"""R2b stage 1, the hybrid: real tokens out of the board (doc 10, R2b).

The fabric does every GEMV of the 8B (all of the DDR traffic, 253 per token, `gemv_axi`); the
A53 runs the REST of the bit-true spec (`ref/fabric_ref.py`: RMSNorm, RoPE, attention over the
KV cache, SiLU, residuals, argmax) in torch on the CPU. The spec code is the same file the GPU
gates use; only the GEMV object is swapped for one that talks to the PL. So:

    tokens here == the spec's free-run tokens   <=>   the board generated them correctly.

What the number means: `fabric ms/token` is PL cycles for one pass of the 253 matrices (the
weight stream). `wall s/token` is dominated by the A53: loading each activation and reading
each result over AXI-Lite, and the vector path in Python. The LM head is read back in windows
of 16,384 rows, which re-streams it 10 times per generated token; those extra passes are
counted separately. Moving the vector path into the fabric (R2b proper) removes all of that.

    # on the Kria (venv with torch + this tree), gemv_axi bitstream loaded:
    sudo ~/bigkev/venv/bin/python -m fabric.big.board.pl_token --prompts ~/bigkev/data/prompts.json -n 80

prompts.json: {name: {"ids": [...], "expect": [...] (optional), "text": "..." (optional)}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from ..pack_weights import Layout
from ..ref.fabric_ref import GEMV_BITS, FabricRef, bfp_quant
from ..ref.gguf_q1 import GGUF
from ..ref.qwen3_ref import GG, NORMS, QK, Q1Matrix, cfg_from_gguf
from .pl_gemv import KINDS, Gemv, build_image, set_fclk


class Stub:
    """A matrix that lives only in DDR: the spec passes it to the GEMV object by name."""

    def __init__(self, out, inn):
        self.out, self.inn = out, inn


def load_lean(path, mats):
    """cfg + the weights the HOST needs: norm gains and the embedding rows. No 1-bit matrix
    is loaded into the A53's RAM except the (tied) embedding, for the per-token row lookup."""
    g = GGUF(path)
    cfg = cfg_from_gguf(g)

    def vec(n):
        return torch.from_numpy(g.dequant(n).reshape(-1).copy())

    ti = g.tensors["token_embd.weight"]
    d, qs = g.q1_parts("token_embd.weight")
    rows, inn = ti.n // ti.shape[0], ti.shape[0]
    emb = Q1Matrix(torch.from_numpy(np.ascontiguousarray(qs).reshape(rows, inn // 8)),
                   torch.from_numpy(d.copy().reshape(rows, inn // QK)))
    W = {"embed": emb, "out_norm": vec("output_norm.weight")}
    W["lm_head"] = Stub(mats["lm_head"]["entry"]["rows"], mats["lm_head"]["entry"]["groups"] * 128)
    for L in range(cfg.n_layer):
        for k, gk in GG.items():
            n = f"l{L}.{k}"
            W[n] = vec(f"blk.{L}.{gk}.weight") if k in NORMS else \
                Stub(mats[n]["entry"]["rows"], mats[n]["entry"]["groups"] * 128)
    return cfg, W


class BoardGemv:
    """Drop-in for ExactBfpGemv: quantise on the host exactly as the spec does, stream the
    matrix from DDR in the fabric, read the fp32 rows back."""

    def __init__(self, eng, mats, lay, maxout):
        self.eng, self.mats, self.lay, self.maxout = eng, mats, lay, maxout
        self.skip_head = False
        self.gold = None                       # (npz, pos) -> compare every GEMV at that position
        self.pos = 0
        self.gold_bad = self.gold_n = 0
        self._x = None
        self.reset_counters()

    def reset_counters(self):
        self.cyc = self.cyc_extra = self.runs = self.stall = 0
        self.t_act = self.t_run = self.t_read = 0.0

    def mm(self, W, X, name="", tr=None):
        eng, m = self.eng, self.mats[name]
        e = m["entry"]
        out = []
        for t in range(X.shape[0]):
            if name == "lm_head" and self.skip_head:
                out.append(torch.zeros(e["rows"]))
                continue
            t0 = time.perf_counter()
            q, ex = bfp_quant(X[t:t + 1], GEMV_BITS, 128)
            xq, xe = q[0].numpy().astype(np.int16), ex[0].numpy().astype(np.int8)
            if self._x is None or not (np.array_equal(xq, self._x[0]) and np.array_equal(xe, self._x[1])):
                eng._act = None
                eng.load_act(name, xq, xe)
                self._x = (xq, xe)
            t1 = time.perf_counter()
            y = np.empty(e["rows"], dtype=np.uint32)
            for s in range(0, e["rows"], eng.res_words):
                ta = time.perf_counter()
                c = eng.run(m["pa"], e, self.lay, self.maxout, res_start=s)
                tb = time.perf_counter()
                if (c["status"] & 0xC) or c["err"] or c["rows"] != e["rows"] or c["tuples"] != e["tuples"]:
                    raise RuntimeError(f"{name}: engine error status {c['status']:#x} err {c['err']} "
                                       f"rows {c['rows']}/{e['rows']} tuples {c['tuples']}/{e['tuples']}")
                if any(p["xsum"] != x for p, x in zip(c["ports"], m["xsum"])):
                    raise RuntimeError(f"{name}: the beats read from DDR are not the packed image (page moved?)")
                k = min(eng.res_words, e["rows"] - s)
                y[s:s + k] = eng.results(k)
                self.t_run += tb - ta
                self.t_read += time.perf_counter() - tb
                if s == 0:
                    self.cyc += c["cyc"]
                    self.stall += c["stall_in"]
                    self.runs += 1
                else:
                    self.cyc_extra += c["cyc"]
            self.t_act += t1 - t0
            if self.gold is not None and self.pos == self.gold[1]:
                g = self.gold[0]
                ok = np.array_equal(xq, g[name + ".xq"]) and np.array_equal(xe, g[name + ".xe"]) \
                    and np.array_equal(y, g[name + ".y"])
                self.gold_n += 1
                self.gold_bad += 0 if ok else 1
            out.append(torch.from_numpy(y.view(np.float32).copy()))
        return torch.stack(out)

    def __call__(self, W, x, name=""):
        return self.mm(W, x[None], name)[0]


def argmax_low(lg):
    """the LOWEST index holding the maximum (the spec's tie rule)."""
    return int(torch.nonzero(lg == lg.max())[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.expanduser("~ubuntu/bigkev/data/Bonsai-8B-Q1_0.gguf"))
    ap.add_argument("--golden", default=os.path.expanduser("~ubuntu/bigkev/data/Bonsai-8B-Q1_0_token.npz"))
    ap.add_argument("--tokenizer", default=os.path.expanduser("~ubuntu/bigkev/data/tokenizer.json"))
    ap.add_argument("--work", default=os.path.expanduser("~ubuntu/bigkev/work"))
    ap.add_argument("--pool", default="/dev/hugepages/bigkev")
    ap.add_argument("--fclk", type=float, default=333.333e6)
    ap.add_argument("--maxout", type=int, default=8)
    ap.add_argument("--prompts", help="JSON {name: {ids, expect?, text?}}")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--text", help="one user message (needs the tokenizers package)")
    ap.add_argument("--think", action="store_true", help="with --text: leave Qwen3 thinking on")
    ap.add_argument("-n", type=int, default=80)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    for k, v in dict(chunk=256, slots=0, port_slots="", rebuild=False, spare_pages=64, min_pages=256).items():
        setattr(a, k, v)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)

    tk = None
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tokenizer)
    except Exception as ex:                                   # ids in, ids out still works
        print(f"NOTE no tokenizer ({type(ex).__name__}); printing token ids only")
    stop = {tk.token_to_id(s) for s in ("<|im_end|>", "<|endoftext|>")} - {None} if tk else {151645, 151643}

    jobs = {}
    gold = np.load(a.golden)
    if a.prompts:
        jobs.update(json.load(open(a.prompts)))
    if a.text:
        assert tk is not None, "--text needs the tokenizers package"
        fmt = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n" + ("" if a.think else "<think>\n\n</think>\n\n")
        jobs["text"] = dict(ids=tk.encode(fmt.format(a.text), add_special_tokens=False).ids)
    if not jobs:
        jobs["golden"] = dict(ids=[int(t) for t in gold["tokens"]], expect=[int(gold["argmax"])])
    if a.only:
        jobs = {k: v for k, v in jobs.items() if k in a.only}

    fclk = set_fclk(a.fclk)
    eng = Gemv()
    lay = Layout(nports=eng.nports, chunk=256, align="stripe")
    n_layer = max(int(k.split(".")[0][1:]) for k in gold.files if k[0] == "l" and k[1].isdigit() and k.endswith(".y")) + 1
    names = [f"l{l}.{k}" for l in range(n_layer) for k in KINDS] + ["lm_head"]
    pool, mats = build_image(a, lay, names)
    t0 = time.time()
    cfg, W = load_lean(a.gguf, mats)
    fab = FabricRef(cfg, W, device="cpu")
    g = BoardGemv(eng, mats, lay, a.maxout)
    fab.gemv = g
    print(f"PL_TOKEN fclk {fclk / 1e6:.2f} MHz, {cfg.n_layer} layers d {cfg.d}, host weights loaded in {time.time() - t0:.0f}s, "
          f"torch {torch.__version__} x{a.threads}")

    res, all_ok = {}, True
    for name, job in jobs.items():
        ids, expect = [int(t) for t in job["ids"]], job.get("expect")
        is_gold = ids == [int(t) for t in gold["tokens"]]
        g.gold = (gold, int(gold["pos"])) if is_gold else None
        g.gold_bad = g.gold_n = 0
        fab.reset()
        g.reset_counters()
        tp = time.perf_counter()
        for i, t in enumerate(ids):                           # prefill: no LM head until the last token
            g.skip_head, g.pos = i < len(ids) - 1, i
            lg = fab.step(t)
        tp = time.perf_counter() - tp
        pre = dict(s=tp, fabric_ms=g.cyc / fclk * 1e3, runs=g.runs)
        print(f"[{name}] prompt {len(ids)} tokens: prefill {tp:.1f}s ({tp / len(ids):.2f} s/token), "
              f"fabric {pre['fabric_ms'] / len(ids):.1f} ms/token"
              + (f"; golden position {int(gold['pos'])}: {g.gold_n - g.gold_bad}/{g.gold_n} GEMVs identical "
                 f"(input mantissas, exponents and output bits) to the GPU spec" if is_gold else ""))
        if is_gold:
            all_ok &= g.gold_bad == 0 and g.gold_n == len(names)
        gen, per = [], []
        g.skip_head = False
        n = min(a.n, len(expect)) if expect and a.n <= 0 else a.n
        while True:
            nxt = argmax_low(lg)
            gen.append(nxt)
            if tk:
                sys.stdout.write(tk.decode([nxt], skip_special_tokens=False))
                sys.stdout.flush()
            if len(gen) >= n or nxt in stop:
                break
            g.reset_counters()
            ts = time.perf_counter()
            g.pos = len(ids) + len(gen) - 1
            lg = fab.step(nxt)
            per.append(dict(wall=time.perf_counter() - ts, fabric_ms=g.cyc / fclk * 1e3,
                            head_extra_ms=g.cyc_extra / fclk * 1e3, act=g.t_act, run=g.t_run, read=g.t_read,
                            stall=g.stall / max(1, g.cyc)))
        print()
        r = dict(n_prompt=len(ids), gen=gen, prefill=pre, per_token=per)
        if tk:
            r["text"] = tk.decode(gen, skip_special_tokens=False)
        if per:
            wall = float(np.median([p["wall"] for p in per]))
            fms = float(np.median([p["fabric_ms"] for p in per]))
            xms = float(np.median([p["head_extra_ms"] for p in per]))
            r.update(wall_s_per_token=wall, fabric_ms_per_token=fms, head_extra_ms_per_token=xms)
            print(f"[{name}] {len(gen)} tokens generated: wall {wall:.2f} s/token = {1 / wall:.2f} tok/s end to end (hybrid, MEASURED); "
                  f"fabric weight stream {fms:.1f} ms/token (= {1e3 / fms:.2f} tok/s if the host cost nothing), "
                  f"+{xms:.0f} ms re-streaming the LM head for readback; host: act-load {np.median([p['act'] for p in per]):.2f}s "
                  f"run-wait {np.median([p['run'] for p in per]):.2f}s readback {np.median([p['read'] for p in per]):.2f}s")
        if expect:
            k = min(len(expect), len(gen))
            same = sum(x == y for x, y in zip(gen[:k], expect[:k]))
            first = next((i for i in range(k) if gen[i] != expect[i]), None)
            r.update(expect_n=k, expect_same=same, first_div=first)
            all_ok &= same == k
            print(f"[{name}] vs the spec's free-run: {same}/{k} tokens identical" + ("" if first is None else f", first difference at {first}"))
        res[name] = r
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(fclk=fclk, n_layer=cfg.n_layer, verdict="PASS" if all_ok else "FAIL", prompts=res), f)
    print(f"PL_TOKEN_VERDICT {'PASS' if all_ok else 'FAIL'}")
    pool.close(free=False)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
