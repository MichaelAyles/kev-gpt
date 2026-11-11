"""R2b stage 2, the hybrid with a C host: real tokens out of the board, faster (doc 10, R2b).

Same split as `pl_token.py` (the fabric does all 253 GEMVs of a token from DDR, the A53 does
the rest of the bit-true spec), but the A53 side is `host/bk_host.c` instead of torch, and it
needs only numpy. The check is the same: tokens here == the spec's free-run tokens.

    sudo ~/bigkev/venv/bin/python -m fabric.big.board.pl_token_c --prompts ~/bigkev/data/prompts.json -n 80

Per generated token the report splits the wall time into
    fabric   PL cycles for the 253 matrices (one pass each), from the PL's own counter
    host     the C vector path (bk_host's own timers)
    io       activation load + result readback + polling over AXI-Lite, and the extra LM-head
             passes that the 16,384-row readback window costs (9 re-streams per token)
With the version-2 wrapper (`--dma`, when the bitstream has it) activations and results go
through DDR and the argmax is taken in the fabric, so `io` shrinks to register writes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .. import gemv_axi_regs as R
from ..host.bk_host import BkHost, mat_name
from ..pack_weights import Layout
from .pl_gemv import KINDS, Gemv, build_image, set_fclk


class BoardGemv:
    """bk_host GEMV callback -> gemv_axi (version-1 path: AXI-Lite activation + result window)."""

    def __init__(self, eng, mats, lay, maxout, n_layer):
        self.eng, self.mats, self.lay, self.maxout, self.n_layer = eng, mats, lay, maxout, n_layer
        self.reset()

    def reset(self):
        self.cyc = self.cyc_extra = self.stall = 0
        self.t_act = self.t_run = self.t_read = 0.0

    def __call__(self, mid, xq, xe, y, same):
        eng, w32 = self.eng, self.eng.w32
        name = mat_name(mid, self.n_layer)
        m = self.mats[name]
        e = m["entry"]
        t0 = time.perf_counter()
        if not same:
            eng._act = None
            eng.load_act(name, xq, xe)
        t1 = time.perf_counter()
        self.t_act += t1 - t0
        yu = y.view(np.uint32)
        for s in range(0, e["rows"], eng.res_words):
            ta = time.perf_counter()
            c = eng.run(m["pa"], e, self.lay, self.maxout, res_start=s)
            tb = time.perf_counter()
            if (c["status"] & 0xC) or c["err"] or c["rows"] != e["rows"] or c["tuples"] != e["tuples"]:
                raise RuntimeError(f"{name}: engine error status {c['status']:#x} err {c['err']} "
                                   f"rows {c['rows']}/{e['rows']} tuples {c['tuples']}/{e['tuples']}")
            if any(p_["xsum"] != x for p_, x in zip(c["ports"], m["xsum"])):
                raise RuntimeError(f"{name}: the beats read from DDR are not the packed image (page moved?)")
            k = min(eng.res_words, e["rows"] - s)
            yu[s:s + k] = eng.results(k)
            self.t_run += tb - ta
            self.t_read += time.perf_counter() - tb
            if s == 0:
                self.cyc += c["cyc"]
                self.stall += c["stall_in"]
            else:
                self.cyc_extra += c["cyc"]
        return 0


class DmaStats:
    """the same counters as BoardGemv, read from the C callback's context (version-2 path)."""

    def __init__(self, dma, fclk):
        self.dma, self.fclk = dma, fclk

    def reset(self):
        self.dma.reset()

    def __getattr__(self, k):
        c = self.dma.c
        return {"cyc": c.cyc, "cyc_extra": 0, "stall": c.stall_in, "t_act": (c.ns_act + c.ns_regs) / 1e9,
                "t_run": c.ns_wait / 1e9, "t_read": c.ns_read / 1e9}[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.expanduser("~ubuntu/bigkev/data/Bonsai-8B-Q1_0.gguf"))
    ap.add_argument("--golden", default=os.path.expanduser("~ubuntu/bigkev/data/Bonsai-8B-Q1_0_token.npz"))
    ap.add_argument("--tables", default=os.path.expanduser("~ubuntu/bigkev/data/host_tables_8B.npz"))
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
    ap.add_argument("--max-ctx", type=int, default=1024)
    ap.add_argument("--no-dma", action="store_true", help="use the AXI-Lite path even on a version-2 bitstream")
    ap.add_argument("--full-head", action="store_true", help="with DMA: read all logits back instead of the fabric argmax")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    for k, v in dict(chunk=256, slots=0, port_slots="", rebuild=False, spare_pages=64, min_pages=256).items():
        setattr(a, k, v)

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
    dma = None
    if (eng.rd(R.PARAMS) >> 24) >= 2 and not a.no_dma:
        from .pl_dma import PlDma
        dma = PlDma(eng, mats, lay, names, pool, a.maxout, head_argmax=not a.full_head)
        g = DmaStats(dma, fclk)
        host = BkHost.from_gguf(a.gguf, tables=a.tables, max_ctx=a.max_ctx)
        host.set_c_gemv(dma.lib.bkpl_gemv, dma.ctx)
    else:
        g = BoardGemv(eng, mats, lay, a.maxout, n_layer)
        host = BkHost.from_gguf(a.gguf, gemv=g, tables=a.tables, max_ctx=a.max_ctx)
    print(f"PL_TOKEN_C {'DMA (v2)' if dma else 'AXI-Lite (v1)'} fclk {fclk / 1e6:.2f} MHz, {n_layer} layers, host loaded in {time.time() - t0:.0f}s, lib {os.path.basename(host.lib._name)}")

    res, all_ok = {}, True
    for name, job in jobs.items():
        ids, expect = [int(t) for t in job["ids"]], job.get("expect")
        host.reset()
        host.reset_stats()
        g.reset()
        tp = time.perf_counter()
        for i, t in enumerate(ids):                           # prefill: no LM head until the last token
            lg = host.step(t, want_logits=i == len(ids) - 1, copy=False)
        tp = time.perf_counter() - tp
        print(f"[{name}] prompt {len(ids)} tokens: prefill {tp:.1f}s ({tp / len(ids) * 1e3:.0f} ms/token), "
              f"fabric {g.cyc / fclk * 1e3 / len(ids):.1f} ms/token")
        gen, per = [], []
        while True:
            nxt = int(np.flatnonzero(lg == lg.max())[0])      # the LOWEST index holding the maximum
            gen.append(nxt)
            if tk:
                sys.stdout.write(tk.decode([nxt], skip_special_tokens=False))
                sys.stdout.flush()
            if len(gen) >= a.n or nxt in stop:
                break
            g.reset()
            host.reset_stats()
            ts = time.perf_counter()
            lg = host.step(nxt, copy=False)
            wall = time.perf_counter() - ts
            st = host.stats()
            cb = g.t_act + g.t_run + g.t_read
            per.append(dict(wall=wall, fabric_ms=g.cyc / fclk * 1e3, head_extra_ms=g.cyc_extra / fclk * 1e3,
                            host_ms=st["host_ns"] / 1e6, glue_ms=(wall - cb) * 1e3 - st["host_ns"] / 1e6,
                            act_ms=g.t_act * 1e3, run_ms=g.t_run * 1e3, read_ms=g.t_read * 1e3,
                            ctx=len(ids) + len(gen) - 1, ns=st["ns"]))
        print()
        r = dict(n_prompt=len(ids), gen=gen, prefill_s=tp, per_token=per)
        if tk:
            r["text"] = tk.decode(gen, skip_special_tokens=False)
        if per:
            med = lambda k: float(np.median([p[k] for p in per]))          # noqa: E731
            wall = med("wall")
            r.update(wall_s_per_token=wall, fabric_ms_per_token=med("fabric_ms"), host_ms_per_token=med("host_ms"),
                     act_ms=med("act_ms"), run_ms=med("run_ms"), read_ms=med("read_ms"),
                     head_extra_ms_per_token=med("head_extra_ms"))
            print(f"[{name}] {len(gen)} tokens generated: {wall * 1e3:.0f} ms/token = {1 / wall:.2f} tok/s end to end (hybrid, MEASURED) "
                  f"at context {per[0]['ctx']}..{per[-1]['ctx']}; fabric weight stream {med('fabric_ms'):.1f} ms, "
                  f"C host {med('host_ms'):.0f} ms, Python glue {med('glue_ms'):.0f} ms, host<->fabric io: act-load {med('act_ms'):.0f} ms + run/poll {med('run_ms'):.0f} ms "
                  f"(of which {med('fabric_ms') + med('head_extra_ms'):.0f} ms is the fabric streaming, "
                  f"{med('head_extra_ms'):.0f} ms of it LM-head re-streams) + readback {med('read_ms'):.0f} ms")
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
            json.dump(dict(fclk=fclk, n_layer=n_layer, verdict="PASS" if all_ok else "FAIL", prompts=res), f, default=float)
    print(f"PL_TOKEN_C_VERDICT {'PASS' if all_ok else 'FAIL'}")
    pool.close(free=False)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
