"""R1d: ternary vs binary at equal bytes per token (doc 10, R1 leftovers, task 4).

Decode speed is DDR bandwidth / bytes streamed per token, so the model question is quality
per streamed byte. For each GGUF this reports

  * bytes/token, exact, from tensor sizes: every matrix that is streamed once per token (each
    layer's q/k/v/o/gate/up/down plus the LM head), EXCLUDING the embedding lookup (one row per
    token; a tied head is the embedding matrix streamed once as the head) and EXCLUDING KV
    traffic. As stored in the GGUF, and at the best fixed packing (ternary: 5 trits/byte =
    1.6 bpw + fp16 scales; binary: 1 bpw + scales, i.e. as stored).
  * DERIVED tok/s = 15.54e9 (R0 MEASURED) / bytes per token: a weights-only roofline.
  * float-reference teacher-forced NLL (nats/token) on the SAME token ids for every model:
    r1b_sweep.eval_texts + the cached binary-8B chat text (the r1c_gate set).

Gates: binary-8B pooled NLL must reproduce results/r1c_gate.json `float_nll`; each ternary
model must give a sensible greedy continuation. The model choice is the human's call: this
tool reports, it does not decide.

    python -u -m fabric.big.ref.r1d_ternary                    # full eval (GPU, ~20 min)
    python -u -m fabric.big.ref.r1d_ternary --bytes-only       # the bytes/tok-s table, no forward
    python -u -m fabric.big.ref.r1d_ternary --pin              # pin the block layout vs -unpacked
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import struct
import time
from pathlib import Path

import numpy as np
import torch

from .gguf_q1 import GGUF, Q1_BLOCK_BYTES, Q2_TYPES, QK
from .qwen3_ref import GG, NORMS, FloatGemv, Qwen3Ref, cfg_from_gguf, from_gguf

ROOT = Path(__file__).resolve().parents[3]
BW = 15.54e9   # R0 MEASURED PL<-DDR bytes/s (HP0+HP1+HP3 @ 333 MHz)
MODELS = ["data/big/Bonsai-8B-Q1_0.gguf", "data/big/Ternary-Bonsai-4B-Q2_0.gguf",
          "data/big/Ternary-Bonsai-8B-Q2_0.gguf"]
MATS = [k for k in GG if k not in NORMS]
PROBE = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DENSE_BYTES = {0: 4, 1: 2, 30: 2}


# ----------------------------------------------------------------------------- bytes per token
def is_tern(g) -> bool:
    return any(t in Q2_TYPES for t in g.type_hist())


def label(g, path) -> str:
    m = re.search(r"(\d+(?:\.\d+)?B)", Path(path).name)
    size = m.group(1) if m else g.meta.get("general.size_label") or Path(path).stem   # file name: "8B", not "8.2B"
    return f"{'ternary' if is_tern(g) else 'binary'}-{size}"


def tok_hash(g) -> str:
    m = g.meta
    s = "\x00".join(m["tokenizer.ggml.tokens"]) + "\x01" + "\x00".join(m["tokenizer.ggml.merges"])
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def mat_bytes(g, name):
    """(weights, bytes as stored, bytes at the best fixed packing, scale bytes) of one matrix."""
    ti = g.tensors[name]
    rows, inn, grp = ti.n // ti.shape[0], ti.shape[0], g.q2_group(name)
    if grp:      # ternary: 5 trits/byte per row (what TernMatrix holds) + one fp16 scale per group
        sc = ti.n // grp * 2
        st, best = ti.n // grp * (2 + grp // 4), rows * -(-inn // 5) + sc
    elif g.is_q1(name):
        sc = ti.n // QK * 2
        st = best = ti.n // QK * Q1_BLOCK_BYTES
    else:
        sc, st = 0, ti.n * DENSE_BYTES[ti.ggml_type]
        best = st
    assert st <= ti.nbytes < st + 64, f"{name}: {ti.nbytes} bytes on disk vs {st} from the format"
    return ti.n, st, best, sc


def stream_names(g, cfg):
    head = "token_embd.weight" if cfg.tied else "output.weight"
    return [f"blk.{L}.{GG[k]}.weight" for L in range(cfg.n_layer) for k in MATS] + [head]


def code_hist(g, names):
    """Weights at -1 / 0 / +1 / (+2, reserved) over the named 2-bit tensors."""
    tab = np.zeros((256, 4), np.int64)
    for b in range(256):
        for k in range(4):
            tab[b, (b >> 2 * k) & 3] += 1
    h = np.zeros(256, np.int64)
    for n in names:
        if not g.q2_group(n):
            continue
        _, qs, _ = g.q2_parts(n)
        for i in range(0, len(qs), 1 << 21):
            h += np.bincount(np.ascontiguousarray(qs[i:i + (1 << 21)]).ravel(), minlength=256)
    return h @ tab


def stream_table(g, cfg, hist=True) -> dict:
    names = stream_names(g, cfg)
    n, st, best, sc = (sum(x) for x in zip(*(mat_bytes(g, nm) for nm in names)))
    every = set(names) | {"token_embd.weight"}
    dev = sum(mat_bytes(g, nm)[2] for nm in every)
    dev += sum(ti.n * 4 for nm, ti in g.tensors.items() if nm not in every)
    r = dict(n_weights=n, bytes_stored=st, bpw_stored=8 * st / n, bytes_best=best, bpw_best=8 * best / n,
             scale_bytes=sc, toks_stored=BW / st, toks_best=BW / best, head_tied=cfg.tied,
             embed_bytes_stored=mat_bytes(g, "token_embd.weight")[1], device_bytes_packed=dev,
             types={str(k): v for k, v in g.type_hist().items()},
             group=max(g.q2_group(nm) for nm in names) or QK)
    if is_tern(g):
        r["bytes_ptq1_0"] = n // 128 * 28     # PrismML's own g128 base-3 block (1.75 bpw), not verified here
        if hist:
            h = code_hist(g, names)
            p = h[:3] / h[:3].sum()
            r.update(frac_neg=p[0], frac_zero=p[1], frac_pos=p[2], code3=int(h[3]),
                     h0_bpw=float(-(p * np.log2(p)).sum()))
    return r


def print_table(rows):
    print("bytes/token = matrices streamed once per token (7 per layer + LM head); EXCLUDES the embedding "
          "row lookup and all KV traffic.\ntok/s DERIVED = 15.54e9 B/s (R0 MEASURED) / bytes per token: "
          "a weights-only roofline, KV reads/writes come off it.")
    print(f"{'model':13s} {'weights':>14s} {'stored MB':>10s} {'bpw':>6s} {'tok/s':>6s} | "
          f"{'best MB':>9s} {'bpw':>6s} {'tok/s':>6s} | {'device MB':>9s}  notes")
    for lab, r in rows.items():
        note = "head tied to embed" if r["head_tied"] else "separate LM head"
        if "frac_zero" in r:
            note += (f"; -1/0/+1 = {r['frac_neg']:.3f}/{r['frac_zero']:.3f}/{r['frac_pos']:.3f} "
                     f"H0 {r['h0_bpw']:.3f} b/w, code3 {r['code3']}")
        print(f"{lab:13s} {r['n_weights']:14,d} {r['bytes_stored'] / 1e6:10.1f} {r['bpw_stored']:6.3f} "
              f"{r['toks_stored']:6.2f} | {r['bytes_best'] / 1e6:9.1f} {r['bpw_best']:6.3f} "
              f"{r['toks_best']:6.2f} | {r['device_bytes_packed'] / 1e6:9.1f}  {note}")
    print("stored = as in the GGUF (MEASURED from tensor sizes); best = ternary 5 trits/byte + fp16 scale "
          "per group, binary as stored (DERIVED); device = whole model packed incl. embedding and norms "
          "(DERIVED; what from_gguf puts on --device).")


# ----------------------------------------------------------------------------- layout pinning
def _st_dtype(buf, dt):
    if dt == "F16":
        return np.frombuffer(buf, np.float16).astype(np.float32)
    if dt == "BF16":
        return (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
    assert dt == "F32", dt
    return np.frombuffer(buf, np.float32)


class RemoteSafetensors:
    """Row ranges of tensors in a HF repo's safetensors shards over anonymous HTTP range
    requests (the 4B/8B unpacked releases are 8-16 GB; pinning needs a few MB)."""

    def __init__(self, repo, tries=8):
        import requests
        from huggingface_hub import hf_hub_url
        self.repo, self.url, self.s, self.tries, self.fetched = repo, hf_hub_url, requests.Session(), tries, 0
        r = self._get("model.safetensors.index.json", None)
        self.where = json.loads(r)["weight_map"] if r is not None else None
        self.hdr = {}

    def _get(self, fn, rng):
        h = {"Range": f"bytes={rng[0]}-{rng[1] - 1}"} if rng else {}
        for i in range(self.tries):
            try:
                r = self.s.get(self.url(self.repo, fn), headers=h, timeout=60)
                if r.status_code == 404 and not rng:
                    return None
                r.raise_for_status()
                assert not rng or len(r.content) == rng[1] - rng[0]
                self.fetched += len(r.content)
                return r.content
            except Exception as e:       # flaky link: back off and retry
                if i == self.tries - 1:
                    raise
                print(f"    retry {fn} {rng}: {type(e).__name__}")
                time.sleep(3 * (i + 1))

    def rows(self, name, r0, r1):
        """(float32 [r1-r0, in], dtype string) of rows [r0, r1) of tensor `name`."""
        fn = self.where[name] if self.where else "model.safetensors"
        if fn not in self.hdr:
            n = struct.unpack("<Q", self._get(fn, (0, 8)))[0]
            self.hdr[fn] = (8 + n, json.loads(self._get(fn, (8, 8 + n))))
        base, hd = self.hdr[fn]
        t = hd[name]
        out, inn = t["shape"]
        es = (t["data_offsets"][1] - t["data_offsets"][0]) // (out * inn)
        a = base + t["data_offsets"][0]
        buf = self._get(fn, (a + r0 * inn * es, a + r1 * inn * es))
        return _st_dtype(buf, t["dtype"]).reshape(r1 - r0, inn), t["dtype"]


def pin(paths, a):
    """Prove the GGUF block layout against PrismML's unpacked release of the same model: zeros
    and signs agree 100%, and every GGUF value equals sign(u) * max|u| over its scale group of
    the unpacked weights (after the unpacked dtype's rounding). `exact` is plain equality: it
    falls short of 1 only where the unpacked release itself carries two magnitudes inside one
    group (`mixed_groups`), which the one-scale-per-group block cannot represent."""
    from .qwen3_ref import HF
    res, ok = {}, True
    for path in paths:
        g = GGUF(path)
        cfg = cfg_from_gguf(g)
        repo = a.pin_repo or "prism-ml/" + re.sub(r"-(P?Q[12]_0(_g64)?)$", "", Path(path).stem) + "-unpacked"
        rs = RemoteSafetensors(repo)
        want = [("token_embd.weight", "model.embed_tokens.weight", 0, a.pin_rows),
                ("token_embd.weight", "model.embed_tokens.weight", 100000, 100000 + a.pin_rows),
                ("blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight", 0, None)]
        if not cfg.tied:
            want.append(("output.weight", "lm_head.weight", 5000, 5000 + a.pin_rows))
        for L in sorted({0, cfg.n_layer // 2, cfg.n_layer - 1}):
            for k in MATS:
                out = g.tensors[f"blk.{L}.{GG[k]}.weight"].n // g.tensors[f"blk.{L}.{GG[k]}.weight"].shape[0]
                r0 = out // 3
                want.append((f"blk.{L}.{GG[k]}.weight", f"model.layers.{L}.{HF[k]}.weight", r0, r0 + a.pin_rows))
        tot = dict(n=0, zero_bad=0, sign_bad=0, exact=0, groupmax=0, groups=0, mixed_groups=0, maxdiff=0.0,
                   maxrel=0.0, tensors=0, hist=[0, 0, 0])
        t0 = time.time()
        print(f"R1D_PIN {Path(path).name} vs {repo}  types {g.type_hist()}")
        for gn, hn, r0, r1 in want:
            ti = g.tensors[gn]
            r1 = r1 or ti.n // ti.shape[0]
            G = g.dequant_rows(gn, r0, r1)
            U, dt = rs.rows(hn, r0, r1)
            assert U.shape == G.shape, (gn, U.shape, G.shape)
            Gr = torch.from_numpy(G).to(torch.bfloat16).float().numpy() if dt == "BF16" else G
            zb = int(((G == 0) != (U == 0)).sum())
            sb = int((np.sign(G) != np.sign(U)).sum())
            grp = g.q2_group(gn) or QK
            A = np.abs(U).reshape(U.shape[0], -1, grp)
            gm = A.max(-1, keepdims=True)
            mixed = int((np.where(A > 0, A, gm).min(-1, keepdims=True) != gm).sum())
            ex, rd = int((G == U).sum()), int((Gr == (np.sign(U).reshape(A.shape) * gm).reshape(U.shape)).sum())
            md = float(np.abs(G.astype(np.float64) - U).max())
            rel = float((np.abs(G.astype(np.float64) - U) / np.maximum(np.abs(U), 1e-30)).max())
            for i, v in enumerate((-1, 0, 1)):
                tot["hist"][i] += int((np.sign(G) == v).sum())
            for k, v in (("n", G.size), ("zero_bad", zb), ("sign_bad", sb), ("exact", ex), ("groupmax", rd),
                         ("groups", A.shape[0] * A.shape[1]), ("mixed_groups", mixed), ("tensors", 1)):
                tot[k] += v
            tot["maxdiff"], tot["maxrel"] = max(tot["maxdiff"], md), max(tot["maxrel"], rel)
            print(f"  {gn:26s} type {ti.ggml_type:3d} G={grp:3d} rows [{r0},{r1}) vs {dt}: n={G.size} "
                  f"zero_mismatch={zb} sign_mismatch={sb} eq_sign*groupmax={rd / G.size:.6f} "
                  f"exact={ex / G.size:.6f} mixed_groups={mixed}/{A.shape[0] * A.shape[1]} max|d|={md:.3e}")
        good = tot["zero_bad"] == 0 and tot["sign_bad"] == 0 and tot["groupmax"] == tot["n"]
        ok &= good
        tot.update(repo=repo, fetched_bytes=rs.fetched, verdict="PASS" if good else "FAIL",
                   types={str(k): v for k, v in g.type_hist().items()})
        res[Path(path).name] = tot
        print(f"  => {tot['tensors']} samples, {tot['n']:,} weights: zero_mismatch {tot['zero_bad']} "
              f"sign_mismatch {tot['sign_bad']} eq_sign*groupmax {tot['groupmax']:,} exact {tot['exact']:,} "
              f"({tot['exact'] / tot['n']:.6%}); unpacked groups with 2 magnitudes {tot['mixed_groups']}/"
              f"{tot['groups']}; max|d| {tot['maxdiff']:.3e} (rel {tot['maxrel']:.2%}); "
              f"-1/0/+1 {tot['hist']}; fetched {rs.fetched / 1e6:.1f} MB "
              f"in {time.time() - t0:.0f}s  {'PASS' if good else 'FAIL'}")
    out = Path(a.pin_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"R1D_PIN_VERDICT {'PASS' if ok and res else 'FAIL'} -> {out}")


# ----------------------------------------------------------------------------- evaluation
def device_bytes(W) -> int:
    seen, n = set(), 0
    for w in W.values():
        for t in ([w] if torch.is_tensor(w) else [v for v in vars(w).values() if torch.is_tensor(v)]):
            if id(t) not in seen:
                seen.add(id(t))
                n += t.numel() * t.element_size()
    return n


def eval_model(path, texts, tok, a) -> dict:
    cuda = torch.device(a.device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    cfg, W, g = from_gguf(path, device=a.device)
    gemv = FloatGemv()
    gemv.CHUNK = a.chunk
    ref = Qwen3Ref(cfg, W, device=a.device, gemv=gemv)
    r = dict(device_bytes_packed_measured=device_bytes(W), load_s=time.time() - t0, per_text={}, tok_nll={})
    print(f"    loaded in {r['load_s']:.0f}s, {r['device_bytes_packed_measured'] / 1e6:.1f} MB packed on "
          f"{a.device} (MEASURED)")
    tern, eos = is_tern(g), g.meta.get("tokenizer.ggml.eos_token_id")
    del g
    tot = n = 0.0
    for name, t in texts:
        t0 = time.time()
        lg = ref.forward_seq(t).cpu().double()
        nl = torch.nn.functional.cross_entropy(lg[:-1], torch.tensor(t[1:]), reduction="none")
        del lg
        r["per_text"][name] = dict(n=len(nl), nll=nl.mean().item())
        r["tok_nll"][name] = [round(v, 5) for v in nl.tolist()]
        tot, n = tot + nl.sum().item(), n + len(nl)
        print(f"    {name:13s} n={len(nl)} NLL {nl.mean().item():.4f} nats/tok ({time.time() - t0:.0f}s)")
    r.update(nll=tot / n, ntok=int(n))
    neutral = [v for k, v in r["per_text"].items() if k != "chat_selfgen"]
    r["nll_ex_selfgen"] = sum(v["nll"] * v["n"] for v in neutral) / max(1, sum(v["n"] for v in neutral))
    r["sane"] = bool(r["nll"] < 0.5 * math.log(cfg.vocab))
    if a.gen and (tern or a.gen_all):
        ids, out = tok(PROBE).input_ids, []
        t0 = time.time()
        for _ in range(a.gen):   # forward_seq on the growing prompt: one matrix sweep per token
            out.append(int(ref.forward_seq(ids + out)[-1].argmax()))
            if out[-1] == eos:
                break
        r["greedy"] = tok.decode(out)
        print(f"    greedy {len(out)} tok ({time.time() - t0:.0f}s): {r['greedy']!r}")
    if cuda:
        r["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
        print(f"    GPU peak {r['gpu_peak_bytes'] / 1e9:.2f} GB (MEASURED)")
    del ref, W, gemv
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return r


def paired(a, b, skip=()):
    """mean and standard error of the per-token NLL difference a - b over the shared texts."""
    d = np.concatenate([np.array(a["tok_nll"][k]) - np.array(b["tok_nll"][k]) for k in a["tok_nll"]
                        if k not in skip])
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def decide(res) -> list:
    out = []
    done = {k: v for k, v in res.items() if "nll" in v}
    bins = [k for k in done if k.startswith("binary")]
    if not bins:
        return ["R1D_RESULT no binary reference model evaluated: no comparison"]
    bl = "binary-8B" if "binary-8B" in done else bins[0]
    b = done[bl]
    for tl, t in done.items():
        if not tl.startswith("ternary"):
            continue
        dm, se = paired(t, b)
        dn, sn = paired(t, b, skip=("chat_selfgen",))
        ds, db = t["bytes_stored"] / b["bytes_stored"] - 1, t["bytes_best"] / b["bytes_stored"] - 1
        out.append(f"R1D_CMP {tl} vs {bl}: pooled NLL {t['nll']:.4f} vs {b['nll']:.4f} (d {dm:+.4f} +- {se:.4f} "
                   f"paired SE, {t['ntok']} tok, MEASURED); ex-selfgen {t['nll_ex_selfgen']:.4f} vs "
                   f"{b['nll_ex_selfgen']:.4f} (d {dn:+.4f} +- {sn:.4f}); bytes/token stored {t['bytes_stored'] / 1e6:.1f} vs "
                   f"{b['bytes_stored'] / 1e6:.1f} MB ({ds:+.1%}), best packing {t['bytes_best'] / 1e6:.1f} MB "
                   f"({db:+.1%}) (DERIVED)")
        nw = tl if t["nll"] < b["nll"] else bl
        if tl == "ternary-4B" and bl == "binary-8B" and t["nll"] < b["nll"] and db <= 0:
            out.append(f"R1D_FLAG ternary-4B dominates: lower pooled NLL at <= the bytes/token of binary-8B "
                       f"(at the 5-trit packing, {db:+.1%}; as stored Q2_0 it is {ds:+.1%})")
        else:
            out.append(f"R1D_RESULT {tl} vs {bl}: NLL winner {nw}; bytes winner as stored "
                       f"{tl if ds < 0 else bl}, at best packing {tl if db < 0 else bl}")
        if (t["nll"] < b["nll"]) != (t["nll_ex_selfgen"] < b["nll_ex_selfgen"]):
            out.append(f"R1D_NOTE {tl}: the NLL winner flips when chat_selfgen is excluded (that text is the "
                       f"binary Bonsai-8B float model's own greedy answer, so it favours binary-8B)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--tok", default="data/big/Bonsai-8B-meta", help="HF tokenizer dir (all Bonsai GGUFs share one vocab; checked)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--out", default="fabric/big/results/r1d_ternary.json")
    ap.add_argument("--chat-cache", default="fabric/big/results/Bonsai-8B-Q1_0_chatgen_T256.json")
    ap.add_argument("--r1c", default="fabric/big/results/r1c_gate.json")
    ap.add_argument("--gen", type=int, default=24, help="greedy sanity tokens per ternary model, stops at EOS (0 = skip)")
    ap.add_argument("--gen-all", action="store_true", help="greedy sanity on binary models too")
    ap.add_argument("--chunk", type=int, default=FloatGemv.CHUNK, help="GEMV row chunk (lower = less device memory)")
    ap.add_argument("--bytes-only", action="store_true", help="bytes/token table only, no forward pass")
    ap.add_argument("--no-hist", action="store_true", help="skip the -1/0/+1 histogram (reads every weight byte)")
    ap.add_argument("--pin", action="store_true", help="pin the GGUF block layout against the unpacked release")
    ap.add_argument("--pin-repo", default="", help="unpacked HF repo (default prism-ml/<model>-unpacked)")
    ap.add_argument("--pin-rows", type=int, default=64)
    ap.add_argument("--pin-out", default="fabric/big/results/r1d_pin.json")
    a = ap.parse_args()
    paths = []
    for p in a.models:
        if Path(p).exists():
            paths.append(p)
        else:
            print(f"NOTE skipping {p}: file not found")
    if a.pin:
        return pin(paths, a)

    # 1. bytes per token + tokenizer identity (no forward pass)
    res, hashes = {}, {}
    for p in paths:
        g = GGUF(p)
        cfg = cfg_from_gguf(g)
        lab = label(g, p)
        lab = lab if lab not in res else f"{lab}:{Path(p).stem}"
        res[lab] = dict(path=p, cfg=vars(cfg), tok_hash=tok_hash(g), **stream_table(g, cfg, not a.no_hist))
        hashes[lab] = res[lab]["tok_hash"]
        del g
    print_table(res)
    same = len(set(hashes.values())) <= 1
    print(f"tokenizer (GGUF tokens+merges sha256/16): {hashes} -> {'IDENTICAL' if same else 'DIFFERENT'}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(bw_bytes_per_s=BW, T=a.T, device=a.device, tokenizer_identical=same, models=res)
    if a.bytes_only:
        out.write_text(json.dumps(doc, indent=1))
        print(f"R1D_VERDICT DONE (bytes only) -> {out}")
        return
    assert same, "tokenizers differ: per-token NLL is not comparable across these models"

    # 2. the shared token ids (r1c_gate's four texts)
    from transformers import AutoTokenizer
    from .r1b_sweep import eval_texts
    tok = AutoTokenizer.from_pretrained(a.tok)
    g = GGUF(paths[0])
    vt = g.meta["tokenizer.ggml.tokens"]
    texts, chat = eval_texts(tok, a.T)
    cc = Path(a.chat_cache)
    if cc.exists():
        texts.append(("chat_selfgen", (tok(chat).input_ids + json.loads(cc.read_text()))[:a.T]))
    else:
        print(f"NOTE no cached chat text at {cc}: evaluating {len(texts)} texts only")
    used = sorted({i for _, t in texts for i in t})
    assert len(vt) == len(tok) and all(tok.convert_ids_to_tokens(i) == vt[i] for i in used), \
        f"--tok {a.tok} does not match the GGUF vocabulary"
    del g, vt
    doc["texts"] = {n: len(t) for n, t in texts}

    # 3. one model at a time
    for lab, r in res.items():
        print(f"[{lab}] {r['path']}")
        try:
            r.update(eval_model(r["path"], texts, tok, a))
        except torch.cuda.OutOfMemoryError as e:
            r["error"] = f"out of memory on {a.device}: {str(e)[:200]}"
            print(f"    NOTE {r['error']}  (retry with --chunk 2048, or --device cpu)")
            gc.collect()
            if torch.device(a.device).type == "cuda":
                torch.cuda.empty_cache()
        out.write_text(json.dumps(doc, indent=1))

    # 4. gates + decision
    gates = []
    rc = Path(a.r1c)
    for lab, r in res.items():
        if "nll" not in r:
            continue
        ref = json.loads(rc.read_text()) if rc.exists() else {}
        if ref.get("model") == Path(r["path"]).name and r["ntok"] == ref.get("ntok"):
            d = abs(r["nll"] - ref["float_nll"])
            gates.append((lab, d < 1e-3, f"pooled NLL {r['nll']:.6f} vs r1c_gate float_nll {ref['float_nll']:.6f} |d|={d:.2e}"))
        gates.append((lab, r["sane"], f"pooled NLL {r['nll']:.4f} < {0.5 * math.log(r['cfg']['vocab']):.2f} (half of uniform)"))
    for lab, okk, msg in gates:
        print(f"R1D_SANITY {lab}: {msg} {'PASS' if okk else 'FAIL'}")
    for lab, r in res.items():
        if "greedy" in r:
            print(f"R1D_GREEDY {lab}: {r['greedy']!r}")
    lines = decide(res)
    print("\n".join(lines))
    doc.update(sanity=[dict(model=l, ok=o, what=m) for l, o, m in gates], decision=lines)
    out.write_text(json.dumps(doc, indent=1))
    print(f"R1D_VERDICT DONE -> {out}")


if __name__ == "__main__":
    main()
