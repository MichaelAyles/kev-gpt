"""R1 long-context check (HANDOFF task 1): the bit-true FABRIC spec vs the FLOAT reference
at T = 1024 and 2048, teacher-forced, reported by position bucket.

  [0] invariants, on one short text (always run first; alone with --check-chunked):
        fab.prefill_chunked(t, chunk) == fab.forward_seq(t) bit-identical, for two chunk
          sizes (one not a multiple of 16), forward_seq on the stock ExactBfpGemv
        prefill_chunked(t[:T/2]) == forward_seq(t)[:T/2]   (a shorter run is a prefix)
        float forward_seq_chunked vs forward_seq: max |dlogit| <= --float-tol
  [1] per long text (tinystories / readme / master_doc): float logits to host RAM as fp32
      (chunked), then the fabric logits (prefill_chunked), then per-position argmax / NLL /
      logit cosine in float64 a block at a time. Nothing [T, vocab] ever sits on the GPU.
  [2] per T: pooled, per text and per position bucket (0-256, 256-512, ...) top-1 / dNLL /
      cos, and the trend across buckets.

Each text is run ONCE, at the largest requested T it reaches; a shorter T is the prefix of
that run (both models are causal; [0] proves the fabric prefix is bit-identical, and the
float prefix differs only by float rounding, ~1e-6 nats). --independent reruns every T
from scratch instead. A text shorter than T is skipped at that T and said so.

Memory (4 GB card). Two things in the stock spec code do not fit T = 2048 next to the
weights, so this harness works around both from the outside (values untouched, and [0]
gates the result bit-for-bit against the stock forward_seq; --stock turns both off):
  * the stock gemv caches every weight scale as two int64 (~1.0 GB on the 8B); LeanGemv
    keeps the same integers as int16 + int8 (~0.2 GB) and uses smaller row blocks
  * _layers builds the new K/V history with torch.cat while the old one is still
    referenced (2x the KV cache, 1.2 GB at T = 2048); _Drain lets go of the old history
    layer by layer

Verdict (R1_LONGCTX_VERDICT PASS|FAIL). PASS needs all of:
  * [0] holds
  * pooled top-1 >= 0.99 at each T (at least one text must reach it)
  * no downward trend in the pooled bucket top-1 at each T, defined as BOTH
      last bucket >= mean of the earlier buckets - 1.0 pp, and
      least-squares fit over the buckets drops < 1.0 pp from first to last
    (--trend-pp). One token in a 256 bucket is 0.39 pp and a bucket's binomial sd is
    ~0.3 pp with 3 texts (~0.5 pp with one), so 1.0 pp is ~2-3 sd: smaller wiggles are
    noise. Trailing buckets under half size are reported but left out of the trend.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m fabric.big.ref.r1_longctx
    python -u -m fabric.big.ref.r1_longctx --check-chunked
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from .fabric_ref import ExactBfpGemv, FabricRef, fp16_decompose
from .qwen3_ref import FloatGemv, Qwen3Ref, from_gguf
from .r1b_sweep import eval_texts

CE = torch.nn.functional.cross_entropy


class LeanGemv(ExactBfpGemv):
    """ExactBfpGemv, same integers, less memory: decomposed scales cached as int16 mantissa
    (|m| <= 2047) + int8 exponent ([-24, 6]) instead of 2 x int64, and 4x smaller row blocks.
    torch promotes both to int64 inside mm(), so every product, shift and sum is unchanged."""
    ELEMS = 2 * 1024 * 1024

    def _scales(self, W):
        k = id(W)
        if k not in self._dec:
            m, e = fp16_decompose(W.scales)
            self._dec[k] = (m.to(torch.int16), e.to(torch.int8))
        return self._dec[k]


def free(dev):
    gc.collect()
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()


def peak(dev):
    """' peak GPU a/r GB' (allocated / reserved since the last call), '' on CPU."""
    if not str(dev).startswith("cuda"):
        return ""
    s = f", peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f}/{torch.cuda.max_memory_reserved() / 2**30:.2f} GB"
    torch.cuda.reset_peak_memory_stats()
    return s


def same(a, b):
    return bool(torch.equal(a, b) and torch.equal(a.view(torch.int32), b.view(torch.int32)))


class _Drain(list):
    """A K/V history that drops layer L-1 as soon as layer L is read (see module doc)."""

    def __getitem__(self, L):
        if isinstance(L, int) and L > 0:
            self[L - 1] = None
        return super().__getitem__(L)


def fab_chunked(fab, toks, chunk, label=None, drain=True):
    """fab.prefill_chunked, used as-is, with the bound _layers wrapped for the duration of
    the call: a progress line per chunk (if label) and the K/V history drained (if drain)."""
    inner, t0, done = fab._layers, time.time(), [0]

    def _layers(x, pos, kv_hist=None, *a, **k):
        if drain and kv_hist is not None:
            old, kv_hist = kv_hist, _Drain(kv_hist)
            old.clear()                     # prefill_chunked never reads the old list again
        r = inner(x, pos, kv_hist, *a, **k)
        done[0] += x.shape[0]
        if label:
            print(f"    {label} fabric {done[0]}/{len(toks)} ({time.time() - t0:.0f}s)")
        return r

    fab._layers = _layers
    try:
        return fab.prefill_chunked(toks, chunk, "cpu")
    finally:
        del fab._layers


# ----------------------------------------------------------------------------- [0]
def check_chunked(fab, flt, t, chunks, lean, dev, tol):
    r = dict(T=len(t), chunks=chunks, gemv="lean+drain" if lean is not None else "stock")
    t0 = time.time()
    ref = fab.forward_seq(t).cpu()                  # stock gemv: the path the R1c gate ran
    print(f"[0] fabric forward_seq T={len(t)} on the stock gemv ({time.time() - t0:.0f}s{peak(dev)})")
    if lean is not None:
        fab.gemv._dec.clear()
        fab.gemv = lean
        free(dev)
    r["chunked_eq_seq"] = {}
    for ch in chunks:
        t0 = time.time()
        q = fab_chunked(fab, t, ch, drain=lean is not None)
        eq = r["chunked_eq_seq"][str(ch)] = same(q, ref)
        bad = "" if eq else f"  max|d| {(q - ref).abs().max():.3e}, {int((q != ref).any(-1).sum())} positions differ"
        print(f"[0] prefill_chunked(chunk={ch}, {r['gemv']}) == forward_seq bit-identical: {eq}"
              f"{bad} ({time.time() - t0:.0f}s{peak(dev)})")
        del q
    h = len(t) // 2
    r["prefix_eq"] = same(fab_chunked(fab, t[:h], chunks[-1], drain=lean is not None), ref[:h])
    print(f"[0] prefill_chunked(t[:{h}]) == forward_seq(t)[:{h}] bit-identical: {r['prefix_eq']}")
    del ref
    free(dev)
    t0 = time.time()
    fs = flt.forward_seq(t).cpu()
    fc = flt.forward_seq_chunked(t, chunks[0], "cpu")
    r["float_chunked_max_abs"] = (fs - fc).abs().max().item()
    r["float_chunked_argmax_diff"] = int((fs.argmax(-1) != fc.argmax(-1)).sum())
    r["float_tol"] = tol
    print(f"[0] float forward_seq_chunked(chunk={chunks[0]}) vs forward_seq: max |dlogit| "
          f"{r['float_chunked_max_abs']:.2e} (tol {tol:g}), argmax differs at "
          f"{r['float_chunked_argmax_diff']}/{len(t)} ({time.time() - t0:.0f}s{peak(dev)})")
    del fs, fc
    free(dev)
    r["ok"] = bool(all(r["chunked_eq_seq"].values()) and r["prefix_eq"] and r["float_chunked_max_abs"] <= tol)
    return r


# ----------------------------------------------------------------------------- metrics
def per_position(f, q, toks, blk=64):
    """f, q [T, V] fp32 on the host -> per-position tensors over positions 0..T-2 (position i
    predicts token i+1), in float64 a block at a time. gap = float's logit preference for
    its own top-1 over the fabric's top-1 (0 where they agree)."""
    n = len(toks) - 1
    y = torch.tensor(toks[1:])
    o = dict(agree=torch.empty(n, dtype=torch.bool), **{k: torch.empty(n, dtype=torch.float64)
                                                         for k in ("nf", "nq", "cos", "gap")})
    for i in range(0, n, blk):
        j = min(i + blk, n)
        a, b = f[i:j].double(), q[i:j].double()
        af, aq = a.argmax(-1), b.argmax(-1)
        o["agree"][i:j] = af == aq
        o["nf"][i:j] = CE(a, y[i:j], reduction="none")
        o["nq"][i:j] = CE(b, y[i:j], reduction="none")
        o["cos"][i:j] = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        o["gap"][i:j] = a.gather(1, af[:, None])[:, 0] - a.gather(1, aq[:, None])[:, 0]
    return o


def pool(pps, lo, hi):
    """pooled stats over positions [lo, hi) of each per-position record."""
    c = {k: torch.cat([p[k][lo:hi] for p in pps]) for k in pps[0]}
    gap = c["gap"][~c["agree"]]
    return dict(n=len(c["agree"]), top1=c["agree"].double().mean().item(),
                dnll=(c["nq"] - c["nf"]).mean().item(), cos_mean=c["cos"].mean().item(),
                cos_min=c["cos"].min().item(), float_nll=c["nf"].mean().item(),
                n_disagree=len(gap), gap_med=gap.median().item() if len(gap) else 0.0,
                gap_max=gap.max().item() if len(gap) else 0.0)


def fmt(s):
    return (f"n={s['n']} top1 {s['top1']:.4f} dNLL {s['dnll']:+.4f} cos mean {s['cos_mean']:.6f} "
            f"min {s['cos_min']:.5f}")


def trend(bk, full_n, tol_pp):
    """trend of the pooled bucket top-1 (see module doc); slopes are per bucket."""
    use = [b for b in bk if b["n"] >= full_n / 2]
    if len(use) < 2:
        return dict(n_buckets=len(use), ok=True, note="fewer than 2 buckets: no trend to test")
    x = np.arange(len(use))
    t1 = np.array([100 * b["top1"] for b in use])
    sl = lambda k, m=1.0: float(np.polyfit(x, [m * b[k] for b in use], 1)[0])
    r = dict(n_buckets=len(use), top1_slope_pp=sl("top1", 100), dnll_slope=sl("dnll"), cos_slope=sl("cos_mean"),
             top1_first_pp=float(t1[0]), top1_last_pp=float(t1[-1]),
             last_vs_earlier_pp=float(t1[-1] - t1[:-1].mean()), tol_pp=tol_pp)
    r["fit_first_to_last_pp"] = r["top1_slope_pp"] * (len(use) - 1)
    r["ok"] = bool(r["last_vs_earlier_pp"] >= -tol_pp and r["fit_first_to_last_pp"] >= -tol_pp)
    return r


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="data/big/Bonsai-8B-Q1_0.gguf")
    ap.add_argument("--tok", default="data/big/Bonsai-8B-meta")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--T", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--chunk", type=int, default=64, help="fabric prefill chunk (query positions per pass)")
    ap.add_argument("--fchunk", type=int, default=128, help="float reference chunk")
    ap.add_argument("--bucket", type=int, default=256, help="position bucket size")
    ap.add_argument("--texts", nargs="+", help="subset of: tinystories readme master_doc")
    ap.add_argument("--out", default="fabric/big/results/r1_longctx.json")
    ap.add_argument("--check-chunked", action="store_true",
                    help="run only the [0] invariants (written to <out>_check.json)")
    ap.add_argument("--check-T", type=int, default=256)
    ap.add_argument("--check-chunks", type=int, nargs="+", default=[64, 100])
    ap.add_argument("--independent", action="store_true", help="rerun every T instead of slicing the longest run")
    ap.add_argument("--stock", action="store_true",
                    help="no memory workarounds: stock ExactBfpGemv, undrained K/V (+~0.8 GB, +1x KV)")
    ap.add_argument("--trend-pp", type=float, default=1.0)
    ap.add_argument("--float-tol", type=float, default=1e-3)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    dev, Ts = a.device, sorted(set(a.T))
    t0 = time.time()
    cfg, W, _ = from_gguf(a.gguf, device=dev)
    flt = Qwen3Ref(cfg, W, device=dev, gemv=FloatGemv())          # shares W with fab
    fab = FabricRef(cfg, W, device=dev, max_pos=max(4096, Ts[-1]))
    print(f"loaded {Path(a.gguf).name} on {dev} ({time.time() - t0:.0f}s{peak(dev)})")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    texts, _ = eval_texts(tok, 1 << 30)             # whole texts; cut to T below
    res = dict(model=Path(a.gguf).name, device=dev, T=Ts, chunk=a.chunk, fchunk=a.fchunk, bucket=a.bucket,
               independent=a.independent, top1_min=0.99, trend_pp=a.trend_pp)

    # [0] invariants
    res["check"] = ck = check_chunked(fab, flt, texts[0][1][:a.check_T], a.check_chunks,
                                      None if a.stock else LeanGemv(), dev, a.float_tol)
    print(f"R1_LONGCTX_CHUNKED_CHECK {'PASS' if ck['ok'] else 'FAIL'}")
    if a.check_chunked:
        out.with_name(out.stem + "_check.json").write_text(json.dumps(res, indent=1))
        return
    fab.gemv.stats = type(fab.gemv.stats)()         # GEMV stats below cover the long runs only

    # [1] one float pass + one fabric pass per text (per T with --independent)
    if a.texts:
        texts = [(n, t) for n, t in texts if n in a.texts]
    rec, res["texts"] = {}, {}
    for name, ids in texts:
        fit = [T for T in Ts if T <= len(ids)]
        for T in Ts:
            if T > len(ids):
                print(f"[1] {name}: only {len(ids)} tokens, skipped at T={T}")
        rec[name], res["texts"][name] = {}, dict(len=len(ids), runs={})
        for Tr in (fit if a.independent else fit[-1:]):
            t = ids[:Tr]
            lab = f"{name} T={Tr}"
            t0 = time.time()
            f = flt.forward_seq_chunked(t, a.fchunk, "cpu", progress=lambda d, n: print(
                f"    {lab} float {d}/{n} ({time.time() - t0:.0f}s)"))
            tf, pf = time.time() - t0, peak(dev)
            free(dev)
            t0 = time.time()
            q = fab_chunked(fab, t, a.chunk, lab, not a.stock)
            tq, pq = time.time() - t0, peak(dev)
            free(dev)
            pp = rec[name][Tr] = per_position(f, q, t)
            del f, q
            s = pool([pp], 0, Tr - 1)
            s.update(float_s=round(tf, 1), fabric_s=round(tq, 1),
                     disagree_pos=torch.nonzero(~pp["agree"])[:, 0].tolist())
            res["texts"][name]["runs"][str(Tr)] = s
            print(f"[1] {lab:17s} {fmt(s)} (float {tf:.0f}s{pf}; fabric {tq:.0f}s{pq})")
            out.write_text(json.dumps(res, indent=1))           # partial, in case a later run dies

    # [2] per T: pooled / per text / per bucket / trend
    res["by_T"], ok = {}, ck["ok"]
    for T in Ts:
        pps = {n: (r[T] if T in r else r[max(r)]) for n, r in rec.items() if r and max(r) >= T}
        if not pps:
            res["by_T"][str(T)] = dict(ok=False, note="no text reaches this length")
            print(f"[2] T={T}: no text reaches this length -> FAIL")
            ok = False
            continue
        L = list(pps.values())
        p = pool(L, 0, T - 1)
        print(f"[2] T={T} pooled {fmt(p)} ({len(L)} texts; {p['n_disagree']} disagreements, "
              f"float logit gap median {p['gap_med']:.3f} max {p['gap_max']:.3f})")
        bk = []
        for lo in range(0, T - 1, a.bucket):
            hi = min(lo + a.bucket, T - 1)
            b = dict(lo=lo, hi=hi, **pool(L, lo, hi), per_text_top1={n: pool([x], lo, hi)["top1"] for n, x in pps.items()})
            bk.append(b)
            print(f"    bucket {lo:5d}-{hi:5d} {fmt(b)} | " + " ".join(f"{n} {v:.4f}" for n, v in b["per_text_top1"].items()))
        tr = trend(bk, a.bucket * len(L), a.trend_pp)
        if "note" in tr:
            print(f"    trend: {tr['note']}")
        else:
            print(f"    trend over {tr['n_buckets']} buckets: top1 slope {tr['top1_slope_pp']:+.3f} pp/bucket "
                  f"(fit first->last {tr['fit_first_to_last_pp']:+.2f} pp), last vs earlier mean "
                  f"{tr['last_vs_earlier_pp']:+.2f} pp, first {tr['top1_first_pp']:.2f} last {tr['top1_last_pp']:.2f}; "
                  f"dNLL slope {tr['dnll_slope']:+.5f}/bucket, cos slope {tr['cos_slope']:+.2e}/bucket "
                  f"-> {'ok' if tr['ok'] else 'DOWNWARD'}")
        okT = bool(p["top1"] >= 0.99 and tr["ok"])
        res["by_T"][str(T)] = dict(texts=list(pps), pooled=p, per_text={n: pool([x], 0, T - 1) for n, x in pps.items()},
                                   buckets=bk, trend=tr, ok=okT)
        ok = ok and okT
    st = fab.gemv.stats
    res["gemv"] = dict(kind=ck["gemv"], min_shift=st.min_shift, max_shift=st.max_shift,
                       floor_frac=st.floor_terms / max(1, st.terms), max_acc_bits=st.max_acc_bits)
    print(f"[3] GEMV shifts [{st.min_shift}..{st.max_shift}] floor-terms {res['gemv']['floor_frac']:.2e} "
          f"peak acc bits {st.max_acc_bits}/64")
    res["verdict"] = "PASS" if ok else "FAIL"
    out.write_text(json.dumps(res, indent=1))
    print(f"R1_LONGCTX_VERDICT {res['verdict']}")


if __name__ == "__main__":
    main()
