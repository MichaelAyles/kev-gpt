"""R3a: the group scales (doc 10, rung R3). 0.125 of Bonsai-8B's 1.125 bits/weight is the fp16
scale of every 128-group: 11% of the bytes read per token. This tool measures how small that
can get, per scheme, and what each lossy scheme costs in model quality.

  stage stats   statistics of the real 8B scales (per matrix, per kind, overall): distinct
                values, entropies (raw code, exponent, mantissa, conditional on the previous
                group / the previous row), separable / low-rank fits in the log domain.
  stage bytes   every candidate scheme: EXACT bytes per token for the full stream order
                (36 layers x q,k,v,o,gate,up,down + lm_head), in two stream formats, plus the
                scale-domain error it introduces.
  stage eval    float-reference teacher-forced NLL of every LOSSY scheme on the R1c/R1d token
                ids (4 texts, T = 256, 1,020 predicted tokens): difference vs stock binary-8B
                with paired SE, pooled and without chat_selfgen, top-1 agreement with the stock
                argmax. Same signs, replaced scales, FloatGemv unchanged. GATE: the stock model
                must first reproduce results/r1c_gate.json float_nll to 1e-6.
  stage report  the summary table, R3A_VERDICT and the one-line recommendation.

Every decoded scale is an fp16 value (the engine multiplies by fp16_decompose's (m_d, e_d)), so
any scheme here needs no change to the GEMV arithmetic, only to how the scale reaches it.

Stream formats (bytes are exact, 128-bit beats, NPORTS = 3):
  rowal   today's rule (pack_weights.py): every row = its scale beats + its sign beats, padded
          to a whole tuple (3 beats). No RTL change beyond the scale decode.
  packed  scale codes packed across rows (floor(128/bits) codes per beat, a code never straddles
          a beat), matrices padded to a tuple. Needs the engine to let rows start mid-tuple.

    CUDA_VISIBLE_DEVICES="" python -u -m fabric.big.ref.r3a_scales --stage stats bytes
    flock /tmp/kevbuild/gpu.lock python -u -m fabric.big.ref.r3a_scales --stage eval
    python -u -m fabric.big.ref.r3a_scales --stage report
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BW = 15.54e9                       # R0 MEASURED PL<-DDR bytes/s
KV_CTX512 = 36 * 2376 * 512        # bytes of KV read per token at ctx 512 (doc 10 section 1)
STREAM_ORDER = ("q", "k", "v", "o", "gate", "up", "down")
GGN = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output",
       "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}
NPORTS, CHUNK = 3, 256
NLOG = 64                          # log-code resolution: codes per octave (sep / log-add decode)
MANT = np.round(1024.0 * 2.0 ** (np.arange(NLOG) / NLOG))   # 11-bit mantissas 1024..2047


# ----------------------------------------------------------------------------- data
def kind_of(name: str) -> str:
    return name.split(".")[-1]


def stream_names(n_layer: int):
    return [f"l{L}.{k}" for L in range(n_layer) for k in STREAM_ORDER] + ["lm_head"]


def load_scales(path):
    """{short name: fp16 [R, G]} for every matrix streamed once per token, in stream order."""
    from .gguf_q1 import GGUF, QK
    g = GGUF(path)
    n_layer = int(g.meta[g.meta["general.architecture"] + ".block_count"])
    out = {}
    for nm in stream_names(n_layer):
        if nm == "lm_head":
            gn = "output.weight" if "output.weight" in g.tensors else "token_embd.weight"
        else:
            L, k = nm[1:].split(".")
            gn = f"blk.{L}.{GGN[k]}.weight"
        ti = g.tensors[gn]
        assert g.is_q1(gn), gn
        d, _ = g.q1_parts(gn)
        out[nm] = d.reshape(ti.n // ti.shape[0], ti.shape[0] // QK)
    return out


# ----------------------------------------------------------------------------- entropy helpers
def H_counts(c, mm=False):
    """order-0 entropy in bits of a count vector (plug-in; mm = Miller-Madow bias corrected)."""
    c = np.asarray(c, dtype=np.float64)
    c = c[c > 0]
    n = c.sum()
    p = c / n
    h = float(-(p * np.log2(p)).sum())
    if mm:
        h += (len(c) - 1) / (2 * n * math.log(2))
    return h


def H_of(x, mm=False):
    _, c = np.unique(np.asarray(x).ravel(), return_counts=True)
    return H_counts(c, mm)


def H_cond(x, y, mm=True):
    """H(X | Y) = H(X, Y) - H(Y) for small-alphabet integer arrays."""
    x = np.asarray(x).ravel().astype(np.int64)
    y = np.asarray(y).ravel().astype(np.int64)
    j = (y - y.min()) * (x.max() - x.min() + 1) + (x - x.min())
    return H_of(j, mm) - H_of(y, mm)


def bucket16(k):
    """16 equal-population buckets of an integer code array (the conditioning context)."""
    q = np.quantile(k.ravel(), np.linspace(0, 1, 17)[1:-1])
    return np.searchsorted(q, k, side="right")


# ----------------------------------------------------------------------------- stage: stats
def sep_fit(L):
    """least-squares additive fit L[r, g] ~ la[r] + lb[g] (= rank-1 separable scale a[r] * b[g]
    in the linear domain). Complete matrix: closed form."""
    la = L.mean(1)
    lb = L.mean(0) - L.mean()
    return la, lb


def matrix_stats(S):
    R, G = S.shape
    u = S.view(np.uint16)
    f = S.astype(np.float64)
    L = np.log2(f)
    k = (u >> 3).astype(np.int64)                       # monotone bf16 code (if low 3 bits are 0)
    vals, cnt = np.unique(u, return_counts=True)
    st = dict(R=R, G=G, n=R * G, distinct=int(len(vals)), H0=H_counts(cnt), H0_mm=H_counts(cnt, True),
              min=float(f.min()), max=float(f.max()), n_neg=int((u >> 15).sum()), n_zero=int((f == 0).sum()),
              low3_nonzero=int(((u & 7) != 0).sum()),
              H_exp=H_of((u >> 10) & 31), H_mant=H_of(u & 1023), H_mant7=H_of((u >> 3) & 127),
              n_exp=int(len(np.unique((u >> 10) & 31))),
              exp_min=int(((u >> 10) & 31).min()), exp_max=int(((u >> 10) & 31).max()),
              log2_mean=float(L.mean()), log2_std=float(L.std()))
    # neighbours: delta of the monotone code, and H(X | 16-bucket context)
    st["H_dleft"] = H_of(k[:, 1:] - k[:, :-1], True)
    st["H_dup"] = H_of(k[1:] - k[:-1], True)
    b = bucket16(k)
    st["H_given_left16"] = H_cond(k[:, 1:], b[:, :-1])
    st["H_given_up16"] = H_cond(k[1:], b[:-1])
    st["H_given_left16_up16"] = H_cond(k[1:, 1:], b[1:, :-1] * 16 + b[:-1, 1:])
    # separable / low-rank structure in the log domain
    la, lb = sep_fit(L)
    res = L - la[:, None] - lb[None, :]
    tot = L.std()
    st["rms_const"] = float(tot)
    st["rms_row"] = float((L - L.mean(1, keepdims=True)).std())
    st["rms_col"] = float((L - L.mean(0, keepdims=True)).std())
    st["rms_sep"] = float(np.sqrt((res ** 2).mean()))
    st["row_std"], st["col_std"] = float(la.std()), float(lb.std())
    sv = np.linalg.svd(L, compute_uv=False)
    e = np.sqrt(np.maximum(0, (sv ** 2).sum() - np.cumsum(sv ** 2)))    # ||L - L_k||_F
    fro = math.sqrt((sv ** 2).sum())
    for r in (1, 2, 3):
        st[f"svd{r}_rms"] = float(e[r - 1] / math.sqrt(L.size))
        st[f"svd{r}_relfro"] = float(e[r - 1] / fro)
    sl = np.linalg.svd(f, compute_uv=False)                             # linear domain
    el = np.sqrt(np.maximum(0, (sl ** 2).sum() - np.cumsum(sl ** 2)))
    st["lin_svd1_rel"], st["lin_svd2_rel"] = float(el[0] / math.sqrt((sl ** 2).sum())), float(el[1] / math.sqrt((sl ** 2).sum()))
    st["sep_rel_of_std"] = float(st["rms_sep"] / tot)
    # is the separable residual white? lag-1 correlations along the row and down the column
    st["res_corr_left"] = float(np.corrcoef(res[:, 1:].ravel(), res[:, :-1].ravel())[0, 1])
    st["res_corr_up"] = float(np.corrcoef(res[1:].ravel(), res[:-1].ravel())[0, 1])
    st["res_kurt"] = float(((res / res.std()) ** 4).mean())
    # lossless with the separable predictor: residual of the monotone code vs the predicted code
    pred = np.float16(2.0 ** (la[:, None] + lb[None, :])).view(np.uint16).astype(np.int64) >> 3
    st["H_sep_resid"] = H_of(k - pred, True)
    return st, cnt, vals


def stage_stats(scales, doc):
    t0 = time.time()
    per, kinds = {}, {}
    pooled = {}                                    # kind -> {u16 value: count}
    for nm, S in scales.items():
        st, cnt, vals = matrix_stats(S)
        per[nm] = st
        kd = kind_of(nm)
        d = pooled.setdefault(kd, np.zeros(65536, np.int64))
        d[vals] += cnt
    keys = ["H0", "H0_mm", "H_exp", "H_mant", "H_mant7", "H_dleft", "H_dup", "H_given_left16", "H_given_up16",
            "H_given_left16_up16", "H_sep_resid", "rms_const", "rms_row", "rms_col", "rms_sep", "row_std",
            "col_std", "svd1_rms", "svd2_rms", "svd3_rms", "svd1_relfro", "svd2_relfro", "lin_svd1_rel",
            "lin_svd2_rel", "sep_rel_of_std", "res_corr_left", "res_corr_up", "res_kurt", "log2_mean", "log2_std"]

    def agg(names):
        w = np.array([per[n]["n"] for n in names], dtype=np.float64)
        r = {k: float((np.array([per[n][k] for n in names]) * w).sum() / w.sum()) for k in keys}
        r.update(n_matrices=len(names), n_scales=int(w.sum()),
                 distinct_min=min(per[n]["distinct"] for n in names),
                 distinct_max=max(per[n]["distinct"] for n in names),
                 n_exp_max=max(per[n]["n_exp"] for n in names),
                 min=min(per[n]["min"] for n in names), max=max(per[n]["max"] for n in names),
                 n_neg=sum(per[n]["n_neg"] for n in names), n_zero=sum(per[n]["n_zero"] for n in names),
                 low3_nonzero=sum(per[n]["low3_nonzero"] for n in names))
        return r

    for kd in list(STREAM_ORDER) + ["lm_head"]:
        names = [n for n in per if kind_of(n) == kd]
        kinds[kd] = agg(names)
        c = pooled[kd]
        kinds[kd]["pooled_distinct"] = int((c > 0).sum())
        kinds[kd]["pooled_H0"] = H_counts(c)
    allc = sum(pooled.values())
    overall = agg(list(per))
    overall["pooled_distinct"] = int((allc > 0).sum())
    overall["pooled_H0"] = H_counts(allc)
    # distribution of the fp16 values, overall: quantiles of log2(scale)
    nz = np.nonzero(allc)[0]
    v = np.array(nz, dtype=np.uint16).view(np.float16).astype(np.float64)
    o = np.argsort(v)
    cdf = np.cumsum(allc[nz][o]) / allc.sum()
    overall["quantiles"] = {str(q): float(v[o][np.searchsorted(cdf, q)]) for q in (0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999)}
    doc["stats"] = dict(per_kind=kinds, overall=overall, per_matrix=per,
                        note="entropies in bits per scale, weighted by scale count across matrices "
                             "(per-matrix models); rms_* in log2 units (octaves); *_mm / conditional "
                             "entropies carry the Miller-Madow correction")
    print(f"[stats] {len(per)} matrices, {overall['n_scales']:,} scales ({time.time() - t0:.0f}s)")
    hdr = (f"{'kind':8s} {'scales':>10s} {'distinct':>9s} {'H0':>5s} {'Hexp':>5s} {'Hmant':>5s} {'H|left':>6s} "
           f"{'H|up':>5s} {'H|l,u':>5s} {'Hdl':>5s} {'Hdu':>5s} {'Hsep':>5s} | {'std':>5s} {'-row':>5s} {'-col':>5s} "
           f"{'sep':>5s} {'svd1':>5s} {'svd2':>5s} {'sep/std':>7s} {'rc_l':>6s} {'rc_u':>6s}")
    print(hdr)
    for kd, r in list(kinds.items()) + [("ALL", overall)]:
        print(f"{kd:8s} {r['n_scales']:10,d} {r['distinct_min']:4d}-{r['distinct_max']:<4d} {r['H0']:5.2f} "
              f"{r['H_exp']:5.2f} {r['H_mant']:5.2f} {r['H_given_left16']:6.2f} {r['H_given_up16']:5.2f} "
              f"{r['H_given_left16_up16']:5.2f} {r['H_dleft']:5.2f} {r['H_dup']:5.2f} {r['H_sep_resid']:5.2f} | "
              f"{r['rms_const']:5.3f} {r['rms_row']:5.3f} {r['rms_col']:5.3f} {r['rms_sep']:5.3f} "
              f"{r['svd1_rms']:5.3f} {r['svd2_rms']:5.3f} {r['sep_rel_of_std']:7.3f} {r['res_corr_left']:+6.3f} "
              f"{r['res_corr_up']:+6.3f}")
    print(f"        fp16 sign bit set: {overall['n_neg']}, zero scales: {overall['n_zero']}, low-3 mantissa bits "
          f"non-zero: {overall['low3_nonzero']} (0 = every scale is a bf16 value), exponent fields per matrix <= "
          f"{overall['n_exp_max']}, range {overall['min']:.3g}..{overall['max']:.3g}")


# ----------------------------------------------------------------------------- quantisers
def lloyd_1d(x, w, K, iters=200):
    """weighted 1-D Lloyd on points x (sorted ascending) with weights w -> K centroids."""
    if len(x) <= K:
        return x.copy()
    cw = np.cumsum(w) / w.sum()
    c = np.interp((np.arange(K) + 0.5) / K, cw, x)
    c = np.unique(c)
    while len(c) < K:                               # split the widest gap until K distinct
        i = np.argmax(np.diff(c)) if len(c) > 1 else 0
        c = np.sort(np.append(c, (c[i] + c[min(i + 1, len(c) - 1)]) / 2 + 1e-9 * (len(c) == 1)))
    for _ in range(iters):
        a = np.searchsorted((c[1:] + c[:-1]) / 2, x)
        sw = np.bincount(a, w, K)
        sx = np.bincount(a, w * x, K)
        n = np.where(sw > 0, sx / np.maximum(sw, 1e-300), c)
        n = np.sort(n)
        if np.allclose(n, c, atol=1e-12):
            break
        c = n
    return c


def assign(x, c):
    return np.searchsorted((c[1:] + c[:-1]) / 2, x)


def hist(S):
    vals, inv, cnt = np.unique(S.view(np.uint16), return_inverse=True, return_counts=True)
    return vals.view(np.float16).astype(np.float64), inv.reshape(S.shape), cnt.astype(np.float64)


def q_codebook(S, bits):
    """per-matrix 2^bits-entry codebook, Lloyd in the log domain, centroids rounded to fp16."""
    v, inv, cnt = hist(S)
    x = np.log2(v)
    c = lloyd_1d(x, cnt, 1 << bits)
    cb = np.float16(2.0 ** c)
    return cb[assign(x, c)][inv]


def q_uniform(S, bits):
    """per-matrix uniform levels in log2, MSE-optimal step/offset (outliers clamp), decoded to
    fp16."""
    v, inv, cnt = hist(S)
    x = np.log2(v)
    K = 1 << bits
    lo, hi = x.min(), x.max()
    if len(x) <= 1 or hi == lo:
        return S.copy()
    best = None
    full = (hi - lo) / (K - 1)
    for step in full * np.linspace(0.15, 1.0, 35):
        span = step * (K - 1)
        for base in np.linspace(lo, hi - span, 25) if hi - span > lo else [lo]:
            kq = np.clip(np.round((x - base) / step), 0, K - 1)
            e = (cnt * (base + kq * step - x) ** 2).sum()
            if best is None or e < best[0]:
                best = (e, step, base)
    _, step, base = best
    kq = np.clip(np.round((x - base) / step), 0, K - 1)
    return np.float16(2.0 ** (base + kq * step))[inv]


def q_ulog(S, bits):
    """per-matrix uniform levels in log2 across the FULL range [min, max] (no clipping: the
    minimax quantiser, max relative error = half a step everywhere), decoded to fp16."""
    v, inv, cnt = hist(S)
    x = np.log2(v)
    K = 1 << bits
    lo, hi = x.min(), x.max()
    if hi == lo:
        return S.copy()
    step = (hi - lo) / (K - 1)
    return np.float16(2.0 ** (lo + np.round((x - lo) / step) * step))[inv]


def compand_levels(x, w, K, alpha):
    """K levels with point density proportional to (histogram density)^alpha over [min, max]:
    alpha = 0 is uniform (minimax), alpha = 1/3 is the MSE-optimal density. Both ends are levels."""
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.array([lo])
    nb = 512
    h, edges = np.histogram(x, bins=nb, range=(lo, hi), weights=w)
    h = np.convolve(h, np.ones(9) / 9, mode="same") + 1e-6 * h.sum()
    lam = h ** alpha
    cdf = np.concatenate([[0], np.cumsum(lam)]) / lam.sum()
    return np.interp(np.arange(K) / (K - 1), cdf, edges)


def q_compand(S, bits, alpha):
    v, inv, cnt = hist(S)
    x = np.log2(v)
    c = np.unique(compand_levels(x, cnt, 1 << bits, alpha))
    if len(c) == 1:
        return S.copy()
    return np.float16(2.0 ** c)[assign(x, c)][inv]


def q_clip(S, pct, side):
    """DIAGNOSTIC: clamp only the top / bottom pct percent of each matrix's scales, rest exact."""
    a = S.astype(np.float64)
    if side == "hi":
        return np.float16(np.minimum(a, np.percentile(a, 100 - pct, method="lower")))
    return np.float16(np.maximum(a, np.percentile(a, pct, method="higher")))


def log_decode(c):
    """integer log code c (NLOG per octave) -> fp16: MANT[c mod NLOG] * 2^(c div NLOG - 10)."""
    c = np.asarray(c, dtype=np.int64)
    return np.float16(MANT[np.mod(c, NLOG)] / 1024.0 * 2.0 ** np.floor_divide(c, NLOG))


def q_sep(S, rbits, mode="sep", resid="lloyd", alpha=0.0):
    """separable scale in integer log codes: c[r, g] = ia[r] + ib[g] (+ off[k[r, g]] with an
    rbits-bit residual index into a per-matrix table of integer code offsets), decoded through
    the NLOG-entry mantissa table. mode: sep | row (ib = 0) | const (one value per matrix)."""
    L = NLOG * np.log2(S.astype(np.float64))
    if mode == "const":
        return np.broadcast_to(log_decode(np.round(L.mean())), S.shape).copy()
    la = L.mean(1)
    lb = (L.mean(0) - L.mean()) if mode == "sep" else np.zeros(S.shape[1])
    ia, ib = np.round(la).astype(np.int64), np.round(lb).astype(np.int64)
    c = ia[:, None] + ib[None, :]
    if rbits:
        e = L - c
        if resid == "lloyd":       # MSE-optimal levels: the tails clamp
            h, edges = np.histogram(e, bins=np.arange(math.floor(e.min()) - 0.125, e.max() + 0.5, 0.25))
            off = np.round(lloyd_1d((edges[:-1] + edges[1:]) / 2, h.astype(np.float64), 1 << rbits))
        else:                      # companded over the full residual range (alpha = 0: uniform)
            off = np.round(compand_levels(e.ravel(), None, 1 << rbits, alpha))
        off = np.unique(off)
        c = c + off[assign(e, off)].astype(np.int64)
    return log_decode(c)


def q_bfp(S, bits, blk=32):
    """block floating point for scales: every `blk` consecutive groups of a row share a header
    (10-bit log code of the block's LARGEST scale + 3-bit shift); each group stores a `bits`-bit
    code k and decodes to log code  top - (k << shift)  (NLOG codes per octave) through the
    mantissa ROM. The largest scale of every block is kept to half a log code (0.54%): the top
    tail is where the model is sensitive (clip_hi diagnostics); the bottom of a block clamps.
    The shift is the one of 0..7 with the least squared error of the block's scales in the LINEAR
    domain (= least L2 weight error, which protects the large scales)."""
    R, G = S.shape
    assert G % blk == 0
    L = (NLOG * np.log2(S.astype(np.float64))).reshape(R, G // blk, blk)
    top = np.round(L.max(-1, keepdims=True))
    best_e = np.full(top.shape, np.inf)
    best = np.zeros(L.shape)
    for sh in range(8):
        rec = top - np.clip(np.round((top - L) / (1 << sh)), 0, (1 << bits) - 1) * (1 << sh)
        e = ((2.0 ** (rec / NLOG) - 2.0 ** (L / NLOG)) ** 2).sum(-1, keepdims=True)
        m = e < best_e
        best = np.where(m, rec, best)
        best_e = np.where(m, e, best_e)
    return log_decode(best.reshape(R, G))


def q_group(S, k):
    """one scale per 128*k weights: the L2-optimal shared scale is the mean of the member scales."""
    R, G = S.shape
    m = S.astype(np.float64).reshape(R, G // k, k).mean(-1)
    return np.repeat(np.float16(m), k, axis=1)


def q_bf16(S):
    return S.copy()


def q_noise(S, sigma, mode):
    """DIAGNOSTIC (not a codec): multiply the scales by exp(sigma * N(0, 1)), drawn per group /
    per row / per group column / per matrix, to see where the model's sensitivity lives."""
    import zlib
    rng = np.random.default_rng(zlib.crc32(S.tobytes()[:4096]) + S.size)
    R, G = S.shape
    shape = dict(grp=(R, G), row=(R, 1), col=(1, G), mat=(1, 1))[mode]
    return np.float16(S.astype(np.float64) * np.exp(sigma * rng.standard_normal(shape)))


DIAG = {f"noise_{m}{pc}": (lambda S, m=m, pc=pc: q_noise(S, float(pc) / 100, m))
        for m in ("grp", "row", "col", "mat") for pc in ("0.3", "1", "3", "6", "10")}
DIAG.update({f"clip_{sd}{pc}": (lambda S, sd=sd, pc=pc: q_clip(S, float(pc), sd))
             for sd in ("hi", "lo") for pc in ("0.01", "0.1", "1")})


def q_noise_top(S, sigma, pct):
    """DIAGNOSTIC: group noise on the top pct percent of each matrix's scales only."""
    a = S.astype(np.float64)
    N = q_noise(S, sigma, "grp")
    return np.where(a >= np.percentile(a, 100 - pct), N, S)


DIAG.update({f"noisetop{pc}_{sg}": (lambda S, pc=pc, sg=sg: q_noise_top(S, float(sg) / 100, float(pc)))
             for pc in ("0.1", "1") for sg in ("3", "10")})
TARGETS = {"ffn": ("gate", "up", "down"), "attn": ("q", "k", "v", "o")}


def scheme_fn(name):
    """'base' or 'base@target' (target = a matrix kind, 'ffn' or 'attn') -> fn(S, matrix name)."""
    base, _, tgt = name.partition("@")
    fn = SCHEMES[base]["fn"] if base in SCHEMES else DIAG[base]
    kinds = TARGETS.get(tgt, (tgt,))
    return lambda S, nm: fn(S) if not tgt or kind_of(nm) in kinds else S


# name -> (fn, lossy, bits per scale, weights per scale, row bits, header bits fn(G), decoder)
def _cb_hdr(bits):
    return lambda G: (1 << bits) * 16


SCHEMES = {
    "stock_fp16": dict(fn=q_bf16, lossy=False, bits=16, gw=128, row=0, hdr=lambda G: 0,
                       dec="none (today's stream)"),
    "bf16_12": dict(fn=q_bf16, lossy=False, bits=12, gw=128, row=0, hdr=lambda G: 0,
                    dec="wire only: 5-bit exponent + 7-bit mantissa (sign and low 3 mantissa bits are always 0)"),
    "bf16_10": dict(fn=q_bf16, lossy=False, bits=10, gw=128, row=0, hdr=lambda G: 16,
                    dec="3-bit exponent offset + 7-bit mantissa; one 5-bit add of a per-matrix exponent base"),
    "dict_lossless": dict(fn=q_bf16, lossy=False, bits="dict", gw=128, row=0, hdr="dict",
                          dec="per-matrix dictionary of its distinct fp16 values: 1024 x 16 b RAM per lane, loaded "
                              "from a matrix header"),
    "cb8": dict(fn=lambda S: q_codebook(S, 8), lossy=True, bits=8, gw=128, row=0, hdr=_cb_hdr(8),
                dec="per-matrix 256 x 16 b LUT (Lloyd, log domain), loaded from a 512 B matrix header"),
    "cb6": dict(fn=lambda S: q_codebook(S, 6), lossy=True, bits=6, gw=128, row=0, hdr=_cb_hdr(6),
                dec="per-matrix 64 x 16 b LUT, 128 B header"),
    "cb4": dict(fn=lambda S: q_codebook(S, 4), lossy=True, bits=4, gw=128, row=0, hdr=_cb_hdr(4),
                dec="per-matrix 16 x 16 b LUT, 32 B header"),
    "cb3": dict(fn=lambda S: q_codebook(S, 3), lossy=True, bits=3, gw=128, row=0, hdr=_cb_hdr(3),
                dec="per-matrix 8 x 16 b LUT, 16 B header"),
    "cb2": dict(fn=lambda S: q_codebook(S, 2), lossy=True, bits=2, gw=128, row=0, hdr=_cb_hdr(2),
                dec="per-matrix 4 x 16 b LUT, 8 B header"),
    "u8": dict(fn=lambda S: q_uniform(S, 8), lossy=True, bits=8, gw=128, row=0, hdr=_cb_hdr(8),
               dec="uniform log levels: per-matrix 256 x 16 b LUT (or base + k*step into a mantissa ROM)"),
    "u6": dict(fn=lambda S: q_uniform(S, 6), lossy=True, bits=6, gw=128, row=0, hdr=_cb_hdr(6),
               dec="uniform log levels: per-matrix 64 x 16 b LUT"),
    "u4": dict(fn=lambda S: q_uniform(S, 4), lossy=True, bits=4, gw=128, row=0, hdr=_cb_hdr(4),
               dec="uniform log levels: per-matrix 16 x 16 b LUT"),
    "ulog8": dict(fn=lambda S: q_ulog(S, 8), lossy=True, bits=8, gw=128, row=0, hdr=lambda G: 32,
                  dec="full-range uniform log: code * step + base (two per-matrix constants) into the mantissa "
                      "ROM, or a per-matrix 256 x 16 b LUT"),
    "ulog6": dict(fn=lambda S: q_ulog(S, 6), lossy=True, bits=6, gw=128, row=0, hdr=_cb_hdr(6),
                  dec="full-range uniform log: per-matrix 64 x 16 b LUT, 128 B header"),
    "ulog5": dict(fn=lambda S: q_ulog(S, 5), lossy=True, bits=5, gw=128, row=0, hdr=_cb_hdr(5),
                  dec="full-range uniform log: per-matrix 32 x 16 b LUT, 64 B header"),
    "ulog4": dict(fn=lambda S: q_ulog(S, 4), lossy=True, bits=4, gw=128, row=0, hdr=_cb_hdr(4),
                  dec="full-range uniform log: per-matrix 16 x 16 b LUT, 32 B header"),
    "ulog3": dict(fn=lambda S: q_ulog(S, 3), lossy=True, bits=3, gw=128, row=0, hdr=_cb_hdr(3),
                  dec="full-range uniform log: per-matrix 8 x 16 b LUT, 16 B header"),
    "cmp4": dict(fn=lambda S: q_compand(S, 4, 0.15), lossy=True, bits=4, gw=128, row=0, hdr=_cb_hdr(4),
                 dec="companded full-range levels (density^0.15): per-matrix 16 x 16 b LUT, 32 B header"),
    "cmp3": dict(fn=lambda S: q_compand(S, 3, 0.15), lossy=True, bits=3, gw=128, row=0, hdr=_cb_hdr(3),
                 dec="companded full-range levels: per-matrix 8 x 16 b LUT, 16 B header"),
    "sepu4": dict(fn=lambda S: q_sep(S, 4, resid="u"), lossy=True, bits=4, gw=128, row=16,
                  hdr=lambda G: G * 8 + 16 * 8, dec="sep0 + 4-bit full-range uniform residual (16-entry offset table)"),
    "sepu3": dict(fn=lambda S: q_sep(S, 3, resid="u"), lossy=True, bits=3, gw=128, row=16,
                  hdr=lambda G: G * 8 + 8 * 8, dec="sep0 + 3-bit full-range uniform residual (8-entry offset table)"),
    "sepu2": dict(fn=lambda S: q_sep(S, 2, resid="u"), lossy=True, bits=2, gw=128, row=16,
                  hdr=lambda G: G * 8 + 4 * 8, dec="sep0 + 2-bit full-range uniform residual (4-entry offset table)"),
    "sepc3": dict(fn=lambda S: q_sep(S, 3, resid="c", alpha=0.15), lossy=True, bits=3, gw=128, row=16,
                  hdr=lambda G: G * 8 + 8 * 8, dec="sep0 + 3-bit companded full-range residual (8-entry offset table)"),
    "sepc2": dict(fn=lambda S: q_sep(S, 2, resid="c", alpha=0.15), lossy=True, bits=2, gw=128, row=16,
                  hdr=lambda G: G * 8 + 4 * 8, dec="sep0 + 2-bit companded full-range residual (4-entry offset table)"),
    "bfp4": dict(fn=lambda S: q_bfp(S, 4), lossy=True, bits=4, gw=128, row=16, blk=32, hdr=lambda G: 0,
                 dec="as bfp3 with 4-bit codes (a header + 32 codes no longer fit one beat)"),
    "bfp3": dict(fn=lambda S: q_bfp(S, 3), lossy=True, bits=3, gw=128, row=16, blk=32, hdr=lambda G: 0,
                 dec="block floating point: one scale beat per 32 sign beats = 16-bit header (10-bit top log code, "
                     "3-bit shift) + 32 x 3-bit codes; per group one shift, one 10-bit subtract, 64-entry mantissa "
                     "ROM; no per-matrix tables"),
    "bfp2": dict(fn=lambda S: q_bfp(S, 2), lossy=True, bits=2, gw=128, row=16, blk=32, hdr=lambda G: 0,
                 dec="as bfp3 with 2-bit codes (80 bits per 32 groups)"),
    "sep0": dict(fn=lambda S: q_sep(S, 0), lossy=True, bits=0, gw=128, row=16, hdr=lambda G: G * 8,
                 dec="rank-1: 16-bit log code per row + 8-bit log code per group column (G-entry table per "
                     "matrix); one add, 64-entry mantissa ROM, exponent = high bits"),
    "sep1": dict(fn=lambda S: q_sep(S, 1), lossy=True, bits=1, gw=128, row=16, hdr=lambda G: G * 8 + 2 * 8,
                 dec="sep0 + 1-bit residual: second add of a 2-entry offset table"),
    "sep2": dict(fn=lambda S: q_sep(S, 2), lossy=True, bits=2, gw=128, row=16, hdr=lambda G: G * 8 + 4 * 8,
                 dec="sep0 + 2-bit residual: second add of a 4-entry offset table"),
    "sep3": dict(fn=lambda S: q_sep(S, 3), lossy=True, bits=3, gw=128, row=16, hdr=lambda G: G * 8 + 8 * 8,
                 dec="sep0 + 3-bit residual: second add of an 8-entry offset table"),
    "sep4": dict(fn=lambda S: q_sep(S, 4), lossy=True, bits=4, gw=128, row=16, hdr=lambda G: G * 8 + 16 * 8,
                 dec="sep0 + 4-bit residual: second add of a 16-entry offset table"),
    "row_only": dict(fn=lambda S: q_sep(S, 0, "row"), lossy=True, bits=0, gw=128, row=16, hdr=lambda G: 0,
                     dec="one 16-bit log code per row, mantissa ROM"),
    "matrix_const": dict(fn=lambda S: q_sep(S, 0, "const"), lossy=True, bits=0, gw=128, row=0, hdr=lambda G: 16,
                         dec="one scale per matrix (anchor: how much do the scales matter at all)"),
    "g256": dict(fn=lambda S: q_group(S, 2), lossy=True, bits=16, gw=256, row=0, hdr=lambda G: 0,
                 dec="none: one fp16 scale held for 2 sign beats"),
    "g512": dict(fn=lambda S: q_group(S, 4), lossy=True, bits=16, gw=512, row=0, hdr=lambda G: 0,
                 dec="none: one fp16 scale held for 4 sign beats"),
    "g1024": dict(fn=lambda S: q_group(S, 8), lossy=True, bits=16, gw=1024, row=0, hdr=lambda G: 0,
                  dec="none: one fp16 scale held for 8 sign beats"),
    "g256_cb8": dict(fn=lambda S: q_codebook(q_group(S, 2), 8), lossy=True, bits=8, gw=256, row=0, hdr=_cb_hdr(8),
                     dec="256 x 16 b LUT, one code held for 2 sign beats"),
}


# ----------------------------------------------------------------------------- bytes
def scheme_bytes(shapes, bits, gw=128, row=0, hdr=lambda G: 0, distinct=None, blk=0):
    """exact stream bytes per token -> dict(rowal, packed, payload_bits). shapes: [(name, R, G)].
    row = header bits per row (or per `blk` groups of a row when blk is set)."""
    t_row = t_pack = 0                 # tuples
    payload = 0
    row_bits = row
    for name, R, G in shapes:
        row = row_bits * (G // blk) if blk else row_bits
        b = bits
        hb = hdr
        if bits == "dict":
            b = max(1, math.ceil(math.log2(distinct[name])))
            hb = lambda G_, d=distinct[name]: d * 16
        ns = G * 128 // gw             # scales per row
        hbeats = -(-hb(G) // 128)
        payload += R * (G * 128 + ns * b + row) + hb(G)
        # row-aligned: whole beats per row; the row code (if any) leads the row's first scale beat
        if b == 0:
            sb = 1 if row else 0
        else:
            first = (128 - row) // b
            sb = 1 + max(0, -(-(ns - first) // (128 // b)))
        rb = G + sb
        t_row += R * -(-rb // NPORTS) + -(-hbeats // NPORTS)
        # packed: codes packed across rows, row codes in their own beats
        sbeats = (-(-(R * ns) // (128 // b)) if b else 0) + -(-(R * row) // 128)
        t_pack += -(-(R * G + sbeats + hbeats) // NPORTS)
    pad = lambda t: (t + (-t) % CHUNK) * NPORTS * 16
    return dict(rowal=pad(t_row), packed=pad(t_pack), payload_bits=payload)


def stage_bytes(scales, doc, redo=False):
    shapes = [(n, S.shape[0], S.shape[1]) for n, S in scales.items()]
    nw = sum(R * G * 128 for _, R, G in shapes)
    distinct = {n: len(np.unique(S.view(np.uint16))) for n, S in scales.items()}
    # gate: my accounting of today's format == pack_weights.plan
    from ..pack_weights import Layout, plan
    _, tt = plan(shapes, Layout())
    stock = scheme_bytes(shapes, 16)
    assert stock["rowal"] == tt * NPORTS * 16 == 1064116224, (stock, tt * 48)
    print(f"[bytes] stock stream {stock['rowal']:,} B/token == pack_weights.plan (gate OK), {nw:,} weights, "
          f"{len(shapes)} matrices")
    sch = doc.setdefault("schemes", {})
    t0 = time.time()
    for name, s in SCHEMES.items():
        if "bytes_packed" in sch.get(name, {}) and not redo:
            continue
        by = scheme_bytes(shapes, s["bits"], s["gw"], s["row"], s["hdr"], distinct, s.get("blk", 0))
        # scale-domain error over the whole stream
        se = sl = sw = n = 0.0
        mx = 0.0
        exact = True
        for nm, S in scales.items():
            Q = s["fn"](S)
            assert Q.dtype == np.float16 and Q.shape == S.shape and np.isfinite(Q).all() and (Q > 0).all(), (name, nm)
            a, b = S.astype(np.float64), Q.astype(np.float64)
            d = np.log2(b) - np.log2(a)
            sl += (d ** 2).sum()
            se += ((b - a) ** 2).sum()
            sw += (a ** 2).sum()
            n += a.size
            mx = max(mx, float(np.abs(d).max()))
            exact &= bool(np.array_equal(S, Q))
        r = sch.setdefault(name, {})
        r.update(lossy=s["lossy"], bits_per_scale=s["bits"] if s["bits"] != "dict" else
                 max(math.ceil(math.log2(d)) for d in distinct.values()),
                 weights_per_scale=s["gw"], row_bits=s["row"], decoder=s["dec"],
                 bytes_rowal=by["rowal"], bytes_packed=by["packed"], bpw_rowal=8 * by["rowal"] / nw,
                 bpw_packed=8 * by["packed"] / nw, bpw_payload=by["payload_bits"] / nw,
                 rms_log2_err=math.sqrt(sl / n), max_log2_err=mx, rel_l2_weight_err=math.sqrt(se / sw),
                 bit_exact=exact)
        for f in ("rowal", "packed"):
            r[f"toks_{f}"] = BW / by[f]
            r[f"toks_{f}_kv512"] = BW / (by[f] + KV_CTX512)
        assert exact or s["lossy"], f"{name} is declared lossless but changed a scale"
        print(f"  {name:13s} rowal {by['rowal'] / 1e6:8.1f} MB ({r['bpw_rowal']:.4f} bpw)  packed "
              f"{by['packed'] / 1e6:8.1f} MB ({r['bpw_packed']:.4f} bpw)  rms log2 err {r['rms_log2_err']:.4f} "
              f"rel L2 {r['rel_l2_weight_err']:.4f}{'  BIT-EXACT' if exact else ''}  ({time.time() - t0:.0f}s)")
    # lossless entropy bounds (not beat-granular: what an ideal entropy coder would reach)
    st = doc.get("stats", {}).get("per_matrix")
    if st:
        sign_bytes = nw // 8
        for key, lab, side in (("H0_mm", "entropy_H0", lambda R, G: 0),
                               ("H_given_left16_up16", "entropy_ctx_left_up", lambda R, G: 0),
                               ("H_sep_resid", "entropy_sep_resid", lambda R, G: (R + G) * 16)):
            bits = sum(st[n][key] * R * G + side(R, G) for n, R, G in shapes)
            by = sign_bytes + bits / 8
            sch[lab] = dict(lossy=False, bound=True, bits_per_scale=bits / (nw / 128), bytes_packed=by,
                            bytes_rowal=None, bpw_packed=8 * by / nw, toks_packed=BW / by,
                            toks_packed_kv512=BW / (by + KV_CTX512), bit_exact=True,
                            decoder="variable-length entropy decoder at ~2.6 scales/cycle (a bound, not a design)")
            print(f"  {lab:20s} bound {by / 1e6:8.1f} MB ({8 * by / nw:.4f} bpw, {bits / (nw / 128):.2f} bits/scale)")
    doc["n_weights"] = nw
    doc["bytes_signs_only"] = nw // 8
    doc["dict_bits_per_matrix"] = {str(b): sum(1 for d in distinct.values() if math.ceil(math.log2(d)) == b)
                                   for b in sorted({math.ceil(math.log2(d)) for d in distinct.values()})}
    print(f"  per-matrix dictionary width (bits: matrices): {doc['dict_bits_per_matrix']}")


# ----------------------------------------------------------------------------- stage: eval
def load_texts(a):
    from transformers import AutoTokenizer
    from .r1b_sweep import eval_texts
    tok = AutoTokenizer.from_pretrained(a.tok)
    texts, chat = eval_texts(tok, a.T)
    cc = Path(a.chat_cache)
    assert cc.exists(), f"missing {cc}"
    texts.append(("chat_selfgen", (tok(chat).input_ids + json.loads(cc.read_text()))[:a.T]))
    return texts


HOLDOUT = [("tinystories_b", "data/TinyStories-valid.txt", 100000), ("tinystories_c", "data/TinyStories-valid.txt", 200000),
           ("tinystories_d", "data/TinyStories-valid.txt", 300000), ("readme_b", "README.md", 9000),
           ("readme_c", "README.md", 14000), ("master_b", "docs/0-master.md", 5000),
           ("doc6", "docs/6-past-the-stream-ceiling.md", 1000), ("doc6_b", "docs/6-past-the-stream-ceiling.md", 6000)]


def holdout_texts(a):
    """a second token set, disjoint from the R1c one, used only to CONFIRM schemes that were
    selected on the R1c set (8 texts x T tokens, all neutral prose: no self-generated text)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    out = []
    for name, f, off in HOLDOUT:
        ids = tok((ROOT / f).read_text(errors="ignore")[off:off + 20000]).input_ids[:a.T]
        assert len(ids) == a.T, (name, len(ids))
        out.append((name, ids))
    return out


def eval_model(ref, texts):
    """-> (per-token NLL {text: np[T-1]}, argmax {text: np[T-1]})."""
    import torch
    nll, am = {}, {}
    for name, t in texts:
        lg = ref.forward_seq(t).cpu().double()
        nll[name] = torch.nn.functional.cross_entropy(lg[:-1], torch.tensor(t[1:]), reduction="none").numpy()
        am[name] = lg[:-1].argmax(-1).numpy()
        del lg
    return nll, am


def compare(nll, am, nll0, am0):
    def pool(skip=()):
        d = np.concatenate([nll[k] - nll0[k] for k in nll if k not in skip])
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    d, se = pool()
    dn, sn = pool(("chat_selfgen",))
    n = sum(len(v) for v in nll.values())
    return dict(nll=float(sum(v.sum() for v in nll.values()) / n), dnll=d, se=se, dnll_ex_selfgen=dn,
                se_ex_selfgen=sn, top1=float(sum((am[k] == am0[k]).sum() for k in am) / n), ntok=int(n),
                per_text={k: dict(nll=float(nll[k].mean()), dnll=float((nll[k] - nll0[k]).mean()),
                                  top1=float((am[k] == am0[k]).mean())) for k in nll})


def stage_eval(scales, doc, a, save):
    import torch
    from .qwen3_ref import FloatGemv, Q1Matrix, Qwen3Ref, from_gguf
    t_start = time.time()
    texts = load_texts(a)
    torch.cuda.reset_peak_memory_stats()
    cfg, W, g = from_gguf(a.gguf, device=a.device)
    del g
    ref = Qwen3Ref(cfg, W, device=a.device, gemv=FloatGemv())
    stock = {n: W[n] for n in scales}
    for n, S in scales.items():                                  # the loader and I see the same scales
        assert tuple(W[n].scales.shape) == S.shape
    print(f"[eval] model loaded ({time.time() - t_start:.0f}s)")
    nll0, am0 = eval_model(ref, texts)
    n = sum(len(v) for v in nll0.values())
    pooled = float(sum(v.sum() for v in nll0.values()) / n)
    want = json.loads(Path(a.r1c).read_text())["float_nll"]
    ok = abs(pooled - want) < 1e-6 and n == 1020
    print(f"R3A_SANITY stock binary-8B pooled NLL {pooled:.6f} vs r1c_gate float_nll {want:.6f} "
          f"|d|={abs(pooled - want):.2e} over {n} tokens {'PASS' if ok else 'FAIL'}")
    doc["eval"] = dict(gpu_seconds=doc.get("eval", {}).get("gpu_seconds", 0),
                       stock_nll=pooled, r1c_float_nll=want, ntok=n, sanity_ok=ok, T=a.T,
                       texts={k: len(t) for k, t in texts},
                       stock_per_text={k: float(v.mean()) for k, v in nll0.items()})
    assert ok, "stock model does not reproduce r1c_gate float_nll: stop and fix"
    sch = doc.setdefault("schemes", {})
    todo = [s for s in (a.schemes or SCHEMES) if s not in SCHEMES or SCHEMES[s]["lossy"] or s == "bf16_10"]
    if a.holdout:
        texts = holdout_texts(a)
        nll0, am0 = eval_model(ref, texts)
        n = sum(len(v) for v in nll0.values())
        doc["holdout"] = dict(stock_nll=float(sum(v.sum() for v in nll0.values()) / n), ntok=n,
                              texts=[list(h) for h in HOLDOUT], stock_per_text={k: float(v.mean()) for k, v in nll0.items()})
        print(f"[eval] HOLDOUT set: {len(texts)} texts, {n} tokens, stock NLL {doc['holdout']['stock_nll']:.4f}")
    for name in todo:
        if ("holdout" if a.holdout else "dnll") in sch.get(name, {}) and not a.redo:
            continue
        if time.time() - t_start > a.budget_s:
            print(f"[eval] budget of {a.budget_s}s reached: rerun to continue")
            break
        t0 = time.time()
        fn = scheme_fn(name)
        se = sw = 0.0
        for nm, S in scales.items():
            Q = fn(S, nm)
            se += ((Q.astype(np.float64) - S.astype(np.float64)) ** 2).sum()
            sw += (S.astype(np.float64) ** 2).sum()
            W[nm] = Q1Matrix(stock[nm].signs, torch.from_numpy(Q).to(a.device))
        tb = time.time() - t0
        nll, am = eval_model(ref, texts)
        r = compare(nll, am, nll0, am0)
        if name not in SCHEMES:
            r.update(diag=True, lossy=True, rel_l2_weight_err=math.sqrt(se / sw))
        if a.holdout:
            r = dict(holdout={k: r[k] for k in ("nll", "dnll", "se", "top1", "ntok", "per_text")})
        sch.setdefault(name, {}).update(r)
        r = r.get("holdout", r)
        r.setdefault("dnll_ex_selfgen", r["dnll"])
        r.setdefault("se_ex_selfgen", r["se"])
        for nm in scales:
            W[nm] = stock[nm]
        save()
        print(f"  {name:13s} dNLL {r['dnll']:+.5f} +- {r['se']:.5f}  ex-selfgen {r['dnll_ex_selfgen']:+.5f} +- "
              f"{r['se_ex_selfgen']:.5f}  top1 {r['top1']:.4f}  (build {tb:.0f}s, eval {time.time() - t0 - tb:.0f}s)")
    doc["eval"]["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
    doc["eval"]["gpu_seconds"] = doc["eval"].get("gpu_seconds", 0) + time.time() - t_start
    print(f"[eval] GPU peak {doc['eval']['gpu_peak_bytes'] / 1e9:.2f} GB, {time.time() - t_start:.0f}s")


# ----------------------------------------------------------------------------- stage: report
def within_noise(r):
    return abs(r["dnll"]) < 2 * r["se"] and abs(r["dnll"]) < 0.01


def stage_report(doc):
    sch = doc.get("schemes", {})
    order = sorted((k for k in sch if not sch[k].get("diag")), key=lambda k: -(sch[k].get("bytes_packed") or 0))
    ho = doc.get("holdout", {})
    print("\nR3a summary. bytes = full stream order (253 matrices), weights only; tok/s DERIVED = 15.54e9 B/s "
          "(R0 MEASURED) / bytes; +KV = with 43.8 MB of KV read per token at ctx 512. 'packed' = scale codes "
          "packed across rows; 'rowal' = today's row-aligned tuples. dNLL = teacher-forced float NLL minus stock "
          "binary-8B, nats/token, +- paired SE, on the R1c set (1,020 tokens) and on the confirmation set "
          f"({ho.get('ntok', 0)} tokens of other text), MEASURED. top1 = argmax agreement with stock.")
    print(f"{'scheme':20s} {'bpw':>7s} {'MB/tok':>8s} {'tok/s':>6s} {'+KV':>6s} | {'rowal bpw':>9s} {'MB':>8s} "
          f"{'tok/s':>6s} | {'dNLL +- SE (R1c set)':>21s} {'top1':>6s} | {'dNLL +- SE (confirm)':>21s} {'top1':>6s} | "
          f"{'rms log2':>8s}")
    for k in order:
        r = sch[k]
        ra = (f"{r['bpw_rowal']:9.4f} {r['bytes_rowal'] / 1e6:8.1f} {r['toks_rowal']:6.2f}"
              if r.get("bytes_rowal") else f"{'bound':>9s} {'':8s} {'':6s}")
        if "dnll" in r:
            q = f"{r['dnll']:+10.5f} +- {r['se']:.5f} {r['top1']:6.4f}"
        else:
            q = f"{'lossless (bit-exact)' if r.get('bit_exact') else 'not evaluated':>21s} {'':6s}"
        h = r.get("holdout")
        qh = f"{h['dnll']:+10.5f} +- {h['se']:.5f} {h['top1']:6.4f}" if h else f"{'':21s} {'':6s}"
        print(f"{k:20s} {r['bpw_packed']:7.4f} {r['bytes_packed'] / 1e6:8.1f} {r['toks_packed']:6.2f} "
              f"{r['toks_packed_kv512']:6.2f} | {ra} | {q} | {qh} | {r.get('rms_log2_err', 0):8.4f}")
    print("decoders:")
    for k in order:
        print(f"  {k:20s} {sch[k].get('decoder', '')}")
    dg = [k for k in sch if sch[k].get("diag")]
    if dg:
        print("diagnostics (not codecs): where the sensitivity to scale error lives")
        for k in dg:
            r = sch[k]
            print(f"  {k:20s} rel L2 {r['rel_l2_weight_err']:.4f}  dNLL {r['dnll']:+9.5f} +- {r['se']:.5f}  "
                  f"ex-selfgen {r['dnll_ex_selfgen']:+9.5f}  top1 {r['top1']:6.4f}")
    lossy = [k for k in sch if "dnll" in sch[k] and sch[k]["lossy"] and not sch[k].get("diag") and sch[k].get("bytes_packed")]
    ok = [k for k in lossy if within_noise(sch[k])]
    # both sets as two strata: the pooled difference and its SE, and a non-inferiority test at 0.01 nats
    print("pooled over both token sets (stratified): dNLL +- SE, upper 95% bound")
    for k in sorted((k for k in lossy if "holdout" in sch[k]), key=lambda k: sch[k]["bytes_packed"]):
        r, h = sch[k], sch[k]["holdout"]
        n1, n2 = r["ntok"], h["ntok"]
        d = (n1 * r["dnll"] + n2 * h["dnll"]) / (n1 + n2)
        se = math.sqrt((n1 * r["se"]) ** 2 + (n2 * h["se"]) ** 2) / (n1 + n2)
        r["pooled"] = dict(dnll=d, se=se, ucb=d + 2 * se, ntok=n1 + n2,
                           top1=(n1 * r["top1"] + n2 * h["top1"]) / (n1 + n2))
        print(f"  {k:10s} {r['bytes_packed'] / 1e6:7.1f} MB  {d:+.5f} +- {se:.5f}  ucb {d + 2 * se:+.5f}  top1 "
              f"{r['pooled']['top1']:.4f}  {'rule-A' if k in ok else '      '} {'non-inferior@0.01' if d + 2 * se < 0.01 else ''}")
    conf = [k for k in ok if "pooled" in sch[k] and sch[k]["pooled"]["ucb"] < 0.01]
    safe = [k for k in conf if sch[k]["top1"] >= 0.99 and sch[k]["holdout"]["top1"] >= 0.99]
    mb = lambda k, f: round(sch[k][f] / 1e6, 1)
    pick = lambda ks, f: min(ks, key=lambda k: (mb(k, f), -sch[k]["top1"])) if ks else None
    rec = dict(rule="rule A: |dNLL| < 2 SE and < 0.01 nats on the R1c set (1,020 tokens). 'confirmed' = rule A and, "
                    "pooled over the R1c set and the second set, dNLL + 2 SE < 0.01 nats (non-inferiority at 0.01). "
                    "'safe' = confirmed and top-1 agreement >= 0.99 on both sets (the R1c bar). Ties in bytes "
                    "(0.1 MB) go to the higher top-1.",
               within_noise=sorted(ok, key=lambda k: sch[k]["bytes_packed"]),
               confirmed=sorted(conf, key=lambda k: sch[k]["bytes_packed"]),
               failed_confirmation=sorted(k for k in ok if "holdout" in sch[k] and k not in conf),
               strict_both_sets=sorted(k for k in ok if "holdout" in sch[k] and within_noise(sch[k]["holdout"])),
               not_on_confirmation_set=sorted(k for k in ok if "holdout" not in sch[k]),
               best_packed=pick(conf, "bytes_packed"), best_rowal=pick(conf, "bytes_rowal"),
               safe_packed=pick(safe, "bytes_packed"), safe_rowal=pick(safe, "bytes_rowal"),
               lossless_fixed_rate="dict_lossless")
    doc["recommendation"] = rec
    print("R3A_VERDICT DONE")
    st = sch["stock_fp16"]

    def line(k, f):
        r = sch[k]
        t = "packed" if f == "bytes_packed" else "rowal"
        q = (f"dNLL {r['dnll']:+.4f} +- {r['se']:.4f} on the R1c set, {r['holdout']['dnll']:+.4f} +- "
             f"{r['holdout']['se']:.4f} on the second set, pooled {r['pooled']['dnll']:+.4f} +- {r['pooled']['se']:.4f}; "
             f"top-1 {r['top1']:.3f} / {r['holdout']['top1']:.3f}") if "dnll" in r and "holdout" in r else "bit-exact"
        return (f"{k} [{t} stream] {8 * r[f] / doc['n_weights']:.4f} bpw, {r[f] / 1e6:.1f} MB/token, "
                f"{r['toks_' + t]:.2f} tok/s weights-only, {r['toks_' + t + '_kv512']:.2f} with KV at ctx 512; {q}")

    if rec["best_rowal"]:
        print(f"R3A_RECOMMEND {line(rec['best_rowal'], 'bytes_rowal')}. Stock: {st['bytes_rowal'] / 1e6:.1f} MB, "
              f"{st['toks_rowal']:.2f} / {st['toks_rowal_kv512']:.2f} tok/s (DERIVED).")
        print(f"R3A_ALSO smallest confirmed with a packed scale stream: {line(rec['best_packed'], 'bytes_packed')}")
        if rec["safe_packed"]:
            print(f"R3A_ALSO smallest confirmed that also keeps >= 99% of stock's argmaxes: {line(rec['safe_packed'], 'bytes_packed')}")
        print(f"R3A_ALSO lossless, fixed rate: {line('dict_lossless', 'bytes_packed')}")
    else:
        print("R3A_RECOMMEND no lossy scheme is confirmed within noise yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["stats", "bytes", "eval", "report"],
                    choices=["stats", "bytes", "eval", "report"])
    ap.add_argument("--gguf", default="data/big/Bonsai-8B-Q1_0.gguf")
    ap.add_argument("--tok", default="data/big/Bonsai-8B-meta")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--out", default="fabric/big/results/r3a_scales.json")
    ap.add_argument("--chat-cache", default="fabric/big/results/Bonsai-8B-Q1_0_chatgen_T256.json")
    ap.add_argument("--r1c", default="fabric/big/results/r1c_gate.json")
    ap.add_argument("--schemes", nargs="+", default=None, help="subset of schemes to evaluate")
    ap.add_argument("--redo", action="store_true", help="re-evaluate schemes already in the JSON")
    ap.add_argument("--holdout", action="store_true", help="evaluate on the second (confirmation) token set")
    ap.add_argument("--budget-s", type=float, default=420, help="stop starting new evals after this many seconds")
    a = ap.parse_args()
    out = Path(a.out)
    doc = json.loads(out.read_text()) if out.exists() else {}
    doc.update(model=Path(a.gguf).name, bw_bytes_per_s=BW, kv_bytes_ctx512=KV_CTX512, nports=NPORTS, nlog=NLOG)

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(out)

    scales = load_scales(a.gguf) if set(a.stage) - {"report"} else None
    if "stats" in a.stage:
        stage_stats(scales, doc)
        save()
    if "bytes" in a.stage:
        stage_bytes(scales, doc, a.redo)
        save()
    if "eval" in a.stage:
        stage_eval(scales, doc, a, save)
        save()
    if "report" in a.stage:
        stage_report(doc)
        save()


if __name__ == "__main__":
    main()
