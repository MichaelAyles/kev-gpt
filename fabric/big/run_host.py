"""Gate of the Big-Kev host vector path (`fabric/big/host/bk_host.c`) against the bit-true
spec (`fabric/big/ref/fabric_ref.py`). Prints one final line: HOST_VERDICT PASS|FAIL (...).

The C library does everything of a token that is not a GEMV. It is correct iff, fed the
same GEMV results, it hands the GEMVs the same inputs and ends with the same logits as the
spec. Stages (--stages, default all):

  prim    every exported primitive against the spec's torch function on adversarial inputs
          (fsum, rsqrt_fab, exp_fab, silu, rmsnorm, rope, bfp_quant in the three formats
          with exact ties and threshold cases, dequant, embedding rows, argmax ties)
  kern    the library has two sets of kernels: the scalar reference kernels (the readable
          statement of the spec) and the fast ones (SIMD, hand-ordered for the Cortex-A53).
          Fast vs reference bit for bit on adversarial inputs (block-boundary sizes,
          exponents all over fp32, subnormals, signed zeros, ties, NaN / inf, the attention
          of one kv head for rep 1..9 and Tk 1..257 with exponents that take every fall-back
          path); the reference kernels vs the spec's vectors of stage prim; a -DBK_NO_SIMD
          build; bugs planted in the fast kernels only must be seen by fast-vs-reference
  tables  the numpy-only tables == the spec's tables, byte for byte (tiny, 1.7B, 8B
          configs); .npz round trip; writes the table files for the board
  a       tiny random Q1 models on the CPU, free-running: EVERY GEMV input (mantissas and
          exponents), every GEMV output, the logits and the intermediate taps (residuals,
          RoPE, attention scores / probabilities / ctx, SiLU, the KV cache as stored) of
          every step are bit-identical between FabricRef.step and the C library, whose
          callback is the same ExactBfpGemv. Also: prefill without logits, cache growth,
          truncate + redo, the three build variants (-O3 default, the plain -O2 command,
          -O0, -march=native), and the numpy-only replay file for the board.
  threads the tiny models free-running with 1 / 3 / 4 / 2 threads and both kernel sets, all
          taps of every run against the spec's record (so every run is identical to every
          other), the worker threads provably used; the same matrix through the numpy-only
          replay with one digest over all taps and the KV cache
  bugs    planted-bug builds (-DBK_BUG_*) must each be caught by prim or by the free-run,
          with the fast kernels (default threads) AND with the reference kernels (1 thread)
  b       real Bonsai-1.7B, a real chat prompt + 20 generated tokens: tokens and every GEMV
          input identical, spec and C library in lockstep (GPU if there is one)
  c       real Bonsai-8B: the golden token file (golden_token.py); the C library runs the
          prompt with the GPU spec's ExactBfpGemv as callback; at the golden position all
          253 inputs and outputs and the argmax must match
  time    host ms/token of the 8B shape by function (stand-in GEMV; plus the real-data
          numbers of stage c when it ran) -> fabric/big/results/host_timing.json

    .venv/bin/python -u -m fabric.big.run_host --stages prim,kern,tables,a,threads,bugs,time   # CPU only
    flock /tmp/kevbuild/gpu.lock env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python -u -m fabric.big.run_host                                  # everything
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .host import bk_host as bh
from .ref import fabric_ref as fr
from .ref.fabric_ref import FabricRef, Trace
from .ref.qwen3_ref import Cfg, Q1Matrix, cfg_from_gguf, from_gguf

RES = Path(__file__).resolve().parent / "results"
VEC = Path("/tmp/kevbuild/big_vectors")
PROMPT_B = ("<|im_start|>user\nWhy is the sky blue? Answer in one short sentence.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")

BUGS = {            # name -> (defines, extra flags, extra libs, what it is)
    "TREE": (["BK_BUG_TREE"], [], [], "pairwise tree sums adjacent lanes instead of [0:8]+[8:16]"),
    "ROUND_AWAY": (["BK_BUG_ROUND_AWAY"], [], [], "bfp_quant rounds half away from zero, not to even"),
    "REASSOC": (["BK_BUG_REASSOC"], [], [], "RMSNorm sum of squares in one sequential accumulator"),
    "FMA": (["BK_BUG_FMA"], [], ["-lm"], "a*b+c fused in the lane sums (what -ffp-contract=fast emits)"),
    "EXP_TRUNC": (["BK_BUG_EXP_TRUNC"], [], [], "exp_fab table index truncated instead of RNE"),
    "NEWTON1": (["BK_BUG_NEWTON1"], [], [], "rsqrt_fab with one Newton step"),
    "CTX_LANE": (["BK_BUG_CTX_LANE"], [], [], "ctx sum: every position in lane 0 (no lane striding)"),
    "ARGMAX_LAST": (["BK_BUG_ARGMAX_LAST"], [], [], "argmax returns the highest index on ties"),
}


def log(*a):
    print(*a, flush=True)


def bits(a):
    a = np.ascontiguousarray(a)
    if a.dtype == np.float32:
        return a.view(np.uint32)
    if a.dtype == np.float64:
        return a.view(np.uint64)
    return a


def beq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.size == b.size and a.dtype == b.dtype and np.array_equal(bits(a).reshape(-1), bits(b).reshape(-1))


def nbad(a, b):
    return int((bits(np.asarray(a)).reshape(-1) != bits(np.asarray(b)).reshape(-1)).sum())


def argmax_low(lg):
    """the spec's rule: the LOWEST index holding the maximum."""
    return int(torch.nonzero(lg == lg.max())[0, 0])


# ----------------------------------------------------------------------------- tiny models
TINY = {    # exercise: several layers, n_head != n_kv, head_dim 128, d a multiple of 128
    "A": dict(cfg=dict(n_layer=3, d=256, ffn=384, n_head=4, n_kv=2, head_dim=128, vocab=320,
                       rope_theta=1e6, eps=1e-6, tied=True), seed=0, n=48, prompt=4),
    # rep 3, d not a power of two (the spec's 2^-int(log2(n)) mean), YaRN, untied head, theta 1e4
    "B": dict(cfg=dict(n_layer=2, d=384, ffn=640, n_head=6, n_kv=2, head_dim=128, vocab=257,
                       rope_theta=1e4, eps=1e-5, tied=False, yarn_factor=4.0, yarn_orig_ctx=8192), seed=1, n=40, prompt=3),
    # rep 1 (n_head == n_kv), one group per vector
    "C": dict(cfg=dict(n_layer=2, d=128, ffn=256, n_head=2, n_kv=2, head_dim=128, vocab=200,
                       rope_theta=1e6, eps=1e-6, tied=True), seed=2, n=40, prompt=5),
}


def tiny_model(spec):
    """A random Q1 model as tests/test_big_golden.py makes one: ~10% negative scales; the
    gate/up scales of layer L are multiplied by 1e-4^L (layer 1: fp16 subnormals and GEMV
    terms below the accumulator LSB; layer 2: scales underflow to 0 -> all-zero vectors)."""
    g = torch.Generator().manual_seed(spec["seed"])
    c = Cfg(**spec["cfg"])

    def mat(o, i, s=1.0):
        sc = (torch.rand(o, i // 128, generator=g) * 0.08 + 0.01) * (1 - 2 * (torch.rand(o, i // 128, generator=g) < 0.1)) * s
        return Q1Matrix(torch.randint(0, 256, (o, i // 8), generator=g, dtype=torch.uint8), sc.to(torch.float16))

    def vec(n):
        return torch.rand(n, generator=g) + 0.5

    W = {"embed": mat(c.vocab, c.d), "out_norm": vec(c.d)}
    W["lm_head"] = W["embed"] if c.tied else mat(c.vocab, c.d)
    for L in range(c.n_layer):
        p = f"l{L}."
        W.update({p + "q": mat(c.n_head * c.head_dim, c.d), p + "k": mat(c.n_kv * c.head_dim, c.d),
                  p + "v": mat(c.n_kv * c.head_dim, c.d), p + "o": mat(c.d, c.n_head * c.head_dim),
                  p + "gate": mat(c.ffn, c.d, 1e-4 ** L), p + "up": mat(c.ffn, c.d, 1e-4 ** L),
                  p + "down": mat(c.d, c.ffn),
                  p + "attn_norm": vec(c.d), p + "ffn_norm": vec(c.d),
                  p + "q_norm": vec(c.head_dim), p + "k_norm": vec(c.head_dim)})
    toks = torch.randint(0, c.vocab, (spec["prompt"],), generator=g).tolist()
    return FabricRef(c, W, device="cpu", max_pos=96), toks


def names_of(c):
    return [bh.mat_name(m, c.n_layer) for m in range(7 * c.n_layer + 1)]


def host_gains(fab):
    c = fab.c
    g = {k: np.stack([fab.W[f"l{L}.{k}"].cpu().numpy() for L in range(c.n_layer)]) for k in bh.NORMS}
    g["out_norm"] = fab.W["out_norm"].cpu().numpy()
    return g


def spec_tables(fab):
    """the tables the spec instance actually uses."""
    t = dict(EXP=fab.EXP, RS_LO=fab.RS_LO, RS_HI=fab.RS_HI, COS=fab.COS, SIN=fab.SIN)
    t = {k: v.cpu().numpy().copy() for k, v in t.items()}
    t.update(eps=np.float32(fab.eps32.item()), inv_sqrt_hd=np.float32(fab.inv_sqrt_hd.item()))
    return t


# ----------------------------------------------------------------------------- GEMV callback
class _Rec:
    """the part of Trace that ExactBfpGemv.mm needs: keeps the input as mm itself quantised it."""

    def __init__(self):
        self.d = {}

    def tap(self, name, t, *a, **k):
        self.d[name.rsplit(".", 1)[1]] = t

    def window(self, name, n_out):
        return 0, 0, False


class SpecGemv:
    """The callback of the C library: the spec's own ExactBfpGemv on the spec's own weights.
    The quantised input is turned back into fp32 (exact), mm re-quantises it, and we check
    that it sees the mantissas / exponents the C library sent (`requant_bad` must stay 0)."""

    def __init__(self, fab, on_call=None):
        self.fab, self.names, self.on_call = fab, names_of(fab.c), on_call
        self.requant_bad = self.calls = 0

    def __call__(self, mid, xq, xe, y, same):
        fab, name = self.fab, self.names[mid]
        X = np.ldexp(xq.astype(np.float64).reshape(-1, 128), xe.astype(np.int32)[:, None]).reshape(1, -1)
        rec = _Rec()
        out = fab.gemv.mm(fab.W[name], torch.from_numpy(X).to(torch.float32).to(fab.dev), name, rec)
        ok = np.array_equal(rec.d["xq"][0].cpu().numpy().astype(np.int16), xq) and \
            np.array_equal(rec.d["xe"][0].cpu().numpy().astype(np.int8), xe)
        self.requant_bad += 0 if ok else 1
        self.calls += 1
        yv = out[0].cpu().numpy()
        y[:] = yv
        if self.on_call is not None:
            self.on_call(mid, name, xq, xe, yv)


class XTrace:
    """A Trace for the big models: the spec's own taps (same call sites, same values), but
    only the signals this gate compares are copied off the device."""
    layers = positions = None
    final = True
    KEEP = {"xq", "xe", "x_in", "x_mid", "x_out", "q_rope", "k_rope", "s", "p", "ctx", "silu", "act",
            "k_q", "k_e", "v_q", "v_e", "out_norm"}

    def __init__(self):
        self.sig = {}

    def begin(self, pos):
        pass

    def on(self, L):
        return True

    def window(self, name, n_out):
        return 0, 0, False

    def tap(self, name, t, tdim=0, fill=None, cache=False):
        if name.rsplit(".", 1)[-1] in self.KEEP:
            self.sig[name] = t.detach().movedim(tdim, 0).cpu()

    def __getitem__(self, name):
        return self.sig[name]

    def __contains__(self, name):
        return name in self.sig


# ----------------------------------------------------------------------------- stage: prim
def prim_reference(fab):
    """Inputs + the spec's answers for every primitive (computed once, reused per build)."""
    rng = np.random.RandomState(1234)
    R = {}
    f32 = np.float32

    def rnd_bits(n, lo=1, hi=0x7F7FFFFF):               # random positive finite fp32, any exponent
        return rng.randint(lo, hi, size=n, dtype=np.int64).astype(np.uint32).view(f32)

    def wide(n, emin=-30, emax=30):                     # sign * mantissa * 2^e
        return (rng.standard_normal(n) * np.exp2(rng.randint(emin, emax + 1, n))).astype(f32)

    # fsum: lengths that are not multiples of 16, values whose sum depends on the order
    xs = [wide(n, -12, 12) for n in list(range(1, 70)) + [127, 128, 129, 255, 256, 300, 1000, 4096] * 8]
    R["fsum"] = (xs, [fr.fsum(torch.from_numpy(x)).numpy() for x in xs])

    # rsqrt_fab: every exponent incl. subnormals, table-bin edges, 0
    v = np.concatenate([rnd_bits(150000), rnd_bits(4000, 1, 0x00800000),
                        np.array([0.0, 1e-6, 1.0, 2.0, 0.5, 4.0, 1.17549435e-38, 1e-45, 3.4028235e38], dtype=f32),
                        (np.arange(0x3F000000, 0x40000000, 1 << 13, dtype=np.int64)).astype(np.uint32).view(f32),
                        (np.arange(0x3F000000, 0x40000000, 1 << 13, dtype=np.int64) - 1).astype(np.uint32).view(f32)])
    R["rsqrt"] = (v, fab.rsqrt_fab(torch.from_numpy(v)).numpy())

    # exp_fab: dense, the 0 / inf cut-offs, -0, +-inf, integers of y, huge
    y_int = (np.arange(-130, 131, dtype=np.float64) / float(fr.LOG2E)).astype(f32)
    x = np.concatenate([(rng.rand(600000) * 200 - 100).astype(f32), wide(100000, -20, 8), -np.abs(wide(100000, -10, 5)),
                        y_int, np.nextafter(y_int, f32(np.inf)), np.nextafter(y_int, f32(-np.inf)),
                        np.array([0.0, -0.0, np.inf, -np.inf, 1e30, -1e30, 88.7, -87.3, -87.4, 88.8, 1e-40, -1e-40], dtype=f32)])
    R["exp"] = (x, fab.exp_fab(torch.from_numpy(x)).numpy())

    x = np.concatenate([wide(300000, -8, 7), (rng.standard_normal(300000) * 3).astype(f32),
                        np.array([0.0, -0.0, 100.0, -100.0, 200.0, -200.0, 1e-30, -1e-30], dtype=f32)])
    R["silu"] = (x, fab.silu(torch.from_numpy(x)).numpy())

    # rmsnorm: sizes of the real models and non-powers of two, many scales, an all-zero vector
    cases = []
    for n in (128, 128, 256, 384, 640, 2048, 4096, 4096):
        for sc in (1e-20, 1e-6, 1.0, 37.0, 1e12):
            xx, w = (rng.standard_normal(n) * sc).astype(f32), (rng.rand(n) + 0.5).astype(f32)
            cases.append((xx, w))
    cases.append((np.zeros(256, dtype=f32), np.ones(256, dtype=f32)))
    eps = f32(fab.eps32.item())
    R["rmsnorm"] = (cases, eps, [fab.rmsnorm(torch.from_numpy(a), torch.from_numpy(b)).numpy() for a, b in cases])

    hd = fab.c.head_dim
    xr = wide(40 * hd, -6, 6).reshape(40, hd)
    pos = rng.randint(0, fab.COS.shape[0], 40)
    R["rope"] = (xr, pos, torch.stack([fab.rope_tab(torch.from_numpy(xr[i])[None], torch.tensor([int(pos[i])]))[0]
                                       for i in range(40)]).numpy())

    # bfp_quant, the three formats. Crafted: exact .5 ties, the 511/512 exponent threshold,
    # zero groups, subnormal and huge groups.
    for name, bits_, grp, width in (("gemv", 10, 128, 16), ("k", 10, 16, 16), ("v", 8, 128, 8)):
        qmax = 2 ** (bits_ - 1) - 1
        rows = []
        for _ in range(300):
            rows.append(wide(grp, -4, 4) * f32(np.exp2(rng.randint(-40, 30))))
        for _ in range(300):                             # exact ties: (k + 0.5) * 2^e, max pins e
            e = int(rng.randint(-20, 20))
            k = rng.randint(-qmax, qmax, grp).astype(np.float64) + 0.5
            k[rng.randint(grp)] = qmax                   # group max = qmax * 2^e exactly -> exponent e
            rows.append(np.ldexp(k, e).astype(f32))
        for m in (qmax, qmax + 0.5, qmax - 0.5, qmax + 1, 2 ** (bits_ - 2), 2 ** (bits_ - 1)):     # threshold
            for e in (-9, 0, 5):
                r = np.ldexp(rng.randint(-qmax // 2, qmax // 2, grp).astype(np.float64) * 0.25, e)
                r[rng.randint(grp)] = np.ldexp(float(m), e)
                rows.append(r.astype(f32))
                r2 = r.astype(f32)
                i = int(np.argmax(np.abs(r2)))
                rows.append(np.where(np.arange(grp) == i, np.nextafter(r2[i], f32(0)), r2).astype(f32))
                rows.append(np.where(np.arange(grp) == i, np.nextafter(r2[i], f32(np.inf)), r2).astype(f32))
        rows.append(np.zeros(grp, dtype=f32))
        rows.append((rng.standard_normal(grp) * 1e-33).astype(f32))      # e ~ -118: near the int8 edge
        rows.append((rng.standard_normal(grp) * 1e-36).astype(f32))      # e ~ -128
        rows.append((rng.standard_normal(grp) * 3e-39).astype(f32))      # subnormals: e < -128
        rows.append((rng.standard_normal(grp) * 1e-44).astype(f32))
        rows.append((rng.standard_normal(grp) * 1e37).astype(f32))
        X = np.stack(rows)
        q, e = fr.bfp_quant(torch.from_numpy(X), bits_, grp)
        R["quant_" + name] = (X, bits_, grp, width, q.numpy().astype(np.int64), e.numpy()[:, 0])

    # dequant: every int8 exponent incl. the fp32 underflow / overflow ends
    for name, bits_, grp, dt in (("k", 10, 16, np.int16), ("v", 8, 128, np.int8)):
        qmax = 2 ** (bits_ - 1) - 1
        q = rng.randint(-qmax, qmax + 1, (256, grp)).astype(np.float64)
        e = np.arange(-128, 128, dtype=np.int64)
        R["dequant_" + name] = (q.astype(dt), e.astype(np.int8), grp,
                                fr.bfp_dequant(torch.from_numpy(q), torch.from_numpy(e)[:, None], grp).numpy())

    # embedding rows: random signs, fp16 scales incl. negative, subnormal, zero
    d, rows_ = 384, 64
    signs = rng.randint(0, 256, (rows_, d // 8)).astype(np.uint8)
    sc = (rng.standard_normal((rows_, d // 128)) * 0.05).astype(np.float16)
    sc[0, 0], sc[1, 1], sc[2, 2], sc[3, 0] = np.float16(0), np.float16(-0.0), np.float16(1e-7), np.float16(-6e-8)
    R["embed"] = (signs, sc, Q1Matrix(torch.from_numpy(signs), torch.from_numpy(sc)).dequant().numpy())

    # argmax: ties, the spec's formula
    am = []
    for _ in range(200):
        n = int(rng.randint(1, 400))
        a = rng.randint(-3, 4, n).astype(f32)            # many ties
        t = torch.from_numpy(a)
        i = torch.arange(n)
        am.append((a, int(torch.where(t == t.amax(-1, keepdim=True), i, n).amin(-1))))
    R["argmax"] = am
    R["tables"] = spec_tables(fab)
    return R


def prim_check(lib, R, ref=False):
    """{name: (ok, detail)} for one library build. ref: its scalar reference kernels."""
    pr = bh.Prims(lib, R["tables"], ref=ref)
    out = {}
    xs, ref = R["fsum"]
    bad = sum(not beq(pr.fsum(x), np.float32(r)) for x, r in zip(xs, ref))
    out["fsum"] = (bad == 0, f"{len(xs)} vectors, {bad} differ")
    for k, fn in (("rsqrt", pr.rsqrt), ("exp", pr.exp), ("silu", pr.silu)):
        x, ref = R[k]
        b = nbad(fn(x), ref)
        out[k] = (b == 0, f"{x.size} values, {b} differ")
    x, ref = R["exp"]                                    # the one-value entry point, a sample + the edge cases
    idx = np.concatenate([np.arange(0, x.size, 97), np.arange(x.size - 900, x.size)])
    b = sum(not beq(pr.exp1(x[i]), ref[i]) for i in idx)
    out["exp1"] = (b == 0, f"{idx.size} values, {b} differ")
    cases, eps, ref = R["rmsnorm"]
    bad = sum(not beq(pr.rmsnorm(a, b, eps), r) for (a, b), r in zip(cases, ref))
    out["rmsnorm"] = (bad == 0, f"{len(cases)} vectors, {bad} differ")
    xr, pos, ref = R["rope"]
    T = R["tables"]
    bad = sum(not beq(pr.rope(T["COS"][p], T["SIN"][p], x), r) for x, p, r in zip(xr, pos, ref))
    out["rope"] = (bad == 0, f"{len(xr)} vectors, {bad} differ")
    for name in ("gemv", "k", "v"):
        X, bits_, grp, width, qr, er = R["quant_" + name]
        bad = rng_err = ties = 0
        for x, q0, e0 in zip(X, qr, er):
            q, e, rc = pr.quant(x, bits_, grp, width)
            if -128 <= e0 <= 127:
                bad += int(rc != 0 or int(e[0]) != int(e0) or not np.array_equal(q.astype(np.int64), q0))
            else:                                       # the spec's exponent does not fit int8
                rng_err += 1
                bad += int(rc != -5)
        out["quant_" + name] = (bad == 0, f"{len(X)} groups (bits {bits_}, group {grp}), {bad} differ, "
                                          f"{rng_err} out-of-int8 exponents refused")
    for name in ("k", "v"):
        q, e, grp, ref = R["dequant_" + name]
        b = nbad(pr.dequant(q.reshape(-1), e, grp), ref.reshape(-1))
        out["dequant_" + name] = (b == 0, f"{q.size} values, exponents -128..127, {b} differ")
    signs, sc, ref = R["embed"]
    d = signs.shape[1] * 8
    blocks = np.concatenate([sc.view(np.uint8).reshape(-1, 2), signs.reshape(-1, 16)], 1).copy()   # the GGUF layout
    e1, e2 = bh.emb_from_split(signs, sc), bh.emb_from_blocks(blocks.reshape(-1), 18)
    bad = sum(not beq(pr.embed_row(e, d, r), ref[r]) for e in (e1, e2) for r in range(signs.shape[0]))
    out["embed"] = (bad == 0, f"{signs.shape[0]} rows x 2 layouts, {bad} differ")
    bad = sum(pr.argmax(a) != i for a, i in R["argmax"])
    out["argmax"] = (bad == 0, f"{len(R['argmax'])} vectors with ties, {bad} differ")
    return out


# ----------------------------------------------------------------------------- stage: a
EXT = ("x_in", "x_mid", "x_out", "q_rope", "k_rope", "att_s", "att_p", "ctx", "silu", "act", "h_out")


def step_signals(tr, c, names, lg, want=True):
    """One spec step as numpy: every GEMV input (+ output when traced), the logits and the
    intermediates the C library can tap, from a Trace / XTrace of that step."""
    nm = names if want else names[:-1]
    s = dict(logits=lg.cpu().numpy().copy(),
             xq=[tr[m + ".xq"][0].numpy().astype(np.int16) for m in nm],
             xe=[tr[m + ".xe"][0].numpy().astype(np.int8) for m in nm])
    if names[0] + ".y" in tr:
        s["y"] = [tr[m + ".y"][0].numpy().copy() for m in nm]
    L_ = range(c.n_layer)
    st = lambda k, dt=np.float32: np.stack([tr[f"l{L}.{k}"][0].numpy().astype(dt) for L in L_])   # noqa: E731
    fl = lambda k: st(k).reshape(c.n_layer, -1)                                                     # noqa: E731
    s["ext"] = dict(x_in=fl("x_in"), x_mid=fl("x_mid"), x_out=fl("x_out"), q_rope=fl("q_rope"), k_rope=fl("k_rope"),
                    ctx=fl("ctx"), silu=fl("silu"), act=fl("act"), h_out=tr["out_norm"][0].numpy(),
                    att_s=st("att.s"), att_p=st("att.p"),
                    kq=st("k_q", np.int16), ke=st("k_e", np.int8), vq=st("v_q", np.int8), ve=st("v_e", np.int8))
    return s


def compare_step(host, s, i, want, names, gemv_seen=None):
    """The C library's step i (just run, taps on) against the spec's signals `s`.
    Returns (list of mismatch descriptions, GEMV inputs compared, other signals compared)."""
    c, bad, n_in, n_ext = host.cfg, [], 0, 0
    nm = len(names) if want else len(names) - 1
    for m in range(nm):
        xq, xe, y = host.tap_gemv(m) if gemv_seen is None else gemv_seen[m][1:]
        n_in += 1
        if not (beq(xq, s["xq"][m]) and beq(xe, s["xe"][m])):
            bad.append(f"{names[m]}.xq/xe ({nbad(xq, s['xq'][m])} mantissas, {nbad(xe, s['xe'][m])} exponents)")
        elif "y" in s and y is not None and not beq(y, s["y"][m]):
            bad.append(f"{names[m]}.y")
    if gemv_seen is not None and len(gemv_seen) != nm:
        bad.append(f"{len(gemv_seen)} GEMV calls, expected {nm}")
    if "x_out" in host.taps:
        e, Tk = s["ext"], i + 1
        for k in EXT:
            if k == "h_out" and not want:
                continue
            got = host.taps[k][:, :, :Tk] if k in ("att_s", "att_p") else host.taps[k]
            n_ext += 1
            if not beq(got, e[k]):
                bad.append(f"tap {k} ({nbad(got, e[k])} values)")
        for L in range(c.n_layer):
            for g in range(c.n_kv):
                kq, ke, vq, ve = host.kv_row(L, g, i)
                n_ext += 1
                if not (beq(kq, e["kq"][L, g]) and beq(ke, e["ke"][L, g])
                        and beq(vq, e["vq"][L, g]) and beq(ve, e["ve"][L, g])):
                    bad.append(f"KV cache l{L} head {g}")
    return bad, n_in, n_ext


def spec_record(fab, prompt, n):
    """The spec free-running for n tokens (FabricRef.step with a full Trace per step)."""
    names = names_of(fab.c)
    fab.reset()
    steps, tok = [], None
    for i in range(n):
        tok = prompt[i] if i < len(prompt) else tok
        tr = Trace(rows=None)
        lg = fab.step(tok, trace=tr)
        s = step_signals(tr, fab.c, names, lg)
        s.update(token=int(tok), want=True)
        assert int(tr["argmax"][0]) == argmax_low(lg)
        steps.append(s)
        tok = argmax_low(lg)
    return steps


def make_host(fab, lib, tables=None, **kw):
    E = fab.W["embed"]
    return bh.BkHost(fab.c, host_gains(fab), bh.emb_from_split(E.signs.cpu().numpy(), E.scales.cpu().numpy()),
                     tables if tables is not None else bh.build_tables(fab.c, fab.COS.shape[0]), lib=lib, **kw)


def freerun_check(fab, lib, rec, prompt, prefill=0, taps=True, init_ctx=0, **hkw):
    """The C library free-running against the spec record `rec`; stops at the first step
    with a mismatch. Returns dict(ok, steps, gemv_inputs, first=...). hkw: threads / kernels
    of the host."""
    names = names_of(fab.c)
    cb = SpecGemv(fab)
    host = make_host(fab, lib, gemv=cb, init_ctx=init_ctx, **hkw)
    if taps:
        host.enable_taps(("xq", "xe", "y") + EXT, att_cap=len(rec) + 1)
    seen = []
    if not taps:
        cb.on_call = lambda mid, name, xq, xe, y: seen.append((mid, xq.copy(), xe.copy(), y.copy()))
    n_in = n_ext = 0
    tok, first = None, None
    for i, s in enumerate(rec):
        tok = prompt[i] if i < len(prompt) else tok
        if tok != s["token"]:
            first = f"step {i}: token {tok} != spec {s['token']}"
            break
        want = i >= prefill
        seen.clear()
        lg = host.step(tok, want_logits=want)
        bad, a_, b_ = compare_step(host, s, i, want, names, None if taps else seen)
        n_in += a_
        n_ext += b_
        if want and not beq(lg, s["logits"]):
            bad.append("logits")
        if bad:
            first = f"step {i}: " + ", ".join(bad[:4]) + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else "")
            break
        if want:
            tok = bh.argmax(lg, lib)
            if tok != argmax_low(torch.from_numpy(s["logits"])):
                first = f"step {i}: argmax {tok} != spec"
                break
        else:
            tok = argmax_low(torch.from_numpy(s["logits"]))
    ok = first is None and cb.requant_bad == 0
    st = host.stats()
    return dict(ok=ok, steps=len(rec), gemv_inputs=n_in, ext_signals=n_ext, first=first, requant_bad=cb.requant_bad,
                host=host, host_threads=host.threads, host_kernels=host.kernels, worker_units=st["worker_units"],
                regions=st["regions"])


def truncate_check(fab, lib, rec, prompt):
    """bk_truncate + redo reproduces the logits; reset + rerun too."""
    host = make_host(fab, lib, gemv=SpecGemv(fab), init_ctx=2)
    for s in rec[:20]:
        host.step(s["token"], want_logits=False)
    ok = True
    for back in (13, 7, 0):
        host.truncate(back) if back else host.reset()
        for i in range(back, 20):
            lg = host.step(rec[i]["token"])
            ok &= beq(lg, rec[i]["logits"])
    return ok


def stage_prim(ctx):
    R = ctx["prim_ref"]
    res = prim_check(ctx["lib"], R)
    for k, (ok, d) in res.items():
        log(f"  prim {k:10s} {'ok  ' if ok else 'FAIL'} {d}")
    ok = all(v[0] for v in res.values())
    return ok, dict(checks={k: dict(ok=v[0], detail=v[1]) for k, v in res.items()}), \
        f"{sum(v[0] for v in res.values())}/{len(res)} primitives"


def stage_a(ctx):
    ok, out, tot_in, tot_steps = True, {}, 0, 0
    variants = dict(default=None, O2=bh.BASE_FLAGS, O0=["-O0"] + bh.BASE_FLAGS[1:],
                    native=["-O3", "-march=native"] + bh.BASE_FLAGS[1:])
    for name, spec in TINY.items():
        fab, prompt, rec = ctx["tiny"][name]
        c = fab.c
        r = freerun_check(fab, ctx["lib"], rec, prompt)
        tot_in += r["gemv_inputs"]
        tot_steps += r["steps"]
        host = r.pop("host")
        grow = host.lib.bk_capacity(host._s)
        r2 = freerun_check(fab, ctx["lib"], rec, prompt, prefill=len(rec) // 2, taps=False, init_ctx=3)
        r2.pop("host")
        tr_ok = truncate_check(fab, ctx["lib"], rec, prompt)
        var = {}
        for vn, flags in variants.items():
            if flags is None:
                continue
            lv = bh.load_lib(flags=flags)
            pv = prim_check(lv, ctx["prim_ref"]) if name == "A" else {}
            rv = freerun_check(fab, lv, rec, prompt)
            rv.pop("host")
            var[vn] = rv["ok"] and all(v[0] for v in pv.values())
        good = r["ok"] and r2["ok"] and tr_ok and all(var.values())
        ok &= good
        out[name] = dict(cfg=bh.HostCfg.of(c).asdict(), tokens=r["steps"], gemv_inputs=r["gemv_inputs"],
                         ext_signals=r["ext_signals"], first=r["first"], requant_bad=r["requant_bad"],
                         prefill_no_logits=r2["ok"], prefill_first=r2["first"], truncate_redo=tr_ok, variants=var,
                         cache_capacity=grow)
        log(f"  a[{name}] L{c.n_layer} d{c.d} ffn{c.ffn} heads {c.n_head}/{c.n_kv} hd{c.head_dim} vocab {c.vocab}: "
            f"{r['steps']} free-running tokens, {r['gemv_inputs']} GEMV inputs + outputs + logits + {r['ext_signals']} "
            f"taps/KV rows {'identical' if r['ok'] else 'MISMATCH ' + str(r['first'])}; prefill-without-logits "
            f"{'ok' if r2['ok'] else 'FAIL ' + str(r2['first'])}; truncate/reset redo {'ok' if tr_ok else 'FAIL'}; "
            f"builds {' '.join(k + ('=ok' if v else '=FAIL') for k, v in var.items())}")
    # the numpy-only replay file (for the board): tiny model A, recorded GEMV outputs
    fab, prompt, rec = ctx["tiny"]["A"]
    E = fab.W["embed"]
    steps = [dict(s, want=(i >= 6)) for i, s in enumerate(rec)]
    for s in steps:
        if not s["want"]:
            s["xq"], s["xe"], s["y"] = s["xq"][:-1], s["xe"][:-1], s["y"][:-1]
    rp = ctx["vec"] / "host_replay_tiny.npz"
    bh.save_replay(rp, fab.c, host_gains(fab), E.signs.numpy(), E.scales.numpy(), bh.build_tables(fab.c, 96), steps)
    rok, summ = bh.replay(rp, ctx["lib"], verbose=False)
    ok &= rok
    out["replay"] = dict(file=str(rp), ok=rok, bytes=rp.stat().st_size, **summ)
    log(f"  a[replay] {rp} ({rp.stat().st_size / 1e3:.0f} kB): numpy-only replay {'PASS' if rok else 'FAIL'} "
        f"({summ['steps']} tokens, {summ['gemvs']} GEMVs) -> on the board: python3 -m fabric.big.host.bk_host --replay <file>")
    return ok, out, f"{tot_steps} tokens / {tot_in} GEMV inputs on {len(TINY)} tiny models"


# ----------------------------------------------------------------------------- stage: kern
FAST_BUGS = {       # bugs planted in the FAST kernels only: the fast-vs-reference check must see them
    "FAST_FOLD": (["BK_BUG_FAST_FOLD"], "ctx: 2^ve folded into p although p * 2^ve underflowed to zero"),
    "FAST_CLAMP": (["BK_BUG_FAST_CLAMP"], "silu: exponent clamp at 127 instead of 128 (no overflow to inf)"),
    "FAST_TINY": (["BK_BUG_FAST_TINY"], "scores: K exponents below -126 not sent to the reference path"),
}


def stage_kern(ctx):
    """The fast kernels vs the scalar reference kernels (adversarial inputs), the reference
    kernels vs the spec (the stage `prim` vectors), and the fast-only planted bugs."""
    lib = ctx["lib"]
    simd = bh.SIMD_NAMES[lib.bk_simd()]
    kok, res = bh.kern_check(lib, verbose=False)
    for k, (c, b) in res.items():
        log(f"  kern {k:13s} {'ok  ' if b == 0 else 'FAIL'} fast [{simd}] vs reference: {c} cases, {b} differ")
    pr = prim_check(lib, ctx["prim_ref"], ref=True)
    rfail = [k for k, v in pr.items() if not v[0]]
    log(f"  kern reference kernels vs the spec: {len(pr) - len(rfail)}/{len(pr)} primitives"
        + (f" FAIL {rfail}" if rfail else ""))
    # a build without SIMD: the fast names are the reference kernels; it must still be the spec
    ln = bh.load_lib(defines=["BK_NO_SIMD"])
    pn = prim_check(ln, ctx["prim_ref"])
    fab, prompt, rec = ctx["tiny"]["A"]
    rn = freerun_check(fab, ln, rec, prompt)
    rn.pop("host")
    nosimd = ln.bk_simd() == 0 and all(v[0] for v in pn.values()) and rn["ok"]
    log(f"  kern -DBK_NO_SIMD build (simd {bh.SIMD_NAMES[ln.bk_simd()]}): prim + free-run {'ok' if nosimd else 'FAIL'}")
    bugs, caught = {}, 0
    for name, (defs, what) in FAST_BUGS.items():
        lb = bh.load_lib(defines=defs)
        bok, bres = bh.kern_check(lb, verbose=False)
        hit = not bok
        caught += hit
        bugs[name] = dict(what=what, caught=hit, differ={k: b for k, (c, b) in bres.items() if b})
        log(f"  kern bug {name:10s} {'CAUGHT' if hit else 'MISSED'}: {what}; differ: {bugs[name]['differ'] or '-'}")
    ok = kok and not rfail and nosimd and caught == len(FAST_BUGS)
    n_cases = sum(c for c, _ in res.values())
    return ok, dict(simd=simd, checks={k: dict(cases=c, bad=b) for k, (c, b) in res.items()}, ref_vs_spec_fail=rfail,
                    no_simd_build=nosimd, fast_bugs=bugs), \
        (f"fast [{simd}] == reference kernels on {n_cases} adversarial cases, reference == spec, "
         f"{caught}/{len(FAST_BUGS)} fast-only bugs caught")


# ----------------------------------------------------------------------------- stage: threads
THREAD_RUNS = ((1, "fast"), (3, "fast"), (4, "fast"), (2, "fast"), (1, "ref"), (3, "ref"), (4, "ref"))


def stage_threads(ctx):
    """Tiny models free-running with 1 / 3 / 4 (/ 2) threads and both kernel sets: every run
    is compared with the spec's record on every tap (GEMV inputs, residual stream, q/k after
    RoPE, scores, softmax, context, silu, act, KV rows, logits), so the runs are identical to
    each other. Then the numpy-only replay with the same matrix (what the board runs)."""
    ok, out, n_runs = True, {}, 0
    for name in TINY:
        fab, prompt, rec = ctx["tiny"][name]
        rows = {}
        for th, kern in THREAD_RUNS:
            r = freerun_check(fab, ctx["lib"], rec, prompt, threads=th, kernels=kern)
            r.pop("host")
            used = th == 1 or r["worker_units"] > 0          # the workers must really have run units
            good = r["ok"] and r["host_threads"] == th and r["host_kernels"] == kern and used
            ok &= good
            n_runs += 1
            rows[f"{kern}/{th}"] = dict(ok=good, first=r["first"], worker_units=r["worker_units"], regions=r["regions"],
                                        ext_signals=r["ext_signals"], gemv_inputs=r["gemv_inputs"])
        out[name] = rows
        log(f"  threads[{name}] {len(rec)} free-running tokens, all taps vs the spec: "
            + " ".join(f"{k}={'ok' if v['ok'] else 'FAIL(' + str(v['first']) + ')'}[{v['worker_units']}wu]"
                       for k, v in rows.items()))
    rp = ctx["vec"] / "host_replay_tiny.npz"
    if rp.exists():
        tok, runs = bh.threads_check(rp, ctx["lib"], verbose=False)
        out["replay"] = dict(ok=tok, digests=sorted({r["digest"] for r in runs}),
                             runs=[dict(threads=r["threads"], kernels=r["kernels"], worker_units=r["worker_units"],
                                        bad=r["bad"]) for r in runs])
        log(f"  threads[replay] {len(runs)} runs (threads 1/3/4/2 x ref/fast): {'one digest ' + runs[0]['digest'] if len(out['replay']['digests']) == 1 else 'DIGESTS DIFFER'}"
            f" over all taps + KV cache, {'PASS' if tok else 'FAIL'} -> on the board: python3 -m fabric.big.host.bk_host --threads-check <file>")
        ok &= tok
        n_runs += len(runs)
    else:
        log(f"  threads[replay] {rp} missing (stage a writes it): FAIL")
        ok = False
    return ok, out, f"{n_runs} runs with 1/2/3/4 threads x fast/ref kernels bit-identical on every tap"


# ----------------------------------------------------------------------------- stage: tables
def stage_tables(ctx):
    ok, out = True, {}
    cfgs = {n: ctx["tiny"][n][0].c for n in TINY}
    for tag, f in (("1.7B", "Bonsai-1.7B-Q1_0.gguf"), ("8B", "Bonsai-8B-Q1_0.gguf")):
        p = ctx["data"] / f
        if p.exists():
            from .ref.gguf_q1 import GGUF
            g = GGUF(p)
            cfgs[tag] = cfg_from_gguf(g)
            same = bh.cfg_from_gguf(g).asdict() == bh.HostCfg.of(cfgs[tag]).asdict()
            ok &= same
            out[tag + "_cfg_np==torch"] = same
    for tag, c in cfgs.items():
        mp = 96 if tag in TINY else ctx["max_pos"]
        tn, tt = bh.build_tables(c, mp), bh.build_tables_torch(c, mp)
        bad = bh.tables_equal(tn, tt)
        inv_same = beq(tn["inv_freq"], tt["inv_freq"])
        r = dict(numpy_eq_spec=not bad, differing=bad, inv_freq_eq=inv_same, source=str(tn["source"]),
                 digest=bh.tables_digest(tt), max_pos=mp)
        if tag in TINY:                                  # and equal to what the FabricRef instance holds
            r["eq_fabricref"] = not bh.tables_equal(tn, dict(spec_tables(ctx["tiny"][tag][0])))
            ok &= r["eq_fabricref"]
        else:                                            # ship the spec's tables; reload and compare
            fp = ctx["vec"] / f"host_tables_{tag}.npz"
            bh.save_tables(fp, tt)
            back = bh.load_tables(fp, c, mp)
            r.update(file=str(fp), roundtrip=not bh.tables_equal(back, tt))
            ok &= r["roundtrip"]
        ok &= not bad and inv_same
        out[tag] = r
        log(f"  tables[{tag}] numpy == spec: {'yes' if not bad else 'NO ' + str(bad)} (inv_freq via {r['source']}), "
            f"digest {r['digest']}, {mp} positions" + (f", saved {r['file']}" if "file" in r else ""))
    # the pin is real: without it the numpy pow differs from torch's at exactly the pinned entries
    pins = bh._POW_PINS
    bh._POW_PINS = {}
    try:
        c = cfgs["A"]
        raw = bh.rope_params_np(c)[0]
    finally:
        bh._POW_PINS = pins
    from .ref.qwen3_ref import rope_params
    d = np.nonzero(bits(raw) != bits(rope_params(c)[0].numpy()))[0].tolist()
    out["pow_unpinned_diff_idx"] = d
    log(f"  tables: torch's fp32 pow vs the correctly rounded value (head_dim 128, theta 1e6): differs at index {d} "
        f"(pinned in bk_host._POW_PINS)")
    return ok, out, f"numpy tables == spec for {len(cfgs)} configs"


# ----------------------------------------------------------------------------- stage: bugs
def stage_bugs(ctx):
    fab, prompt, rec = ctx["tiny"]["A"]
    out, caught = {}, 0
    for name, (defs, flags, libs, what) in BUGS.items():
        lib = bh.load_lib(defines=defs, flags=bh.DEFAULT_FLAGS + flags, libs=libs)
        pr = prim_check(lib, ctx["prim_ref"])
        fails = [k for k, v in pr.items() if not v[0]]
        r = freerun_check(fab, lib, rec, prompt)
        r.pop("host")
        # the same bug in the scalar reference kernels, single-threaded
        pr_r = prim_check(lib, ctx["prim_ref"], ref=True)
        fails_r = [k for k, v in pr_r.items() if not v[0]]
        rr = freerun_check(fab, lib, rec, prompt, kernels="ref", threads=1)
        rr.pop("host")
        hit_f, hit_r = bool(fails) or not r["ok"], bool(fails_r) or not rr["ok"]
        hit = hit_f and hit_r
        caught += hit
        out[name] = dict(what=what, caught=hit, caught_fast=hit_f, caught_ref=hit_r, prim_fail=fails,
                         freerun_first=r["first"], ref_prim_fail=fails_r, ref_freerun_first=rr["first"])
        log(f"  bug {name:11s} {'CAUGHT' if hit else 'MISSED'}: {what}\n      fast kernels, {r['host_threads']} threads: "
            f"prim: {', '.join(fails) or '-'}; free-run: {r['first'] or 'no mismatch'}\n      reference kernels, 1 thread: "
            f"prim: {', '.join(fails_r) or '-'}; free-run: {rr['first'] or 'no mismatch'}")
    n = len(BUGS)
    # the real thing: let the compiler contract (x86 needs -mfma to have an FMA at all)
    if platform.machine() == "x86_64" and "fma" in Path("/proc/cpuinfo").read_text():
        fl = ["-O3", "-mfma", "-ffp-contract=fast", "-fno-fast-math", "-shared", "-fPIC"]
        l1 = bh.load_lib(defines=["BK_ALLOW_CONTRACT"], flags=fl)
        st = l1.bk_selftest()
        try:
            make_host(fab, l1, gemv=SpecGemv(fab))
            refused = False
        except bh.BkError as ex:
            refused = ex.rc == -7
        l2 = bh.load_lib(defines=["BK_ALLOW_CONTRACT", "BK_SKIP_SELFTEST"], flags=fl)
        pr = prim_check(l2, ctx["prim_ref"])
        fails = [k for k, v in pr.items() if not v[0]]
        r = freerun_check(fab, l2, rec, prompt)
        r.pop("host")
        hit = st != 0 and refused and (bool(fails) or not r["ok"])
        l3 = bh.load_lib(flags=fl)                       # same flags, but the source's pragma is active
        r3 = freerun_check(fab, l3, rec, prompt)
        r3.pop("host")
        out["REAL_CONTRACT"] = dict(what="-mfma -ffp-contract=fast with the source pragma disabled", caught=hit,
                                    selftest_mask=st, init_refused=refused, prim_fail=fails, freerun_first=r["first"],
                                    pragma_alone_protects=r3["ok"] and l3.bk_selftest() == 0)
        caught += hit
        n += 1
        log(f"  bug REAL_CONTRACT {'CAUGHT' if hit else 'MISSED'}: built -mfma -ffp-contract=fast, pragma off: selftest mask {st:#x}, "
            f"bk_init refused: {refused}; with the selftest skipped: prim {', '.join(fails) or '-'}; free-run: {r['first'] or 'no mismatch'}\n"
            f"      same flags with the source's fp-contract pragma left on: {'bit-identical' if out['REAL_CONTRACT']['pragma_alone_protects'] else 'MISMATCH'}")
    return caught == n, out, f"{caught}/{n} planted bugs caught"


# ----------------------------------------------------------------------------- stage: b
def tokenizer(ctx):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(ctx["data"] / "Bonsai-8B-meta"))


def lockstep(fab, host, cb, toks, n_prompt, n_gen, on_pos=None, want_from=None, tag=""):
    """Spec (FabricRef.step, XTrace) and C library token by token on the same tokens; the C
    library decides the generated ones. Every GEMV input, the logits, the taps and the KV rows
    of every step are compared. want_from: first step whose logits the C library computes."""
    c, names = fab.c, names_of(fab.c)
    want_from = n_prompt - 1 if want_from is None else want_from
    host.enable_taps(("xq", "xe") + EXT, att_cap=len(toks) + n_gen + 1)
    r = dict(n_in=0, n_ext=0, bad_steps=0, bad_logits=0, bad_tokens=0, first=None, gen=[], per=[])
    toks = list(toks)
    fab.reset()
    i, t0 = 0, time.time()
    while i < len(toks):
        tok, want = toks[i], i >= want_from
        if on_pos is not None:
            on_pos(i)
        tr = XTrace()
        ts = time.time()
        lg_ref = fab.step(tok, trace=tr)
        t_spec = time.time() - ts
        host.reset_stats()
        lg = host.step(tok, want_logits=want, copy=False)
        st = host.stats()
        r["per"].append(dict(Tk=i + 1, host_ms=st["host_ns"] / 1e6, **{k + "_ms": v / 1e6 for k, v in st["ns"].items()}))
        s = step_signals(tr, c, names, lg_ref)
        bad, a_, b_ = compare_step(host, s, i, want, names)
        r["n_in"] += a_
        r["n_ext"] += b_
        if want:
            if not beq(lg, s["logits"]):
                bad.append("logits")
                r["bad_logits"] += 1
            a, b = bh.argmax(lg, host.lib), argmax_low(lg_ref)
            r["bad_tokens"] += int(a != b)
            if i >= n_prompt - 1 and len(r["gen"]) < n_gen:
                r["gen"].append(a)
                toks.append(a)
        if bad:
            r["bad_steps"] += 1
            r["first"] = r["first"] or f"step {i}: " + ", ".join(bad[:4])
        i += 1
        log(f"    {tag} position {i - 1}: host {st['host_ns'] / 1e6:6.2f} ms, callback {st['gemv_ns'] / 1e9:5.2f} s, spec step {t_spec:5.2f} s, "
            f"{'ok' if not bad else 'MISMATCH ' + bad[0]}  ({time.time() - t0:.0f}s)")
    r["tokens"] = toks
    r["ok"] = r["bad_steps"] == 0 and r["bad_tokens"] == 0 and cb.requant_bad == 0
    return r


def stage_b(ctx):
    p = ctx["data"] / "Bonsai-1.7B-Q1_0.gguf"
    dev = ctx["device"]
    tk = tokenizer(ctx)
    prompt = tk(PROMPT_B).input_ids
    t0 = time.time()
    cfg, W, _ = from_gguf(str(p), device=dev)
    fab = FabricRef(cfg, W, device=dev, max_pos=ctx["max_pos"])
    cb = SpecGemv(fab)
    host = bh.BkHost.from_gguf(str(p), gemv=cb, max_pos=ctx["max_pos"])      # numpy-only loader + numpy tables
    tab_bad = bh.tables_equal(host.tables, spec_tables(fab))
    log(f"  b: Bonsai-1.7B on {dev} (L{cfg.n_layer} d{cfg.d} heads {cfg.n_head}/{cfg.n_kv}), prompt {len(prompt)} tokens, "
        f"loaded in {time.time() - t0:.0f}s; C tables == spec tables: {not tab_bad}")
    r = lockstep(fab, host, cb, prompt, len(prompt), ctx["n_gen"], tag="b")
    ok = r["ok"] and not tab_bad
    text = tk.decode(r["gen"], skip_special_tokens=False)
    log(f"  b: {len(r['tokens'])} tokens ({len(prompt)} prompt + {len(r['gen'])} generated): {r['n_in']} GEMV inputs and "
        f"{r['n_ext']} taps / KV rows compared, steps with a mismatch {r['bad_steps']}; logits differing {r['bad_logits']}; "
        f"tokens differing {r['bad_tokens']}; text {text!r}")
    out = dict(model=p.name, device=dev, prompt=PROMPT_B, prompt_tokens=len(prompt), generated=r["gen"], text=text,
               gemv_inputs=r["n_in"], ext_signals=r["n_ext"], bad_steps=r["bad_steps"], bad_logits=r["bad_logits"],
               bad_tokens=r["bad_tokens"], requant_bad=cb.requant_bad, first=r["first"], tables_eq_spec=not tab_bad,
               per_position=r["per"])
    del fab, W, host, cb
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return ok, out, f"1.7B {len(r['tokens'])} tokens ({len(prompt)}+{len(r['gen'])}) / {r['n_in']} GEMV inputs"


# ----------------------------------------------------------------------------- stage: c
def stage_c(ctx):
    p, gp = ctx["data"] / "Bonsai-8B-Q1_0.gguf", ctx["golden"]
    dev = ctx["device"]
    gold = np.load(gp)
    toks, gpos, gmax = [int(t) for t in gold["tokens"]], int(gold["pos"]), int(gold["argmax"])
    t0 = time.time()
    cfg, W, _ = from_gguf(str(p), device=dev)
    fab = FabricRef(cfg, W, device=dev, max_pos=ctx["max_pos"])
    names = names_of(cfg)
    cur = dict(pos=-1)
    cmp_ = dict(n=0, bad_x=0, bad_y=0, first=None)

    def on_call(mid, name, xq, xe, y):                   # the C library's GEMVs at the golden position
        if cur["pos"] != gpos:
            return
        cmp_["n"] += 1
        okx = beq(xq, gold[name + ".xq"]) and beq(xe, gold[name + ".xe"])
        oky = np.array_equal(y.view(np.uint32), gold[name + ".y"])
        cmp_["bad_x"] += int(not okx)
        cmp_["bad_y"] += int(not oky)
        if not (okx and oky) and cmp_["first"] is None:
            cmp_["first"] = name

    cb = SpecGemv(fab, on_call)
    host = bh.BkHost.from_gguf(str(p), gemv=cb, max_pos=ctx["max_pos"])      # embedding straight from the mmap'd GGUF
    tab_bad = bh.tables_equal(host.tables, spec_tables(fab))
    log(f"  c: Bonsai-8B on {dev} (L{cfg.n_layer} d{cfg.d} ffn{cfg.ffn} heads {cfg.n_head}/{cfg.n_kv}), {len(toks)} prompt tokens, "
        f"golden position {gpos}, loaded in {time.time() - t0:.0f}s; C tables == spec tables: {not tab_bad}")
    if ctx["c_lockstep"]:                                # also: the spec stepping alongside, all positions
        r = lockstep(fab, host, cb, toks[:gpos + 1], gpos + 1, 0, on_pos=lambda i: cur.update(pos=i),
                     want_from=gpos, tag="c")
        per, lock_ok = r["per"], r["ok"]
    else:
        per, r, lock_ok = [], None, True
        for i, t in enumerate(toks[:gpos + 1]):
            cur["pos"] = i
            host.reset_stats()
            host.step(t, want_logits=(i == gpos), copy=False)
            st = host.stats()
            per.append(dict(Tk=i + 1, host_ms=st["host_ns"] / 1e6, **{k + "_ms": v / 1e6 for k, v in st["ns"].items()}))
            log(f"    c position {i}: host {st['host_ns'] / 1e6:.2f} ms, callback {st['gemv_ns'] / 1e9:.1f} s  ({time.time() - t0:.0f}s)")
    am = bh.argmax(host.logits, host.lib)
    ok = cmp_["n"] == len(names) and cmp_["bad_x"] == 0 and cmp_["bad_y"] == 0 and am == gmax \
        and cb.requant_bad == 0 and not tab_bad and lock_ok
    log(f"  c: golden file, position {gpos}: {cmp_['n']} GEMVs, inputs differing {cmp_['bad_x']}, outputs differing {cmp_['bad_y']}"
        f"{' (first ' + str(cmp_['first']) + ')' if cmp_['first'] else ''}; argmax {am} vs golden {gmax}"
        + (f"; lockstep with the spec over positions 0..{gpos}: {r['n_in']} GEMV inputs + {r['n_ext']} taps / KV rows, "
           f"steps with a mismatch {r['bad_steps']}" if r else ""))
    out = dict(model=p.name, device=dev, golden=str(gp), prompt_tokens=len(toks), golden_pos=gpos, gemvs=cmp_["n"],
               bad_inputs=cmp_["bad_x"], bad_outputs=cmp_["bad_y"], first=cmp_["first"], argmax=am, golden_argmax=gmax,
               requant_bad=cb.requant_bad, tables_eq_spec=not tab_bad, per_position=per)
    if r:
        out.update(lockstep_gemv_inputs=r["n_in"], lockstep_ext_signals=r["n_ext"], lockstep_bad_steps=r["bad_steps"],
                   lockstep_first=r["first"])
    ctx["real_8b"] = per
    del fab, W, host, cb
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return ok, out, f"8B golden position {gpos}: {cmp_['n'] - cmp_['bad_x']}/{len(names)} inputs, " \
                    f"{cmp_['n'] - cmp_['bad_y']}/{len(names)} outputs, argmax {'ok' if am == gmax else 'DIFFERS'}" \
                    + (f", lockstep {r['n_in']} inputs over {gpos + 1} positions" if r else "")


# ----------------------------------------------------------------------------- stage: time
def stage_time(ctx):
    lib = ctx["lib"]
    cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")), platform.machine())
    log(f"  time: 8B shape (L36 d4096 ffn12288 heads 32/8 hd128), stand-in GEMV, {cpu}, 1 thread, median of {ctx['time_reps']}\n"
        f"  build: {' '.join(bh.build_cmd('libbk_host.so'))}")
    r = bh.bench("8b", ctx["time_ctx"], reps=ctx["time_reps"], lib=lib)
    r.update(build=" ".join(bh.build_cmd("libbk_host.so")), cpu=cpu)
    # the same with the plain -O2 command and with -march=native (same results, other speed)
    alt = {}
    for tag, flags in (("O2", bh.BASE_FLAGS), ("O3_native", ["-O3", "-march=native"] + bh.BASE_FLAGS[1:])):
        log(f"  time[{tag}]: {' '.join(bh.build_cmd('libbk_host.so', flags))}")
        alt[tag] = bh.bench("8b", (20, 256), reps=ctx["time_reps"], lib=bh.load_lib(flags=flags))
        alt[tag]["build"] = " ".join(bh.build_cmd("libbk_host.so", flags))
    r["alt_builds"] = alt
    if ctx.get("real_8b"):
        r["real_8b_per_position"] = ctx["real_8b"]
    (RES / "host_timing.json").write_text(json.dumps(r, indent=1))
    row = next((x for x in r["rows"] if x["Tk"] == 20), r["rows"][0])
    att = row["att_score_ms"] + row["att_softmax_ms"] + row["att_ctx_ms"]
    rest = row["host_ms"] - att - row["rmsnorm_ms"] - row["quant_ms"] - row["silu_ms"]
    log(f"  time: host ms/token at Tk {row['Tk']}: total {row['host_ms']:.2f} = rmsnorm {row['rmsnorm_ms']:.2f} + quant {row['quant_ms']:.2f} "
        f"+ attention {att:.2f} + silu {row['silu_ms']:.2f} + rest {rest:.2f} (rope {row['rope_ms']:.3f}, kv-quant {row['kvq_ms']:.3f}, "
        f"residual {row['resid_ms']:.3f}, embed {row['embed_ms']:.3f}); attention grows {r.get('attention_us_per_pos', 0):.1f} us per "
        f"cached position")
    return True, dict(file=str(RES / "host_timing.json"), non_attention_ms=r.get("non_attention_ms"),
                      attention_us_per_pos=r.get("attention_us_per_pos"), host_ms_tk20=row["host_ms"]), \
        f"8B-shape host {row['host_ms']:.2f} ms/token at Tk {row['Tk']} (x86, excluding the callback)"


STAGES = dict(prim=stage_prim, kern=stage_kern, tables=stage_tables, a=stage_a, threads=stage_threads, bugs=stage_bugs,
              b=stage_b, c=stage_c, time=stage_time)


def main():
    ap = argparse.ArgumentParser(description="gate of the Big-Kev host vector path (bk_host.c) vs fabric_ref.py")
    ap.add_argument("--stages", default="prim,kern,tables,a,threads,bugs,b,c,time")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default="data/big")
    ap.add_argument("--golden", default=str(VEC / "Bonsai-8B-Q1_0_token.npz"))
    ap.add_argument("--vec", default=str(VEC), help="where the replay / table files for the board go")
    ap.add_argument("--max-pos", type=int, default=4096)
    ap.add_argument("--n-gen", type=int, default=20)
    ap.add_argument("--time-ctx", default="1,20,64,256,1024")
    ap.add_argument("--time-reps", type=int, default=5)
    ap.add_argument("--no-c-lockstep", action="store_true", help="stage c: golden file only, no spec stepping alongside")
    ap.add_argument("--out", default=str(RES / "host_gate.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    stages = [s for s in a.stages.split(",") if s]
    assert all(s in STAGES for s in stages), f"stages: {list(STAGES)}"
    t0 = time.time()
    lib = bh.load_lib()
    ctx = dict(lib=lib, device=a.device, data=Path(a.data), golden=Path(a.golden), vec=Path(a.vec), max_pos=a.max_pos,
               n_gen=a.n_gen, c_lockstep=not a.no_c_lockstep, time_ctx=tuple(int(x) for x in a.time_ctx.split(",")), time_reps=a.time_reps)
    ctx["vec"].mkdir(parents=True, exist_ok=True)
    log(f"run_host: lib {lib._path}\n  build: {' '.join(bh.build_cmd('libbk_host.so'))}\n  selftest mask {lib.bk_selftest():#x}, "
        f"{platform.machine()}, torch {torch.__version__}, numpy {np.__version__}, device {a.device}")
    if any(s in stages for s in ("prim", "kern", "tables", "a", "threads", "bugs")):
        ctx["tiny"] = {}
        for name, spec in TINY.items():
            fab, prompt = tiny_model(spec)
            ctx["tiny"][name] = (fab, prompt, spec_record(fab, prompt, spec["n"]))
        ctx["prim_ref"] = prim_reference(ctx["tiny"]["A"][0])
    res, parts, ok = {}, [], lib.bk_selftest() == 0
    for s in stages:
        ts = time.time()
        log(f"--- stage {s}")
        try:
            sok, detail, summ = STAGES[s](ctx)
        except Exception as ex:                          # a stage that cannot run is a FAIL, not a skip
            import traceback
            traceback.print_exc()
            sok, detail, summ = False, dict(error=repr(ex)), f"error {type(ex).__name__}"
        res[s] = dict(ok=sok, seconds=round(time.time() - ts, 1), summary=summ, **detail)
        parts.append(f"{s}: {summ}{'' if sok else ' FAIL'}")
        ok &= sok
    skipped = [s for s in STAGES if s not in stages]
    verdict = "PASS" if ok else "FAIL"
    out = Path(a.out)
    prev = {}
    if out.exists():
        try:
            prev = json.loads(out.read_text()).get("stages", {})
        except Exception:
            prev = {}
    prev.update(res)
    out.write_text(json.dumps(dict(verdict_last_run=verdict, stages_last_run=stages, lib_build=" ".join(bh.build_cmd("libbk_host.so")),
                                   machine=platform.machine(), torch=torch.__version__, numpy=np.__version__,
                                   stages=prev), indent=1, default=str))
    log(f"HOST_VERDICT {verdict} ({'; '.join(parts)}"
        + (f"; NOT RUN: {','.join(skipped)}" if skipped else "") + f"; {time.time() - t0:.0f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
