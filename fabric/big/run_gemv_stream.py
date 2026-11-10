"""R2a gate: the streaming 1-bit x BFP10 GEMV RTL is bit-exact against the fabric spec.

For every case: build Q1 matrices + an activation vector, pack them with
`pack_weights.pack` into the port-interleaved DDR image, cut the image back into what each
read port delivers (`port_streams`, from the address map), compile `rtl/gemv_stream.v` +
`tb/tb_gemv_stream.sv` with iverilog, run with randomised per-port valid gaps and output
back-pressure, and compare every fp32 output word bit-for-bit against

  * the REFERENCE: `fabric_ref.ExactBfpGemv.mm` (imported, CPU) -- the spec; and
  * an INDEPENDENT model written here in Python integers (own fp16 decode, own shift,
    own int64 -> fp32 round-to-nearest-even), as a cross-check on both.

The activation the RTL is loaded with is the reference's own `bfp_quant(x, 10, 128)`.

Cases
  adversarial  max |inner| = 65408; every fp16 scale class (zero, -0, subnormal, min/max
               normal, exp = 31); negative shifts (floor) and left shifts up to 63 with
               accumulator wrap; rows that are exactly zero; crafted RNE ties in the
               int64 -> fp32 conversion; random shapes G = 1..GMAX chained on one stream
               (matrix boundaries, tuple- and stripe-aligned); one-tuple rows into a sink
               that is ready 3% of the time (output credits).
  ext_shift    shifts outside [-63, 63]. torch's int64 shift is not defined there, so the RTL
               is gated against the independent model (mod-2^64 semantics: 0, or floor) and
               the reference's agreement is printed as information only.
  real         64 rows of real Bonsai matrices: 1.7B l0.q (G=16) on a real layer-0 hidden
               state, 1.7B l0.down (G=48) on silu(gate)*up of that state, and (if the 8B file
               is present) 8B blk.0 attn_q (G=32) / ffn_down (G=96).

    python -m fabric.big.run_gemv_stream            # full gate
    python -m fabric.big.run_gemv_stream --quick    # adversarial only, NPORTS 1 and 3
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fabric.big import pack_weights as pw
from fabric.big.ref.fabric_ref import ACC_LSB, GEMV_BITS, ExactBfpGemv, bfp_quant
from fabric.big.ref.qwen3_ref import Q1Matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SIM = Path(os.environ.get("KEV_SIM_DIR", "/tmp/kevbuild")) / "gemv_stream"
IVERILOG = os.environ.get("IVERILOG", str(Path.home() / ".local/bin/iverilog"))
VVP = os.environ.get("VVP", str(Path.home() / ".local/bin/vvp"))
RTL = Path(os.environ.get("GEMV_STREAM_RTL", HERE / "rtl" / "gemv_stream.v"))   # override: mutation tests
GMAX, EXW = 96, 9
REGMASK = 0x7F              # sign-tree levels registered (bit l = level l); --regmask overrides
M64 = (1 << 64) - 1
GGUF_17 = ROOT / "data/big/Bonsai-1.7B-Q1_0.gguf"
GGUF_8 = ROOT / "data/big/Bonsai-8B-Q1_0.gguf"


# ----------------------------------------------------------------------------- models
def quantise(x: np.ndarray):
    """the reference's BFP10 quantiser -> (q int64 [n], e int64 [G])."""
    q, e = bfp_quant(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None], GEMV_BITS, 128)
    return q[0].to(torch.int64).numpy(), e[0].numpy()


def ref_gemv(signs, scales, x):
    """the spec: ExactBfpGemv.mm on CPU -> (fp32 bits uint32 [R], stats)."""
    W = Q1Matrix(torch.from_numpy(np.ascontiguousarray(signs)),
                 torch.from_numpy(np.ascontiguousarray(scales).view(np.float16).copy()))
    g = ExactBfpGemv()
    y = g.mm(W, torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None])[0]
    return y.numpy().view(np.uint32).copy(), g.stats


def i64_to_f32_bits(a: int):
    """RNE_fp32(a) * 2^-40 as IEEE bits, in integer arithmetic. a is a signed 64-bit int.
    -> (bits, was_tie)."""
    if a == 0:
        return 0, False
    sgn, mag = (1, -a) if a < 0 else (0, a)
    top = mag.bit_length() - 1
    tie = False
    if top > 23:
        sh = top - 23
        man, rem, half = mag >> sh, mag & ((1 << sh) - 1), 1 << (sh - 1)
        tie = rem == half
        if rem > half or (tie and man & 1):
            man += 1
            if man == 1 << 24:
                man >>= 1
                top += 1
    else:
        man = mag << (23 - top)
    return (sgn << 31) | ((top - ACC_LSB + 127) << 23) | (man & 0x7FFFFF), tie


def indep_gemv(signs, scales, q, e):
    """independent integer model -> (bits uint32 [R], ties, sh_min, sh_max, acc_bits)."""
    R, G = scales.shape
    pm = np.unpackbits(signs, axis=1, bitorder="little").astype(np.int64) * 2 - 1
    inner = (pm.reshape(R, G, 128) * q.reshape(1, G, 128)).sum(-1)
    out = np.zeros(R, dtype=np.uint32)
    ties, smin, smax, abits = 0, 999, -999, 0
    for r in range(R):
        acc = 0
        for g in range(G):
            b = int(scales[r, g])
            ex5, fr = (b >> 10) & 31, b & 0x3FF
            m, ed = (fr, -24) if ex5 == 0 else (fr | 0x400, ex5 - 25)
            if b >> 15:
                m = -m
            t = int(inner[r, g]) * m
            sh = ed + int(e[g]) + ACC_LSB
            smin, smax = min(smin, sh), max(smax, sh)
            acc += (t << sh) if sh >= 0 else (t >> -sh)          # python >> floors
        acc &= M64
        if acc >> 63:
            acc -= 1 << 64
        abits = max(abits, abs(acc).bit_length())
        out[r], tie = i64_to_f32_bits(acc)
        ties += tie
    return out, ties, smin, smax, abits


# ----------------------------------------------------------------------------- cases
@dataclass
class Run:
    name: str
    signs: np.ndarray            # uint8 [R, G*16]
    scales: np.ndarray           # uint16 [R, G]
    x: np.ndarray                # fp32 [G*128]


@dataclass
class Case:
    name: str
    runs: list
    ref_domain: bool = True      # False: shifts outside torch's defined range
    note: str = ""


def mk_x(q, e):
    """x = q * 2^e exactly in fp32 (q [G,128] ints, e [G])."""
    x64 = np.ldexp(np.asarray(q, dtype=np.float64), np.asarray(e)[:, None])
    x = x64.astype(np.float32)
    assert np.array_equal(x.astype(np.float64), x64), "activation not exact in fp32"
    return x.reshape(-1)


def rand_q(rng, G):
    q = rng.integers(-511, 512, (G, 128))
    q[np.arange(G), rng.integers(0, 128, G)] = rng.choice([-511, 511, 256, -256, 300], G)
    return q


def f16(m: int, ex5: int = 15, neg: bool = False) -> int:
    """fp16 bits whose fp16_decompose mantissa is m (normal if m >= 1024, else subnormal)."""
    return (int(neg) << 15) | ((ex5 << 10) | (m & 0x3FF) if m >= 1024 else m)


SPECIAL_SCALES = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0200, 0x0400, 0x8400,
                  0x7BFF, 0xFBFF, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0xFFFF, 0x7FFF,
                  0x3555, 0xB555, 0x0155, 0x4400]


def case_maxinner():
    G = 4
    q = np.zeros((G, 128), dtype=np.int64)
    q[0] = 511
    q[1] = -511
    q[2] = np.where(np.arange(128) % 2, -511, 511)
    q[3] = 511
    x = mk_x(q, [0, -3, 5, -7])
    alt = np.packbits((np.arange(128) % 2 == 0).astype(np.uint8), bitorder="little")
    ones, zeros = np.full(16, 0xFF, np.uint8), np.zeros(16, np.uint8)
    pats = [[ones, zeros, alt, ones], [zeros, ones, ~alt, zeros], [ones, ones, ones, ones],
            [zeros, zeros, zeros, zeros], [ones, zeros, alt, zeros], [zeros, ones, alt, ones],
            [ones, zeros, ~alt, ones], [zeros, zeros, alt, ones]]
    signs = np.stack([np.concatenate(p) for p in pats])
    sc = np.array([[0x7BFF] * 4, [0xFBFF] * 4, [0x7C00, 0xFFFF, 0x7BFF, 0xFBFF], [0x3C00, 0xBC00, 0x0001, 0x83FF],
                   [0x7BFF, 0x7BFF, 0x7BFF, 0x7BFF], [0xFBFF, 0x7BFF, 0xFBFF, 0x7BFF],
                   [0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF], [0x03FF, 0x0400, 0x8400, 0x0001]], dtype=np.uint16)
    return Case("adv_maxinner", [Run("maxinner", signs, sc, x)], note="|inner| = 65408")


def case_scales(rng):
    runs = []
    for G in (8, 9, 22):
        R = 16
        sc = rng.choice(np.array(SPECIAL_SCALES, dtype=np.uint16), (R, G))
        sc[::3] = rng.integers(0, 65536, (len(sc[::3]), G), dtype=np.uint16)
        for i, s in enumerate(SPECIAL_SCALES[:min(R * G, len(SPECIAL_SCALES))]):
            sc[i // G, i % G] = s
        signs = rng.integers(0, 256, (R, G * 16), dtype=np.uint8)
        x = (rng.standard_normal(G * 128) * 0.7).astype(np.float32)
        runs.append(Run(f"scales_g{G}", signs, sc, x))
    return Case("adv_scales", runs, note="zero/-0/subnormal/min/max/exp31 scales")


def case_negshift(rng):
    runs = []
    for G in (6, 13):
        R = 12
        e = rng.integers(-75, -37, G)
        x = mk_x(rand_q(rng, G), e)
        sc = rng.integers(0, 65536, (R, G), dtype=np.uint16)
        sc[:4] = rng.choice(np.array([0x0001, 0x8001, 0x03FF, 0x83FF, 0x0155, 0x8155, 0x0400, 0x8400], np.uint16), (4, G))
        runs.append(Run(f"negshift_g{G}", rng.integers(0, 256, (R, G * 16), dtype=np.uint8), sc, x))
    return Case("adv_negshift", runs, note="right shifts (floor) down to -59")


def case_bigshift(rng):
    runs = []
    for G in (5, 16):
        R = 12
        e = rng.integers(0, 18, G)
        e[0] = 17
        x = mk_x(rand_q(rng, G), e)
        sc = (rng.integers(24, 32, (R, G)) << 10 | rng.integers(0, 1024, (R, G)) |
              rng.integers(0, 2, (R, G)) << 15).astype(np.uint16)
        sc[0, 0] = 0x7FFF
        runs.append(Run(f"bigshift_g{G}", rng.integers(0, 256, (R, G * 16), dtype=np.uint8), sc, x))
    return Case("adv_bigshift", runs, note="left shifts up to 63, accumulator wraps")


def case_zero_rows(rng):
    G, R = 4, 10
    qh = rand_q(rng, 2)
    q = np.concatenate([qh[:1], qh[:1], qh[1:], qh[1:]])
    x = mk_x(q, [3, 3, -6, -6])
    half = rng.integers(0, 256, (R, 2, 16), dtype=np.uint8)
    signs = np.stack([half[:, 0], ~half[:, 0], half[:, 1], ~half[:, 1]], 1).reshape(R, G * 16)
    s2 = rng.integers(0, 65536, (R, 2), dtype=np.uint16)
    sc = np.stack([s2[:, 0], s2[:, 0], s2[:, 1], s2[:, 1]], 1)
    r1 = Run("cancel", signs, sc, x)                                   # every row sums to 0
    sc0 = rng.choice(np.array([0x0000, 0x8000], np.uint16), (6, 3))
    r2 = Run("zero_scale", rng.integers(0, 256, (6, 48), dtype=np.uint8), sc0,
             (rng.standard_normal(384) * 3).astype(np.float32))
    xz = (rng.standard_normal(384)).astype(np.float32)
    xz[128:256] = 0.0
    r3 = Run("zero_group", rng.integers(0, 256, (6, 48), dtype=np.uint8),
             rng.integers(0, 65536, (6, 3), dtype=np.uint16), xz)
    r4 = Run("zero_x", rng.integers(0, 256, (4, 32), dtype=np.uint8),
             rng.integers(0, 65536, (4, 2), dtype=np.uint16), np.zeros(256, np.float32))
    return Case("adv_zero_rows", [r1, r2, r3, r4], note="rows that sum to exactly zero")


def craft_q(target: int):
    """q [128] with sum(q) == target, max |q| >= 256 (so BFP keeps q and the exponent)."""
    q = np.zeros(128, dtype=np.int64)
    q[0] = 300
    rem = target - 300
    for j in range(1, 128):
        v = max(-511, min(511, rem))
        q[j] = v
        rem -= v
    assert rem == 0, target
    return q


def _find_tie(rng, mlo, mhi, lo, hi, mod, res):
    """(inner, m) with lo <= inner*m < hi and inner*m % mod == res, inner reachable by craft_q."""
    while True:
        m = int(rng.integers(mlo, mhi))
        a, b = lo // m + 1, min(hi // m, 64000)
        if a >= b:
            continue
        inner = int(rng.integers(a, b))
        if lo <= inner * m < hi and (inner * m) % mod == res:
            return inner, m


def case_rne_ties(rng):
    """acc = P * 2^30 with P = inner * m chosen so the 24-bit rounding is an exact tie."""
    want = [(18631, 1801)]                                             # P = 2^25 - 1: carry into the exponent
    spec = [(1 << 24, 1 << 25, 4, 1), (1 << 24, 1 << 25, 4, 3),       # 25 bits: tie, even / odd lsb
            (1 << 25, 1 << 26, 8, 2), (1 << 25, 1 << 26, 8, 6),       # 26 bits
            (1 << 26, 1 << 27, 16, 4), (1 << 26, 1 << 27, 16, 12),    # 27 bits
            (1 << 24, 1 << 25, 4, 2), (1 << 24, 1 << 25, 4, 0)]       # exact: no rounding at all
    for lo, hi, mod, res in spec:
        for _ in range(3):
            want.append(_find_tie(rng, 1024, 2048, lo, hi, mod, res))
    for lo, hi, mod, res in spec[:2]:                                  # subnormal scale mantissas
        want.append(_find_tie(rng, 600, 1024, lo, hi, mod, res))
    ones, zeros = np.full(16, 0xFF, np.uint8), np.zeros(16, np.uint8)
    runs = []
    for i, (inner, m) in enumerate(want):
        ex5 = 15 if m >= 1024 else 0                                   # e_d = -10 | -24
        e0 = 0 if m >= 1024 else 14                                    # sh = 30 either way
        q = craft_q(inner)
        if i % 3 == 2:                                                 # near-ties: a second group adds +-16
            q2 = np.zeros(128, dtype=np.int64)
            q2[5] = 256
            x = mk_x(np.stack([q, q2]), [e0, -20])
            signs = np.stack([np.concatenate(p) for p in
                              ([ones, ones], [ones, zeros], [zeros, ones], [zeros, zeros])])
            sc = np.array([[f16(m, ex5), 0x0001], [f16(m, ex5), 0x0001],
                           [f16(m, ex5), 0x0001], [f16(m, ex5, True), 0x8001]], dtype=np.uint16)
        else:
            x = mk_x(q[None], [e0])
            signs = np.stack([ones, zeros, ones, zeros])
            sc = np.array([[f16(m, ex5)], [f16(m, ex5)], [f16(m, ex5, True)], [f16(m, ex5, True)]], dtype=np.uint16)
        runs.append(Run(f"tie{i}", signs, sc, x))
    return Case("adv_rne_ties", runs, note="crafted int64->fp32 ties")


def rand_run(rng, name, G, R, elo=-45, ehi=15):
    signs = rng.integers(0, 256, (R, G * 16), dtype=np.uint8)
    sc = rng.integers(0, 65536, (R, G), dtype=np.uint16)
    k = rng.random((R, G)) < 0.15
    sc[k] = rng.choice(np.array(SPECIAL_SCALES, dtype=np.uint16), int(k.sum()))
    x = rng.standard_normal((G, 128)) * np.exp2(rng.integers(elo + 9, ehi + 1, G))[:, None]
    x[rng.random(G) < 0.1] = 0.0
    return Run(name, signs, sc, x.astype(np.float32).reshape(-1))


def case_random(rng, name, shapes):
    return Case(name, [rand_run(rng, f"g{G}r{R}", G, R) for G, R in shapes], note="random, chained")


def case_ext_shift(rng):
    runs = []
    for i, (elo, ehi) in enumerate(((24, 118), (-149, -80), (-149, 118))):
        G, R = 7, 10
        x = mk_x(rand_q(rng, G), rng.integers(elo, ehi + 1, G))
        runs.append(Run(f"ext{i}", rng.integers(0, 256, (R, G * 16), dtype=np.uint8),
                        rng.integers(0, 65536, (R, G), dtype=np.uint16), x))
    return Case("ext_shift", runs, ref_domain=False, note="shifts beyond +-63 (outside torch's defined range)")


def cases_real(nrows=64, r0=100):
    out = []
    if GGUF_17.exists():
        from fabric.big.ref.qwen3_ref import from_gguf
        cfg, W, _ = from_gguf(str(GGUF_17), "cpu")

        def rms(v, w):
            return (v * torch.rsqrt(v.pow(2).mean() + cfg.eps) * w).to(torch.float32)

        emb = W["embed"].dequant(9707, 9708)[0].to(torch.float32)           # one real embedding row
        h = rms(emb, W["l0.attn_norm"])                                      # the real input of l0.q/k/v
        h2 = rms(emb, W["l0.ffn_norm"])
        g = ExactBfpGemv()
        act = torch.nn.functional.silu(g(W["l0.gate"], h2)) * g(W["l0.up"], h2)   # down's input shape

        def rows(key):
            return W[key].signs[r0:r0 + nrows].numpy().copy(), W[key].scales[r0:r0 + nrows].numpy().view(np.uint16).copy()

        out.append(Case("real_1p7B_l0.q_G16", [Run("l0.q", *rows("l0.q"), h.numpy())],
                        note=f"rows {r0}..{r0 + nrows}, x = attn_norm(embed row) (real layer-0 input)"))
        out.append(Case("real_1p7B_l0.down_G48", [Run("l0.down", *rows("l0.down"), act.numpy())],
                        note=f"rows {r0}..{r0 + nrows}, x = silu(gate h)*(up h), attention residual skipped"))
        out.append(Case("real_1p7B_chain_q_k_v", [Run(k, *rows(k), h.numpy()) for k in ("l0.q", "l0.k", "l0.v")],
                        note="q,k,v back to back on one stream, same activation"))
    else:
        print(f"NOTE {GGUF_17} missing: no 1.7B real-weight cases")
    if GGUF_8.exists():
        from fabric.big.ref.gguf_q1 import GGUF
        g8 = GGUF(str(GGUF_8))
        rng = np.random.default_rng(88)

        def rows8(name):
            R, G = pw.gguf_shape(g8, name)
            ti = g8.tensors[name]
            b = np.asarray(g8.raw(name)[r0 * G * 18:(r0 + nrows) * G * 18]).reshape(nrows * G, 18)
            return (np.ascontiguousarray(b[:, 2:]).reshape(nrows, G * 16),
                    np.ascontiguousarray(b[:, :2]).view(np.uint16).reshape(nrows, G))

        e_s, e_sc = (lambda b: (np.ascontiguousarray(b[:, 2:]).reshape(1, -1), np.ascontiguousarray(b[:, :2]).view(np.uint16).reshape(1, -1)))(
            np.asarray(g8.raw("token_embd.weight")[9707 * 32 * 18:9708 * 32 * 18]).reshape(32, 18))
        emb = (np.unpackbits(e_s, axis=1, bitorder="little").astype(np.float32) * 2 - 1) * \
            np.repeat(e_sc.view(np.float16).astype(np.float32), 128, axis=1)
        a = g8.meta["general.architecture"]
        eps = float(g8.meta.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6))
        h = (emb[0] / np.sqrt(np.mean(emb[0] ** 2) + eps) * g8.dequant("blk.0.attn_norm.weight").reshape(-1)).astype(np.float32)
        out.append(Case("real_8B_l0.q_G32", [Run("l0.q", *rows8("blk.0.attn_q.weight"), h)],
                        note=f"rows {r0}..{r0 + nrows}, x = attn_norm(embed row)"))
        out.append(Case("real_8B_l0.down_G96", [Run("l0.down", *rows8("blk.0.ffn_down.weight"),
                                                    (rng.standard_normal(12288) * 0.05).astype(np.float32))],
                        note=f"rows {r0}..{r0 + nrows}, x = synthetic normal(0, 0.05)"))
    else:
        print(f"NOTE {GGUF_8} missing: no 8B real-weight cases")
    return out


# ----------------------------------------------------------------------------- sim
_built = {}


def build(nports: int) -> Path:
    if nports in _built:
        return _built[nports]
    SIM.mkdir(parents=True, exist_ok=True)
    out = SIM / f"tb_n{nports}.vvp"
    cmd = [IVERILOG, "-g2012", f"-Ptb_gemv_stream.NPORTS={nports}", f"-Ptb_gemv_stream.GMAX={GMAX}",
           f"-Ptb_gemv_stream.EXW={EXW}", f"-Ptb_gemv_stream.TREE_REGMASK={REGMASK}", "-o", str(out),
           str(RTL), str(HERE / "tb" / "tb_gemv_stream.sv")]
    subprocess.run(cmd, check=True)
    _built[nports] = out
    return out


@dataclass
class Stim:
    gap: int = 0
    bp: int = 0
    seed: int = 1
    chunk: int = 16
    align: str = "tuple"
    slots: int = 0
    port_slots: tuple = ()

    def tag(self):
        s = f"gap{self.gap}/bp{self.bp} chunk{self.chunk} {self.align}"
        return s + (f" slots{self.slots}{list(self.port_slots)}" if self.slots else "")


def run_case(case: Case, nports: int, st: Stim, models: dict):
    """-> (ok, line)."""
    slots, ps = (st.slots, st.port_slots) if st.slots and len(st.port_slots) == nports else (0, ())
    lay = pw.Layout(nports, st.chunk, slots, ps, st.align)
    mats = [(r.name, (r.signs, r.scales)) for r in case.runs]
    img, index = pw.pack(mats, lay)
    rt = pw.roundtrip_ok(mats, lay)
    ports = pw.port_streams(img, lay)                           # [nports, beats, 16]
    d = SIM / f"{case.name}_n{nports}"
    d.mkdir(parents=True, exist_ok=True)
    nb = ports.shape[1]
    for k in range(nports):
        with open(d / f"port{k}.mem", "w") as f:
            f.write("\n".join(bytes(b[::-1]).hex() for b in ports[k]) + "\n")
    am, ae, rl = [], [], []
    for r, e in zip(case.runs, index):
        m = models[r.name]
        for g in range(e.groups):
            w = 0
            for j, v in enumerate(m["q"][g * 128:(g + 1) * 128]):
                w |= (int(v) & 0x3FF) << (10 * j)
            am.append(f"{w:0320x}")
            assert -(1 << (EXW - 1)) <= int(m["e"][g]) < (1 << (EXW - 1))
            ae.append(f"{int(m['e'][g]) & ((1 << EXW) - 1):03x}")
        rl += [f"{e.groups:08x}", f"{e.rows:08x}", f"{e.pad:08x}"]
    (d / "act_m.mem").write_text("\n".join(am) + "\n")
    (d / "act_e.mem").write_text("\n".join(ae) + "\n")
    (d / "runs.mem").write_text("\n".join(rl) + "\n")
    (d / "out.hex").unlink(missing_ok=True)
    cmd = [VVP, "-n", str(build(nports)), f"+dir={d}", f"+nbeats={nb}", f"+nruns={len(case.runs)}",
           f"+gap={st.gap}", f"+bp={st.bp}", f"+seed={st.seed}",
           f"+timeout={60 * sum(e.tuples + e.pad for e in index) + 400 * sum(e.rows for e in index) + 5000 * len(index) + 20000}"]
    so = subprocess.run(cmd, capture_output=True, text=True).stdout
    md = re.search(r"GEMV_TB_DONE outputs=(\d+) xout=(\d+) pops=(\d+)", so)
    if not md or "GEMV_TB_TIMEOUT" in so:
        return False, f"sim did not finish: {so[-300:]!r}", None
    nout, xout, pops = (int(v) for v in md.groups())
    words = [ln.split() for ln in (d / "out.hex").read_text().split("\n") if ln.strip()]
    got = np.array([int(w[0], 16) if "x" not in w[0].lower() else 0xFFFFFFFF for w in words], dtype=np.uint64)
    last = [int(w[1]) for w in words]
    ref = np.concatenate([models[r.name]["ref"] for r in case.runs]).astype(np.uint64)
    ind = np.concatenate([models[r.name]["ind"] for r in case.runs]).astype(np.uint64)
    n = len(ref)
    bad = []
    if not rt:
        bad.append("pack round-trip")
    if xout or len(got) != n:
        bad.append(f"outputs {len(got)}/{n} x={xout}")
    exp_pops = sum(e.tuples + e.pad for e in index)
    if pops != exp_pops:
        bad.append(f"tuples popped {pops} != {exp_pops}")
    want_last = []
    for e in index:
        want_last += [0] * (e.rows - 1) + [1]
    if last != want_last:
        bad.append("m_last wrong")
    m_ref = int((got[:n] == ref[:len(got)]).sum()) if len(got) else 0
    m_ind = int((got[:n] == ind[:len(got)]).sum()) if len(got) else 0
    if m_ind != n:
        bad.append("RTL != independent model")
    if case.ref_domain and m_ref != n:
        bad.append("RTL != ExactBfpGemv")
    if not bad and len(got) == n and m_ref != n and case.ref_domain:
        bad.append("ref mismatch")
    cyc = [(int(a), int(b), int(c)) for a, b, c in re.findall(r"GEMV_TB_RUN run=\d+ cycles=(\d+) rows=(\d+) groups=(\d+)", so)]
    gpc = max((r * g / c for c, r, g in cyc), default=0.0)
    ok = not bad
    line = (f"{'PASS' if ok else 'FAIL'} {case.name:24s} n={nports} {st.tag():28s} rows={n:4d} "
            f"rtl==ref {m_ref}/{n}{'' if case.ref_domain else ' (info)'} rtl==indep {m_ind}/{n} "
            f"tuples={pops}" + (f"  !! {'; '.join(bad)}" if bad else ""))
    return ok, line, (gpc, cyc)


def model_case(case: Case):
    """reference + independent results per run, and the case's stats line."""
    models, ties, zeros, smin, smax, abits, ind_ref = {}, 0, 0, 999, -999, 0, True
    for r in case.runs:
        q, e = quantise(r.x)
        ref, _ = ref_gemv(r.signs, r.scales, r.x)
        ind, t, lo, hi, ab = indep_gemv(r.signs, r.scales, q, e)
        models[r.name] = dict(q=q, e=e, ref=ref, ind=ind)
        ties += t
        zeros += int((ind == 0).sum())
        smin, smax, abits = min(smin, lo), max(smax, hi), max(abits, ab)
        ind_ref &= bool(np.array_equal(ref, ind))
    info = (f"     {case.name}: {len(case.runs)} run(s), {sum(r.scales.shape[0] for r in case.runs)} rows, "
            f"G={sorted({r.scales.shape[1] for r in case.runs})}, shifts {smin}..{smax}, |acc| <= {abits} bits, "
            f"exact RNE ties {ties}, zero rows {zeros}, ref==indep {'yes' if ind_ref else 'NO'}"
            + (f"  [{case.note}]" if case.note else ""))
    return models, info, ind_ref


def main():
    global REGMASK
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="adversarial cases only, NPORTS 1 and 3")
    ap.add_argument("--nports", type=int, nargs="*", default=None)
    ap.add_argument("--regmask", type=lambda v: int(v, 0), default=REGMASK,
                    help="sign-tree register mask (bit 6 must be set), e.g. 0x55 for a 4-stage tree")
    a = ap.parse_args()
    REGMASK = a.regmask
    assert REGMASK & 0x40 and 0 < REGMASK < 0x80
    t_start = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(20261110)

    adv = [case_maxinner(), case_scales(rng), case_negshift(rng), case_bigshift(rng), case_zero_rows(rng),
           case_rne_ties(rng),
           case_random(rng, "adv_random_small", [(1, 1), (1, 5), (2, 3), (7, 4), (8, 2), (9, 3), (15, 1), (17, 6), (1, 2)]),
           case_random(rng, "adv_random_big", [(32, 6), (96, 4), (48, 3), (GMAX, 2), (95, 3), (16, 9)]),
           case_ext_shift(rng),
           case_random(rng, "adv_credits", [(1, 150), (2, 80), (8, 40), (1, 60)])]
    real = [] if a.quick else cases_real()
    sweep = a.nports or ([1, 3] if a.quick else [1, 2, 3, 4])

    ok_all = pw.selftest()
    n_lines = n_rows = 0
    best = {}
    for case in adv + real:
        models, info, ind_ref = model_case(case)
        print(info)
        if case.ref_domain and not ind_ref:
            ok_all = False
            print(f"FAIL {case.name}: independent model disagrees with ExactBfpGemv")
        is_real = case.name.startswith("real")
        for n in sweep:
            if is_real and n not in (1, 3) and not a.nports:
                continue
            stims = [Stim(0, 0, 1, 16, "tuple"),
                     Stim(35, 30, 2 + n, 4, "stripe"),
                     Stim(70, 5, 5 + n, 256, "tuple", 4 if n == 3 else 0, (0, 1, 3) if n == 3 else ())]
            if is_real:
                stims = stims[:2]
            if case.name == "adv_credits":      # short rows + a nearly-stuck sink: the output FIFO
                stims = [Stim(0, 97, 11 + n, 16, "tuple"), Stim(20, 90, 12 + n, 4, "stripe")]   # must not overflow
            for st in stims:
                ok, line, perf = run_case(case, n, st, models)
                ok_all &= ok
                n_lines += 1
                n_rows += sum(r.scales.shape[0] for r in case.runs)
                print(line)
                if perf and st.gap == 0 and st.bp == 0 and is_real:
                    for c, r, g in perf[1]:
                        if r * g / c > best.get(n, (0,))[0]:
                            best[n] = (r * g / c, c, r, g, case.name)
    for n, (gpc, c, r, g, name) in sorted(best.items()):
        print(f"GEMV_STREAM_RATE n={n} {gpc:.3f} groups/cycle = {gpc * 128:.1f} weights/cycle "
              f"(MEASURED in sim, no stalls: {r} rows x {g} groups in {c} cycles incl. pipeline drain, {name}; "
              f"ideal {n * 8 / 9:.3f})")
    print(f"GEMV_STREAM_VERDICT {'PASS' if ok_all else 'FAIL'} ({n_lines} sims, {n_rows} output rows, "
          f"NPORTS {sweep}, tree regmask {REGMASK:#04x}, {time.time() - t_start:.0f} s)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
