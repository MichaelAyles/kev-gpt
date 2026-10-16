"""layernorm_vec_gendiv RTL gate: bit-true vs a floor-divide-generalized
integer LayerNorm reference, at ASR's real D=288 -- the non-power-of-2 case
fabric/stage3/run_layernorm._ln_int_quantized explicitly refuses (its own
assertion: "d is not a power of 2; the shift-based mean/var divide ... is
only exact for one"), and the case the "Two attention shapes" doc's Stage 2
table originally, wrongly, called a trivial parametric reuse.

Reuses checkpoint C's own proven fixed-point pieces directly, not
re-derived: fabric.stage3.run_layernorm's format constants (QX/G_FRAC/
A_FRAC/Y_FRAC/OUT_FRAC/VAR_FRAC/EPS_A), rsqrt_int (the seed+2-Newton rsqrt,
D-independent), and seed_table() (the 64-entry ROM). Only the mean/var
divide step is new: an exact floor-divide by D (round toward -infinity,
matching Verilog's `>>>` bit-for-bit for a power-of-2 D too -- this
reference and the RTL are both regression-checked against the ORIGINAL
D=256 case as well as the new D=288 one).

    python -m fabric.asr_seq.run_layernorm_gendiv            # D=288, P=8, 64 cases
    python -m fabric.asr_seq.run_layernorm_gendiv --d 256     # regression: matches
                                                                 fabric.stage3.run_layernorm exactly
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np

from fabric.stage3._simdir import kevbuild
from fabric.stage3.run_layernorm import (
    QX, G_FRAC, A_FRAC, Y_FRAC, OUT_FRAC, VAR_FRAC, EPS_A,
    rsqrt_int, seed_table, _ln_int_quantized,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_STAGE3 = os.path.join(os.path.dirname(HERE), "stage3", "rtl")


def _floordiv(num: int, d: int) -> int:
    """floor(num/d), d>0 -- matches Verilog `>>> $clog2(d)` bit-for-bit when d
    IS a power of 2 (Python's own `//` already floors, so this is just for
    clarity/parity with the RTL's floordiv40/floordiv72 functions)."""
    return num // d


def ln_int_gendiv(X, G):
    """Same integer LayerNorm as _ln_int_quantized, but the mean/var divide is
    a real floor-divide by d=len(X) instead of a power-of-2-only shift --
    correct for ANY d, including ASR's D=288."""
    X = [int(v) for v in X]
    G = [int(v) for v in G]
    d_model = len(X)
    S = sum(X)
    mean = _floordiv(S, d_model)
    d = [v - mean for v in X]
    SS = sum(di * di for di in d)
    var = _floordiv(SS, d_model)
    A = (var >> (VAR_FRAC - A_FRAC)) + EPS_A
    if A <= 0:
        A = 1
    Yr = rsqrt_int(A)
    sh = (QX + Y_FRAC + G_FRAC) - OUT_FRAC
    out = []
    for i in range(d_model):
        t = d[i] * Yr
        t = t * G[i]
        yv = t >> sh if sh >= 0 else t << (-sh)
        out.append(yv)
    return out, mean, var, A, Yr


def _rand_cases(n, d_model, seed=0):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        x = rng.uniform(-4.0, 4.0, size=d_model)
        g = rng.uniform(-2.0, 2.0, size=d_model)
        X = np.round(x * (1 << QX)).astype(object)
        G = np.round(g * (1 << G_FRAC)).astype(object)
        cases.append((X, G))
    return cases


def _pack_rows(vec, P):
    rows = []
    for r in range(len(vec) // P):
        val = 0
        for k in range(P):
            val |= (int(vec[r * P + k]) & 0xFFFFFFFF) << (32 * k)
        rows.append(val)
    return rows


def _w(path, vals, nib):
    mask = (1 << (4 * nib)) - 1
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v) & mask:0{nib}x}" for v in vals) + "\n")


def run(sim_dir: str, d_model: int = 288, n_cases: int = 64, seed: int = 0, P: int = 8) -> bool:
    os.makedirs(sim_dir, exist_ok=True)
    assert d_model % P == 0
    cases = _rand_cases(n_cases, d_model, seed)

    xrows, grows, gold = [], [], []
    for X, G in cases:
        xrows += _pack_rows(X, P)
        grows += _pack_rows(G, P)
        out, *_ = ln_int_gendiv(list(X), list(G))
        gold += [int(o) & 0xFFFFFFFFFFFFFFFF for o in out]

    # regression cross-check: for a power-of-2 d_model, this generalized
    # reference must match the ORIGINAL power-of-2-only reference exactly
    if d_model & (d_model - 1) == 0:
        mism_ref = 0
        for X, G in cases:
            out_gen, *_ = ln_int_gendiv(list(X), list(G))
            out_orig, *_ = _ln_int_quantized(list(X), list(G))
            mism_ref += sum(1 for a, b in zip(out_gen, out_orig) if a != b)
        print(f"REGRESSION_VS_ORIGINAL_REF mismatches={mism_ref}")
        assert mism_ref == 0, "gendiv reference disagrees with the original power-of-2 reference"

    nib_row = (P * 32) // 4
    _w(os.path.join(sim_dir, "x.mem"), xrows, nib_row)
    _w(os.path.join(sim_dir, "g.mem"), grows, nib_row)
    _w(os.path.join(sim_dir, "seed.mem"), seed_table(), 5)

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                         f"-DNCASE={n_cases}", f"-DPVAL={P}", f"-DDVAL={d_model}",
                         os.path.join(HERE, "tb", "tb_layernorm_gendiv.sv"),
                         os.path.join(HERE, "rtl", "layernorm_vec_gendiv.sv")],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout); print(cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout[-2000:]); print(rp.stderr[-1000:]); return False

    with open(os.path.join(sim_dir, "y.out")) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    def _hx(s):
        return None if ("x" in s or "z" in s) else int(s, 16) & 0xFFFFFFFFFFFFFFFF

    got = [_hx(s) for s in lines]
    n = min(len(got), len(gold))
    mism = sum(1 for i in range(n) if got[i] != gold[i]) + abs(len(got) - len(gold))
    ok = (mism == 0) and (len(got) == len(gold))
    print(f"LN_GENDIV_VERDICT bitexact={ok} mismatches={mism}/{len(gold)} "
          f"d_model={d_model} P={P} ncase={n_cases} got={len(got)}")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.run_layernorm_gendiv")
    p.add_argument("--d", type=int, default=288)
    p.add_argument("--p", type=int, default=8)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    ok = run(kevbuild("asr_ln_gendiv"), d_model=a.d, n_cases=a.n, seed=a.seed, P=a.p)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
