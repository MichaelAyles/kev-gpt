"""gelu_wide_vec RTL gate: bit-true vs a piecewise fixed-point reference,
P lanes wide, 32-bit/lane -- the fix for conv_front_end_seq.sv's own
Q4.12-domain GELU clipping limitation (see that file's header and
gelu_wide_vec.sv's own header for the real-audio magnitude finding this
closes: conv3's raw pre-GELU output reaching |.|~1004 real units, vastly
outside gelu_lut2.sv's own fixed +-8 domain).

    python -m fabric.asr_seq.run_gelu_wide

Reference (gelu_wide_q412): gelu_q (fabric.stage3.run_gelu, unchanged) on
the sat16-clipped value for in-domain/very-negative inputs -- ALREADY
correct there (GELU saturates to ~0 fast on the negative side); the WIDE
input value itself, unclipped, for x > +32767 (real > +8) -- GELU(x)->x
just as fast on the positive side, which gelu_lut2.sv's own domain can't
represent. Same 8192-entry LUT/format as every other Q4.12 gate here.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

from fabric.stage3._simdir import kevbuild
from fabric.stage3.run_gelu import gelu_table, gelu_q

HERE = os.path.dirname(os.path.abspath(__file__))


def gelu_wide_q412(x_wide, lut):
    """Matches gelu_wide_vec.sv exactly: LUT on the sat16-clipped value
    (correct in-domain and for very-negative x), overridden by a direct
    passthrough of the WIDE value itself when x > +32767."""
    x_wide = np.asarray(x_wide, dtype=np.int64)
    x_clip = np.clip(x_wide, -32768, 32767)
    lut_out = gelu_q(x_clip, lut)
    return np.where(x_wide > 32767, x_wide, lut_out)


def run(sim_dir, N=2048, P=8, seed=0):
    os.makedirs(sim_dir, exist_ok=True)
    lut = gelu_table()
    rng = np.random.default_rng(seed)

    # Mix: in-domain (full int16 range), wide positive passthrough (up to
    # conv3's own observed real scale, |.|~1004 real units => ~4.1M in
    # Q4.12), and wide negative (should still land near 0 via the LUT path,
    # not passthrough -- regression coverage for "no change on that side").
    n_indom = N // 2
    n_wide_pos = N // 4
    n_wide_neg = N - n_indom - n_wide_pos
    x_indom = rng.integers(-32768, 32768, size=n_indom).astype(np.int64)
    x_wide_pos = rng.integers(32768, 6_000_000, size=n_wide_pos).astype(np.int64)
    x_wide_neg = rng.integers(-6_000_000, -32768, size=n_wide_neg).astype(np.int64)
    x = np.concatenate([x_indom, x_wide_pos, x_wide_neg])
    rng.shuffle(x)
    edges = np.array([-32768, -22528, 32767, 32768, 40000, 4_096_000, -40000, -4_096_000],
                      dtype=np.int64)
    x[:len(edges)] = edges

    y = gelu_wide_q412(x, lut)

    with open(os.path.join(sim_dir, "gelu_lut_e.mem"), "w") as fh:
        fh.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in lut[0::2]) + "\n")
    with open(os.path.join(sim_dir, "gelu_lut_o.mem"), "w") as fh:
        fh.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in lut[1::2]) + "\n")
    with open(os.path.join(sim_dir, "xin.mem"), "w") as fh:
        for row in x.reshape(-1, P):
            word = 0
            for k in range(P):
                word |= (int(row[k]) & 0xFFFFFFFF) << (32 * k)
            fh.write(f"{word:0{8 * P}x}\n")

    n_rows = len(x) // P
    gold = y.reshape(-1, P)

    vvp = os.path.join(sim_dir, "sim.vvp")
    stage3_rtl = os.path.join(os.path.dirname(HERE), "stage3", "rtl")
    cp = subprocess.run(
        ["iverilog", "-g2012", "-o", vvp, f"-DN={n_rows}", f"-DP={P}",
         os.path.join(HERE, "tb", "tb_gelu_wide_vec.sv"),
         os.path.join(HERE, "rtl", "gelu_wide_vec.sv"),
         os.path.join(stage3_rtl, "vec_gelu.sv"),
         os.path.join(stage3_rtl, "gelu_lut2.sv")],
        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout, cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout, rp.stderr); return False

    with open(os.path.join(sim_dir, "y.out")) as fh:
        rows = [line.strip() for line in fh if line.strip()]

    total = n_rows * P
    mism = 0
    if len(rows) != n_rows:
        mism = total
    else:
        for r in range(n_rows):
            word = int(rows[r], 16)
            for k in range(P):
                got = (word >> (32 * k)) & 0xFFFFFFFF
                gv = int(gold[r, k])
                gold_bits = gv & 0xFFFFFFFF
                if got != gold_bits:
                    mism += 1

    ok = (mism == 0) and (len(rows) == n_rows)
    print(f"GELU_WIDE_VERDICT bitexact={ok} mismatches={mism}/{total} P={P}")
    return ok


def main(argv=None):
    sim_dir = kevbuild("asr_gelu_wide")
    raise SystemExit(0 if run(sim_dir) else 1)


if __name__ == "__main__":
    main()
