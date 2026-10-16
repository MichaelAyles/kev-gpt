"""vec_silu RTL gate: bit-true vs the fixed-point reference, P lanes wide.

    python -m fabric.asr_seq.run_vec_silu

vec_silu (rtl/vec_silu.sv) instantiates P independent copies of the
single-lane silu_lut, so each lane must reproduce silu_q
(fabric/asr_seq/run_silu.silu_q) EXACTLY. Same gate shape as checkpoint C's
run_vec_gelu.py. Format: I/O signed Q4.12 (16-bit, +-8).
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

from fabric.stage3._simdir import kevbuild

from fabric.asr_seq.run_silu import silu_q, silu_table

HERE = os.path.dirname(os.path.abspath(__file__))


def run(sim_dir, N=1024, P=8, seed=0):
    os.makedirs(sim_dir, exist_ok=True)
    lut = silu_table()
    rng = np.random.default_rng(seed)

    x = rng.integers(-32768, 32768, size=(N, P)).astype(np.int64)
    edges = np.array([-32768, -22528, -4096, 0, 1, 4096, 22528, 32767], dtype=np.int64)
    x[0, :min(P, edges.size)] = edges[:P]

    y = silu_q(x, lut)

    with open(os.path.join(sim_dir, "silu_lut.mem"), "w") as fh:
        fh.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in lut) + "\n")
    with open(os.path.join(sim_dir, "xin.mem"), "w") as fh:
        for row in x:
            word = 0
            for k in range(P):
                word |= (int(row[k]) & 0xFFFF) << (16 * k)
            fh.write(f"{word:0{4 * P}x}\n")

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(
        ["iverilog", "-g2012", "-o", vvp, f"-DN={N}", f"-DP={P}",
         os.path.join(HERE, "tb", "tb_vec_silu.sv"),
         os.path.join(HERE, "rtl", "vec_silu.sv"),
         os.path.join(HERE, "rtl", "silu_lut.sv")],
        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout, cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout, rp.stderr); return False

    with open(os.path.join(sim_dir, "y.out")) as fh:
        rows = [line.strip() for line in fh if line.strip()]

    total = N * P
    mism = 0
    if len(rows) != N:
        mism = total
    else:
        for r in range(N):
            word = int(rows[r], 16)
            for k in range(P):
                got = (word >> (16 * k)) & 0xFFFF
                gold = int(y[r, k]) & 0xFFFF
                if got != gold:
                    mism += 1

    ok = (mism == 0) and (len(rows) == N)
    print(f"VEC_SILU_VERDICT bitexact={ok} mismatches={mism}/{total} P={P}")
    return ok


def main(argv=None):
    sim_dir = kevbuild("asr_vec_silu")
    raise SystemExit(0 if run(sim_dir) else 1)


if __name__ == "__main__":
    main()
