"""SiLU-LUT RTL gate: bit-true vs the fixed-point reference.

    python -m fabric.asr_seq.run_silu

Format: I/O signed Q4.12 (16-bit, +-8), identical shape to checkpoint C's
gelu_lut.sv (fabric/stage3/run_gelu.py) -- 8192-entry LUT, index =
(x+0x8000)>>3, low 3 bits are the linear-interp fraction. silu_q() is the
integer reference rtl/silu_lut.sv (fabric/asr_seq/rtl/) must reproduce
exactly. This is the decoder MLP's real gate activation (test_generate_kv.c's
SwiGLU stage: h1 = fc1(xn3); value, gate = split(h1); gate = silu_(gate);
combined = value * gate) -- genuinely new, nothing in kevgpt_seq implements
SiLU (gelu_lut.sv/gelu_lut2.sv are GELU-shaped).

No end-to-end cosine check here (unlike run_gelu.py): there is no existing
full fixed-point ASR-decoder reference to plug this into yet -- that's the
top-level FSM's own eventual job. The module-level bit-exact LUT gate is the
binding check, same rigor this project applies to every other LUT.
"""
from __future__ import annotations

import os
import subprocess
import sys

from fabric.stage3._simdir import kevbuild

import numpy as np

FRAC = 12
SCALE = 1 << FRAC                     # 4096
N_LUT = 8192
HERE = os.path.dirname(os.path.abspath(__file__))


def silu_float(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def silu_table():
    """8192 Q4.12 LUT entries; entry i ~= silu(i/512 - 8)."""
    xs = np.arange(N_LUT) / 512.0 - 8.0
    return np.clip(np.round(silu_float(xs) * SCALE), -32768, 32767).astype(np.int64)


def silu_q(x_int, lut):
    """Integer SiLU on Q4.12 inputs -- the exact arithmetic silu_lut.sv does."""
    x_int = np.asarray(x_int, dtype=np.int64)
    u = x_int + 32768                                   # 0..65535
    i = u >> 3
    f = u & 7
    i1 = np.minimum(i + 1, N_LUT - 1)
    step = (lut[i1] - lut[i]) * f
    return lut[i] + (step >> 3)                         # arithmetic shift (matches >>>)


def run(sim_dir, N=4096, seed=0):
    os.makedirs(sim_dir, exist_ok=True)
    lut = silu_table()
    rng = np.random.default_rng(seed)
    x = rng.integers(-32768, 32768, size=N).astype(np.int64)          # full Q4.12 range
    x[:8] = np.array([-32768, -22528, -4096, 0, 1, 4096, 22528, 32767])  # edges
    y = silu_q(x, lut)

    with open(os.path.join(sim_dir, "silu_lut.mem"), "w") as fh:
        fh.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in lut) + "\n")
    with open(os.path.join(sim_dir, "x.mem"), "w") as fh:
        fh.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in x) + "\n")

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp, f"-DN={N}",
                         os.path.join(HERE, "tb", "tb_silu.sv"),
                         os.path.join(HERE, "rtl", "silu_lut.sv")],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout, cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout, rp.stderr); return False

    def _hex(line):
        try:
            return int(line, 16) & 0xFFFF
        except ValueError:
            return -1
    with open(os.path.join(sim_dir, "y.out")) as fh:
        yout = [_hex(line) for line in fh if line.strip()]
    gold = [int(v) & 0xFFFF for v in y]

    best_L, best_mis = 0, N + 1
    for L in range(6):
        seg = yout[L:L + N]
        if len(seg) < N:
            continue
        mis = sum(1 for a, b in zip(gold, seg) if a != b)
        if mis < best_mis:
            best_L, best_mis = L, mis
    ok = best_mis == 0
    print(f"SILU_VERDICT bitexact={ok} mismatches={best_mis}/{N} latency={best_L}")
    return ok


def main(argv=None):
    ok = run(kevbuild("asr_silu"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
