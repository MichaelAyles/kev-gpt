"""Sim gate for rope_apply_vec.sv: gen ROM+vectors -> iverilog -> vvp ->
bit-exact check against pack_rope.py's own integer reference.

    python -m fabric.asr_seq.run_rope

Mirrors this project's own gate-harness convention (fabric/stage3/run_*.py):
one sentinel line, ROPE_APPLY_VEC_VERDICT.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from fabric.stage3._simdir import kevbuild

from . import pack_rope

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "rope_apply_vec.sv")
TB = os.path.join(HERE, "tb", "tb_rope.sv")


def run(sim_dir: str, seed: int) -> bool:
    os.makedirs(sim_dir, exist_ok=True)
    pack_rope.write_case(sim_dir, seed)

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp, RTL, TB],
                         cwd=sim_dir, capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout); print(cp.stderr)
        return False

    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    sys.stdout.write(rp.stdout)
    if rp.returncode != 0 or "TB_DONE" not in rp.stdout:
        print("VVP_RUN_FAIL"); print(rp.stderr)
        return False

    return pack_rope.check(sim_dir)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.run_rope")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dir", default=kevbuild("asr_seq_rope"))
    a = p.parse_args(argv)
    ok = run(a.dir, a.seed)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
