"""Sim gate for groupnorm1_vec.sv -- Stage 1 conv front-end's groupnorm1:
real conv1+tanh output (real audio, real weights), real groupnorm gamma/
beta, RTL checked bit-exact against pack_groupnorm1.py's own integer
reference (gn_int) across every one of the C*T real output elements.

    python -m fabric.asr_seq.run_groupnorm1

One sentinel line: GROUPNORM1_VERDICT.
"""
from __future__ import annotations

import os
import subprocess
import sys

from fabric.stage3._simdir import kevbuild

from . import pack_groupnorm1

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")
TB = os.path.join(HERE, "tb", "tb_groupnorm1_vec.sv")


def run(sim_dir: str) -> bool:
    os.makedirs(sim_dir, exist_ok=True)
    manifest = pack_groupnorm1.main_gen(sim_dir)
    C, T, P = manifest["C"], manifest["T"], manifest["P"]
    gold = manifest["gold"]

    vvp = os.path.join(sim_dir, "sim.vvp")
    sources = [
        os.path.join(RTL_DIR, "groupnorm1_vec.sv"),
        TB,
    ]
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                          f"-DPVAL={P}", f"-DCVAL={C}", f"-DTVAL={T}"]
                         + sources, cwd=sim_dir, capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout); print(cp.stderr)
        return False

    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True,
                         timeout=1800)
    sys.stdout.write(rp.stdout)
    if rp.returncode != 0 or "TB_DONE" not in rp.stdout:
        print("VVP_RUN_FAIL"); print(rp.stderr)
        return False

    def _hx(s):
        return None if ("x" in s or "z" in s) else int(s, 16) & 0xFFFFFFFFFFFFFFFF

    with open(os.path.join(sim_dir, "y.out")) as fh:
        got = [_hx(ln.strip()) for ln in fh if ln.strip()]

    n = min(len(got), len(gold))
    mism = sum(1 for i in range(n) if got[i] != gold[i]) + abs(len(got) - len(gold))
    ok = (mism == 0) and (len(got) == len(gold))
    print(f"GROUPNORM1_VERDICT bitexact={ok} mismatches={mism}/{len(gold)} "
          f"C={C} T={T} N={manifest['N']} got={len(got)}")
    return ok


def main(argv=None):
    ok = run(kevbuild("asr_groupnorm1"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
