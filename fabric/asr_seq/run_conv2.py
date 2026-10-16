"""Sim gate for conv1d_seq.sv, parameterized as conv2 (+bias, real CIN=288
already P-aligned): real moonshine-tiny weights, real audio-derived
conv1+tanh+groupnorm1 input, RTL checked bit-exact against pack_conv2.py's
own integer reference.

    python -m fabric.asr_seq.run_conv2

One sentinel line: CONV2_VERDICT.
"""
from __future__ import annotations

import os
import subprocess
import sys

from fabric.stage3._simdir import kevbuild

from . import pack_conv2

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")
STAGE3_RTL = os.path.join(os.path.dirname(HERE), "stage3", "rtl")
TB = os.path.join(HERE, "tb", "tb_conv1d_seq.sv")


def run(sim_dir: str) -> bool:
    os.makedirs(sim_dir, exist_ok=True)
    manifest = pack_conv2.main_gen(sim_dir)
    gold = manifest["gold_rows"]

    vvp = os.path.join(sim_dir, "sim.vvp")
    sources = [
        os.path.join(STAGE3_RTL, "gemv_banked_resident_vec.sv"),
        os.path.join(STAGE3_RTL, "weight_bank_tdp.sv"),
        os.path.join(STAGE3_RTL, "vec_dequant.sv"),
        os.path.join(RTL_DIR, "conv1d_seq.sv"),
        TB,
    ]
    defs = [f"-DPVAL={manifest['P']}", f"-DCINVAL={manifest['CIN']}",
            f"-DCOUTVAL={manifest['COUT']}", f"-DKWVAL={manifest['KW']}",
            f"-DSTRIDEVAL={manifest['STRIDE']}", f"-DTINVAL={manifest['TIN']}",
            "-DHASBIAS=1", f"-DNWORDS={manifest['N_WORDS']}",
            f"-DWWORDSVAL={manifest['WWORDS']}"]
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp] + defs + sources,
                         cwd=sim_dir, capture_output=True, text=True)
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
    print(f"CONV2_VERDICT bitexact={ok} mismatches={mism}/{len(gold)} "
          f"TOUT={manifest['TOUT']} COUT={manifest['COUT']} got={len(got)}")
    return ok


def main(argv=None):
    ok = run(kevbuild("asr_conv2"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
