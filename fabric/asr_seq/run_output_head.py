"""Sim gate for output_head_seq.sv -- Stage 4 of the ASR accelerator: real
final decoder LayerNorm, real tied lm_head weight (INT8, per-row
quantized), real per-row (mant,exp) dequant table, real per-step decoder
hidden state, argmax checked bit-exact against pack_output_head.py's own
golden, once per real decode step.

    python -m fabric.asr_seq.run_output_head

One sentinel line: OUTPUT_HEAD_SEQ_VERDICT.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from fabric.stage3._simdir import kevbuild

from . import pack_output_head

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")
STAGE3_RTL = os.path.join(os.path.dirname(HERE), "stage3", "rtl")
TB = os.path.join(HERE, "tb", "tb_output_head_seq.sv")


def run(sim_dir: str) -> bool:
    os.makedirs(sim_dir, exist_ok=True)
    manifest = pack_output_head.main_gen(sim_dir)
    n_words = manifest["n_words_total"]
    n_steps = manifest["n_steps"]

    vvp = os.path.join(sim_dir, "sim.vvp")
    sources = [
        os.path.join(RTL_DIR, "layernorm_vec_gendiv.sv"),
        os.path.join(STAGE3_RTL, "gemv_banked_resident_vec.sv"),
        os.path.join(STAGE3_RTL, "weight_bank_tdp.sv"),
        os.path.join(STAGE3_RTL, "vec_dequant.sv"),
        os.path.join(RTL_DIR, "output_head_seq.sv"),
        TB,
    ]
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                          f"-DNWORDS={n_words}", f"-DNSTEPS={n_steps}"]
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

    return "OUTPUT_HEAD_SEQ_VERDICT,bitexact=1" in rp.stdout


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.run_output_head")
    p.add_argument("--dir", default=kevbuild("asr_output_head"))
    a = p.parse_args(argv)
    ok = run(a.dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
