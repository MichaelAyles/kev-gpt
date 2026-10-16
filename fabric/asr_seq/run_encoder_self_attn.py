"""Sim gate for the "static full-attend" access pattern (ASR's encoder
bidirectional self-attention): rope_apply_vec.sv (new) + kv_bank.sv
(checkpoint C's real RTL, UNMODIFIED, used in write-once/read-repeatedly
mode) + vec_attn_w.sv (checkpoint C's real RTL, unmodified), chained for
real ASR data, checked bit-exact against pack_encoder_self_attn.py's own
golden reference.

    python -m fabric.asr_seq.run_encoder_self_attn

One sentinel line: ENCODER_SELF_ATTN_VERDICT.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from fabric.stage3._simdir import kevbuild

from . import pack_encoder_self_attn

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")
STAGE3_RTL = os.path.join(os.path.dirname(HERE), "stage3", "rtl")
TB = os.path.join(HERE, "tb", "tb_encoder_self_attn.sv")


def run(sim_dir: str) -> bool:
    os.makedirs(sim_dir, exist_ok=True)
    pack_encoder_self_attn.main_gen(sim_dir)

    vvp = os.path.join(sim_dir, "sim.vvp")
    sources = [
        os.path.join(RTL_DIR, "rope_apply_vec.sv"),
        os.path.join(STAGE3_RTL, "kv_bank.sv"),
        os.path.join(STAGE3_RTL, "vec_attn_w.sv"),
        os.path.join(STAGE3_RTL, "softmax_f.sv"),
        TB,
    ]
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp] + sources,
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

    return "ENCODER_SELF_ATTN_VERDICT,bitexact=1" in rp.stdout


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.run_encoder_self_attn")
    p.add_argument("--dir", default=kevbuild("asr_seq_encattn"))
    a = p.parse_args(argv)
    ok = run(a.dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
