"""Sim gate for encoder_block_seq.sv -- the MULTI-LAYER functional gate for
the real, sized encoder-block top-level FSM: real moonshine-tiny weights
(all 6 real encoder layers), real per-position conv-front-end input (all
T2 real positions), one full encoder-layer forward pass per layer
(chained, layer L's own residual output feeding layer L+1's own input),
each checked bit-exact against pack_encoder_block.py's own per-(layer,
position) golden xres_out.

    python -m fabric.asr_seq.run_encoder_block

One sentinel line: ENCODER_BLOCK_SEQ_VERDICT.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from fabric.stage3._simdir import kevbuild

from . import pack_encoder_block

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")
STAGE3_RTL = os.path.join(os.path.dirname(HERE), "stage3", "rtl")
TB = os.path.join(HERE, "tb", "tb_encoder_block_seq.sv")


def run(sim_dir: str) -> bool:
    os.makedirs(sim_dir, exist_ok=True)
    manifest = pack_encoder_block.main_gen(sim_dir)
    n_words = manifest["n_words_total"]
    n_layer = manifest["n_layer"]

    vvp = os.path.join(sim_dir, "sim.vvp")
    sources = [
        os.path.join(RTL_DIR, "rope_apply_vec.sv"),
        os.path.join(STAGE3_RTL, "gelu_lut2.sv"),
        os.path.join(STAGE3_RTL, "vec_gelu.sv"),
        os.path.join(RTL_DIR, "layernorm_vec_gendiv.sv"),
        os.path.join(STAGE3_RTL, "kv_bank.sv"),
        os.path.join(STAGE3_RTL, "vec_attn_w.sv"),
        os.path.join(STAGE3_RTL, "softmax_f.sv"),
        os.path.join(STAGE3_RTL, "gemv_banked_resident_vec.sv"),
        os.path.join(STAGE3_RTL, "weight_bank_tdp.sv"),
        os.path.join(RTL_DIR, "encoder_block_seq.sv"),
        TB,
    ]
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                          f"-DNWORDS={n_words}", f"-DNLAYER={n_layer}"]
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

    return "ENCODER_BLOCK_SEQ_VERDICT,bitexact=1" in rp.stdout


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.run_encoder_block")
    p.add_argument("--dir", default=kevbuild("asr_encoder_block"))
    a = p.parse_args(argv)
    ok = run(a.dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
