"""Sim gate for decoder_block_seq.sv -- the MULTI-STEP functional gate for
the real, sized decoder-block top-level FSM: real moonshine-tiny layer-0
weights, real cross K/V (T2=6), real per-step initial residuals, one full
decoder-layer forward pass per real decode step (step 0..N_STEPS-1,
self-attn KV cache growing causally across steps), each checked bit-exact
against pack_decoder_block.py's own per-step golden xres3.

    python -m fabric.asr_seq.run_decoder_block

One sentinel line: DECODER_BLOCK_SEQ_VERDICT.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from fabric.stage3._simdir import kevbuild

from . import pack_decoder_block

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")
STAGE3_RTL = os.path.join(os.path.dirname(HERE), "stage3", "rtl")
TB = os.path.join(HERE, "tb", "tb_decoder_block_seq.sv")


def run(sim_dir: str) -> bool:
    os.makedirs(sim_dir, exist_ok=True)
    manifest = pack_decoder_block.main_gen(sim_dir)
    n_words = manifest["n_words_total"]
    n_steps = manifest["n_steps"]
    n_layer = manifest["n_layer"]

    vvp = os.path.join(sim_dir, "sim.vvp")
    sources = [
        os.path.join(RTL_DIR, "rope_apply_vec.sv"),
        os.path.join(RTL_DIR, "silu_lut.sv"),
        os.path.join(RTL_DIR, "vec_silu.sv"),
        os.path.join(RTL_DIR, "layernorm_vec_gendiv.sv"),
        os.path.join(STAGE3_RTL, "kv_bank.sv"),
        os.path.join(STAGE3_RTL, "vec_attn_w.sv"),
        os.path.join(STAGE3_RTL, "softmax_f.sv"),
        os.path.join(STAGE3_RTL, "gemv_banked_resident_vec.sv"),
        os.path.join(STAGE3_RTL, "weight_bank_tdp.sv"),
        os.path.join(RTL_DIR, "decoder_block_seq.sv"),
        TB,
    ]
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                          f"-DNWORDS={n_words}", f"-DNSTEPS={n_steps}", f"-DNLAYER={n_layer}"]
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

    return "DECODER_BLOCK_SEQ_VERDICT,bitexact=1" in rp.stdout


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.run_decoder_block")
    p.add_argument("--dir", default=kevbuild("asr_decoder_block"))
    a = p.parse_args(argv)
    ok = run(a.dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
