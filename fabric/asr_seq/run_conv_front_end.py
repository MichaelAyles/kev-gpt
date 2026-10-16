"""Sim gate for conv_front_end_seq.sv -- the Stage 1 conv front-end's own
top-level FSM, end to end: real moonshine-tiny weights, real audio
(torch.manual_seed(0), L=3000), RTL checked bit-exact against
pack_conv_front_end.py's own integer reference across every real output
position/channel of the whole conv1->tanh->groupnorm1->conv2->gelu->conv3->
gelu chain.

    python -m fabric.asr_seq.run_conv_front_end

One sentinel line: CONV_FRONT_END_VERDICT.
"""
from __future__ import annotations

import os
import subprocess
import sys

from fabric.stage3._simdir import kevbuild

from . import pack_conv_front_end

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")
STAGE3_RTL = os.path.join(os.path.dirname(HERE), "stage3", "rtl")
TB = os.path.join(HERE, "tb", "tb_conv_front_end_seq.sv")


def run(sim_dir: str) -> bool:
    os.makedirs(sim_dir, exist_ok=True)
    manifest = pack_conv_front_end.main_gen(sim_dir)
    gold = manifest["gold_rows"]

    vvp = os.path.join(sim_dir, "sim.vvp")
    sources = [
        os.path.join(STAGE3_RTL, "gemv_banked_resident_vec.sv"),
        os.path.join(STAGE3_RTL, "weight_bank_tdp.sv"),
        os.path.join(STAGE3_RTL, "vec_gelu.sv"),
        os.path.join(STAGE3_RTL, "gelu_lut2.sv"),
        os.path.join(STAGE3_RTL, "vec_dequant.sv"),
        os.path.join(RTL_DIR, "gelu_wide_vec.sv"),
        os.path.join(RTL_DIR, "conv1d_seq.sv"),
        os.path.join(RTL_DIR, "tanh_lut.sv"),
        os.path.join(RTL_DIR, "vec_tanh.sv"),
        os.path.join(RTL_DIR, "groupnorm1_vec.sv"),
        os.path.join(RTL_DIR, "conv_front_end_seq.sv"),
        TB,
    ]
    defs = [f"-DTIN1VAL={manifest['TIN1']}", f"-DNWORDS1={manifest['NWORDS1']}",
            f"-DNWORDS2={manifest['NWORDS2']}", f"-DNWORDS3={manifest['NWORDS3']}"]
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp] + defs + sources,
                         cwd=sim_dir, capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout); print(cp.stderr)
        return False

    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True,
                         timeout=3000)
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
    print(f"CONV_FRONT_END_VERDICT bitexact={ok} mismatches={mism}/{len(gold)} "
          f"TOUT3={manifest['TOUT3']} COUT3={manifest['COUT3']} got={len(got)}")
    return ok


def main(argv=None):
    ok = run(kevbuild("asr_conv_front_end"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
