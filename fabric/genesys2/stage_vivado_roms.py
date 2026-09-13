"""Generates the ROM .mem files sequencer_vec.sv/layernorm_vec.sv/kv_bank_ddr.sv/
gelu_lut2.sv/softmax_f.sv `$readmemh` at SYNTHESIS TIME -- baked into the bitstream
itself, completely separate from the UART/DDR3-loaded big weight matrix
(fabric/genesys2/send_weights.py) -- and stages them into a Vivado build directory.

This closes a real gap flagged (but never fixed) back in PORT-NOTES.md's Phase 5:
"this session's inline ROM-file generation should be consolidated into one script
before Phase 6". Skipping this step doesn't fail loudly: Vivado logs a
`CRITICAL WARNING: could not open $readmem data file '...'; ... ignoring` per
missing file and continues, silently zero-initializing the dequant-scale/gamma/
gumbel ROMs -- which zeroes every dequantized value regardless of the real weight
data (a real hardware bug this project hit and root-caused the hard way: it also
LOOKS like a big, scary synthesis-non-determinism DSP-count mystery, since the
optimizer treats the now-effectively-constant-zero logic as dead code and infers
far fewer DSP48E1 instances than a build with the ROMs correctly populated --
803/840 with the fix, ~550/840 without it, same RTL either way).

Needs BOTH: the Vivado project root, AND `<project>.runs/synth_1/` specifically --
`launch_runs synth_1` spawns synthesis as its own `vivado -mode batch` subprocess
with a DIFFERENT CWD than the project root, so `$readmemh`'s bare relative
filenames only resolve there. If `reset_run synth_1` has already run, that
directory gets recreated EMPTY -- always stage AFTER any reset_run, immediately
before launching synthesis, not before.

Reuses fabric.stage3.run_sequencer.write_mems_wideword (produces gamma_w/inv_sact/
dqm_w/dqe_w/seed/exp_lut/gelu_lut/wrom -- the same code path run_vec_kv.py's gate
uses) plus the exact gelu-split/inv-lut/gumbel-lut generation from run_vec_kv.py's
own run(), verbatim -- not re-derived.

    python -m fabric.genesys2.stage_vivado_roms \\
        --npz fabric/export_stories/goformer.npz --lanes 64 --p 8 \\
        --dest /path/to/genesys2_kevgpt-vivado \\
        --dest /path/to/genesys2_kevgpt-vivado/*.runs/synth_1
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile

from fabric.stage3 import seq_ref
from fabric.stage3 import gumbel
from fabric.stage3.run_sequencer import write_mems_wideword
from fabric.stage3.seq_ref import q_round_div

NEEDED = [
    "gamma_w.mem", "inv_sact.mem", "dqm_w.mem", "dqe_w.mem",
    "seed.mem", "exp_lut.mem",
    "gelu_lut_e.mem", "gelu_lut_o.mem",
    "inv_lut_lo.mem", "inv_lut_hi.mem",
    "gumbel_lut.mem",
]


def stage(npz: str, lanes: int, p: int, dests: list[str]) -> None:
    goformer_p, cfg = seq_ref.build(npz)
    nlayer = len(goformer_p["blocks"])
    iseq = seq_ref.IntSequencer(goformer_p, cfg)

    with tempfile.TemporaryDirectory() as td:
        write_mems_wideword(td, iseq, lanes, nlayer, p)

        with open(os.path.join(td, "gelu_lut.mem")) as fh:
            lut = [ln.strip() for ln in fh if ln.strip()]
        with open(os.path.join(td, "gelu_lut_e.mem"), "w") as fh:
            fh.write("\n".join(lut[0::2]) + "\n")
        with open(os.path.join(td, "gelu_lut_o.mem"), "w") as fh:
            fh.write("\n".join(lut[1::2]) + "\n")

        with open(os.path.join(td, "inv_lut_lo.mem"), "w") as fh:
            for s4 in range(4096):
                fh.write(f"{q_round_div(1 << 24, max(s4, 1)) & 0x1FFFFFF:07x}\n")
        with open(os.path.join(td, "inv_lut_hi.mem"), "w") as fh:
            for s4 in range(4096, 16512):
                fh.write(f"{q_round_div(1 << 24, s4) & 0x1FFF:04x}\n")

        with open(os.path.join(td, "gumbel_lut.mem"), "w") as fh:
            for g in gumbel.make_gumbel_lut():
                fh.write(f"{g & 0xFFFFFFFF:08x}\n")

        for dest in dests:
            os.makedirs(dest, exist_ok=True)
            for fname in NEEDED:
                shutil.copy(os.path.join(td, fname), os.path.join(dest, fname))
            print(f"staged {len(NEEDED)} ROM files -> {dest}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.genesys2.stage_vivado_roms")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--lanes", type=int, default=64)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--dest", action="append", required=True,
                     help="a directory to stage ROM files into; repeat for multiple "
                          "(project root AND .runs/synth_1/ both need it)")
    a = ap.parse_args(argv)
    stage(a.npz, a.lanes, a.p, a.dest)


if __name__ == "__main__":
    main()
