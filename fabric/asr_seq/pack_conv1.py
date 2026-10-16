"""Golden reference + RTL test vectors for conv1d_seq.sv, parameterized as
conv1 (gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md Stage 1 table: `conv1d(conv1)
[1][16000] -> [288][249], k=127 s=64` -- real shape; this gate uses the same
L=3000 audio convention as every other ASR gate here, giving T=45 output
positions instead of 249, real weights either way).

conv1's real input has 1 channel (raw audio) -- does not divide P=8, so it
is padded to CIN=8 with the weight's extra 7 lanes zeroed (conv1d_seq.sv's
own header explains why this trade is taken instead of a CIN=1 special
case). No bias (moonshine's own `nn.Conv1d(1, embed_dim, kernel_size=127,
stride=64, bias=False)`).

    .venv/bin/python -m fabric.asr_seq.pack_conv1 gen --dir <sim_dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GEN2ASR_SW = os.path.expanduser("~/gen2asr/sw_model")
sys.path.insert(0, GEN2ASR_SW)
sys.path.insert(0, os.path.join(GEN2ASR_SW, "model"))

from fabric.asr_seq import conv1d_ref as cr  # noqa: E402
from fabric.stage3.pack_banked_resident_vec import build_resident  # noqa: E402
from fabric.stage3.seq_ref import quantize_scale_24  # noqa: E402

P = 8
COUT = 288
KW = 127
STRIDE = 64
LANES, WBW = 128, 8
AUDIO_SEED = 0
AUDIO_LEN = 3000


def load_real():
    from transformers import MoonshineForConditionalGeneration
    model = MoonshineForConditionalGeneration.from_pretrained(
        "UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    conv1 = model.model.encoder.conv1

    torch.manual_seed(AUDIO_SEED)
    audio = torch.randn(1, AUDIO_LEN, dtype=torch.float32)
    with torch.no_grad():
        conv1_out_real = conv1(audio.unsqueeze(1))[0]      # (COUT, TOUT), real float

    w_real = conv1.weight.detach().numpy().astype(np.float64)   # (COUT,1,KW)
    x_real = audio[0].detach().numpy().astype(np.float64)[None, :]  # (1, TIN)
    return x_real, w_real, conv1_out_real.detach().numpy().astype(np.float64)


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    x_real, w_real, conv1_out_real = load_real()   # x_real (1,TIN), w_real (COUT,1,KW)
    tin = x_real.shape[1]
    print(f"conv1: CIN_real=1 COUT={COUT} KW={KW} STRIDE={STRIDE} TIN={tin} "
          f"max|x|={np.max(np.abs(x_real)):.4f} max|w|={np.max(np.abs(w_real)):.4f}",
          file=sys.stderr)

    x_pad = cr.pad_cin_input(x_real, P)             # (CIN=8, TIN)
    w_pad = cr.pad_cin_weight(w_real, P)             # (COUT,8,KW)
    cin = x_pad.shape[0]

    ashift = cr.choose_ashift(x_real)
    x_int8 = cr.quantize_act(x_pad, ashift)          # (CIN,TIN) int8 (padded rows all 0)

    w_flat = cr.transpose_weight(w_pad)              # (COUT, KW*CIN)
    w_int8, wshift = cr.quantize_weight_per_row(w_flat)   # (COUT,) per-output-channel shift

    # per-row dequant scale: TRUE_real[row] = gemvy[row] * 2^(-(ashift+wshift[row])) ,
    # want dq_val[row] = round(TRUE_real[row] * 2^QX) = gemvy[row] * 2^(QX-ashift-wshift[row])
    scale = np.exp2((cr.QX - ashift - wshift).astype(np.float64))   # (COUT,)
    mant, expo = quantize_scale_24(scale)
    mant = np.asarray(mant, dtype=np.int64)
    expo = np.asarray(expo, dtype=np.int64)
    print(f"ashift={ashift} wshift range=[{wshift.min()},{wshift.max()}]", file=sys.stderr)

    gold_tc = cr.conv1d_int_ref(x_int8, w_int8, KW, STRIDE, COUT, cin,
                                 mant, expo, bias_q=None)   # (TOUT,COUT) Q6.25
    tout = gold_tc.shape[0]

    # informational-only float check against PyTorch's own real conv1 output
    y_real_from_int = gold_tc.astype(np.float64) / (1 << cr.QX)          # (TOUT,COUT)
    y_real_ref = conv1_out_real.T                                          # (TOUT,COUT)
    cos = float(np.dot(y_real_from_int.reshape(-1), y_real_ref.reshape(-1)) /
                (np.linalg.norm(y_real_from_int) * np.linalg.norm(y_real_ref) + 1e-30))
    print(f"cosine(quantized_int_output, real_float_conv1_output)={cos:.6f} (informational only)",
          file=sys.stderr)

    # ---- RTL vectors ----------------------------------------------------
    all_words, meta = build_resident([w_int8], LANES, WBW)
    n_words = meta[0]["n_words"]
    wbits = LANES * WBW
    hexw = (wbits + 3) // 4
    with open(os.path.join(out_dir, "w.mem"), "w") as f:
        f.write("\n".join(format(w, f"0{hexw}x") for w in all_words) + "\n")

    xt_flat = x_int8.T.reshape(-1)              # (TIN,CIN) -> flat, row r = t*CGRP+cg (CGRP=1 here)
    xt_rows = cr.pack_rows_p8(xt_flat, P)
    with open(os.path.join(out_dir, "xt.mem"), "w") as f:
        nib = (P * 8) // 4
        f.write("\n".join(f"{v & ((1 << (P*8)) - 1):0{nib}x}" for v in xt_rows) + "\n")

    mant_rows = cr.pack_rows_pw(mant, P, 24)
    exp_rows = cr.pack_rows_pw(expo, P, 8)
    with open(os.path.join(out_dir, "dq_mant.mem"), "w") as f:
        f.write("\n".join(f"{v & ((1 << (P*24)) - 1):0{(P*24)//4}x}" for v in mant_rows) + "\n")
    with open(os.path.join(out_dir, "dq_exp.mem"), "w") as f:
        f.write("\n".join(f"{v & ((1 << (P*8)) - 1):0{(P*8)//4}x}" for v in exp_rows) + "\n")

    gold_flat = gold_tc.reshape(-1)              # (TOUT*COUT,), row-major t*COUT+c
    gold_rows = cr.pack_rows_p32(gold_flat, P)
    gold_masked = [v & 0xFFFFFFFFFFFFFFFF for v in gold_rows]  # comparison mask, see run_conv1.py

    with open(os.path.join(out_dir, "gold.txt"), "w") as f:
        f.write("\n".join(f"{v & ((1 << (P*32)) - 1):0{(P*32)//4}x}" for v in gold_rows) + "\n")

    return {"P": P, "CIN": cin, "COUT": COUT, "KW": KW, "STRIDE": STRIDE, "TIN": tin,
            "TOUT": tout, "N_WORDS": n_words, "WWORDS": max(n_words, 1),
            "gold_rows": gold_masked}


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_conv1")
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gen")
    g.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    if a.cmd == "gen":
        m = main_gen(a.dir)
        print(f"CIN={m['CIN']} COUT={m['COUT']} TOUT={m['TOUT']} n_words={m['N_WORDS']}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
