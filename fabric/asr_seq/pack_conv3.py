"""Golden reference + RTL test vectors for conv1d_seq.sv, parameterized as
conv3 (gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md Stage 1 table: `conv1d(conv3)
[576][81] -> [288][40], k=3 s=2, +bias`). CIN=576 already divides P=8 -- no
padding needed. Real input is conv2's own real output after real gelu
(moonshine's own `nn.functional.gelu(self.conv3(hidden_states))`, hidden_states
the real post-conv2-gelu tensor) -- reuses the real HF ops directly.

    .venv/bin/python -m fabric.asr_seq.pack_conv3 gen --dir <sim_dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GEN2ASR_SW = os.path.expanduser("~/gen2asr/sw_model")
sys.path.insert(0, GEN2ASR_SW)
sys.path.insert(0, os.path.join(GEN2ASR_SW, "model"))

from fabric.asr_seq import conv1d_ref as cr  # noqa: E402
from fabric.stage3.pack_banked_resident_vec import build_resident  # noqa: E402
from fabric.stage3.seq_ref import quantize_scale_24  # noqa: E402

P = 8
COUT = 288
KW = 3
STRIDE = 2
LANES, WBW = 128, 8
AUDIO_SEED = 0
AUDIO_LEN = 3000


def load_real():
    from transformers import MoonshineForConditionalGeneration
    model = MoonshineForConditionalGeneration.from_pretrained(
        "UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    enc = model.model.encoder

    torch.manual_seed(AUDIO_SEED)
    audio = torch.randn(1, AUDIO_LEN, dtype=torch.float32)
    with torch.no_grad():
        x = torch.tanh(enc.conv1(audio.unsqueeze(1)))
        gn_out = enc.groupnorm(x)
        c2 = F.gelu(enc.conv2(gn_out))                     # (1,576,T2), real float, conv3's real input
        conv3_out_real = enc.conv3(c2)[0]                   # (COUT,TOUT), pre-gelu, real float

    w_real = enc.conv3.weight.detach().numpy().astype(np.float64)   # (COUT,576,KW)
    b_real = enc.conv3.bias.detach().numpy().astype(np.float64)     # (COUT,)
    x_real = c2[0].detach().numpy().astype(np.float64)               # (576, T2)
    return x_real, w_real, b_real, conv3_out_real.detach().numpy().astype(np.float64)


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    x_real, w_real, b_real, conv3_out_real = load_real()
    cin_real, tin = x_real.shape
    print(f"conv3: CIN={cin_real} COUT={COUT} KW={KW} STRIDE={STRIDE} TIN={tin} "
          f"max|x|={np.max(np.abs(x_real)):.4f} max|w|={np.max(np.abs(w_real)):.4f} "
          f"max|b|={np.max(np.abs(b_real)):.4f}", file=sys.stderr)

    x_pad = cr.pad_cin_input(x_real, P)
    w_pad = cr.pad_cin_weight(w_real, P)
    cin = x_pad.shape[0]

    ashift = cr.choose_ashift(x_real)
    x_int8 = cr.quantize_act(x_pad, ashift)

    w_flat = cr.transpose_weight(w_pad)
    w_int8, wshift = cr.quantize_weight_per_row(w_flat)   # (COUT,) per-output-channel shift

    scale = np.exp2((cr.QX - ashift - wshift).astype(np.float64))   # (COUT,)
    mant, expo = quantize_scale_24(scale)
    mant = np.asarray(mant, dtype=np.int64)
    expo = np.asarray(expo, dtype=np.int64)
    bias_q = np.round(b_real * (1 << cr.QX)).astype(np.int64)
    print(f"ashift={ashift} wshift range=[{wshift.min()},{wshift.max()}]", file=sys.stderr)

    gold_tc = cr.conv1d_int_ref(x_int8, w_int8, KW, STRIDE, COUT, cin,
                                 mant, expo, bias_q=bias_q)
    tout = gold_tc.shape[0]

    y_real_from_int = gold_tc.astype(np.float64) / (1 << cr.QX)
    y_real_ref = conv3_out_real.T
    cos = float(np.dot(y_real_from_int.reshape(-1), y_real_ref.reshape(-1)) /
                (np.linalg.norm(y_real_from_int) * np.linalg.norm(y_real_ref) + 1e-30))
    print(f"cosine(quantized_int_output, real_float_conv3_output)={cos:.6f} (informational only -- "
          "this gate targets Q6.25 (QX=25) directly, and conv3's own raw output can reach real "
          "magnitude ~1004, which overflows Q6.25's ~+-64 32-bit-storage ceiling for a handful of "
          "elements (wraps, same class of issue conv_front_end_seq.sv's own final widen step hit -- "
          "see that file's header); this standalone gate never re-targets a narrower format the way "
          "the real front-end integration does (Q4.12, ~8192x smaller scale, comfortably in range), "
          "so a low number here does NOT indicate the per-row dequant itself is wrong -- the RTL-vs-"
          "Python bit-exact check above is unaffected either way)", file=sys.stderr)

    all_words, meta = build_resident([w_int8], LANES, WBW)
    n_words = meta[0]["n_words"]
    wbits = LANES * WBW
    hexw = (wbits + 3) // 4
    with open(os.path.join(out_dir, "w.mem"), "w") as f:
        f.write("\n".join(format(w, f"0{hexw}x") for w in all_words) + "\n")

    xt_flat = x_int8.T.reshape(-1)
    xt_rows = cr.pack_rows_p8(xt_flat, P)
    with open(os.path.join(out_dir, "xt.mem"), "w") as f:
        nib = (P * 8) // 4
        f.write("\n".join(f"{v & ((1 << (P*8)) - 1):0{nib}x}" for v in xt_rows) + "\n")

    b_rows = cr.pack_rows_p32(bias_q, P)
    with open(os.path.join(out_dir, "b.mem"), "w") as f:
        nib = (P * 32) // 4
        f.write("\n".join(f"{v & ((1 << (P*32)) - 1):0{nib}x}" for v in b_rows) + "\n")

    mant_rows = cr.pack_rows_pw(mant, P, 24)
    exp_rows = cr.pack_rows_pw(expo, P, 8)
    with open(os.path.join(out_dir, "dq_mant.mem"), "w") as f:
        f.write("\n".join(f"{v & ((1 << (P*24)) - 1):0{(P*24)//4}x}" for v in mant_rows) + "\n")
    with open(os.path.join(out_dir, "dq_exp.mem"), "w") as f:
        f.write("\n".join(f"{v & ((1 << (P*8)) - 1):0{(P*8)//4}x}" for v in exp_rows) + "\n")

    gold_flat = gold_tc.reshape(-1)
    gold_rows = cr.pack_rows_p32(gold_flat, P)
    gold_masked = [v & 0xFFFFFFFFFFFFFFFF for v in gold_rows]

    return {"P": P, "CIN": cin, "COUT": COUT, "KW": KW, "STRIDE": STRIDE, "TIN": tin,
            "TOUT": tout, "N_WORDS": n_words, "WWORDS": max(n_words, 1),
            "gold_rows": gold_masked}


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_conv3")
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
