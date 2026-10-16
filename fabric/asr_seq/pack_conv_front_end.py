"""Golden reference + RTL test vectors for conv_front_end_seq.sv -- the
Stage 1 conv front-end's own top-level FSM: conv1 -> tanh_ -> groupnorm1 ->
conv2 -> gelu -> conv3 -> gelu (permute is free, see that file's header),
end to end, against real moonshine-tiny weights and real audio
(torch.manual_seed(0), L=3000, this project's own gate-audio convention).

Reconstructs the SAME integer pipeline conv_front_end_seq.sv's own RTL
computes, stage by stage, reusing every existing block's own reference
function (conv1d_ref.conv1d_int_ref, pack_groupnorm1.gn_int, run_tanh.tanh_q,
run_gelu.gelu_q) -- no new arithmetic here either, only the SAME format-glue
choices (per-row (mant,exp) dequant targeting Q4.12 instead of the
standalone conv gates' own Q6.25, sat16, actquant, widen-by-13, PER-CHANNEL
activation quantization at the groupnorm1->conv2 and gelu1->conv3
boundaries via conv1d_ref.quantize_act_per_channel/
rescale_weight_cols_per_channel) the RTL's own header documents.

    .venv/bin/python -m fabric.asr_seq.pack_conv_front_end gen --dir <sim_dir>
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
from fabric.asr_seq.run_tanh import tanh_table, tanh_q  # noqa: E402
from fabric.stage3.run_gelu import gelu_table, gelu_q  # noqa: E402
from fabric.asr_seq.pack_groupnorm1 import gn_int, BETA_FRAC  # noqa: E402
from fabric.stage3.run_layernorm import G_FRAC, seed_table  # noqa: E402
from fabric.stage3.pack_banked_resident_vec import build_resident  # noqa: E402
from fabric.stage3.seq_ref import quantize_scale_24  # noqa: E402

P = 8
LANES, WBW = 128, 8
Q412 = 12
Q625 = 25
OUT_FRAC_GN = 22   # groupnorm1_vec.sv's own OUT_FRAC
AUDIO_SEED = 0
AUDIO_LEN = 3000

COUT1, KW1, STRIDE1 = 288, 127, 64
COUT2, KW2, STRIDE2 = 576, 7, 3
COUT3, KW3, STRIDE3 = 288, 3, 2


def sat16(x):
    return np.clip(np.asarray(x, dtype=np.int64), -32768, 32767)


def widen1312(x_q412):
    return np.asarray(x_q412, dtype=np.int64) << 13


def widen1312_sat(x_q412_wide):
    """Matches conv_front_end_seq.sv's own widen1312_sat exactly: x<<<13,
    SATURATING (not wrapping) whenever that would overflow signed int32 --
    x > +262143 (real > ~+64, Q6.25's own representable ceiling) clips to
    INT32_MAX, x < -262144 clips to INT32_MIN, otherwise the shift is
    exact. Chosen over decoder/encoder/output_head's own xres_bank
    precedent (which wraps) because this is NEW code with a real choice to
    make, not an already-shipped, already-gated design being revisited --
    see conv_front_end_seq.sv's own header for the reasoning."""
    x = np.asarray(x_q412_wide, dtype=np.int64)
    return np.where(x > 262143, 0x7FFFFFFF,
           np.where(x < -262144, -0x80000000, x << 13))


def gelu_wide_q412(x_wide, lut):
    """Matches gelu_wide_vec.sv exactly (run_gelu_wide.py's own reference,
    reused here directly in spirit): gelu_q on the sat16-clipped value for
    in-domain/very-negative inputs (already correct there), overridden by
    the WIDE value itself, unclipped, when x > +32767 (real >+8 units) --
    GELU(x)->x that fast on the positive side, which gelu_lut2.sv's own
    fixed Q4.12 domain can't represent."""
    x_wide = np.asarray(x_wide, dtype=np.int64)
    x_clip = sat16(x_wide)
    lut_out = gelu_q(x_clip, lut)
    return np.where(x_wide > 32767, x_wide, lut_out)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def load_real():
    from transformers import MoonshineForConditionalGeneration
    model = MoonshineForConditionalGeneration.from_pretrained(
        "UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    enc = model.model.encoder

    torch.manual_seed(AUDIO_SEED)
    audio = torch.randn(1, AUDIO_LEN, dtype=torch.float32)
    with torch.no_grad():
        c1 = enc.conv1(audio.unsqueeze(1))          # (1,288,T1)
        t1 = torch.tanh(c1)
        gn = enc.groupnorm(t1)                        # (1,288,T1)
        c2 = enc.conv2(gn)                              # (1,576,T2)
        g2 = F.gelu(c2)
        c3 = enc.conv3(g2)                              # (1,288,T3)
        g3 = F.gelu(c3)
        front_end_real = g3.permute(0, 2, 1)[0]          # (T3,288), real float reference

    return dict(
        audio=audio[0].detach().numpy().astype(np.float64),
        w1=enc.conv1.weight.detach().numpy().astype(np.float64),
        gn_gamma=enc.groupnorm.weight.detach().numpy().astype(np.float64),
        gn_beta=enc.groupnorm.bias.detach().numpy().astype(np.float64),
        w2=enc.conv2.weight.detach().numpy().astype(np.float64),
        b2=enc.conv2.bias.detach().numpy().astype(np.float64),
        w3=enc.conv3.weight.detach().numpy().astype(np.float64),
        b3=enc.conv3.bias.detach().numpy().astype(np.float64),
        # real intermediate references, used for informational per-stage
        # cosine prints only (not needed for calibration any more -- the
        # per-channel activation-quantization fix, conv1d_ref.
        # quantize_act_per_channel, doesn't need a search against these at
        # all, unlike the shared-shift calibration attempt that preceded it)
        c2_real=c2[0].detach().numpy().astype(np.float64).T,     # (T2,COUT2) pre-gelu1
        c3_real=c3[0].detach().numpy().astype(np.float64).T,     # (T3,COUT3) pre-gelu2
        front_end_real=front_end_real.detach().numpy().astype(np.float64),
    )


def _wmem(path, vals, nib):
    mask = (1 << (4 * nib)) - 1
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v) & mask:0{nib}x}" for v in vals) + "\n")


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    d = load_real()
    audio = d["audio"][None, :]         # (1, TIN1)
    tin1 = audio.shape[1]

    # ================= conv1 (dq target = Q4.12, feeds tanh directly) =====
    x1_pad = cr.pad_cin_input(audio, P)              # (8, TIN1)
    w1_pad = cr.pad_cin_weight(d["w1"], P)             # (288,8,127)
    cin1 = x1_pad.shape[0]
    ashift1 = cr.choose_ashift(audio)
    x1_int8 = cr.quantize_act(x1_pad, ashift1)
    w1_flat = cr.transpose_weight(w1_pad)
    w1_int8, wshift1 = cr.quantize_weight_per_row(w1_flat)   # (COUT1,) per-output-channel shift
    scale1 = np.exp2((Q412 - ashift1 - wshift1).astype(np.float64))
    mant1, expo1 = quantize_scale_24(scale1)
    mant1 = np.asarray(mant1, dtype=np.int64); expo1 = np.asarray(expo1, dtype=np.int64)
    c1_out = cr.conv1d_int_ref(x1_int8, w1_int8, KW1, STRIDE1, COUT1, cin1,
                                mant1, expo1, bias_q=None)          # (TOUT1,COUT1) Q4.12
    c1_sat = sat16(c1_out)
    tout1 = c1_out.shape[0]
    print(f"conv1: ashift={ashift1} wshift range=[{wshift1.min()},{wshift1.max()}] TOUT1={tout1}",
          file=sys.stderr)

    # ================= tanh =================================================
    lut_tanh = tanh_table()
    t1_q412 = tanh_q(c1_sat, lut_tanh)                  # (TOUT1,COUT1) Q4.12

    # ================= widen Q4.12 -> Q6.25, groupnorm1 =====================
    gn_x_625 = widen1312(t1_q412)                        # (TOUT1,COUT1) Q6.25
    c_gn, t_gn = COUT1, tout1
    gn_x_flat = gn_x_625.reshape(-1)                       # t-major/channel-minor
    g_gn = np.round(d["gn_gamma"] * (1 << G_FRAC)).astype(np.int64)
    b_gn = np.round(d["gn_beta"] * (1 << BETA_FRAC)).astype(np.int64)
    gn_out_list, gn_mean, gn_var, gn_A, gn_Yr = gn_int(
        list(gn_x_flat.astype(object)), list(g_gn.astype(object)), list(b_gn.astype(object)))
    gn_out = np.array([int(v) for v in gn_out_list], dtype=np.int64)   # (T*C,) Q.22
    print(f"groupnorm1: mean={gn_mean} var={gn_var} A={gn_A} Yr={gn_Yr}", file=sys.stderr)

    # ================= actquant groupnorm1 -> conv2 input (PER-CHANNEL) =====
    # True per-channel activation quantization, not a single shared shift
    # (shift-calibration alone got whole-chain cosine to ~0.96; this closes
    # nearly all of the rest, see conv_front_end_seq.sv's own header for the
    # full derivation and history). gn_out (T*C flat, Q.22) -> real -> one
    # ashift PER CHANNEL (quantize_act_per_channel) -> conv2's own weight
    # COLUMNS rescaled to compensate (rescale_weight_cols_per_channel), so
    # the per-row dequant scale below needs NO separate ashift term at all
    # -- it cancels out algebraically (see that function's own docstring).
    cin2 = COUT1                                          # = CIN2, 288
    x2_int8, gn_rshift = cr.quantize_act_per_channel_from_int(gn_out.reshape(t_gn, c_gn).T)  # (CIN2,T2), (CIN2,)
    ashift_gn = OUT_FRAC_GN - gn_rshift               # effective multiply-exponent, for weight rescale only
    print(f"groupnorm1->conv2: per-channel rshift range=[{gn_rshift.min()},{gn_rshift.max()}]",
          file=sys.stderr)

    w2_pad = cr.pad_cin_weight(d["w2"], P)                 # no-op, 288 already /8
    w2_flat = cr.transpose_weight(w2_pad)
    w2_flat_c = cr.rescale_weight_cols_per_channel(w2_flat, ashift_gn, cin2)
    w2_int8, wshift2 = cr.quantize_weight_per_row(w2_flat_c)   # (COUT2,) per-output-channel shift
    scale2 = np.exp2((Q412 - wshift2).astype(np.float64))       # no ashift term -- absorbed above
    mant2, expo2 = quantize_scale_24(scale2)
    mant2 = np.asarray(mant2, dtype=np.int64); expo2 = np.asarray(expo2, dtype=np.int64)
    b2_q = np.round(d["b2"] * (1 << Q412)).astype(np.int64)   # bias in Q4.12 (matches conv2's own target)

    # ================= conv2 (dq target = Q4.12, feeds gelu1) ===============
    c2_out = cr.conv1d_int_ref(x2_int8, w2_int8, KW2, STRIDE2, COUT2, cin2,
                                mant2, expo2, bias_q=b2_q)          # (TOUT2,COUT2) Q4.12, WIDE (not sat16'd)
    tout2 = c2_out.shape[0]
    print(f"conv2: wshift range=[{wshift2.min()},{wshift2.max()}] TOUT2={tout2} "
          f"max|c2_out|={int(np.max(np.abs(c2_out)))} "
          f"cos_vs_real={_cos(c2_out.astype(np.float64) / (1 << Q412), d['c2_real']):.6f}",
          file=sys.stderr)

    # ================= gelu1 (gelu_wide_q412 -- see that function's own
    # docstring and conv_front_end_seq.sv's own header for the Q4.12
    # clipping fix this closes) ================================================
    lut_gelu = gelu_table()
    g1_q412 = gelu_wide_q412(c2_out, lut_gelu)            # (TOUT2,COUT2) Q4.12, WIDE

    # ================= actquant gelu1 -> conv3 input (PER-CHANNEL) ==========
    # Same per-channel fix as groupnorm1->conv2 above -- this is the
    # boundary where it mattered most (conv3's narrow KW3=3 kernel doesn't
    # average a coarse shared activation scale away the way a wider
    # reduction would).
    cin3 = COUT2                                              # = CIN3, 576
    x3_int8, ge1_rshift = cr.quantize_act_per_channel_from_int(g1_q412.T)   # (CIN3=576, T2)
    ashift_ge1 = Q412 - ge1_rshift                    # effective multiply-exponent, for weight rescale only
    print(f"gelu1->conv3: per-channel rshift range=[{ge1_rshift.min()},{ge1_rshift.max()}]",
          file=sys.stderr)

    w3_pad = cr.pad_cin_weight(d["w3"], P)                    # no-op, 576 already /8
    w3_flat = cr.transpose_weight(w3_pad)
    w3_flat_c = cr.rescale_weight_cols_per_channel(w3_flat, ashift_ge1, cin3)
    w3_int8, wshift3 = cr.quantize_weight_per_row(w3_flat_c)    # (COUT3,) per-output-channel shift
    scale3 = np.exp2((Q412 - wshift3).astype(np.float64))        # no ashift term -- absorbed above
    mant3, expo3 = quantize_scale_24(scale3)
    mant3 = np.asarray(mant3, dtype=np.int64); expo3 = np.asarray(expo3, dtype=np.int64)
    b3_q = np.round(d["b3"] * (1 << Q412)).astype(np.int64)

    # ================= conv3 (dq target = Q4.12, feeds gelu2) ===============
    c3_out = cr.conv1d_int_ref(x3_int8, w3_int8, KW3, STRIDE3, COUT3, cin3,
                                mant3, expo3, bias_q=b3_q)          # (TOUT3,COUT3) Q4.12, WIDE
    tout3 = c3_out.shape[0]
    print(f"conv3: wshift range=[{wshift3.min()},{wshift3.max()}] TOUT3={tout3} "
          f"max|c3_out|={int(np.max(np.abs(c3_out)))} "
          f"cos_vs_real={_cos(c3_out.astype(np.float64) / (1 << Q412), d['c3_real']):.6f}",
          file=sys.stderr)

    # ================= gelu2 (gelu_wide_q412, same fix as gelu1) ============
    g2_q412 = gelu_wide_q412(c3_out, lut_gelu)            # (TOUT3,COUT3) Q4.12, WIDE

    # ================= widen Q4.12 -> Q6.25: FINAL OUTPUT (saturating) ======
    # widen1312_sat: g2_q412 can now genuinely exceed Q6.25's own ~+-64
    # representable range post-widen -- SATURATE (not wrap), matching
    # conv_front_end_seq.sv's own widen1312_sat exactly.
    final_out = widen1312_sat(g2_q412)                       # (TOUT3,COUT3) Q6.25 (saturated)

    # informational-only float check against the real HF model's own conv
    # front-end output
    y_real_from_int = final_out.astype(np.float64) / (1 << Q625)
    y_real_ref = d["front_end_real"]
    cos = float(np.dot(y_real_from_int.reshape(-1), y_real_ref.reshape(-1)) /
                (np.linalg.norm(y_real_from_int) * np.linalg.norm(y_real_ref) + 1e-30))
    print(f"cosine(quantized_int_output, real_float_front_end_output)={cos:.6f} (informational only)",
          file=sys.stderr)

    # ================= RTL vectors ============================================
    all_w1, meta1 = build_resident([w1_int8], LANES, WBW)
    all_w2, meta2 = build_resident([w2_int8], LANES, WBW)
    all_w3, meta3 = build_resident([w3_int8], LANES, WBW)
    hexw = (LANES * WBW + 3) // 4
    _wmem(os.path.join(out_dir, "w1.mem"), all_w1, hexw)
    _wmem(os.path.join(out_dir, "w2.mem"), all_w2, hexw)
    _wmem(os.path.join(out_dir, "w3.mem"), all_w3, hexw)

    xt1_flat = x1_int8.T.reshape(-1)              # (TIN1,CIN1) -> flat, CGRP1=1
    _wmem(os.path.join(out_dir, "xt1.mem"), cr.pack_rows_p8(xt1_flat, P), (P * 8) // 4)

    for tag, mant, expo in [("1", mant1, expo1), ("2", mant2, expo2), ("3", mant3, expo3)]:
        _wmem(os.path.join(out_dir, f"dq{tag}_mant.mem"), cr.pack_rows_pw(mant, P, 24), (P * 24) // 4)
        _wmem(os.path.join(out_dir, f"dq{tag}_exp.mem"), cr.pack_rows_pw(expo, P, 8), (P * 8) // 4)

    _wmem(os.path.join(out_dir, "g.mem"), cr.pack_rows_p32(g_gn, P), (P * 32) // 4)
    _wmem(os.path.join(out_dir, "b.mem"), cr.pack_rows_p32(b_gn, P), (P * 32) // 4)
    _wmem(os.path.join(out_dir, "b2.mem"), cr.pack_rows_p32(b2_q, P), (P * 32) // 4)
    _wmem(os.path.join(out_dir, "b3.mem"), cr.pack_rows_p32(b3_q, P), (P * 32) // 4)
    _wmem(os.path.join(out_dir, "seed.mem"), seed_table(), 5)

    with open(os.path.join(out_dir, "tanh_lut.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in lut_tanh) + "\n")
    # vec_gelu.sv instantiates gelu_lut2 (BRAM-shared even/odd-banked core),
    # not gelu_lut directly -- same split pack_encoder_block.py's own export
    # already documented finding (that file's own header note on this).
    with open(os.path.join(out_dir, "gelu_lut_e.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in lut_gelu[0::2]) + "\n")
    with open(os.path.join(out_dir, "gelu_lut_o.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in lut_gelu[1::2]) + "\n")

    # gn_rshift/ge1_rshift are ALREADY the right-shift amount
    # conv_front_end_seq.sv's own actq() port expects directly (see
    # quantize_act_per_channel_from_int's own docstring for the two real
    # bugs found and fixed getting here: (1) an earlier version wrote the
    # "ashift" multiply-exponent into this table instead of the matching
    # right-shift (target_frac - ashift); (2) an earlier version of the
    # quantization itself used round-to-nearest-on-real-values instead of
    # actq()'s own floor-shift-on-the-integer, a 1-LSB-per-element
    # mismatch that compounded into a large bit-exact failure).
    _wmem(os.path.join(out_dir, "gn_ash.mem"), cr.pack_rows_pw(gn_rshift, P, 8), (P * 8) // 4)
    _wmem(os.path.join(out_dir, "ge1_ash.mem"), cr.pack_rows_pw(ge1_rshift, P, 8), (P * 8) // 4)

    gold_flat = final_out.reshape(-1)
    gold_rows = cr.pack_rows_p32(gold_flat, P)
    gold_masked = [v & 0xFFFFFFFFFFFFFFFF for v in gold_rows]

    return {
        "TIN1": tin1, "NWORDS1": meta1[0]["n_words"], "NWORDS2": meta2[0]["n_words"],
        "NWORDS3": meta3[0]["n_words"], "TOUT3": tout3, "COUT3": COUT3,
        "gold_rows": gold_masked,
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_conv_front_end")
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gen")
    g.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    if a.cmd == "gen":
        m = main_gen(a.dir)
        print(f"TOUT3={m['TOUT3']} COUT3={m['COUT3']}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
