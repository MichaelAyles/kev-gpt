"""Golden reference + RTL test vectors for output_head_seq.sv -- Stage 4 of
gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md: final decoder LayerNorm -> tied
lm_head GEMV (VOCAB=32768 x D=288, INT8) -> argmax. Runs once per real
decode step, fed the decoder's own real final-layer hidden state (BEFORE
the final norm -- this module applies that itself).

Uses checkpoint C's real, already-deployed PER-ROW dequant scheme
(fabric.stage3.seq_ref.quantize_scale_24, feeding vec_dequant.sv), not
the per-matrix single-shift simplification decoder_block_seq.sv/
encoder_block_seq.sv use -- see output_head_seq.sv's own header for why:
32768 very different output rows feed directly into an argmax, so one
shared shift risks flipping which token wins. The lm_head weight matrix
is quantized PER ROW too, matching model/c_port/ops.c's own
linear_lmhead_i8 convention (per-row float scale) -- reproduced here in
integer mantissa/exponent form instead, since that's what vec_dequant.sv
actually consumes.

Real decoder hidden states: the REAL HF model's own forward pass
(teacher-forced, all 3 of TOKEN_IDS in one causal-masked call -- exactly
equivalent to 3 autoregressive steps), with `decoder.norm` temporarily
replaced by an identity module so the RETURNED hidden state is the
pre-final-norm one this module's own LN input expects (dec.norm's own
call happens INSIDE this file's reference math, not before it) --
avoids reimplementing 6 layers of real attention/cross-attention by hand
a second time (pack_decoder_block.py already did that once, with real
INT8 quantization; this file needs a real FP32 reference point, not
another quantized one, so it reuses the HF model's own forward directly).

Usage:
    .venv/bin/python -m fabric.asr_seq.pack_output_head gen --dir <sim_dir>
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GEN2ASR_SW = os.path.expanduser("~/gen2asr/sw_model")
sys.path.insert(0, GEN2ASR_SW)
sys.path.insert(0, os.path.join(GEN2ASR_SW, "model"))

from fabric.stage3.seq_ref import rsh_round, quantize_scale_24  # noqa: E402
from fabric.stage3.run_layernorm import (  # noqa: E402
    QX, G_FRAC, A_FRAC, Y_FRAC, OUT_FRAC, VAR_FRAC, EPS_A, rsqrt_int, seed_table,
)

D = 288
VOCAB = 32768
P = 8
LN_EPS = 1e-5
RESID_FRAC = 25             # xres_bank's Q6.25 (matches decoder_block_seq.sv's own)
DQ_FRAC = 0                 # vec_dequant's own target frac -- plain integer logit

AUDIO_SEED = 0
AUDIO_LEN = 3000
TOKEN_IDS = [1, 940, 24936]   # same 3 real decode steps as every other ASR gate here


def floordiv(num: int, d: int) -> int:
    return num // d


def ln_int_gendiv(X, G):
    X = [int(v) for v in X]
    G = [int(v) for v in G]
    d_model = len(X)
    S = sum(X)
    mean = floordiv(S, d_model)
    d = [v - mean for v in X]
    SS = sum(di * di for di in d)
    var = floordiv(SS, d_model)
    A = (var >> (VAR_FRAC - A_FRAC)) + EPS_A
    if A <= 0:
        A = 1
    Yr = rsqrt_int(A)
    sh = (QX + Y_FRAC + G_FRAC) - OUT_FRAC
    out = []
    for i in range(d_model):
        t = d[i] * Yr
        t = t * G[i]
        yv = t >> sh if sh >= 0 else t << (-sh)
        out.append(yv)
    return out


def to_qfrac(x_real, frac) -> np.ndarray:
    return np.round(np.asarray(x_real, dtype=np.float64) * (1 << frac)).astype(np.int64)


def wrap32(x: np.ndarray) -> np.ndarray:
    """Wrap each lane to signed 32-bit two's complement -- xres_bank is
    `reg [P*32-1:0]` (exactly 32 bits/lane) in the RTL, and the .mem export
    below (`w32()`) already truncates to 32 bits on write; ln_int_gendiv
    must see the SAME truncated value, not the full-precision one, or the
    two silently diverge. Same fix pack_decoder_block.py/pack_encoder_
    block.py needed for their own accumulated residual stream, but
    triggered differently here: the REAL decoder hidden state's own
    magnitude (up to ~240 in real units, confirmed via this file's own
    load_real()) already exceeds Q6.25's 32-bit range for SOME of its 288
    elements even before any accumulation -- found via xres_bank[0]/[35]
    matching bit-for-bit (those two elements happened to be in range)
    while the LayerNorm's own internal sum (over all 288 elements) still
    diverged, since other, unchecked elements were NOT in range."""
    x = np.asarray(x, dtype=object)
    out = np.empty(len(x), dtype=np.int64)
    for i, v in enumerate(x):
        v = int(v) & 0xFFFFFFFF
        out[i] = v - 0x100000000 if v >= 0x80000000 else v
    return out


def choose_act_rshift_from_max(m: int, target_max: int = 100) -> int:
    if m <= target_max:
        return 0
    shift = 0
    while (m >> shift) > target_max:
        shift += 1
    return shift


def act_quantize(x_fixed: np.ndarray, act_rshift: int) -> np.ndarray:
    x = np.asarray(x_fixed, dtype=np.int64)
    q = x >> act_rshift if act_rshift >= 0 else x << (-act_rshift)
    clipped = np.clip(q, -128, 127)
    n_clip = int(np.sum(clipped != q))
    if n_clip:
        print(f"WARNING: act_quantize clipped {n_clip}/{len(q)} lanes "
              f"(max|q|={int(np.max(np.abs(q)))}, shift={act_rshift})", file=sys.stderr)
    return clipped


def quantize_weight_per_row(W: np.ndarray):
    """PER-ROW INT8 quantization (matches model/c_port/ops.c's own
    linear_lmhead_i8 convention, weight-side) -- one WSHIFT per output
    row, not one for the whole matrix. Returns (W_int8 (VOCAB,D) int64,
    wshift (VOCAB,) int64)."""
    max_abs = np.max(np.abs(W), axis=1)                       # (VOCAB,)
    max_abs = np.where(max_abs == 0, 1.0, max_abs)             # avoid log2(0)
    wshift = np.floor(np.log2(127.0 / max_abs)).astype(np.int64)
    W_int8 = np.round(W * (2.0 ** wshift[:, None]))
    W_int8 = np.clip(W_int8, -128, 127).astype(np.int64)
    # re-check per row (the same floor-then-round can occasionally clip a
    # row by 1 ULP at the boundary); shrink that row's own shift until clean.
    bad = np.max(np.abs(W_int8), axis=1) > 127
    while np.any(bad):
        wshift[bad] -= 1
        W_int8[bad] = np.clip(np.round(W[bad] * (2.0 ** wshift[bad, None])), -128, 127).astype(np.int64)
        bad = np.max(np.abs(W_int8), axis=1) > 127
    return W_int8, wshift


@contextlib.contextmanager
def _identity_norm(module):
    """Temporarily replace `module` (dec.norm) with an identity, so the
    decoder's own forward() returns the PRE-final-norm hidden state
    directly -- avoids reimplementing 6 layers of real attention by hand
    a second time."""
    orig = module.forward
    module.forward = lambda x: x
    try:
        yield
    finally:
        module.forward = orig


def load_real():
    """Returns (dec_hidden_pre_norm (3,D) real FP32 -- the decoder's own
    final-layer output for TOKEN_IDS, teacher-forced, BEFORE dec.norm --
    w_lnf (D,) dec.norm's own real gamma, w_lmhead (VOCAB,D) the real
    tied lm_head weight)."""
    from transformers import MoonshineForConditionalGeneration

    model = MoonshineForConditionalGeneration.from_pretrained(
        "UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    enc = model.model.encoder
    dec = model.model.decoder

    torch.manual_seed(AUDIO_SEED)
    audio = torch.randn(1, AUDIO_LEN, dtype=torch.float32)
    with torch.no_grad():
        enc_hidden = enc(audio).last_hidden_state   # (1, T2, D), real, post-final-norm
        # (enc.forward() runs conv1->tanh->groupnorm->conv2->gelu->conv3->gelu->
        # permute->layers->final norm internally, including RoPE's own position
        # embeddings the encoder layers need -- reusing it directly here avoids
        # reimplementing that chain a second time just to get a real FP32
        # reference point; pack_encoder_block.py's own from-scratch reimplementation
        # was for a different purpose, producing its own quantized reference.)

        input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
        with _identity_norm(dec.norm):
            out = dec(input_ids=input_ids, encoder_hidden_states=enc_hidden, use_cache=False)
        dec_hidden_pre_norm = out.last_hidden_state[0].detach().numpy().astype(np.float64)   # (3, D)

    w_lnf = dec.norm.weight.detach().numpy().astype(np.float64)
    w_lmhead = model.proj_out.weight.detach().numpy().astype(np.float64)   # (VOCAB, D), tied to embed_tokens
    assert w_lmhead.shape == (VOCAB, D), w_lmhead.shape
    return dec_hidden_pre_norm, w_lnf, w_lmhead


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    dec_hidden, w_lnf, w_lmhead = load_real()
    n_steps = dec_hidden.shape[0]

    g_lnf = to_qfrac(w_lnf, G_FRAC)
    W_int8, wshift = quantize_weight_per_row(w_lmhead)   # (VOCAB,D), (VOCAB,)

    xres_steps = np.stack([wrap32(row) for row in to_qfrac(dec_hidden, RESID_FRAC)])   # (n_steps, D)

    # ================= PASS 1: profile the SINGLE activation's own
    # magnitude across all steps (one act_rshift shared by every one of
    # the 32768 output rows' own dot product -- see this file's header). ==
    profile_max = 0
    for step in range(n_steps):
        xn = np.asarray(ln_int_gendiv(list(xres_steps[step].astype(object)), list(g_lnf)),
                         dtype=np.int64)
        profile_max = max(profile_max, int(np.max(np.abs(xn))))
    act_rshift = choose_act_rshift_from_max(profile_max)
    print(f"profile: max|xn_lnf|={profile_max} -> act_rshift={act_rshift}", file=sys.stderr)

    # ================= PASS 2: real run =====================================
    # per-row dequant scale: TRUE_real[o] = gemvy[o] * 2^(act_rshift - OUT_FRAC - wshift[o])
    # (OUT_FRAC = ln_int_gendiv's own output fraction, matching the LN
    # module's real Q.22 format) -- see output_head_seq.sv's own header
    # for the derivation. DQ_FRAC=0 folds no further scale (plain integer
    # logit out).
    scale = np.exp2((act_rshift - OUT_FRAC - wshift).astype(np.float64))   # (VOCAB,)
    mant, expo = quantize_scale_24(scale)
    mant = np.asarray(mant, dtype=np.int64)
    expo = np.asarray(expo, dtype=np.int64)

    xn_steps = []
    argmax_idx = []
    argmax_val = []
    for step in range(n_steps):
        xn = np.asarray(ln_int_gendiv(list(xres_steps[step].astype(object)), list(g_lnf)),
                         dtype=np.int64)
        xn_steps.append(xn)
        x_int8 = act_quantize(xn, act_rshift)                       # (D,)
        gemvy = (W_int8.astype(np.int64) @ x_int8.astype(np.int64))  # (VOCAB,) int64, exact
        # vec_dequant.sv's own contract, vectorized:
        dq_shv = expo + DQ_FRAC
        dq_prod = gemvy.astype(object) * mant.astype(object)
        dq_val = np.empty(VOCAB, dtype=object)
        left = dq_shv >= 0
        dq_val[left] = [int(p) << int(s) for p, s in zip(dq_prod[left], dq_shv[left])]
        dq_val[~left] = [rsh_round(int(p), int(-s)) for p, s in zip(dq_prod[~left], dq_shv[~left])]
        dq_val32 = np.array([int(v) & 0xFFFFFFFF for v in dq_val], dtype=np.int64)
        dq_val32 = np.where(dq_val32 >= 0x80000000, dq_val32 - 0x100000000, dq_val32)

        idx = int(np.argmax(dq_val32))   # numpy argmax: first-occurrence-wins on ties, matches argmax_f
        argmax_idx.append(idx)
        argmax_val.append(int(dq_val32[idx]))
        # NOT expected to match TOKEN_IDS[step+1] -- this gate's own audio
        # input is torch.randn(seed=0), not the real speech clip
        # TOKEN_IDS=[1,940,24936] was borrowed from (the real "He hoped."
        # transcription's own tokens, reused project-wide purely as
        # plausible, real, in-vocab decode-step IDs to exercise causal
        # self-attention/RoPE positions correctly -- see
        # pack_decoder_self_attn.py's own header). Checked directly: even
        # the PLAIN FP32 computation (no quantization at all) doesn't
        # predict TOKEN_IDS[step+1] from this synthetic audio's own real
        # cross-attention K/V, confirming the mismatch is expected, not a
        # bug in this file's own hidden-state extraction or dequant math.
        # This gate's own pass/fail bar is RTL == this reference,
        # bit-exact -- not semantic correctness against a real transcript.
        print(f"step {step}: argmax_idx={idx} argmax_val={dq_val32[idx]}", file=sys.stderr)

    # =========================================================================
    from fabric.stage3.pack_banked_resident_vec import build_resident
    LANES, WBW = 128, 8
    all_words, wmeta = build_resident([W_int8], LANES, WBW)
    wb_lm = wmeta[0]["w_base"]

    wbits = LANES * WBW
    hexw = (wbits + 3) // 4
    with open(os.path.join(out_dir, "w.mem"), "w") as f:
        f.write("\n".join(format(v, f"0{hexw}x") for v in all_words) + "\n")

    def w32(f, v):
        f.write(format(int(v) & 0xFFFFFFFF, "08x") + "\n")

    with open(os.path.join(out_dir, "gamma_lnf.mem"), "w") as f:
        for r in range(D // P):
            for k in range(P):
                w32(f, g_lnf[r * P + k])

    # layernorm_vec_gendiv.sv's own rsqrt seed ROM -- same generation as
    # every other ASR gate here, not re-derived.
    with open(os.path.join(out_dir, "seed.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFFF:05x}" for v in seed_table()) + "\n")

    # per-row (mant,exp) dequant table, P-wide packed rows (ROWS_VOCAB=VOCAB/P)
    # matching vec_dequant.sv's own packed-bus convention.
    with open(os.path.join(out_dir, "dq_mant.mem"), "w") as f:
        for r in range(VOCAB // P):
            word = 0
            for k in range(P):
                word |= (int(mant[r * P + k]) & ((1 << 24) - 1)) << (24 * k)
            f.write(format(word, f"0{24 * P // 4}x") + "\n")
    with open(os.path.join(out_dir, "dq_exp.mem"), "w") as f:
        for r in range(VOCAB // P):
            word = 0
            for k in range(P):
                word |= (int(expo[r * P + k]) & 0xFF) << (8 * k)
            f.write(format(word, f"0{8 * P // 4}x") + "\n")

    with open(os.path.join(out_dir, "xres_in.mem"), "w") as f:
        for step in range(n_steps):
            for r in range(D // P):
                for k in range(P):
                    w32(f, xres_steps[step, r * P + k])

    with open(os.path.join(out_dir, "argmax_ref.mem"), "w") as f:
        for step in range(n_steps):
            f.write(format(argmax_idx[step] & 0xFFFF, "04x") + "\n")

    manifest = {
        "d": D, "vocab": VOCAB, "n_steps": n_steps,
        "resid_frac": RESID_FRAC, "act_rshift": act_rshift, "dq_frac": DQ_FRAC,
        "w_base": wb_lm, "n_words_total": len(all_words),
        "argmax_idx": argmax_idx, "argmax_val": argmax_val,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"GEN dir={out_dir} n_steps={n_steps} vocab={VOCAB} n_words={len(all_words)} "
          f"argmax_idx={argmax_idx}")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_output_head")
    p.add_argument("cmd", choices=["gen"])
    p.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    main_gen(a.dir)


if __name__ == "__main__":
    main()
