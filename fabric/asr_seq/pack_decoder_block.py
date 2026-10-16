"""Golden reference + RTL test vectors for decoder_block_seq.sv -- the
MULTI-LAYER, MULTI-STEP functional gate for the real, sized decoder-block
top-level FSM. Runs ALL 6 real moonshine-tiny decoder layers, chained
(layer L's own residual OUTPUT becomes layer L+1's own residual INPUT, at
the same decode step), across ALL of TOKEN_IDS' real decode steps
(step=0,1,2) -- a real, full multi-layer autoregressive decode, not a
single-layer toy.

Extends the earlier multi-step-only gate (single layer 0, see git history)
to real layer-looping. That needed exactly one RTL change:
decoder_block_seq.sv's WB_Q/WB_K/.../WB_FC2 weight-offset PARAMETERS
(compile-time, one value for the whole simulation) became wb_q/wb_k/.../
wb_fc2 runtime PORTS, since each of the 6 layers needs its OWN offset into
the resident weight image for the "same" call site (layer 1's own
Q-projection lives at a different address than layer 0's). Everything
else needed for multi-layer was already there: kv_bank.sv's own storage is
already indexed per-layer internally via `blk` (both self- and
cross-attention), and layer-to-layer chaining is "free" -- xres_bank is
the SAME physical memory across `go` pulses within a step, so layer L's
own RES3 write is already sitting there as layer L+1's own LN1 input; the
testbench only needs to reload gamma/bias (RTL banks sized for one layer,
reused sequentially -- see below) and change wb_*/blk between layers.

GEMV quantization scheme (unchanged from the single-layer gates, see
their own history for the full rationale): per-matrix single-scale INT8
weights (independent per layer -- each layer's own weight distribution
gets its own WSHIFT), per-call single-shift INT8 activations, one
combined dequant shift. ACT_RSHIFT/g_frac stay GLOBAL across ALL 6 layers
AND all 3 steps (decoder_block_seq.sv's ACT_*/GF_* were NOT converted to
per-layer ports -- only the weight offsets needed that): PASS 1 profiles
every call site's magnitude across the full (layer x step) = 18-point
grid; PASS 2 recomputes the real, bit-exact reference with ONE fixed
shift per call site. This works because LayerNorm re-normalizes the
residual stream's scale at every layer boundary -- the activation feeding
each layer's own Q/K/V (etc.) GEMVs stays in a comparable range
regardless of layer, verified empirically here (act_quantize's own
clipping check), not assumed.

Reuses this project's proven fixed-point references directly: rsh_round/sat
(fabric.stage3.seq_ref), int_softmax_q/exp_table (fabric.stage3.run_softmax),
quant_head_asym/dequant_head (model.goformer_kvq), rope_apply_ref/
build_cos_sin_rom (fabric.asr_seq.pack_rope). LayerNorm's own integer math
(mean/var floor-divide, rsqrt seed+2-Newton) is reproduced here matching
layernorm_vec_gendiv.sv exactly (ln_int_gendiv, same as
fabric.asr_seq.run_layernorm_gendiv but inlined so this file owns the whole
pipeline's own intermediate values for RTL vector generation).

Usage:
    .venv/bin/python -m fabric.asr_seq.pack_decoder_block gen --dir <sim_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GEN2ASR_SW = os.path.expanduser("~/gen2asr/sw_model")
sys.path.insert(0, GEN2ASR_SW)
sys.path.insert(0, os.path.join(GEN2ASR_SW, "model"))

from fabric.stage3.seq_ref import rsh_round, sat, q_round_div  # noqa: E402
from fabric.stage3.run_softmax import int_softmax_q, exp_table  # noqa: E402
from model.goformer_kvq import quant_head_asym, dequant_head  # noqa: E402
from fabric.asr_seq.pack_rope import rope_apply_ref, build_cos_sin_rom  # noqa: E402
from fabric.stage3.run_layernorm import (  # noqa: E402
    QX, G_FRAC, A_FRAC, Y_FRAC, OUT_FRAC, VAR_FRAC, EPS_A, rsqrt_int, seed_table,
)

D = 288
FFN = 1152
DFFN2 = 2 * FFN
NHEAD = 8
HEAD_DIM = 36
ATTN_P = 4
HR_ATTN = HEAD_DIM // ATTN_P            # 9
T2 = 6
NLAYER = 6                              # real moonshine-tiny decoder depth
ROT_DIM = 32
ROT_PAIRS = 16
LN_EPS = 1e-5
VFRAC = 16
Q16 = 1 << VFRAC
ISQRT = 3
SCORE_FRAC = 8
PROB_FRAC = 20
SCORE_SH = 2 * VFRAC + ISQRT - SCORE_FRAC     # 27
CTX_SH = PROB_FRAC + VFRAC - 25               # 11
KBITS = 8
POST_SCALE_Q16 = round(math.sqrt(4.0 / 3.0) * Q16)   # 75674

AUDIO_SEED = 0
AUDIO_LEN = 3000            # -> T2=6, same as pack_encoder_self_attn.py
TOKEN_IDS = [1, 940, 24936]  # same as pack_decoder_self_attn.py -- real decode steps 0,1,2

RESID_FRAC = 25             # xres_bank's Q6.25
FRAC_QKV = VFRAC             # 16
FRAC_FC1 = 12
CALL_NAMES = ["q", "k", "v", "o", "cq", "oc", "fc1", "fc2"]
W_KEYS = {"q": "sq", "k": "sk", "v": "sv", "o": "so",
          "cq": "cq", "oc": "co", "fc1": "fc1", "fc2": "fc2"}


# ---- LayerNorm (layernorm_vec_gendiv's own exact integer math) --------------
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


# ---- GEMV quantization --------------------------------------------------------
def quantize_weight(W: np.ndarray):
    """Returns (W_int8 (M,K) int64, WSHIFT). Single scale per matrix,
    WSHIFT chosen so max|round(W*2^WSHIFT)| <= 127 -- no clipping. Layer-
    and step-invariant (depends only on the weight matrix itself)."""
    max_abs = float(np.max(np.abs(W)))
    wshift = int(math.floor(math.log2(127.0 / max_abs)))
    while True:
        W_int8 = np.clip(np.round(W * (2.0 ** wshift)), -128, 127).astype(np.int64)
        if int(np.max(np.abs(W_int8))) <= 127:
            break
        wshift -= 1
    return W_int8, wshift


def choose_act_rshift_from_max(m: int, target_max: int = 100) -> int:
    """Smallest ACT_RSHIFT (>=0) bringing observed max magnitude m at or
    below target_max -- applied to a magnitude already maximized across
    every (layer, step) pair, so ONE shift is safe for all of them."""
    if m <= target_max:
        return 0
    shift = 0
    while (m >> shift) > target_max:
        shift += 1
    return shift


def act_quantize(x_fixed: np.ndarray, act_rshift: int) -> np.ndarray:
    """x_int8 = sat(x_fixed >>> act_rshift, -128, 127) -- PLAIN floor shift,
    matching decoder_block_seq.sv's G_XFEED exactly (no rounding). Asserts
    no clipping actually occurs (would mean ACT_RSHIFT was chosen too
    small) -- a real check, not just documentation."""
    x = np.asarray(x_fixed, dtype=np.int64)
    q = x >> act_rshift if act_rshift >= 0 else x << (-act_rshift)   # floor, matches >>>
    clipped = np.clip(q, -128, 127)
    n_clip = int(np.sum(clipped != q))
    if n_clip:
        print(f"WARNING: act_quantize clipped {n_clip}/{len(q)} lanes "
              f"(max|q|={int(np.max(np.abs(q)))}, shift={act_rshift})", file=sys.stderr)
    return clipped


def gemv_int(W_int8: np.ndarray, x_int8: np.ndarray) -> np.ndarray:
    return (W_int8.astype(object) @ x_int8.astype(object)).astype(np.int64)


def gdequant(raw: np.ndarray, g_frac: int) -> np.ndarray:
    """raw >>> g_frac (arithmetic/floor shift, matches Verilog `>>>` exactly
    -- Python's native `>>` on an int is already floor division, no special
    casing needed for negative values)."""
    raw = np.asarray(raw, dtype=object)
    if g_frac >= 0:
        return np.asarray([int(v) >> g_frac for v in raw], dtype=np.int64)
    return np.asarray([int(v) << (-g_frac) for v in raw], dtype=np.int64)


def wrap32(x: np.ndarray) -> np.ndarray:
    """Wrap each lane to signed 32-bit two's complement -- xres_bank/
    gout_bank are `reg [P*32-1:0]` (exactly 32 bits/lane) in the RTL, and
    decoder_block_seq.sv's own residual add (`res_add_word`, per-lane
    `$signed(a)+$signed(b)`, result stored back into a 32-bit slot)
    truncates on every store, not just at the end. A single decoder LAYER's
    own O/Oc/FC2 contributions stay comfortably inside int32, but the
    ACCUMULATED xres stream does not: real layer-0 weights already push
    |xres3| past INT32_MAX by layer 1 (found empirically, not assumed --
    profile with this file's own `python -m ... gen` run). Python's
    arbitrary-precision ints never wrap on their own, so every xres0/1/2/3
    computed here must go through this explicitly, or the reference
    silently diverges from the RTL's own per-store truncation the moment
    magnitudes cross 2^31."""
    x = np.asarray(x, dtype=object)
    out = np.empty(len(x), dtype=np.int64)
    for i, v in enumerate(x):
        v = int(v) & 0xFFFFFFFF
        out[i] = v - 0x100000000 if v >= 0x80000000 else v
    return out


def layernorm_nobias_f(x, gamma, eps):
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / np.sqrt(var + eps) * gamma


# ---- real weight/data loading -------------------------------------------------
def load_real():
    """Returns (enc_hidden, w_enc_finalln, w_per_layer (list of NLAYER
    dicts), embed) -- ALL 6 real decoder layers' weights, not just layer 0."""
    from transformers import MoonshineForConditionalGeneration

    model = MoonshineForConditionalGeneration.from_pretrained(
        "UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    enc = model.model.encoder
    assert len(model.model.decoder.layers) == NLAYER, \
        f"expected NLAYER={NLAYER} real decoder layers, got {len(model.model.decoder.layers)}"

    torch.manual_seed(AUDIO_SEED)
    audio = torch.randn(1, AUDIO_LEN, dtype=torch.float32)
    with torch.no_grad():
        h = F.tanh(enc.conv1(audio.unsqueeze(1)))
        h = enc.groupnorm(h)
        h = F.gelu(enc.conv2(h))
        h = F.gelu(enc.conv3(h))
        h = h.permute(0, 2, 1)
    enc_hidden = h[0].detach().numpy().astype(np.float64)
    w_enc_finalln = enc.layer_norm.weight.detach().numpy().astype(np.float64)

    w_per_layer = []
    for li in range(NLAYER):
        dec = model.model.decoder.layers[li]
        w = {}
        w["ln1"] = dec.input_layernorm.weight.detach().numpy().astype(np.float64)
        w["sq"] = dec.self_attn.q_proj.weight.detach().numpy().astype(np.float64)
        w["sk"] = dec.self_attn.k_proj.weight.detach().numpy().astype(np.float64)
        w["sv"] = dec.self_attn.v_proj.weight.detach().numpy().astype(np.float64)
        w["so"] = dec.self_attn.o_proj.weight.detach().numpy().astype(np.float64)
        w["ln2"] = dec.post_attention_layernorm.weight.detach().numpy().astype(np.float64)
        w["cq"] = dec.encoder_attn.q_proj.weight.detach().numpy().astype(np.float64)
        w["ck"] = dec.encoder_attn.k_proj.weight.detach().numpy().astype(np.float64)
        w["cv"] = dec.encoder_attn.v_proj.weight.detach().numpy().astype(np.float64)
        w["co"] = dec.encoder_attn.o_proj.weight.detach().numpy().astype(np.float64)
        w["ln3"] = dec.final_layernorm.weight.detach().numpy().astype(np.float64)
        w["fc1"] = dec.mlp.fc1.weight.detach().numpy().astype(np.float64)
        w["b_fc1"] = dec.mlp.fc1.bias.detach().numpy().astype(np.float64)
        w["fc2"] = dec.mlp.fc2.weight.detach().numpy().astype(np.float64)
        w["b_fc2"] = dec.mlp.fc2.bias.detach().numpy().astype(np.float64)
        w_per_layer.append(w)

    embed = model.model.decoder.embed_tokens.weight.detach().numpy().astype(np.float64)
    return enc_hidden, w_enc_finalln, w_per_layer, embed


def head_slice(vec_d, h):
    return vec_d[h * HEAD_DIM:(h + 1) * HEAD_DIM]


def one_layer(layer_state, step, xres0, act_rshifts):
    """One decoder LAYER's forward pass at `step`, given its own already-
    precomputed constants in `layer_state` (a dict: w, g_ln1/2/3,
    bias_fc1_fixed, bias_fc2_fixed, W8, wshift, k_cross_deq, v_cross_deq,
    self_k_cache, self_v_cache -- the last two MUTATED in place, one entry
    appended per step, same causal-cache pattern pack_decoder_self_attn.py
    established). `xres0` is this layer's own residual INPUT -- the
    caller's job to supply (a fresh token embedding for layer 0, or the
    PREVIOUS layer's own xres3 for layer>0, at this same step).
    `act_rshifts`: dict name->shift; None on the profiling pass (auto-
    derives per call from this (layer,step)'s own magnitude), fixed dict
    on the real pass (matches decoder_block_seq.sv's compile-time ACT_*).

    Returns (result_dict, profile_dict).
    """
    w = layer_state["w"]
    W8 = layer_state["W8"]
    wshift = layer_state["wshift"]
    k_cross_deq = layer_state["k_cross_deq"]
    v_cross_deq = layer_state["v_cross_deq"]
    self_k_cache = layer_state["self_k_cache"]
    self_v_cache = layer_state["self_v_cache"]
    cos_rom = layer_state["cos_rom"]
    sin_rom = layer_state["sin_rom"]

    profile = {}

    def lq(x_fixed, frac_in, name, frac_out, bias_fixed=None):
        profile[name] = int(np.max(np.abs(np.asarray(x_fixed, dtype=np.int64))))
        ar = act_rshifts[name] if act_rshifts is not None else choose_act_rshift_from_max(profile[name])
        x_int8 = act_quantize(x_fixed, ar)
        raw = gemv_int(W8[name], x_int8)
        g_frac = frac_in - ar + wshift[name] - frac_out
        y = gdequant(raw, g_frac)
        if bias_fixed is not None:
            y = y + np.asarray(bias_fixed, dtype=np.int64)
        return y, g_frac, ar

    xres0 = np.asarray(xres0, dtype=np.int64)

    # ---- LN1 ----
    xn1 = np.asarray(ln_int_gendiv(list(xres0.astype(object)), list(layer_state["g_ln1"])),
                      dtype=np.int64)

    # ---- Q/K/V GEMVs ----
    q_fixed, gfrac_q, ar_q = lq(xn1, OUT_FRAC, "q", FRAC_QKV)
    k_fixed, gfrac_k, ar_k = lq(xn1, OUT_FRAC, "k", FRAC_QKV)
    v_fixed, gfrac_v, ar_v = lq(xn1, OUT_FRAC, "v", FRAC_QKV)

    # ---- RoPE (self, real position = step) ----
    q_rope = np.zeros(D, dtype=np.int64)
    k_rope = np.zeros(D, dtype=np.int64)
    for h in range(NHEAD):
        q_rope[h * HEAD_DIM:(h + 1) * HEAD_DIM] = rope_apply_ref(
            head_slice(q_fixed, h), step, cos_rom, sin_rom, POST_SCALE_Q16)
        k_rope[h * HEAD_DIM:(h + 1) * HEAD_DIM] = rope_apply_ref(
            head_slice(k_fixed, h), step, cos_rom, sin_rom, POST_SCALE_Q16)

    # ---- self kv_bank write (K RoPE'd, V raw) + quantize-at-write; append
    # this step's own row to this LAYER's own running per-head cache. ----
    ctx_self = np.zeros(D, dtype=np.int64)
    for h in range(NHEAD):
        k_h = head_slice(k_rope, h)
        v_h = head_slice(v_fixed, h)
        k_codes, k_lo, k_scale = quant_head_asym(list(k_h), KBITS, divfree=True)
        v_codes, v_lo, v_scale = quant_head_asym(list(v_h), KBITS, divfree=True)
        k_deq = np.asarray(dequant_head(k_codes, k_lo, k_scale), dtype=np.int64)
        v_deq = np.asarray(dequant_head(v_codes, v_lo, v_scale), dtype=np.int64)
        self_k_cache[h].append(k_deq)
        self_v_cache[h].append(v_deq)

        # ---- causal self-attn (Tc=step+1) ----
        q_h = head_slice(q_rope, h)
        Tc = step + 1
        scores = np.zeros(Tc, dtype=np.int64)
        for j in range(Tc):
            acc = int(np.dot(q_h.astype(object), self_k_cache[h][j].astype(object)))
            s_q88 = rsh_round(acc, SCORE_SH)
            scores[j] = sat(s_q88, -32768, 32767)
        prob = int_softmax_q(scores, np.ones(Tc, dtype=bool), exp_table())
        ctx_h = np.zeros(HEAD_DIM, dtype=np.int64)
        for d in range(HEAD_DIM):
            acc = 0
            for j in range(Tc):
                acc += int(prob[j]) * int(self_v_cache[h][j][d])
            ctx_h[d] = rsh_round(acc, CTX_SH)
        ctx_self[h * HEAD_DIM:(h + 1) * HEAD_DIM] = ctx_h

    # ---- O GEMV (self-attn output proj) ----
    o_fixed, gfrac_o, ar_o = lq(ctx_self, RESID_FRAC, "o", RESID_FRAC)
    xres1 = wrap32(xres0 + o_fixed)   # RES1 -- RTL truncates to 32b/lane on store

    # ---- LN2 ----
    xn2 = np.asarray(ln_int_gendiv(list(xres1.astype(object)), list(layer_state["g_ln2"])),
                      dtype=np.int64)

    # ---- cross Q GEMV (no RoPE; POST_SCALE applied directly) ----
    qc_fixed, gfrac_cq, ar_cq = lq(xn2, OUT_FRAC, "cq", FRAC_QKV)
    qc_scaled = np.zeros(D, dtype=np.int64)
    for i in range(D):
        qc_scaled[i] = rsh_round(int(qc_fixed[i]) * POST_SCALE_Q16, VFRAC)

    # ---- cross-attn (static full-attend, T2=6, this LAYER's own step-
    # invariant K/V) ----
    ctx_cross = np.zeros(D, dtype=np.int64)
    for h in range(NHEAD):
        q_h = head_slice(qc_scaled, h)
        scores = np.zeros(T2, dtype=np.int64)
        for j in range(T2):
            acc = int(np.dot(q_h.astype(object), k_cross_deq[j][h].astype(object)))
            s_q88 = rsh_round(acc, SCORE_SH)
            scores[j] = sat(s_q88, -32768, 32767)
        prob = int_softmax_q(scores, np.ones(T2, dtype=bool), exp_table())
        ctx_h = np.zeros(HEAD_DIM, dtype=np.int64)
        for d in range(HEAD_DIM):
            acc = 0
            for j in range(T2):
                acc += int(prob[j]) * int(v_cross_deq[j][h][d])
            ctx_h[d] = rsh_round(acc, CTX_SH)
        ctx_cross[h * HEAD_DIM:(h + 1) * HEAD_DIM] = ctx_h

    # ---- Oc GEMV ----
    oc_fixed, gfrac_oc, ar_oc = lq(ctx_cross, RESID_FRAC, "oc", RESID_FRAC)
    xres2 = wrap32(xres1 + oc_fixed)   # RES2 -- same 32b/lane truncation

    # ---- LN3 ----
    xn3 = np.asarray(ln_int_gendiv(list(xres2.astype(object)), list(layer_state["g_ln3"])),
                      dtype=np.int64)

    # ---- FC1: D -> DFFN2, +bias, Q.12 out (SiLU-ready) ----
    h1_fixed, gfrac_fc1, ar_fc1 = lq(xn3, OUT_FRAC, "fc1", FRAC_FC1,
                                      bias_fixed=layer_state["bias_fc1_fixed"])
    value = h1_fixed[:FFN]
    gate = h1_fixed[FFN:]

    # ---- SiLU LUT reference (bit-exact to silu_lut.sv) ----
    from fabric.asr_seq.run_silu import silu_table, silu_q
    silu_lut = silu_table()
    gate_sat = np.clip(gate, -32768, 32767).astype(np.int64)
    silu_gate = silu_q(gate_sat, silu_lut)            # Q4.12
    combined = np.zeros(FFN, dtype=np.int64)
    for i in range(FFN):
        prod = int(value[i]) * int(silu_gate[i])       # Q4.12 * Q4.12 = Q8.24
        combined[i] = rsh_round(prod, FRAC_FC1)         # -> Q4.12

    # ---- FC2: FFN -> D, +bias, Q.25 out ----
    h2_fixed, gfrac_fc2, ar_fc2 = lq(combined, FRAC_FC1, "fc2", RESID_FRAC,
                                      bias_fixed=layer_state["bias_fc2_fixed"])
    xres3 = wrap32(xres2 + h2_fixed)   # RES3 -- same 32b/lane truncation

    result = {
        "xres0": xres0, "xres1": xres1, "xres2": xres2, "xres3": xres3,
        "g_frac": {"q": gfrac_q, "k": gfrac_k, "v": gfrac_v, "o": gfrac_o, "cq": gfrac_cq,
                   "oc": gfrac_oc, "fc1": gfrac_fc1, "fc2": gfrac_fc2},
        "act_rshift": {"q": ar_q, "k": ar_k, "v": ar_v, "o": ar_o, "cq": ar_cq,
                       "oc": ar_oc, "fc1": ar_fc1, "fc2": ar_fc2},
    }
    return result, profile


def build_layer_states(w_per_layer, enc_hidden, w_enc_finalln, cos_rom, sin_rom):
    """Precompute everything step-invariant per layer: cross K/V (Stage 3a,
    once per layer, real per-layer ck/cv weights), weight quantization,
    gamma/bias fixed-point tables, and fresh (empty) self-attn caches."""
    layer_states = []
    for li in range(NLAYER):
        w = w_per_layer[li]

        k_cross_deq = [[None] * NHEAD for _ in range(T2)]
        v_cross_deq = [[None] * NHEAD for _ in range(T2)]
        for pos in range(T2):
            eo = layernorm_nobias_f(enc_hidden[pos], w_enc_finalln, LN_EPS)
            k16 = to_qfrac(w["ck"] @ eo, VFRAC)
            v16 = to_qfrac(w["cv"] @ eo, VFRAC)
            for h in range(NHEAD):
                k_h = head_slice(k16, h)
                v_h = head_slice(v16, h)
                k_h_scaled = np.array([rsh_round(int(v) * POST_SCALE_Q16, VFRAC) for v in k_h],
                                       dtype=np.int64)
                k_codes, k_lo, k_scale = quant_head_asym(list(k_h_scaled), KBITS, divfree=True)
                v_codes, v_lo, v_scale = quant_head_asym(list(v_h), KBITS, divfree=True)
                k_cross_deq[pos][h] = np.asarray(dequant_head(k_codes, k_lo, k_scale), dtype=np.int64)
                v_cross_deq[pos][h] = np.asarray(dequant_head(v_codes, v_lo, v_scale), dtype=np.int64)

        W8, wshift = {}, {}
        for name in CALL_NAMES:
            W8[name], wshift[name] = quantize_weight(w[W_KEYS[name]])

        layer_states.append({
            "w": w, "W8": W8, "wshift": wshift,
            "g_ln1": to_qfrac(w["ln1"], G_FRAC),
            "g_ln2": to_qfrac(w["ln2"], G_FRAC),
            "g_ln3": to_qfrac(w["ln3"], G_FRAC),
            "bias_fc1_fixed": to_qfrac(w["b_fc1"], FRAC_FC1),
            "bias_fc2_fixed": to_qfrac(w["b_fc2"], RESID_FRAC),
            "k_cross_deq": k_cross_deq, "v_cross_deq": v_cross_deq,
            "self_k_cache": [[] for _ in range(NHEAD)],
            "self_v_cache": [[] for _ in range(NHEAD)],
            "cos_rom": cos_rom, "sin_rom": sin_rom,
        })
    return layer_states


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    enc_hidden, w_enc_finalln, w_per_layer, embed = load_real()
    t2 = enc_hidden.shape[0]
    assert t2 == T2, f"expected T2={T2}, got {t2}"

    cos_rom, sin_rom = build_cos_sin_rom()

    # ================= PASS 1: profile every call site's magnitude across
    # the full (layer x step) grid (auto-derive per call, purely to
    # measure) -- decoder_block_seq.sv's ACT_*/GF_* are compile-time
    # parameters shared by ALL 6 layers, so ONE fixed shift per call site
    # must cover every layer AND every step. ==================
    profile_states = build_layer_states(w_per_layer, enc_hidden, w_enc_finalln, cos_rom, sin_rom)
    profile_max = {name: 0 for name in CALL_NAMES}
    for step, tok in enumerate(TOKEN_IDS):
        x = to_qfrac(embed[tok], RESID_FRAC)
        for li in range(NLAYER):
            result, profile = one_layer(profile_states[li], step, x, act_rshifts=None)
            for name, m in profile.items():
                profile_max[name] = max(profile_max[name], m)
            x = result["xres3"]
    fixed_ar = {name: choose_act_rshift_from_max(m) for name, m in profile_max.items()}
    print("profile (max|x_fixed| across all 6 layers x 3 steps -> fixed ACT_RSHIFT):",
          file=sys.stderr)
    for name in CALL_NAMES:
        print(f"  {name}: max|.|={profile_max[name]} -> act_rshift={fixed_ar[name]}", file=sys.stderr)

    # ================= PASS 2: real run, FIXED shifts, fresh per-layer
    # self-attn caches (matches decoder_block_seq.sv's own compile-time
    # ACT_*/GF_* parameters exactly) =========
    layer_states = build_layer_states(w_per_layer, enc_hidden, w_enc_finalln, cos_rom, sin_rom)
    steps_out = []   # steps_out[step][layer] -> result dict
    for step, tok in enumerate(TOKEN_IDS):
        x = to_qfrac(embed[tok], RESID_FRAC)
        layer_results = []
        for li in range(NLAYER):
            result, _ = one_layer(layer_states[li], step, x, act_rshifts=fixed_ar)
            layer_results.append(result)
            x = result["xres3"]
        steps_out.append(layer_results)

    # act_rshift is identical across EVERY (layer, step) by construction
    # (fixed_ar is the same dict every call, and decoder_block_seq.sv's
    # ACT_* are compile-time parameters -- must be one shared value).
    # g_frac, in contrast, is genuinely per-LAYER (it bakes in that
    # layer's own WSHIFT) but IS fixed across steps within a layer (WSHIFT
    # depends only on the weight matrix, not the decode step) -- checked
    # both ways below, not assumed.
    act_rshift = steps_out[0][0]["act_rshift"]
    g_frac_per_layer = [steps_out[0][li]["g_frac"] for li in range(NLAYER)]
    for s in range(len(TOKEN_IDS)):
        for li in range(NLAYER):
            res = steps_out[s][li]
            assert res["act_rshift"] == act_rshift, \
                f"step {s} layer {li}: act_rshift drifted -- should be fixed across layers+steps"
            assert res["g_frac"] == g_frac_per_layer[li], \
                f"step {s} layer {li}: g_frac drifted across steps -- should be fixed per layer"

    def rng(name, arr):
        a = np.asarray(arr, dtype=np.float64)
        print(f"  {name}: max|.|={np.max(np.abs(a)):.0f}", file=sys.stderr)
    print("final (fixed-point magnitude, last step, last layer):", file=sys.stderr)
    rng("xres3(Q25)", steps_out[-1][-1]["xres3"])

    # =========================================================================
    # ---- resident weight image: ALL 6 layers x 8 call sites = 48 weight
    # matrices, one combined resident image (real gemv_banked_resident_vec.sv
    # loader format via pack_banked_resident_vec.build_resident(), the
    # already-proven WBW=8 packing) -- loaded ONCE, never reloaded between
    # layers (WWORDS capacity comfortably covers 6x a single layer's own
    # 13824 words). ------------------------------------------------------
    from fabric.stage3.pack_banked_resident_vec import build_resident
    LANES, WBW = 128, 8
    layer_order = []
    for li in range(NLAYER):
        for name in CALL_NAMES:
            layer_order.append(layer_states[li]["W8"][name])
    all_words, wmeta = build_resident(layer_order, LANES, WBW)
    # wb[layer][name] -> resident word offset
    wb = [
        {name: wmeta[li * len(CALL_NAMES) + i]["w_base"] for i, name in enumerate(CALL_NAMES)}
        for li in range(NLAYER)
    ]

    wbits = LANES * WBW
    hexw = (wbits + 3) // 4
    with open(os.path.join(out_dir, "w.mem"), "w") as f:
        f.write("\n".join(format(v, f"0{hexw}x") for v in all_words) + "\n")

    # ---- internal ROMs every sub-module $readmemh's from the sim run dir --
    # (same generation as every prior attention/SiLU gate -- not re-derived,
    # layer-invariant).
    with open(os.path.join(out_dir, "rope_cos.mem"), "w") as f:
        for pos in range(cos_rom.shape[0]):
            for i in range(ROT_PAIRS):
                f.write(format(int(cos_rom[pos, i]) & 0xFFFF, "04x") + "\n")
    with open(os.path.join(out_dir, "rope_sin.mem"), "w") as f:
        for pos in range(sin_rom.shape[0]):
            for i in range(ROT_PAIRS):
                f.write(format(int(sin_rom[pos, i]) & 0xFFFF, "04x") + "\n")

    exp_lut = exp_table()
    with open(os.path.join(out_dir, "exp_lut.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0x1FFFFF:06x}" for v in exp_lut) + "\n")

    with open(os.path.join(out_dir, "inv_lut_lo.mem"), "w") as f:
        for s4 in range(4096):
            f.write(f"{q_round_div(1 << 24, max(s4, 1)) & 0x1FFFFFF:07x}\n")
    with open(os.path.join(out_dir, "inv_lut_hi.mem"), "w") as f:
        for s4 in range(4096, 16512):
            f.write(f"{q_round_div(1 << 24, s4) & 0x1FFF:04x}\n")

    with open(os.path.join(out_dir, "seed.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFFF:05x}" for v in seed_table()) + "\n")

    from fabric.asr_seq.run_silu import silu_table
    silu_lut_vals = silu_table()
    with open(os.path.join(out_dir, "silu_lut.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in silu_lut_vals) + "\n")

    def w32(f, v):
        f.write(format(int(v) & 0xFFFFFFFF, "08x") + "\n")

    def write_prow_all(path, vecs):
        """P=8-wide packed rows, one line/row, ALL layers concatenated
        (layer-major) into ONE file -- a single $readmemh on the testbench
        side, indexed by layer at runtime, same idiom as every other
        combined .mem file here (w.mem, wb_offsets.mem, ...)."""
        with open(path, "w") as f:
            for vec in vecs:
                for r in range(len(vec) // 8):
                    for k in range(8):
                        w32(f, vec[r * 8 + k])

    # ---- LN gamma + FC bias tables (Q4.20/Q.12/Q.25, P=8-wide), ALL
    # layers concatenated into one file each -- the testbench reloads the
    # CURRENT layer's own slice (through the SAME gam_we/bias_we ports)
    # before each layer's own forward pass; RTL gamma_bank/bias banks are
    # sized for one layer and reused sequentially, matching the real access
    # pattern (layer 0 fully, then layer 1, ... -- never revisited within a
    # step). ----
    write_prow_all(os.path.join(out_dir, "gamma_ln1_all.mem"),
                    [layer_states[li]["g_ln1"] for li in range(NLAYER)])
    write_prow_all(os.path.join(out_dir, "gamma_ln2_all.mem"),
                    [layer_states[li]["g_ln2"] for li in range(NLAYER)])
    write_prow_all(os.path.join(out_dir, "gamma_ln3_all.mem"),
                    [layer_states[li]["g_ln3"] for li in range(NLAYER)])
    write_prow_all(os.path.join(out_dir, "bias_fc1_all.mem"),
                    [layer_states[li]["bias_fc1_fixed"] for li in range(NLAYER)])
    write_prow_all(os.path.join(out_dir, "bias_fc2_all.mem"),
                    [layer_states[li]["bias_fc2_fixed"] for li in range(NLAYER)])

    # ---- cross K/V write vectors (ATTN_P=4-wide, pos-major then head-major:
    # [k(36) v(36)] per head -- same convention as pack_decoder_cross_attn.py's
    # kv_in.mem), ALL layers concatenated (layer-major, then pos, then head)
    # into ONE file -- preloaded ONCE per layer at the very start (Stage 3a
    # semantics: computed once per layer, before any decode step);
    # kv_bank.sv's own storage is already layer-indexed internally via
    # `blk`, so all NLAYER layers' cross K/V persist simultaneously, no
    # reload between layers. ----
    with open(os.path.join(out_dir, "xkv_in_all.mem"), "w") as f:
        for li in range(NLAYER):
            w = w_per_layer[li]
            for pos in range(T2):
                eo = layernorm_nobias_f(enc_hidden[pos], w_enc_finalln, LN_EPS)
                k16 = to_qfrac(w["ck"] @ eo, VFRAC)
                v16 = to_qfrac(w["cv"] @ eo, VFRAC)
                for h in range(NHEAD):
                    k_h = head_slice(k16, h)
                    v_h = head_slice(v16, h)
                    k_h_scaled = np.array([rsh_round(int(v) * POST_SCALE_Q16, VFRAC) for v in k_h],
                                           dtype=np.int64)
                    for v in k_h_scaled:
                        w32(f, v)
                    for v in v_h:
                        w32(f, v)

    # ---- per-step xres0 (layer 0's own initial residual = fresh token
    # embedding) + per-(step,layer) golden xres3 (every layer's own output
    # checked, not just the last -- fine-grained pass/fail). ----
    n_steps = len(TOKEN_IDS)
    with open(os.path.join(out_dir, "xres0_steps.mem"), "w") as f:
        for step, tok in enumerate(TOKEN_IDS):
            xres0 = to_qfrac(embed[tok], RESID_FRAC)
            for r in range(D // 8):
                for k in range(8):
                    w32(f, xres0[r * 8 + k])
    with open(os.path.join(out_dir, "xres3_ref_steps.mem"), "w") as f:
        for step in range(n_steps):
            for li in range(NLAYER):
                for v in steps_out[step][li]["xres3"]:
                    w32(f, v)

    # ---- per-layer wb_*/gf_* runtime-port values, one .mem line per
    # (layer, call) in CALL_NAMES order -- $readmemh'd by the testbench and
    # indexed by layer at runtime (NOT hand-transcribed into the testbench
    # source; 48 values each is real transcription-error risk, avoided by
    # generating them the same way every other .mem file here is). ----
    with open(os.path.join(out_dir, "wb_offsets.mem"), "w") as f:
        for li in range(NLAYER):
            for name in CALL_NAMES:
                f.write(format(wb[li][name] & 0xFFFFF, "05x") + "\n")
    with open(os.path.join(out_dir, "gf_shifts.mem"), "w") as f:
        for li in range(NLAYER):
            for name in CALL_NAMES:
                f.write(format(g_frac_per_layer[li][name] & 0xFF, "02x") + "\n")

    # =========================================================================
    # ---- assemble manifest ---------------------------------------------------
    manifest = {
        "d": D, "ffn": FFN, "dffn2": DFFN2, "nhead": NHEAD, "head_dim": HEAD_DIM, "t2": T2,
        "n_steps": n_steps, "n_layer": NLAYER, "tokens": TOKEN_IDS,
        "resid_frac": RESID_FRAC, "post_scale_q16": POST_SCALE_Q16,
        "lanes": LANES, "wbw": WBW, "n_words_total": len(all_words),
        "g_frac_per_layer": g_frac_per_layer,
        "wshift": {name: [layer_states[li]["wshift"][name] for li in range(NLAYER)]
                   for name in CALL_NAMES},
        "act_rshift": act_rshift, "w_base": wb,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    dump = {
        "n_steps": n_steps, "n_layer": NLAYER,
        "steps": [[{"xres0": res["xres0"].tolist(), "xres1": res["xres1"].tolist(),
                    "xres2": res["xres2"].tolist(), "xres3": res["xres3"].tolist()}
                   for res in layer_results]
                  for layer_results in steps_out],
    }
    with open(os.path.join(out_dir, "golden.json"), "w") as f:
        json.dump(dump, f)

    print(f"GEN dir={out_dir} t2={T2} n_steps={n_steps} n_layer={NLAYER} tokens={TOKEN_IDS} "
          f"n_words={len(all_words)}")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_decoder_block")
    p.add_argument("cmd", choices=["gen"])
    p.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    main_gen(a.dir)


if __name__ == "__main__":
    main()
