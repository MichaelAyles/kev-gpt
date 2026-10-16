"""Golden reference + RTL test vectors for encoder_block_seq.sv -- the
MULTI-LAYER functional gate for the real, sized encoder-block top-level
FSM. Runs ALL 6 real moonshine-tiny encoder layers, chained (layer L's own
per-position OUTPUT becomes layer L+1's own per-position INPUT), each
layer processing ALL T2=6 real positions in one `go` pulse (the encoder is
NOT autoregressive -- every position is known up front, no step/decode
loop) -- a real, full multi-layer encoder forward pass, not a single-layer
or single-position toy.

Reuses pack_encoder_self_attn.py's OWN "static full-attend" access pattern
(write all T2 positions' own K/V once, then every one of the T2 query
positions attends the full, unchanging set -- bidirectional, no causal
masking) and pack_decoder_block.py's OWN two-pass profiling + wrap32()
technique for real multi-layer quantization (see that file's header for
the full rationale -- WSHIFT is genuinely per-layer, ACT_RSHIFT is not;
xres_bank is 32 bits/lane and the RTL truncates on every store, so the
Python reference needs the SAME explicit wraparound or it silently
diverges once magnitudes cross 2^31, which they do by layer 1 here same
as on the decoder side).

Real structural differences from pack_decoder_block.py, matching
encoder_block_seq.sv's own real differences from decoder_block_seq.sv:
no cross-attention, only 2 LayerNorms/layer, bidirectional self-attention
(tcount=T2 for every query, always), plain GELU MLP (fc1->GELU->fc2, no
SwiGLU gate*value), and every layer processes ALL T2 positions per `go`
(not one token per `go` like the decoder).

Usage:
    .venv/bin/python -m fabric.asr_seq.pack_encoder_block gen --dir <sim_dir>
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
from fabric.stage3.run_gelu import gelu_table, gelu_q  # noqa: E402

D = 288
FFN = 1152
NHEAD = 8
HEAD_DIM = 36
ATTN_P = 4
HR_ATTN = HEAD_DIM // ATTN_P            # 9
NLAYER = 6                              # real moonshine-tiny encoder depth
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
AUDIO_LEN = 3000            # -> T2=6, same as every other ASR gate in this repo

RESID_FRAC = 25             # xres_bank's Q6.25
FRAC_QKV = VFRAC             # 16
FRAC_FC1 = 12
CALL_NAMES = ["q", "k", "v", "o", "fc1", "fc2"]
W_KEYS = {"q": "sq", "k": "sk", "v": "sv", "o": "so", "fc1": "fc1", "fc2": "fc2"}


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


def quantize_weight(W: np.ndarray):
    max_abs = float(np.max(np.abs(W)))
    wshift = int(math.floor(math.log2(127.0 / max_abs)))
    while True:
        W_int8 = np.clip(np.round(W * (2.0 ** wshift)), -128, 127).astype(np.int64)
        if int(np.max(np.abs(W_int8))) <= 127:
            break
        wshift -= 1
    return W_int8, wshift


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


def gemv_int(W_int8: np.ndarray, x_int8: np.ndarray) -> np.ndarray:
    return (W_int8.astype(object) @ x_int8.astype(object)).astype(np.int64)


def gdequant(raw: np.ndarray, g_frac: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=object)
    if g_frac >= 0:
        return np.asarray([int(v) >> g_frac for v in raw], dtype=np.int64)
    return np.asarray([int(v) << (-g_frac) for v in raw], dtype=np.int64)


def wrap32(x: np.ndarray) -> np.ndarray:
    """Same real necessity as pack_decoder_block.py's own wrap32 -- see
    that file's header. xres_bank is 32 bits/lane, RTL truncates on every
    store, Python's arbitrary-precision ints don't -- must be applied
    explicitly after every residual add or the reference silently
    diverges once the accumulated stream crosses 2^31 (confirmed here
    too, empirically, not assumed)."""
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


def load_real():
    """Returns (hidden_states[T2,D] real FP32, w_per_layer (list of NLAYER
    dicts)) -- the SAME real conv-front-end input pack_encoder_self_attn.py
    already established (torch.manual_seed(0), L=3000), and ALL 6 real
    encoder layers' weights, not just layer 0."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained("UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    enc = model.encoder
    assert len(enc.layers) == NLAYER, f"expected NLAYER={NLAYER}, got {len(enc.layers)}"

    torch.manual_seed(AUDIO_SEED)
    audio = torch.randn(1, AUDIO_LEN, dtype=torch.float32)
    with torch.no_grad():
        h = F.tanh(enc.conv1(audio.unsqueeze(1)))
        h = enc.groupnorm(h)
        h = F.gelu(enc.conv2(h))
        h = F.gelu(enc.conv3(h))
        h = h.permute(0, 2, 1)
    hidden_states = h[0].detach().numpy().astype(np.float64)   # (T2, D)

    w_per_layer = []
    for li in range(NLAYER):
        layer = enc.layers[li]
        w = {}
        w["ln1"] = layer.input_layernorm.weight.detach().numpy().astype(np.float64)
        w["sq"] = layer.self_attn.q_proj.weight.detach().numpy().astype(np.float64)
        w["sk"] = layer.self_attn.k_proj.weight.detach().numpy().astype(np.float64)
        w["sv"] = layer.self_attn.v_proj.weight.detach().numpy().astype(np.float64)
        w["so"] = layer.self_attn.o_proj.weight.detach().numpy().astype(np.float64)
        w["ln2"] = layer.post_attention_layernorm.weight.detach().numpy().astype(np.float64)
        w["fc1"] = layer.mlp.fc1.weight.detach().numpy().astype(np.float64)
        w["b_fc1"] = layer.mlp.fc1.bias.detach().numpy().astype(np.float64)
        w["fc2"] = layer.mlp.fc2.weight.detach().numpy().astype(np.float64)
        w["b_fc2"] = layer.mlp.fc2.bias.detach().numpy().astype(np.float64)
        w_per_layer.append(w)

    return hidden_states, w_per_layer


def head_slice(vec_d, h):
    return vec_d[h * HEAD_DIM:(h + 1) * HEAD_DIM]


def one_layer(layer_state, xres_in, cos_rom, sin_rom, act_rshifts):
    """One encoder LAYER's forward pass over ALL T2 positions.
    `xres_in`: (T2, D) int64 -- this layer's own residual INPUT per
    position (layer 0's own fresh conv-front-end embedding, or the
    PREVIOUS layer's own per-position xres_out). `act_rshifts`: dict
    name->shift; None on the profiling pass (auto-derives per call from
    this layer's own observed magnitude across all T2 positions), fixed
    dict on the real pass. Returns (xres_out (T2,D), g_frac dict,
    act_rshift dict, profile dict)."""
    w = layer_state["w"]
    W8 = layer_state["W8"]
    wshift = layer_state["wshift"]
    T2 = xres_in.shape[0]

    profile = {name: 0 for name in CALL_NAMES}
    g_frac_out = {}
    ar_out = {}

    def lq(x_fixed, frac_in, name, frac_out, bias_fixed=None):
        m = int(np.max(np.abs(np.asarray(x_fixed, dtype=np.int64))))
        profile[name] = max(profile[name], m)
        ar = act_rshifts[name] if act_rshifts is not None else choose_act_rshift_from_max(profile[name])
        ar_out[name] = ar
        x_int8 = act_quantize(x_fixed, ar)
        raw = gemv_int(W8[name], x_int8)
        g_frac = frac_in - ar + wshift[name] - frac_out
        g_frac_out[name] = g_frac
        y = gdequant(raw, g_frac)
        if bias_fixed is not None:
            y = y + np.asarray(bias_fixed, dtype=np.int64)
        return y

    # ================= PHASE A: for pos=0..T2-1, LN1->QKV->RoPE->KV-write ===
    xn1_all = np.zeros((T2, D), dtype=np.int64)
    q_rope_all = np.zeros((T2, D), dtype=np.int64)
    self_k_cache = [[] for _ in range(NHEAD)]   # [h] -> list of (pos-ordered) dequant K rows
    self_v_cache = [[] for _ in range(NHEAD)]
    for pos in range(T2):
        xn1 = np.asarray(ln_int_gendiv(list(xres_in[pos].astype(object)), list(layer_state["g_ln1"])),
                          dtype=np.int64)
        xn1_all[pos] = xn1
        q_fixed = lq(xn1, OUT_FRAC, "q", FRAC_QKV)
        k_fixed = lq(xn1, OUT_FRAC, "k", FRAC_QKV)
        v_fixed = lq(xn1, OUT_FRAC, "v", FRAC_QKV)

        for h in range(NHEAD):
            q_rope_all[pos, h * HEAD_DIM:(h + 1) * HEAD_DIM] = rope_apply_ref(
                head_slice(q_fixed, h), pos, cos_rom, sin_rom, POST_SCALE_Q16)
            k_rope = rope_apply_ref(head_slice(k_fixed, h), pos, cos_rom, sin_rom, POST_SCALE_Q16)
            v_h = head_slice(v_fixed, h)
            k_codes, k_lo, k_scale = quant_head_asym(list(k_rope), KBITS, divfree=True)
            v_codes, v_lo, v_scale = quant_head_asym(list(v_h), KBITS, divfree=True)
            self_k_cache[h].append(np.asarray(dequant_head(k_codes, k_lo, k_scale), dtype=np.int64))
            self_v_cache[h].append(np.asarray(dequant_head(v_codes, v_lo, v_scale), dtype=np.int64))

    # ================= PHASE B: for pos=0..T2-1, attend + MLP ===============
    xres_out = np.zeros((T2, D), dtype=np.int64)
    for pos in range(T2):
        # ---- self-attn (bidirectional, tcount=T2 always) ----
        ctx_self = np.zeros(D, dtype=np.int64)
        for h in range(NHEAD):
            q_h = head_slice(q_rope_all[pos], h)
            scores = np.zeros(T2, dtype=np.int64)
            for j in range(T2):
                acc = int(np.dot(q_h.astype(object), self_k_cache[h][j].astype(object)))
                s_q88 = rsh_round(acc, SCORE_SH)
                scores[j] = sat(s_q88, -32768, 32767)
            prob = int_softmax_q(scores, np.ones(T2, dtype=bool), exp_table())
            ctx_h = np.zeros(HEAD_DIM, dtype=np.int64)
            for d in range(HEAD_DIM):
                acc = 0
                for j in range(T2):
                    acc += int(prob[j]) * int(self_v_cache[h][j][d])
                ctx_h[d] = rsh_round(acc, CTX_SH)
            ctx_self[h * HEAD_DIM:(h + 1) * HEAD_DIM] = ctx_h

        o_fixed = lq(ctx_self, RESID_FRAC, "o", RESID_FRAC)
        xres1 = wrap32(xres_in[pos] + o_fixed)   # RES1

        xn2 = np.asarray(ln_int_gendiv(list(xres1.astype(object)), list(layer_state["g_ln2"])),
                          dtype=np.int64)

        h1_fixed = lq(xn2, OUT_FRAC, "fc1", FRAC_FC1, bias_fixed=layer_state["bias_fc1_fixed"])
        h1_sat = np.clip(h1_fixed, -32768, 32767).astype(np.int64)
        g1 = gelu_q(h1_sat, layer_state["gelu_lut"])          # Q4.12

        h2_fixed = lq(g1, FRAC_FC1, "fc2", RESID_FRAC, bias_fixed=layer_state["bias_fc2_fixed"])
        xres_out[pos] = wrap32(xres1 + h2_fixed)   # RES2

    return xres_out, g_frac_out, ar_out, profile


def build_layer_states(w_per_layer, gelu_lut):
    layer_states = []
    for li in range(NLAYER):
        w = w_per_layer[li]
        W8, wshift = {}, {}
        for name in CALL_NAMES:
            W8[name], wshift[name] = quantize_weight(w[W_KEYS[name]])
        layer_states.append({
            "w": w, "W8": W8, "wshift": wshift,
            "g_ln1": to_qfrac(w["ln1"], G_FRAC),
            "g_ln2": to_qfrac(w["ln2"], G_FRAC),
            "bias_fc1_fixed": to_qfrac(w["b_fc1"], FRAC_FC1),
            "bias_fc2_fixed": to_qfrac(w["b_fc2"], RESID_FRAC),
            "gelu_lut": gelu_lut,
        })
    return layer_states


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    hidden_states, w_per_layer = load_real()
    T2 = hidden_states.shape[0]

    cos_rom, sin_rom = build_cos_sin_rom()
    gelu_lut = gelu_table()

    # ================= PASS 1: profile every call site's magnitude across
    # the full (layer x position) grid -- ACT_* are compile-time parameters
    # shared by ALL 6 layers AND all T2 positions. ==================
    profile_states = build_layer_states(w_per_layer, gelu_lut)
    profile_max = {name: 0 for name in CALL_NAMES}
    x = to_qfrac(hidden_states, RESID_FRAC)   # (T2, D)
    for li in range(NLAYER):
        x, _, _, profile = one_layer(profile_states[li], x, cos_rom, sin_rom, act_rshifts=None)
        for name, m in profile.items():
            profile_max[name] = max(profile_max[name], m)
    fixed_ar = {name: choose_act_rshift_from_max(m) for name, m in profile_max.items()}
    print("profile (max|x_fixed| across all 6 layers x 6 positions -> fixed ACT_RSHIFT):",
          file=sys.stderr)
    for name in CALL_NAMES:
        print(f"  {name}: max|.|={profile_max[name]} -> act_rshift={fixed_ar[name]}", file=sys.stderr)

    # ================= PASS 2: real run, FIXED shifts ========================
    layer_states = build_layer_states(w_per_layer, gelu_lut)
    x = to_qfrac(hidden_states, RESID_FRAC)
    layer_outs = []   # layer_outs[li] -> (T2, D) xres_out
    g_frac_per_layer = []
    for li in range(NLAYER):
        x, g_frac, ar, _ = one_layer(layer_states[li], x, cos_rom, sin_rom, act_rshifts=fixed_ar)
        layer_outs.append(x)
        g_frac_per_layer.append(g_frac)
        assert ar == fixed_ar, f"layer {li}: act_rshift drifted -- should be fixed everywhere"

    def rng(name, arr):
        a = np.asarray(arr, dtype=np.float64)
        print(f"  {name}: max|.|={np.max(np.abs(a)):.0f}", file=sys.stderr)
    print("final (fixed-point magnitude, last layer):", file=sys.stderr)
    rng("xres_out(Q25)", layer_outs[-1])

    # =========================================================================
    from fabric.stage3.pack_banked_resident_vec import build_resident
    LANES, WBW = 128, 8
    layer_order = []
    for li in range(NLAYER):
        for name in CALL_NAMES:
            layer_order.append(layer_states[li]["W8"][name])
    all_words, wmeta = build_resident(layer_order, LANES, WBW)
    wb = [
        {name: wmeta[li * len(CALL_NAMES) + i]["w_base"] for i, name in enumerate(CALL_NAMES)}
        for li in range(NLAYER)
    ]

    wbits = LANES * WBW
    hexw = (wbits + 3) // 4
    with open(os.path.join(out_dir, "w.mem"), "w") as f:
        f.write("\n".join(format(v, f"0{hexw}x") for v in all_words) + "\n")

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

    # gelu_lut2.sv's OWN documented split: lut_e[i]=lut[2i], lut_o[i]=lut[2i+1]
    # (NOT the single gelu_lut.mem run_vec_gelu.py writes -- vec_gelu.sv
    # instantiates gelu_lut2, the BRAM-shared even/odd-banked core, not
    # gelu_lut directly; found by reading vec_gelu.sv's own source, not
    # assumed from run_vec_gelu.py, which turned out to compile the wrong
    # module for its own current vec_gelu.sv -- a pre-existing issue, out
    # of scope here).
    with open(os.path.join(out_dir, "gelu_lut_e.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in gelu_lut[0::2]) + "\n")
    with open(os.path.join(out_dir, "gelu_lut_o.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in gelu_lut[1::2]) + "\n")

    def w32(f, v):
        f.write(format(int(v) & 0xFFFFFFFF, "08x") + "\n")

    def write_prow_all(path, vecs):
        with open(path, "w") as f:
            for vec in vecs:
                for r in range(len(vec) // 8):
                    for k in range(8):
                        w32(f, vec[r * 8 + k])

    write_prow_all(os.path.join(out_dir, "gamma_ln1_all.mem"),
                    [layer_states[li]["g_ln1"] for li in range(NLAYER)])
    write_prow_all(os.path.join(out_dir, "gamma_ln2_all.mem"),
                    [layer_states[li]["g_ln2"] for li in range(NLAYER)])
    write_prow_all(os.path.join(out_dir, "bias_fc1_all.mem"),
                    [layer_states[li]["bias_fc1_fixed"] for li in range(NLAYER)])
    write_prow_all(os.path.join(out_dir, "bias_fc2_all.mem"),
                    [layer_states[li]["bias_fc2_fixed"] for li in range(NLAYER)])

    # ---- per-layer wb_*/gf_* runtime-port values, one .mem line per
    # (layer, call) in CALL_NAMES order -- see pack_decoder_block.py's own
    # identical export for why (avoids hand-transcribing 36 values). ----
    with open(os.path.join(out_dir, "wb_offsets.mem"), "w") as f:
        for li in range(NLAYER):
            for name in CALL_NAMES:
                f.write(format(wb[li][name] & 0xFFFFF, "05x") + "\n")
    with open(os.path.join(out_dir, "gf_shifts.mem"), "w") as f:
        for li in range(NLAYER):
            for name in CALL_NAMES:
                f.write(format(g_frac_per_layer[li][name] & 0xFF, "02x") + "\n")

    # ---- xres_in (layer 0's own real per-position input) + per-layer
    # golden xres_out (every layer's own per-position output checked). ----
    xres0 = to_qfrac(hidden_states, RESID_FRAC)   # (T2, D)
    with open(os.path.join(out_dir, "xres_in.mem"), "w") as f:
        for pos in range(T2):
            for r in range(D // 8):
                for k in range(8):
                    w32(f, xres0[pos, r * 8 + k])
    with open(os.path.join(out_dir, "xres_out_ref.mem"), "w") as f:
        for li in range(NLAYER):
            for pos in range(T2):
                for v in layer_outs[li][pos]:
                    w32(f, v)

    manifest = {
        "d": D, "ffn": FFN, "nhead": NHEAD, "head_dim": HEAD_DIM, "t2": T2,
        "n_layer": NLAYER, "resid_frac": RESID_FRAC, "post_scale_q16": POST_SCALE_Q16,
        "lanes": LANES, "wbw": WBW, "n_words_total": len(all_words),
        "g_frac_per_layer": g_frac_per_layer,
        "wshift": {name: [layer_states[li]["wshift"][name] for li in range(NLAYER)]
                   for name in CALL_NAMES},
        "act_rshift": fixed_ar, "w_base": wb,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    dump = {"n_layer": NLAYER, "t2": T2,
            "layers": [layer_outs[li].tolist() for li in range(NLAYER)]}
    with open(os.path.join(out_dir, "golden.json"), "w") as f:
        json.dump(dump, f)

    print(f"GEN dir={out_dir} t2={T2} n_layer={NLAYER} n_words={len(all_words)}")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_encoder_block")
    p.add_argument("cmd", choices=["gen"])
    p.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    main_gen(a.dir)


if __name__ == "__main__":
    main()
