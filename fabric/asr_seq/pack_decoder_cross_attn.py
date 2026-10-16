"""Golden reference + RTL test vectors for the decoder CROSS-attention block
gate: the RoPE-free simpler case of the same "static full-attend" access
pattern the encoder self-attention gate proved needs zero new storage
RTL -- does checkpoint C's real, UNMODIFIED kv_bank.sv + vec_attn_w.sv
correctly compute it too, closing the "not separately gated" honest gap
left by pack_encoder_self_attn.py?

Per MoonshineAttention.forward's own `if not is_cross_attention:` guard,
RoPE is skipped ENTIRELY for cross-attention -- neither Q nor K gets
apply_rotary_pos_emb. So this gate has NO rope_apply_vec.sv in its RTL
chain at all: just kv_bank.sv (write the encoder's cross K/V ONCE, read it
with a FIXED tcount every decode step) + vec_attn_w.sv, unmodified.

vec_attn_w.sv's SCORE_SH=27 mismatch (hardcodes 1/sqrt(HEAD_DIM)=1/8 for
checkpoint C's HEAD_DIM=64; ASR's HEAD_DIM=36 needs 1/6, not a power of
2) still applies regardless of RoPE -- it's purely a function of
HEAD_DIM, not of whether rotation happens. With no RoPE stage available
to fold the sqrt(4/3) correction into this time, it's applied directly in
this reference (and fed pre-scaled into the RTL testbench's kv_bank/
vec_attn_w inputs) using rope_apply_vec.sv's OWN pass-through-lane
formula (`rsh_round(v * POST_SCALE_Q16, 16)`, identical to
pack_rope.rope_apply_ref's un-rotated-dim path) -- Q/K/V generation
(including this upstream scale) is out of scope for this gate, same as
LayerNorm/GEMV in both prior attention gates.

Data, deliberately reusing prior gates' real sources rather than
re-deriving new ones:
  - Cross K/V: the SAME real conv-front-end output as
    pack_encoder_self_attn.py (real conv1/conv2/conv3/groupnorm weights,
    same fixed-seed waveform, T2=6), run through the encoder's own real
    final `layer_norm` (a stand-in for the true 6-layer encoder's real
    final output -- out of scope here, same simplification the encoder
    self-attn gate made for its own input), then decoder layer 0's real
    `encoder_attn.k_proj`/`v_proj` -- the true Stage 3a op
    ("cross_k/cross_v computed once, never recomputed, never grows").
  - Query: the SAME real decoder token embeddings + TOKEN_IDS as
    pack_decoder_self_attn.py, through the real
    `post_attention_layernorm` + `encoder_attn.q_proj` -- one query row
    per decode step, each attending the SAME fixed T2=6-row cross K/V set
    (tcount=T2 always, never growing, unlike the decoder's own
    self-attention K/V cache).

Reuses this project's proven fixed-point references verbatim (same as
both prior attention gates): fabric.stage3.seq_ref.rsh_round/sat/
q_round_div, fabric.stage3.run_softmax.int_softmax_q/exp_table,
model.goformer_kvq.quant_head_asym/dequant_head.

Usage:
    .venv/bin/python -m fabric.asr_seq.pack_decoder_cross_attn gen --dir <sim_dir>
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
from model.goformer_kvq import quant_head_asym, dequant_head, INV_SH as KVQ_INV_SH  # noqa: E402

D = 288
HEAD_DIM = 36
NHEAD = 8
LN_EPS = 1e-5
VFRAC = 16
Q16 = 1 << VFRAC
ISQRT = 3
SCORE_FRAC = 8
PROB_FRAC = 20
SCORE_SH = 2 * VFRAC + ISQRT - SCORE_FRAC   # 27, vec_attn_w.sv's own localparam
CTX_SH = PROB_FRAC + VFRAC - 25             # 11, vec_attn_w.sv's own localparam
KBITS = 8
POST_SCALE_Q16 = round(math.sqrt(4.0 / 3.0) * Q16)   # 75674, same HEAD_DIM=36 correction

AUDIO_SEED = 0
AUDIO_LEN = 3000   # -> T2=6, same conv-front-end input as pack_encoder_self_attn.py
TOKEN_IDS = [1, 940, 24936]   # same real decoder tokens as pack_decoder_self_attn.py


def layernorm_nobias(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / np.sqrt(var + eps) * gamma


def to_q16(x_real) -> np.ndarray:
    return np.round(np.asarray(x_real, dtype=np.float64) * Q16).astype(np.int64)


def post_scale_q16(v_q16: np.ndarray) -> np.ndarray:
    """rope_apply_vec.sv's OWN pass-through-lane formula (no rotation
    involved -- identical math to pack_rope.rope_apply_ref's un-rotated
    dims), applied to every lane since cross-attention has no RoPE stage
    to fold this into."""
    out = np.zeros_like(v_q16)
    for i in range(len(v_q16)):
        out[i] = rsh_round(int(v_q16[i]) * POST_SCALE_Q16, VFRAC)
    return out


def load_real_cross_attn():
    """Returns (enc_hidden[T2,D], w_enc_finalln, w_ck, w_cv, w_postln, w_cq,
    embed) -- all real, unmodified moonshine-tiny weights; enc_hidden from
    the real conv front-end (same as pack_encoder_self_attn.py)."""
    from transformers import AutoModel, MoonshineForConditionalGeneration

    model = MoonshineForConditionalGeneration.from_pretrained(
        "UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    enc = model.model.encoder
    dec0 = model.model.decoder.layers[0]

    torch.manual_seed(AUDIO_SEED)
    audio = torch.randn(1, AUDIO_LEN, dtype=torch.float32)
    with torch.no_grad():
        h = F.tanh(enc.conv1(audio.unsqueeze(1)))
        h = enc.groupnorm(h)
        h = F.gelu(enc.conv2(h))
        h = F.gelu(enc.conv3(h))
        h = h.permute(0, 2, 1)
    enc_hidden = h[0].detach().numpy().astype(np.float64)   # (T2, D)

    w_enc_finalln = enc.layer_norm.weight.detach().numpy().astype(np.float64)
    w_ck = dec0.encoder_attn.k_proj.weight.detach().numpy().astype(np.float64)
    w_cv = dec0.encoder_attn.v_proj.weight.detach().numpy().astype(np.float64)
    w_postln = dec0.post_attention_layernorm.weight.detach().numpy().astype(np.float64)
    w_cq = dec0.encoder_attn.q_proj.weight.detach().numpy().astype(np.float64)
    embed = model.model.decoder.embed_tokens.weight.detach().numpy().astype(np.float64)
    return enc_hidden, w_enc_finalln, w_ck, w_cv, w_postln, w_cq, embed


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    enc_hidden, w_enc_finalln, w_ck, w_cv, w_postln, w_cq, embed = load_real_cross_attn()
    T2 = enc_hidden.shape[0]

    # ---- Stage 3a: cross K/V computed ONCE from the (stand-in) encoder
    # output, never recomputed, never grows. No LN on this specific path
    # (enc_out is already-normalized upstream by construction) -- matches
    # the real op order exactly, only the "upstream 6 encoder layers"
    # depth is the (already-established, out-of-scope) simplification. --
    k_deq = [[None] * NHEAD for _ in range(T2)]
    v_deq = [[None] * NHEAD for _ in range(T2)]
    for pos in range(T2):
        enc_out_pos = layernorm_nobias(enc_hidden[pos], w_enc_finalln, LN_EPS)
        k16 = to_q16(w_ck @ enc_out_pos)
        v16 = to_q16(w_cv @ enc_out_pos)
        for h in range(NHEAD):
            base = h * HEAD_DIM
            k_h = post_scale_q16(k16[base:base + HEAD_DIM])
            v_h = v16[base:base + HEAD_DIM]   # V never scaled, never RoPE'd -- matches self-attn gates
            k_codes, k_lo, k_scale = quant_head_asym(list(k_h), KBITS, divfree=True)
            v_codes, v_lo, v_scale = quant_head_asym(list(v_h), KBITS, divfree=True)
            k_deq[pos][h] = np.asarray(dequant_head(k_codes, k_lo, k_scale), dtype=np.int64)
            v_deq[pos][h] = np.asarray(dequant_head(v_codes, v_lo, v_scale), dtype=np.int64)

    # ---- Stage 3b (attention part only): one query row per decode step,
    # each attending the SAME fixed T2-row cross K/V set (tcount=T2 always,
    # never growing -- the actual access pattern under test). ----------
    steps_out = []
    for step, tok in enumerate(TOKEN_IDS):
        x = embed[tok]
        xn = layernorm_nobias(x, w_postln, LN_EPS)
        q16 = to_q16(w_cq @ xn)

        ctx_q25 = np.zeros(D, dtype=np.int64)
        step_dump = {"step": step, "token": tok, "heads": []}
        for h in range(NHEAD):
            base = h * HEAD_DIM
            q_h = post_scale_q16(q16[base:base + HEAD_DIM])

            scores = np.zeros(T2, dtype=np.int64)
            for j in range(T2):
                acc = int(np.dot(q_h.astype(object), k_deq[j][h].astype(object)))
                s_q88 = rsh_round(acc, SCORE_SH)
                scores[j] = sat(s_q88, -32768, 32767)
            prob_q20 = int_softmax_q(scores, np.ones(T2, dtype=bool), exp_table())

            ctx_h = np.zeros(HEAD_DIM, dtype=np.int64)
            for d in range(HEAD_DIM):
                acc = 0
                for j in range(T2):
                    acc += int(prob_q20[j]) * int(v_deq[j][h][d])
                ctx_h[d] = rsh_round(acc, CTX_SH)
            ctx_q25[base:base + HEAD_DIM] = ctx_h

            step_dump["heads"].append({
                "scores_q88": scores.tolist(), "T": T2, "ctx_q25": ctx_h.tolist(),
            })
        steps_out.append(step_dump)

    manifest = {"nhead": NHEAD, "head_dim": HEAD_DIM, "d": D, "t2": T2,
                "post_scale_q16": POST_SCALE_Q16, "tokens": TOKEN_IDS,
                "audio_len": AUDIO_LEN, "audio_seed": AUDIO_SEED}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "steps.json"), "w") as f:
        json.dump(steps_out, f, indent=2)

    exp_lut = exp_table()
    with open(os.path.join(out_dir, "exp_lut.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0x1FFFFF:06x}" for v in exp_lut) + "\n")

    with open(os.path.join(out_dir, "inv_lut_lo.mem"), "w") as f:
        for s4 in range(4096):
            f.write(f"{q_round_div(1 << 24, max(s4, 1)) & 0x1FFFFFF:07x}\n")
    with open(os.path.join(out_dir, "inv_lut_hi.mem"), "w") as f:
        for s4 in range(4096, 16512):
            f.write(f"{q_round_div(1 << 24, s4) & 0x1FFF:04x}\n")

    def w32(f, v):
        f.write(format(int(v) & 0xFFFFFFFF, "08x") + "\n")

    # cross K/V: ALREADY POST_SCALE'd (K) / raw (V) Q.16 vectors, pos-major
    # then head-major -- written to kv_bank ONCE, before any query read.
    with open(os.path.join(out_dir, "kv_in.mem"), "w") as f:
        for pos in range(T2):
            enc_out_pos = layernorm_nobias(enc_hidden[pos], w_enc_finalln, LN_EPS)
            k16 = to_q16(w_ck @ enc_out_pos)
            v16 = to_q16(w_cv @ enc_out_pos)
            for h in range(NHEAD):
                base = h * HEAD_DIM
                k_h = post_scale_q16(k16[base:base + HEAD_DIM])
                v_h = v16[base:base + HEAD_DIM]
                for v in k_h:
                    w32(f, v)
                for v in v_h:
                    w32(f, v)

    # query: ALREADY POST_SCALE'd Q.16 vectors, step-major then head-major.
    with open(os.path.join(out_dir, "q_in.mem"), "w") as f:
        for step, tok in enumerate(TOKEN_IDS):
            x = embed[tok]
            xn = layernorm_nobias(x, w_postln, LN_EPS)
            q16 = to_q16(w_cq @ xn)
            for h in range(NHEAD):
                base = h * HEAD_DIM
                q_h = post_scale_q16(q16[base:base + HEAD_DIM])
                for v in q_h:
                    w32(f, v)

    with open(os.path.join(out_dir, "ctx_ref.mem"), "w") as f:
        for sd in steps_out:
            for hd in sd["heads"]:
                for v in hd["ctx_q25"]:
                    w32(f, v)

    print(f"GEN dir={out_dir} t2={T2} n_steps={len(steps_out)} post_scale_q16={POST_SCALE_Q16}")
    return steps_out


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_decoder_cross_attn")
    p.add_argument("cmd", choices=["gen"])
    p.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    main_gen(a.dir)


if __name__ == "__main__":
    main()
