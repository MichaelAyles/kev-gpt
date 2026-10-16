"""Golden reference + RTL test vectors for the encoder self-attention block
gate: does the "static full-attend" access pattern -- writing a FIXED K/V
set once, then reading the SAME set repeatedly (once per query row) --
work correctly through checkpoint C's real, UNMODIFIED kv_bank.sv +
vec_attn_w.sv, chained with rope_apply_vec.sv?

Why this matters: ASR-ACCELERATOR-OP-SEQUENCE.md's "Two attention shapes,
one primitive" section classified the encoder's bidirectional self-attention
(and every decoder layer's cross-attention) as needing NEW storage RTL
("architecturally closer to weight_bank_tdp.sv... write once at setup,
stream-read repeatedly"), because kv_bank.sv's own design is described as
built for a cache that grows. A cheap isolated probe (fabric/asr_seq/tb,
not committed -- see PORT-NOTES.md's "static full-attend, kv_bank.sv reused
unmodified" note) showed kv_bank.sv's read port has NO actual incremental
dependency baked into a single rd_start call: write T positions once, then
call rd_start repeatedly with the SAME fixed tcount, and every read returns
bit-identical data. "Growing" is purely a caller convention (the LLM/decoder
varying tcount step by step), not something kv_bank.sv's RTL enforces. This
gate is the real, model-weight-driven proof of that claim: NHEAD=8 heads,
T2=6 real encoder positions (from moonshine-tiny's own conv front-end on a
short fixed waveform), every one of the 6 query rows attending the full,
already-fully-written 6-row K/V set -- bidirectional, no causal masking,
zero new storage RTL.

Scope, deliberate, same as pack_decoder_self_attn.py: LayerNorm + the Q/K/V
GEMVs are treated as already-proven elsewhere (layernorm_vec.sv,
gemv_banked_resident_vec.sv WBW=8) -- this gate uses REAL FP32 Q/K/V
computed directly from the real HF model's own encoder layer-0 weights
and focuses entirely on the attention access pattern + RoPE + kv_bank's
INT8 quantize-at-write/dequantize-at-read + vec_attn_w's score/softmax/ctx
math, reusing the SAME proven fixed-point references as the decoder gate:
  - fabric.stage3.seq_ref: rsh_round, sat, q_round_div
  - fabric.stage3.run_softmax: int_softmax_q, exp_table
  - model.goformer_kvq: quant_head_asym(divfree=True)/dequant_head
  - fabric.asr_seq.pack_rope: rope_apply_ref, build_cos_sin_rom,
    POST_SCALE_Q16 correction (identical HEAD_DIM=36/NHEAD=8 shape, same
    vec_attn_w.sv SCORE_SH=27 mismatch as the decoder gate)

Encoder input: real conv1->tanh->groupnorm->conv2->gelu->conv3->gelu
front-end (moonshine-tiny's own real conv/groupnorm weights, unmodified),
run on a short FIXED-SEED random waveform (torch.manual_seed(0), L=3000
samples) purely to get a small, real-weights-derived T2=6 sequence for a
fast gate -- not an end-to-end transcription claim, exactly as the
decoder gate used real weights against a manageable few-step slice
instead of the full DECODE_STEPS/6-layer stack.

Usage:
    .venv/bin/python -m fabric.asr_seq.pack_encoder_self_attn gen --dir <sim_dir>
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
from fabric.asr_seq.pack_rope import rope_apply_ref, build_cos_sin_rom  # noqa: E402

D = 288
HEAD_DIM = 36
NHEAD = 8
ROT_DIM = 32
ROT_PAIRS = 16
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
AUDIO_LEN = 3000   # -> T2=6 encoder positions through the real conv front-end


def layernorm_nobias(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / np.sqrt(var + eps) * gamma


def to_q16(x_real) -> np.ndarray:
    return np.round(np.asarray(x_real, dtype=np.float64) * Q16).astype(np.int64)


def load_real_encoder_layer0():
    """Returns (hidden_states[T2,D] real FP32, w_ln1, w_q, w_k, w_v) --
    hidden_states from the model's OWN real conv front-end run on a short
    fixed-seed waveform; weights from encoder layer 0, all real/unmodified."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained("UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    enc = model.encoder

    torch.manual_seed(AUDIO_SEED)
    audio = torch.randn(1, AUDIO_LEN, dtype=torch.float32)
    with torch.no_grad():
        h = F.tanh(enc.conv1(audio.unsqueeze(1)))
        h = enc.groupnorm(h)
        h = F.gelu(enc.conv2(h))
        h = F.gelu(enc.conv3(h))
        h = h.permute(0, 2, 1)      # (1, T2, D)

    hidden_states = h[0].detach().numpy().astype(np.float64)   # (T2, D)

    layer0 = enc.layers[0]
    w_ln1 = layer0.input_layernorm.weight.detach().numpy().astype(np.float64)
    w_q = layer0.self_attn.q_proj.weight.detach().numpy().astype(np.float64)
    w_k = layer0.self_attn.k_proj.weight.detach().numpy().astype(np.float64)
    w_v = layer0.self_attn.v_proj.weight.detach().numpy().astype(np.float64)
    return hidden_states, w_ln1, w_q, w_k, w_v


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    hidden_states, w_ln1, w_q, w_k, w_v = load_real_encoder_layer0()
    T2 = hidden_states.shape[0]

    cos_rom, sin_rom = build_cos_sin_rom()

    # ---- pass 1: RoPE'd K and raw V for every position, every head --------
    # (the "write the full K/V set before any query reads it" half of the
    # static full-attend pattern -- no incremental growth, all T2 written up
    # front, exactly matching kv_bank.sv's own write port used T2 times)
    k_deq = [[None] * NHEAD for _ in range(T2)]   # [pos][head] -> (HEAD_DIM,) dequantized Q.16
    v_deq = [[None] * NHEAD for _ in range(T2)]
    q_rope_all = [[None] * NHEAD for _ in range(T2)]   # query rows, RoPE'd once each

    for pos in range(T2):
        x = hidden_states[pos]
        xn = layernorm_nobias(x, w_ln1, LN_EPS)
        q_real = w_q @ xn
        k_real = w_k @ xn
        v_real = w_v @ xn
        q16 = to_q16(q_real)
        k16 = to_q16(k_real)
        v16 = to_q16(v_real)
        for h in range(NHEAD):
            base = h * HEAD_DIM
            q_h = q16[base:base + HEAD_DIM]
            k_h = k16[base:base + HEAD_DIM]
            v_h = v16[base:base + HEAD_DIM]

            q_rope = rope_apply_ref(q_h, pos, cos_rom, sin_rom, POST_SCALE_Q16)
            k_rope = rope_apply_ref(k_h, pos, cos_rom, sin_rom, POST_SCALE_Q16)
            q_rope_all[pos][h] = q_rope

            k_codes, k_lo, k_scale = quant_head_asym(list(k_rope), KBITS, divfree=True)
            v_codes, v_lo, v_scale = quant_head_asym(list(v_h), KBITS, divfree=True)
            k_deq[pos][h] = np.asarray(dequant_head(k_codes, k_lo, k_scale), dtype=np.int64)
            v_deq[pos][h] = np.asarray(dequant_head(v_codes, v_lo, v_scale), dtype=np.int64)

    # ---- pass 2: EVERY query position attends the FULL, unchanging T2-row
    # K/V set -- bidirectional, no causal masking, tcount==T2 for every
    # single query (the actual "static full-attend" read pattern under test:
    # kv_bank.sv's read port called T2 separate times with the SAME fixed
    # tcount, on data written once in pass 1 and never rewritten). ----------
    queries_out = []
    for qpos in range(T2):
        ctx_q25 = np.zeros(D, dtype=np.int64)
        q_dump = {"qpos": qpos, "heads": []}
        for h in range(NHEAD):
            q_rope = q_rope_all[qpos][h]
            scores = np.zeros(T2, dtype=np.int64)
            for j in range(T2):
                acc = int(np.dot(q_rope.astype(object), k_deq[j][h].astype(object)))
                s_q88 = rsh_round(acc, SCORE_SH)
                scores[j] = sat(s_q88, -32768, 32767)
            prob_q20 = int_softmax_q(scores, np.ones(T2, dtype=bool), exp_table())

            base = h * HEAD_DIM
            ctx_h = np.zeros(HEAD_DIM, dtype=np.int64)
            for d in range(HEAD_DIM):
                acc = 0
                for j in range(T2):
                    acc += int(prob_q20[j]) * int(v_deq[j][h][d])
                ctx_h[d] = rsh_round(acc, CTX_SH)
            ctx_q25[base:base + HEAD_DIM] = ctx_h

            q_dump["heads"].append({
                "scores_q88": scores.tolist(), "T": T2,
                "ctx_q25": ctx_h.tolist(),
            })
        queries_out.append(q_dump)

    manifest = {"nhead": NHEAD, "head_dim": HEAD_DIM, "d": D, "t2": T2,
                "post_scale_q16": POST_SCALE_Q16, "audio_len": AUDIO_LEN,
                "audio_seed": AUDIO_SEED}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "queries.json"), "w") as f:
        json.dump(queries_out, f, indent=2)

    # ---- RTL testbench inputs -----------------------------------------------
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

    def w32(f, v):
        f.write(format(int(v) & 0xFFFFFFFF, "08x") + "\n")

    # per-(pos,head) raw (pre-RoPE) k/v Q.16 vectors -- written to kv_bank
    # ONCE, in pass-1 position order, before any query read happens.
    with open(os.path.join(out_dir, "kv_in.mem"), "w") as f:
        for pos in range(T2):
            x = hidden_states[pos]
            xn = layernorm_nobias(x, w_ln1, LN_EPS)
            k16 = to_q16(w_k @ xn)
            v16 = to_q16(w_v @ xn)
            for h in range(NHEAD):
                base = h * HEAD_DIM
                for v in k16[base:base + HEAD_DIM]:
                    w32(f, v)
                for v in v16[base:base + HEAD_DIM]:
                    w32(f, v)

    # per-(qpos,head) raw (pre-RoPE) q Q.16 vectors -- fed through RoPE at
    # the query's OWN position, then straight into vec_attn_w as q_data.
    with open(os.path.join(out_dir, "q_in.mem"), "w") as f:
        for qpos in range(T2):
            x = hidden_states[qpos]
            xn = layernorm_nobias(x, w_ln1, LN_EPS)
            q16 = to_q16(w_q @ xn)
            for h in range(NHEAD):
                base = h * HEAD_DIM
                for v in q16[base:base + HEAD_DIM]:
                    w32(f, v)

    with open(os.path.join(out_dir, "ctx_ref.mem"), "w") as f:
        for qd in queries_out:
            for hd in qd["heads"]:
                for v in hd["ctx_q25"]:
                    w32(f, v)

    print(f"GEN dir={out_dir} t2={T2} post_scale_q16={POST_SCALE_Q16}")
    return queries_out


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_encoder_self_attn")
    p.add_argument("cmd", choices=["gen"])
    p.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    main_gen(a.dir)


if __name__ == "__main__":
    main()
