"""Golden reference + RTL test vectors for the decoder self-attention block
gate: does chaining checkpoint C's real RTL (kv_bank.sv's INT8 K/V cache +
vec_attn_w.sv's score/softmax/ctx pipeline) with the newly-built
rope_apply_vec.sv correctly compute ASR's decoder self-attention?

Scope, deliberate: Q/K/V generation (LayerNorm + the Q/K/V GEMVs) is
treated as already-proven -- layernorm_vec.sv and gemv_banked_resident_vec.sv
(WBW=8) were each separately gated in earlier sessions. This gate uses REAL
FP32 Q/K/V from the actual HF model directly (attention correctness doesn't
depend on whether Q/K/V came from an FP32 or INT8-dequantized GEMV, already
established) and focuses entirely on the genuinely new combination: RoPE ->
kv_bank's own INT8 (K8/V8) quantize-at-write/dequantize-at-read -> vec_attn_w's
score/softmax/ctx math.

Reuses this project's own proven Python fixed-point references directly,
not re-derived:
  - fabric.stage3.seq_ref: rsh_round, sat, q_round_div (the RTL's exact
    rounding/saturation conventions)
  - fabric.stage3.run_softmax: int_softmax_q, exp_table (softmax_f.sv's
    own exact integer softmax, PROB_FRAC=20 Q1.20)
  - model.goformer_kvq: quant_head_asym(divfree=True)/dequant_head --
    kv_bank.sv's OWN K8/V8 asymmetric per-head quantization formula,
    verified INV_SH=24 matches kv_bank.sv's own parameter
  - fabric.asr_seq.pack_rope: rope_apply_ref, generalized here with the
    POST_SCALE_Q16 correction rope_apply_vec.sv's own header explains
    (vec_attn_w.sv's SCORE_SH=27 hardcodes a HEAD_DIM=64-specific 1/8
    score scaling; ASR's HEAD_DIM=36 needs 1/6, not a power of 2 --
    scaling q and k each by sqrt(4/3) in Q.16 makes their dot product
    pick up the missing factor)

Two real decode steps tested (T=1 then T=2, reusing the KV cache written
at step 0) -- not just the degenerate T=1 softmax-of-one case.

Usage:
    .venv/bin/python -m fabric.asr_seq.pack_decoder_self_attn gen --dir <sim_dir>
    .venv/bin/python -m fabric.asr_seq.pack_decoder_self_attn check --dir <sim_dir>
"""
from __future__ import annotations

import argparse
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
ISQRT = 3            # vec_attn_w.sv's own hardcoded assumption (1/sqrt(64)=1/8)
SCORE_FRAC = 8        # run_softmax.SCORE_FRAC
PROB_FRAC = 20        # run_softmax.PROB_FRAC
SCORE_SH = 2 * VFRAC + ISQRT - SCORE_FRAC   # 27, matches vec_attn_w.sv's own localparam
CTX_SH = PROB_FRAC + VFRAC - 25             # 11, matches vec_attn_w.sv's own localparam (RESID_FRAC=25)
KBITS = 8             # kv_bank.sv's own KBITS=8 (K8/V8)
POST_SCALE_Q16 = round(math.sqrt(4.0 / 3.0) * Q16)   # 75674, see rope_apply_vec.sv's header

TOKEN_IDS = [1, 940, 24936]   # real decoder_start_token_id + 2 real next tokens


def layernorm_nobias(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / np.sqrt(var + eps) * gamma


def to_q16(x_real) -> np.ndarray:
    return np.round(np.asarray(x_real, dtype=np.float64) * Q16).astype(np.int64)


def load_real_layer0_qkv():
    """Returns (w_ln1, w_q, w_k, w_v) real FP32 numpy arrays for decoder layer 0."""
    from transformers import MoonshineForConditionalGeneration

    model = MoonshineForConditionalGeneration.from_pretrained(
        "UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    layer0 = model.model.decoder.layers[0]
    w_ln1 = layer0.input_layernorm.weight.detach().numpy().astype(np.float64)
    w_q = layer0.self_attn.q_proj.weight.detach().numpy().astype(np.float64)
    w_k = layer0.self_attn.k_proj.weight.detach().numpy().astype(np.float64)
    w_v = layer0.self_attn.v_proj.weight.detach().numpy().astype(np.float64)
    embed = model.model.decoder.embed_tokens.weight.detach().numpy().astype(np.float64)
    return w_ln1, w_q, w_k, w_v, embed


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    w_ln1, w_q, w_k, w_v, embed = load_real_layer0_qkv()

    cos_rom, sin_rom = build_cos_sin_rom()

    self_k_cache = [[] for _ in range(NHEAD)]   # per head: list of dequantized Q.16 vectors
    self_v_cache = [[] for _ in range(NHEAD)]

    steps_out = []
    for step, tok in enumerate(TOKEN_IDS):
        x = embed[tok]                                    # (D,) real embedding
        xn = layernorm_nobias(x, w_ln1, LN_EPS)
        q_real = w_q @ xn                                   # (D,)
        k_real = w_k @ xn
        v_real = w_v @ xn

        q16 = to_q16(q_real)
        k16 = to_q16(k_real)
        v16 = to_q16(v_real)

        ctx_q25 = np.zeros(D, dtype=np.int64)
        step_dump = {"step": step, "token": tok, "heads": []}
        for h in range(NHEAD):
            base = h * HEAD_DIM
            q_h = q16[base:base + HEAD_DIM]
            k_h = k16[base:base + HEAD_DIM]
            v_h = v16[base:base + HEAD_DIM]

            q_rope = rope_apply_ref(q_h, step, cos_rom, sin_rom, POST_SCALE_Q16)
            k_rope = rope_apply_ref(k_h, step, cos_rom, sin_rom, POST_SCALE_Q16)

            # kv_bank's own quantize-at-write, matching kv_bank.sv exactly
            k_codes, k_lo, k_scale = quant_head_asym(list(k_rope), KBITS, divfree=True)
            v_codes, v_lo, v_scale = quant_head_asym(list(v_h), KBITS, divfree=True)
            k_deq = np.asarray(dequant_head(k_codes, k_lo, k_scale), dtype=np.int64)
            v_deq = np.asarray(dequant_head(v_codes, v_lo, v_scale), dtype=np.int64)

            self_k_cache[h].append(k_deq)
            self_v_cache[h].append(v_deq)
            T = len(self_k_cache[h])

            scores = np.zeros(T, dtype=np.int64)
            for j in range(T):
                acc = int(np.dot(q_rope.astype(object), self_k_cache[h][j].astype(object)))
                s_q88 = rsh_round(acc, SCORE_SH)
                scores[j] = sat(s_q88, -32768, 32767)
            prob_q20 = int_softmax_q(scores, np.ones(T, dtype=bool), exp_table())

            ctx_h = np.zeros(HEAD_DIM, dtype=np.int64)
            for d in range(HEAD_DIM):
                acc = 0
                for j in range(T):
                    acc += int(prob_q20[j]) * int(self_v_cache[h][j][d])
                ctx_h[d] = rsh_round(acc, CTX_SH)
            ctx_q25[base:base + HEAD_DIM] = ctx_h

            step_dump["heads"].append({
                "q_q16": q_h.tolist(), "k_q16": k_h.tolist(), "v_q16": v_h.tolist(),
                "q_rope_q16": q_rope.tolist(), "k_rope_q16": k_rope.tolist(),
                "k_deq_q16": k_deq.tolist(), "v_deq_q16": v_deq.tolist(),
                "scores_q88": scores.tolist(), "T": T,
                "ctx_q25": ctx_h.tolist(),
            })
        steps_out.append(step_dump)

    manifest = {"nhead": NHEAD, "head_dim": HEAD_DIM, "d": D,
                "post_scale_q16": POST_SCALE_Q16, "tokens": TOKEN_IDS,
                "n_steps": len(steps_out)}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "steps.json"), "w") as f:
        json.dump(steps_out, f, indent=2)

    # ---- RTL testbench inputs -------------------------------------------------
    # rope_apply_vec's own ROM (identical to pack_rope.py's own -- reused, not
    # regenerated differently).
    with open(os.path.join(out_dir, "rope_cos.mem"), "w") as f:
        for pos in range(cos_rom.shape[0]):
            for i in range(ROT_PAIRS):
                f.write(format(int(cos_rom[pos, i]) & 0xFFFF, "04x") + "\n")
    with open(os.path.join(out_dir, "rope_sin.mem"), "w") as f:
        for pos in range(sin_rom.shape[0]):
            for i in range(ROT_PAIRS):
                f.write(format(int(sin_rom[pos, i]) & 0xFFFF, "04x") + "\n")

    # softmax_f.sv's own exp LUT -- copied verbatim from
    # fabric/stage3/run_softmax.py's own generation (the proven, already-
    # gated formula; not re-derived). exp_table() is imported above, same
    # call already used to build the Python reference's own probabilities.
    exp_lut = exp_table()
    with open(os.path.join(out_dir, "exp_lut.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0x1FFFFF:06x}" for v in exp_lut) + "\n")

    # kv_bank.sv's own inv_lut ROMs -- pure function of INV_SH=24/KBITS=8,
    # copied verbatim from fabric/stage3/run_vec_kv.py's own generation (the
    # proven, already-gated formula; not re-derived).
    with open(os.path.join(out_dir, "inv_lut_lo.mem"), "w") as f:
        for s4 in range(4096):
            f.write(f"{q_round_div(1 << 24, max(s4, 1)) & 0x1FFFFFF:07x}\n")
    with open(os.path.join(out_dir, "inv_lut_hi.mem"), "w") as f:
        for s4 in range(4096, 16512):
            f.write(f"{q_round_div(1 << 24, s4) & 0x1FFF:04x}\n")

    # per-(step,head) raw (pre-RoPE) q/k/v Q.16 vectors -- the testbench drives
    # these INTO rope_apply_vec (for q,k) and straight into kv_bank's write
    # port (for v, which never gets RoPE), then checks ctx against ctx_ref.mem.
    def w32(f, v):
        f.write(format(int(v) & 0xFFFFFFFF, "08x") + "\n")

    with open(os.path.join(out_dir, "qkv_in.mem"), "w") as f:
        for sd in steps_out:
            for hd in sd["heads"]:
                for v in hd["q_q16"]:
                    w32(f, v)
                for v in hd["k_q16"]:
                    w32(f, v)
                for v in hd["v_q16"]:
                    w32(f, v)
    with open(os.path.join(out_dir, "ctx_ref.mem"), "w") as f:
        for sd in steps_out:
            for hd in sd["heads"]:
                for v in hd["ctx_q25"]:
                    w32(f, v)

    print(f"GEN dir={out_dir} n_steps={len(steps_out)} post_scale_q16={POST_SCALE_Q16}")
    return steps_out


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_decoder_self_attn")
    p.add_argument("cmd", choices=["gen"])
    p.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    main_gen(a.dir)


if __name__ == "__main__":
    main()
