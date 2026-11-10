"""Bonsai (Qwen3-architecture, 1-bit) reference forward — doc 10, R1.

One implementation of the model, written from scratch (no transformers), with the
GEMV pluggable so the same forward serves as:
  * the FLOAT reference  (`FloatGemv`: dequantised weights, fp32 math), and
  * the FIXED reference  (`fixed.py`: the bit-true integer arithmetic the fabric runs).

Weights stay PACKED (1 sign bit + fp16 scale per 128) on the device and are expanded
one matrix at a time, so the 8B model fits a 4 GB laptop GPU.

Sources: `from_gguf` (the Q1_0 release = ground truth) or `from_unpacked` (bf16
safetensors, used only to validate this code against HF transformers).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

QK = 128


@dataclass
class Cfg:
    n_layer: int
    d: int
    ffn: int
    n_head: int
    n_kv: int
    head_dim: int
    vocab: int
    rope_theta: float
    eps: float
    tied: bool
    yarn_factor: float = 0.0       # 0 = plain RoPE
    yarn_orig_ctx: int = 0


# ----------------------------------------------------------------------------- RoPE
def rope_params(cfg: Cfg):
    """(inv_freq [hd/2] float32, attention_factor) — plain RoPE, or YaRN exactly as
    transformers' _compute_yarn_parameters (beta_fast 32, beta_slow 1, truncate=True,
    attention_factor = 0.1 ln(factor) + 1), which HF applies at every sequence length."""
    hd, base = cfg.head_dim, cfg.rope_theta
    pos_freqs = base ** (torch.arange(0, hd, 2, dtype=torch.int64).to(torch.float32) / hd)
    if not cfg.yarn_factor:
        return 1.0 / pos_freqs, 1.0
    f, orig = cfg.yarn_factor, cfg.yarn_orig_ctx
    att = 0.1 * math.log(f) + 1.0

    def corr_dim(nrot):
        return (hd * math.log(orig / (nrot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr_dim(32)), 0)
    high = min(math.ceil(corr_dim(1)), hd - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(hd // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extrap = 1 - ramp
    inv = (1.0 / (f * pos_freqs)) * (1 - extrap) + (1.0 / pos_freqs) * extrap
    return inv, att


# ----------------------------------------------------------------------------- weights
class Q1Matrix:
    """A packed 1-bit matrix [out, in]: signs [out, in/8] uint8 (bit j%8 of byte j//8,
    1 = +), scales [out, in/128] fp16."""

    def __init__(self, signs: torch.Tensor, scales: torch.Tensor):
        self.signs, self.scales = signs, scales
        self.out = signs.shape[0]
        self.inn = signs.shape[1] * 8

    def dequant(self, r0=0, r1=None, dtype=torch.float32) -> torch.Tensor:
        s = self.signs[r0:r1]
        bits = (s.unsqueeze(-1) >> torch.arange(8, device=s.device, dtype=torch.uint8)) & 1
        pm = bits.reshape(s.shape[0], -1).to(dtype) * 2 - 1
        sc = self.scales[r0:r1].to(dtype).repeat_interleave(QK, dim=1)
        return pm * sc

    def sign_pm(self, r0=0, r1=None, dtype=torch.float32) -> torch.Tensor:
        s = self.signs[r0:r1]
        bits = (s.unsqueeze(-1) >> torch.arange(8, device=s.device, dtype=torch.uint8)) & 1
        return bits.reshape(s.shape[0], -1).to(dtype) * 2 - 1


class DenseMatrix:
    """Unpacked weights (validation path only)."""

    def __init__(self, w: torch.Tensor):
        self.w = w
        self.out, self.inn = w.shape

    def dequant(self, r0=0, r1=None, dtype=torch.float32):
        return self.w[r0:r1].to(dtype)


# ----------------------------------------------------------------------------- GEMV
class FloatGemv:
    """y = W x in fp32, row-chunked so the expanded matrix never exceeds ~CHUNK rows."""
    CHUNK = 8192

    def __call__(self, W, x: torch.Tensor, name: str = "") -> torch.Tensor:
        ys = []
        for r0 in range(0, W.out, self.CHUNK):
            ys.append(W.dequant(r0, r0 + self.CHUNK) @ x)
        return torch.cat(ys)

    def mm(self, W, X: torch.Tensor, name: str = "") -> torch.Tensor:
        """Y [T, out] = X [T, in] @ W^T, row-chunked."""
        ys = []
        for r0 in range(0, W.out, self.CHUNK):
            ys.append(X @ W.dequant(r0, r0 + self.CHUNK).T)
        return torch.cat(ys, 1)


# ----------------------------------------------------------------------------- model
class Qwen3Ref:
    def __init__(self, cfg: Cfg, W: dict, device="cuda", gemv=None):
        self.c, self.W, self.dev = cfg, W, device
        self.gemv = gemv or FloatGemv()
        inv, self.attn_factor = rope_params(cfg)
        self.inv_freq = inv.to(device)
        self.reset()

    def reset(self):
        c = self.c
        self.k_cache = [[] for _ in range(c.n_layer)]
        self.v_cache = [[] for _ in range(c.n_layer)]
        self.pos = 0

    # --- non-linears (fp32; fixed.py may override) ---
    def rmsnorm(self, x, w):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.c.eps) * w

    def rope(self, x, pos):  # x [n, hd], NeoX half-rotation
        hd = x.shape[-1]
        ang = (pos * self.inv_freq).to(torch.float32)
        cos, sin = torch.cos(ang) * self.attn_factor, torch.sin(ang) * self.attn_factor
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def softmax(self, s):
        return torch.softmax(s, -1)

    def silu(self, x):
        return x * torch.sigmoid(x)

    def embed(self, tok: int):
        return self.W["embed"].dequant(tok, tok + 1)[0].to(torch.float32)

    # --- one token, KV-cached ---
    def step(self, tok: int, trace: dict | None = None) -> torch.Tensor:
        c, W, g = self.c, self.W, self.gemv
        x = self.embed(tok)
        pos = self.pos
        for L in range(c.n_layer):
            p = f"l{L}."
            h = self.rmsnorm(x, W[p + "attn_norm"])
            q = g(W[p + "q"], h, p + "q").view(c.n_head, c.head_dim)
            k = g(W[p + "k"], h, p + "k").view(c.n_kv, c.head_dim)
            v = g(W[p + "v"], h, p + "v").view(c.n_kv, c.head_dim)
            q = self.rope(self.rmsnorm(q, W[p + "q_norm"]), pos)
            k = self.rope(self.rmsnorm(k, W[p + "k_norm"]), pos)
            self.k_cache[L].append(k)
            self.v_cache[L].append(v)
            K = torch.stack(self.k_cache[L], 1)       # [n_kv, T, hd]
            V = torch.stack(self.v_cache[L], 1)
            rep = c.n_head // c.n_kv
            qg = q.view(c.n_kv, rep, c.head_dim)
            s = torch.einsum("grd,gtd->grt", qg, K) / math.sqrt(c.head_dim)
            a = self.softmax(s)
            ctx = torch.einsum("grt,gtd->grd", a, V).reshape(-1)
            x = x + g(W[p + "o"], ctx, p + "o")
            h = self.rmsnorm(x, W[p + "ffn_norm"])
            gt = g(W[p + "gate"], h, p + "gate")
            up = g(W[p + "up"], h, p + "up")
            x = x + g(W[p + "down"], self.silu(gt) * up, p + "down")
            if trace is not None and L in trace.get("layers", ()):
                trace[f"x{L}"] = x.detach().clone()
        h = self.rmsnorm(x, W["out_norm"])
        logits = g(W["lm_head"], h, "lm_head")
        self.pos += 1
        return logits

    # --- whole sequence at once (teacher forcing / evaluation) ---
    def rope_seq(self, x, T):  # x [h, T, hd]
        hd = x.shape[-1]
        ang = (torch.arange(T, device=x.device, dtype=torch.float64)[:, None] * self.inv_freq[None]).to(torch.float32)
        cos, sin = torch.cos(ang) * self.attn_factor, torch.sin(ang) * self.attn_factor
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    @torch.no_grad()
    def forward_seq(self, toks) -> torch.Tensor:
        """Causal forward over the whole sequence; logits [T, V]. Same math as step();
        the GEMV is called with X [T, in] (row-wise semantics identical to per-token)."""
        c, W, g = self.c, self.W, self.gemv
        T = len(toks)
        E = W["embed"]
        x = torch.stack([E.dequant(int(t), int(t) + 1)[0].to(torch.float32) for t in toks])
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        rep = c.n_head // c.n_kv
        for L in range(c.n_layer):
            p = f"l{L}."
            h = self.rmsnorm(x, W[p + "attn_norm"])
            q = g.mm(W[p + "q"], h, p + "q").view(T, c.n_head, c.head_dim).transpose(0, 1)
            k = g.mm(W[p + "k"], h, p + "k").view(T, c.n_kv, c.head_dim).transpose(0, 1)
            v = g.mm(W[p + "v"], h, p + "v").view(T, c.n_kv, c.head_dim).transpose(0, 1)
            q = self.rope_seq(self.rmsnorm(q, W[p + "q_norm"]), T)
            k = self.rope_seq(self.rmsnorm(k, W[p + "k_norm"]), T)
            k = k.repeat_interleave(rep, 0)
            v = v.repeat_interleave(rep, 0)
            s = q @ k.transpose(1, 2) / math.sqrt(c.head_dim) + mask
            ctx = (self.softmax(s) @ v).transpose(0, 1).reshape(T, -1)
            x = x + g.mm(W[p + "o"], ctx, p + "o")
            h = self.rmsnorm(x, W[p + "ffn_norm"])
            gt = g.mm(W[p + "gate"], h, p + "gate")
            up = g.mm(W[p + "up"], h, p + "up")
            x = x + g.mm(W[p + "down"], self.silu(gt) * up, p + "down")
        h = self.rmsnorm(x, W["out_norm"])
        return g.mm(W["lm_head"], h, "lm_head")

    @torch.no_grad()
    def prefill(self, toks):
        lg = None
        for t in toks:
            lg = self.step(int(t))
        return lg

    @torch.no_grad()
    def greedy(self, toks, n):
        self.reset()
        lg = self.prefill(toks)
        out = []
        for _ in range(n):
            t = int(torch.argmax(lg))
            out.append(t)
            lg = self.step(t)
        return out


# ----------------------------------------------------------------------------- loaders
GG = {"attn_norm": "attn_norm", "q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output",
      "q_norm": "attn_q_norm", "k_norm": "attn_k_norm", "ffn_norm": "ffn_norm",
      "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}
HF = {"attn_norm": "input_layernorm", "q": "self_attn.q_proj", "k": "self_attn.k_proj",
      "v": "self_attn.v_proj", "o": "self_attn.o_proj", "q_norm": "self_attn.q_norm",
      "k_norm": "self_attn.k_norm", "ffn_norm": "post_attention_layernorm",
      "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
NORMS = ("attn_norm", "q_norm", "k_norm", "ffn_norm")


def cfg_from_gguf(g) -> Cfg:
    m = g.meta
    a = m["general.architecture"]
    d = int(m[f"{a}.embedding_length"])
    nh = int(m[f"{a}.attention.head_count"])
    return Cfg(
        n_layer=int(m[f"{a}.block_count"]), d=d, ffn=int(m[f"{a}.feed_forward_length"]),
        n_head=nh, n_kv=int(m[f"{a}.attention.head_count_kv"]),
        head_dim=int(m.get(f"{a}.attention.key_length", d // nh)),
        vocab=int(g.tensors["token_embd.weight"].shape[1]),
        rope_theta=float(m.get(f"{a}.rope.freq_base", 1e6)),
        eps=float(m.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)),
        tied="output.weight" not in g.tensors,
        yarn_factor=float(m.get(f"{a}.rope.scaling.factor", 0.0)) if m.get(f"{a}.rope.scaling.type") == "yarn" else 0.0,
        yarn_orig_ctx=int(m.get(f"{a}.rope.scaling.original_context_length", 0)))


def from_gguf(path, device="cuda"):
    from .gguf_q1 import GGUF, Q1_BLOCK_BYTES
    g = GGUF(path)
    cfg = cfg_from_gguf(g)

    def mat(name):
        ti = g.tensors[name]
        if g.is_q1(name):
            d, qs = g.q1_parts(name)
            rows, inn = ti.n // ti.shape[0], ti.shape[0]
            signs = torch.from_numpy(np.ascontiguousarray(qs).reshape(rows, inn // 8)).to(device)
            scales = torch.from_numpy(d.copy().reshape(rows, inn // QK)).to(device)
            return Q1Matrix(signs, scales)
        return DenseMatrix(torch.from_numpy(g.dequant(name)).to(device))

    def vec(name):
        return torch.from_numpy(g.dequant(name).reshape(-1)).to(device)

    W = {"embed": mat("token_embd.weight"), "out_norm": vec("output_norm.weight")}
    W["lm_head"] = W["embed"] if cfg.tied else mat("output.weight")
    for L in range(cfg.n_layer):
        for k, gk in GG.items():
            n = f"blk.{L}.{gk}.weight"
            W[f"l{L}.{k}"] = vec(n) if k in NORMS else mat(n)
    return cfg, W, g


def from_unpacked(dirpath, device="cuda", dtype=torch.float32):
    from safetensors.torch import load_file
    d = Path(dirpath)
    hc = json.loads((d / "config.json").read_text())
    cfg = Cfg(n_layer=hc["num_hidden_layers"], d=hc["hidden_size"], ffn=hc["intermediate_size"],
              n_head=hc["num_attention_heads"], n_kv=hc["num_key_value_heads"],
              head_dim=hc.get("head_dim", hc["hidden_size"] // hc["num_attention_heads"]),
              vocab=hc["vocab_size"], rope_theta=float(hc.get("rope_theta", 1e6)),
              eps=float(hc.get("rms_norm_eps", 1e-6)), tied=bool(hc.get("tie_word_embeddings", False)))
    rs = hc.get("rope_scaling") or {}
    if rs.get("rope_type", rs.get("type")) == "yarn":
        cfg.yarn_factor = float(rs["factor"])
        cfg.yarn_orig_ctx = int(rs.get("original_max_position_embeddings", hc["max_position_embeddings"]))
    sd = {}
    for f in sorted(d.glob("*.safetensors")):
        sd.update(load_file(str(f)))

    def t(n):
        return sd[n].to(device=device, dtype=dtype)

    W = {"embed": DenseMatrix(t("model.embed_tokens.weight")), "out_norm": t("model.norm.weight")}
    W["lm_head"] = W["embed"] if cfg.tied else DenseMatrix(t("lm_head.weight"))
    for L in range(cfg.n_layer):
        for k, hk in HF.items():
            x = t(f"model.layers.{L}.{hk}.weight")
            W[f"l{L}.{k}"] = x if k in NORMS else DenseMatrix(x)
    return cfg, W, sd
