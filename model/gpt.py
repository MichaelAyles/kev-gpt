"""A compact GPT, nanoGPT-lineage, sized for the Kevin-on-Kria budget.

Deliberately small: 2-4M params, short context, so the whole thing can later be
quantised to INT4 and baked into the KV260's on-chip memory (doc 2). Nothing
here is exotic — the constraint is size, not architecture. Attention uses
torch's fused scaled_dot_product_attention so it is fast on CUDA and MPS alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 256        # context length; short on purpose (KV stays on-chip)
    vocab_size: int = 256        # set from the tokenizer at build time
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False           # RMSNorm-style: no biases, cheaper in fabric
    attn_fp32: bool = False      # diagnostic: do QK^T + softmax in explicit fp32
                                  # instead of trusting SDPA's opaque autocast-dtype
                                  # path, matching GPT-Neo's own precision convention
                                  # (transformers' GPTNeoSelfAttention._attn upcasts
                                  # q/k to fp32 before the matmul, casts back after
                                  # softmax). Tests whether kev-gpt's width-scaling
                                  # AdamW divergence (see model/SCALE-UP-LOG.md) is
                                  # partly a precision-path artifact of the fused
                                  # SDPA call rather than purely an Adam-eps issue.
                                  # 0.0/False = off, unchanged SDPA behavior.
    z_loss_coef: float = 0.0     # penalizes logsumexp(logits)^2 -- see forward().
                                  # 0.0 = off (unchanged default behavior); PaLM/
                                  # ST-MoE use ~1e-4. Added while investigating a
                                  # width-scaling training instability where logit
                                  # scale grew unboundedly (unchecked by anything
                                  # in this architecture) throughout training,
                                  # amplifying wrong predictions once ranking
                                  # accuracy plateaued -- see model/train.py's
                                  # weight-decay param-grouping comment for the
                                  # fuller investigation.


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.attn_fp32 = cfg.attn_fp32
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.attn_fp32:
            # Manual attention, explicit fp32 QK^T + softmax regardless of the
            # active autocast dtype (mirrors GPT-Neo's GPTNeoSelfAttention._attn:
            # query/key upcast to fp32 before the matmul, attn_weights cast back
            # to value's dtype after softmax). Still scales by 1/sqrt(head_dim)
            # -- GPT-Neo itself omits that scaling, but that's a separate
            # architectural choice, not the precision-path variable under test.
            head_dim = C // self.n_head
            qf = q.float()
            kf = k.float()
            att = (qf @ kf.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            att = att.masked_fill(causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1).to(v.dtype)
            if self.training and self.dropout > 0:
                att = F.dropout(att, p=self.dropout)
            y = att @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying: embedding and output share weights, saves params (matters
        # at this scale) and a touch of on-chip memory later.
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        # GPT-2-paper residual-projection scaling: the two matrices that write
        # DIRECTLY into the residual stream (attn.proj, the second MLP Linear)
        # get std scaled by 1/sqrt(2*n_layer) instead of the flat 0.02 every
        # other Linear uses -- without it, residual-stream variance grows
        # with depth AND width uncontrolled, since nothing here dampens each
        # block's addition to the running sum. Missing this is a real,
        # previously-undocumented gap in this codebase: every prior
        # scale-up (D=256 char-level, D=384 word-level) hit the SAME
        # train+val-climb-together instability signature shortly after
        # warmup ends, at multiple different peak LR/warmup combos --
        # pointing at an architectural cause, not just a tuning miss.
        for pn, p in self.named_parameters():
            if pn.endswith("attn.proj.weight") or pn.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # subtract the (tied) positional table to report the trainable headline
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            flat_logits = logits.view(-1, logits.size(-1))
            loss = F.cross_entropy(flat_logits, targets.view(-1))
            if self.cfg.z_loss_coef > 0:
                # caps logit scale directly (log-sum-exp = the log-normalizer
                # cross-entropy already computes internally) instead of hoping
                # weight decay controls it indirectly -- see GPTConfig.z_loss_coef.
                z = torch.logsumexp(flat_logits, dim=-1)
                loss = loss + self.cfg.z_loss_coef * z.pow(2).mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
