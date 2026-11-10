"""Activation-quantised 1-bit GEMV variants (doc 10, R1b) — the arithmetic choices
the fabric could make, simulated in fp32 so they can be ranked by accuracy.

Weights are always exact: sign * fp16 group scale. Only the GEMV *input* vector is
quantised, groupwise along the input dimension:

  int<b>  g : x_q = round(x / s), s = absmax_g / (2^(b-1)-1)      (float scale per group)
  bfp<b>  g : x_q = round(x / 2^e), e = ceil(log2(absmax_g / (2^(b-1)-1)))
              (power-of-two shared exponent: the fabric does sign-adds of b-bit
               integers and an exponent shift — no multiplier in the inner loop)
  vec<b>    : one float scale for the whole vector

`QuantGemv.mm` keeps the FloatGemv interface used by Qwen3Ref.forward_seq.
"""
from __future__ import annotations

import torch

from .qwen3_ref import FloatGemv


def quant_act(X: torch.Tensor, kind: str, bits: int, group: int) -> torch.Tensor:
    """Return the dequantised (simulated) activations, same shape as X [T, n]."""
    qmax = 2 ** (bits - 1) - 1
    if kind == "none":
        return X
    if kind == "vec":
        s = X.abs().amax(-1, keepdim=True).clamp_min(1e-30) / qmax
        return torch.clamp(torch.round(X / s), -qmax, qmax) * s
    T, n = X.shape
    Xg = X.view(T, n // group, group)
    am = Xg.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    if kind == "int":
        s = am / qmax
    elif kind == "bfp":
        s = torch.exp2(torch.ceil(torch.log2(am / qmax)))
    else:
        raise ValueError(kind)
    return (torch.clamp(torch.round(Xg / s), -qmax, qmax) * s).view(T, n)


class QuantGemv(FloatGemv):
    def __init__(self, kind="bfp", bits=8, group=128, skip=()):
        self.kind, self.bits, self.group, self.skip = kind, bits, group, tuple(skip)

    def mm(self, W, X, name=""):
        if not any(s in name for s in self.skip):
            X = quant_act(X, self.kind, self.bits, self.group)
        return super().mm(W, X, name)

    def __call__(self, W, x, name=""):
        return self.mm(W, x[None], name)[0]

    def __repr__(self):
        return f"{self.kind}{self.bits}" + ("" if self.kind == "vec" else f"_g{self.group}") + \
            (f"_skip{'+'.join(self.skip)}" if self.skip else "")
