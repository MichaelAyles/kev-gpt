"""gumbel.py — the EXACT on-chip-sampling spec, shared by the RTL and the gate.

Sampling from softmax(logit/T) is identically argmax_i( logit_i + T*g_i ), with
g_i = -log(-log(u_i)), u_i ~ U(0,1)  (the Gumbel-max trick). So the fabric needs
NO softmax and NO host logit readback: it adds a precomputed noise value to each
head logit and the existing argmax winner IS the sample. The host writes ONE seed
per request instead of reading VOCAB logits per token.

This module is the single source of truth for BOTH:
  * the gumbel_lut.mem ROM the RTL loads, and
  * the bit-exact Python reference the gate checks the fabric against.

Fixed point: head logits are Q6.25 signed ints; noise is added on the same scale.
Determinism: a 32-bit xorshift seeded by the SEED register, advancing once per
logit and PERSISTING across a request's tokens (so every token gets fresh noise).
seed == 0 => greedy (noise forced to 0) => argmax, bit-exact to the old behaviour.
"""

from __future__ import annotations

import math

TEMP = 0.45          # baked into the LUT (the deployed host temperature).
                      # Recalibrated (was 0.58) after a real-hardware finding:
                      # 0.58 was tuned on 2026-08-26 (commit 6b49ed2) against
                      # the NLAYER=8 checkpoint then deployed (head-logit std
                      # ~4.78, Q6.25 real units) -- but NLAYER grew 8->12 the
                      # SAME day, ~4h45m later (commit ddce9f7), and TEMP was
                      # never rechecked against the new depth. Measured directly
                      # (data/ckpt_stepC_d128_v16384.qat.pt, the model.
                      # tinystories_hf_repro Phase-2 investigation's D=128
                      # retarget -- see model/SCALE-UP-LOG.md) via the SAME
                      # proven-correct golden reference the real bit-exact
                      # gates use (seq_ref.build + IntKVQSequencer(kbits=8,
                      # vbits=8, rotate=False, divfree=True).step(), NOT the
                      # stale seq_ref.IntSequencer directly -- that class still
                      # hardcodes C=256/NHEAD/HEAD_DIM from the old KV260 D=256
                      # shape and throws IndexError at D=128): mean head-logit
                      # std 3.734 across 15 steps / 5 prompts (min 2.15, max
                      # 6.65) -- ~22% LOWER than the stale 4.78 calibration
                      # point, the SAME direction (smaller real signal vs a
                      # too-large fixed noise magnitude) as the original
                      # noise-dominated-garbling failure this file's own history
                      # already documents once. Rescaled proportionally to
                      # preserve the noise-to-signal ratio: 0.58 * (3.734/4.78)
                      # ~= 0.4531, rounded to 0.45. If a future checkpoint's
                      # logit scale drifts again, recheck this the same way
                      # (real head-logit std via seq_ref.build + IntKVQSequencer
                      # .step(), not guessed, not the parent IntSequencer class).
GLBITS = 10          # u resolution = 1/1024; LUT = 1024 x 32b = one BRAM tile
FRAC = 25            # logits are Q6.25
VOCAB = 193
MASK32 = 0xFFFFFFFF


def make_gumbel_lut(temp: float = TEMP, glbits: int = GLBITS,
                    frac: int = FRAC) -> list[int]:
    """gumbel_lut[idx] = round( temp * (-log(-log((idx+0.5)/2^glbits))) * 2^frac ),
    signed int. idx is the top `glbits` bits of an xorshift draw."""
    n = 1 << glbits
    out = []
    for i in range(n):
        u = (i + 0.5) / n                      # in (0,1), never exactly 0 or 1
        g = -math.log(-math.log(u))            # Gumbel(0,1) sample for that u
        out.append(int(round(temp * g * (1 << frac))))
    return out


def xorshift32(x: int) -> int:
    x &= MASK32
    x ^= (x << 13) & MASK32
    x ^= (x >> 17)
    x ^= (x << 5) & MASK32
    return x & MASK32


class GumbelRng:
    """The persistent sampler state for one request. Construct from the seed the
    host wrote; call sample_token() once per generated token, in order."""

    def __init__(self, seed: int, lut: list[int] | None = None,
                 glbits: int = GLBITS, vocab: int = VOCAB):
        self.state = seed & MASK32
        self.enabled = (self.state != 0)       # seed 0 => greedy
        self.lut = lut if lut is not None else make_gumbel_lut(glbits=glbits)
        self.glbits = glbits
        self.vocab = vocab

    def sample_token(self, logits) -> int:
        """logits: sequence of Q6.25 ints (len >= vocab). Advance the rng once per
        logit (j = 0..vocab-1), perturb, return the argmax index (first wins ties).
        State persists for the next token."""
        best_v = None
        best_i = 0
        shift = 32 - self.glbits
        for j in range(self.vocab):
            self.state = xorshift32(self.state)
            v = int(logits[j])
            if self.enabled:
                v += self.lut[self.state >> shift]
            if best_v is None or v > best_v:    # strict > : first index wins ties
                best_v = v
                best_i = j
        return best_i


if __name__ == "__main__":
    # sanity: the Gumbel-max distribution must match softmax(logit/T) sampling
    import numpy as np
    rng = np.random.default_rng(0)
    logits_f = rng.normal(0, 4, VOCAB)                 # arbitrary logits
    lq = [int(round(x * (1 << FRAC))) for x in logits_f]
    lut = make_gumbel_lut()
    # empirical Gumbel-max histogram
    from collections import Counter
    c = Counter()
    g = GumbelRng(seed=0x12345678, lut=lut)
    for _ in range(20000):
        c[g.sample_token(lq)] += 1
    # reference softmax(logit/T)
    p = np.exp((logits_f - logits_f.max()) / TEMP); p /= p.sum()
    top = np.argsort(p)[::-1][:5]
    print("token  softmax_p   gumbel_freq")
    for t in top:
        print(f"  {t:3d}   {p[t]:.4f}     {c[t]/20000:.4f}")
    print(f"LUT[{1<<GLBITS}] range [{min(lut)}, {max(lut)}]  "
          f"(logit Q6.25 ~ {min(lq)}..{max(lq)})")
