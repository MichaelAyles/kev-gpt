"""Golden reference + RTL test vectors for groupnorm1_vec.sv -- Stage 1 conv
front-end's groupnorm1 (gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md Stage 1 table:
`conv1d(conv1) -> tanh_ -> groupnorm1 -> conv2 -> gelu -> conv3 -> gelu ->
permute`), moonshine's own real `nn.GroupNorm(num_groups=1, num_channels=288,
eps=1e-5)` (modeling_moonshine.py: `hidden_states = self.groupnorm(hidden_states)`,
called right after `tanh(conv1(...))`).

Real conv1 + real tanh output, real audio (torch.manual_seed(0), L=3000 --
same AUDIO_SEED/AUDIO_LEN convention every other ASR gate in this project
uses), fed through this file's own integer GroupNorm reference (gn_int) --
the SAME mean/var/rsqrt core as layernorm_vec_gendiv.sv's own ln_int_gendiv
(run_layernorm_gendiv.py), generalized to ONE reduction over ALL C*T
elements plus a per-channel bias term LayerNorm's own convention doesn't
have (see groupnorm1_vec.sv's header). The gate is bit-exact RTL vs this
Python integer reference -- GroupNorm is arithmetic, not a transcendental
LUT, so CLAUDE.md's cosine>0.9999 allowance doesn't apply here (that's for
LUTs only); a real-vs-quantized cosine number is printed for information,
not as the pass/fail gate.

    .venv/bin/python -m fabric.asr_seq.pack_groupnorm1 gen --dir <sim_dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GEN2ASR_SW = os.path.expanduser("~/gen2asr/sw_model")
sys.path.insert(0, GEN2ASR_SW)
sys.path.insert(0, os.path.join(GEN2ASR_SW, "model"))

from fabric.stage3.run_layernorm import (  # noqa: E402
    QX, G_FRAC, A_FRAC, Y_FRAC, OUT_FRAC, VAR_FRAC, EPS_A, rsqrt_int, seed_table,
)

P = 8
C = 288
BETA_FRAC = 22             # == OUT_FRAC, chosen so the final add needs no shift
GN_EPS = 1e-5

AUDIO_SEED = 0
AUDIO_LEN = 3000           # same L=3000 gate convention as every other ASR gate here


def to_qfrac(x_real, frac) -> np.ndarray:
    return np.round(np.asarray(x_real, dtype=np.float64) * (1 << frac)).astype(np.int64)


def gn_int(X, G, B):
    """Integer GroupNorm(num_groups=1): X flat C*T elements (Q6.25,
    t-major/channel-minor layout -- flat index i = t*C + c, matching
    groupnorm1_vec.sv's own xbank row-major-by-time storage), G/B
    per-channel gamma/beta (C elements, Q4.20 / Q10.22). Same mean/var/
    rsqrt core as ln_int_gendiv (run_layernorm_gendiv.py), generalized:
    ONE mean/var over all len(X) elements (not per-row), and a
    per-channel affine WITH bias (ln_int_gendiv has none)."""
    X = [int(v) for v in X]
    G = [int(v) for v in G]
    B = [int(v) for v in B]
    N = len(X)
    Cn = len(G)
    assert N % Cn == 0
    S = sum(X)
    mean = S // N
    d = [v - mean for v in X]
    SS = sum(di * di for di in d)
    var = SS // N
    A = (var >> (VAR_FRAC - A_FRAC)) + EPS_A
    if A <= 0:
        A = 1
    Yr = rsqrt_int(A)
    sh = (QX + Y_FRAC + G_FRAC) - OUT_FRAC
    out = []
    for i in range(N):
        c = i % Cn
        t = d[i] * Yr
        t = t * G[c]
        yv = (t >> sh if sh >= 0 else t << (-sh)) + B[c]
        out.append(yv)
    return out, mean, var, A, Yr


def _pack_rows(vec, p):
    rows = []
    for r in range(len(vec) // p):
        val = 0
        for k in range(p):
            val |= (int(vec[r * p + k]) & 0xFFFFFFFF) << (32 * k)
        rows.append(val)
    return rows


def _w(path, vals, nib):
    mask = (1 << (4 * nib)) - 1
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v) & mask:0{nib}x}" for v in vals) + "\n")


def load_real():
    """Returns (tanh_out (C,T) real FP32 -- conv1's real output after real
    tanh, gamma (C,) real groupnorm weight, beta (C,) real groupnorm bias,
    gn_out_real (C,T) real FP32 -- PyTorch's own GroupNorm output, for an
    informational cosine check only)."""
    from transformers import MoonshineForConditionalGeneration

    model = MoonshineForConditionalGeneration.from_pretrained(
        "UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    enc = model.model.encoder

    torch.manual_seed(AUDIO_SEED)
    audio = torch.randn(1, AUDIO_LEN, dtype=torch.float32)
    with torch.no_grad():
        x = audio.unsqueeze(1)                       # (1,1,L)
        conv1_out = enc.conv1(x)                      # (1,C,T)
        tanh_out = torch.tanh(conv1_out)               # (1,C,T)
        gn_out_real = enc.groupnorm(tanh_out)           # (1,C,T), real float

    gamma = enc.groupnorm.weight.detach().numpy().astype(np.float64)   # (C,)
    beta = enc.groupnorm.bias.detach().numpy().astype(np.float64)      # (C,)
    assert gamma.shape == (C,) and beta.shape == (C,)
    return (tanh_out[0].detach().numpy().astype(np.float64),
            gamma, beta, gn_out_real[0].detach().numpy().astype(np.float64))


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tanh_np, gamma, beta, gn_out_real = load_real()   # tanh_np: (C,T)
    T = tanh_np.shape[1]
    print(f"conv1+tanh: C={C} T={T} max|tanh|={np.max(np.abs(tanh_np)):.4f} "
          f"gamma range=[{gamma.min():.5f},{gamma.max():.5f}] "
          f"beta range=[{beta.min():.5f},{beta.max():.5f}]", file=sys.stderr)

    x_flat = tanh_np.T.reshape(-1)               # (T,C) -> flat, i = t*C+c
    x_q = to_qfrac(x_flat, QX)                    # Q6.25; |tanh|<=1 so well within int32
    g_q = to_qfrac(gamma, G_FRAC)                 # Q4.20
    b_q = to_qfrac(beta, BETA_FRAC)                # Q10.22

    y_int, mean, var, A, Yr = gn_int(list(x_q.astype(object)), list(g_q.astype(object)),
                                      list(b_q.astype(object)))
    print(f"gn_int: mean={mean} var={var} A={A} Yr={Yr}", file=sys.stderr)

    # informational-only float check against PyTorch's own real GroupNorm
    y_real_from_int = np.array([int(v) for v in y_int], dtype=np.float64) / (1 << OUT_FRAC)
    y_real_ref = gn_out_real.T.reshape(-1)        # same (t*C+c) flat order
    cos = float(np.dot(y_real_from_int, y_real_ref) /
                (np.linalg.norm(y_real_from_int) * np.linalg.norm(y_real_ref) + 1e-30))
    print(f"cosine(quantized_int_output, real_float_groupnorm_output)={cos:.6f} (informational only)",
          file=sys.stderr)

    N = len(x_q)
    CROWS = C // P
    ROWS = N // P
    xrows = _pack_rows(x_q, P)
    grows = _pack_rows(g_q, P)
    brows = _pack_rows(b_q, P)
    gold = [int(v) & 0xFFFFFFFFFFFFFFFF for v in y_int]

    nib_row = (P * 32) // 4
    _w(os.path.join(out_dir, "x.mem"), xrows, nib_row)
    _w(os.path.join(out_dir, "g.mem"), grows, nib_row)
    _w(os.path.join(out_dir, "b.mem"), brows, nib_row)
    _w(os.path.join(out_dir, "seed.mem"), seed_table(), 5)

    with open(os.path.join(out_dir, "gold.txt"), "w") as fh:
        fh.write("\n".join(f"{v:016x}" for v in gold) + "\n")

    return {"C": C, "T": T, "P": P, "CROWS": CROWS, "ROWS": ROWS, "N": N, "gold": gold}


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_groupnorm1")
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gen")
    g.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    if a.cmd == "gen":
        manifest = main_gen(a.dir)
        print(f"C={manifest['C']} T={manifest['T']} N={manifest['N']}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
