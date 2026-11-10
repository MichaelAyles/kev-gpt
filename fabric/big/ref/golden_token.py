"""One whole token of GEMV work as a compact replay file for the board (doc 10, R2a).

For one real position of a real prompt, record the input and output of EVERY GEMV of the
fabric spec (36 layers x q,k,v,o,gate,up,down + lm_head = 253 on Bonsai-8B):

    <name>.xq   int16  [n]      BFP10 mantissas of the GEMV input
    <name>.xe   int8   [G]      exponent per 128-group
    <name>.y    uint32 [R]      the fp32 result, raw IEEE bits

`board/pl_gemv.py` loads each (xq, xe) into the engine, streams the matrix from DDR and
compares the result words with `y`: bit-exact on all 253 means the board did 100% of a token's
weight traffic and arithmetic correctly.

    flock /tmp/kevbuild/gpu.lock .venv/bin/python -u -m fabric.big.ref.golden_token
        -> /tmp/kevbuild/big_vectors/Bonsai-8B-Q1_0_token.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .fabric_ref import FabricRef, Trace
from .golden import PROMPT
from .qwen3_ref import from_gguf

KINDS = ("q", "k", "v", "o", "gate", "up", "down")


def gemv_names(n_layer):
    return [f"l{L}.{k}" for L in range(n_layer) for k in KINDS] + ["lm_head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="data/big/Bonsai-8B-Q1_0.gguf")
    ap.add_argument("--tok", default="data/big/Bonsai-8B-meta")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--pos", type=int, default=-1, help="traced position (default: the last prompt token)")
    ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    toks = tok(a.prompt).input_ids
    pos = a.pos % len(toks)
    cfg, W, _ = from_gguf(a.gguf, device=a.device)
    fab = FabricRef(cfg, W, device=a.device)
    t0 = time.time()
    tr = Trace(layers=None, positions={pos}, rows=None, final=True)
    lg = fab.forward_seq(toks, trace=tr)
    d = dict(tokens=np.array(toks, dtype=np.int32), pos=np.int32(pos), argmax=np.int32(int(lg[pos].argmax())))
    for n in gemv_names(cfg.n_layer):
        xq, xe, y = tr[n + ".xq"][0], tr[n + ".xe"][0], tr[n + ".y"][0]
        assert int(xq.abs().max()) <= 511 and y.dtype == torch.float32
        d[n + ".xq"] = xq.numpy().astype(np.int16)
        d[n + ".xe"] = xe.numpy().astype(np.int8)
        d[n + ".y"] = y.contiguous().numpy().view(np.uint32).copy()
        W_ = W[n]
        assert d[n + ".y"].shape == (W_.out,) and d[n + ".xq"].shape[0] == d[n + ".xe"].shape[0] * 128
    out = Path(a.out or f"/tmp/kevbuild/big_vectors/{Path(a.gguf).stem}_token.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    print(f"GOLDEN_TOKEN {len(gemv_names(cfg.n_layer))} GEMVs at position {pos} of {len(toks)} "
          f"(next token {tok.decode([int(d['argmax'])])!r}) -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
