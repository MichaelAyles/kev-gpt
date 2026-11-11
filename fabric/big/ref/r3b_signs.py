"""R3b: sub-1-bit signs, a KILL-OR-KEEP feasibility study (doc 10, rung R3).

The plan hopes for ~0.75 bits/weight by vector-quantising groups of sign bits against a codebook
decoded in fabric. This tool measures whether Bonsai-8B's sign planes leave room for that
without retraining:

  stage lossless  entropy of the sign bits of all 253 streamed matrices (bit, byte, 16-bit word,
                  conditional on the left / upper neighbour, on the previous byte, on the column,
                  on the row, and on a 13-bit causal context), each next to the SAME estimator run
                  on random bytes of the same shape (the estimator's own floor), plus zstd / xz /
                  bz2 on the raw sign bytes of three matrices. One number: achievable lossless bpw.
  stage vqfit     lossy VQ with no retraining: per-matrix codebooks of b-bit sign blocks (b = 8,
                  16; K entries -> log2(K)/b bits per weight), k-modes under Hamming distance on
                  each matrix's full block histogram, started from a generic codebook (a perfect
                  Hamming code where one exists). Reports the fraction of sign bits flipped per
                  kind, next to the generic codebook and the rate-distortion bound for random bits.
  stage eval      model quality of the damaged model (float reference, the R1c/R1d 1,020 token
                  ids): random flips at rate f (the damage an ideal code would do), the VQ
                  settings, perfect-Hamming-code "syndrome" coding (a table-free decoder), on all
                  matrices / FFN only / attention only / one layer / LM head only, with and
                  without the closed-form scale re-fit  s' = s * (1 - 2 * flips_in_group / 128).
                  GATE: stock pooled NLL == results/r1c_gate.json float_nll to 1e-6.
  stage report    table + R3B_VERDICT KEEP|KILL. KEEP only if some setting reaches <= 0.875 total
                  bpw with dNLL < 0.05 nats without training.

    CUDA_VISIBLE_DEVICES="" python -u -m fabric.big.ref.r3b_signs --stage lossless
    flock /tmp/kevbuild/gpu.lock python -u -m fabric.big.ref.r3b_signs --stage vqfit
    flock /tmp/kevbuild/gpu.lock python -u -m fabric.big.ref.r3b_signs --stage eval
    python -u -m fabric.big.ref.r3b_signs --stage report
"""
from __future__ import annotations

import argparse
import bz2
import json
import lzma
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .r3a_scales import BW, GGN, KV_CTX512, STREAM_ORDER, H_counts, compare, eval_model, kind_of, load_texts, stream_names

KINDS = list(STREAM_ORDER) + ["lm_head"]
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
# within-byte adjacent bit pairs (bit i, bit i+1), i = 0..6 -> counts of (left, cur) = 2*left + cur
PAIR = np.zeros((256, 4), dtype=np.int64)
for _v in range(256):
    for _i in range(7):
        PAIR[_v, 2 * ((_v >> _i) & 1) + ((_v >> (_i + 1)) & 1)] += 1
VQ_SETTINGS = [(16, 8192), (16, 4096), (8, 64), (16, 256), (8, 16)]     # (block bits, codebook entries)


def hinv(y):
    """inverse binary entropy on [0, 0.5]: the rate-distortion bound D(R) = hinv(1 - R) for
    uniform random bits under Hamming distortion."""
    lo, hi = 0.0, 0.5
    for _ in range(60):
        m = (lo + hi) / 2
        h = -(m * math.log2(m) + (1 - m) * math.log2(1 - m)) if m > 0 else 0.0
        lo, hi = (m, hi) if h < y else (lo, m)
    return (lo + hi) / 2


def h2(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1 - 1e-16)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ----------------------------------------------------------------------------- sign planes
def open_signs(path):
    """-> (GGUF, {short name: (gguf name, R, in)}) in stream order. Signs stay on the memmap."""
    from .gguf_q1 import GGUF
    g = GGUF(path)
    n_layer = int(g.meta[g.meta["general.architecture"] + ".block_count"])
    idx = {}
    for nm in stream_names(n_layer):
        if nm == "lm_head":
            gn = "output.weight" if "output.weight" in g.tensors else "token_embd.weight"
        else:
            L, k = nm[1:].split(".")
            gn = f"blk.{L}.{GGN[k]}.weight"
        ti = g.tensors[gn]
        idx[nm] = (gn, ti.n // ti.shape[0], ti.shape[0])
    return g, idx


def sign_rows(g, gn, R, inn, r0, r1):
    """packed sign bytes uint8 [r1-r0, in/8] of rows [r0, r1) (a copy)."""
    _, qs = g.q1_parts(gn)
    G = inn // 128
    return np.ascontiguousarray(qs[r0 * G:r1 * G]).reshape(r1 - r0, inn // 8)


# ----------------------------------------------------------------------------- stage: lossless
class Acc:
    def __init__(self):
        self.bits = self.ones = 0
        self.byte = np.zeros(256, np.int64)
        self.w16 = np.zeros(65536, np.int64)
        self.bpair = np.zeros(65536, np.int64)
        self.left = np.zeros(4, np.int64)
        self.up = np.zeros(4, np.int64)
        self.deep = np.zeros(1 << 14, np.int64)
        self.hcol = self.hrow = self.wcol = 0.0        # sum of bits * H(bit | column / row), per matrix

    def chunk(self, B, prev_row):
        self.bits += B.size * 8
        self.ones += int(POP[B].sum())
        self.byte += np.bincount(B.ravel(), minlength=256)
        self.w16 += np.bincount(B.view(np.uint16).ravel(), minlength=65536)
        self.bpair += np.bincount(((B[:, :-1].astype(np.uint16) << 8) | B[:, 1:]).ravel(), minlength=65536)
        self.left += (np.bincount(B.ravel(), minlength=256)[:, None] * PAIR).sum(0)
        cross = 2 * (B[:, :-1] >> 7).astype(np.int64) + (B[:, 1:] & 1)
        self.left += np.bincount(cross.ravel(), minlength=4)
        A = B if prev_row is None else np.concatenate([prev_row[None], B])
        a, b = A[:-1], A[1:]
        na, nb = ~a, ~b
        self.up += np.array([POP[na & nb].sum(), POP[na & b].sum(), POP[a & nb].sum(), POP[a & b].sum()])

    def deep_ctx(self, B):
        """13-bit causal context: 10 bits to the left in the row, and up-left / up / up-right."""
        U = np.unpackbits(B, axis=1, bitorder="little")
        n = U.shape[1]
        ctx = np.zeros((U.shape[0] - 1, n - 11), dtype=np.uint16)
        for i in range(1, 11):
            ctx |= U[1:, 10 - i:n - 1 - i].astype(np.uint16) << (i - 1)
        ctx |= U[:-1, 9:n - 2].astype(np.uint16) << 10
        ctx |= U[:-1, 10:n - 1].astype(np.uint16) << 11
        ctx |= U[:-1, 11:n].astype(np.uint16) << 12
        self.deep += np.bincount(((ctx.astype(np.int64) << 1) | U[1:, 10:n - 1]).ravel(), minlength=1 << 14)

    def result(self):
        n = self.bits
        r = dict(bits=int(n), frac_ones=self.ones / n, H_bit=float(h2(self.ones / n)),
                 H_byte=H_counts(self.byte, True) / 8, H_word16=H_counts(self.w16, True) / 16,
                 H_byte_given_prev_byte=(H_counts(self.bpair, True) - H_counts(self.bpair.reshape(256, 256).sum(1), True)) / 8,
                 H_given_left=H_counts(self.left, True) - H_counts(self.left.reshape(2, 2).sum(1), True),
                 H_given_up=H_counts(self.up, True) - H_counts(self.up.reshape(2, 2).sum(1), True),
                 agree_left=float((self.left[0] + self.left[3]) / self.left.sum()),
                 agree_up=float((self.up[0] + self.up[3]) / self.up.sum()),
                 H_given_col=self.hcol / self.wcol if self.wcol else None,
                 H_given_row=self.hrow / self.wcol if self.wcol else None)
        if self.deep.sum():
            d = self.deep.reshape(-1, 2)
            r["H_given_ctx13"] = H_counts(self.deep, True) - H_counts(d.sum(1), True)
            r["ctx13_bits"] = int(self.deep.sum())
        r["min_bpw"] = min(v for k, v in r.items() if k.startswith("H_") and v is not None)
        return r


def matrix_pass(acc, get_rows, R, inn, deep):
    """accumulate one matrix into acc. get_rows(r0, r1) -> uint8 [r, in/8]."""
    step = max(1, (1 << 28) // inn)                        # <= 32 MB of packed bytes per chunk
    col = np.zeros(inn, np.int64)
    rows = []
    prev = None
    for r0 in range(0, R, step):
        B = get_rows(r0, min(R, r0 + step))
        acc.chunk(B, prev)
        prev = B[-1].copy()
        rows.append(POP[B].sum(1))
        for s in range(0, B.shape[0], 2048):
            col += np.unpackbits(B[s:s + 2048], axis=1, bitorder="little").sum(0, dtype=np.int64)
        if deep and r0 == 0:
            acc.deep_ctx(B[:min(R, 1024)])
    rows = np.concatenate(rows)
    mm = lambda n: 1 / (2 * n * math.log(2))               # Miller-Madow for a binary variable
    acc.hcol += R * inn * float((h2(col / R) + mm(R)).mean())
    acc.hrow += R * inn * float((h2(rows / inn) + mm(inn)).mean())
    acc.wcol += R * inn


def compress_test(g, idx, names, scratch):
    out = {}
    scratch.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    items = [(n, None) for n in names] + [("random_control", 6291456)]
    for nm, nrand in items:
        if nrand:
            raw = rng.integers(0, 256, size=nrand, dtype=np.uint8).tobytes()
        else:
            gn, R, inn = idx[nm]
            raw = sign_rows(g, gn, R, inn, 0, R).tobytes()
        f = scratch / "signs.bin"
        f.write_bytes(raw)
        r = dict(raw_bytes=len(raw))
        for tool, cmd in (("zstd-19", ["zstd", "-19", "--long=27", "-q", "-c", str(f)]),
                          ("zstd-22", ["zstd", "--ultra", "-22", "--long=27", "-q", "-c", str(f)]),
                          ("xz-9e", ["xz", "-9e", "-q", "-c", str(f)])):
            if shutil.which(cmd[0]):
                r[tool] = len(subprocess.run(cmd, capture_output=True, check=True).stdout) / len(raw)
        r["lzma-9e(py)"] = len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)) / len(raw)
        r["bz2-9(py)"] = len(bz2.compress(raw, 9)) / len(raw)
        # bit-transposed (column-major) too: does a compressor find structure down the columns?
        if not nrand:
            gn, R, inn = idx[nm]
            U = np.unpackbits(np.frombuffer(raw, np.uint8).reshape(R, inn // 8), axis=1, bitorder="little")
            rawT = np.packbits(np.ascontiguousarray(U.T), axis=1, bitorder="little").tobytes()
            r["lzma-9e(py), column-major"] = len(lzma.compress(rawT, preset=9 | lzma.PRESET_EXTREME)) / len(rawT)
        f.unlink()
        out[nm] = r
        print(f"  compress {nm:15s} {len(raw) / 1e6:6.2f} MB: " + "  ".join(f"{k} {v:.5f}" for k, v in r.items() if k != "raw_bytes"))
    return out


def stage_lossless(a, doc, save):
    g, idx = open_signs(a.gguf)
    t0 = time.time()
    rng = np.random.default_rng(1)
    real = {k: Acc() for k in KINDS}
    ctrl = {k: Acc() for k in KINDS}
    deep_layers = {"l0", "l17", "l35", "lm_head"}
    for i, (nm, (gn, R, inn)) in enumerate(idx.items()):
        kd = kind_of(nm)
        deep = nm.split(".")[0] in deep_layers
        matrix_pass(real[kd], lambda r0, r1: sign_rows(g, gn, R, inn, r0, r1), R, inn, deep)
        matrix_pass(ctrl[kd], lambda r0, r1: rng.integers(0, 256, size=(r1 - r0, inn // 8), dtype=np.uint8), R, inn, deep)
        if i % 28 == 27 or nm == "lm_head":
            print(f"  [{i + 1}/{len(idx)}] {nm} ({time.time() - t0:.0f}s)")
    res = {}
    tot = sum(real[k].bits for k in KINDS)
    for k in KINDS:
        res[k] = dict(real=real[k].result(), random_control=ctrl[k].result())
    keys = [k for k in res["q"]["real"] if k.startswith("H_") or k in ("frac_ones", "agree_left", "agree_up", "min_bpw")]
    allr = {w: {k: float(sum(res[kd][w][k] * res[kd][w]["bits"] for kd in KINDS) / tot) for k in keys}
            for w in ("real", "random_control")}
    for w in allr:
        allr[w]["bits"] = int(tot)
    res["ALL"] = allr
    cols = ["frac_ones", "H_bit", "H_byte", "H_word16", "H_given_left", "H_given_up", "H_byte_given_prev_byte",
            "H_given_col", "H_given_row", "H_given_ctx13"]
    print(f"[lossless] sign-plane entropies, bits per sign bit ({tot:,} bits, {time.time() - t0:.0f}s). 'ctl' = the "
          f"same estimator on random bytes of the same shapes.")
    print(f"{'kind':8s} " + " ".join(f"{c.replace('H_', 'H:').replace('given_', '|')[:14]:>14s}" for c in cols))
    for k in KINDS + ["ALL"]:
        for w, lab in (("real", k), ("random_control", "  ctl")):
            print(f"{lab:8s} " + " ".join(f"{res[k][w][c]:14.6f}" for c in cols))
    comp = compress_test(g, idx, ["l0.q", "l17.gate", "l35.down"], Path(a.scratch))
    best_est = min(res["ALL"]["real"][c] for c in cols if c.startswith("H_"))
    best_ctl = min(res["ALL"]["random_control"][c] for c in cols if c.startswith("H_"))
    best_comp = min(v for nm, r in comp.items() if nm != "random_control" for k, v in r.items() if k != "raw_bytes")
    # the honest single number: estimates are only meaningful relative to their random control
    gain = max(0.0, best_ctl - best_est)
    doc["lossless"] = dict(per_kind=res, compressors=comp, best_entropy_estimate=best_est,
                           best_entropy_estimate_random_control=best_ctl, best_compressor_ratio=best_comp,
                           achievable_lossless_bpw=1.0 - gain,
                           note="achievable = 1 - (control - real) for the best context model: an estimator "
                                "that reads below 1.0 on random bytes is showing its bias, not structure")
    print(f"R3B_LOSSLESS best context model {best_est:.6f} b/bit (random control {best_ctl:.6f}); best real "
          f"compressor {best_comp:.5f} of raw; achievable lossless sign bpw = {1.0 - gain:.4f} (MEASURED)")
    save()


# ----------------------------------------------------------------------------- stage: vqfit
def hamming_codewords(n, extra=0):
    """all words of n + extra bits whose low n bits are a perfect Hamming [n, n - m] codeword
    (n = 2^m - 1; syndrome = XOR of the 1-based positions of the set bits = 0)."""
    w = np.arange(1 << (n + extra), dtype=np.int64)
    syn = np.zeros_like(w)
    for j in range(n):
        syn ^= ((w >> j) & 1) * (j + 1)
    return w[syn == 0]


def kmodes(counts, K, b, init, iters, dev):
    """weighted k-modes on the full pattern histogram. counts [2^b]; init int64 [K] patterns.
    -> (lut int64 [2^b] pattern -> codeword, flip fraction, codebook int64 [K])."""
    import torch
    N = 1 << b
    ar = torch.arange(N, device=dev)
    sh = torch.arange(b, device=dev)
    P = (((ar[:, None] >> sh) & 1).float() * 2 - 1)                    # [N, b]
    w = torch.as_tensor(counts, dtype=torch.float32, device=dev)
    C = (((torch.as_tensor(init, device=dev)[:, None] >> sh) & 1).float() * 2 - 1)

    def assign(C):
        return torch.cat([(P[i:i + 4096] @ C.T).argmax(1) for i in range(0, N, 4096)])

    for _ in range(iters):
        a = assign(C)
        S = torch.zeros(K, b, device=dev).index_add_(0, a, P * w[:, None])
        n = torch.where(S > 0, 1.0, torch.where(S < 0, -1.0, C))
        if torch.equal(n, C):
            break
        C = n
    a = assign(C)
    rec = C[a]
    dist = (b - (P * rec).sum(1)) / 2
    flips = float((w.double() * dist.double()).sum() / (w.double().sum() * b))
    code = ((C > 0).long() << sh).sum(1)
    return code[a].cpu().numpy(), flips, code.cpu().numpy()


def generic_codebook(b, K, dev, restarts=4, iters=40):
    """the codebook for structureless (uniform) blocks: a perfect Hamming code where one fits,
    else the best of a few random-start k-modes runs on a flat histogram."""
    flat = np.ones(1 << b)
    best = None
    inits = []
    if (b, K) == (16, 4096):
        inits.append(("hamming[15,11] + 1 raw bit", hamming_codewords(15, 1)))
    if (b, K) == (16, 8192):       # two cosets of the same code: every word within distance 1, 1/8 exact
        cw = hamming_codewords(15, 1)
        inits.append(("two cosets of hamming[15,11] + 1 raw bit", np.concatenate([cw, cw ^ 1])))
    rng = np.random.default_rng(b * 100003 + K)
    for i in range(restarts):
        inits.append((f"random start {i}", rng.choice(1 << b, size=K, replace=False)))
    for lab, init in inits:
        lut, fl, code = kmodes(flat, K, b, init, iters, dev)
        if best is None or fl < best[1]:
            best = (lut, fl, code, lab)
    return best


def stage_vqfit(a, doc, save):
    import torch
    dev = a.device
    g, idx = open_signs(a.gguf)
    lut_dir = Path(a.lut_dir)
    lut_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # block histograms of every matrix (CPU; the whole data set of a b-bit VQ is its histogram)
    h16, h8 = {}, {}
    for nm, (gn, R, inn) in idx.items():
        c16 = np.zeros(65536, np.int64)
        c8 = np.zeros(256, np.int64)
        step = max(1, (1 << 28) // inn)
        for r0 in range(0, R, step):
            B = sign_rows(g, gn, R, inn, r0, min(R, r0 + step))
            c16 += np.bincount(B.view(np.uint16).ravel(), minlength=65536)
            c8 += np.bincount(B.ravel(), minlength=256)
        h16[nm], h8[nm] = c16, c8
    print(f"[vqfit] histograms of {len(idx)} matrices ({time.time() - t0:.0f}s)")
    out = doc.setdefault("vq", {})
    for b, K in VQ_SETTINGS:
        key = f"vq{b}_{K}"
        rate = math.log2(K) / b
        glut, gfl, gcode, glab = generic_codebook(b, K, dev)
        hh = h16 if b == 16 else h8
        luts, per = {}, {}
        x = np.arange(1 << b) ^ glut
        d0 = (POP[x & 255] + POP[x >> 8]).astype(np.float64)
        for nm in idx:
            # flips of the generic codebook on this matrix, then per-matrix refinement from it
            cnt = hh[nm]
            f_gen = float((cnt * d0).sum() / (cnt.sum() * b))
            lut, fl, _ = kmodes(cnt, K, b, gcode, a.vq_iters, dev)
            luts[nm] = lut.astype(np.uint16 if b == 16 else np.uint8)
            per[nm] = dict(flips=fl, flips_generic=f_gen, bits=int(cnt.sum() * b))
        np.savez(lut_dir / f"{key}.npz", **luts)
        kinds = {}
        for kd in KINDS + ["ALL"]:
            names = [n for n in per if kd == "ALL" or kind_of(n) == kd]
            w = np.array([per[n]["bits"] for n in names], dtype=np.float64)
            kinds[kd] = dict(flips=float((np.array([per[n]["flips"] for n in names]) * w).sum() / w.sum()),
                             flips_generic=float((np.array([per[n]["flips_generic"] for n in names]) * w).sum() / w.sum()))
        out[key] = dict(block_bits=b, K=K, sign_bpw=rate, flips_uniform_generic=gfl, generic_init=glab,
                        rd_bound_flips=hinv(1 - rate), per_kind=kinds,
                        decoder=f"{K} x {b} b codebook ROM per matrix ({K * b / 8 / 1024:.1f} KB, reloaded per matrix), "
                                f"{128 // b} lookups per 128-group")
        print(f"  {key:10s} {rate:.4f} sign bpw: flips per-matrix codebook {kinds['ALL']['flips']:.4%}, generic "
              f"codebook {kinds['ALL']['flips_generic']:.4%} (on uniform blocks {gfl:.4%}, from {glab}), "
              f"rate-distortion bound for random bits {hinv(1 - rate):.4%}  ({time.time() - t0:.0f}s)")
        print("      per kind: " + "  ".join(f"{kd} {kinds[kd]['flips']:.4%}" for kd in KINDS))
        save()
    # perfect Hamming codes inside each 128-group: analytic rates (verified at eval time)
    ham = {}
    for n in (7, 15, 31, 63, 127):
        m = int(math.log2(n + 1))
        nb = 128 // n
        ham[f"ham{n}"] = dict(n=n, blocks_per_group=nb, raw_bits_per_group=128 - nb * n,
                              sign_bpw=(nb * (n - m) + 128 - nb * n) / 128,
                              flips_expected=nb * (n / (n + 1)) / 128,
                              rd_bound_flips=hinv(1 - (nb * (n - m) + 128 - nb * n) / 128),
                              decoder=f"no table: {nb} x {m} parity bits per group, each an XOR of {(n + 1) // 2 - 1} stored bits")
        print(f"  ham{n:<3d}     {ham[f'ham{n}']['sign_bpw']:.4f} sign bpw: flips {ham[f'ham{n}']['flips_expected']:.4%} "
              f"(DERIVED), bound {ham[f'ham{n}']['rd_bound_flips']:.4%}")
    doc["hamming"] = ham
    save()


# ----------------------------------------------------------------------------- sign transforms (GPU)
def t_lut(signs, lut, b):
    """signs uint8 [R, nb] -> signs with every b-bit block replaced by its codeword."""
    import torch
    out = torch.empty_like(signs)
    flat, o = signs.view(-1), out.view(-1)
    for i in range(0, flat.numel(), 1 << 24):
        if b == 8:
            o[i:i + (1 << 24)] = lut[flat[i:i + (1 << 24)].long()]
            continue
        w = flat[i:i + (1 << 24)].view(torch.int16).to(torch.int32) & 0xFFFF
        o[i:i + (1 << 24)] = lut[w.long()].view(torch.uint8)
    return out


def _pack(bits):
    import torch
    sh = torch.arange(8, device=bits.device, dtype=torch.uint8)
    return (bits.view(bits.shape[0], -1, 8) << sh).sum(-1).to(torch.uint8)


def _unpack(s):
    import torch
    sh = torch.arange(8, device=s.device, dtype=torch.uint8)
    return ((s.unsqueeze(-1) >> sh) & 1).view(s.shape[0], -1)


def t_hamming(signs, n):
    """every 128-group: floor(128/n) blocks of n bits snapped to the nearest perfect-Hamming
    codeword (flip the bit the syndrome names), the remaining bits stored raw."""
    import torch
    dev = signs.device
    m = int(math.log2(n + 1))
    nb = 128 // n
    PB = (((torch.arange(1, n + 1, device=dev)[:, None] >> torch.arange(m, device=dev)) & 1)).float()   # [n, m]
    pw = (1 << torch.arange(m, device=dev)).float()
    out = torch.empty_like(signs)
    R, nbytes = signs.shape
    step = max(1, (1 << 24) // (nbytes * 8))
    for r0 in range(0, R, step):
        bits = _unpack(signs[r0:r0 + step]).view(-1, 128)                # [groups, 128]
        blk = bits[:, :nb * n].reshape(-1, nb, n)
        syn = ((blk.float() @ PB).remainder(2) * pw).sum(-1).long()     # [groups, nb] in 0..n
        idx = (syn - 1).clamp(min=0).unsqueeze(-1)
        cur = blk.gather(-1, idx)
        blk.scatter_(-1, idx, torch.where(syn.unsqueeze(-1) > 0, 1 - cur, cur))
        bits = torch.cat([blk.view(-1, nb * n), bits[:, nb * n:]], 1)
        out[r0:r0 + step] = _pack(bits.reshape(-1, nbytes * 8))
    return out


def t_randflip(signs, f, gen):
    import torch
    out = torch.empty_like(signs)
    R, nbytes = signs.shape
    step = max(1, (1 << 24) // (nbytes * 8))
    sh = torch.arange(8, device=signs.device, dtype=torch.uint8)
    for r0 in range(0, R, step):
        s = signs[r0:r0 + step]
        m = (torch.rand(s.shape[0], nbytes, 8, device=signs.device, generator=gen) < f).to(torch.uint8)
        out[r0:r0 + step] = s ^ (m << sh).sum(-1).to(torch.uint8)
    return out


def flips_per_group(a, b, pop):
    """[R, G] number of differing sign bits per 128-group."""
    import torch
    step = max(1, (1 << 24) // a.shape[1])
    return torch.cat([pop[(a[r:r + step] ^ b[r:r + step]).long()].view(-1, a.shape[1] // 16, 16).sum(-1)
                      for r in range(0, a.shape[0], step)])


def refit_scales(scales, fl):
    """closed form: argmin_s' || s*w - s'*w' ||^2 over the group = s * (w . w') / 128."""
    import torch
    s = scales.to(torch.float64) * (1 - 2 * fl.to(torch.float64) / 128)
    return s.to(torch.float16)


# scheme name -> (transform spec, target, refit), most important first
def scheme_list():
    L = []
    for f in (0.001, 0.003, 0.01, 0.03):
        L.append((f"rand{f:g}", ("rand", f), "all", False))
    for n in (127, 63, 31, 15):
        L.append((f"ham{n}", ("ham", n), "all", False))
    for n in (31, 15, 127):
        L.append((f"ham{n}+refit", ("ham", n), "all", True))
    L.append(("vq16_4096", ("vq", 16, 4096), "all", False))
    L.append(("vq16_4096+refit", ("vq", 16, 4096), "all", True))
    for b, K in ((16, 8192), (8, 64), (16, 256)):
        L.append((f"vq{b}_{K}", ("vq", b, K), "all", False))
    for tgt in ("ffn", "attn", "layer17", "lm_head"):
        L.append((f"ham31@{tgt}", ("ham", 31), tgt, False))
    L.append(("ham31+refit@ffn", ("ham", 31), "ffn", True))
    for tgt in ("ffn", "layer17"):
        L.append((f"vq16_4096@{tgt}", ("vq", 16, 4096), tgt, False))
    for f in (0.0003, 0.0001):
        L.append((f"rand{f:g}", ("rand", f), "all", False))
    for tgt in ("ffn", "attn", "layer17", "layer0", "layer35", "lm_head"):
        L.append((f"rand0.01@{tgt}", ("rand", 0.01), tgt, False))
    L.append(("rand0.01+refit", ("rand", 0.01), "all", True))
    # protected variants: the top p% of each matrix's groups BY SCALE keep their raw sign bits (R3a
    # found the large scales to be where the model is sensitive); the decoder can tell from the scale
    for nm, spec in (("ham31^p1", ("ham", 31)), ("ham31^p10", ("ham", 31)), ("ham63^p10", ("ham", 63))):
        L.append((nm, spec, "all", False))
    return L


def in_target(nm, tgt):
    kd = kind_of(nm)
    if tgt == "all":
        return True
    if tgt == "ffn":
        return kd in ("gate", "up", "down")
    if tgt == "attn":
        return kd in ("q", "k", "v", "o")
    if tgt == "lm_head":
        return nm == "lm_head"
    if tgt.startswith("layer"):
        return nm.startswith(f"l{tgt[5:]}.")
    raise ValueError(tgt)


def stage_eval(a, doc, save):
    import torch
    from .qwen3_ref import FloatGemv, Q1Matrix, Qwen3Ref, from_gguf
    t_start = time.time()
    dev = a.device
    texts = load_texts(a)
    torch.cuda.reset_peak_memory_stats()
    cfg, W, g = from_gguf(a.gguf, device=dev)
    _, idx = open_signs(a.gguf)
    ref = Qwen3Ref(cfg, W, device=dev, gemv=FloatGemv())
    names = list(idx)
    stock_scales = {n: W[n].scales for n in names}
    pop = torch.as_tensor(POP, device=dev).to(torch.int16)
    nbits = {n: idx[n][1] * idx[n][2] for n in names}
    print(f"[eval] model loaded ({time.time() - t_start:.0f}s)")
    nll0, am0 = eval_model(ref, texts)
    n = sum(len(v) for v in nll0.values())
    pooled = float(sum(v.sum() for v in nll0.values()) / n)
    want = json.loads(Path(a.r1c).read_text())["float_nll"]
    ok = abs(pooled - want) < 1e-6 and n == 1020
    print(f"R3B_SANITY stock binary-8B pooled NLL {pooled:.6f} vs r1c_gate float_nll {want:.6f} "
          f"|d|={abs(pooled - want):.2e} over {n} tokens {'PASS' if ok else 'FAIL'}")
    doc["eval"] = dict(doc.get("eval", {}), stock_nll=pooled, r1c_float_nll=want, ntok=n, sanity_ok=ok)
    assert ok, "stock model does not reproduce r1c_gate float_nll: stop and fix"
    res = doc.setdefault("schemes", {})
    luts = {}
    todo = [s for s in scheme_list() if not a.schemes or s[0] in a.schemes]
    for name, spec, tgt, refit in todo:
        if name in res and not a.redo:
            continue
        if time.time() - t_start > a.budget_s:
            print(f"[eval] budget of {a.budget_s}s reached: rerun to continue")
            break
        t0 = time.time()
        if spec[0] == "vq":
            key = f"vq{spec[1]}_{spec[2]}"
            if key not in luts:
                luts.clear()
                f = Path(a.lut_dir) / f"{key}.npz"
                assert f.exists(), f"run --stage vqfit first ({f})"
                luts[key] = np.load(f)
        gen = torch.Generator(device=dev).manual_seed(1234)
        touched, nflip, nb_t, nprot = [], 0, 0, 0
        protect = float(name.split("^p")[1]) / 100 if "^p" in name else 0.0
        for nm in names:
            if not in_target(nm, tgt):
                continue
            M = W[nm]
            if spec[0] == "rand":
                new = t_randflip(M.signs, spec[1], gen)
            elif spec[0] == "ham":
                new = t_hamming(M.signs, spec[1])
            else:
                lut = torch.as_tensor(luts[key][nm].astype(np.int64), device=dev)
                lut = lut.to(torch.uint8) if spec[1] == 8 else lut.to(torch.int16)   # 16-bit: two's complement view
                new = t_lut(M.signs, lut, spec[1])
            if protect:
                sc32 = M.scales.to(torch.float32)
                keep = sc32 >= torch.quantile(sc32.flatten(), 1 - protect)          # [R, G]
                new = torch.where(keep.repeat_interleave(16, dim=1), M.signs, new)
                nprot += int(keep.sum()) * 128
            fl = flips_per_group(M.signs, new, pop)
            nflip += int(fl.sum())
            nb_t += nbits[nm]
            sc = refit_scales(M.scales, fl) if refit else M.scales
            M.signs.copy_(new)                       # in place: never two copies of the sign planes
            del new, fl
            W[nm] = Q1Matrix(M.signs, sc)
            touched.append(nm)
        tb = time.time() - t0
        nll, am = eval_model(ref, texts)
        r = compare(nll, am, nll0, am0)
        r.update(target=tgt, refit=refit, spec=list(spec), flips=nflip / max(1, nb_t),
                 frac_of_stream_bits=(nb_t - nprot) / sum(nbits.values()), protected_frac=nprot / max(1, nb_t))
        res[name] = r
        # restore the touched sign planes from the GGUF and the stock scales
        for nm in touched:
            gn, R, inn = idx[nm]
            _, qs = g.q1_parts(gn)
            W[nm].signs.copy_(torch.from_numpy(np.array(qs).reshape(R, inn // 8)))
            W[nm] = Q1Matrix(W[nm].signs, stock_scales[nm])
        save()
        print(f"  {name:22s} flips {r['flips']:.4%} of {r['frac_of_stream_bits']:.1%} of the bits: dNLL "
              f"{r['dnll']:+.4f} +- {r['se']:.4f}  ex-selfgen {r['dnll_ex_selfgen']:+.4f}  top1 {r['top1']:.4f}  "
              f"(build {tb:.0f}s, total {time.time() - t0:.0f}s)")
    # the restore path must give the stock model back bit-for-bit
    nll1, _ = eval_model(ref, texts)
    back = all(np.array_equal(nll0[k], nll1[k]) for k in nll0)
    doc["eval"].update(restore_bit_exact=back, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                       gpu_seconds=doc["eval"].get("gpu_seconds", 0) + time.time() - t_start)
    print(f"R3B_RESTORE stock model after all schemes: per-token NLL identical = {back}")
    print(f"[eval] GPU peak {doc['eval']['gpu_peak_bytes'] / 1e9:.2f} GB, {time.time() - t_start:.0f}s")
    assert back
    save()


# ----------------------------------------------------------------------------- stage: report
def sign_bpw_of(spec, doc):
    if spec[0] == "ham":
        return doc["hamming"][f"ham{spec[1]}"]["sign_bpw"]
    if spec[0] == "vq":
        return math.log2(spec[2]) / spec[1]
    f = spec[1]
    return 1 - float(h2(f))          # the rate an IDEAL lossy code needs for this flip rate


def stage_report(a, doc, save):
    res = doc.get("schemes", {})
    sc_bpw = a.scale_bpw
    nw = 7567003648
    print(f"\nR3b summary. dNLL vs stock binary-8B (nats/token, paired SE, 1,020 tokens, float reference, MEASURED). "
          f"total bpw = sign bpw on the targeted matrices (1.0 elsewhere) + {sc_bpw} bpw of scales (the R3a "
          f"row-aligned 4-bit scheme). rand* rows: sign bpw is the rate-distortion bound 1 - h(f), i.e. what an "
          f"ideal code of unbounded block length would need for that many flips.")
    print(f"{'scheme':22s} {'target':8s} {'flips':>8s} {'sign bpw':>8s} {'total bpw':>9s} {'tok/s':>6s} | "
          f"{'dNLL +- SE':>18s} {'ex-selfgen':>10s} {'top1':>6s}")
    rows = []
    for k, r in res.items():
        sb = sign_bpw_of(r["spec"], doc)
        fr = r["frac_of_stream_bits"]
        tot = fr * sb + (1 - fr) * 1.0 + sc_bpw
        r["sign_bpw"], r["total_bpw"] = sb, tot
        r["toks"] = BW / (tot * nw / 8)
        r["toks_kv512"] = BW / (tot * nw / 8 + KV_CTX512)
        rows.append((k, r))
        print(f"{k:22s} {r['target']:8s} {r['flips']:8.4%} {sb:8.4f} {tot:9.4f} {r['toks']:6.2f} | "
              f"{r['dnll']:+9.4f} +- {r['se']:.4f} {r['dnll_ex_selfgen']:+10.4f} {r['top1']:6.4f}")
    keep = [(k, r) for k, r in rows if r["total_bpw"] <= 0.875 + 1e-9 and r["dnll"] < 0.05 and r["spec"][0] != "rand"]
    ll = doc.get("lossless", {})
    ev = []
    if ll:
        ev.append(f"lossless sign bpw {ll['achievable_lossless_bpw']:.4f} (best compressor {ll['best_compressor_ratio']:.4f} of raw)")
    cand = [(k, r) for k, r in rows if r["total_bpw"] <= 0.875 + 1e-9 and r["spec"][0] != "rand" and r["target"] == "all"]
    if cand:
        k, r = min(cand, key=lambda kr: kr[1]["dnll"])
        ev.append(f"best all-matrix setting at <= 0.875 total bpw: {k} ({r['total_bpw']:.4f} bpw, {r['flips']:.2%} of "
                  f"signs flipped) dNLL {r['dnll']:+.3f} +- {r['se']:.3f}, top-1 {r['top1']:.3f}")
    rnd = sorted([(r["flips"], r["dnll"], r["se"], k) for k, r in rows if r["spec"][0] == "rand" and r["target"] == "all"
                  and not r["refit"]])
    okf = [x for x in rnd if x[1] < 0.05]
    if rnd:
        bad = [x for x in rnd if x[1] >= 0.05]
        if okf and bad and okf[-1][1] > 0:
            # log-log interpolation of the flip rate at which random flips cost 0.05 nats
            (f0, d0, _, _), (f1, d1, _, _) = okf[-1], bad[0]
            fs = f0 * (0.05 / d0) ** (math.log(f1 / f0) / math.log(d1 / d0))
            doc["flip_tolerance"] = dict(f_at_0p05_nats=fs, bracket=[[f0, d0], [f1, d1]], ideal_bpw_saved=float(h2(fs)))
            ev.append(f"random sign flips cost 0.05 nats at about {fs:.2%} flipped (interpolated between {f0:.1%}: "
                      f"{d0:+.3f} and {f1:.1%}: {d1:+.3f}), which even an ideal code of unbounded block length turns into "
                      f"only h({fs:.4f}) = {float(h2(fs)):.3f} bpw saved; 0.75 bpw needs >= {hinv(0.25):.1%} flipped")
        elif okf:
            f = okf[-1][0]
            ev.append(f"largest random flip rate with dNLL < 0.05: {f:.3%} (dNLL {okf[-1][1]:+.3f}), which even an ideal "
                      f"code turns into only {float(h2(f)):.4f} bpw saved")
        else:
            ev.append(f"no tested random flip rate stays under dNLL 0.05 (smallest tested {rnd[0][0]:.3%}: dNLL {rnd[0][1]:+.3f})")
    # the cheapest point that does stay under 0.05 nats, whatever its bpw (the honest consolation prize)
    under = [(k, r) for k, r in rows if r["dnll"] < 0.05 and r["spec"][0] != "rand" and r["target"] == "all"
             and r["total_bpw"] < 1.0 + sc_bpw - 1e-9]
    if under:
        k, r = min(under, key=lambda kr: kr[1]["total_bpw"])
        ev.append(f"smallest all-matrix setting under 0.05 nats: {k} at {r['total_bpw']:.4f} total bpw ({r['toks']:.2f} tok/s "
                  f"weights-only), dNLL {r['dnll']:+.3f} +- {r['se']:.3f}, top-1 {r['top1']:.3f}")
    rf = [(res[k]["dnll"], res[k + "+refit"]["dnll"], k) for k in res if k + "+refit" in res]
    if rf:
        ev.append("closed-form scale re-fit: " + ", ".join(f"{k} {a_:+.3f} -> {b_:+.3f}" for a_, b_, k in rf)
                  + (" (it does not help)" if all(b_ >= a_ - 0.01 for a_, b_, _ in rf) else ""))
    verdict = "KEEP" if keep else "KILL"
    r3a = Path(a.out).with_name("r3a_scales.json")
    instead = ("what it would take instead: retraining (quantisation-aware, with the codebook / code constraint in the "
               "loop) so the model learns signs that live on the code, which is out of scope here")
    if r3a.exists():
        d3 = json.loads(r3a.read_text())
        rec = d3.get("recommendation", {}).get("best_rowal")
        if rec:
            q, st = d3["schemes"][rec], d3["schemes"]["stock_fp16"]
            instead += (f"; the realistic R3 saving is therefore R3a only: {rec} scales, {st['bytes_rowal'] / 1e6:.1f} -> "
                        f"{q['bytes_rowal'] / 1e6:.1f} MB/token, {st['toks_rowal']:.2f} -> {q['toks_rowal']:.2f} tok/s "
                        f"weights-only ({st['toks_rowal_kv512']:.2f} -> {q['toks_rowal_kv512']:.2f} with KV at ctx 512), DERIVED")
    doc["verdict"] = dict(verdict=verdict, evidence=ev, keep_settings=[k for k, _ in keep], scale_bpw_assumed=sc_bpw,
                          instead=instead,
                          rule="KEEP only if some setting reaches <= 0.875 total bpw with dNLL < 0.05 nats, no training")
    print(f"R3B_VERDICT {verdict}: " + "; ".join(ev) + (". " + instead[0].upper() + instead[1:] + "." if not keep else ""))
    save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["lossless", "vqfit", "eval", "report"],
                    choices=["lossless", "vqfit", "eval", "report"])
    ap.add_argument("--gguf", default="data/big/Bonsai-8B-Q1_0.gguf")
    ap.add_argument("--tok", default="data/big/Bonsai-8B-meta")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--out", default="fabric/big/results/r3b_signs.json")
    ap.add_argument("--chat-cache", default="fabric/big/results/Bonsai-8B-Q1_0_chatgen_T256.json")
    ap.add_argument("--r1c", default="fabric/big/results/r1c_gate.json")
    ap.add_argument("--lut-dir", default="/tmp/kevbuild/r3b")
    ap.add_argument("--scratch", default="/tmp/kevbuild/r3b/scratch")
    ap.add_argument("--vq-iters", type=int, default=8)
    ap.add_argument("--schemes", nargs="+", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--budget-s", type=float, default=400)
    ap.add_argument("--scale-bpw", type=float, default=0.03125, help="scale cost added to every total (R3a cb4/sep3)")
    a = ap.parse_args()
    out = Path(a.out)
    doc = json.loads(out.read_text()) if out.exists() else {}
    doc.update(model=Path(a.gguf).name, bw_bytes_per_s=BW)

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(out)

    if "lossless" in a.stage:
        stage_lossless(a, doc, save)
    if "vqfit" in a.stage:
        stage_vqfit(a, doc, save)
    if "eval" in a.stage:
        stage_eval(a, doc, save)
    if "report" in a.stage:
        stage_report(a, doc, save)


if __name__ == "__main__":
    main()
