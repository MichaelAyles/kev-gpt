"""Transposed wide-word packing + numpy golden for gemv_banked_resident_vec —
the P-WIDE-boundary resident GEMV core, generalized this session to support
INT8 weights (WBW=8) alongside checkpoint C's own INT4 scheme (WBW=4). See
that module's own header and gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md's
"INT4-specific compute, not just storage" finding for why this generalization
exists.

Same transposed layout as pack_banked_resident.py (the older, P=1 core this
one boundary-wraps), generalized from a hardcoded 4 bits/lane to WBW bits/lane:

    word(layer, g, k) = w_base[layer] + g*K_layer + k          (word units)
    word(g,k) bit [L*WBW +: WBW] = lane(W[g*LANES + L, k])      L = 0..LANES-1

w_base[layer] is the cumulative wide-word count of every earlier layer, same
formula as pack_banked_resident.py (LANES-dependent, not WBW-dependent —
the number of WORDS per layer doesn't change with lane width, only each
word's bit width does):
    words_per_layer(M,K) = ceil(M/LANES) * K
    w_base[0] = 0;  w_base[i] = w_base[i-1] + words_per_layer(M_{i-1}, K_{i-1})

Bit-exact contract: y[m] = sum_k W[m,k]*x[k], signed WBW-bit weights, signed
INT8 activations, exact int32 accumulation. Pure numpy.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def gemv_int(int_w: np.ndarray, x_int8: np.ndarray) -> np.ndarray:
    """Exact integer GEMV — the same datapath the RTL reproduces, int32 accum."""
    return np.asarray(int_w, dtype=np.int64) @ np.asarray(x_int8, dtype=np.int64)


def words_per_layer(M: int, K: int, lanes: int) -> int:
    groups = (M + lanes - 1) // lanes
    return groups * K


def pack_transposed(int_w: np.ndarray, lanes: int, wbw: int) -> list[int]:
    """(M,K) signed WBW-bit ints -> list of (g*K+k) wide words, each LANES*WBW bits.

    Rows are zero-padded up to a multiple of LANES; padded lanes contribute a 0
    weight and their outputs are never read (rd_addr < M)."""
    int_w = np.asarray(int_w, dtype=np.int64)
    M, K = int_w.shape
    groups = (M + lanes - 1) // lanes
    pad = groups * lanes - M
    if pad:
        int_w = np.vstack([int_w, np.zeros((pad, K), dtype=np.int64)])
    mask = (1 << wbw) - 1
    words: list[int] = []
    for g in range(groups):
        rows = int_w[g * lanes:(g + 1) * lanes]          # (LANES, K)
        for k in range(K):
            w = 0
            col = rows[:, k]
            for L in range(lanes):
                w |= (int(col[L]) & mask) << (L * wbw)
            words.append(w)
    return words


def build_resident(layers: list[np.ndarray], lanes: int, wbw: int):
    """layers: list of (M,K) signed WBW-bit arrays in MODEL ORDER.
    Returns (all_words, layer_meta) where layer_meta[i] = dict(w_base, M, K, n_words)
    and all_words is the concatenated wide-word stream (w_base = running index)."""
    all_words: list[int] = []
    meta = []
    wbase = 0
    for iw in layers:
        iw = np.asarray(iw, dtype=np.int64)
        M, K = int(iw.shape[0]), int(iw.shape[1])
        words = pack_transposed(iw, lanes, wbw)
        meta.append({"w_base": wbase, "M": M, "K": K, "n_words": len(words)})
        all_words.extend(words)
        wbase += len(words)
        assert len(words) == words_per_layer(M, K, lanes)
    return all_words, meta


def write_case(out_dir: str, shapes: list[tuple[int, int]], lanes: int, wbw: int,
               p: int, seed: int = 0):
    """Generate random signed-WBW W + INT8 x PER LAYER, write the concatenated
    resident weight stream (w.mem), per-layer x/y mem files (P-wide, matching
    gemv_banked_resident_vec's own boundary), and a manifest."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    wlo, whi = -(1 << (wbw - 1)), (1 << (wbw - 1))   # signed WBW-bit range [wlo, whi-1]
    layers_w, layers_x, layers_y = [], [], []
    for (M, K) in shapes:
        assert K % p == 0, f"K={K} must be a multiple of P={p} (xmem row boundary)"
        iw = rng.integers(wlo, whi, size=(M, K)).astype(np.int64)
        x = rng.integers(-128, 128, size=K).astype(np.int64)
        layers_w.append(iw)
        layers_x.append(x)
        layers_y.append(gemv_int(iw, x))

    all_words, meta = build_resident(layers_w, lanes, wbw)
    wbits = lanes * wbw
    hexw = (wbits + 3) // 4  # hex digits per word
    with open(os.path.join(out_dir, "w.mem"), "w") as f:
        f.write("\n".join(format(w, f"0{hexw}x") for w in all_words) + "\n")

    # x/y files: P values per line (P INT8 acts per x_we write; P INT32 outputs
    # per rd_addr read), matching the core's own boundary width exactly.
    for i, (x, y) in enumerate(zip(layers_x, layers_y)):
        K = len(x)
        with open(os.path.join(out_dir, f"x{i}.mem"), "w") as f:
            for row in range(K // p):
                chunk = x[row * p:(row + 1) * p]
                # P*8 bits, lane l at bits [l*8 +: 8] -> lane 0 is the LOW byte
                packed = 0
                for l, v in enumerate(chunk):
                    packed |= (int(v) & 0xFF) << (l * 8)
                f.write(format(packed, f"0{p * 2}x") + "\n")
        with open(os.path.join(out_dir, f"y{i}.mem"), "w") as f:
            f.write("\n".join(format(int(v) & 0xFFFFFFFF, "08x") for v in y) + "\n")

    manifest = {"LANES": lanes, "WBW": wbw, "P": p, "seed": seed,
                "n_layers": len(shapes), "n_words_total": len(all_words),
                "layers": meta}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def check(out_dir: str) -> bool:
    """Compare each layer's testbench dump y{i}.out vs golden y{i}.mem at its
    w_base. Prints ONE sentinel line aggregating all layers/offsets."""
    with open(os.path.join(out_dir, "manifest.json")) as f:
        man = json.load(f)

    def load(name):
        with open(os.path.join(out_dir, name)) as f:
            return [int(line, 16) & 0xFFFFFFFF for line in f if line.strip()]

    total_mis = 0
    total_m = 0
    all_ok = True
    per_layer = []
    for i, lm in enumerate(man["layers"]):
        M = lm["M"]
        gold = load(f"y{i}.mem")
        got = load(f"y{i}.out")
        ok = len(got) >= M and gold[:M] == got[:M]
        # Only count real value mismatches among the first M (the actual
        # output rows) plus a genuine shortfall (len(got) < M, meaning the
        # RTL didn't even produce enough data). len(got) > M is EXPECTED
        # and benign whenever M isn't a multiple of P -- the P-wide readback
        # reads whole P-groups, so the tail group's extra (M % P == 0 ? 0 :
        # P - M % P) rows were never meant to be compared; counting that
        # gap as a "mismatch" (as the P=1 pack_banked_resident.py's own
        # formula does, where it's always 0 since P=1 never overshoots)
        # would contradict `ok` for a case that's actually bit-exact.
        nmis = sum(1 for a, b in zip(gold[:M], got[:M]) if a != b) + max(0, M - len(got))
        per_layer.append((i, lm["w_base"], M, lm["K"], ok, nmis))
        total_mis += nmis
        total_m += M
        all_ok = all_ok and ok

    for (i, wb, M, K, ok, nmis) in per_layer:
        print(f"  layer{i}: w_base={wb:6d} M={M:5d} K={K:5d} ok={ok} mismatches={nmis}")
    print(f"BANKED_RESIDENT_VEC_VERDICT bitexact={all_ok} mismatches={total_mis} "
          f"n_layers={man['n_layers']} LANES={man['LANES']} WBW={man['WBW']} "
          f"P={man['P']} total_elems={total_m}")
    return all_ok


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.stage3.pack_banked_resident_vec")
    p.add_argument("cmd", choices=["gen", "check"])
    p.add_argument("--dir", required=True)
    p.add_argument("--lanes", type=int, default=64)
    p.add_argument("--wbw", type=int, default=4)
    p.add_argument("--p", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shapes", default="256x128,128x128",
                    help="comma list of MxK layer shapes")
    a = p.parse_args(argv)
    shapes = [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]
    if a.cmd == "gen":
        man = write_case(a.dir, shapes, a.lanes, a.wbw, a.p, a.seed)
        print(f"GEN dir={a.dir} layers={man['n_layers']} "
              f"n_words_total={man['n_words_total']} LANES={a.lanes} WBW={a.wbw} P={a.p}")
    else:
        ok = check(a.dir)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
