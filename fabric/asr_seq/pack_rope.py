"""Golden reference + ROM/vector packer for rope_apply_vec.sv -- gates the
ONE genuinely new RTL primitive needed to reuse checkpoint C's already-
proven causal-attention RTL (kv_bank.sv + vec_attn_w.sv) for ASR's
decoder self-attention. See rope_apply_vec.sv's own header for the full
rationale and fixed-point convention (Q.16 head lanes, Q1.15 cos/sin ROM,
rsh_round-to-Q.16 after the Q.16 x Q1.15 = Q.31 product).

Integer-exact reference (not float-tolerance): builds the cos/sin ROM at
Q1.15 from real float64 math (same rope_table_at formula ops.c uses --
inv_freq[i] = 1/theta^(2i/rot_dim)), then reproduces the RTL's OWN
fixed-point rounding (round-half-away-from-zero, rsh_round) in Python so
the check is bit-exact against what the RTL actually computes -- not
"close to the true float rotation", which would let a real fixed-point
bug hide inside an accepted tolerance.

Usage:
    .venv/bin/python -m fabric.asr_seq.pack_rope gen --dir <sim_dir>
    .venv/bin/python -m fabric.asr_seq.pack_rope check --dir <sim_dir>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HEAD_DIM = 36
ROT_DIM = 32
ROT_PAIRS = 16
TMAX = 128
THETA = 10000.0
Q16 = 1 << 16
Q15 = 1 << 15


def rsh_round(v: int, s: int) -> int:
    """round-half-away-from-zero right shift -- bit-for-bit the RTL's own
    rsh_round function (vec_attn_w.sv/sequencer_fast.sv's convention)."""
    if s <= 0:
        return v << (-s)
    half = 1 << (s - 1)
    if v >= 0:
        return (v + half) >> s
    return -(((-v) + half) >> s)


def build_cos_sin_rom():
    """[TMAX][ROT_PAIRS] Q1.15 signed 16-bit ints, matching rope_table_at's
    own inv_freq/angle formula exactly (float64, then rounded to Q1.15 --
    the ROM's own quantization, a real, deliberate, one-time approximation
    of the true irrational cos/sin values, not a bug)."""
    cos_rom = np.zeros((TMAX, ROT_PAIRS), dtype=np.int32)
    sin_rom = np.zeros((TMAX, ROT_PAIRS), dtype=np.int32)
    for i in range(ROT_PAIRS):
        inv_freq = 1.0 / (THETA ** (2.0 * i / ROT_DIM))
        for pos in range(TMAX):
            angle = pos * inv_freq
            c = int(np.round(np.cos(angle) * Q15))
            s = int(np.round(np.sin(angle) * Q15))
            c = max(-Q15, min(Q15 - 1, c))
            s = max(-Q15, min(Q15 - 1, s))
            cos_rom[pos, i] = c
            sin_rom[pos, i] = s
    return cos_rom, sin_rom


def rope_apply_ref(head_q16: np.ndarray, position: int, cos_rom, sin_rom,
                    post_scale_q16: int = 65536) -> np.ndarray:
    """Integer-exact reference for what the RTL computes -- same Q.16 x
    Q1.15 -> rsh_round(.., 15) -> Q.16 pipeline, element by element, then
    POST_SCALE_Q16 applied uniformly to every lane (default 65536 = 1.0,
    a true no-op -- see rope_apply_vec.sv's own header for why this
    exists: compensating vec_attn_w.sv's HEAD_DIM=64-specific SCORE_SH,
    not part of RoPE's own math)."""
    out = np.zeros_like(head_q16)
    for i in range(ROT_PAIRS):
        x1, x2 = int(head_q16[2 * i]), int(head_q16[2 * i + 1])
        c, s = int(cos_rom[position, i]), int(sin_rom[position, i])
        r1 = rsh_round(x1 * c - x2 * s, 15)
        r2 = rsh_round(x2 * c + x1 * s, 15)
        out[2 * i] = rsh_round(r1 * post_scale_q16, 16)
        out[2 * i + 1] = rsh_round(r2 * post_scale_q16, 16)
    for i in range(ROT_DIM, HEAD_DIM):
        out[i] = rsh_round(int(head_q16[i]) * post_scale_q16, 16)
    return out


def to_u32_hex(v: int) -> str:
    return format(v & 0xFFFFFFFF, "08x")


def to_u16_hex(v: int) -> str:
    return format(v & 0xFFFF, "04x")


def write_case(out_dir: str, seed: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    cos_rom, sin_rom = build_cos_sin_rom()
    with open(os.path.join(out_dir, "rope_cos.mem"), "w") as f:
        for pos in range(TMAX):
            for i in range(ROT_PAIRS):
                f.write(to_u16_hex(int(cos_rom[pos, i])) + "\n")
    with open(os.path.join(out_dir, "rope_sin.mem"), "w") as f:
        for pos in range(TMAX):
            for i in range(ROT_PAIRS):
                f.write(to_u16_hex(int(sin_rom[pos, i])) + "\n")

    # Real decode-step positions (0..3, this project's own DECODE_STEPS)
    # plus a few positions spanning the rest of TMAX, so the gate isn't
    # only exercising the narrow range every other native gate already
    # covers -- matches this project's own "verify with seeded sweep"
    # discipline (don't trust one narrow sample).
    positions = [0, 1, 2, 3, 39, 63, 100, 127]
    cases = []
    for idx, pos in enumerate(positions):
        head_q16 = rng.integers(-(1 << 20), 1 << 20, size=HEAD_DIM).astype(np.int64)
        ref = rope_apply_ref(head_q16, pos, cos_rom, sin_rom)
        cases.append({"position": pos, "head_in": head_q16.tolist(), "ref_out": ref.tolist()})

    with open(os.path.join(out_dir, "head_in.mem"), "w") as f:
        for c in cases:
            for v in c["head_in"]:
                f.write(to_u32_hex(int(v)) + "\n")
    with open(os.path.join(out_dir, "ref_out.mem"), "w") as f:
        for c in cases:
            for v in c["ref_out"]:
                f.write(to_u32_hex(int(v)) + "\n")
    with open(os.path.join(out_dir, "positions.mem"), "w") as f:
        for c in cases:
            f.write(format(c["position"], "08x") + "\n")

    manifest = {"n_cases": len(cases), "head_dim": HEAD_DIM, "rot_dim": ROT_DIM,
                "rot_pairs": ROT_PAIRS, "tmax": TMAX, "positions": positions}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def check(out_dir: str) -> bool:
    with open(os.path.join(out_dir, "manifest.json")) as f:
        man = json.load(f)

    def load(name):
        with open(os.path.join(out_dir, name)) as f:
            return [int(line, 16) for line in f if line.strip()]

    def to_signed32(v):
        v &= 0xFFFFFFFF
        return v - (1 << 32) if v & 0x80000000 else v

    ref = [to_signed32(v) for v in load("ref_out.mem")]
    got = [to_signed32(v) for v in load("got_out.mem")]

    n = man["n_cases"] * man["head_dim"]
    mismatches = sum(1 for a, b in zip(ref[:n], got[:n]) if a != b)
    ok = mismatches == 0 and len(got) >= n
    print(f"ROPE_APPLY_VEC_VERDICT bitexact={ok} mismatches={mismatches} "
          f"n_cases={man['n_cases']} head_dim={man['head_dim']} total_elems={n}")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_rope")
    p.add_argument("cmd", choices=["gen", "check"])
    p.add_argument("--dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "gen":
        man = write_case(a.dir, a.seed)
        print(f"GEN dir={a.dir} n_cases={man['n_cases']} positions={man['positions']}")
    else:
        ok = check(a.dir)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
