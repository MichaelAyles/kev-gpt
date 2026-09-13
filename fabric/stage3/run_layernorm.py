"""LayerNorm RTL gate: bit-true vs the integer fixed-point reference, + token-stream.

    python -m fabric.stage3.run_layernorm

Op: gamma-only LayerNorm over a d=256 vector (eps=1e-5, population variance).
  mean = (1/256) sum X ; var = (1/256) sum (X-mean)^2 ; y = (X-mean)*rsqrt(var+eps)*gamma

Fixed-point datapath (the exact integer arithmetic rtl/layernorm.sv reproduces):
  - input  x     : signed Q6.25 (32-bit residual stream, pinned)
  - gamma        : signed Q4.20 (G_FRAC=20)
  - sum          : sum_i X      (40-bit), mean = sum >>> 8           [Q6.25]
  - d = X - mean : Q6.25 ; d*d  : Q12.50 (64-bit) ; SS = sum d*d ; var = SS >>> 8 [Q12.50]
  - A = (var >>> (50-26)) + eps : Q.26   (rsqrt input)
  - rsqrt(A)     : 64-entry seed LUT (priority-encoder normalize) + 2 Newton steps -> Q.26
  - y = ((d * rsqrt) * gamma) >>> (25+26+20 - OUT_FRAC)              [Q.OUT_FRAC, OUT_FRAC=22]

The output Q.22 feeds the GEMV INT8 requant (ample fraction bits). ln_int() below is the
integer reference; the RTL must reproduce it with mismatches=0 (module gate). The same
function plugged into goformer must emit a token stream IDENTICAL to the exact float model
(the binding gate) — LayerNorm is exact (cosine 1.0) so this holds cleanly.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

from fabric.stage3._simdir import kevbuild

# ---- pinned fixed-point formats ---------------------------------------------
QX = 25                 # input x: Q6.25 (signed 32-bit residual stream)
G_FRAC = 20             # gamma: signed Q4.20
A_FRAC = 26             # rsqrt input A = var+eps in Q.26
Y_FRAC = 26             # rsqrt result in Q.26
OUT_FRAC = 22           # LayerNorm output y in Q.22 (feeds GEMV requant)
VAR_FRAC = 50           # d*d is Q12.50
SEED_IDX_BITS = 6       # 64-entry rsqrt seed LUT (mantissa fraction index)
SEED_OUT_FRAC = 16      # seed stored Q1.16 (value in [1/sqrt2, 1))
EPS_A = int(round(1e-5 * (1 << A_FRAC)))   # eps in Q.26
SQRT2_Q15 = int(round(np.sqrt(2.0) * (1 << 15)))
D = 256
HERE = os.path.dirname(os.path.abspath(__file__))


def seed_table():
    """64-entry 1/sqrt(mant) seed LUT, mant in [1,2), Q1.16 (the RTL ROM)."""
    return np.array(
        [int(round((1.0 / np.sqrt(1.0 + (i + 0.5) / (1 << SEED_IDX_BITS)))
                   * (1 << SEED_OUT_FRAC))) for i in range(1 << SEED_IDX_BITS)],
        dtype=np.int64)


_SEED = seed_table()


def _msb(x: int) -> int:
    return x.bit_length() - 1


def rsqrt_int(A: int) -> int:
    """1/sqrt(a) with a = A/2^26. Returns Q.26. 64-entry seed LUT + 2 Newton steps —
    the exact integer arithmetic the RTL does."""
    A = int(A)
    if A <= 0:
        A = 1
    msb = _msb(A)
    if msb >= SEED_IDX_BITS:
        idx = (A >> (msb - SEED_IDX_BITS)) & ((1 << SEED_IDX_BITS) - 1)
    else:
        idx = (A << (SEED_IDX_BITS - msb)) & ((1 << SEED_IDX_BITS) - 1)
    seed_m = int(_SEED[idx])                       # Q1.16
    # a = mant * 2^E, E = msb - A_FRAC ; 1/sqrt(a) = seed_m * 2^(-E/2)
    E = msb - A_FRAC
    shift = Y_FRAC - SEED_OUT_FRAC
    Y0 = seed_m << shift if shift >= 0 else seed_m >> (-shift)
    half = -E
    if half >= 0:
        q, r = divmod(half, 2)
    else:
        q = -((-half + 1) // 2)
        r = half - 2 * q                           # r in {0,1}
    Y0 = Y0 << q if q >= 0 else Y0 >> (-q)
    if r == 1:
        Y0 = (Y0 * SQRT2_Q15) >> 15
    Y = Y0
    one_p5 = (3 << (2 * Y_FRAC)) >> 1              # 1.5 in Q(2*Y_FRAC)
    for _ in range(2):
        yy = Y * Y                                 # Q(2*Y_FRAC)
        ayy = (A * yy) >> A_FRAC                    # Q(2*Y_FRAC)
        term = one_p5 - (ayy >> 1)                  # 1.5 - 0.5*a*y*y
        Y = (Y * term) >> (2 * Y_FRAC)              # back to Q.Y_FRAC
    return Y


def _ln_int_quantized(X, G):
    """Core integer LayerNorm on already-quantized inputs.
    X: list/array of int Q6.25 (len d). G: int Q4.20 gamma (len d).
    Returns (Y_out_ints Q.OUT_FRAC, mean, var, A, Yr) — pure integer arithmetic.

    d = len(X) is derived from the input, not the module-level D=256 constant
    (which stays what run_layernorm.py's own standalone _validate() self-test
    exercises) -- the Genesys2 port trains smaller embedding widths, and this
    is the function the shared golden reference (seq_ref.py's _ln, used by
    model.goformer_kvq.IntKVQSequencer) actually calls per-token. The mean/var
    divide is an exact arithmetic right-shift (matching layernorm_vec.sv's own
    `sum >>> $clog2(D)`), which is only bit-exact for a power-of-2 d — every
    embedding width this project trains at (128, 256, ...) is one."""
    X = [int(v) for v in X]
    G = [int(v) for v in G]
    d_model = len(X)
    assert d_model & (d_model - 1) == 0, (
        f"_ln_int_quantized: d={d_model} is not a power of 2; the shift-based "
        "mean/var divide (matching layernorm_vec.sv) is only exact for one")
    d_shift = d_model.bit_length() - 1
    S = sum(X)                                     # Q6.25, up to ~40 bits
    mean = S >> d_shift                             # /d (arith shift / floor)
    d = [v - mean for v in X]                       # Q6.25
    SS = sum(di * di for di in d)                   # Q12.50
    var = SS >> d_shift                             # /d -> Q12.50
    A = (var >> (VAR_FRAC - A_FRAC)) + EPS_A        # Q.26
    if A <= 0:
        A = 1
    Yr = rsqrt_int(A)                               # Q.26
    sh = (QX + Y_FRAC + G_FRAC) - OUT_FRAC          # 25+26+20-22 = 49
    out = []
    for i in range(d_model):
        t = d[i] * Yr                               # Q(QX+Y_FRAC)
        t = t * G[i]                                # Q(QX+Y_FRAC+G_FRAC)
        yv = t >> sh if sh >= 0 else t << (-sh)     # arith shift -> Q.OUT_FRAC
        out.append(yv)
    return out, mean, var, A, Yr


def ln_int(x, gamma):
    """Float-in / float-out LayerNorm matching the RTL exactly. Quantizes x->Q6.25,
    gamma->Q4.20, runs the integer datapath, returns y = Yint / 2^OUT_FRAC."""
    x = np.asarray(x, dtype=np.float64)
    X = np.round(x * (1 << QX)).astype(object)
    G = np.round(np.asarray(gamma, dtype=np.float64) * (1 << G_FRAC)).astype(object)
    out, *_ = _ln_int_quantized(list(X), list(G))
    return np.array([o / (1 << OUT_FRAC) for o in out], dtype=np.float64)


# --- token-stream gate: integer LN inside goformer vs exact float reference ----
def _token_stream():
    from model.goformer_full import GoformerFull, load_params

    class IntLNGoformer(GoformerFull):
        def _ln(self, x, gamma):
            if np.asarray(x).ndim == 1:
                return ln_int(x, gamma)
            return np.stack([ln_int(x[t], gamma) for t in range(x.shape[0])])

        def forward(self, idx):
            idx = np.asarray(idx)
            T = len(idx)
            xs = self.p["tok_emb"][idx] + self.p["pos_emb"][np.arange(T)]
            for blk in self.p["blocks"]:
                xs = xs + self._attn(self._ln(xs, blk["ln1"]), blk)
                xs = xs + self._mlp(self._ln(xs, blk["ln2"]), blk)
            xs = self._ln(xs, self.p["ln_f"])
            return self.qlin(xs, self.p["head"])

    p = load_params("fabric/export/goformer.npz")
    ref = GoformerFull(p)
    got = IntLNGoformer(p)
    rng = np.random.default_rng(0)
    prompt = list(rng.integers(0, p["tok_emb"].shape[0], size=8))
    n_gen = 60

    def greedy(engine, idx0, n):
        out = list(idx0)
        for _ in range(n):
            out.append(int(np.argmax(engine.forward(out[-256:])[-1])))
        return out

    ref_stream = greedy(ref, prompt, n_gen)
    got_stream = greedy(got, prompt, n_gen)
    same = ref_stream == got_stream
    n_match = sum(1 for a, b in zip(ref_stream, got_stream) if a == b)
    # worst-step logit cosine (informational, not the gate)
    worst = 1.0
    for t in range(len(ref_stream)):
        a = ref.forward(ref_stream[:t + 1][-256:])[-1]
        b = got.forward(ref_stream[:t + 1][-256:])[-1]
        worst = min(worst, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    return same, n_match, len(ref_stream), worst


# --- module bit-true gate: RTL vs ln_int integer reference --------------------
def _rand_cases(n, seed=0):
    """n test vectors of (X Q6.25, G Q4.20). Includes the model's real range plus edges."""
    rng = np.random.default_rng(seed)
    cases = []
    for c in range(n):
        if c == 0:
            x = np.zeros(D)                                  # all-zero -> var=0 -> A=eps
        elif c == 1:
            x = np.full(D, 19.5)                             # max magnitude, ~zero var
        elif c == 2:
            x = np.linspace(-19.5, 14.8, D)                  # wide spread, large var
        else:
            scale = rng.uniform(0.05, 8.0)
            x = rng.standard_normal(D) * scale + rng.uniform(-1, 1)
            x = np.clip(x, -31.9, 31.9)                      # stay in Q6.25 range
        g = rng.uniform(0.04, 1.12, D)
        X = np.round(x * (1 << QX)).astype(np.int64)
        G = np.round(g * (1 << G_FRAC)).astype(np.int64)
        cases.append((X, G))
    return cases


def run(sim_dir, n_cases=64, seed=0):
    os.makedirs(sim_dir, exist_ok=True)
    cases = _rand_cases(n_cases, seed)

    # reference outputs (Q.OUT_FRAC ints), one row of 256 per case
    gold = []
    for X, G in cases:
        out, _, _, _, _ = _ln_int_quantized(list(X), list(G))
        gold.append([int(o) & 0xFFFFFFFFFFFFFFFF for o in out])   # 64-bit two's comp

    # write stimulus: x.mem (Q6.25, 32-bit hex), gamma.mem (Q4.20, 32-bit hex), seed LUT
    def w(name, vals, width_nib):
        mask = (1 << (4 * width_nib)) - 1
        with open(os.path.join(sim_dir, name), "w") as fh:
            fh.write("\n".join(f"{int(v) & mask:0{width_nib}x}" for v in vals) + "\n")

    xs = np.concatenate([c[0] for c in cases])
    gs = np.concatenate([c[1] for c in cases])
    w("x.mem", xs, 8)              # 32-bit
    w("gamma.mem", gs, 8)          # 32-bit
    w("seed.mem", _SEED, 5)        # 20-bit (Q1.16 fits, store 5 nibbles)

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp, f"-DNCASE={n_cases}",
                         os.path.join(HERE, "tb", "tb_layernorm.sv"),
                         os.path.join(HERE, "rtl", "layernorm.sv")],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout)
        print(cp.stderr)
        return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL")
        print(rp.stdout)
        print(rp.stderr)
        return False

    # y.out: one 16-hex-nibble (64-bit) value per output element, NCASE*256 lines,
    # case rows separated by case boundaries (TB writes them in order, no fill).
    with open(os.path.join(sim_dir, "y.out")) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    def _hx(s):
        if "x" in s or "z" in s:
            return None
        return int(s, 16) & 0xFFFFFFFFFFFFFFFF

    got = [_hx(s) for s in lines]
    flat_gold = [v for row in gold for v in row]
    mism = 0
    n = min(len(got), len(flat_gold))
    for i in range(n):
        if got[i] != flat_gold[i]:
            mism += 1
    mism += abs(len(got) - len(flat_gold))    # length disagreement counts
    total = len(flat_gold)
    ok = (mism == 0)

    same, n_match, n_tot, worst = _token_stream()
    print(f"LN_VERDICT bitexact={ok} mismatches={mism}/{total} "
          f"token_stream_identical={same} stream_match={n_match}/{n_tot} "
          f"worst_step_cosine={worst:.7f} pass={ok and same}")
    return ok and same


def main():
    sim_dir = kevbuild("stage3_ln")
    raise SystemExit(0 if run(sim_dir) else 1)


if __name__ == "__main__":
    main()
