"""Golden vectors for the Big-Kev RTL (HANDOFF task 2): trace FabricRef, export `.mem`.

The analogue of fabric/stage3/seq_ref.py `block0_phase_signals()`. One forward_seq over a
short prompt with a `Trace` attached records every intermediate of the selected layers in
pipeline order; this module writes them as $readmemh files plus MANIFEST.json, and
`verify()` re-reads ONLY those files and recomputes every block with the spec functions.

    python -u -m fabric.big.ref.golden --verify                      # dump the default 8B scope, then verify
    python -u -m fabric.big.ref.golden --verify-only --out DIR       # verify an existing dump (CPU, no model)

File format (every `.mem`): one value per line, a fixed number of hex digits, no comments,
no address directives. A tensor of shape `shape` is flattened in C order (first axis
slowest, last fastest), so a testbench loads it into a FLAT memory and indexes by hand —
never a 2D-array row (iverilog cannot $readmemh one):

    reg [31:0] x_in [0:T*D-1];   initial $readmemh("l0.x_in.mem", x_in);     // (pos, i) at pos*D + i
    $readmemh("l0.q.w.md.mem", rom, BASE, BASE + R*G - 1);   // ranged: stack files in one flat ROM

(The range of a ranged $readmemh is where the file LANDS in the memory; it always reads the
file from its first line, so size the range to the file's MANIFEST `depth`.)

Encodings: `ieee754` = raw fp32 bits (8 digits); `twos-complement` = the value masked to
`width` bits (ceil(width/4) digits, so `reg signed [width-1:0]` reads it back); `unsigned`;
`bits` = a packed word (w.sign: bit j of the 128-bit word is weight 128*group + j, 1 = +).

Axes. `pos` = ordinal into MANIFEST.positions (the traced absolute positions); `cpos` =
ordinal into MANIFEST.cache_positions (the KV cache rows, kept for ALL positions since
attention at a traced position reads the whole history); `t` = cache position, entries
past the query position are masked (score -inf, e = p = +0); `row` = ordinal into the
GEMV's exported output-row window MANIFEST.gemv[name].rows.

Weights are NOT dumped wholesale: per traced GEMV only the row window (sign bits, fp16
scale bits, and the decomposed scale m_d / e_d), plus the norm weight vectors of the traced
layers and the embedding rows of the traced tokens.
"""
from __future__ import annotations

import argparse
import binascii
import dataclasses
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import fabric_ref as fr
from .fabric_ref import (ACC_LSB, EXP_K, GEMV_BITS, K_BITS, KV_BITS, LOG2E, P, ExactBfpGemv, FabricRef, Trace,
                         bfp_dequant, bfp_quant, fp16_decompose)
from .qwen3_ref import Cfg, Q1Matrix

FORMAT = "big-kev-golden/1"
PROMPT = "<|im_start|>user\nWho are you, and why use few word?<|im_end|>\n<|im_start|>assistant\n"   # 18 tokens
F32 = (32, "ieee754")
GEMV_IN = {"q": "attn_norm", "k": "attn_norm", "v": "attn_norm", "o": "ctx",
           "gate": "ffn_norm", "up": "ffn_norm", "down": "act"}          # GEMV -> its fp32 input signal
NORM_W = ("attn_norm", "q_norm", "k_norm", "ffn_norm")


def S(w):
    return (w, "twos-complement")


def _norm(what, x):
    return {"": (F32, f"{what} = w * (x * r), x = {x}"),
            ".ss": (F32, f"{what}: fsum(x*x), lane-strided, x = {x}"),
            ".mean": (F32, f"{what}: ss * 2^-log2(n)"),
            ".r": (F32, f"{what}: rsqrt_fab(mean + eps)")}


# signal leaf (name without the `l<L>.` prefix) -> (dims, (width, encoding), spec line)
SIG = {
    "embed": (("pos", "d"), F32, "embedding row of token[pos]: sign * fp16 scale (Q1 row of token_embd)"),
    "x_in": (("pos", "d"), F32, "layer input residual (layer 0: == embed)"),
    "q_rope": (("pos", "head", "hd"), F32, "rope(q_norm): [x1*c - x2*s, x2*c + x1*s] with tab.rope_cos/sin[pos]"),
    "k_rope": (("pos", "kvhead", "hd"), F32, "rope(k_norm)"),
    "k_q": (("cpos", "kvhead", "hd"), S(K_BITS), "K cache mantissa: BFP10 of k_rope, exponent per 16 channels"),
    "k_e": (("cpos", "kvhead", "kgroup"), S(8), "K cache exponent (K = k_q * 2^k_e)"),
    "v_q": (("cpos", "kvhead", "hd"), S(KV_BITS), "V cache mantissa: BFP8 of v.y, one exponent per head vector"),
    "v_e": (("cpos", "kvhead", "vgroup"), S(8), "V cache exponent (V = v_q * 2^v_e)"),
    "att.s": (("pos", "head", "t"), F32, "score = fsum_d(q_rope * K_t) * fp32(1/sqrt(hd)); K of head h = kvhead h // rep"),
    "att.m": (("pos", "head"), F32, "max_t score"),
    "att.e": (("pos", "head", "t"), F32, "exp_fab(score - m)"),
    "att.z": (("pos", "head"), F32, "Z = fsum_t(e), lane = t % P"),
    "att.rz": (("pos", "head"), F32, "1 / Z (fp32 divide)"),
    "att.p": (("pos", "head", "t"), F32, "p = e * (1/Z)"),
    "ctx": (("pos", "head", "hd"), F32, "ctx_d = fsum_t(p_t * V_t,d), lane = t % P; flattened = the o GEMV input"),
    "x_mid": (("pos", "d"), F32, "x_in + o.y (post-attention residual)"),
    "silu": (("pos", "ffn"), F32, "silu(gate.y) = x * (1 / (1 + exp_fab(-x)))"),
    "act": (("pos", "ffn"), F32, "silu * up.y (the down GEMV input)"),
    "x_out": (("pos", "d"), F32, "x_mid + down.y (layer output)"),
    "x_final": (("pos", "d"), F32, "out_norm input = x_out of the last layer"),
    "argmax": (("pos",), (32, "unsigned"), "lowest index of the maximum logit (lm_head.y)"),
}
for _n, _x, _d in (("attn_norm", "x_in", ("pos", "d")), ("ffn_norm", "x_mid", ("pos", "d")),
                   ("out_norm", "x_final", ("pos", "d")), ("q_norm", "q.y as [head, hd]", ("pos", "head", "hd")),
                   ("k_norm", "k.y as [kvhead, hd]", ("pos", "kvhead", "hd"))):
    for _s, (_f, _doc) in _norm(_n, _x).items():
        SIG[_n + _s] = (_d if _s == "" else _d[:-1], _f, _doc)

# GEMV sub-signals `<gemv>.<suffix>` (gemv = l<L>.{q,k,v,o,gate,up,down} or lm_head)
GEMV_SIG = {
    "xq": (("pos", "in"), S(GEMV_BITS), "BFP10 mantissa of the GEMV input: RNE(x / 2^xe[in // 128]), |q| <= 511"),
    "xe": (("pos", "group"), S(8), "BFP10 exponent per 128-group: smallest e with absmax <= 511 * 2^e (0 if all-zero)"),
    "y": (("pos", "out"), F32, "GEMV output, ALL rows: RNE_fp32(acc) * 2^-ACC_LSB"),
    "acc": (("pos", "row"), S(64), "final accumulator = sum over groups of term (LSB 2^-ACC_LSB)"),
    "inner": (("pos", "row", "group"), S(17), "sum_j sign_j * q_j over the group's 128 weights"),
    "prod": (("pos", "row", "group"), S(28), "inner * m_d"),
    "sh": (("pos", "row", "group"), S(8), "e_d + xe + ACC_LSB"),
    "term": (("pos", "row", "group"), S(64), "prod << sh, or prod >>> -sh (arithmetic = floor) when sh < 0"),
    "accp": (("pos", "row", "group"), S(64), "running accumulator after adding groups 0..group"),
}
W_SIG = {
    "w.sign": (("row", "group"), (128, "bits"), "sign bits: bit j = weight 128*group + j, 1 = +scale, 0 = -scale"),
    "w.scale": (("row", "group"), (16, "unsigned"), "fp16 scale, raw bits"),
    "w.md": (("row", "group"), S(12), "scale = w.md * 2^w.ed exactly (|m_d| <= 2047)"),
    "w.ed": (("row", "group"), S(8), "scale exponent e_d"),
    "w": (("elem",), F32, "RMSNorm weight vector"),
}
TAB_SIG = {
    "tab.exp": (("k",), F32, "exp_fab table: fp32(2^(k/4096)), k = 0..4096"),
    "tab.rsqrt_lo": (("i",), F32, "rsqrt seed, mantissa in [0.5, 1): fp32(1/sqrt(0.5 + (i + 0.5)/1024))"),
    "tab.rsqrt_hi": (("i",), F32, "rsqrt seed, mantissa in [1, 2): fp32(1/sqrt(1 + (i + 0.5)/512))"),
    "tab.rope_cos": (("pos", "hd2"), F32, "fp32(cos(position * inv_freq) * yarn_attn_factor)"),
    "tab.rope_sin": (("pos", "hd2"), F32, "fp32(sin(position * inv_freq) * yarn_attn_factor)"),
}


def sigspec(name):
    """signal name -> (layer | None, dims, (width, encoding), spec line)."""
    m = re.match(r"l(\d+)\.(.*)", name)
    layer, leaf = (int(m.group(1)), m.group(2)) if m else (None, name)
    if leaf in SIG:
        return (layer,) + SIG[leaf]
    if name in TAB_SIG:
        return (None,) + TAB_SIG[name]
    for suf, tab in ((leaf.split(".", 1)[-1], W_SIG), (leaf.rsplit(".", 1)[-1], GEMV_SIG)):
        if suf in tab:
            return (layer,) + tab[suf]
    raise KeyError(name)


# ----------------------------------------------------------------------------- hex I/O
def _digits(width):
    return (width + 3) // 4


def encode(t: torch.Tensor, width, enc) -> np.ndarray:
    """tensor -> big-endian byte matrix [n, ceil(digits/2)] (range-checked, never truncating)."""
    if enc == "bits":                                   # uint8 [..., width/8], byte 0 = bits 7..0
        return np.ascontiguousarray(t.numpy().reshape(-1, width // 8)[:, ::-1])
    if enc == "ieee754":
        assert t.dtype == torch.float32, t.dtype
        u = t.contiguous().view(torch.int32).numpy().astype(np.int64).reshape(-1) & 0xFFFFFFFF
    else:
        v = t.to(torch.int64)
        assert bool((v == t).all()), "non-integer value in an integer signal"
        lo, hi = (-(1 << (width - 1)), (1 << (width - 1)) - 1) if enc == "twos-complement" else (0, (1 << width) - 1)
        assert v.numel() == 0 or (int(v.min()) >= lo and int(v.max()) <= hi), \
            f"value range [{int(v.min())}, {int(v.max())}] does not fit {width}-bit {enc}"
        u = v.numpy().reshape(-1)
    u = u.astype(np.int64).view(np.uint64) & np.uint64((1 << width) - 1)
    return u.astype(">u8").view(np.uint8).reshape(-1, 8)[:, 8 - (_digits(width) + 1) // 2:]


def decode(be: np.ndarray, width, enc) -> torch.Tensor:
    if enc == "bits":
        return torch.from_numpy(np.ascontiguousarray(be[:, ::-1]))
    full = np.zeros((be.shape[0], 8), np.uint8)
    full[:, 8 - be.shape[1]:] = be
    u = full.view(">u8").reshape(-1).astype(np.uint64)
    if enc == "ieee754":
        return torch.from_numpy(u.astype(np.uint32).view(np.float32).copy())
    v = u.view(np.int64)
    if enc == "twos-complement" and width < 64:
        v = (v ^ (1 << (width - 1))) - (1 << (width - 1))
    return torch.from_numpy(v.copy())


def write_mem(path: Path, be: np.ndarray, digits):
    hx = np.frombuffer(binascii.hexlify(be.tobytes()), dtype="S1").reshape(be.shape[0], -1)
    out = np.full((be.shape[0], digits + 1), b"\n", dtype="S1")
    out[:, :digits] = hx[:, hx.shape[1] - digits:]
    path.write_bytes(out.tobytes())


def read_mem(path: Path, digits) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype="S1")
    assert raw.size % (digits + 1) == 0, f"{path.name}: not {digits} hex digits per line"
    a = raw.reshape(-1, digits + 1)
    assert bool((a[:, digits] == b"\n").all()), f"{path.name}: ragged lines"
    hx = a[:, :digits]
    if digits % 2:
        hx = np.concatenate([np.full((a.shape[0], 1), b"0", dtype="S1"), hx], 1)
    return np.frombuffer(binascii.unhexlify(hx.tobytes()), np.uint8).reshape(a.shape[0], -1)


def f32_bits(x) -> str:
    return f"{int(np.float32(x).view(np.uint32)):08x}"


# ----------------------------------------------------------------------------- dump
def dump(fab: FabricRef, toks, out, layers, positions=None, rows=(0, 64), partials=None, meta=None,
         selfcheck=False, final=True):
    """Trace forward_seq(toks) and write the vector set + MANIFEST.json into `out`. Returns the manifest.
    rows / partials: see Trace (a rows window covering a whole matrix exports all of its weights, which
    makes the traced layer recomputable end to end); final=False drops out_norm / logits / argmax."""
    c, W = fab.c, fab.W
    assert fab.K_FMT is not None and fab.V_FMT is not None, "golden vectors need the quantised KV cache"
    layers = sorted(set(layers))
    partials = dict(partials or {})
    for n in partials:
        m = re.fullmatch(r"l(\d+)\.(\w+)", n)
        assert n == "lm_head" or (m and int(m.group(1)) in layers and m.group(2) in GEMV_IN), \
            f"--partials {n}: not a GEMV of a traced layer"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in list(out.glob("*.mem")) + list(out.glob("MANIFEST.json")):
        f.unlink()
    tr = Trace(layers=layers, positions=positions, rows=rows, partials=partials, final=final)
    t0 = time.time()
    lg = fab.forward_seq(toks, trace=tr)
    print(f"traced forward: {len(toks)} tok, layers {layers}, {time.time() - t0:.1f}s")
    if selfcheck:
        ok = torch.equal(lg, fab.forward_seq(toks))
        print(f"selfcheck: traced logits == untraced logits: {ok}")
        assert ok
    pos = tr.pos
    assert pos, "no traced positions"
    assert tr.cache_pos == list(range(len(toks)))
    files, total = [], 0

    def put(name, t, index, dims=None):
        nonlocal total
        layer, sdims, (width, enc), doc = sigspec(name)
        dims = dims or sdims
        shape = list(t.shape[:-1]) if enc == "bits" else list(t.shape)
        assert len(shape) == len(dims), (name, shape, dims)
        be = encode(t.cpu(), width, enc)
        write_mem(out / (name + ".mem"), be, _digits(width))
        total += be.shape[0] * (_digits(width) + 1)
        files.append(dict(file=name + ".mem", signal=name, layer=layer, shape=shape, dims=list(dims),
                          width=width, hex_digits=_digits(width), signed=enc == "twos-complement",
                          encoding=enc, depth=be.shape[0], index=index,
                          order="C order over dims (" + ", ".join(dims) + "); last axis fastest", spec=doc))

    # 1. the trace, in pipeline order
    for name in tr.names():
        put(name, tr[name], "cpos" if sigspec(name)[1][0] == "cpos" else "pos")
    # 2. the row window of every traced GEMV + the norm vectors + the traced embedding rows
    gem = {}
    for name in [n[:-3] for n in tr.names() if n.endswith(".xq")]:
        M = W[name]
        r0, r1, part = tr.window(name, M.out)
        inp = "out_norm" if name == "lm_head" else name.rsplit(".", 1)[0] + "." + GEMV_IN[name.rsplit(".", 1)[1]]
        gem[name] = dict(rows=[r0, max(r0, r1)], n_in=M.inn, n_out=M.out, input=inp, partials=part)
        if r0 < r1:
            md, ed = fp16_decompose(M.scales[r0:r1])
            put(name + ".w.sign", M.signs[r0:r1].reshape(r1 - r0, M.inn // 128, 16), "static")
            put(name + ".w.scale", M.scales[r0:r1].view(torch.int16).to(torch.int64) & 0xFFFF, "static")
            put(name + ".w.md", md, "static")
            put(name + ".w.ed", ed, "static")
    for n in [f"l{L}.{k}" for L in layers for k in NORM_W] + ["out_norm"] * final:
        put(n + ".w", W[n].to(torch.float32), "static")
    tk = torch.tensor([int(toks[p]) for p in pos], device=W["embed"].signs.device)
    put("embed.w.sign", W["embed"].signs[tk].reshape(len(pos), c.d // 128, 16), "pos", ("pos", "group"))
    put("embed.w.scale", W["embed"].scales[tk].view(torch.int16).to(torch.int64) & 0xFFFF, "pos", ("pos", "group"))
    # 3. the three small tables the RTL needs
    pt = torch.tensor(pos, device=fab.COS.device)
    for n, t in (("tab.exp", fab.EXP), ("tab.rsqrt_lo", fab.RS_LO), ("tab.rsqrt_hi", fab.RS_HI),
                 ("tab.rope_cos", fab.COS[pt]), ("tab.rope_sin", fab.SIN[pt])):
        put(n, t, "pos" if "rope" in n else "static")

    def git(*a):
        try:
            return subprocess.run(("git",) + a, capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
        except OSError:
            return ""
    man = dict(
        format=FORMAT, **(meta or {}), cfg=dataclasses.asdict(c), tokens=[int(t) for t in toks],
        positions=pos, cache_positions=tr.cache_pos, layers=layers,
        spec=dict(P=P, ACC_LSB=ACC_LSB, GEMV_BITS=GEMV_BITS, GEMV_GROUP=128, KV_BITS=KV_BITS, K_BITS=K_BITS, EXP_K=EXP_K,
                  K_FMT=list(fab.K_FMT), V_FMT=list(fab.V_FMT), LOG2E=f32_bits(LOG2E),
                  eps=f32_bits(fab.eps32.item()), inv_sqrt_hd=f32_bits(fab.inv_sqrt_hd.item()),
                  note="K_FMT / V_FMT = [mantissa bits, channels per exponent]; LOG2E / eps / inv_sqrt_hd = fp32 bits"),
        git=dict(head=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no"))),
        conventions=dict(
            file="one value per line, fixed hex digits, C-order flatten of `shape` (last axis fastest); "
                 "load into a flat memory, index by hand, or use ranged $readmemh(file, mem, start, end)",
            ieee754="raw fp32 bits", twos_complement="value masked to `width` bits", bits="packed word, see spec",
            pos="axis ordinal i <-> absolute position positions[i]",
            cpos="axis ordinal i <-> cache_positions[i]; the KV cache rows are kept for every position",
            t="cache position; entries past the query position are masked (att.s = -inf, att.e = att.p = +0)",
            row="ordinal into gemv[name].rows = [r0, r1): output row r0 + i",
            heads="q/ctx use n_head heads, k/v use n_kv; head h attends K/V of kvhead h // (n_head // n_kv)",
            gemv="gemv[name].input is the fp32 vector the GEMV quantises (flattened per position); q/k/v share "
                 "one input (and so identical xq/xe), as do gate/up",
            weights="only the row window of each traced GEMV (w.sign / w.scale / w.md / w.ed), the RMSNorm "
                    "weight vectors of the traced layers and the embedding rows of the traced tokens"),
        gemv=gem, files=files, total_bytes=total)
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {len(files)} files, {total / 1e6:.1f} MB -> {out}")
    return man


# ----------------------------------------------------------------------------- verify
class Dump:
    """A vector set read back from disk: D["l0.q.xq"] -> tensor of the manifest shape."""

    def __init__(self, d):
        self.dir = Path(d)
        self.m = json.loads((self.dir / "MANIFEST.json").read_text())
        self.f = {e["signal"]: e for e in self.m["files"]}
        self._c = {}

    def __contains__(self, name):
        return name in self.f

    def __getitem__(self, name):
        if name not in self._c:
            e = self.f[name]
            be = read_mem(self.dir / e["file"], e["hex_digits"])
            assert be.shape[0] == e["depth"], f"{e['file']}: {be.shape[0]} lines, manifest says {e['depth']}"
            t = decode(be, e["width"], e["encoding"])
            self._c[name] = t.reshape(e["shape"] + ([e["width"] // 8] if e["encoding"] == "bits" else []))
        return self._c[name]

    def q1(self, name) -> Q1Matrix:
        """The exported rows of a 1-bit matrix, from <name>.w.sign / .w.scale."""
        sg = self[name + ".w.sign"]
        sc = self[name + ".w.scale"].numpy().astype(np.uint16).view(np.float16)
        return Q1Matrix(sg.reshape(sg.shape[0], -1), torch.from_numpy(sc.copy()))


def beq(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-exact equality (fp32 compared as raw bits, so -0 / inf / nan count)."""
    if a.shape != b.shape or a.is_floating_point() != b.is_floating_point():
        return False
    if a.is_floating_point():
        a, b = a.to(torch.float32).contiguous().view(torch.int32), b.to(torch.float32).contiguous().view(torch.int32)
    return torch.equal(a, b)


def acc_to_f32_bits(v: int) -> int:
    """fp32 bits of RNE_fp32(v) * 2^-ACC_LSB in pure integer arithmetic (the RTL normaliser)."""
    if v == 0:
        return 0
    s, a = int(v < 0), abs(v)
    n = a.bit_length()
    if n > 24:
        k = n - 24
        m, rem, half = a >> k, a & ((1 << k) - 1), 1 << (k - 1)
        m += rem > half or (rem == half and m & 1)
        if m == 1 << 24:
            m, n = m >> 1, n + 1
    else:
        m = a << (24 - n)
    e = n - 1 - ACC_LSB + 127
    assert 0 < e < 255
    return (s << 31) | (e << 23) | (m & 0x7FFFFF)


def verify(out) -> bool:
    """Recompute every block from the .mem files + MANIFEST alone; True if all bit-exact."""
    D = Dump(out)
    m, sp = D.m, D.m["spec"]
    c = Cfg(**m["cfg"])
    pos, cpos, T = m["positions"], m["cache_positions"], len(m["positions"])
    res = []

    def ck(name, **sub):
        bad = [k for k, v in sub.items() if not v]
        res.append(not bad)
        print(f"  [{'FAIL' if bad else ' ok '}] {name:26s} " + (f"MISMATCH: {' '.join(bad)}" if bad else " ".join(sub)))

    def trace():
        t = Trace(layers=None, rows=None)
        t.begin(range(T))
        return t

    print(f"verify {out}: {m.get('model', '?')}, {len(D.f)} files, positions {pos}, layers {m['layers']}")
    ck("spec constants", format=m["format"] == FORMAT,
       consts=all(sp[k] == getattr(fr, k) for k in ("P", "ACC_LSB", "GEMV_BITS", "KV_BITS", "K_BITS", "EXP_K")),
       LOG2E=sp["LOG2E"] == f32_bits(LOG2E), eps=sp["eps"] == f32_bits(c.eps),
       inv_sqrt_hd=sp["inv_sqrt_hd"] == f32_bits(1 / np.sqrt(c.head_dim)))
    # a weightless FabricRef: tables rebuilt from the formulas, compared, then REPLACED by the files
    fab = FabricRef(c, {}, device="cpu", max_pos=max(pos) + 1)
    fab.K_FMT, fab.V_FMT = tuple(sp["K_FMT"]), tuple(sp["V_FMT"])
    pt = torch.tensor(pos)
    ck("tables", exp=beq(D["tab.exp"], fab.EXP), rsqrt_lo=beq(D["tab.rsqrt_lo"], fab.RS_LO),
       rsqrt_hi=beq(D["tab.rsqrt_hi"], fab.RS_HI), rope_cos=beq(D["tab.rope_cos"], fab.COS[pt]),
       rope_sin=beq(D["tab.rope_sin"], fab.SIN[pt]))
    fab.EXP, fab.RS_LO, fab.RS_HI = D["tab.exp"], D["tab.rsqrt_lo"], D["tab.rsqrt_hi"]
    fab.COS, fab.SIN = torch.full_like(fab.COS, float("nan")), torch.full_like(fab.SIN, float("nan"))
    fab.COS[pt], fab.SIN[pt] = D["tab.rope_cos"], D["tab.rope_sin"]
    ck("embed", rows=beq(D.q1("embed").dequant().to(torch.float32), D["embed"]))

    def norm(name, x, shape=None):
        t = trace()
        fab.rmsnorm(x if shape is None else x.view(T, *shape), D[name + ".w"], t, "n")
        ck(name, **{k or "y": beq(t["n" + (k and "." + k)], D[name + (k and "." + k)]) for k in ("ss", "mean", "r", "")})

    def gemv(name):
        g = m["gemv"][name]
        r0, r1 = g["rows"]
        x = D[g["input"]].reshape(T, -1)
        xq, xe = D[name + ".xq"], D[name + ".xe"]
        q, e = bfp_quant(x, GEMV_BITS, 128)
        sub = dict(xq=beq(q.to(torch.int64), xq), xe=beq(e, xe))
        if r0 < r1:
            Wm, R, G = D.q1(name), r1 - r0, g["n_in"] // 128
            md, ed = fp16_decompose(Wm.scales)
            sub.update(md=beq(md, D[name + ".w.md"]), ed=beq(ed, D[name + ".w.ed"]))
            # (1) the spec function on the fp32 input and the exported weight rows
            t = Trace(rows=(0, R), partials={name: (0, R)} if g["partials"] else None)
            t.begin(range(T))
            y = ExactBfpGemv().mm(Wm, x, name, t)
            sub.update(y=beq(y, D[name + ".y"][:, r0:r1]), acc=beq(t[name + ".acc"], D[name + ".acc"]))
            # (2) the integer datapath from the mantissa files, as the module doc states it
            pm = (Wm.sign_pm(dtype=torch.float64)).view(R, G, 128)
            inner = torch.einsum("tgj,rgj->trg", xq.double().view(T, G, 128), pm).to(torch.int64)
            prod = inner * D[name + ".w.md"][None]
            sh = D[name + ".w.ed"][None] + xe[:, None, :] + ACC_LSB
            term = torch.where(sh >= 0, prod << sh.clamp(min=0), prod >> (-sh).clamp(min=0))
            acc = term.sum(-1)
            yb = torch.tensor([acc_to_f32_bits(v) for v in acc.reshape(-1).tolist()], dtype=torch.int64)
            sub.update(int_acc=beq(acc, D[name + ".acc"]),
                       int_y=torch.equal(yb, D[name + ".y"][:, r0:r1].contiguous().view(torch.int32).reshape(-1).to(torch.int64) & 0xFFFFFFFF))
            if g["partials"]:
                for k, v in (("inner", inner), ("prod", prod), ("sh", sh), ("term", term), ("accp", term.cumsum(-1))):
                    sub["int_" + k] = beq(v, D[f"{name}.{k}"])
                    sub[k] = beq(t[f"{name}.{k}"], D[f"{name}.{k}"])
        ck(f"{name} gemv [{r0}:{r1}]", **sub)

    rep = c.n_head // c.n_kv
    for L in m["layers"]:
        p = f"l{L}."
        xin, xmid = D[p + "x_in"], D[p + "x_mid"]
        if L == 0:
            ck(p + "x_in", embed=beq(xin, D["embed"]))
        elif f"l{L - 1}.x_out" in D:
            ck(p + "x_in", prev_x_out=beq(xin, D[f"l{L - 1}.x_out"]))
        norm(p + "attn_norm", xin)
        for g in ("q", "k", "v"):
            gemv(p + g)
        norm(p + "q_norm", D[p + "q.y"], (c.n_head, c.head_dim))
        norm(p + "k_norm", D[p + "k.y"], (c.n_kv, c.head_dim))
        ck(p + "rope", q=beq(fab.rope_tab(D[p + "q_norm"].transpose(0, 1), pt).transpose(0, 1), D[p + "q_rope"]),
           k=beq(fab.rope_tab(D[p + "k_norm"].transpose(0, 1), pt).transpose(0, 1), D[p + "k_rope"]))
        ci = torch.tensor([cpos.index(x) for x in pos])
        kq, ke = bfp_quant(D[p + "k_rope"], *fab.K_FMT)
        vq, ve = bfp_quant(D[p + "v.y"].view(T, c.n_kv, c.head_dim), *fab.V_FMT)
        ck(p + "kv quant", k_q=beq(kq.to(torch.int64), D[p + "k_q"][ci]), k_e=beq(ke, D[p + "k_e"][ci]),
           v_q=beq(vq.to(torch.int64), D[p + "v_q"][ci]), v_e=beq(ve, D[p + "v_e"][ci]))
        # attention per traced position against the cache rows 0..position (the step() view)
        K = bfp_dequant(D[p + "k_q"].double(), D[p + "k_e"], fab.K_FMT[1]).transpose(0, 1).repeat_interleave(rep, 0)
        V = bfp_dequant(D[p + "v_q"].double(), D[p + "v_e"], fab.V_FMT[1]).transpose(0, 1).repeat_interleave(rep, 0)
        sub = {k: True for k in ("s", "m", "e", "z", "rz", "p", "ctx", "masked")}
        for i, ap in enumerate(pos):
            assert cpos[:ap + 1] == list(range(ap + 1)), "attention needs the cache rows 0..position"
            t = Trace(rows=None)
            t.begin([ap])
            fab.attend(D[p + "q_rope"][i][:, None, :], K[:, :ap + 1], V[:, :ap + 1],
                       torch.ones(1, ap + 1, dtype=torch.bool), t, "")
            for k in ("s", "e", "p"):
                f = D[p + "att." + k][i]
                sub[k] &= beq(t["att." + k][0], f[:, :ap + 1])
                sub["masked"] &= bool((f[:, ap + 1:] == (float("-inf") if k == "s" else 0.0)).all())
            for k in ("m", "z", "rz"):
                sub[k] &= beq(t["att." + k][0], D[p + "att." + k][i])
            sub["ctx"] &= beq(t["ctx"][0], D[p + "ctx"][i])
        ck(p + "attention", **sub)
        gemv(p + "o")
        ck(p + "x_mid", add=beq(xin + D[p + "o.y"], xmid))
        norm(p + "ffn_norm", xmid)
        gemv(p + "gate")
        gemv(p + "up")
        ck(p + "silu*up", silu=beq(fab.silu(D[p + "gate.y"]), D[p + "silu"]),
           act=beq(D[p + "silu"] * D[p + "up.y"], D[p + "act"]))
        gemv(p + "down")
        ck(p + "x_out", add=beq(xmid + D[p + "down.y"], D[p + "x_out"]))
    if "x_final" in D:
        last = f"l{c.n_layer - 1}.x_out"
        if last in D:
            ck("x_final", last_x_out=beq(D["x_final"], D[last]))
        norm("out_norm", D["x_final"])
        gemv("lm_head")
        lg = D["lm_head.y"]
        am = torch.where(lg == lg.amax(-1, keepdim=True), torch.arange(lg.shape[-1]), lg.shape[-1]).amin(-1)
        ck("argmax", lowest_max=beq(am, D["argmax"]))
    ok = all(res)
    print(f"GOLDEN_VERDICT {'PASS' if ok else 'FAIL'} ({sum(res)}/{len(res)} checks)")
    return ok


# ----------------------------------------------------------------------------- CLI
def _ints(s, n):
    """'all' | '0,last,-2,4:8' -> sorted ints in [0, n) (None for all)."""
    if s == "all":
        return None
    o = set()
    for part in s.split(","):
        if ":" in part:
            a, b = part.split(":")
            o.update(range(int(a or 0), int(b) if b else n))
        else:
            o.add(n - 1 if part == "last" else int(part) % n)
    return sorted(x for x in o if 0 <= x < n)


def _rng(s):
    a, b = s.split(":")
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gguf", default="data/big/Bonsai-8B-Q1_0.gguf")
    ap.add_argument("--tok", default="data/big/Bonsai-8B-meta")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--layers", default="0,last", help="'all' or e.g. 0,last or 0:4,35")
    ap.add_argument("--positions", default="all", help="'all' or e.g. 0,17 or 12:18")
    ap.add_argument("--partials", default="l0.q:0:64",
                    help="NAME:R0:R1[,..] GEMVs that also get per-group partials on rows [R0, R1); 'none'")
    ap.add_argument("--rows", default="0:64",
                    help="R0:R1 output-row window (accumulator + weights) of every other traced GEMV; 'none'")
    ap.add_argument("--no-final", action="store_true", help="skip out_norm / lm_head (logits) / argmax")
    ap.add_argument("--out", default=None, help="default /tmp/kevbuild/big_vectors/<gguf stem>/")
    ap.add_argument("--verify", action="store_true", help="round-trip the dump from its files after writing it")
    ap.add_argument("--verify-only", action="store_true", help="only verify the existing dump in --out")
    ap.add_argument("--selfcheck", action="store_true", help="also assert traced logits == untraced logits")
    a = ap.parse_args()
    out = Path(a.out or Path("/tmp/kevbuild/big_vectors") / Path(a.gguf).stem)
    if not a.verify_only:
        from transformers import AutoTokenizer

        from .qwen3_ref import from_gguf
        toks = AutoTokenizer.from_pretrained(a.tok)(a.prompt).input_ids
        cfg, W, _ = from_gguf(a.gguf, device=a.device)
        fab = FabricRef(cfg, W, device=a.device)
        parts = {} if a.partials == "none" else \
            {s.split(":", 1)[0]: _rng(s.split(":", 1)[1]) for s in a.partials.split(",")}
        dump(fab, toks, out, _ints(a.layers, cfg.n_layer) or range(cfg.n_layer), _ints(a.positions, len(toks)),
             None if a.rows == "none" else _rng(a.rows), parts, selfcheck=a.selfcheck, final=not a.no_final,
             meta=dict(model=Path(a.gguf).stem, gguf=str(a.gguf), prompt=a.prompt, device=a.device))
        st = fab.gemv.stats
        print(f"GEMV shifts [{st.min_shift}..{st.max_shift}] peak acc bits {st.max_acc_bits}/64")
    if a.verify or a.verify_only:
        raise SystemExit(0 if verify(out) else 1)


if __name__ == "__main__":
    main()
