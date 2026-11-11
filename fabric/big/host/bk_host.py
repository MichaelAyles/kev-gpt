"""bk_host: ctypes wrapper of the Big-Kev host vector path (`bk_host.c`), numpy only.

The C library is the part of a Bonsai token that is not a GEMV (RMSNorm, RoPE, KV-cache
quantise, attention, softmax, SiLU, residuals, BFP10 quantise, argmax), bit-identical to
`fabric/big/ref/fabric_ref.py`. This module builds it on demand, builds / saves / loads the
tables it needs, loads the host-side weights from a GGUF, and drives it one token at a time
with the GEMV given as a Python callable (or a C function pointer).

    from fabric.big.host.bk_host import BkHost, argmax
    host = BkHost.from_gguf("Bonsai-8B-Q1_0.gguf", gemv=my_gemv, tables="tables_8b.npz")
    for t in prompt[:-1]:
        host.step(t, want_logits=False)              # prefill: no LM head
    lg = host.step(prompt[-1])                       # np.float32 [vocab]
    nxt = argmax(lg)                                 # lowest index holding the maximum

    def my_gemv(mat_id, xq, xe, y, same_input):      # fill y in place
        # xq int16 [groups*128] BFP10 mantissas, xe int8 [groups], y float32 [rows]
        # same_input: xq/xe are exactly the previous call's input (q/k/v, gate/up)
        ...

mat_id = 7*layer + (0 q, 1 k, 2 v, 3 o, 4 gate, 5 up, 6 down); 7*n_layer = lm_head
(`mat_name` / `mat_id`).

TABLES. The engine never computes exp / rsqrt / cos / sin: they are tables built by the
spec's numpy code (`build_tables`, a copy of FabricRef.__init__ / exp_table / rsqrt_tables).
Two things in them are NOT reproducible from first principles on another machine:
  * numpy float64 cos / sin / pow are not correctly rounded, and their last bit can differ
    between numpy builds and CPUs;
  * the RoPE frequencies come from torch's fp32 `base ** x`, which is not correctly rounded
    either (x86 AVX2 torch 2.4: 1 of the 64 frequencies of head_dim 128 / theta 1e6 is one
    ulp off the correctly rounded value). `rope_params_np` reproduces the spec's value with
    a correctly rounded pow plus a pinned exception table (`_POW_PINS`).
So: build the tables ONCE where the spec runs (`save_tables`), ship the .npz, and
`load_tables` it everywhere else. `tables_digest` names a table set.

Imports without torch (the board's system python has numpy only); `build_tables_torch` is
the only function that needs it.

    python -m fabric.big.host.bk_host --selftest
    python -m fabric.big.host.bk_host --tables t.npz --gguf data/big/Bonsai-8B-Q1_0.gguf
    python -m fabric.big.host.bk_host --bench --shape 8b --ctx 20,64,256     # host cost, no PL
    python -m fabric.big.host.bk_host --replay host_replay_tiny.npz           # numpy-only gate
    python -m fabric.big.host.bk_host --kern-check --threads-check host_replay_tiny.npz
        # numpy-only: fast kernels == reference kernels on adversarial inputs
        # (HOST_KERN_VERDICT); the replay with 1/3/4/2 threads x both kernel sets, one digest
        # over every tap and the KV cache (HOST_THREADS_VERDICT)
    python -m fabric.big.host.bk_host --bench --shape 8b --ctx 1,20,64,256 --threads 1,4
        # [--kernels ref] [--spin-us N] [--gap-us N: the stand-in GEMV waits N us, as the fabric would]
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from ctypes import POINTER, c_float, c_int, c_int8, c_int16, c_int32, c_int64, c_size_t, c_uint64, c_void_p
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC, HDR = HERE / "bk_host.c", HERE / "bk_host.h"

# the mandatory part: -ffp-contract=off (aarch64 gcc fuses a*b+c by default), never fast-math
BASE_FLAGS = ["-O2", "-ffp-contract=off", "-fno-fast-math", "-pthread", "-shared", "-fPIC"]
# default build: the optimisation level cannot change an fp32 result (the gate checks it)
DEFAULT_FLAGS = ["-O3", "-ffp-contract=off", "-fno-fast-math", "-pthread", "-shared", "-fPIC"]
API_VERSION = 2
KERN_AUTO, KERN_FAST, KERN_REF = 0, 1, 2
KERN_NAMES = {KERN_FAST: "fast", KERN_REF: "ref"}
SIMD_NAMES = {0: "none", 1: "neon", 2: "sse2"}

P, EXP_K, QK = 16, 4096, 128
KINDS = ("q", "k", "v", "o", "gate", "up", "down")
T_NAMES = ("gemv", "embed", "rmsnorm", "quant", "rope", "kvq", "att_score", "att_softmax", "att_ctx",
           "silu", "resid", "taps", "other")
ERRS = {0: "OK", -1: "BK_ERR_ARG", -2: "BK_ERR_CTX_FULL", -3: "BK_ERR_NOMEM", -4: "BK_ERR_GEMV",
        -5: "BK_ERR_EXP_RANGE", -6: "BK_ERR_NONFINITE", -7: "BK_ERR_FPENV"}
NORMS = ("attn_norm", "q_norm", "k_norm", "ffn_norm")
GG = {"attn_norm": "attn_norm", "q_norm": "attn_q_norm", "k_norm": "attn_k_norm", "ffn_norm": "ffn_norm"}


# ----------------------------------------------------------------------------- build
def build_cmd(out="libbk_host.so", flags=None, defines=(), cc=None):
    cc = cc or os.environ.get("CC") or "gcc"
    if flags is None:
        env = os.environ.get("BK_HOST_CFLAGS")
        flags = shlex.split(env) if env else DEFAULT_FLAGS
    return [cc, *flags, *[f"-D{d}" for d in defines], "-o", str(out), str(SRC)]


def build_lib(flags=None, defines=(), cc=None, out_dir=None, force=False, libs=()):
    """Compile bk_host.c -> a cached .so (named by a hash of source, header, compiler, flags).
    Cache dir: `out_dir`, else $BK_HOST_BUILD_DIR, else ~/.cache/bk_host, else the temp dir."""
    cmd = build_cmd("@OUT@", flags, defines, cc) + list(libs)
    key = hashlib.sha1(SRC.read_bytes() + HDR.read_bytes() + " ".join(cmd).encode()
                       + platform.machine().encode()).hexdigest()[:12]
    cands = [out_dir, os.environ.get("BK_HOST_BUILD_DIR"), Path.home() / ".cache" / "bk_host",
             Path(tempfile.gettempdir()) / f"bk_host-{os.getuid()}"]
    last = None
    for d in cands:
        if not d:
            continue
        try:
            d = Path(d)
            d.mkdir(parents=True, exist_ok=True)
            so = d / f"libbk_host-{key}.so"
            if so.exists() and not force:
                return so
            tmp = d / f".{so.name}.{os.getpid()}.tmp"
            run = [str(tmp) if c == "@OUT@" else c for c in cmd]
            r = subprocess.run(run, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"bk_host build failed: {' '.join(run)}\n{r.stderr}")
            os.replace(tmp, so)
            return so
        except OSError as ex:           # read-only dir: try the next one
            last = ex
    raise RuntimeError(f"bk_host: no writable build dir ({last})")


class BkConfig(ctypes.Structure):
    _fields_ = [(n, c_int32) for n in ("n_layer", "d", "ffn", "n_head", "n_kv", "head_dim", "vocab",
                                       "max_ctx", "max_pos", "init_ctx")] + \
               [("eps", c_float), ("inv_sqrt_hd", c_float)] + \
               [(n, c_int32) for n in ("n_threads", "spin_us", "kernels", "_pad")]


class BkWeights(ctypes.Structure):
    _fields_ = [(n, c_void_p) for n in ("attn_norm", "q_norm", "k_norm", "ffn_norm", "out_norm",
                                        "emb_signs", "emb_scales")] + \
               [("emb_sign_stride", c_int64), ("emb_scale_stride", c_int64)] + \
               [(n, c_void_p) for n in ("exp_tab", "rs_lo", "rs_hi", "cos_tab", "sin_tab")]


TAP_PTRS = ("xq", "xe", "y", "x_in", "x_mid", "x_out", "q_rope", "k_rope", "att_s", "att_p", "ctx",
            "silu", "act", "h_out")


class BkTaps(ctypes.Structure):
    _fields_ = [(n, c_void_p) for n in TAP_PTRS] + [("att_cap", c_int32), ("_pad", c_int32)]


class BkStats(ctypes.Structure):
    _fields_ = [("ns", c_uint64 * len(T_NAMES)), ("steps", c_uint64), ("gemv_calls", c_uint64),
                ("att_pos", c_uint64), ("cpu_ns", c_uint64 * len(T_NAMES)), ("regions", c_uint64),
                ("threads", c_int32), ("kernels", c_int32), ("worker_units", c_uint64)]


GEMV_FN = ctypes.CFUNCTYPE(c_int, c_void_p, c_int, c_void_p, c_void_p, c_int, c_void_p, c_int)
_LIBS = {}


def load_lib(path=None, **build_kw):
    """ctypes handle of the library with every prototype set; checks the struct ABI and the
    floating-point self-test (skipped for the planted-bug builds, which say so)."""
    path = Path(path) if path else build_lib(**build_kw)
    if str(path) in _LIBS:
        return _LIBS[str(path)]
    lib = ctypes.CDLL(str(path))
    vp = c_void_p
    sig = {
        "bk_init": (vp, [POINTER(BkConfig), POINTER(BkWeights), POINTER(c_int)]),
        "bk_free": (None, [vp]), "bk_reset": (None, [vp]), "bk_pos": (c_int, [vp]),
        "bk_truncate": (c_int, [vp, c_int]), "bk_capacity": (c_int, [vp]),
        "bk_step": (c_int, [vp, c_int32, c_int, vp, vp, vp, vp]),
        "bk_last_gemv_rc": (c_int, [vp]), "bk_argmax": (c_int, [vp, c_int]),
        "bk_n_mats": (c_int, [vp]),
        "bk_mat_info": (c_int, [vp, c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int64),
                                POINTER(c_int64), POINTER(c_int64)]),
        "bk_tap_sizes": (None, [vp, POINTER(c_int64), POINTER(c_int64), POINTER(c_int64)]),
        "bk_kv_rows": (c_int, [vp, c_int, c_int, POINTER(vp), POINTER(vp), POINTER(vp), POINTER(vp)]),
        "bk_get_stats": (None, [vp, POINTER(BkStats)]), "bk_reset_stats": (None, [vp]),
        "bk_selftest": (c_int, []), "bk_api_version": (c_int, []), "bk_sizeof": (c_size_t, [c_int]),
        "bk_fsum": (c_float, [vp, c_int]), "bk_rsqrt_fab": (c_float, [vp, vp, c_float]),
        "bk_exp_fab": (c_float, [vp, c_float]), "bk_exp_fab_n": (None, [vp, vp, vp, c_int]),
        "bk_silu": (None, [vp, vp, vp, c_int]),
        "bk_rmsnorm": (None, [vp, vp, c_float, vp, vp, c_int, vp]),
        "bk_rope": (None, [vp, vp, vp, c_int, vp]),
        "bk_bfp_quant16": (c_int, [vp, c_int, c_int, c_int, vp, vp]),
        "bk_bfp_quant8": (c_int, [vp, c_int, c_int, c_int, vp, vp]),
        "bk_bfp_dequant16": (None, [vp, vp, c_int, c_int, vp]),
        "bk_bfp_dequant8": (None, [vp, vp, c_int, c_int, vp]),
        "bk_embed_row": (None, [vp, c_int64, vp, c_int64, c_int, c_int64, vp]),
        "bk_bench_gemv": (c_int, [vp, c_int, vp, vp, c_int, vp, c_int]),
        "bk_ref_fsum": (c_float, [vp, c_int]), "bk_ref_exp_fab_n": (None, [vp, vp, vp, c_int]),
        "bk_ref_silu": (None, [vp, vp, vp, c_int]),
        "bk_ref_rmsnorm": (None, [vp, vp, c_float, vp, vp, c_int, vp]),
        "bk_ref_rope": (None, [vp, vp, vp, c_int, vp]),
        "bk_ref_bfp_quant16": (c_int, [vp, c_int, c_int, c_int, vp, vp]),
        "bk_ref_bfp_quant8": (c_int, [vp, c_int, c_int, c_int, vp, vp]),
        "bk_silu_mul": (None, [c_int, vp, vp, vp, vp, vp, c_int]),
        "bk_attend_kv": (c_int, [c_int, vp, c_int, c_int, c_int, c_float, vp, vp, vp, vp, vp, vp, vp, vp]),
        "bk_simd": (c_int, []), "bk_threads": (c_int, [vp]), "bk_kernels": (c_int, [vp]),
    }
    for name, (res, args) in sig.items():
        f = getattr(lib, name)
        f.restype, f.argtypes = res, args
    want = [ctypes.sizeof(s) for s in (BkConfig, BkWeights, BkTaps, BkStats)]
    got = [lib.bk_sizeof(i) for i in range(4)]
    if want != got or lib.bk_api_version() != API_VERSION:
        raise RuntimeError(f"bk_host ABI mismatch: struct sizes {got} vs the binding's {want}")
    lib._path = str(path)
    _LIBS[str(path)] = lib
    return lib


def _ptr(a):
    return a.ctypes.data_as(c_void_p)


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


# ----------------------------------------------------------------------------- config
class HostCfg:
    """The fields of qwen3_ref.Cfg (any object with these attributes is accepted instead)."""
    FIELDS = ("n_layer", "d", "ffn", "n_head", "n_kv", "head_dim", "vocab", "rope_theta", "eps", "tied",
              "yarn_factor", "yarn_orig_ctx")

    def __init__(self, n_layer, d, ffn, n_head, n_kv, head_dim, vocab, rope_theta=1e6, eps=1e-6,
                 tied=True, yarn_factor=0.0, yarn_orig_ctx=0):
        self.n_layer, self.d, self.ffn, self.n_head, self.n_kv = n_layer, d, ffn, n_head, n_kv
        self.head_dim, self.vocab, self.rope_theta, self.eps, self.tied = head_dim, vocab, rope_theta, eps, tied
        self.yarn_factor, self.yarn_orig_ctx = yarn_factor, yarn_orig_ctx

    @classmethod
    def of(cls, c):
        return cls(**{k: getattr(c, k) for k in cls.FIELDS})

    def asdict(self):
        return {k: getattr(self, k) for k in self.FIELDS}

    def __repr__(self):
        return f"HostCfg({self.asdict()})"


def cfg_from_gguf(g):
    """numpy-only copy of qwen3_ref.cfg_from_gguf (run_host.py checks they agree)."""
    m = g.meta
    a = m["general.architecture"]
    d = int(m[f"{a}.embedding_length"])
    nh = int(m[f"{a}.attention.head_count"])
    return HostCfg(
        n_layer=int(m[f"{a}.block_count"]), d=d, ffn=int(m[f"{a}.feed_forward_length"]),
        n_head=nh, n_kv=int(m[f"{a}.attention.head_count_kv"]),
        head_dim=int(m.get(f"{a}.attention.key_length", d // nh)),
        vocab=int(g.tensors["token_embd.weight"].shape[1]),
        rope_theta=float(m.get(f"{a}.rope.freq_base", 1e6)),
        eps=float(m.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)),
        tied="output.weight" not in g.tensors,
        yarn_factor=float(m.get(f"{a}.rope.scaling.factor", 0.0)) if m.get(f"{a}.rope.scaling.type") == "yarn" else 0.0,
        yarn_orig_ctx=int(m.get(f"{a}.rope.scaling.original_context_length", 0)))


def mat_name(mat_id, n_layer):
    return "lm_head" if mat_id == 7 * n_layer else f"l{mat_id // 7}.{KINDS[mat_id % 7]}"


def mat_id(name, n_layer):
    if name == "lm_head":
        return 7 * n_layer
    L, k = name.split(".")
    return 7 * int(L[1:]) + KINDS.index(k)


# ----------------------------------------------------------------------------- tables
def exp_table_np():
    k = np.arange(EXP_K + 1, dtype=np.float64)
    return (2.0 ** (k / EXP_K)).astype(np.float32)


def rsqrt_tables_np():
    lo = 0.5 + (np.arange(512) + 0.5) / 1024
    hi = 1.0 + (np.arange(512) + 0.5) / 512
    return (1 / np.sqrt(lo)).astype(np.float32), (1 / np.sqrt(hi)).astype(np.float32)


# torch's fp32 `base ** x` where it is NOT the correctly rounded value:
# (bits of fp32(base), head_dim) -> {index: bits torch returns}.  Measured against torch 2.4.1
# (x86-64, AVX2 kernels): theta 1e6, head_dim 128, index 37 (x = 0.578125): the exact value is
# 0.514 ulp above torch's 0x4537EBA2, so the correctly rounded result is 0x4537EBA3.
_POW_PINS = {(0x49742400, 128): {37: 0x4537EBA2}}


def rope_params_np(cfg):
    """numpy copy of qwen3_ref.rope_params: (inv_freq float32 [hd/2], attention factor, source).
    source = 'numpy+pin' when the config has a `_POW_PINS` entry (verified against torch),
    'numpy' otherwise (correctly rounded pow: equals torch wherever torch rounds correctly)."""
    hd, base = int(cfg.head_dim), float(cfg.rope_theta)
    one = np.float32(1.0)
    ex = np.arange(0, hd, 2, dtype=np.int64).astype(np.float32) / np.float32(hd)
    b32 = np.float32(base)
    pos_freqs = (np.float64(b32) ** ex.astype(np.float64)).astype(np.float32)
    pins = _POW_PINS.get((int(np.array(b32).view(np.uint32)), hd))
    if pins:
        pf = pos_freqs.view(np.uint32)
        for i, bits in pins.items():
            pf[i] = bits
    src = "numpy+pin" if pins else "numpy"
    if not cfg.yarn_factor:
        return one / pos_freqs, 1.0, src
    f, orig = float(cfg.yarn_factor), int(cfg.yarn_orig_ctx)
    att = 0.1 * math.log(f) + 1.0

    def corr_dim(nrot):
        return (hd * math.log(orig / (nrot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr_dim(32)), 0)
    high = min(math.ceil(corr_dim(1)), hd - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(hd // 2, dtype=np.float32) - np.float32(low)) / np.float32(high - low),
                   np.float32(0), one).astype(np.float32)
    extrap = one - ramp
    inv = (one / (np.float32(f) * pos_freqs)) * (one - extrap) + (one / pos_freqs) * extrap
    return inv.astype(np.float32), att, src


def _tables_from(cfg, max_pos, inv, att, source):
    ang = np.arange(max_pos, dtype=np.float64)[:, None] * np.asarray(inv, dtype=np.float32).astype(np.float64)[None]
    lo, hi = rsqrt_tables_np()
    return dict(EXP=exp_table_np(), RS_LO=lo, RS_HI=hi,
                COS=(np.cos(ang) * att).astype(np.float32), SIN=(np.sin(ang) * att).astype(np.float32),
                inv_freq=np.asarray(inv, dtype=np.float32).copy(), att=np.float64(att),
                eps=np.float32(cfg.eps), inv_sqrt_hd=np.float32(1 / math.sqrt(cfg.head_dim)),
                max_pos=np.int64(max_pos), head_dim=np.int64(cfg.head_dim),
                rope_theta=np.float64(cfg.rope_theta), yarn_factor=np.float64(cfg.yarn_factor),
                yarn_orig_ctx=np.int64(cfg.yarn_orig_ctx), source=np.array(source))


def build_tables(cfg, max_pos=4096):
    """All tables with numpy only, by the spec's code (FabricRef.__init__, exp_table,
    rsqrt_tables) on top of `rope_params_np`."""
    inv, att, src = rope_params_np(cfg)
    return _tables_from(cfg, max_pos, inv, att, src)


def build_tables_torch(cfg, max_pos=4096):
    """The same tables taken from the spec itself (needs torch): rope_params, exp_table and
    rsqrt_tables are imported from fabric/big/ref. This is the reference `build_tables` is
    checked against."""
    from ..ref import fabric_ref as fr
    from ..ref.qwen3_ref import rope_params
    inv, att = rope_params(cfg)
    t = _tables_from(cfg, max_pos, inv.numpy(), att, "torch")
    lo, hi = fr.rsqrt_tables("cpu")
    t.update(EXP=fr.exp_table("cpu").numpy().copy(), RS_LO=lo.numpy().copy(), RS_HI=hi.numpy().copy())
    return t


TABLE_ARRAYS = ("EXP", "RS_LO", "RS_HI", "COS", "SIN")


def tables_digest(t):
    h = hashlib.sha256()
    for k in TABLE_ARRAYS:
        h.update(np.ascontiguousarray(t[k], dtype=np.float32).tobytes())
    h.update(np.float32(t["eps"]).tobytes() + np.float32(t["inv_sqrt_hd"]).tobytes())
    return h.hexdigest()[:16]


def tables_equal(a, b):
    """byte identity of two table sets; returns the list of differing names (empty = equal)."""
    bad = [k for k in TABLE_ARRAYS if not (a[k].shape == b[k].shape and np.array_equal(
        np.asarray(a[k]).view(np.uint32), np.asarray(b[k]).view(np.uint32)))]
    for k in ("eps", "inv_sqrt_hd"):
        if np.float32(a[k]).tobytes() != np.float32(b[k]).tobytes():
            bad.append(k)
    return bad


def save_tables(path, t):
    np.savez_compressed(path, digest=np.array(tables_digest(t)), **{k: np.asarray(v) for k, v in t.items() if k != "digest"})
    return tables_digest(t)


def load_tables(path, cfg=None, max_pos=None):
    """Tables saved by `save_tables` (checked against their stored digest and, when `cfg` is
    given, against the model's RoPE / eps parameters)."""
    z = np.load(path, allow_pickle=False)
    t = {k: z[k] for k in z.files}
    for k in TABLE_ARRAYS:
        t[k] = _f32(t[k])
    if str(t["digest"]) != tables_digest(t):
        raise ValueError(f"{path}: table digest mismatch (file damaged?)")
    if cfg is not None:
        ok = (int(t["head_dim"]) == cfg.head_dim and float(t["rope_theta"]) == float(cfg.rope_theta)
              and float(t["yarn_factor"]) == float(cfg.yarn_factor)
              and int(t["yarn_orig_ctx"]) == int(cfg.yarn_orig_ctx)
              and np.float32(t["eps"]).tobytes() == np.float32(cfg.eps).tobytes())
        if not ok:
            raise ValueError(f"{path}: tables were built for another model configuration")
    if max_pos is not None and int(t["max_pos"]) < max_pos:
        raise ValueError(f"{path}: tables cover {int(t['max_pos'])} positions, {max_pos} wanted")
    return t


# ----------------------------------------------------------------------------- weights
def load_gguf_host(path):
    """(cfg, gains, emb, gguf) with numpy only: the RMSNorm gains and the embedding, which is
    NOT copied (the C library reads rows straight out of the memory-mapped GGUF blocks)."""
    from ..ref.gguf_q1 import GGUF, Q1_BLOCK_BYTES
    g = GGUF(path)
    cfg = cfg_from_gguf(g)

    def vec(n):
        return _f32(g.dequant(n).reshape(-1))

    gains = {k: np.stack([vec(f"blk.{L}.{gk}.weight") for L in range(cfg.n_layer)]) for k, gk in GG.items()}
    gains["out_norm"] = vec("output_norm.weight")
    ti = g.tensors["token_embd.weight"]
    assert g.is_q1("token_embd.weight") and ti.shape[0] == cfg.d, "embedding is not a Q1_0 [vocab, d] tensor"
    emb = emb_from_blocks(g.raw("token_embd.weight"), Q1_BLOCK_BYTES)
    return cfg, gains, emb, g


def emb_from_blocks(raw, block_bytes=18):
    """Embedding as the GGUF stores it: blocks of fp16 scale || 16 sign bytes."""
    return dict(keep=raw, base=raw.ctypes.data, scales_off=0, signs_off=2, sign_stride=block_bytes,
                scale_stride=block_bytes)


def emb_from_split(signs, scales):
    """Embedding as two arrays: signs uint8 [vocab, d/8], scales fp16 [vocab, d/128]."""
    signs = np.ascontiguousarray(signs, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float16)
    return dict(keep=(signs, scales), signs_ptr=signs.ctypes.data, scales_ptr=scales.ctypes.data,
                sign_stride=16, scale_stride=2)


def argmax(lg, lib=None):
    """the LOWEST index holding the maximum (the spec's tie rule)."""
    lib = lib or load_lib()
    lg = _f32(lg)
    return int(lib.bk_argmax(_ptr(lg), lg.size))


# ----------------------------------------------------------------------------- the host
class BkError(RuntimeError):
    def __init__(self, rc, what=""):
        super().__init__(f"{ERRS.get(rc, rc)} {what}".strip())
        self.rc = rc


class BkHost:
    """One decode stream. gains: dict attn_norm [L, d], q_norm [L, hd], k_norm [L, hd],
    ffn_norm [L, d], out_norm [d] (a per-layer dict 'l3.attn_norm' ... as in the spec's W is
    accepted too). emb: `emb_from_blocks` / `emb_from_split`. tables: `build_tables*` /
    `load_tables` (or a path). gemv: Python callable (mat_id, xq, xe, y, same_input).
    threads: None = $BK_THREADS, else min(4, CPUs); 1 = no worker threads. kernels: None =
    $BK_KERNELS, else "fast"; "ref" = the scalar reference kernels. spin_us: how long idle
    workers spin before sleeping (None = $BK_SPIN_US, else 200; 0 = sleep at once).
    The result does not depend on any of the three."""

    def __init__(self, cfg, gains, emb, tables, gemv=None, max_ctx=None, init_ctx=0, lib=None,
                 threads=None, kernels=None, spin_us=None):
        self.cfg = cfg if isinstance(cfg, HostCfg) else HostCfg.of(cfg)
        c = self.cfg
        self.lib = lib or load_lib()
        if isinstance(tables, (str, os.PathLike)):
            tables = load_tables(tables, c)
        self.tables = tables
        max_pos = int(tables["COS"].shape[0])
        max_ctx = max_pos if max_ctx is None else int(max_ctx)
        if max_ctx > max_pos:
            raise ValueError(f"max_ctx {max_ctx} > the {max_pos} positions of the RoPE tables")
        if tables["COS"].shape[1] != c.head_dim // 2:
            raise ValueError("RoPE tables are for another head_dim")
        if "attn_norm" not in gains:            # the spec's flat W dict
            gains = dict({k: np.stack([np.asarray(gains[f"l{L}.{k}"]) for L in range(c.n_layer)]) for k in NORMS},
                         out_norm=np.asarray(gains["out_norm"]))
        self._g = {k: _f32(gains[k]) for k in (*NORMS, "out_norm")}
        want = dict(attn_norm=(c.n_layer, c.d), q_norm=(c.n_layer, c.head_dim), k_norm=(c.n_layer, c.head_dim),
                    ffn_norm=(c.n_layer, c.d), out_norm=(c.d,))
        for k, shp in want.items():
            if self._g[k].shape != shp:
                raise ValueError(f"gain {k}: shape {self._g[k].shape}, want {shp}")
        self._t = {k: _f32(tables[k]) for k in TABLE_ARRAYS}
        assert self._t["EXP"].shape == (EXP_K + 1,) and self._t["RS_LO"].shape == (512,) and self._t["RS_HI"].shape == (512,)
        self._emb = emb
        w = BkWeights()
        for k in (*NORMS, "out_norm"):
            setattr(w, k, self._g[k].ctypes.data)
        if "base" in emb:
            w.emb_signs, w.emb_scales = emb["base"] + emb["signs_off"], emb["base"] + emb["scales_off"]
        else:
            w.emb_signs, w.emb_scales = emb["signs_ptr"], emb["scales_ptr"]
        w.emb_sign_stride, w.emb_scale_stride = emb["sign_stride"], emb["scale_stride"]
        w.exp_tab, w.rs_lo, w.rs_hi = (self._t[k].ctypes.data for k in ("EXP", "RS_LO", "RS_HI"))
        w.cos_tab, w.sin_tab = self._t["COS"].ctypes.data, self._t["SIN"].ctypes.data
        kern = {None: KERN_AUTO, "auto": KERN_AUTO, "fast": KERN_FAST, "ref": KERN_REF}.get(kernels, kernels)
        bc = BkConfig(c.n_layer, c.d, c.ffn, c.n_head, c.n_kv, c.head_dim, c.vocab, max_ctx, max_pos, init_ctx,
                      float(np.float32(tables["eps"])), float(np.float32(tables["inv_sqrt_hd"])),
                      int(threads or 0), 0 if spin_us is None else (int(spin_us) if spin_us > 0 else -1),
                      int(kern), 0)
        err = c_int(0)
        self._s = self.lib.bk_init(ctypes.byref(bc), ctypes.byref(w), ctypes.byref(err))
        if not self._s:
            raise BkError(err.value, "bk_init (BK_ERR_FPENV: the library was built with the wrong flags)"
                          if err.value == -7 else "bk_init")
        self._w, self._bc = w, bc
        self.n_mats = self.lib.bk_n_mats(self._s)
        self.threads = self.lib.bk_threads(self._s)             # as resolved by the library
        self.kernels = KERN_NAMES[self.lib.bk_kernels(self._s)]
        self.logits = np.zeros(c.vocab, dtype=np.float32)
        self._gemv = gemv
        self._exc = None
        self._last_xq = None
        self._cb = GEMV_FN(self._tramp)
        self._c_fn = None
        self._taps = None
        self.taps = {}
        self.info = []
        for m in range(self.n_mats):
            n_in, rows, a, b, y = c_int(), c_int(), c_int64(), c_int64(), c_int64()
            self.lib.bk_mat_info(self._s, m, ctypes.byref(n_in), ctypes.byref(rows), ctypes.byref(a),
                                 ctypes.byref(b), ctypes.byref(y))
            self.info.append(dict(name=mat_name(m, c.n_layer), n_in=n_in.value, rows=rows.value,
                                  xq_off=a.value, xe_off=b.value, y_off=y.value))

    @classmethod
    def from_gguf(cls, path, gemv=None, tables=None, max_pos=4096, **kw):
        """Host weights from a GGUF. tables: a dict, an .npz path, or None = build them here
        with numpy (see the module docstring for why a shipped .npz is the safe choice)."""
        cfg, gains, emb, g = load_gguf_host(path)
        if tables is None:
            tables = build_tables(cfg, max_pos)
        h = cls(cfg, gains, emb, tables, gemv=gemv, **kw)
        h._gguf = g
        return h

    def close(self):
        """Free the state and stop its worker threads (also done on garbage collection)."""
        s, self._s = getattr(self, "_s", None), None
        if s:
            self.lib.bk_free(s)

    __del__ = close

    # --- GEMV plumbing ---
    def _tramp(self, _ctx, mid, xq_p, xe_p, groups, y_p, rows):
        try:
            n = groups * QK
            xq = np.frombuffer((c_int16 * n).from_address(xq_p), dtype=np.int16)
            xe = np.frombuffer((c_int8 * groups).from_address(xe_p), dtype=np.int8)
            y = np.frombuffer((c_float * rows).from_address(y_p), dtype=np.float32)
            same = xq_p == self._last_xq
            self._last_xq = xq_p
            r = self._gemv(mid, xq, xe, y, same)
            return int(r or 0)
        except BaseException as ex:             # never unwind through C
            self._exc = ex
            return -1

    def set_gemv(self, fn):
        self._gemv, self._c_fn = fn, None

    def set_c_gemv(self, fn_ptr, ctx=None):
        """Use a C function (bk_gemv_fn) directly, e.g. `host.lib.bk_bench_gemv`."""
        self._c_fn = (ctypes.cast(fn_ptr, c_void_p), ctx)

    # --- taps ---
    def enable_taps(self, names=("xq", "xe", "y"), att_cap=None):
        """Allocate tap buffers; after each step `self.taps[name]` holds that step's values
        (see bk_taps in bk_host.h for the shapes). names=() turns taps off."""
        c = self.cfg
        if not names:
            self._taps, self.taps = None, {}
            return
        nxq, nxe, ny = c_int64(), c_int64(), c_int64()
        self.lib.bk_tap_sizes(self._s, ctypes.byref(nxq), ctypes.byref(nxe), ctypes.byref(ny))
        att_cap = int(att_cap or self._bc.max_ctx)
        nq, nkv = c.n_head * c.head_dim, c.n_kv * c.head_dim
        shapes = dict(xq=((nxq.value,), np.int16), xe=((nxe.value,), np.int8), y=((ny.value,), np.float32),
                      x_in=((c.n_layer, c.d), np.float32), x_mid=((c.n_layer, c.d), np.float32),
                      x_out=((c.n_layer, c.d), np.float32), q_rope=((c.n_layer, nq), np.float32),
                      k_rope=((c.n_layer, nkv), np.float32),
                      att_s=((c.n_layer, c.n_head, att_cap), np.float32),
                      att_p=((c.n_layer, c.n_head, att_cap), np.float32),
                      ctx=((c.n_layer, nq), np.float32), silu=((c.n_layer, c.ffn), np.float32),
                      act=((c.n_layer, c.ffn), np.float32), h_out=((c.d,), np.float32))
        t = BkTaps()
        t.att_cap = att_cap
        self.taps = {}
        for n in names:
            shp, dt = shapes[n]
            self.taps[n] = np.zeros(shp, dtype=dt)
            setattr(t, n, self.taps[n].ctypes.data)
        self._taps = t

    def tap_gemv(self, m):
        """(xq, xe, y) views of GEMV `m` (id or name) in the tap buffers of the last step."""
        if isinstance(m, str):
            m = mat_id(m, self.cfg.n_layer)
        i = self.info[m]
        g = lambda k, off, n: self.taps[k][off:off + n] if k in self.taps else None  # noqa: E731
        return (g("xq", i["xq_off"], i["n_in"]), g("xe", i["xe_off"], i["n_in"] // QK),
                g("y", i["y_off"], i["rows"]))

    # --- stepping ---
    @property
    def pos(self):
        return self.lib.bk_pos(self._s)

    def reset(self):
        self.lib.bk_reset(self._s)

    def truncate(self, pos):
        rc = self.lib.bk_truncate(self._s, int(pos))
        if rc:
            raise BkError(rc, "bk_truncate")

    def step(self, token, want_logits=True, copy=True):
        """One token. Returns the logits (float32 [vocab]) or None when want_logits is False.
        copy=False returns the internal buffer, overwritten by the next step."""
        if self._c_fn is not None:
            fn, ctx = self._c_fn
        else:
            if self._gemv is None:
                raise ValueError("no GEMV callback set")
            fn, ctx = ctypes.cast(self._cb, c_void_p), None
        self._exc = None
        rc = self.lib.bk_step(self._s, int(token), 1 if want_logits else 0, fn, ctx,
                              self.logits.ctypes.data if want_logits else None,
                              ctypes.byref(self._taps) if self._taps is not None else None)
        if rc:
            if self._exc is not None:
                ex, self._exc = self._exc, None
                raise ex
            raise BkError(rc, f"bk_step at position {self.pos}"
                          + (f" (callback returned {self.lib.bk_last_gemv_rc(self._s)})" if rc == -4 else ""))
        if not want_logits:
            return None
        return self.logits.copy() if copy else self.logits

    def kv_rows(self, layer, kv_head):
        """The cache as stored for (layer, kv head): kq int16 [pos, hd], ke int8 [pos, hd/16],
        vq int8 [pos, hd], ve int8 [pos, hd/128] (copies)."""
        ps = [c_void_p() for _ in range(4)]
        rc = self.lib.bk_kv_rows(self._s, layer, kv_head, *[ctypes.byref(p) for p in ps])
        if rc:
            raise BkError(rc, "bk_kv_rows")
        n, hd = self.pos, self.cfg.head_dim
        out = []
        for p, (ct, dt, w) in zip(ps, ((c_int16, np.int16, hd), (c_int8, np.int8, hd // 16),
                                       (c_int8, np.int8, hd), (c_int8, np.int8, hd // QK))):
            out.append(np.frombuffer((ct * (n * w)).from_address(p.value), dtype=dt).reshape(n, w).copy()
                       if n else np.zeros((0, w), dtype=dt))
        return out

    def kv_row(self, layer, kv_head, pos):
        """One cached position (kq [hd], ke [hd/16], vq [hd], ve [hd/128]), copies."""
        ps = [c_void_p() for _ in range(4)]
        rc = self.lib.bk_kv_rows(self._s, layer, kv_head, *[ctypes.byref(p) for p in ps])
        if rc or not 0 <= pos < self.pos:
            raise BkError(rc or -1, "bk_kv_rows")
        hd = self.cfg.head_dim
        return [np.frombuffer((ct * w).from_address(p.value + pos * w * ctypes.sizeof(ct)), dtype=dt).copy()
                for p, (ct, dt, w) in zip(ps, ((c_int16, np.int16, hd), (c_int8, np.int8, hd // 16),
                                               (c_int8, np.int8, hd), (c_int8, np.int8, hd // QK)))]

    def stats(self, reset=False):
        """Timing since the last reset: ns per bucket (T_NAMES; wall time on the calling
        thread, a parallel region's time split over its buckets), host_ns = everything but the
        callback and the tap copies, cpu_ns = the same buckets summed over the threads, plus
        counters."""
        st = BkStats()
        self.lib.bk_get_stats(self._s, ctypes.byref(st))
        d = {n: int(st.ns[i]) for i, n in enumerate(T_NAMES)}
        cpu = {n: int(st.cpu_ns[i]) for i, n in enumerate(T_NAMES)}
        out = dict(ns=d, steps=int(st.steps), gemv_calls=int(st.gemv_calls), att_pos=int(st.att_pos),
                   gemv_ns=d["gemv"], host_ns=sum(v for k, v in d.items() if k not in ("gemv", "taps")),
                   cpu_ns=cpu, host_cpu_ns=sum(v for k, v in cpu.items() if k not in ("gemv", "taps")),
                   regions=int(st.regions), threads=int(st.threads), kernels=KERN_NAMES.get(st.kernels, "?"),
                   worker_units=int(st.worker_units))
        if reset:
            self.lib.bk_reset_stats(self._s)
        return out

    def reset_stats(self):
        self.lib.bk_reset_stats(self._s)


# ----------------------------------------------------------------------------- primitives
class Prims:
    """numpy front end of the exported spec primitives (unit gates)."""

    def __init__(self, lib=None, tables=None, ref=False):
        """ref=True: the scalar reference kernels (bk_ref_*) where the library has two."""
        self.lib = lib or load_lib()
        self.ref = bool(ref)
        lo, hi = rsqrt_tables_np()
        t = tables or dict(EXP=exp_table_np(), RS_LO=lo, RS_HI=hi)
        self.EXP, self.LO, self.HI = _f32(t["EXP"]), _f32(t["RS_LO"]), _f32(t["RS_HI"])

    def _fn(self, name):
        return getattr(self.lib, ("bk_ref_" if self.ref else "bk_") + name)

    def fsum(self, x):
        x = _f32(x)
        return np.float32(self._fn("fsum")(_ptr(x), x.size))

    def silu_mul(self, gate, up):
        """(silu(gate), silu(gate) * up): the FFN activation kernel."""
        g, u = _f32(gate).reshape(-1), _f32(up).reshape(-1)
        sil, act = np.empty_like(g), np.empty_like(g)
        self.lib.bk_silu_mul(KERN_REF if self.ref else KERN_FAST, _ptr(self.EXP), _ptr(g), _ptr(u), _ptr(sil),
                             _ptr(act), g.size)
        return sil, act

    def attend(self, q, kq, ke, vq, ve, inv_sqrt_hd=None):
        """The attention of one kv head. q [rep, hd] fp32, the cache as stored: kq int16
        [Tk, hd], ke int8 [Tk, hd/16], vq int8 [Tk, hd], ve int8 [Tk, hd/128]. Returns
        (scores [rep, Tk], p [rep, Tk], ctx [rep, hd])."""
        q = _f32(q)
        rep_, hd = q.shape
        kq, ke = np.ascontiguousarray(kq, dtype=np.int16), np.ascontiguousarray(ke, dtype=np.int8)
        vq, ve = np.ascontiguousarray(vq, dtype=np.int8), np.ascontiguousarray(ve, dtype=np.int8)
        Tk = kq.shape[0]
        assert kq.shape == (Tk, hd) and ke.shape == (Tk, hd // 16) and vq.shape == (Tk, hd) and ve.shape == (Tk, hd // QK)
        isq = np.float32(1.0) / np.sqrt(np.float32(hd)) if inv_sqrt_hd is None else np.float32(inv_sqrt_hd)
        sc, pp = np.empty((rep_, Tk), dtype=np.float32), np.empty((rep_, Tk), dtype=np.float32)
        ctx = np.empty((rep_, hd), dtype=np.float32)
        rc = self.lib.bk_attend_kv(KERN_REF if self.ref else KERN_FAST, _ptr(self.EXP), hd, rep_, Tk, float(isq),
                                   _ptr(q), _ptr(kq), _ptr(ke), _ptr(vq), _ptr(ve), _ptr(sc), _ptr(pp), _ptr(ctx))
        if rc:
            raise BkError(rc, "bk_attend_kv")
        return sc, pp, ctx

    def rsqrt(self, v):
        v = _f32(v).reshape(-1)
        return np.array([self.lib.bk_rsqrt_fab(_ptr(self.LO), _ptr(self.HI), float(x)) for x in v], dtype=np.float32)

    def exp(self, x):
        x = _f32(x).reshape(-1)
        y = np.empty_like(x)
        self._fn("exp_fab_n")(_ptr(self.EXP), _ptr(x), _ptr(y), x.size)
        return y

    def exp1(self, x):
        return np.float32(self.lib.bk_exp_fab(_ptr(self.EXP), float(np.float32(x))))

    def silu(self, x):
        x = _f32(x).reshape(-1)
        y = np.empty_like(x)
        self._fn("silu")(_ptr(self.EXP), _ptr(x), _ptr(y), x.size)
        return y

    def rmsnorm(self, x, w, eps):
        x, w = _f32(x), _f32(w)
        y = np.empty_like(x)
        self._fn("rmsnorm")(_ptr(self.LO), _ptr(self.HI), float(np.float32(eps)), _ptr(x), _ptr(w), x.size, _ptr(y))
        return y

    def rope(self, cos, sin, x):
        cos, sin, x = _f32(cos), _f32(sin), _f32(x)
        y = np.empty_like(x)
        self._fn("rope")(_ptr(cos), _ptr(sin), _ptr(x), x.size, _ptr(y))
        return y

    def quant(self, x, bits, group, width=16):
        """(q, e, rc): the spec's bfp_quant over a flat vector."""
        x = _f32(x).reshape(-1)
        q = np.zeros(x.size, dtype=np.int16 if width == 16 else np.int8)
        e = np.zeros(x.size // group, dtype=np.int8)
        fn = self._fn("bfp_quant16" if width == 16 else "bfp_quant8")
        rc = fn(_ptr(x), x.size, group, bits, _ptr(q), _ptr(e))
        return q, e, rc

    def dequant(self, q, e, group):
        q, e = np.ascontiguousarray(q), np.ascontiguousarray(e, dtype=np.int8)
        x = np.empty(q.size, dtype=np.float32)
        (self.lib.bk_bfp_dequant16 if q.dtype == np.int16 else self.lib.bk_bfp_dequant8)(
            _ptr(q), _ptr(e), q.size, group, _ptr(x))
        return x

    def embed_row(self, emb, d, row):
        x = np.empty(d, dtype=np.float32)
        if "base" in emb:
            sg, sc = emb["base"] + emb["signs_off"], emb["base"] + emb["scales_off"]
        else:
            sg, sc = emb["signs_ptr"], emb["scales_ptr"]
        self.lib.bk_embed_row(sg, emb["sign_stride"], sc, emb["scale_stride"], d, row, _ptr(x))
        return x

    def argmax(self, x):
        x = _f32(x)
        return int(self.lib.bk_argmax(_ptr(x), x.size))


# ----------------------------------------------------------------------------- replay file
def save_replay(path, cfg, gains, signs, scales, tables, steps):
    """A self-contained record of a run for a machine that has no torch (the board): model
    config, gains, embedding, tables, and per step the token, want_logits, every GEMV output
    the callback returned and every GEMV input + logits the spec produced.
    steps: list of dict(token, want, y=[per mat float32], xq=[...], xe=[...], logits|None)."""
    c = HostCfg.of(cfg)
    d = dict(cfg=np.array(json.dumps(c.asdict())), signs=np.asarray(signs, dtype=np.uint8),
             scales=np.asarray(scales, dtype=np.float16), tokens=np.array([s["token"] for s in steps], dtype=np.int32),
             want=np.array([int(s["want"]) for s in steps], dtype=np.int8))
    for k in (*NORMS, "out_norm"):
        d["gain." + k] = _f32(gains[k])
    for k, v in tables.items():
        d["tab." + k] = np.asarray(v)
    d["y"] = np.concatenate([np.concatenate([_f32(y).view(np.uint32) for y in s["y"]]) for s in steps])
    d["xq"] = np.concatenate([np.concatenate([np.asarray(q, dtype=np.int16) for q in s["xq"]]) for s in steps])
    d["xe"] = np.concatenate([np.concatenate([np.asarray(e, dtype=np.int8) for e in s["xe"]]) for s in steps])
    d["logits"] = np.concatenate([_f32(s["logits"]).view(np.uint32) for s in steps if s["want"]] or [np.zeros(0, np.uint32)])
    np.savez_compressed(path, **d)


def _same(a, b, nan_ok=False):
    """Bit-identical float32 arrays. nan_ok: a NaN matches any NaN (payloads not compared)."""
    a, b = _f32(a).reshape(-1), _f32(b).reshape(-1)
    if a.size != b.size:
        return False
    ua, ub = a.view(np.uint32), b.view(np.uint32)
    if nan_ok:
        na, nb = np.isnan(a), np.isnan(b)
        return bool(np.array_equal(na, nb) and np.array_equal(ua[~na], ub[~nb]))
    return bool(np.array_equal(ua, ub))


def _adversarial(rng, n, kind):
    """float32 [n] test vectors. kind: 'norm', 'wide' (exponents all over fp32), 'tiny'
    (subnormals and values that underflow), 'edge' (zeros of both signs, ties, limits)."""
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if kind == "norm":
        return rng.standard_normal(n).astype(np.float32) * np.float32(rng.choice([1e-3, 1.0, 7.0, 300.0]))
    if kind == "wide":
        return (rng.standard_normal(n) * np.exp2(rng.integers(-60, 60, n))).astype(np.float32)
    if kind == "tiny":
        return (rng.standard_normal(n) * np.exp2(rng.integers(-160, -110, n))).astype(np.float32)
    x = rng.standard_normal(n).astype(np.float32)
    edge = np.array([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.5, -2.5, 3.5, 88.0, -88.0, 88.73, -87.34, -103.0,
                     -104.0, 127.5, 2 ** -126, -2 ** -126, 2 ** -149, 3.4e38, -3.4e38, 1e-30, 4095.5 / 4096,
                     510.5, 511.5, -511.5, 126.5, 127.49999, 1 + 2 ** -23, 0.999999940395], dtype=np.float32)
    idx = rng.integers(0, n, max(1, n // 3))
    x[idx] = edge[rng.integers(0, edge.size, idx.size)]
    return x


def kern_check(lib=None, seed=0, verbose=True, quick=False):
    """The FAST kernels against the scalar REFERENCE kernels of the same library, bit for
    bit, on adversarial inputs (sizes around the block boundaries, exponents all over fp32,
    subnormals, zeros of both signs, ties, the exponents that take the fall-back paths).
    numpy only, so it runs on the board. Returns (ok, {kernel: (cases, bad)})."""
    lib = lib or load_lib()
    rng = np.random.default_rng(seed)
    F, R = Prims(lib), Prims(lib, ref=True)
    res = {}

    def note(name, same):
        c, b = res.get(name, (0, 0))
        res[name] = (c + 1, b + (not same))

    kinds = ("norm", "wide", "tiny", "edge")
    sizes = [0, 1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257,
             511, 512, 513, 527, 528, 529, 1023, 1024, 1025, 1040, 4096, 12288]
    if quick:
        sizes = sizes[::3]
    for n in sizes:
        for kind in kinds:
            x = _adversarial(rng, n, kind)
            note("fsum", _same(F.fsum(x), R.fsum(x)))
            if n == 0:
                continue
            xe = np.clip(x, -300, 300) if kind != "wide" else x          # exp: also far outside the table range
            note("exp", _same(F.exp(xe), R.exp(xe)))
            note("silu", _same(F.silu(xe), R.silu(xe)))
            u = _adversarial(rng, n, "norm")
            (s1, a1), (s2, a2) = F.silu_mul(xe, u), R.silu_mul(xe, u)
            note("silu_mul", _same(s1, s2) and _same(a1, a2))
            w = _adversarial(rng, n, "norm")
            for eps in (1e-6, 1e-5):
                note("rmsnorm", _same(F.rmsnorm(x, w, eps), R.rmsnorm(x, w, eps)))
            if n % 2 == 0:
                ang = rng.uniform(-np.pi, np.pi, n // 2)
                c_, s_ = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
                note("rope", _same(F.rope(c_, s_, x), R.rope(c_, s_, x)))
            for bits_, grp, width in ((10, 16, 16), (10, 128, 16), (8, 128, 8), (8, 16, 8), (10, 8, 16), (10, 24, 16),
                                      (6, 32, 8)):
                if n % grp:
                    continue
                for xx in (x, np.where(rng.random(n) < 0.5, np.float32(0), x).astype(np.float32),
                           (np.round(np.clip(x, -1e30, 1e30) * 8) / 16).astype(np.float32)):   # zeros; ties
                    qf, ef, rf = F.quant(xx, bits_, grp, width)
                    qr, er, rr = R.quant(xx, bits_, grp, width)
                    note(f"quant{width}", rf == rr and np.array_equal(qf, qr) and np.array_equal(ef, er))
    # NaN / inf: exp is bit-identical; silu gives a NaN wherever the reference does
    for n in (16, 33, 600):
        x = _adversarial(rng, n, "edge")
        x[rng.integers(0, n, 3)] = [np.nan, np.inf, -np.inf]
        note("exp_nan_inf", _same(F.exp(x), R.exp(x)))
        note("silu_nan_inf", _same(F.silu(x), R.silu(x), nan_ok=True))
        u = _adversarial(rng, n, "norm")
        (s1, a1), (s2, a2) = F.silu_mul(x, u), R.silu_mul(x, u)
        note("silu_nan_inf", _same(s1, s2, nan_ok=True) and _same(a1, a2, nan_ok=True))
    # attention of one kv head: scores, softmax and context
    reps = (1, 2, 3, 4, 8, 9)
    tks = (1, 2, 15, 16, 17, 31, 32, 33, 100, 257)
    if quick:
        reps, tks = (1, 3, 4), (1, 16, 17, 100)
    for hd in (128, 256):
        for rep_ in reps:
            for Tk in tks:
                for mode in ("typ", "wide", "tiny", "peaky", "zeros"):
                    q = rng.standard_normal((rep_, hd)).astype(np.float32)
                    kq = rng.integers(-511, 512, (Tk, hd)).astype(np.int16)
                    vq = rng.integers(-127, 128, (Tk, hd)).astype(np.int8)
                    ke = rng.integers(-12, -6, (Tk, hd // 16)).astype(np.int8)
                    ve = rng.integers(-10, -3, (Tk, hd // QK)).astype(np.int8)
                    if mode == "wide":              # normal fp32 scales of every size, products that go subnormal
                        ke = rng.integers(-126, -5, (Tk, hd // 16)).astype(np.int8)
                        ve = rng.integers(-126, 20, (Tk, hd // QK)).astype(np.int8)
                        q *= np.float32(4.0)
                    elif mode == "tiny":            # exponents below -126: 2^e is not a normal fp32
                        ke[rng.random(ke.shape) < 0.2] = rng.integers(-128, -126)
                        ve[rng.random(ve.shape) < 0.3] = rng.integers(-128, -126)
                        ve[rng.random(ve.shape) < 0.3] = -120
                    elif mode == "peaky":           # softmax with zeros and subnormals in p
                        q *= np.float32(rng.choice([0.5, 3.0, 12.0]))
                        ke = rng.integers(-9, -2, (Tk, hd // 16)).astype(np.int8)
                        ve = rng.integers(-126, -100, (Tk, hd // QK)).astype(np.int8)
                    elif mode == "zeros":           # zero rows / zero q: sums of signed zeros
                        kq[rng.random(Tk) < 0.5] = 0
                        vq[rng.random(Tk) < 0.5] = 0
                        vq[rng.random((Tk, hd)) < 0.5] = 0
                        q[rng.random(rep_) < 0.3] = 0
                        q[rng.random((rep_, hd)) < 0.3] *= np.float32(-0.0)
                    a, b_ = F.attend(q, kq, ke, vq, ve), R.attend(q, kq, ke, vq, ve)
                    note("att_score", _same(a[0], b_[0]))
                    note("att_softmax", _same(a[1], b_[1]))
                    note("att_ctx", _same(a[2], b_[2]))
    ok = all(b == 0 for _, b in res.values())
    if lib.bk_simd() == 0:
        ok = False                                   # nothing was compared: fast == ref in this build
    if verbose:
        print(f"HOST_KERN_VERDICT {'PASS' if ok else 'FAIL'} (fast kernels [{SIMD_NAMES[lib.bk_simd()]}] vs the scalar "
              f"reference kernels, bit for bit: " + ", ".join(f"{k} {c - b}/{c}" for k, (c, b) in res.items())
              + f"; {platform.machine()}, lib {Path(lib._path).name})")
    return ok, res


def replay(path, lib=None, verbose=True, threads=None, kernels=None, taps=False):
    """Run the C library over a `save_replay` file with the recorded GEMV outputs as the
    callback and compare every GEMV input and every logit with the record. numpy only: this
    is the bit-identity check to run on the board (aarch64). Returns (ok, summary).
    taps=True also returns summary['digest']: a hash over every tap of every step (residual
    stream, q/k after RoPE, scores, softmax, context, silu, act) and the final KV rows, to
    compare runs with different thread counts / kernel sets with each other."""
    z = np.load(path, allow_pickle=False)
    cfg = HostCfg(**json.loads(str(z["cfg"])))
    gains = {k: z["gain." + k] for k in (*NORMS, "out_norm")}
    tables = {k[4:]: z[k] for k in z.files if k.startswith("tab.")}
    host = BkHost(cfg, gains, emb_from_split(z["signs"], z["scales"]), tables, lib=lib, threads=threads,
                  kernels=kernels)
    Y, XQ, XE, LG = z["y"].view(np.float32), z["xq"], z["xe"], z["logits"]
    cur = dict(y=0, xq=0, xe=0, lg=0)
    bad = dict(xq=0, xe=0, logits=0)
    n_gemv = 0
    h = hashlib.sha256()
    ext = ("x_in", "x_mid", "x_out", "q_rope", "k_rope", "att_s", "att_p", "ctx", "silu", "act", "h_out")
    if taps:
        host.enable_taps(("xq", "xe", "y") + ext, att_cap=len(z["tokens"]) + 1)

    def gemv(mid, xq, xe, y, same):
        nonlocal n_gemv
        i = host.info[mid]
        bad["xq"] += int(not np.array_equal(xq, XQ[cur["xq"]:cur["xq"] + i["n_in"]]))
        bad["xe"] += int(not np.array_equal(xe, XE[cur["xe"]:cur["xe"] + i["n_in"] // QK]))
        y[:] = Y[cur["y"]:cur["y"] + i["rows"]]
        cur["xq"] += i["n_in"]
        cur["xe"] += i["n_in"] // QK
        cur["y"] += i["rows"]
        n_gemv += 1

    host.set_gemv(gemv)
    for step, (tok, want) in enumerate(zip(z["tokens"], z["want"])):
        lg = host.step(int(tok), bool(want), copy=False)
        if want:
            ref = LG[cur["lg"]:cur["lg"] + cfg.vocab]
            bad["logits"] += int(not np.array_equal(lg.view(np.uint32), ref))
            cur["lg"] += cfg.vocab
        if taps:
            for k in ext:
                t = host.taps[k]
                if k in ("att_s", "att_p"):
                    t = t[:, :, :step + 1]
                elif k == "h_out" and not want:
                    continue
                h.update(np.ascontiguousarray(t).tobytes())
    if taps:
        for L in range(cfg.n_layer):
            for g in range(cfg.n_kv):
                for a in host.kv_rows(L, g):
                    h.update(a.tobytes())
    st = host.stats()
    ok = not any(bad.values()) and cur["y"] == Y.size and cur["xq"] == XQ.size
    summ = dict(steps=int(len(z["tokens"])), gemvs=n_gemv, bad=bad, machine=platform.machine(),
                tables=tables_digest(tables), threads=host.threads, kernels=host.kernels,
                worker_units=st["worker_units"], regions=st["regions"])
    if taps:
        summ["digest"] = h.hexdigest()[:16]
    if verbose:
        print(f"HOST_REPLAY_VERDICT {'PASS' if ok else 'FAIL'} ({summ['steps']} tokens, {n_gemv} GEMV inputs and "
              f"the logits of {int(z['want'].sum())} steps vs the spec's record; mismatching xq {bad['xq']}, "
              f"xe {bad['xe']}, logits {bad['logits']}; threads {host.threads}, kernels {host.kernels}; "
              f"{platform.machine()}, lib {Path(host.lib._path).name})")
    host.close()
    return ok, summ


def threads_check(path, lib=None, threads=(1, 3, 4, 2), verbose=True):
    """The replay with every thread count x both kernel sets, all taps on: each run must
    match the spec's record, and all runs must agree on every tap and on the KV cache.
    numpy only. Returns (ok, [summary per run])."""
    lib = lib or load_lib()
    runs, ok = [], True
    for k in ("ref", "fast"):
        for th in threads:
            r_ok, summ = replay(path, lib, verbose=False, threads=th, kernels=k, taps=True)
            ok &= r_ok
            runs.append(summ)
    digests = {r["digest"] for r in runs}
    ok &= len(digests) == 1
    # the threaded runs must really have used their workers
    used = all(r["worker_units"] > 0 for r in runs if r["threads"] > 1)
    ok &= used
    if verbose:
        print(f"HOST_THREADS_VERDICT {'PASS' if ok else 'FAIL'} ({len(runs)} runs of the replay, threads "
              f"{'/'.join(str(t) for t in threads)} x kernels ref/fast, each vs the spec's record and "
              f"{'all with the same' if len(digests) == 1 else str(len(digests)) + ' DIFFERENT'} digest over every tap "
              f"and the KV cache ({runs[0]['digest']}); region units run by workers: "
              + ", ".join(f"{r['kernels']}/{r['threads']}t {r['worker_units']}" for r in runs if r["threads"] > 1)
              + f"; {platform.machine()})")
    return ok, runs


# ----------------------------------------------------------------------------- bench
SHAPES = {"8b": dict(n_layer=36, d=4096, ffn=12288, n_head=32, n_kv=8, head_dim=128),
          "1.7b": dict(n_layer=28, d=2048, ffn=6144, n_head=16, n_kv=8, head_dim=128),
          "tiny": dict(n_layer=3, d=256, ffn=384, n_head=4, n_kv=2, head_dim=128)}
HOST_BUCKETS = [k for k in T_NAMES if k not in ("gemv", "taps")]


def bench(shape="8b", ctxs=(20, 64, 256), reps=5, lib=None, vocab=1024, seed=0, verbose=True,
          threads=None, kernels=None, spin_us=None, gap_us=0):
    """Host time per token of a model SHAPE with the C stand-in GEMV (random outputs of a
    plausible magnitude, not the model): what the vector path costs on this CPU, by bucket,
    at each history length Tk in `ctxs`. The callback's time is excluded. gap_us > 0 makes
    every stand-in GEMV busy-wait that long, as the fabric would: the worker threads then
    idle between the parallel regions exactly as on the board (with gap_us > spin_us they go
    to sleep and each region pays their wake-up). Returns a dict."""
    lib = lib or load_lib()
    c = HostCfg(vocab=vocab, yarn_factor=4.0, yarn_orig_ctx=16384, **SHAPES[shape])
    rng = np.random.RandomState(seed)
    gains = dict(attn_norm=rng.rand(c.n_layer, c.d) + 0.5, q_norm=rng.rand(c.n_layer, c.head_dim) + 0.5,
                 k_norm=rng.rand(c.n_layer, c.head_dim) + 0.5, ffn_norm=rng.rand(c.n_layer, c.d) + 0.5,
                 out_norm=rng.rand(c.d) + 0.5)
    signs = rng.randint(0, 256, (vocab, c.d // 8)).astype(np.uint8)
    scales = (rng.rand(vocab, c.d // QK) * 0.08 + 0.01).astype(np.float16)
    top = max(ctxs)
    host = BkHost(c, gains, emb_from_split(signs, scales), build_tables(c, top + 1), lib=lib,
                  threads=threads, kernels=kernels, spin_us=spin_us)
    gap = c_int32(int(gap_us))
    host.set_c_gemv(lib.bk_bench_gemv, ctypes.cast(ctypes.pointer(gap), c_void_p) if gap_us else None)
    if verbose:
        print(f"  {shape}: threads {host.threads}, kernels {host.kernels} (simd {SIMD_NAMES[lib.bk_simd()]}), "
              f"spin_us {'default' if spin_us is None else spin_us}, gap_us {gap_us}", flush=True)
    rows, t = [], 0
    for Tk in sorted(ctxs):
        while host.pos < Tk - 1:                # history: Tk-1 cached positions
            host.step(t % vocab, want_logits=False)
            t += 1
        host.step(t % vocab, want_logits=True, copy=False)      # warm-up at this length
        runs = []
        for _ in range(reps):                   # the same position again: history exactly Tk
            host.truncate(Tk - 1)
            host.reset_stats()
            host.step(t % vocab, want_logits=True, copy=False)
            st = host.stats()
            runs.append(dict(st["ns"], host=st["host_ns"], cpu=st["host_cpu_ns"]))
        t += 1
        med = {k: float(np.median([r[k] for r in runs])) / 1e6 for k in runs[0]}        # ms, median of reps
        r = dict(Tk=Tk, host_ms=med["host"], host_ms_min=min(r["host"] for r in runs) / 1e6, gemv_ms=med["gemv"],
                 host_cpu_ms=med["cpu"], **{k + "_ms": med[k] for k in HOST_BUCKETS})
        rows.append(r)
        if verbose:
            print(f"  Tk {r['Tk']:5d}: host {r['host_ms']:8.3f} ms/token  " +
                  " ".join(f"{k} {r[k + '_ms']:.3f}" for k in HOST_BUCKETS), flush=True)
    out = dict(shape=shape, cfg=c.asdict(), machine=platform.machine(), lib=Path(lib._path).name, rows=rows,
               threads=host.threads, kernels=host.kernels, simd=SIMD_NAMES[lib.bk_simd()],
               spin_us=spin_us, gap_us=gap_us)
    try:
        out["loadavg"] = os.getloadavg()[0]
    except OSError:
        pass
    host.close()
    if len(rows) >= 2:                          # attention = a + b * Tk, least squares
        tk = np.array([r["Tk"] for r in rows])
        for k in ("att_score", "att_softmax", "att_ctx"):
            b, a = np.polyfit(tk, np.array([r[k + "_ms"] for r in rows]), 1)
            out[k + "_fit"] = dict(ms_fixed=float(a), us_per_pos=float(b * 1e3))
        fix = [r["host_ms"] - r["att_score_ms"] - r["att_softmax_ms"] - r["att_ctx_ms"] for r in rows]
        out["non_attention_ms"] = float(np.median(fix))
        out["attention_us_per_pos"] = sum(out[k + "_fit"]["us_per_pos"] for k in ("att_score", "att_softmax", "att_ctx"))
        b, a = np.polyfit(tk, np.array([r["host_ms"] for r in rows]), 1)
        out["host_fit"] = dict(ms_fixed=float(a), us_per_pos=float(b * 1e3))
        if verbose:
            print(f"  -> non-attention {out['non_attention_ms']:.3f} ms/token + attention "
                  f"{out['attention_us_per_pos']:.2f} us per cached position per token; "
                  f"host = {a:.2f} ms + {b * 1e3:.1f} us * Tk (least squares); "
                  f"load average {out.get('loadavg', float('nan')):.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="build the library and print its path")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--shape", default="8b", choices=sorted(SHAPES))
    ap.add_argument("--ctx", default="20,64,256")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--replay", help="a save_replay .npz: numpy-only bit-identity check")
    ap.add_argument("--threads-check", metavar="REPLAY", help="the replay with 1/3/4/2 threads x both kernel sets, "
                    "all taps compared (numpy only)")
    ap.add_argument("--kern-check", action="store_true", help="fast kernels vs reference kernels on adversarial "
                    "inputs (numpy only)")
    ap.add_argument("--tables", help="write the tables of --gguf to this .npz")
    ap.add_argument("--gguf")
    ap.add_argument("--max-pos", type=int, default=4096)
    ap.add_argument("--torch", action="store_true", help="--tables: take them from the spec (needs torch)")
    ap.add_argument("--out", help="--bench: write the result as JSON")
    ap.add_argument("--threads", default=None, help="--bench: thread counts, e.g. 1,4 (default: the library's)")
    ap.add_argument("--kernels", default=None, choices=("fast", "ref"))
    ap.add_argument("--spin-us", type=int, default=None)
    ap.add_argument("--gap-us", type=int, default=0, help="--bench: stand-in GEMV busy-waits this long")
    a = ap.parse_args()
    lib = load_lib()
    print(f"bk_host {lib._path}\n  built: {' '.join(build_cmd(Path(lib._path).name))}")
    ok = True
    if a.selftest or a.build:
        st = lib.bk_selftest()
        print(f"  selftest {'ok' if st == 0 else f'FAILED (mask {st:#x})'} on {platform.machine()}")
        ok &= st == 0
    if a.tables:
        from ..ref.gguf_q1 import GGUF
        cfg = cfg_from_gguf(GGUF(a.gguf))
        t = build_tables_torch(cfg, a.max_pos) if a.torch else build_tables(cfg, a.max_pos)
        print(f"  tables ({t['source']}) digest {save_tables(a.tables, t)} -> {a.tables}")
    if a.replay:
        ok &= replay(a.replay, lib, kernels=a.kernels)[0]
    if a.kern_check:
        ok &= kern_check(lib)[0]
    if a.threads_check:
        ok &= threads_check(a.threads_check, lib)[0]
    if a.bench:
        res = []
        for th in (a.threads.split(",") if a.threads else [None]):
            res.append(bench(a.shape, tuple(int(x) for x in a.ctx.split(",")), a.reps, lib,
                             threads=int(th) if th else None, kernels=a.kernels, spin_us=a.spin_us,
                             gap_us=a.gap_us))
        if a.out:
            Path(a.out).write_text(json.dumps(res if len(res) > 1 else res[0], indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
