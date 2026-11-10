"""Minimal GGUF reader for PrismML Bonsai Q1_0 models (doc 10, R1).

Parses the GGUF container (v2/v3) without the `gguf` package, which doesn't know
PrismML's custom 1-bit type. Exposes metadata, tensor infos, and raw tensor bytes;
`dequant_q1` turns a Q1_0 tensor into float32 under the PINNED block layout:

    block (128 weights) = fp16 d  ||  16 bytes of sign bits
    weight j of the block: bit (j % 8) of byte (j // 8); bit 1 -> +d, bit 0 -> -d

That layout is a hypothesis until `check_q1_layout` confirms it bit-exact against the
unpacked (bf16) release of the same model.

Ternary-Bonsai (R1d) uses PrismML's 2-bit type, `dequant_q2`, PINNED the same way
(`r1d_ternary --pin`, against the fp16 unpacked release: zeros and signs agree 100%, every
value equals sign * the group's max |unpacked| bit-exactly; ~1 group in 10^4 of the unpacked
release carries two magnitudes one bf16 ulp apart, which one scale per group flattens):

    block (G weights) = fp16 d  ||  G/4 bytes of 2-bit codes
    weight j of the block: code q = bits [2*(j%4) +: 2] of byte (j // 4); w = (q - 1) * d
    (q = 0/1/2 -> -d/0/+d; q = 3 would be +2d, reserved and never used by a ternary model)

    ggml type 42  (Q2_0):  G = 128, 34 B/block, 2.125 bpw in the released `*-Q2_0.gguf`;
                           G = 64, 18 B/block, 2.25 bpw in `*-Q2_0_g64.gguf` (what type 42 means
                           in PrismML's current llama.cpp fork). G is read off the tensor size.
    ggml type 142 (PQ2_0): G = 128, the fork's private id for the same g128 blocks.
    (Only type 42 / G = 128 has been pinned on a file; G = 64 and type 142 follow the fork's
    ggml-quants.c and are untested here.)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"
# value types
_T = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
      10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
# standard ggml types we may meet alongside Q1_0
GGML_F32, GGML_F16, GGML_BF16 = 0, 1, 30
QK = 128
Q1_BLOCK_BYTES = 2 + QK // 8   # 18 -> 1.125 bpw
GGML_Q1_0, GGML_Q2_0, GGML_PQ2_0 = 41, 42, 142
Q2_TYPES = (GGML_Q2_0, GGML_PQ2_0)


@dataclass
class TensorInfo:
    name: str
    shape: tuple          # ggml order: ne[0] is the fastest (row length)
    ggml_type: int
    offset: int           # relative to data start
    nbytes: int = 0

    @property
    def n(self) -> int:
        return int(np.prod(self.shape))


class GGUF:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        self.pos = 0
        assert bytes(self.mm[:4]) == GGUF_MAGIC, "not a GGUF file"
        self.pos = 4
        self.version = self._u32()
        n_tensors = self._u64()
        n_kv = self._u64()
        self.meta = {}
        for _ in range(n_kv):
            k = self._str()
            self.meta[k] = self._val(self._u32())
        self.tensors: dict[str, TensorInfo] = {}
        order = []
        for _ in range(n_tensors):
            name = self._str()
            nd = self._u32()
            shape = tuple(self._u64() for _ in range(nd))
            t = self._u32()
            off = self._u64()
            ti = TensorInfo(name, shape, t, off)
            self.tensors[name] = ti
            order.append(ti)
        align = int(self.meta.get("general.alignment", 32))
        self.data_start = (self.pos + align - 1) // align * align
        # byte sizes from offset deltas (robust to unknown types)
        order.sort(key=lambda x: x.offset)
        end = len(self.mm) - self.data_start
        for a, b in zip(order, order[1:] + [None]):
            a.nbytes = (b.offset if b else end) - a.offset

    # --- primitive readers ---
    def _rd(self, fmt):
        n = struct.calcsize(fmt)
        v = struct.unpack(fmt, bytes(self.mm[self.pos:self.pos + n]))[0]
        self.pos += n
        return v

    def _u32(self):
        return self._rd("<I")

    def _u64(self):
        return self._rd("<Q")

    def _str(self):
        n = self._u64()
        s = bytes(self.mm[self.pos:self.pos + n]).decode("utf-8", "replace")
        self.pos += n
        return s

    def _val(self, t):
        if t in _T:
            return self._rd(_T[t])
        if t == STRING:
            return self._str()
        if t == ARRAY:
            et = self._u32()
            n = self._u64()
            if et == STRING:
                return [self._str() for _ in range(n)]
            if et in _T:
                fmt = _T[et]
                sz = struct.calcsize(fmt)
                arr = np.frombuffer(bytes(self.mm[self.pos:self.pos + n * sz]), dtype=np.dtype(fmt))
                self.pos += n * sz
                return arr
            return [self._val(et) for _ in range(n)]
        raise ValueError(f"unknown gguf value type {t}")

    # --- tensors ---
    def raw(self, name: str) -> np.ndarray:
        ti = self.tensors[name]
        a = self.data_start + ti.offset
        return self.mm[a:a + ti.nbytes]

    def type_hist(self):
        h = {}
        for ti in self.tensors.values():
            h[ti.ggml_type] = h.get(ti.ggml_type, 0) + 1
        return h

    def is_q1(self, name: str) -> bool:
        ti = self.tensors[name]
        return ti.n % QK == 0 and ti.nbytes >= ti.n // QK * Q1_BLOCK_BYTES and \
            ti.nbytes < ti.n // QK * Q1_BLOCK_BYTES + 64

    def q1_parts(self, name: str):
        """(scales fp16 [nblk], signs uint8 [nblk,16]) for a Q1_0 tensor, PyTorch row-major view:
        rows = shape[1:] flattened, row length = shape[0]."""
        ti = self.tensors[name]
        nblk = ti.n // QK
        b = np.asarray(self.raw(name)[:nblk * Q1_BLOCK_BYTES]).reshape(nblk, Q1_BLOCK_BYTES)
        d = b[:, :2].copy().view(np.float16).reshape(nblk)
        qs = b[:, 2:]
        return d, qs

    def q2_group(self, name: str) -> int:
        """Group size G (128 or 64) of a 2-bit ternary tensor, 0 if it is not one. The type id
        says "2-bit codes"; G comes from the byte size (G/4 + 2 bytes per G weights)."""
        ti = self.tensors[name]
        if ti.ggml_type not in Q2_TYPES:
            return 0
        for grp in (128, 64):
            want = ti.n // grp * (2 + grp // 4)
            if ti.n % grp == 0 and ti.shape[0] % grp == 0 and want <= ti.nbytes < want + 64:
                return grp
        raise ValueError(f"{name}: type {ti.ggml_type} but {ti.nbytes} bytes fits no Q2_0 group size")

    def q2_parts(self, name: str):
        """(scales fp16 [nblk], codes uint8 [nblk, G/4], G) for a Q2_0 / PQ2_0 tensor."""
        ti, grp = self.tensors[name], self.q2_group(name)
        nblk, bb = ti.n // grp, 2 + grp // 4
        b = np.asarray(self.raw(name)[:nblk * bb]).reshape(nblk, bb)
        return b[:, :2].copy().view(np.float16).reshape(nblk), b[:, 2:], grp

    def dequant_rows(self, name: str, r0: int, r1: int) -> np.ndarray:
        """float32 rows [r0, r1) of a Q1_0 / Q2_0 matrix (PyTorch orientation) without touching
        the rest of the tensor (layout pinning, spot checks)."""
        ti = self.tensors[name]
        inn, grp = ti.shape[0], self.q2_group(name)
        if grp:
            d, qs, _ = self.q2_parts(name)
            sl = slice(r0 * (inn // grp), r1 * (inn // grp))
            return dequant_q2(d[sl], np.ascontiguousarray(qs[sl])).reshape(-1, inn)
        assert self.is_q1(name), f"{name}: not a Q1_0/Q2_0 tensor"
        d, qs = self.q1_parts(name)
        sl = slice(r0 * (inn // QK), r1 * (inn // QK))
        return dequant_q1(d[sl], np.ascontiguousarray(qs[sl])).reshape(-1, inn)

    def dequant(self, name: str) -> np.ndarray:
        """float32 tensor in PyTorch orientation [out, in] (ggml ne[0] = in = row length)."""
        ti = self.tensors[name]
        rows = ti.n // ti.shape[0]
        if ti.ggml_type == GGML_F32:
            return np.asarray(self.raw(name)).view(np.float32).reshape(rows, ti.shape[0]).copy()
        if ti.ggml_type == GGML_F16:
            return np.asarray(self.raw(name)).view(np.float16).astype(np.float32).reshape(rows, ti.shape[0])
        if ti.ggml_type == GGML_BF16:
            u = np.asarray(self.raw(name)).view(np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(rows, ti.shape[0])
        if ti.ggml_type in Q2_TYPES:
            d, qs, _ = self.q2_parts(name)
            return dequant_q2(d, qs).reshape(rows, ti.shape[0])
        d, qs = self.q1_parts(name)
        return dequant_q1(d, qs).reshape(rows, ti.shape[0])


def unpack_signs(qs: np.ndarray) -> np.ndarray:
    """[nblk,16] uint8 -> [nblk,128] in {0,1}, weight j = bit (j%8) of byte (j//8)."""
    return np.unpackbits(qs, axis=1, bitorder="little")


def dequant_q1(d: np.ndarray, qs: np.ndarray) -> np.ndarray:
    s = unpack_signs(qs).astype(np.float32) * 2.0 - 1.0
    return (s * d.astype(np.float32)[:, None]).reshape(-1)


def unpack_codes2(qs: np.ndarray) -> np.ndarray:
    """[nblk, G/4] uint8 -> [nblk, G] 2-bit codes in 0..3, weight j = bits [2*(j%4) +: 2] of
    byte (j//4)."""
    return ((qs[:, :, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3).reshape(qs.shape[0], -1)


def dequant_q2(d: np.ndarray, qs: np.ndarray) -> np.ndarray:
    t = unpack_codes2(qs).astype(np.float32) - 1.0
    return (t * d.astype(np.float32)[:, None]).reshape(-1)


if __name__ == "__main__":
    import sys
    g = GGUF(sys.argv[1])
    print("version", g.version, "tensors", len(g.tensors), "types", g.type_hist())
    for k, v in g.meta.items():
        if not isinstance(v, (list, np.ndarray)) or len(v) < 8:
            print(f"  {k} = {v}")
        else:
            print(f"  {k} = <{type(v).__name__} len {len(v)}>")
    for ti in list(g.tensors.values())[:16]:
        print(f"  {ti.name:40s} {str(ti.shape):22s} type {ti.ggml_type:3d} bytes {ti.nbytes} "
              f"bpw {8 * ti.nbytes / ti.n:.3f}")
