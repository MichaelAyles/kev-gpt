"""Minimal GGUF reader for PrismML Bonsai Q1_0 models (doc 10, R1).

Parses the GGUF container (v2/v3) without the `gguf` package, which doesn't know
PrismML's custom 1-bit type. Exposes metadata, tensor infos, and raw tensor bytes;
`dequant_q1` turns a Q1_0 tensor into float32 under the PINNED block layout:

    block (128 weights) = fp16 d  ||  16 bytes of sign bits
    weight j of the block: bit (j % 8) of byte (j // 8); bit 1 -> +d, bit 0 -> -d

That layout is a hypothesis until `check_q1_layout` confirms it bit-exact against the
unpacked (bf16) release of the same model.
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
        d, qs = self.q1_parts(name)
        return dequant_q1(d, qs).reshape(rows, ti.shape[0])


def unpack_signs(qs: np.ndarray) -> np.ndarray:
    """[nblk,16] uint8 -> [nblk,128] in {0,1}, weight j = bit (j%8) of byte (j//8)."""
    return np.unpackbits(qs, axis=1, bitorder="little")


def dequant_q1(d: np.ndarray, qs: np.ndarray) -> np.ndarray:
    s = unpack_signs(qs).astype(np.float32) * 2.0 - 1.0
    return (s * d.astype(np.float32)[:, None]).reshape(-1)


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
