"""pack_weights — Q1 (1-bit) matrices -> the DDR image the streaming GEMV consumes (doc 10, R2a).

The image is what `rtl/gemv_stream.v` eats, laid out the way R0 measured the bus fastest:
NPORTS 128-bit AXI read ports walking NEIGHBOURING addresses with disjoint data
(port-interleaved stripes). This docstring is the normative definition of the format.

1. Beats
   A beat is 128 bits = 16 bytes, little-endian on the bus: bit b of the beat is bit (b % 8)
   of byte (b // 8), and byte i of a beat sits at address (beat address + i) -- i.e. exactly
   AXI RDATA[8i+7:8i] of a 16-byte-wide read.
     SIGN beat   one 128-group of one output row: the 16 Q1_0 sign bytes verbatim, so beat bit j
                 is the sign of weight j of the group (1 = +scale, 0 = -scale).
     SCALE beat  eight fp16 group scales, raw IEEE bits: the scale of slot i is beat bits
                 [16i+15 : 16i] (bytes 2i, 2i+1). Unused slots are zero.
     PAD beat    all zero.

2. Logical beat order of one matrix [R rows, G = in/128 groups per row]
   Row-major. Each row is
       for c in 0 .. ceil(G/8)-1:
           SCALE beat   scales of groups 8c .. 8c+7 of this row   (slot i = group 8c+i)
           SIGN beats   groups 8c .. min(8c+7, G-1), ascending
       PAD beats up to a multiple of NPORTS
   so a scale always arrives immediately before the first sign beat that needs it (the engine
   holds one scale beat, nothing more), and every row starts on a tuple boundary (see 3).
       RB  = G + ceil(G/8)            beats in a row before padding
       TPR = ceil(RB / NPORTS)        tuples per row
   For every G that is a multiple of 8 and NPORTS in {1, 3}, RB = 9G/8 is a multiple of NPORTS
   and there is NO row padding: 9 beats carry 1024 weights = exactly 1.125 bits/weight. (That
   covers every Bonsai matrix: G = 32 / 96 on the 8B, 16 / 48 on the 1.7B. NPORTS = 2 or 4 is
   also pad-free on the 8B.)

3. Dealing beats to ports (tuples)
   Logical beat i goes to port (i % NPORTS) as that port's beat number (i // NPORTS). A TUPLE is
   the NPORTS consecutive logical beats t*NPORTS .. t*NPORTS+NPORTS-1, one per port; the engine
   pops one whole tuple per cycle, so all ports carry exactly the same number of beats and none
   can run ahead. Matrices follow each other in stream order; matrix m starts at tuple T0[m].
     align = "tuple"  (default) T0[m+1] = T0[m] + R*TPR: no padding between matrices, the ports
                      stream continuously through the whole model.
     align = "stripe" every matrix is padded with PAD tuples to a multiple of CHUNK tuples, so
                      each matrix starts at the top of a burst on every port (the engine is told
                      the pad count and discards it: `cfg_pad`).
   The whole image is padded to a whole number of stripes.

4. Address map (R0's port-interleaved layout)
   CHUNK = beats per AXI burst = a power of two in 1..256 (default 256 = 4 KB). A STRIPE is
   SLOTS chunks of CHUNK beats; port k owns slot PORT_SLOTS[k] of every stripe. Tuple t, port k
   lives at byte offset
       ((t // CHUNK) * SLOTS + PORT_SLOTS[k]) * CHUNK*16  +  (t % CHUNK) * 16
   from the image base. So port k's read master issues INCR bursts of CHUNK beats at
       base + PORT_SLOTS[k]*CHUNK*16 + n * SLOTS*CHUNK*16,   n = 0, 1, 2, ...
   -- `axi_rd_probe` with BURST = CHUNK, STEP = SLOTS*CHUNK*16. The image base must be
   CHUNK*16-byte aligned (4 KB aligned covers every CHUNK); then every burst is CHUNK*16 bytes
   at a CHUNK*16-aligned address and can never cross a 4 KB boundary.
   Default SLOTS = NPORTS, PORT_SLOTS = (0..NPORTS-1): dense, no holes. NOTE (honesty): R0's
   MEASURED 15.54 GB/s on HP0+HP1+HP3 used SLOTS = 4 with PORT_SLOTS = (0, 1, 3) -- HP2's slot was
   left unread. `--slots 4 --port-slots 0,1,3` reproduces that address pattern exactly (the
   image is 4/3 the size; the bytes READ per token are unchanged). The dense default is the
   same idea with tighter addresses but has not itself been timed on silicon.

Bytes per token is exact: logical beats * 16. Overhead is measured against the Q1_0 payload
(18 bytes per 128 weights = 1.125 bpw).

    python -m fabric.big.pack_weights data/big/Bonsai-8B-Q1_0.gguf            # stream report
    python -m fabric.big.pack_weights data/big/Bonsai-1.7B-Q1_0.gguf --names l0.q l0.down \
        -o /tmp/kevbuild/gemv_stream/l0.bin --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BEAT = 16                       # bytes per 128-bit beat
QK = 128                        # weights per group
Q1_BLOCK_BYTES = 18             # fp16 scale + 16 sign bytes
STREAM_ORDER = ("q", "k", "v", "o", "gate", "up", "down")
GG = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output",
      "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}


# ----------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class Layout:
    nports: int = 3
    chunk: int = 256                         # beats per burst (power of two, 1..256)
    slots: int = 0                           # chunks per stripe; 0 -> nports (dense)
    port_slots: tuple = ()                   # slot of each port; () -> 0..nports-1
    align: str = "tuple"                     # "tuple" | "stripe" (per-matrix alignment)

    def __post_init__(self):
        if self.slots == 0:
            object.__setattr__(self, "slots", self.nports)
        if not self.port_slots:
            object.__setattr__(self, "port_slots", tuple(range(self.nports)))
        object.__setattr__(self, "port_slots", tuple(int(s) for s in self.port_slots))
        assert 1 <= self.nports <= 8, "NPORTS 1..8 (at most one scale beat per tuple)"
        assert 1 <= self.chunk <= 256 and self.chunk & (self.chunk - 1) == 0, "chunk: power of two 1..256"
        assert len(self.port_slots) == self.nports and len(set(self.port_slots)) == self.nports
        assert all(0 <= s < self.slots for s in self.port_slots)
        assert self.align in ("tuple", "stripe")

    @property
    def chunk_bytes(self) -> int:
        return self.chunk * BEAT

    @property
    def stripe_bytes(self) -> int:
        return self.slots * self.chunk * BEAT

    def beat_offset(self, t: int, k: int) -> int:
        """byte offset in the image of tuple t, port k (section 4 of the module doc)."""
        return ((t // self.chunk) * self.slots + self.port_slots[k]) * self.chunk * BEAT + (t % self.chunk) * BEAT


def row_beats(G: int) -> int:
    return G + (G + 7) // 8


def tuples_per_row(G: int, nports: int) -> int:
    return -(-row_beats(G) // nports)


def matrix_tuples(R: int, G: int, lay: Layout) -> tuple[int, int]:
    """(data tuples, pad tuples after them) for one matrix."""
    t = R * tuples_per_row(G, lay.nports)
    pad = (-t) % lay.chunk if lay.align == "stripe" else 0
    return t, pad


@dataclass
class Entry:
    name: str
    rows: int
    groups: int
    tuple0: int            # first tuple of the matrix in the stream
    tuples: int            # data tuples (rows * TPR)
    pad: int               # pad tuples after the data (stripe alignment)

    def as_dict(self):
        return dict(name=self.name, rows=self.rows, groups=self.groups, tuple0=self.tuple0,
                    tuples=self.tuples, pad=self.pad)


def plan(shapes, lay: Layout) -> tuple[list[Entry], int]:
    """shapes: iterable of (name, R, G) in stream order -> (entries, total tuples incl. the
    image's final pad to a whole stripe)."""
    out, t = [], 0
    for name, R, G in shapes:
        n, pad = matrix_tuples(R, G, lay)
        out.append(Entry(name, int(R), int(G), t, n, pad))
        t += n + pad
    return out, t + (-t) % lay.chunk


# ----------------------------------------------------------------------------- matrices in/out
def _as_numpy(W):
    """Q1Matrix | (signs, scales) -> (signs uint8 [R, G*16], scales uint16 [R, G] raw fp16 bits)."""
    signs, scales = (W.signs, W.scales) if hasattr(W, "signs") else W
    if hasattr(signs, "detach"):
        signs = signs.detach().cpu().numpy()
    if hasattr(scales, "detach"):
        scales = scales.detach().cpu().numpy()
    signs = np.ascontiguousarray(signs, dtype=np.uint8)
    scales = np.ascontiguousarray(scales)
    if scales.dtype == np.float16:
        scales = scales.view(np.uint16)
    assert scales.dtype == np.uint16, scales.dtype
    R, nb = signs.shape
    assert nb % 16 == 0 and scales.shape == (R, nb // 16), (signs.shape, scales.shape)
    return signs, scales


def _beat_index(G: int):
    """positions within a row of (scale beats [ceil(G/8)], sign beats [G])."""
    g = np.arange(G)
    return np.arange((G + 7) // 8) * 9, g + g // 8 + 1


def matrix_to_tuples(W, nports: int) -> np.ndarray:
    """one matrix -> its tuples, uint8 [R*TPR, nports, 16] (sections 1-3 of the module doc)."""
    signs, scales = _as_numpy(W)
    R, G = scales.shape
    tpr = tuples_per_row(G, nports)
    rows = np.zeros((R, tpr * nports, BEAT), dtype=np.uint8)
    sc_pos, sg_pos = _beat_index(G)
    sc = np.zeros((R, len(sc_pos) * 8), dtype="<u2")
    sc[:, :G] = scales
    rows[:, sc_pos] = sc.view(np.uint8).reshape(R, len(sc_pos), BEAT)
    rows[:, sg_pos] = signs.reshape(R, G, BEAT)
    return rows.reshape(R * tpr, nports, BEAT)


def tuples_to_matrix(tup: np.ndarray, R: int, G: int, strict: bool = True):
    """inverse of matrix_to_tuples -> (signs uint8 [R, G*16], scales uint16 [R, G])."""
    nports = tup.shape[1]
    tpr = tuples_per_row(G, nports)
    rows = np.ascontiguousarray(tup).reshape(R, tpr * nports, BEAT)
    sc_pos, sg_pos = _beat_index(G)
    sc = np.ascontiguousarray(rows[:, sc_pos]).reshape(R, -1).view("<u2")
    if strict:
        used = np.zeros(tpr * nports, dtype=bool)
        used[sc_pos] = used[sg_pos] = True
        assert not rows[:, ~used].any(), "non-zero row padding"
        assert not sc[:, G:].any(), "non-zero unused scale slot"
    signs = np.ascontiguousarray(rows[:, sg_pos]).reshape(R, G * BEAT)
    return signs, sc[:, :G].astype(np.uint16)


# ----------------------------------------------------------------------------- image
def tuples_to_image(tup: np.ndarray, lay: Layout) -> np.ndarray:
    """stream of tuples [T, nports, 16] -> image bytes (padded to whole stripes)."""
    T = tup.shape[0]
    S = -(-T // lay.chunk)
    full = np.zeros((S * lay.chunk, lay.nports, BEAT), dtype=np.uint8)
    full[:T] = tup
    img = np.zeros((S, lay.slots, lay.chunk, BEAT), dtype=np.uint8)
    img[:, list(lay.port_slots)] = full.reshape(S, lay.chunk, lay.nports, BEAT).transpose(0, 2, 1, 3)
    return img.reshape(-1)


def port_streams(image: np.ndarray, lay: Layout) -> np.ndarray:
    """What each port's read master delivers, straight from the address map: for port k,
    bursts of CHUNK beats at PORT_SLOTS[k]*CHUNK*16 + n*STRIPE. -> uint8 [nports, beats, 16].
    Written from the byte-offset formula (not as the inverse of tuples_to_image) so it is an
    independent check of section 4."""
    image = np.asarray(image, dtype=np.uint8).reshape(-1)
    assert image.size % lay.stripe_bytes == 0, "image is not a whole number of stripes"
    S = image.size // lay.stripe_bytes
    out = np.empty((lay.nports, S * lay.chunk, BEAT), dtype=np.uint8)
    for k in range(lay.nports):
        for n in range(S):
            a = lay.port_slots[k] * lay.chunk_bytes + n * lay.stripe_bytes
            out[k, n * lay.chunk:(n + 1) * lay.chunk] = image[a:a + lay.chunk_bytes].reshape(lay.chunk, BEAT)
    return out


def image_to_tuples(image: np.ndarray, lay: Layout) -> np.ndarray:
    return port_streams(image, lay).transpose(1, 0, 2)          # [T, nports, 16]


def pack(mats, lay: Layout = Layout()):
    """mats: list of (name, W) in stream order, W a Q1Matrix or (signs, scales).
    -> (image uint8 [bytes], index list[Entry])."""
    mats = [(n, _as_numpy(W)) for n, W in mats]
    index, total = plan([(n, s.shape[0], sc.shape[1]) for n, (s, sc) in mats], lay)
    tup = np.zeros((total, lay.nports, BEAT), dtype=np.uint8)
    for e, (_, W) in zip(index, mats):
        tup[e.tuple0:e.tuple0 + e.tuples] = matrix_to_tuples(W, lay.nports)
    return tuples_to_image(tup, lay), index


def unpack(image, index, lay: Layout = Layout(), strict: bool = True):
    """inverse of pack -> list of (name, signs uint8 [R, in/8], scales uint16 [R, G])."""
    tup = image_to_tuples(image, lay)
    out, end = [], 0
    for e in index:
        if isinstance(e, dict):
            e = Entry(**e)
        if strict:
            assert not tup[end:e.tuple0].any(), "non-zero bytes between matrices"
        s, sc = tuples_to_matrix(tup[e.tuple0:e.tuple0 + e.tuples], e.rows, e.groups, strict)
        end = e.tuple0 + e.tuples
        out.append((e.name, s, sc))
    if strict:
        assert not tup[end:].any(), "non-zero trailing padding"
    return out


def roundtrip_ok(mats, lay: Layout) -> bool:
    """unpack(pack(W)) == W bit-for-bit, and the image obeys the address-map contract."""
    img, index = pack(mats, lay)
    if img.size % lay.stripe_bytes:
        return False
    back = unpack(img, index, lay)
    for (n, W), (n2, s2, sc2) in zip(mats, back):
        s, sc = _as_numpy(W)
        if n != n2 or not np.array_equal(s, s2) or not np.array_equal(sc, sc2):
            return False
    # spot-check the per-beat offset formula against the image
    tup = image_to_tuples(img, lay)
    rng = np.random.default_rng(1)
    for t in rng.integers(0, tup.shape[0], size=min(64, tup.shape[0])):
        for k in range(lay.nports):
            o = lay.beat_offset(int(t), k)
            if not np.array_equal(img[o:o + BEAT], tup[t, k]):
                return False
    return True


# ----------------------------------------------------------------------------- GGUF side
def gguf_name(short: str, tied: bool) -> str:
    if short == "lm_head":
        return "token_embd.weight" if tied else "output.weight"
    if short == "embed":
        return "token_embd.weight"
    if short.startswith("l") and "." in short:
        L, k = short[1:].split(".", 1)
        if k in GG:
            return f"blk.{int(L)}.{GG[k]}.weight"
    return short


def gguf_shape(g, name):
    ti = g.tensors[name]
    assert g.is_q1(name), f"{name} is not Q1_0"
    inn = int(ti.shape[0])
    assert inn % QK == 0
    return ti.n // inn, inn // QK                    # (rows, groups)


def gguf_matrix(g, name):
    """(signs uint8 [R, in/8], scales uint16 [R, G]) straight from the memmap (no torch)."""
    R, G = gguf_shape(g, name)
    d, qs = g.q1_parts(name)
    return (np.ascontiguousarray(qs).reshape(R, G * 16),
            np.ascontiguousarray(d).view(np.uint16).reshape(R, G))


def stream_shapes(g):
    """The full per-token stream order: every layer q,k,v,o,gate,up,down, then lm_head."""
    a = g.meta["general.architecture"]
    tied = "output.weight" not in g.tensors
    out = []
    for L in range(int(g.meta[f"{a}.block_count"])):
        for k in STREAM_ORDER:
            out.append((f"l{L}.{k}",) + gguf_shape(g, gguf_name(f"l{L}.{k}", tied)))
    out.append(("lm_head",) + gguf_shape(g, gguf_name("lm_head", tied)))
    return out


def report(shapes, lay: Layout) -> dict:
    index, total = plan(shapes, lay)
    weights = sum(e.rows * e.groups * QK for e in index)
    payload = sum(e.rows * e.groups for e in index) * Q1_BLOCK_BYTES
    data = sum(e.tuples for e in index) * lay.nports * BEAT
    mpad = sum(e.pad for e in index) * lay.nports * BEAT
    read = total * lay.nports * BEAT                       # bytes the ports actually read
    image = total // lay.chunk * lay.stripe_bytes          # bytes the image occupies in DDR
    return dict(nports=lay.nports, chunk=lay.chunk, slots=lay.slots, port_slots=list(lay.port_slots),
                align=lay.align, matrices=len(index), weights=weights, payload_bytes=payload,
                data_bytes=data, row_pad_bytes=data - payload, matrix_pad_bytes=mpad,
                tail_pad_bytes=read - data - mpad, read_bytes=read, image_bytes=image,
                bpw_read=8 * read / weights, overhead_pct=100 * (read - payload) / payload)


def _fmt(r: dict, tag: str) -> str:
    return (f"PACK_STREAM {tag} nports={r['nports']} chunk={r['chunk']} align={r['align']} "
            f"slots={r['slots']} matrices={r['matrices']} weights={r['weights']} "
            f"read_bytes={r['read_bytes']} ({r['read_bytes'] / 1e6:.3f} MB) "
            f"payload_1.125bpw={r['payload_bytes']} overhead={r['read_bytes'] - r['payload_bytes']} B "
            f"({r['overhead_pct']:.5f}%) bpw={r['bpw_read']:.6f} image_bytes={r['image_bytes']}")


def selftest(extra=(), lay: Layout = Layout()) -> bool:
    rng = np.random.default_rng(7)
    ok = True
    n = 0
    for nports in (1, 2, 3, 4):
        for chunk in (1, 4, 16, 256):
            for align in ("tuple", "stripe"):
                for slots, ps in ((0, ()), (nports + 1, tuple(range(nports - 1)) + (nports,))):
                    L = Layout(nports, chunk, slots, ps, align)
                    mats = []
                    for i, (R, G) in enumerate(((3, 1), (5, 7), (2, 8), (4, 9), (7, 32), (3, 96), (1, 17))):
                        mats.append((f"m{i}", (rng.integers(0, 256, (R, G * 16), dtype=np.uint8),
                                               rng.integers(0, 65536, (R, G), dtype=np.uint16))))
                    ok &= roundtrip_ok(mats, L)
                    n += 1
    if extra:
        ok &= roundtrip_ok(list(extra), lay)
        n += 1
    print(f"PACK_ROUNDTRIP {'PASS' if ok else 'FAIL'} ({n} layouts, unpack(pack(W)) == W bit-for-bit)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("--names", nargs="*", default=[], help="matrices to pack: l<L>.<q|k|v|o|gate|up|down>, lm_head, or a GGUF tensor name")
    ap.add_argument("-o", "--out", help="write the image here (plus <out>.json index)")
    ap.add_argument("--nports", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=256, help="beats per burst (power of two, 1..256)")
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--port-slots", default="")
    ap.add_argument("--align", default="tuple", choices=("tuple", "stripe"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--all", action="store_true", help="pack the whole stream order (writes ~1 GB for the 8B)")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fabric.big.ref.gguf_q1 import GGUF
    g = GGUF(a.gguf)
    tied = "output.weight" not in g.tensors
    ps = tuple(int(x) for x in a.port_slots.split(",")) if a.port_slots else ()
    lay = Layout(a.nports, a.chunk, a.slots, ps, a.align)

    shapes = stream_shapes(g)
    for tag, L in (("full_stream", lay),
                   ("full_stream_stripe_aligned", Layout(a.nports, a.chunk, a.slots, ps, "stripe"))):
        print(_fmt(report(shapes, L), tag))
    r = report(shapes, lay)
    print(f"PACK_BYTES_PER_TOKEN {r['read_bytes']} B = {r['read_bytes'] / 1e6:.3f} MB "
          f"(DERIVED from tensor shapes; weights only, no KV, embedding row excluded"
          f"{', lm_head = tied embedding' if tied else ''})")

    names = [n for n, _, _ in shapes] if a.all else a.names
    ok = True
    mats = [(n, gguf_matrix(g, gguf_name(n, tied))) for n in names]
    if mats:
        img, index = pack(mats, lay)
        print(_fmt(report([(n, s.shape[0], sc.shape[1]) for n, (s, sc) in mats], lay), "packed"))
        for e in index:
            print(f"  {e.name:12s} rows={e.rows} groups={e.groups} tuple0={e.tuple0} tuples={e.tuples} "
                  f"pad={e.pad} offset={lay.beat_offset(e.tuple0, 0):#x}")
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            img.tofile(a.out)
            Path(a.out + ".json").write_text(json.dumps(dict(
                layout=dict(nports=lay.nports, chunk=lay.chunk, slots=lay.slots,
                            port_slots=list(lay.port_slots), align=lay.align),
                index=[e.as_dict() for e in index]), indent=1))
            print(f"wrote {a.out} ({img.size} bytes) + .json")
    if a.selftest:
        ok = selftest(mats, lay)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
