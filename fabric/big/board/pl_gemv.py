"""R2a on silicon: stream Bonsai-8B's weights from DDR through `gemv_axi` and check every
result against the golden token (doc 10, R2a). Runs ON the Kria, as root.

  1. pack the selected matrices (pack_weights, stripe-aligned) into physically contiguous
     hugepage regions (physmem), one region per matrix; the pool lives in a hugetlbfs file so
     later runs re-use the image
  2. for each GEMV of the golden token: load the BFP10 activation, point the engine at the
     matrix, GO, read back the result rows and the counters
  3. compare: every result word with the golden `y` (bit-exact), the per-port beat counts and
     128-bit xor with the packed image, rows / tuples / RRESP errors
  4. report the stream rate (bytes / cycles x fclk) and, when all 253 GEMVs ran, the time one
     token's weight traffic took

    sudo fpgautil -b ~/kevbit/gemv_axi.bit.bin
    sudo python3 -m fabric.big.board.pl_gemv --fclk 333e6               # everything
    sudo python3 -m fabric.big.board.pl_gemv --layers 1 --no-head       # layer 0 only (quick)

fclk MUST be forced via /sys (fpgautil does not apply the BD's PL0 rate) and is verified.
Honesty: the A53 loads each activation and reads each result over AXI-Lite; that host time is
NOT in the stream numbers (in the real engine both stay on-chip). CYC is GO -> last result row
in PL clock cycles, measured by the PL itself.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import sys
import time
from pathlib import Path

import numpy as np

from .. import gemv_axi_regs as R
from ..pack_weights import Layout, gguf_matrix, gguf_name, pack
from ..ref.gguf_q1 import GGUF
from .physmem import HP, HugePool, evict, hp_settle

FCLK = "/sys/devices/platform/fclk0/set_rate"
KINDS = ("q", "k", "v", "o", "gate", "up", "down")


def set_fclk(hz: float) -> float:
    with open(FCLK, "w") as f:
        f.write(str(int(hz)))
    with open(FCLK) as f:
        got = float(f.read().strip())
    if abs(got - hz) / hz > 0.02:
        print(f"NOTE fclk0 asked {hz / 1e6:.1f} MHz, PLL gave {got / 1e6:.2f} MHz — using the read-back rate")
    return got


def pack10(xq: np.ndarray) -> np.ndarray:
    """int16 [G*128] BFP10 mantissas -> uint32 [G, 40]: element j of a group at bits [10j+9:10j]."""
    v = (xq.astype(np.int32) & 0x3FF).reshape(-1, 128)
    bits = ((v[:, :, None] >> np.arange(10)) & 1).astype(np.uint8).reshape(v.shape[0], 1280)
    return np.packbits(bits, axis=1, bitorder="little").view("<u4")


class Gemv:
    def __init__(self):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.m = mmap.mmap(self.fd, 1 << R.ADDR_BITS, offset=R.BASE_ADDR)
        self.w32 = (ctypes.c_uint32 * ((1 << R.ADDR_BITS) // 4)).from_buffer(self.m)
        idc = self.rd(R.IDCODE)
        if idc != R.IDCODE_VALUE:
            sys.exit(f"IDCODE {idc:08x} != GEMV — is gemv_axi.bit loaded?")
        p = self.rd(R.PARAMS)
        self.nports, self.gmax, self.res_words = p & 15, (p >> 8) & 255, 1 << ((p >> 16) & 255)
        self.wr(R.CTRL, 2)
        self.wr(R.CTRL, 0)
        self._act = None

    def wr(self, off, v):
        self.w32[off >> 2] = int(v) & 0xFFFFFFFF

    def rd(self, off):
        return int(self.w32[off >> 2])

    def rd64(self, lo):
        return self.rd(lo) | (self.rd(lo + 4) << 32)

    def load_act(self, key, xq, xe):
        if self._act == key:
            return 0.0
        t0 = time.perf_counter()
        words = pack10(xq)
        w32, base = self.w32, R.ACT_MANT >> 2
        for g in range(words.shape[0]):
            w32[R.ACT_GROUP >> 2] = g
            w32[R.ACT_EXP >> 2] = int(xe[g]) & 0xFFFFFFFF
            row = words[g].tolist()
            for i in range(R.ACT_MANT_WORDS):
                w32[base + i] = row[i]
            w32[R.ACT_COMMIT >> 2] = 1
        self._act = key
        return time.perf_counter() - t0

    def run(self, pa, e, lay: Layout, maxout, res_start=0, timeout=10.0):
        bursts = (e["tuples"] + e["pad"]) // lay.chunk
        self.wr(R.ROWS, e["rows"])
        self.wr(R.GROUPS, e["groups"])
        self.wr(R.PAD, e["pad"])
        self.wr(R.BURSTS, bursts)
        self.wr(R.CHUNK, lay.chunk)
        self.wr(R.STEP, lay.stripe_bytes)
        self.wr(R.MAXOUT, maxout)
        self.wr(R.RES_START, res_start)
        self.wr(R.BASE_LO, pa & 0xFFFFFFFF)
        self.wr(R.BASE_HI, pa >> 32)
        for k in range(lay.nports):
            self.wr(R.PORT_OFF + 4 * k, lay.port_slots[k] * lay.chunk_bytes)
        self.wr(R.CTRL, 1)
        t0 = time.perf_counter()
        while not self.rd(R.STATUS) & 2:
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError(f"{e['name']}: no DONE after {timeout}s, STATUS={self.rd(R.STATUS):#x} "
                                   f"rows_done={self.rd(R.ROWS_DONE)} tuples={self.rd(R.TUPLES)}")
        st = self.rd(R.STATUS)
        c = dict(status=st, cyc=self.rd64(R.CYC_LO), cyc_stream=self.rd64(R.CYC_STREAM_LO),
                 rows=self.rd(R.ROWS_DONE), sum=self.rd64(R.SUM_LO), xor=self.rd(R.XOR),
                 tuples=self.rd(R.TUPLES), stall_in=self.rd(R.STALL_IN), stall_out=self.rd(R.STALL_OUT),
                 err=self.rd(R.ERRCNT), ports=[])
        for k in range(lay.nports):
            o = R.port_reg(k, 0)
            xs = 0
            for i in range(4):
                xs |= self.rd(o + R.P_XSUM + 4 * i) << (32 * i)
            c["ports"].append(dict(beats=self.rd(o + R.P_BEATS), empty=self.rd(o + R.P_EMPTY),
                                   fifo_max=self.rd(o + R.P_FIFO_MAX), xsum=xs))
        return c

    def results(self, n):
        a = R.RES_BASE >> 2
        return np.array(self.w32[a:a + n], dtype=np.uint32)


def port_xsums(img: np.ndarray, lay: Layout):
    """128-bit xor of every beat each port reads from a packed single-matrix image."""
    ch = img.reshape(-1, lay.slots, lay.chunk, 16)
    out = []
    for k in range(lay.nports):
        x = np.bitwise_xor.reduce(ch[:, lay.port_slots[k]].reshape(-1, 16), axis=0)
        out.append(int.from_bytes(x.tobytes(), "little"))
    return out


def build_image(a, lay: Layout, names):
    """-> (pool, {name: dict(pa, size, entry, xsum, bytes, region)})"""
    idx_path = Path(a.work) / (os.path.basename(a.pool) + "_index.json")
    key = dict(gguf=os.path.basename(a.gguf), nports=lay.nports, chunk=lay.chunk, slots=lay.slots,
               port_slots=list(lay.port_slots), names=names)
    g = GGUF(a.gguf)
    tied = "output.weight" not in g.tensors
    from ..pack_weights import gguf_shape, matrix_tuples
    need = 0
    for n in names:
        rows, G = gguf_shape(g, gguf_name(n, tied))
        t, pad = matrix_tuples(rows, G, lay)
        need += (t + pad) // lay.chunk * lay.stripe_bytes
    need = -(-need // HP)
    pages = max(need + a.spare_pages, a.min_pages)   # small pools come back as isolated 2 MB fragments
    if not a.rebuild and idx_path.exists() and os.path.exists(a.pool):
        old = json.loads(idx_path.read_text())
        same = all(old["key"][k] == key[k] for k in key if k != "names") and set(names) <= set(old["mats"])
        if same and os.path.getsize(a.pool) == old["pages"] * HP:      # a superset image is fine
            pool = HugePool(old["pages"], a.pool)
            mats = {}
            for n, m in old["mats"].items():
                m["region"] = pool.region(m["pa"], m["size"])
                m["xsum"] = [int(x, 16) for x in m["xsum"]]
                mats[n] = m
            print(f"IMAGE re-used: {len(mats)} matrices, {pool.report()}")
            return pool, mats
    if os.path.exists(a.pool):
        os.unlink(a.pool)
    hp_settle(release_free=True)          # give an old pool's pages back first: re-used ones come back scattered
    for f in ("/proc/sys/vm/drop_caches", "/proc/sys/vm/compact_memory"):
        try:
            with open(f, "w") as fh:
                fh.write("3" if "drop" in f else "1")
        except OSError:
            pass
    pool = HugePool(pages, a.pool)
    print("IMAGE pool:", pool.report())
    t0, mats, total = time.time(), {}, 0
    order = sorted(names, key=lambda n: -gguf_shape(g, gguf_name(n, tied))[0] * gguf_shape(g, gguf_name(n, tied))[1])
    for n in order:                                           # biggest first: best use of the runs
        img, index = pack([(n, gguf_matrix(g, gguf_name(n, tied)))], lay)
        e = index[0].as_dict()
        try:
            reg = pool.alloc(len(img))
        except MemoryError:
            pool.close(free=True)
            raise
        reg.write(0, img)
        mats[n] = dict(pa=reg.pa, size=reg.size, entry=e, xsum=port_xsums(img, lay),
                       bytes=len(img), region=reg)
        total += len(img)
    evict()
    print(f"IMAGE packed {len(mats)} matrices, {total / 1e6:.1f} MB, in {time.time() - t0:.0f}s")
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(dict(key=key, pages=pool.n, mats={
        n: dict(pa=m["pa"], size=m["size"], entry=m["entry"], bytes=m["bytes"],
                xsum=[hex(x) for x in m["xsum"]]) for n, m in mats.items()})))
    return pool, mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.expanduser("~ubuntu/bigkev/data/Bonsai-8B-Q1_0.gguf"))
    ap.add_argument("--golden", default=os.path.expanduser("~ubuntu/bigkev/data/Bonsai-8B-Q1_0_token.npz"))
    ap.add_argument("--work", default=os.path.expanduser("~ubuntu/bigkev/work"))
    ap.add_argument("--pool", default="/dev/hugepages/bigkev")
    ap.add_argument("--fclk", type=float, default=333.333e6)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--port-slots", default="")
    ap.add_argument("--maxout", type=int, default=8)
    ap.add_argument("--layers", type=int, default=-1, help="first N layers (default all)")
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--spare-pages", type=int, default=64)
    ap.add_argument("--min-pages", type=int, default=256)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--free", action="store_true", help="release the hugepage pool at exit")
    ap.add_argument("--out", default="")
    ap.add_argument("--pack-only", type=int, default=0, metavar="NPORTS",
                    help="build the DDR image for NPORTS ports and stop (does not touch the PL)")
    a = ap.parse_args()

    gold = np.load(a.golden)
    if a.pack_only:
        fclk, eng = 0.0, argparse.Namespace(nports=a.pack_only, gmax=0, res_words=0)
    else:
        fclk = set_fclk(a.fclk)
        eng = Gemv()
    n_layer = max(int(k.split(".")[0][1:]) for k in gold.files if k[0] == "l" and k[1].isdigit() and k.endswith(".y")) + 1
    L = n_layer if a.layers < 0 else a.layers
    names = a.names or ([f"l{l}.{k}" for l in range(L) for k in KINDS] + ([] if a.no_head else ["lm_head"]))
    ps = tuple(int(x) for x in a.port_slots.split(",")) if a.port_slots else ()
    lay = Layout(nports=eng.nports, chunk=a.chunk, slots=a.slots, port_slots=ps, align="stripe")
    print(f"GEMV_BOARD fclk {fclk / 1e6:.2f} MHz, NPORTS {eng.nports}, GMAX {eng.gmax}, window {eng.res_words}, "
          f"chunk {lay.chunk}, slots {lay.slots} {lay.port_slots}, maxout {a.maxout}, {len(names)} matrices")
    pool, mats = build_image(a, lay, names)
    if a.pack_only:
        for n in names[:3] + names[-1:]:                     # read back through the region map
            m = mats[n]
            img = np.frombuffer(m["region"].read(0, m["bytes"]), dtype=np.uint8)
            assert port_xsums(img, lay) == m["xsum"], n
        print(f"GEMV_BOARD image ready ({len(mats)} matrices), read-back xsum ok")
        pool.close(free=a.free)
        return

    rows_out = []
    all_ok = True
    for rep in range(a.reps):
        tot = dict(cyc=0, cyc_stream=0, bytes=0, host=0.0, bad=0, stall_in=0)
        t_rep = time.perf_counter()
        for n in names:
            m, e = mats[n], mats[n]["entry"]
            y = gold[n + ".y"]
            key = n.rsplit(".", 1)[0] + (".qkv" if n.endswith((".q", ".k", ".v")) else
                                         ".ffn" if n.endswith((".gate", ".up")) else "." + n.rsplit(".", 1)[-1])
            tot["host"] += eng.load_act(key, gold[n + ".xq"], gold[n + ".xe"])
            c = eng.run(m["pa"], e, lay, a.maxout)
            beats = (e["tuples"] + e["pad"])
            ok = (c["status"] & 0xC) == 0 and c["err"] == 0 and c["rows"] == e["rows"] and c["tuples"] == e["tuples"] \
                and all(p["beats"] == beats and p["xsum"] == x for p, x in zip(c["ports"], m["xsum"]))
            data_ok = ok
            ysum, yxor = int(y.astype(np.uint64).sum()), int(np.bitwise_xor.reduce(y))
            ok &= c["sum"] == ysum and c["xor"] == yxor
            nbad = 0
            if e["rows"] <= eng.res_words:
                got = eng.results(e["rows"])
                nbad = int((got != y).sum())
                ok &= nbad == 0
            elif rep == 0 or not ok:                       # windowed read of a tall matrix (lm_head)
                for s in range(0, e["rows"], eng.res_words):
                    c2 = eng.run(m["pa"], e, lay, a.maxout, res_start=s)
                    k = min(eng.res_words, e["rows"] - s)
                    nbad += int((eng.results(k) != y[s:s + k]).sum())
                ok &= nbad == 0
            tot["cyc"] += c["cyc"]
            tot["cyc_stream"] += c["cyc_stream"]
            tot["bytes"] += beats * lay.nports * 16
            tot["stall_in"] += c["stall_in"]
            tot["bad"] += 0 if ok else 1
            all_ok &= ok
            gbs = beats * lay.nports * 16 / (c["cyc_stream"] / fclk) / 1e9 if c["cyc_stream"] else 0
            row = dict(rep=rep, name=n, ok=bool(ok), data_ok=bool(data_ok), bad_rows=nbad, gbs=gbs, **{k: v for k, v in c.items() if k != "ports"},
                       ports=[dict(beats=p["beats"], empty=p["empty"], fifo_max=p["fifo_max"]) for p in c["ports"]])
            rows_out.append(row)
            if not ok or a.reps == 1 and (n.startswith("l0.") or n == "lm_head"):
                print(f"  {n:9s} {'OK ' if ok else 'BAD'} rows {c['rows']}/{e['rows']} bad {nbad} image {'ok' if data_ok else 'BAD'} "
                      f"cyc {c['cyc']} stream {c['cyc_stream']} {gbs:5.2f} GB/s stall_in {c['stall_in']} "
                      f"fifo_max {[p['fifo_max'] for p in c['ports']]} status {c['status']:#x}")
        sec, sec_s = tot["cyc"] / fclk, tot["cyc_stream"] / fclk
        full = len(names) == n_layer * 7 + 1
        print(f"GEMV_BOARD rep {rep}: {len(names) - tot['bad']}/{len(names)} matrices bit-exact, "
              f"{tot['bytes'] / 1e6:.1f} MB in {sec * 1e3:.2f} ms ({tot['bytes'] / sec / 1e9:.2f} GB/s GO->last row; "
              f"{tot['bytes'] / sec_s / 1e9:.2f} GB/s first AR->last beat), stall_in {tot['stall_in'] / max(1, tot['cyc']):.3f}, "
              f"host act-load {tot['host'] * 1e3:.0f} ms, wall {time.perf_counter() - t_rep:.1f} s"
              + (f" | one token's weight stream: {sec * 1e3:.1f} ms -> {1 / sec:.2f} tok/s (weights only, MEASURED)" if full else ""))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as f:
            for r in rows_out:
                f.write(json.dumps(dict(fclk=fclk, chunk=lay.chunk, slots=lay.slots, port_slots=list(lay.port_slots),
                                        maxout=a.maxout, **r)) + "\n")
    print(f"GEMV_BOARD_VERDICT {'PASS' if all_ok else 'FAIL'}")
    pool.close(free=a.free)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
