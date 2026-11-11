"""gemv_axi version 2 on the board: activations and results through DDR, argmax in the fabric.

`PlDma` wraps the C callback `bkpl_gemv` (board/mmio.c) for a loaded image: it is what
`pl_token_c.py --dma` hands to the C host, and this module's own main replays the golden token
through it (the version-2 counterpart of `pl_gemv.py`):

    sudo python3 -m fabric.big.board.pl_dma --fclk 333.333e6 --reps 3      -> GEMV_DMA_VERDICT

Here the wall time of a replay is meaningful: per matrix the host writes the activation
records, cleans them to DDR, writes 8 registers, polls, invalidates the result lines and
copies the rows out. That is all the host<->fabric traffic a token needs.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import numpy as np

from .. import gemv_axi_regs as R
from ..pack_weights import Layout
from .physmem import HP
from .pl_gemv import KINDS, Gemv, build_image, set_fclk

u32, u64, i32, vp = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int32, ctypes.c_void_p

REG_ORDER = ("CTRL", "STATUS", "ROWS", "GROUPS", "PAD", "BURSTS", "CHUNK", "STEP", "MAXOUT", "RES_START",
             "BASE_LO", "BASE_HI", "MODE", "PORT_OFF", "ACT_ADDR_LO", "ACT_ADDR_HI", "RES_ADDR_LO",
             "RES_ADDR_HI", "CYC_LO", "CYC_HI", "ROWS_DONE", "TUPLES", "STALL_IN", "ERRCNT", "ARGMAX_IDX",
             "ARGMAX_VAL", "CYC_ACT", "CYC_RES", "PORT_BASE", "PORT_STRIDE", "P_BEATS", "P_XSUM",
             "BATCH", "ARGMAX_B")                               # enum in mmio.c


class BkplMat(ctypes.Structure):
    _fields_ = [("pa", u64), ("rows", u32), ("groups", u32), ("pad", u32), ("bursts", u32), ("tuples", u32),
                ("beats", u32), ("xsum", (u32 * 4) * 8)]


class Bkpl(ctypes.Structure):                                   # mirrors `bkpl` in mmio.c
    _fields_ = [("regs", vp), ("off", u32 * len(REG_ORDER)),
                ("nports", i32), ("chunk", i32), ("maxout", i32), ("n_mats", i32),
                ("step", u32), ("port_off", u32 * 8),
                ("mats", vp), ("act_va", vp), ("res_va", vp), ("act_pa", u64), ("res_pa", u64),
                ("head_argmax", i32), ("check_xsum", i32), ("spins", u64),
                ("cyc", u64), ("cyc_act", u64), ("cyc_res", u64), ("stall_in", u64), ("runs", u64), ("act_loads", u64),
                ("ns_act", u64), ("ns_regs", u64), ("ns_wait", u64), ("ns_read", u64),
                ("last_argmax", i32), ("err_mat", i32), ("last_status", u32), ("err_code", u32), ("last_xq", vp),
                ("nb_max", i32), ("last_nb", i32), ("argmax_b", i32 * 8), ("res_cap", u64)]


COUNTERS = ("cyc", "cyc_act", "cyc_res", "stall_in", "runs", "act_loads", "ns_act", "ns_regs", "ns_wait", "ns_read")
ERRORS = {-10: "bad mat_id", -11: "shape mismatch", -12: "engine status / rows / tuples / RRESP",
          -13: "beat count", -14: "the beats read from DDR are not the packed image", -15: "argmax index out of range",
          -16: "batch size not supported by this bitstream", -17: "results of this batch do not fit the scratch region"}


class PlDma:
    ACT_OFF, RES_OFF = 0, 1 << 18                               # inside one scratch hugepage (NB * 96 records = 98 kB at NB 4)

    def __init__(self, eng: Gemv, mats, lay: Layout, names, pool, maxout=8, head_argmax=True, check_xsum=True):
        ver = eng.rd(R.PARAMS) >> 24
        if ver < 2:
            sys.exit(f"the loaded gemv_axi is version {ver}: no aux DMA (needs version 2)")
        if eng.mmio is None:
            sys.exit("the DMA path needs the C helper (board/mmio.c) and so a compiler on the board")
        lib = self.lib = eng.mmio
        lib.bkpl_gemv.argtypes, lib.bkpl_gemv.restype = [vp, ctypes.c_int, vp, vp, ctypes.c_int, vp, ctypes.c_int], ctypes.c_int
        lib.bkpl_setup.argtypes, lib.bkpl_setup.restype = [vp], None
        PP = ctypes.POINTER(vp)
        lib.bkpl_gemv_batch.argtypes = [vp, ctypes.c_int, ctypes.c_int, PP, PP, ctypes.c_int, PP, ctypes.c_int, ctypes.c_int]
        lib.bkpl_gemv_batch.restype = ctypes.c_int
        self.nb = ((eng.rd(R.PARAMS) >> 4) & 0xF or 1) if ver >= 3 else 0   # 0: no BATCH register
        lib.bkpl_cache_invalidate.argtypes, lib.bkpl_cache_invalidate.restype = [vp, ctypes.c_size_t], None
        self.names = list(names)
        reg = pool.alloc(HP, align=HP)                          # one hugepage: virtually contiguous
        va = pool.va(reg)
        ctypes.memset(va, 0, HP)
        lib.bkpl_cache_invalidate(va, HP)                       # zeros reach DDR; nothing stays cached
        self.marr = (BkplMat * len(names))()
        for i, n in enumerate(names):
            m, e, t = mats[n], mats[n]["entry"], self.marr[i]
            beats = e["tuples"] + e["pad"]
            t.pa, t.rows, t.groups, t.pad, t.tuples, t.beats = m["pa"], e["rows"], e["groups"], e["pad"], e["tuples"], beats
            t.bursts = beats // lay.chunk
            for k, x in enumerate(m["xsum"]):
                for j in range(4):
                    t.xsum[k][j] = (x >> (32 * j)) & 0xFFFFFFFF
        c = self.c = Bkpl()
        c.regs = eng.base
        for i, n in enumerate(REG_ORDER):
            c.off[i] = getattr(R, n)
        c.nports, c.chunk, c.maxout, c.n_mats = lay.nports, lay.chunk, maxout, len(names)
        c.step = lay.stripe_bytes
        for k in range(lay.nports):
            c.port_off[k] = lay.port_slots[k] * lay.chunk_bytes
        c.mats = ctypes.addressof(self.marr)
        c.act_va, c.act_pa = va + self.ACT_OFF, reg.pa + self.ACT_OFF
        c.res_va, c.res_pa = va + self.RES_OFF, reg.pa + self.RES_OFF
        c.head_argmax, c.check_xsum, c.spins = int(head_argmax), int(check_xsum), 200_000_000
        c.nb_max, c.res_cap = self.nb, HP - self.RES_OFF
        lib.bkpl_setup(ctypes.byref(c))
        self.ctx = ctypes.addressof(c)

    def reset(self):
        for k in COUNTERS:
            setattr(self.c, k, 0)

    def counters(self):
        return {k: int(getattr(self.c, k)) for k in COUNTERS}

    def gemv(self, mid, xq, xe, y):
        """one GEMV from Python (the replay): xq int16 [G*128], xe int8 [G], y float32 [rows] out."""
        rc = self.lib.bkpl_gemv(self.ctx, mid, xq.ctypes.data, xe.ctypes.data, len(xe), y.ctypes.data, len(y))
        if rc:
            raise RuntimeError(f"{self.names[mid]}: bkpl_gemv {rc} ({ERRORS.get(rc, '?')}), status {self.c.last_status:#x}")


    def gemv_batch(self, mid, xqs, xes, ys, reuse=False):
        """one batch GEMV from Python: lists of nb arrays (as gemv's); ys[b] may be None for the argmax-only head."""
        nb = len(xqs)
        A = vp * nb
        rc = self.lib.bkpl_gemv_batch(self.ctx, mid, nb, A(*[x.ctypes.data for x in xqs]), A(*[x.ctypes.data for x in xes]),
                                      len(xes[0]), A(*[None if y is None else y.ctypes.data for y in ys]),
                                      self.marr[mid].rows, int(reuse))
        if rc:
            raise RuntimeError(f"{self.names[mid]}: bkpl_gemv_batch {rc} ({ERRORS.get(rc, '?')}), status {self.c.last_status:#x}")


def batch_replay(a, eng, dma, lay, names, fclk):
    """--batch N: N positions' activations through each matrix in one pass (register map VERSION 3)."""
    gold = np.load(a.golden_batch)
    N = a.batch
    P = len(gold["positions"])
    if N > P or N > dma.nb:
        sys.exit(f"--batch {N}: the golden has {P} positions and the bitstream NB = {dma.nb}")
    xs, last = [], None
    for n in names:
        xq, xe = np.ascontiguousarray(gold[n + ".xq"][P - N:]), np.ascontiguousarray(gold[n + ".xe"][P - N:])
        same = last is not None and np.array_equal(xq, last[0]) and np.array_equal(xe, last[1])
        if same:
            xq, xe = last[0], last[1]
        last = (xq, xe, same)
        xs.append(last)
    ys = [gold[n + ".y"][P - N:] for n in names]
    head = ys[-1].view(np.float32)
    head_idx = [int(np.flatnonzero(h == h.max())[0]) for h in head]
    full_head_fits = head.shape[1] * N * 4 <= dma.c.res_cap
    all_ok = True
    for rep in range(a.reps):
        dma.c.head_argmax = 0 if (rep == 0 and full_head_fits) else 1
        dma.c.last_nb = 0                                    # force a reload on the first matrix
        dma.reset()
        bad = 0
        t0 = time.perf_counter()
        for i, n in enumerate(names):
            y = [np.empty(ys[i].shape[1], dtype=np.float32) for _ in range(N)]
            dma.gemv_batch(i, list(xs[i][0]), list(xs[i][1]), y, reuse=xs[i][2])
            for b in range(N):
                if i == len(names) - 1 and dma.c.head_argmax:
                    ok = dma.c.argmax_b[b] == head_idx[b] and y[b][head_idx[b]].view(np.uint32) == ys[i][b][head_idx[b]]
                else:
                    ok = np.array_equal(y[b].view(np.uint32), ys[i][b])
                if not ok:
                    bad += 1
                    print(f"  {n} element {b} BAD")
        wall = time.perf_counter() - t0
        c = dma.counters()
        nbytes = sum(m.beats for m in dma.marr) * lay.nports * 16
        sec = c["cyc"] / fclk
        all_ok &= bad == 0
        print(f"GEMV_DMA_BATCH rep {rep} (batch {N}, {'full LM-head readback' if not dma.c.head_argmax else 'fabric argmax'}): "
              f"{len(names) * N - bad}/{len(names) * N} (matrix, element) results bit-exact, fabric {sec * 1e3:.2f} ms/pass "
              f"({nbytes / sec / 1e9:.2f} GB/s incl. {c['cyc_act'] / fclk * 1e3:.2f} ms activation fetch; result write tail "
              f"{c['cyc_res'] / fclk * 1e3:.2f} ms) = {N / sec:.1f} positions/s weights-only, wall {wall * 1e3:.1f} ms/pass "
              f"(pack+clean {c['ns_act'] / 1e6:.1f} ms, regs {c['ns_regs'] / 1e6:.1f}, wait {c['ns_wait'] / 1e6:.1f}, readback "
              f"{c['ns_read'] / 1e6:.1f}; {c['act_loads']} activation loads), stall_in {c['stall_in'] / max(1, c['cyc']):.3f}")
    print(f"GEMV_DMA_BATCH_VERDICT {'PASS' if all_ok else 'FAIL'} (batch {N} of NB {dma.nb})")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.expanduser("~ubuntu/bigkev/data/Bonsai-8B-Q1_0.gguf"))
    ap.add_argument("--golden", default=os.path.expanduser("~ubuntu/bigkev/data/Bonsai-8B-Q1_0_token.npz"))
    ap.add_argument("--work", default=os.path.expanduser("~ubuntu/bigkev/work"))
    ap.add_argument("--pool", default="/dev/hugepages/bigkev")
    ap.add_argument("--fclk", type=float, default=333.333e6)
    ap.add_argument("--maxout", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=0, help="replay N positions per pass (needs a VERSION 3 bitstream)")
    ap.add_argument("--golden-batch", default=os.path.expanduser("~ubuntu/bigkev/data/Bonsai-8B-Q1_0_token_b4.npz"))
    a = ap.parse_args()
    for k, v in dict(chunk=256, slots=0, port_slots="", rebuild=False, spare_pages=64, min_pages=256).items():
        setattr(a, k, v)
    fclk = set_fclk(a.fclk)
    eng = Gemv()
    gold = np.load(a.golden)
    n_layer = max(int(k.split(".")[0][1:]) for k in gold.files if k[0] == "l" and k[1].isdigit() and k.endswith(".y")) + 1
    names = [f"l{l}.{k}" for l in range(n_layer) for k in KINDS] + ["lm_head"]
    lay = Layout(nports=eng.nports, chunk=a.chunk, align="stripe")
    pool, mats = build_image(a, lay, names)
    dma = PlDma(eng, mats, lay, names, pool, a.maxout)
    if a.batch:
        ok = batch_replay(a, eng, dma, lay, names, fclk)
        pool.close(free=False)
        sys.exit(0 if ok else 1)
    # one input array per distinct activation, so q/k/v and gate/up exercise the reuse path
    xs, last = [], None
    for n in names:
        xq, xe = np.ascontiguousarray(gold[n + ".xq"]), np.ascontiguousarray(gold[n + ".xe"])
        if last is not None and np.array_equal(xq, last[0]) and np.array_equal(xe, last[1]):
            xq, xe = last
        last = (xq, xe)
        xs.append(last)
    ys = [gold[n + ".y"] for n in names]
    head = ys[-1]
    hv = head.view(np.float32)
    head_idx = int(np.flatnonzero(hv == hv.max())[0])
    all_ok = True
    for rep in range(a.reps):
        dma.c.head_argmax = 0 if rep == 0 else 1            # rep 0 reads the whole LM head back through DDR
        dma.c.last_xq = None
        dma.reset()
        bad = 0
        t0 = time.perf_counter()
        for i, n in enumerate(names):
            y = np.empty(len(ys[i]), dtype=np.float32)
            dma.gemv(i, xs[i][0], xs[i][1], y)
            if i == len(names) - 1 and dma.c.head_argmax:
                ok = dma.c.last_argmax == head_idx and y[head_idx].view(np.uint32) == head[head_idx]
            else:
                ok = np.array_equal(y.view(np.uint32), ys[i])
            if not ok:
                bad += 1
                print(f"  {n} BAD" + (f" argmax {dma.c.last_argmax} != {head_idx}" if i == len(names) - 1 and dma.c.head_argmax else
                                      f" {int((y.view(np.uint32) != ys[i]).sum())} rows"))
        wall = time.perf_counter() - t0
        c = dma.counters()
        nbytes = sum(m.beats for m in dma.marr) * lay.nports * 16
        sec = c["cyc"] / fclk
        all_ok &= bad == 0
        print(f"GEMV_DMA rep {rep} ({'full LM-head readback' if rep == 0 else 'fabric argmax'}): {len(names) - bad}/{len(names)} bit-exact, "
              f"fabric {sec * 1e3:.2f} ms/token ({nbytes / sec / 1e9:.2f} GB/s incl. {c['cyc_act'] / fclk * 1e3:.2f} ms activation fetch; "
              f"result write tail {c['cyc_res'] / fclk * 1e3:.2f} ms), wall {wall * 1e3:.1f} ms/token = {1 / wall:.2f} tok/s for GEMV + all host io "
              f"(pack+clean {c['ns_act'] / 1e6:.1f} ms, regs {c['ns_regs'] / 1e6:.1f}, wait {c['ns_wait'] / 1e6:.1f}, readback {c['ns_read'] / 1e6:.1f}; "
              f"{c['act_loads']} activation loads), stall_in {c['stall_in'] / max(1, c['cyc']):.3f}")
    print(f"GEMV_DMA_VERDICT {'PASS' if all_ok else 'FAIL'}")
    pool.close(free=False)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
