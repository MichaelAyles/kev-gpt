"""pl_mamba_pipe — first-silicon driver for mamba_pipe_axi (the pipelined
NC-stream Mamba-2 WAVE engine; doc 9 P1).

Streams the SAME table images the sim gate used (ms_t0..15.mem + ms_cfg.mem
from run_mamba_pipe's sim dir) over AXI-Lite — including the rsqrt seed table
(WSEL_SEED=15, the silicon garbage-output fix) — preloads the gate's
per-(stream,token) tokens, pulses go once, and after done compares every
stream's per-token argmax (dump_tok readback, DBGSEL=2) against the
laptop-computed reference (mp_ref.npz) — the bit-honest step. The fabric
CYCLES counter gives the MEASURED cyc/token/stream and aggregate tok/s.

Laptop prep (after a green run_mamba_pipe):
  scp <simdir>/ms_t*.mem <simdir>/ms_cfg.mem <simdir>/mp_ref.npz kria:~/kevmem/

On the Kria (root for /dev/mem):
  sudo python3 -m fabric.stage3.board.pl_mamba_pipe --dir ~/kevmem --fclk 100e6
  sudo python3 -m fabric.stage3.board.pl_mamba_pipe --dir ~/kevmem --fclk 150e6 \
      --runs 8 --skip-load          # fclk sweep, tables already resident

The fclk MUST be forced via /sys (a flat fpgautil load does not apply the
BD's PL0_REF FREQMHZ — the known gotcha) and is verified after setting.
"""

from __future__ import annotations

import argparse
import ctypes
import mmap
import os
import struct
import sys
import time

import numpy as np

BASE = 0xA000_0000
R_CTRL, R_STATUS = 0x00, 0x04
R_TSEL, R_TADDR, R_TDATA, R_CYCLES = 0x10, 0x14, 0x18, 0x1C
R_DBGSEL, R_DBGADDR, R_DBGDATA = 0x20, 0x24, 0x28
R_TWADDR, R_TWDATA, R_IDCODE = 0x2C, 0x30, 0x34
IDCODE = 0x4D504950          # "MPIP"
DBG_TOK = 2                  # dump_tok: per-(stream,token) argmax (DBG=0-safe)


class MambaPipe:
    def __init__(self, base=BASE):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.m = mmap.mmap(self.fd, 0x1000, offset=base)
        # ONE 32-bit store per register access: an mmap slice assignment is a
        # memcpy that may emit several narrower bus writes, and the AXI shell
        # ignores WSTRB, so each extra beat is counted as another register
        # write (taddr double-increments and the table lands at 2x stride).
        self.w32 = (ctypes.c_uint32 * (0x1000 // 4)).from_buffer(self.m)

    def wr(self, off, val):
        self.w32[off >> 2] = val & 0xFFFFFFFF

    def rd(self, off):
        return int(self.w32[off >> 2])

    def wait_status(self, bit, timeout=30.0):
        t0 = time.time()
        while not (self.rd(R_STATUS) >> bit) & 1:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"status bit {bit} timeout "
                                   f"(STATUS={self.rd(R_STATUS):08x} "
                                   f"CYCLES={self.rd(R_CYCLES)})")

    def load_table(self, sel, words):
        self.wr(R_TSEL, sel)
        self.wr(R_TADDR, 0)
        w = self.wr
        for v in words:
            w(R_TDATA, int(v))

    def load_tokens(self, toks, tmax):
        """toks: (NC, T) int array; buffer address = stream*TMAX + tok."""
        nc, t = toks.shape
        for s in range(nc):
            for i in range(t):
                self.wr(R_TWADDR, s * tmax + i)
                self.wr(R_TWDATA, int(toks[s, i]) & 0x3FF)

    def go(self):
        self.wr(R_CTRL, 0b01)
        self.wait_status(0)                 # done_l
        return self.rd(R_CYCLES)

    def read_toks(self, nc, t, tmax):
        self.wr(R_DBGSEL, DBG_TOK)
        out = np.zeros((nc, t), dtype=np.int64)
        for s in range(nc):
            for i in range(t):
                self.wr(R_DBGADDR, s * tmax + i)
                out[s, i] = self.rd(R_DBGDATA) & 0x3FF
        return out


def set_and_verify_fclk(hz):
    path = "/sys/devices/platform/fclk0/set_rate"
    if not os.path.exists(path):
        for cand in ("/sys/class/clk/fclk0/set_rate",):
            if os.path.exists(cand):
                path = cand
                break
        else:
            print("WARN: no fclk0 set_rate node; PL clock NOT forced",
                  file=sys.stderr)
            return
    with open(path, "w") as f:
        f.write(str(int(hz)))
    print(f"fclk0 forced to {hz/1e6:.1f} MHz")


def rd16(path):
    with open(path) as f:
        return [int(l, 16) for l in f if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pl_mamba_pipe")
    ap.add_argument("--dir", required=True, help="dir with ms_t*.mem/ms_cfg/mp_ref")
    ap.add_argument("--fclk", type=float, default=100e6)
    ap.add_argument("--tmax", type=int, default=0,
                    help="bitstream TMAX (0 = assume == T from mp_ref.npz)")
    ap.add_argument("--runs", type=int, default=1,
                    help="extra timed go's after the verified one (cycles only; "
                         "state persists so argmax is checked on run 1 only)")
    ap.add_argument("--skip-load", action="store_true",
                    help="tables already resident (same power cycle)")
    ap.add_argument("--chars-per-tok", type=float, default=0.0,
                    help="if set, also print chars/s = tok/s * this")
    args = ap.parse_args(argv)

    ref = np.load(f"{args.dir}/mp_ref.npz")
    toks, ref_am = ref["toks"], ref["ref_argmax"]
    NC, T = toks.shape
    TMAX = args.tmax or T

    set_and_verify_fclk(args.fclk)
    d = MambaPipe()
    ident = d.rd(R_IDCODE)
    if ident != IDCODE:
        sys.exit(f"IDCODE mismatch: {ident:08x} != {IDCODE:08x} — wrong bitstream?")
    print("IDCODE ok (MPIP)")

    d.wr(R_CTRL, 0b10)                     # soft_reset -> state clear sweep
    d.wait_status(2)                        # ready
    print("ready (scan state cleared)")

    cfg = rd16(f"{args.dir}/ms_cfg.mem")
    if not args.skip_load:
        t0 = time.time()
        # identical order to tb_mamba_pipe: t1, t0(gw, count cfg[15]), t1 again,
        # t2..t14, then t15 = rsqrt seed (WSEL_SEED, count cfg[17] — the fix).
        d.load_table(1, rd16(f"{args.dir}/ms_t1.mem"))
        d.load_table(0, rd16(f"{args.dir}/ms_t0.mem"))
        for s in range(1, 15):
            d.load_table(s, rd16(f"{args.dir}/ms_t{s}.mem"))
        d.load_table(15, rd16(f"{args.dir}/ms_t15.mem"))
        print(f"tables loaded ({time.time()-t0:.1f}s, "
              f"{cfg[15] + sum(cfg[1:15]) + cfg[17]} words)")

    d.load_tokens(toks, TMAX)
    print(f"tokens loaded (NC={NC} T={T} TMAX={TMAX})")

    # ---- run 1: the bit-honest gate (fresh state, matches the sim reference) --
    t0 = time.time()
    cyc = d.go()
    wall = time.time() - t0
    got = d.read_toks(NC, T, TMAX)
    bad = 0
    for s in range(NC):
        for t in range(T):
            m = got[s, t] == ref_am[s, t]
            bad += not m
            print(f"  s{s} tok#{t} ({int(toks[s, t])}): argmax rtl {got[s, t]} "
                  f"ref {ref_am[s, t]} {'MATCH' if m else 'DIFF'}")
    verdict = "PASS" if bad == 0 else "FAIL"
    print(f"PL_MAMBA_PIPE_VERDICT: {verdict} ({NC}x{T} argmax on silicon vs "
          f"MambaSeqRef)")

    # ---- bench: cycles are deterministic; extra runs confirm + time the loop --
    cycs = [cyc]
    for _ in range(max(0, args.runs - 1)):
        cycs.append(d.go())
    per = float(np.mean(cycs)) / (NC * T)
    toks_s = args.fclk / per * NC
    print(f"PL_MAMBA_PIPE_BENCH: {per:,.0f} cyc/token/stream @ "
          f"{args.fclk/1e6:.0f} MHz = {args.fclk/per:,.1f} tok/s/stream x{NC} "
          f"= {toks_s:,.0f} tok/s aggregate "
          f"({NC*T/wall:,.0f} tok/s incl AXI wall, runs={len(cycs)}, "
          f"cyc={[int(c) for c in cycs[:4]]}{'...' if len(cycs) > 4 else ''})")
    if args.chars_per_tok:
        print(f"  => {toks_s*args.chars_per_tok:,.0f} chars/s "
              f"(at {args.chars_per_tok} chars/tok)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
