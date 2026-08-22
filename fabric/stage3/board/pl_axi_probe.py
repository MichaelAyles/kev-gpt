"""pl_axi_probe — prove what the AXI table-load path actually does on silicon.

The silicon readback showed every table corrupt with the SAME structure: each
value repeated in two consecutive addresses. Four independent memories cannot
share that pattern unless the AXI write/read path itself is at fault. This
probe writes a SELF-IDENTIFYING table (word i = i) into the constants memory
and reads it straight back, so the address mapping is unambiguous:

    stored[i] == i          -> load path fine, look elsewhere
    stored[i] == i >> 1     -> every second write dropped (addresses advance
                               half as fast as data)
    stored[i] == i * 2      -> writes double-executed (taddr increments twice)

It also re-reads one address repeatedly to separate a WRITE-path fault from a
READ-path artifact, and re-reads out of order to rule out my own loop.

  sudo python3 -m fabric.stage3.board.pl_axi_probe --fclk 80e6
"""

from __future__ import annotations

import argparse
import ctypes
import mmap
import os
import struct
import sys

BASE = 0xA000_0000
R_CTRL, R_STATUS = 0x00, 0x04
R_TSEL, R_TADDR, R_TDATA = 0x10, 0x14, 0x18
R_DBGSEL, R_DBGADDR, R_DBGDATA, R_IDCODE = 0x20, 0x24, 0x28, 0x34
IDCODE = 0x4D504950
SEL_CONSTS, WSEL_CONST = 8, 14
N = 128


class Dev:
    def __init__(self):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.m = mmap.mmap(self.fd, 0x1000, offset=BASE)
        # ONE 32-bit store per register access (see pl_axi_probe): mmap slice
        # assignment is a memcpy and can emit several bus writes, which the
        # WSTRB-ignoring AXI shell counts as several register writes.
        self.w32 = (ctypes.c_uint32 * (0x1000 // 4)).from_buffer(self.m)

    def wr(self, off, val):
        self.w32[off >> 2] = val & 0xFFFFFFFF

    def rd(self, off):
        return int(self.w32[off >> 2])

    def read_const(self, i, settle=2):
        self.wr(R_DBGSEL, SEL_CONSTS)
        self.wr(R_DBGADDR, i)
        for _ in range(settle):
            self.rd(R_DBGDATA)
        return self.rd(R_DBGDATA)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pl_axi_probe")
    ap.add_argument("--fclk", type=float, default=80e6)
    args = ap.parse_args(argv)

    for p in ("/sys/devices/platform/fclk0/set_rate",
              "/sys/class/clk/fclk0/set_rate"):
        if os.path.exists(p):
            open(p, "w").write(str(int(args.fclk)))
            print("fclk0 set %.1f MHz" % (args.fclk / 1e6))
            break

    d = Dev()
    if d.rd(R_IDCODE) != IDCODE:
        sys.exit("IDCODE mismatch — wrong bitstream?")
    print("IDCODE ok (MPIP)")
    d.wr(R_CTRL, 0b10)
    while not (d.rd(R_STATUS) >> 2) & 1:
        pass
    print("ready")

    # ---- write a self-identifying constants table: word i = i --------------
    d.wr(R_TSEL, WSEL_CONST)
    d.wr(R_TADDR, 0)
    for i in range(N):
        d.wr(R_TDATA, i)
    print("wrote %d words (word i = i)\n" % N)

    got = [d.read_const(i) for i in range(N)]
    print("addr : silicon   (expect addr)")
    for i in range(16):
        print("  %3d : %8d" % (i, got[i]))

    exact = sum(1 for i in range(N) if got[i] == i)
    halved = sum(1 for i in range(N) if got[i] == i >> 1)
    doubled = sum(1 for i in range(N) if got[i] == i * 2)
    print("\nmatches  exact %d/%d | i>>1 %d/%d | i*2 %d/%d"
          % (exact, N, halved, N, doubled, N))
    if exact == N:
        print("AXI_PROBE: LOAD PATH OK")
    elif halved > N // 2:
        print("AXI_PROBE: EVERY SECOND WRITE DROPPED (addresses advance twice "
              "per data word)")
    elif doubled > N // 4:
        print("AXI_PROBE: WRITES DOUBLE-EXECUTED (taddr increments twice)")
    else:
        print("AXI_PROBE: CORRUPT, pattern not one of the simple cases")

    # ---- is it the write path or the readback? ----------------------------
    print("\nsame address re-read (addr 5, five times):",
          [d.read_const(5) for _ in range(5)])
    print("descending re-read (7..0):", [d.read_const(i) for i in range(7, -1, -1)])
    print("extra settle reads (addr 5, settle=8):", d.read_const(5, settle=8))
    return 0


if __name__ == "__main__":
    sys.exit(main())
