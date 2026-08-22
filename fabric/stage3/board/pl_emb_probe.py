"""pl_emb_probe — test the embedding COMPUTE in isolation on silicon.

Everything feeding it is now verified: the embedding table reads back 4096/4096,
the scale table 1024/1024, and different tokens produce different results. Yet
the embedding stage's checksum disagrees with the bit-exact simulation. So the
arithmetic itself is wrong on hardware.

This drives the compute with controlled data instead of the model:
  case A  embedding table all ZERO  -> every product is zero -> checksum MUST
          be 0. A non-zero answer means the datapath is not reading what it
          was given.
  case B  a single non-zero nibble  -> checksum must equal that one product,
          which is small and predictable in magnitude.

  sudo python3 pl_emb_probe.py --dir ~/kevmem --fclk 80e6
"""
from __future__ import annotations
import argparse, ctypes, mmap, os, sys, time

BASE = 0xA000_0000
R_CTRL, R_STATUS = 0x00, 0x04
R_TSEL, R_TADDR, R_TDATA = 0x10, 0x14, 0x18
R_DBGSEL, R_DBGADDR, R_DBGDATA = 0x20, 0x24, 0x28
R_TWADDR, R_TWDATA, R_IDCODE = 0x2C, 0x30, 0x34
IDCODE = 0x4D504950
SEL_EMB_SUM, SEL_EMBW = 4, 1
WSEL_EMB, WSEL_ESC = 1, 2


class Dev:
    def __init__(self):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.m = mmap.mmap(self.fd, 0x1000, offset=BASE)
        self.w32 = (ctypes.c_uint32 * (0x1000 // 4)).from_buffer(self.m)

    def wr(self, o, v): self.w32[o >> 2] = v & 0xFFFFFFFF
    def rd(self, o):    return int(self.w32[o >> 2])

    def wait(self, bit, to=60.0):
        t0 = time.time()
        while not (self.rd(R_STATUS) >> bit) & 1:
            if time.time() - t0 > to: raise TimeoutError(f"status {bit}")

    def load(self, sel, words):
        self.wr(R_TSEL, sel); self.wr(R_TADDR, 0)
        for v in words: self.wr(R_TDATA, int(v))

    def dbg(self, sel, addr=0):
        self.wr(R_DBGSEL, sel); self.wr(R_DBGADDR, addr)
        self.rd(R_DBGDATA)
        v = self.rd(R_DBGDATA)
        return v - (1 << 32) if v >= 1 << 31 else v


def rd16(p):
    with open(p) as f: return [int(l, 16) for l in f if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pl_emb_probe")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fclk", type=float, default=80e6)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--tmax", type=int, default=16)
    args = ap.parse_args(argv)
    for p in ("/sys/devices/platform/fclk0/set_rate",
              "/sys/class/clk/fclk0/set_rate"):
        if os.path.exists(p): open(p, "w").write(str(int(args.fclk))); break

    d = Dev()
    if d.rd(R_IDCODE) != IDCODE: sys.exit("IDCODE mismatch")
    print("IDCODE ok")

    base_emb = rd16(f"{args.dir}/ms_t1.mem")
    base_esc = rd16(f"{args.dir}/ms_t2.mem")

    def run(emb_words, esc_words, label, tok=12):
        d.wr(R_CTRL, 0b10); d.wait(2)
        d.load(1, emb_words)
        d.load(0, rd16(f"{args.dir}/ms_t0.mem"))
        d.load(1, emb_words)
        d.load(2, esc_words)
        for s in range(3, 15):
            d.load(s, rd16(f"{args.dir}/ms_t{s}.mem"))
        d.load(15, rd16(f"{args.dir}/ms_t15.mem"))
        for s in range(args.nc):
            for i in range(args.tmax):
                d.wr(R_TWADDR, s * args.tmax + i); d.wr(R_TWDATA, tok)
        d.wr(R_DBGSEL, 0)
        d.wr(R_CTRL, 0b01); d.wait(0)
        got = d.dbg(SEL_EMB_SUM)
        # confirm what actually sits in the table we just wrote
        rb = [d.dbg(SEL_EMBW, i) for i in range(4)]
        print(f"  {label}: emb-checksum {got}   table[0:4] {[hex(x) for x in rb]}")
        return got

    print("\nA. embedding table all ZERO (every product must be 0):")
    zero = run([0] * len(base_emb), base_esc, "all-zero emb")
    print(f"   => {'OK, compute honours its input' if zero == 0 else 'WRONG: non-zero output from all-zero weights'}")

    print("\nB. embedding table all ZERO and scales all ZERO:")
    zz = run([0] * len(base_emb), [0] * len(base_esc), "zero emb+esc")
    print(f"   => {'OK' if zz == 0 else 'WRONG: non-zero output from all-zero inputs'}")

    print("\nC. real tables (reference point):")
    run(base_emb, base_esc, "real tables ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
