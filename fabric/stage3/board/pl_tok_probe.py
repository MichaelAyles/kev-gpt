"""pl_tok_probe — does the input token actually reach the engine on silicon?

Every table now reads back byte-exact from fabric (embedding, embedding-scale,
rsqrt seed, constants, in_proj scales, gemv weights), yet the EMBEDDING stage —
the first thing the engine computes — is wrong, and the output does not vary
with the input. The one input never verified is the token itself.

This runs the same engine twice with deliberately different token sets and
compares. Identical results prove the tokens are not reaching the embedding;
different results clear the token path and point at the embedding compute.
Needs no new bitstream.

  sudo python3 pl_tok_probe.py --dir ~/kevmem --fclk 80e6 --tmax 16
"""
from __future__ import annotations
import argparse, ctypes, mmap, os, struct, sys, time
import numpy as np

BASE = 0xA000_0000
R_CTRL, R_STATUS = 0x00, 0x04
R_TSEL, R_TADDR, R_TDATA, R_CYCLES = 0x10, 0x14, 0x18, 0x1C
R_DBGSEL, R_DBGADDR, R_DBGDATA = 0x20, 0x24, 0x28
R_TWADDR, R_TWDATA, R_IDCODE = 0x2C, 0x30, 0x34
IDCODE = 0x4D504950
SEL_TOK, SEL_EMB = 2, 4


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
            if time.time() - t0 > to:
                raise TimeoutError(f"status {bit}")

    def load_table(self, sel, words):
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
    ap = argparse.ArgumentParser(prog="pl_tok_probe")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fclk", type=float, default=80e6)
    ap.add_argument("--tmax", type=int, default=16)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--t", type=int, default=16)
    args = ap.parse_args(argv)

    for p in ("/sys/devices/platform/fclk0/set_rate",
              "/sys/class/clk/fclk0/set_rate"):
        if os.path.exists(p):
            open(p, "w").write(str(int(args.fclk))); break

    d = Dev()
    if d.rd(R_IDCODE) != IDCODE: sys.exit("IDCODE mismatch")
    print("IDCODE ok")

    NC, T, TMAX = args.nc, args.t, args.tmax

    def run(tokens, label):
        d.wr(R_CTRL, 0b10); d.wait(2)
        # tables must be reloaded: soft_reset does not clear them, but the
        # engine's scan state must start fresh for a fair comparison
        for s, f in [(1, "ms_t1.mem"), (0, "ms_t0.mem")]:
            d.load_table(s, rd16(f"{args.dir}/{f}"))
        for s in range(1, 15):
            d.load_table(s, rd16(f"{args.dir}/ms_t{s}.mem"))
        d.load_table(15, rd16(f"{args.dir}/ms_t15.mem"))
        for s in range(NC):
            for i in range(T):
                d.wr(R_TWADDR, s * TMAX + i)
                d.wr(R_TWDATA, int(tokens[s][i]) & 0x3FF)
        d.wr(R_DBGSEL, 0)
        d.wr(R_CTRL, 0b01); d.wait(0)
        out = []
        d.wr(R_DBGSEL, SEL_TOK)
        for s in range(NC):
            for i in range(T):
                d.wr(R_DBGADDR, s * TMAX + i)
                d.rd(R_DBGDATA)
                out.append(d.rd(R_DBGDATA) & 0x3FF)
        emb = d.dbg(SEL_EMB)
        print(f"  {label}: emb-checksum {emb}  argmax[:8] {out[:8]}")
        return emb, out

    print("\nrunning the SAME engine with two DIFFERENT token sets:")
    a = run([[12] * T for _ in range(NC)], "tokens all 12 ")
    b = run([[500] * T for _ in range(NC)], "tokens all 500")
    c = run([[(s * 97 + i * 13) % 1024 for i in range(T)] for s in range(NC)],
            "tokens varied ")

    same = (a == b == c)
    print("\nTOKEN_VERDICT: " +
          ("IDENTICAL for every token set — the token never reaches the "
           "embedding; the token path is the bug"
           if same else
           "results DIFFER with the tokens — the token path works, so the "
           "embedding compute itself is wrong"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
