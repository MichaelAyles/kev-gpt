"""pl_mamba_diag — silicon localizer for the wave engine (mamba_pipe_axi, the
DBG=1 diagnostic build: NC=1, TMAX=2, full x/logit readback + TABLE readback).

The wave engine runs on silicon with a CYCLE-EXACT match to sim (control path
provably correct) but produces garbage logits, while every RTL/AXI simulation
is bit-exact. That splits the suspects into "what the fabric computes" vs
"what actually got loaded into the fabric's memories" — sim cannot see the
latter. This driver checks the loaded memories FIRST (gemv weight URAM, the
consts, the rsqrt seed, the embedding-scale table), then runs a token and
compares logits/residual against the laptop reference.

  sudo python3 -m fabric.stage3.board.pl_mamba_diag --dir ~/kevmem --fclk 80e6
  sudo python3 -m fabric.stage3.board.pl_mamba_diag --dir ~/kevmem --tables-only

Readback selectors (mamba_pipe dbg_sel): 0=dump_x 1=dump_logit 2=dump_tok
8=consts 9=gemv weight word 10=rsqrt seed 11=emb scale.
"""

from __future__ import annotations

import argparse
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
V, D = 1024, 256
SEL_X, SEL_LOGIT, SEL_TOK = 0, 1, 2
SEL_CONSTS, SEL_GW, SEL_SEED, SEL_ESC = 8, 9, 10, 11


class Diag:
    def __init__(self, base=BASE):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.m = mmap.mmap(self.fd, 0x1000, offset=base)

    def wr(self, off, val):
        self.m[off:off + 4] = struct.pack("<I", val & 0xFFFFFFFF)

    def rd(self, off):
        return struct.unpack("<I", self.m[off:off + 4])[0]

    def wait_status(self, bit, timeout=60.0):
        t0 = time.time()
        while not (self.rd(R_STATUS) >> bit) & 1:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"status bit {bit} timeout "
                                   f"(STATUS={self.rd(R_STATUS):08x})")

    def load_table(self, sel, words):
        self.wr(R_TSEL, sel)
        self.wr(R_TADDR, 0)
        w = self.wr
        for v in words:
            w(R_TDATA, int(v))

    def dbg_read(self, sel, addr):
        self.wr(R_DBGSEL, sel)
        self.wr(R_DBGADDR, addr)
        self.rd(R_DBGDATA)          # discard: the mux is registered
        return self.rd(R_DBGDATA)

    def dbg_block(self, sel, addrs):
        self.wr(R_DBGSEL, sel)
        out = []
        for a in addrs:
            self.wr(R_DBGADDR, int(a))
            self.rd(R_DBGDATA)
            out.append(self.rd(R_DBGDATA))
        return np.asarray(out, dtype=np.int64)


def set_fclk(hz):
    for path in ("/sys/devices/platform/fclk0/set_rate",
                 "/sys/class/clk/fclk0/set_rate"):
        if os.path.exists(path):
            with open(path, "w") as f:
                f.write(str(int(hz)))
            rb = path.replace("set_rate", "get_rate")
            actual = open(rb).read().strip() if os.path.exists(rb) else "?"
            print(f"fclk0 set {hz/1e6:.1f} MHz (readback {actual})")
            return
    print("WARN: no fclk0 node; PL clock NOT forced", file=sys.stderr)


def rd16(path):
    with open(path) as f:
        return [int(l, 16) for l in f if l.strip()]


def cmp_table(name, got, want, limit=6):
    got = np.asarray(got, dtype=np.int64)
    want = np.asarray(want, dtype=np.int64)
    bad = np.flatnonzero(got != want)
    if bad.size == 0:
        print(f"  {name}: OK ({got.size} words match)")
        return True
    print(f"  {name}: MISMATCH {bad.size}/{got.size} words")
    for i in bad[:limit]:
        print(f"      [{i}] silicon {got[i]:#010x} expected {want[i]:#010x}")
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pl_mamba_diag")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fclk", type=float, default=80e6)
    ap.add_argument("--skip-load", action="store_true")
    ap.add_argument("--tables-only", action="store_true",
                    help="verify the loaded memories, skip the run")
    ap.add_argument("--gw-samples", type=int, default=512,
                    help="weight words to spot-check (0 = all, slow)")
    args = ap.parse_args(argv)

    set_fclk(args.fclk)
    d = Diag()
    ident = d.rd(R_IDCODE)
    if ident != IDCODE:
        sys.exit(f"IDCODE mismatch {ident:08x} — wrong bitstream?")
    print("IDCODE ok (MPIP)")

    d.wr(R_CTRL, 0b10)
    d.wait_status(2)
    print("ready (state cleared)")

    cfg = rd16(f"{args.dir}/ms_cfg.mem")
    gw = rd16(f"{args.dir}/ms_t0.mem")
    consts = rd16(f"{args.dir}/ms_t14.mem")
    seed = rd16(f"{args.dir}/ms_t15.mem")
    esc = rd16(f"{args.dir}/ms_t2.mem")

    if not args.skip_load:
        t0 = time.time()
        d.load_table(1, rd16(f"{args.dir}/ms_t1.mem"))
        d.load_table(0, gw)
        for s in range(1, 15):
            d.load_table(s, rd16(f"{args.dir}/ms_t{s}.mem"))
        d.load_table(15, seed)
        print(f"tables loaded ({time.time()-t0:.1f}s)")

    # ---- THE test sim cannot do: did the load actually land in the fabric? ---
    print("\nTABLE READBACK (silicon memories vs the images we wrote):")
    ok = True
    ok &= cmp_table("consts (128w)", d.dbg_block(SEL_CONSTS, range(128)), consts)
    ok &= cmp_table("rsqrt seed (64w)", d.dbg_block(SEL_SEED, range(64)),
                    [s & 0xFFFFF for s in seed])
    n_esc = min(len(esc), 1024)
    ok &= cmp_table(f"emb scale ({n_esc}w)",
                    d.dbg_block(SEL_ESC, range(n_esc)),
                    [e & 0xFFFF for e in esc[:n_esc]])

    n = len(gw)
    if args.gw_samples and args.gw_samples < n:
        # stride sample + the first/last 32 (load-boundary effects show there)
        idx = sorted(set(list(range(32)) + list(range(n - 32, n)) +
                         list(np.linspace(0, n - 1, args.gw_samples,
                                          dtype=np.int64))))
    else:
        idx = list(range(n))
    got = d.dbg_block(SEL_GW, idx)
    want = [gw[i] for i in idx]
    ok &= cmp_table(f"gemv weights ({len(idx)} of {n} words)", got, want)
    print(f"TABLE_VERDICT: {'ALL MATCH' if ok else 'CORRUPT — the load path is '
                                                   'the bug, not the compute'}")
    if args.tables_only:
        return 0 if ok else 1

    # ---- run one token, full readback ---------------------------------------
    # release the shared weight-read port before computing: dbg_sel==9 steers
    # the gemv read pointer to the debug address, so leaving it selected would
    # feed the multiplier debug reads instead of weights.
    d.wr(R_DBGSEL, 0)

    toks = [int(t) for t in rd16(f"{args.dir}/ms_tok.mem")][:2]
    for i, tk in enumerate(toks):
        d.wr(R_TWADDR, i)
        d.wr(R_TWDATA, tk)
    d.wr(R_CTRL, 0b01)
    d.wait_status(0)
    cyc = d.rd(R_CYCLES)
    print(f"\nrun done: {cyc:,} cyc for {len(toks)} tokens")

    ref = np.load(f"{args.dir}/mp_ref.npz")
    for t in range(len(toks)):
        lg = d.dbg_block(SEL_LOGIT, range(t * V, (t + 1) * V))
        lg = np.where(lg >= 1 << 15, lg - (1 << 16), lg)
        rl = ref["ref_logits"][0][t]
        am_r, am_s = int(np.argmax(rl)), int(np.argmax(lg))
        dl = int(np.abs(lg - rl).max())
        print(f"  tok#{t} ({toks[t]}): argmax silicon {am_s} ref {am_r} "
              f"max|logitd|={dl} {'OK' if dl == 0 else 'DIFF'}")
        print(f"      silicon logits[:8] {lg[:8].tolist()}")
        print(f"      ref     logits[:8] {rl[:8].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
