"""pl_bwprobe — R0 on silicon: how many GB/s can the PL actually pull from DDR?

Two jobs:
  --check   data honesty. Fill one 2 MB hugepage per port with random bytes from the
            A53, evict the caches (HP/HPC ports are non-coherent here), have each port
            read its page alone, and compare the 128-bit XOR checksum with numpy's.
  --sweep   bandwidth. Read big regions of the high DDR bank (0x8_0000_0000+, plain
            System RAM; reads are harmless) across port masks x burst x outstanding x
            ids, timing with the fabric's go->all-done cycle counter at the forced fclk.

On the Kria (root):
  sudo fpgautil -b ~/kevbit/bwprobe.bit.bin
  sudo python3 pl_bwprobe.py --fclk 300e6 --check --sweep --out r0_300.jsonl

fclk MUST be forced via /sys (fpgautil does not apply the BD's PL0 rate) and is
read back; GB/s are computed from the read-back rate.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import struct
import sys
import time

import numpy as np

BASE = 0xA000_0000
R_CTRL, R_STATUS, R_ENABLE = 0x00, 0x04, 0x08
R_BASE_LO, R_BASE_HI, R_STRIDE, R_BEATS = 0x0C, 0x10, 0x14, 0x18
R_BURST, R_MAXOUT, R_NIDS = 0x1C, 0x20, 0x24
R_GCYC_LO, R_GCYC_HI, R_IDCODE, R_NM, R_AXPARAM, R_STEP = 0x28, 0x2C, 0x30, 0x34, 0x38, 0x3C
IDCODE = 0x42575052  # "BWPR"
PORTS = ["HP0", "HP1", "HP2", "HP3", "HPC0", "HPC1"]
DDR_PEAK = 2133.33e6 * 8   # DDR4-2133 x 64-bit, read from DPLL/DDR_CTRL/MSTR on this board

MASKS = {
    "HP0": 0b000001,
    "HP0+HP3": 0b001001,
    "HP0+HP1": 0b000011,
    "HP1+HP2": 0b000110,
    "HP0+HP1+HP3": 0b001011,
    "HP0-3": 0b001111,
    "HPC0": 0b010000,
    "HPC0+HPC1": 0b110000,
    "all6": 0b111111,
}

FCLK = "/sys/devices/platform/fclk0/set_rate"


def set_fclk(hz: float) -> float:
    with open(FCLK, "w") as f:
        f.write(str(int(hz)))
    with open(FCLK) as f:
        got = float(f.read().strip())
    if abs(got - hz) / hz > 0.02:
        print(f"NOTE fclk0 asked {hz / 1e6:.1f} MHz, PLL gave {got / 1e6:.2f} MHz — using the read-back rate")
    return got


class Probe:
    def __init__(self):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.m = mmap.mmap(self.fd, 0x1000, offset=BASE)
        self.w32 = (ctypes.c_uint32 * 1024).from_buffer(self.m)
        idc = self.rd(R_IDCODE)
        if idc != IDCODE:
            sys.exit(f"IDCODE {idc:08x} != BWPR — is bwprobe.bit loaded?")
        self.nm = self.rd(R_NM)

    def wr(self, off, v):
        self.w32[off >> 2] = v & 0xFFFFFFFF

    def rd(self, off):
        return int(self.w32[off >> 2])

    def run(self, enable, base, stride, beats, burst, maxout, nids, qos=0, cache=3, step=0, timeout=30.0):
        assert beats % burst == 0 and burst & (burst - 1) == 0 and 1 <= burst <= 256
        assert base % (burst * 16) == 0 and stride % (burst * 16) == 0
        self.wr(R_CTRL, 2)  # soft reset
        self.wr(R_ENABLE, enable)
        self.wr(R_BASE_LO, base & 0xFFFFFFFF)
        self.wr(R_BASE_HI, base >> 32)
        self.wr(R_STRIDE, stride)
        self.wr(R_BEATS, beats)
        self.wr(R_BURST, burst)
        self.wr(R_MAXOUT, maxout)
        self.wr(R_NIDS, nids)
        self.wr(R_AXPARAM, (cache << 4) | qos)
        self.wr(R_STEP, step)
        t0 = time.perf_counter()
        self.wr(R_CTRL, 1)
        full = (1 << self.nm) - 1
        while True:
            st = self.rd(R_STATUS)
            if ((st & full) | (~enable & full)) == full and not (st >> 16) & 1:
                break
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError(f"STATUS={st:08x}")
        wall = time.perf_counter() - t0
        gcyc = self.rd(R_GCYC_LO) | (self.rd(R_GCYC_HI) << 32)
        per = []
        for i in range(self.nm):
            o = 0x100 + i * 0x20
            cyc = self.rd(o) | (self.rd(o + 4) << 32)
            xs = 0
            for k in range(4):
                xs |= self.rd(o + 16 + 4 * k) << (32 * k)
            per.append(dict(cyc=cyc, beats=self.rd(o + 8), err=self.rd(o + 12), xsum=xs))
        return gcyc, wall, per


# ---------------- data check ----------------
def virt_to_phys(addr: int) -> int:
    with open("/proc/self/pagemap", "rb") as f:
        f.seek((addr // 4096) * 8)
        e = struct.unpack("<Q", f.read(8))[0]
    if not (e >> 63) & 1:
        raise RuntimeError("page not present")
    pfn = e & ((1 << 55) - 1)
    if pfn == 0:
        raise RuntimeError("pagemap PFN hidden (need root)")
    return pfn * 4096 + (addr % 4096)


def data_check(p: Probe) -> bool:
    HP = 2 * 1024 * 1024
    with open("/proc/sys/vm/nr_hugepages", "w") as f:
        f.write("16")
    MAP_HUGETLB = 0x40000
    rng = np.random.default_rng(1234)
    pages = []
    for i in range(p.nm):
        mm = mmap.mmap(-1, HP, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_HUGETLB)
        a = np.frombuffer(mm, dtype=np.uint32)
        a[:] = rng.integers(0, 2**32, size=a.size, dtype=np.uint32)
        va = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        pa = virt_to_phys(va)
        exp = np.bitwise_xor.reduce(a.reshape(-1, 4), axis=0)
        expv = sum(int(exp[k]) << (32 * k) for k in range(4))
        pages.append((mm, pa, expv))
    # evict dirty lines to DDR: stream a buffer >> L2 (1 MB) through the cache
    junk = np.ones(64 * 1024 * 1024 // 8, dtype=np.uint64)
    for _ in range(2):
        junk += 1
    ok = True
    for i, (mm, pa, expv) in enumerate(pages):
        gcyc, wall, per = p.run(1 << i, pa, 0, HP // 16, 64, 8, 1)
        r = per[i]
        good = r["xsum"] == expv and r["beats"] == HP // 16 and r["err"] == 0
        ok &= good
        print(f"CHECK {PORTS[i]:5s} pa={pa:#011x} beats={r['beats']} err={r['err']} "
              f"xsum {'MATCH' if r['xsum'] == expv else 'MISMATCH'}")
    # interleaved: the active ports split ONE hugepage into burst-sized chunks,
    # port k taking chunks k, k+n, k+2n, ... (disjoint data, neighbouring addresses)
    mm, pa, _ = pages[0]
    beats_all = np.frombuffer(mm, dtype=np.uint32).reshape(-1, 4)
    for mask in (0b1111, 0b001011):
        act = [i for i in range(p.nm) if (mask >> i) & 1]
        n = len(act)
        for burst in (16, 64, 256):
            chunks = beats_all.reshape(-1, burst, 4)
            nper = (chunks.shape[0] // n) * burst
            # per-port base = pa + rank*burst*16; the probe bases master i at base + i*stride,
            # so run one port at a time with its own base to keep ranks contiguous
            for rank, i in enumerate(act):
                g, w, per = p.run(1 << i, pa + rank * burst * 16, 0, nper, burst, 8, 1,
                                  step=n * burst * 16)
                mine = chunks[rank::n][: nper // burst].reshape(-1, 4)
                e = np.bitwise_xor.reduce(mine, axis=0)
                ev = sum(int(e[k]) << (32 * k) for k in range(4))
                good = per[i]["xsum"] == ev and per[i]["beats"] == nper and per[i]["err"] == 0
                ok &= good
                if not good:
                    print(f"ILV_CHECK FAIL mask={mask:06b} burst={burst} port={PORTS[i]}")
        print(f"ILV_CHECK mask={mask:06b} bursts 16/64/256 {'MATCH' if ok else 'MISMATCH'}")
    print(f"R0_DATA_CHECK {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------- sweep ----------------
def sweep(p: Probe, fclk: float, out, mb_per_port: int, quick: bool):
    base = 0x8_0000_0000
    stride = 0x1000_0000  # 256 MB apart; 6 x 256 MB stays inside 0x8_0000_0000..0x8_6000_0000
    beats = mb_per_port * 1024 * 1024 // 16
    bursts = [16, 64, 256] if not quick else [16, 256]
    outs = [1, 2, 4, 8, 16, 32] if not quick else [4, 16]
    idss = [1, 4] if not quick else [1]
    best = None
    rows = []
    for mname, mask in MASKS.items():
        for burst in bursts:
            for mo in outs:
                for nids in idss:
                    gcyc, wall, per = p.run(mask, base, stride, beats, burst, mo, nids)
                    n = bin(mask).count("1")
                    secs = gcyc / fclk
                    gbs = n * beats * 16 / secs / 1e9
                    errs = sum(per[i]["err"] for i in range(p.nm) if (mask >> i) & 1)
                    beats_ok = all(per[i]["beats"] == beats for i in range(p.nm) if (mask >> i) & 1)
                    row = dict(fclk=fclk, ports=mname, mask=mask, burst=burst, maxout=mo, nids=nids,
                               beats=beats, gcyc=gcyc, wall_s=wall, gbs=gbs,
                               pct_peak=100 * gbs * 1e9 / DDR_PEAK, errs=errs, beats_ok=beats_ok,
                               per_port_gbs={PORTS[i]: per[i]["beats"] * 16 / (per[i]["cyc"] / fclk) / 1e9
                                             for i in range(p.nm) if (mask >> i) & 1 and per[i]["cyc"]})
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    if errs == 0 and beats_ok and (best is None or gbs > best["gbs"]):
                        best = row
                    print(f"{mname:12s} b{burst:<3d} o{mo:<2d} id{nids} {gbs:6.2f} GB/s "
                          f"({row['pct_peak']:4.1f}% peak){'' if errs == 0 and beats_ok else ' ERR'}")
    print("R0_BEST " + json.dumps(best))
    return rows


def ilv_sweep(p: Probe, fclk: float, out, mb_per_port: int, reps: int = 5):
    """Port-interleaved layout: disjoint data per port, neighbouring addresses."""
    import statistics as st
    base = 0x8_0000_0000
    beats = mb_per_port * 1024 * 1024 // 16
    best = None
    for mname in ("HP0-3", "HP0+HP1+HP3", "HP0+HP3", "all6"):
        mask = MASKS[mname]
        n = bin(mask).count("1")
        slots = mask.bit_length()
        for burst in (16, 64, 256):
            chunk = burst * 16
            for mo in (4, 8, 16, 32):
                r = []
                for _ in range(reps):
                    g, w, per = p.run(mask, base, chunk, beats, burst, mo, 1, step=slots * chunk)
                    bad = any(per[i]["err"] or per[i]["beats"] != beats
                              for i in range(p.nm) if (mask >> i) & 1)
                    r.append(float("nan") if bad else n * beats * 16 / (g / fclk) / 1e9)
                med = st.median(r)
                row = dict(fclk=fclk, layout="interleaved", ports=mname, burst=burst, maxout=mo,
                           step=slots * chunk, gbs=r, med=med, pct_peak=100 * med * 1e9 / DDR_PEAK)
                out.write(json.dumps(row) + "\n"); out.flush()
                if best is None or med > best["med"]:
                    best = row
                print(f"ILV {fclk / 1e6:3.0f} {mname:12s} b{burst:<3d} o{mo:<2d} "
                      f"min {min(r):5.2f} med {med:5.2f} max {max(r):5.2f} GB/s ({row['pct_peak']:4.1f}%)")
    print("R0_ILV_BEST " + json.dumps(best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fclk", type=float, default=300e6)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ilv", action="store_true", help="port-interleaved sweep (disjoint data)")
    ap.add_argument("--mb", type=int, default=128, help="MB read per port per run")
    ap.add_argument("--out", default="r0.jsonl")
    a = ap.parse_args()
    fclk = set_fclk(a.fclk)
    print(f"fclk0 = {fclk / 1e6:.2f} MHz; DDR peak (DDR4-2133 x64) = {DDR_PEAK / 1e9:.2f} GB/s")
    p = Probe()
    if a.check:
        data_check(p)
    if a.sweep:
        with open(a.out, "a") as f:
            sweep(p, fclk, f, a.mb, a.quick)
    if a.ilv:
        with open(a.out, "a") as f:
            ilv_sweep(p, fclk, f, a.mb)


if __name__ == "__main__":
    main()
