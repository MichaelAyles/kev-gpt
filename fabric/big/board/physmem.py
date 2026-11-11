"""Physically-contiguous memory for the weight image on the Kria, without a reboot.

The PL reads DDR by physical address, and a packed matrix must sit in ONE physically
contiguous region (its ports walk neighbouring addresses). Boot-time `reserved-memory` would be
the clean way; this gets the same thing from 2 MB hugepages at run time:

  1. grow the hugepage pool, map N pages anonymously, touch them
  2. read each page's physical address from /proc/self/pagemap (root)
  3. sort by physical address and join neighbours into RUNS
  4. hand out regions first-fit from the runs (a region never spans a gap)

A region is written through `Region.write(offset, bytes)`, which maps the physical order back
to the (arbitrary) virtual order of the pages. With `path=` (a file on hugetlbfs, e.g.
/dev/hugepages/bigkev) the pool survives the process: the pages stay pinned at the same
physical addresses, so an image packed once can be re-used by later driver runs. After writing, call `evict()`: dirty lines must
leave the A53 caches before the PL reads the DDR (the PL does not see the CPU cache).

    sudo python3 -m fabric.big.board.physmem --pages 600      # report the runs, then free
"""
from __future__ import annotations

import argparse
import ctypes
import mmap
import os
import struct

import numpy as np

HP = 2 * 1024 * 1024
MAP_HUGETLB = 0x40000
NR = "/proc/sys/vm/nr_hugepages"


def _pfn(va: int) -> int:
    with open("/proc/self/pagemap", "rb") as f:
        f.seek((va // 4096) * 8)
        e = struct.unpack("<Q", f.read(8))[0]
    if not (e >> 63) & 1:
        raise RuntimeError("page not present")
    pfn = e & ((1 << 55) - 1)
    if pfn == 0 and os.geteuid() != 0:               # (as root PFN 0 is real: ZynqMP DDR starts at 0x0)
        raise RuntimeError("pagemap PFN hidden (need root)")
    return pfn


class Region:
    """`size` bytes at physical address `pa`, physically contiguous. `slots` = the virtual page
    index of each 2 MB page it touches, `off0` = offset of `pa` inside the first of them."""

    def __init__(self, pool, pa, size, slots, off0=0):
        self.pool, self.pa, self.size, self.slots, self.off0 = pool, pa, size, slots, off0

    def _spans(self, off, n):
        assert 0 <= off and off + n <= self.size, (off, n, self.size)
        i = 0
        while i < n:
            pg, po = divmod(self.off0 + off + i, HP)
            m = min(n - i, HP - po)
            yield i, self.slots[pg] * HP + po, m
            i += m

    def write(self, off: int, data) -> None:
        b = memoryview(data).cast("B")
        for i, v, m in self._spans(off, len(b)):
            self.pool.buf[v:v + m] = b[i:i + m]

    def read(self, off: int, n: int) -> bytes:
        out = bytearray(n)
        for i, v, m in self._spans(off, n):
            out[i:i + m] = self.pool.buf[v:v + m]
        return bytes(out)


class HugePool:
    def __init__(self, pages: int, path: str | None = None):
        """pages 2 MB hugepages; path = a hugetlbfs file to back them (persistent) or None."""
        self.path, self.reused = path, False
        with open(NR) as f:
            self.before = int(f.read())
        if path and os.path.exists(path) and os.path.getsize(path) == pages * HP:
            self.reused = True                           # pages already allocated and pinned
        else:
            if path and os.path.exists(path):
                os.unlink(path)
            with open(NR, "w") as f:
                f.write(str(self.before + pages))
            with open(NR) as f:
                got = int(f.read()) - self.before
            if got < pages:
                print(f"NOTE asked for {pages} hugepages, kernel gave {got}")
            pages = got
        self.n = pages
        if path:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(fd, self.n * HP)
            self.mm = mmap.mmap(fd, self.n * HP, flags=mmap.MAP_SHARED)
            os.close(fd)
        else:
            self.mm = mmap.mmap(-1, self.n * HP, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_HUGETLB)
        self.buf = memoryview(self.mm)
        va0 = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        pa = []
        for i in range(self.n):
            if self.reused:
                self.mm[i * HP]                          # fault in without modifying
            else:
                self.mm[i * HP] = 0
            pa.append(_pfn(va0 + i * HP) * 4096)
        self.pa = pa
        order = sorted(range(self.n), key=lambda i: pa[i])
        self.runs = []                                  # [start_pa, [virtual page index, ...]]
        for i in order:
            if self.runs and pa[i] == self.runs[-1][0] + len(self.runs[-1][1]) * HP:
                self.runs[-1][1].append(i)
            else:
                self.runs.append([pa[i], [i]])
        self.free = [[r[0], list(r[1]), 0] for r in self.runs]      # [run pa, pages, bytes used]
        self.vpage = {p: i for i, p in enumerate(pa)}

    def region(self, pa: int, size: int) -> Region:
        """re-attach a region recorded by an earlier process (persistent pool): fails if the
        pages are no longer where the record says."""
        first, off0 = pa - pa % HP, pa % HP
        try:
            slots = [self.vpage[first + i * HP] for i in range(-(-(off0 + size) // HP))]
        except KeyError:
            raise RuntimeError(f"physical page for region {pa:#x}+{size} is not in the pool") from None
        return Region(self, pa, size, slots, off0)

    def alloc(self, size: int, align: int = 4096) -> Region:
        """bump-allocate inside a run, tightest sufficient run first (keeps the big runs for
        big matrices). Regions are `align`-aligned, so several small ones share a hugepage."""
        size = -(-size // align) * align
        for r in sorted(self.free, key=lambda r: len(r[1]) * HP - r[2]):
            if len(r[1]) * HP - r[2] >= size:
                pg, off0 = divmod(r[2], HP)
                reg = Region(self, r[0] + r[2], size, r[1][pg:pg + -(-(off0 + size) // HP)], off0)
                r[2] += size
                return reg
        raise MemoryError(f"no contiguous {size / 1e6:.1f} MB left; largest free "
                          f"{max((len(r[1]) * HP - r[2] for r in self.free), default=0) / 1e6:.1f} MB")

    def report(self) -> str:
        ln = sorted((len(r[1]) for r in self.runs), reverse=True)
        return (f"{self.n} hugepages ({self.n * HP / 1e6:.0f} MB) in {len(self.runs)} runs; "
                f"largest {ln[0] * HP / 1e6:.0f} MB; top runs (MB): {[x * 2 for x in ln[:8]]}")

    def close(self, free: bool = True):
        """free=False keeps a persistent pool (the file and its pages) for the next run."""
        self.buf.release()
        self.mm.close()
        if free:
            if self.path and os.path.exists(self.path):
                os.unlink(self.path)
            with open(NR) as f:
                now = int(f.read())
            with open(NR, "w") as f:
                f.write(str(max(0, now - self.n)))


def evict(mb: int = 64, passes: int = 3) -> None:
    """Push dirty lines out to DDR by streaming a buffer much larger than L2 (1 MB)."""
    junk = np.ones(mb * 1024 * 1024 // 8, dtype=np.uint64)
    for _ in range(passes):
        junk += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=600)
    ap.add_argument("--path")
    a = ap.parse_args()
    p = HugePool(a.pages, a.path)
    print("PHYSMEM", p.report())
    r0 = p.alloc(3 * 4096)
    r = p.alloc(88 * 1024 * 1024)
    pat = bytes(range(256)) * 16
    r.write(HP - 2048, pat)
    assert r.read(HP - 2048, len(pat)) == pat and p.region(r.pa, r.size).read(HP - 2048, len(pat)) == pat
    print(f"PHYSMEM alloc 12 KB at pa={r0.pa:#x} + 88 MB at pa={r.pa:#x} ok, write/read across a page boundary ok")
    p.close()


if __name__ == "__main__":
    main()
