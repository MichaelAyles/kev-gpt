"""Physically-contiguous memory for the weight image on the Kria, without a reboot.

The PL reads DDR by physical address, and a packed matrix must sit in ONE physically
contiguous region (its ports walk neighbouring addresses). Boot-time `reserved-memory` would be
the clean way; this gets the same thing from 2 MB hugepages at run time:

  1. grow the hugepage pool, map N pages of a hugetlbfs file, touch them; throw out the pages
     the kernel could later MOVE (the ones in CMA: see pin_longterm) and pin the rest
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
import json
import mmap
import os
import struct

import numpy as np

HP = 2 * 1024 * 1024
MAP_NORESERVE = 0x4000
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


def hp_counts() -> dict:
    """HugePages_{Total,Free,Rsvd,Surp} from /proc/meminfo."""
    out = {}
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("HugePages_"):
                k, v = ln.split(":")
                out[k[10:]] = int(v)
    return out


def hp_settle(release_free: bool = False) -> int:
    """Make nr_hugepages equal the real pool size and return it. Writing nr_hugepages below the
    number of pages in use does not free them, it marks them SURPLUS, and nr_hugepages then
    reads low; a later 'grow by N' would only relabel surplus pages and allocate nothing.
    release_free=True also gives the unused pages back to the kernel."""
    c = hp_counts()
    want = c["Total"] - (c["Free"] - c["Rsvd"] if release_free else 0)
    with open(NR, "w") as f:
        f.write(str(want))
    with open(NR) as f:
        return int(f.read())


def pin_longterm(ranges):
    """Long-term pin the (addr, length) ranges by registering them as io_uring fixed buffers.
    Returns the ring fd (the pin lasts until it is closed) or raises OSError.

    Why: hugepages are MOVABLE on arm64, so the kernel may take them from the CMA area, and any
    later CMA allocation (loading a bitstream through the FPGA manager is one) MIGRATES them to
    other physical pages; the PL would then read stale DDR. A FOLL_LONGTERM pin holds a page
    where it is. On this kernel (5.15) the pin is REFUSED (ENOMEM, one WARN in dmesg) for a
    hugepage that lies in CMA instead of moving it out, which HugePool uses as its test for
    'this page can be moved under us'."""
    libc = ctypes.CDLL(None, use_errno=True)
    step, iovs = 1 << 30, []                                  # at most 1 GB per registered buffer
    for addr, length in ranges:
        for o in range(0, length, step):
            iovs += [addr + o, min(step, length - o)]
    assert len(iovs) // 2 <= 1024, "too many ranges for one io_uring registration"
    params = ctypes.create_string_buffer(120)
    fd = libc.syscall(425, 4, params)                         # io_uring_setup
    if fd < 0:
        raise OSError(ctypes.get_errno(), "io_uring_setup")
    iov = (ctypes.c_size_t * len(iovs))(*iovs)
    if libc.syscall(427, fd, 0, iov, len(iovs) // 2) < 0:     # io_uring_register(REGISTER_BUFFERS)
        e = ctypes.get_errno()
        os.close(fd)
        raise OSError(e, "io_uring_register: " + os.strerror(e))
    return fd


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


def _ranges(idx):
    """sorted page indices -> [(first, count)] of consecutive runs."""
    out = []
    for i in idx:
        if out and i == out[-1][0] + out[-1][1]:
            out[-1][1] += 1
        else:
            out.append([i, 1])
    return out


class HugePool:
    """`pages` 2 MB hugepages that will stay where they are, in the hugetlbfs file `path`.

    The file is sparse: pages the kernel put in CMA are found by a trial pin, replaced, and
    punched out, so the pool's virtual page indices (`kept`) have gaps. `<path>` survives the
    process (the pages keep their physical addresses), `/run/<name>.pool.json` remembers which
    indices are populated, and both die with a reboot."""
    VEXTRA = 640                                         # virtual room for rejected pages

    @staticmethod
    def meta_path(path):
        return "/run/" + os.path.basename(path) + ".pool.json"

    @classmethod
    def reusable(cls, pages, path):
        try:
            with open(cls.meta_path(path)) as f:
                md = json.load(f)
            return md if md["pages"] == pages and os.path.getsize(path) == md["vsize"] else None
        except (OSError, ValueError, KeyError):
            return None

    def __init__(self, pages: int, path: str | None = None):
        self.temp = path is None
        self.path = path = path or f"/dev/hugepages/physmem_{os.getpid()}"
        self.req, self.rejected = pages, 0
        md = self.reusable(pages, path)
        self.reused = md is not None
        if not self.reused and os.path.exists(path):
            os.unlink(path)
        self.before = hp_settle(release_free=not self.reused)
        vpages = md["vsize"] // HP if md else pages + self.VEXTRA
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(self.fd, vpages * HP)
        self.mm = mmap.mmap(self.fd, vpages * HP, flags=mmap.MAP_SHARED | MAP_NORESERVE)
        self.buf = memoryview(self.mm)
        self._anchor = ctypes.c_char.from_buffer(self.mm)
        va0 = self.va0 = ctypes.addressof(self._anchor)
        if self.reused:
            kept = md["kept"]
            for i in kept:
                self.mm[i * HP]                          # fault in without modifying
        else:
            kept, bad, nxt = [], [], 0
            while len(kept) < pages and nxt < vpages:
                got = self._grow(min(pages - len(kept), vpages - nxt))
                if got == 0:
                    break
                idx = list(range(nxt, nxt + got))
                nxt += got
                for i in idx:
                    self.mm[i * HP] = 0
                good = self._pinnable(idx)               # rejected ones stay allocated for now, so
                gs = set(good)                           # the kernel cannot hand them out again
                kept += good
                bad += [i for i in idx if i not in gs]
            libc = ctypes.CDLL(None, use_errno=True)
            libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long]
            for i in bad:                                # PUNCH_HOLE | KEEP_SIZE
                if libc.fallocate(self.fd, 3, i * HP, HP) != 0:
                    raise OSError(ctypes.get_errno(), "fallocate(PUNCH_HOLE)")
            hp_settle(release_free=True)
            self.rejected = len(bad)
            if len(kept) < pages:
                print(f"NOTE asked for {pages} hugepages, got {len(kept)} usable ({len(bad)} rejected)")
            if not self.temp:
                with open(self.meta_path(path), "w") as f:
                    json.dump(dict(pages=pages, vsize=vpages * HP, kept=kept), f)
        self.kept, self.n = kept, len(kept)
        try:                                             # must come BEFORE the addresses are read
            self.pin_fd = pin_longterm([(va0 + i * HP, c * HP) for i, c in _ranges(kept)]) if kept else None
        except OSError as ex:
            self.pin_fd = None
            print(f"WARNING hugepages not pinned ({ex}): a CMA allocation can move them under the PL")
        pa = {i: _pfn(va0 + i * HP) * 4096 for i in kept}
        self.pa = pa
        self.runs = []                                  # [start_pa, [virtual page index, ...]]
        for i in sorted(kept, key=pa.get):
            if self.runs and pa[i] == self.runs[-1][0] + len(self.runs[-1][1]) * HP:
                self.runs[-1][1].append(i)
            else:
                self.runs.append([pa[i], [i]])
        self.free = [[r[0], list(r[1]), 0] for r in self.runs]      # [run pa, pages, bytes used]
        self.vpage = {p: i for i, p in pa.items()}

    @staticmethod
    def _grow(need: int) -> int:
        """raise nr_hugepages by `need`; -> how many free hugepages there are to touch."""
        c = hp_counts()
        with open(NR, "w") as f:
            f.write(str(c["Total"] + max(0, need - c["Free"])))
        return min(need, hp_counts()["Free"])

    def _pinnable(self, idx):
        """the pages of idx (consecutive virtual indices) a long-term pin accepts (bisection)."""
        if not idx:
            return []
        try:
            os.close(pin_longterm([(self.va0 + idx[0] * HP, len(idx) * HP)]))
            return list(idx)
        except OSError:
            if len(idx) == 1:
                return []
            h = len(idx) // 2
            return self._pinnable(idx[:h]) + self._pinnable(idx[h:])

    def region(self, pa: int, size: int) -> Region:
        """re-attach a region recorded by an earlier process (persistent pool): fails if the
        pages are no longer where the record says. Marks its bytes used in the allocator."""
        first, off0 = pa - pa % HP, pa % HP
        try:
            slots = [self.vpage[first + i * HP] for i in range(-(-(off0 + size) // HP))]
        except KeyError:
            raise RuntimeError(f"physical page for region {pa:#x}+{size} is not in the pool") from None
        for r in self.free:
            if r[0] <= pa < r[0] + len(r[1]) * HP:
                r[2] = max(r[2], pa - r[0] + size)
        return Region(self, pa, size, slots, off0)

    def alloc(self, size: int, align: int = 4096) -> Region:
        """bump-allocate inside a run, tightest sufficient run first (keeps the big runs for
        big matrices). Regions start `align`-aligned, so several small ones share a hugepage."""
        size = -(-size // 4096) * 4096
        for r in sorted(self.free, key=lambda r: len(r[1]) * HP - r[2]):
            start = -(-r[2] // align) * align
            if len(r[1]) * HP - start >= size:
                pg, off0 = divmod(start, HP)
                reg = Region(self, r[0] + start, size, r[1][pg:pg + -(-(off0 + size) // HP)], off0)
                r[2] = start + size
                return reg
        raise MemoryError(f"no contiguous {size / 1e6:.1f} MB left; largest free "
                          f"{max((len(r[1]) * HP - r[2] for r in self.free), default=0) / 1e6:.1f} MB")

    def va(self, reg: Region) -> int:
        """virtual address of a region that lies inside ONE hugepage (scratch buffers)."""
        assert len(reg.slots) == 1, "region spans hugepages: not virtually contiguous"
        return self.va0 + reg.slots[0] * HP + reg.off0

    def report(self) -> str:
        ln = sorted((len(r[1]) for r in self.runs), reverse=True)
        return (f"{self.n} hugepages{'' if self.pin_fd is None else ' (pinned)'}"
                f"{f', {self.rejected} movable ones rejected' if self.rejected else ''} ({self.n * HP / 1e6:.0f} MB) in {len(self.runs)} runs; "
                f"largest {ln[0] * HP / 1e6:.0f} MB; top runs (MB): {[x * 2 for x in ln[:8]]}")

    def close(self, free: bool = True):
        """free=False keeps a persistent pool (the file and its pages) for the next run."""
        if self.pin_fd is not None:
            os.close(self.pin_fd)
        self.buf.release()
        del self._anchor
        self.mm.close()
        os.close(self.fd)
        if free or self.temp:
            for f in (self.path, self.meta_path(self.path)):
                if os.path.exists(f):
                    os.unlink(f)
            hp_settle(release_free=True)


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
