"""R0 gate: the bandwidth probe is correct, and fast enough not to be the bottleneck.

For each config: generate the RTL, compile tb_bwprobe with iverilog, run, and check
per master that
  * beats received == beats requested, zero RRESP errors, zero slave protocol errors
    (4 KB crossings, outstanding over the cap, bad size/burst);
  * the XOR checksum equals the Python recompute of the slave's address pattern
    (the same check the board driver makes against A53-written data);
and, for the zero-gap "saturation" configs, that the probe sustains >= 0.95
beats/cycle per master — i.e. on silicon the probe itself can't be the limit.

    python -m fabric.big.run_bwprobe            # full gate
    python -m fabric.big.run_bwprobe --quick
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = Path(os.environ.get("KEV_SIM_DIR", "/tmp/kevbuild")) / "big_r0_sim"
NM = 6
M32 = 0xFFFFFFFF


def pat(a: int) -> int:
    v = 0
    for j in range(4):
        lane = (((a + 4 * j) & M32) * 0x9E3779B1 & M32) ^ ((a >> 32) & 0x1FFFF)
        v |= lane << (32 * j)
    return v


def expected_xsum(base: int, beats: int, burst: int = 1, step: int = 0) -> int:
    x = 0
    for b in range(beats):
        k, j = divmod(b, burst)
        x ^= pat(base + k * (step or burst * 16) + 16 * j)
    return x


def build(lat: int, gap: int) -> Path:
    SIM.mkdir(parents=True, exist_ok=True)
    out = SIM / f"tb_l{lat}_g{gap}.vvp"
    srcs = [HERE / "rtl" / "axi_rd_probe.v", HERE / "rtl" / "bwprobe_axi.v",
            HERE / "tb" / "axi_rd_slave_model.sv", HERE / "tb" / "tb_bwprobe.sv"]
    cmd = ["iverilog", "-g2012", "-I", str(HERE / "tb"), f"-Ptb_bwprobe.LAT={lat}",
           f"-Ptb_bwprobe.GAP_PCT={gap}", "-o", str(out)] + [str(s) for s in srcs]
    subprocess.run(cmd, check=True)
    return out


def run_cfg(name, lat, gap, beats, burst, maxout, nids, enable=(1 << NM) - 1,
            base=0x8_0000_0000, stride=0x0100_0000, step=0, sat=False):
    vvp = build(lat, gap)
    args = ["vvp", "-n", str(vvp), f"+beats={beats}", f"+burst={burst}", f"+maxout={maxout}",
            f"+nids={nids}", f"+enable={enable}", f"+base={base:x}", f"+stride={stride:x}",
            f"+step={step:x}"]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    ok = True
    notes = []
    if "PROBE_TIMEOUT" in out or "IDCODE 42575052" not in out:
        print(out)
        return False, "timeout/idcode"
    for line in out.splitlines():
        if line.startswith("SLAVE_ERR"):
            notes.append(line)
        m = re.match(r"PROBE_M m=(\d+) cyc=(\d+) beats=(\d+) rresp_err=(\d+) slave_err=(\d+) xsum=([0-9a-f]+)", line)
        if not m:
            continue
        i, cyc, got, rerr, serr = (int(m.group(k)) for k in range(1, 6))
        xs = int(m.group(6), 16)
        if not (enable >> i) & 1:
            continue
        exp = expected_xsum(base + i * stride, beats, burst, step)
        eff = got / cyc if cyc else 0.0
        bad = []
        if got != beats: bad.append(f"beats {got}!={beats}")
        if rerr or serr: bad.append(f"errs r{rerr}/s{serr}")
        if xs != exp: bad.append("xsum MISMATCH")
        if sat and eff < 0.95: bad.append(f"eff {eff:.3f}<0.95")
        if bad:
            ok = False
            notes.append(f"m{i}: " + ", ".join(bad))
        else:
            notes.append(f"m{i}:{eff:.3f}b/c")
    return ok, " ".join(notes)


CONFIGS = [
    # name, lat, gap, beats, burst, maxout, nids, kwargs
    ("sat_b16_o8",   40, 0, 4096, 16, 8, 1, dict(sat=True)),
    ("sat_b256_o2",  40, 0, 4096, 256, 2, 4, dict(sat=True)),
    ("sat_b64_o4_id8", 60, 0, 4096, 64, 4, 8, dict(sat=True)),
    ("serial_b1_o1", 20, 0, 512, 1, 1, 1, {}),
    ("gaps_b16_o16", 40, 30, 2048, 16, 16, 4, {}),
    ("gaps_b256_o1", 40, 30, 2048, 256, 1, 1, {}),
    ("mask_3_ports", 40, 0, 1024, 32, 4, 2, dict(enable=0b101001)),
    ("ilv4_b16",     40, 0, 2048, 16, 8, 1, dict(enable=0b1111, stride=256, step=1024, sat=True)),
    ("ilv3_b256",    40, 10, 1024, 256, 4, 2, dict(enable=0b1011, stride=4096, step=12288)),
    ("hi_base_2G",   40, 0, 1024, 16, 8, 1, dict(base=0x8_7F00_0000, stride=0x0010_0000)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    subprocess.run([sys.executable, "-m", "fabric.big.gen_bwprobe_top"], check=True,
                   cwd=HERE.parent.parent, stdout=subprocess.DEVNULL)
    cfgs = CONFIGS[:3] if args.quick else CONFIGS
    allok = True
    for name, lat, gap, beats, burst, maxout, nids, kw in cfgs:
        ok, notes = run_cfg(name, lat, gap, beats, burst, maxout, nids, **kw)
        allok &= ok
        print(f"{'PASS' if ok else 'FAIL'} {name:16s} {notes}")
    print(f"BWPROBE_VERDICT {'PASS' if allok else 'FAIL'} ({len(cfgs)} configs)")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
