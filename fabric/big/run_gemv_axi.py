"""R2a board-wrapper gate: `rtl/gemv_axi.v` (AXI-Lite slave + AXI4 read masters + FIFOs +
`gemv_stream` + result capture) is bit-exact against the fabric spec THROUGH ITS REGISTERS.

Everything the simulation does to the design is a register script: the same sequence of
AXI-Lite writes / reads / polls the board driver (`board/pl_gemv.py`) issues, replayed by the
AXI-Lite BFM in `tb/tb_gemv_axi.sv`, while one memory-backed AXI4 read slave per master
(`tb/axi_rd_mem_slave.sv`) serves the packed weight image and checks the bus protocol
(4 KB rule, burst length, single ARID, ARVALID stability, in-flight cap, RREADY never low).

For each case: pack the matrices with `pack_weights.pack` (align = "stripe") behind a junk
lead-in (so no matrix sits at the start of the image), then run matrix after matrix WITHOUT a
reset in between -- load the activation, write the run configuration, GO, poll DONE, read the
result window and every counter back -- and compare

  * every fp32 result row bit-for-bit with `fabric_ref.ExactBfpGemv.mm` (the spec, CPU) and
    with the independent integer model of `run_gemv_stream.py`;
  * ROWS_DONE, SUM, XOR, TUPLES (= rows x tuples-per-row: DATA tuples), STATUS, ERRCNT;
  * per port P_BEATS (= BURSTS x CHUNK) and the 128-bit P_XSUM, recomputed from the image
    bytes at BASE + PORT_OFF[k] + n x STEP;
  * CYC == L - T and CYC_STREAM == B - A exactly, against the bus events the testbench
    observed (T = first BVALID cycle of the GO write, L = the core's last-row handshake,
    A = first AR handshake, B = last R beat), and the AR / beat counts seen by the slaves;
  * IDCODE / PARAMS, read-back of every RW register, unmapped addresses reading 0.

Cases: the adversarial generators of `run_gemv_stream.py`; real rows of Bonsai-1.7B (l0.q
G=16, l0.down G=48) and Bonsai-8B (attn_q G=32, ffn_down G=96); RES_START windowing on a
matrix taller than the (sim-shrunk) result window, re-run without reloading the activation;
one slow port (starvation: STALL_IN / P_EMPTY / FIFO credits); an RRESP error; SOFT_RST in
the middle of a run with bursts in flight (drain) followed by a GO issued during the drain;
GO while busy; CHUNK in {16, 64, 256}; dense stripes and R0's (4 slots, ports on 0, 1, 3);
NPORTS 3 and 4. Then a MUTATION check: four wrapper bugs are injected into copies of the RTL
and the gate must fail on each. Finally the popped-tuples-per-cycle rate with an ideal slave.

    python -m fabric.big.run_gemv_axi                 # full gate -> GEMV_AXI_VERDICT
    python -m fabric.big.run_gemv_axi --quick         # adversarial subset, NPORTS 3, no mutations
    python -m fabric.big.run_gemv_axi --replay fabric/big/results/gemv_axi_golden.json

GOLDEN FILE (for unit-testing the board driver against the simulated top): the full gate
writes `results/gemv_axi_golden.json` for one real case (Bonsai-1.7B l0.q, 64 rows, G = 16,
NPORTS = 3, CHUNK = 256, dense stripes) and immediately replays it. Schema
("gemv_axi_golden/1", all integers decimal unless the key says hex):

    schema, note
    nports, gmax, res_lg, fifo_lg     parameters of the simulated top
    layout          {nports, chunk, slots, port_slots, align}       (pack_weights.Layout)
    matrix          {name, rows, groups, tuple0, tuples, pad}       (pack_weights.Entry)
    image_base      byte address of image_hex[0] on the AXI bus (what the slaves serve)
    image_hex       the memory image, hex, byte 0 first
    base, bursts, maxout, res_start   the run configuration the ops write
    activation      {q: [G][128] BFP10 mantissas, e: [G] exponents}  (the spec's bfp_quant)
    ops             the register script, in order:
                      ["w", addr, data]          write
                      ["r", addr, expect]        read; expect = null when the value is
                                                 timing-dependent (see measured_in_sim)
                      ["poll", addr, mask, value] read until (data & mask) == value
    expect          {rows_hex: [R fp32 words], sum, xor, tuples, beats, xsum_hex: [per port]}
    measured_in_sim {cyc, cyc_stream, stall_in, stall_out, p_empty, p_fifo_max}  (ideal slave)

To test a driver: make it emit its op list for this matrix / activation / base and either
compare it with `ops`, or run it through the simulated top:

    from fabric.big.run_gemv_axi import simulate_ops
    out = simulate_ops(ops, image_bytes, image_base, nports=3)      # ops as above
    out.reads   # [(addr, data)] of every "r" / "poll" in order; out.ok, out.slave_errors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fabric.big import gemv_axi_regs as R
from fabric.big import pack_weights as pw
from fabric.big import run_gemv_stream as gs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SIM = Path(os.environ.get("KEV_SIM_DIR", "/tmp/kevbuild")) / "gemv_axi"
IVERILOG = os.environ.get("IVERILOG", str(Path.home() / ".local/bin/iverilog"))
VVP = os.environ.get("VVP", str(Path.home() / ".local/bin/vvp"))
RTL_FILES = ("gemv_axi.v", "axi_rd_stream.v", "gemv_stream.v")
GOLDEN = HERE / "results" / "gemv_axi_golden.json"
GMAX, EXW = gs.GMAX, gs.EXW
M32 = 0xFFFFFFFF
RES_LG_SIM = 8                      # result window in most sims (256 rows); the RTL default is 14
MEM_LG = 18                         # beats of slave memory (4 MB)


# ----------------------------------------------------------------------------- register script
def pack10(q) -> list:
    """BFP10 mantissas [G*128] -> per group 40 x 32-bit words (element j at bits [10j+9:10j])."""
    q = np.asarray(q).reshape(-1, 128)
    out = []
    for g in range(q.shape[0]):
        w = 0
        for j, v in enumerate(q[g]):
            w |= (int(v) & 0x3FF) << (10 * j)
        out.append([(w >> (32 * i)) & M32 for i in range(R.ACT_MANT_WORDS)])
    return out


def ops_act(q, e) -> list:
    """the driver's activation load: per group ACT_GROUP, ACT_EXP, 40 x ACT_MANT, ACT_COMMIT."""
    ops = []
    for g, words in enumerate(pack10(q)):
        ops.append(("w", R.ACT_GROUP, g))
        ops.append(("w", R.ACT_EXP, int(e[g]) & M32))
        ops += [("w", R.ACT_MANT + 4 * i, w) for i, w in enumerate(words)]
        ops.append(("w", R.ACT_COMMIT, 1))
    return ops


def run_config(entry, lay: pw.Layout, image_base: int) -> dict:
    """register values of one run, from the packer's layout (gemv_axi_regs docstring)."""
    e = entry if isinstance(entry, dict) else entry.as_dict()
    assert lay.align == "stripe" and e["tuple0"] % lay.chunk == 0 and (e["tuples"] + e["pad"]) % lay.chunk == 0
    return dict(rows=e["rows"], groups=e["groups"], pad=e["pad"], bursts=(e["tuples"] + e["pad"]) // lay.chunk,
                chunk=lay.chunk, step=lay.stripe_bytes,
                base=image_base + (e["tuple0"] // lay.chunk) * lay.stripe_bytes,
                port_off=[s * lay.chunk_bytes for s in lay.port_slots])


def ops_cfg(cfg: dict, maxout: int, res_start: int) -> list:
    """the driver's configuration writes, in pl_gemv.Gemv.run's order."""
    ops = [("w", R.ROWS, cfg["rows"]), ("w", R.GROUPS, cfg["groups"]), ("w", R.PAD, cfg["pad"]),
           ("w", R.BURSTS, cfg["bursts"]), ("w", R.CHUNK, cfg["chunk"]), ("w", R.STEP, cfg["step"]),
           ("w", R.MAXOUT, maxout), ("w", R.RES_START, res_start),
           ("w", R.BASE_LO, cfg["base"] & M32), ("w", R.BASE_HI, cfg["base"] >> 32)]
    ops += [("w", R.PORT_OFF + 4 * k, o) for k, o in enumerate(cfg["port_off"])]
    return ops


COUNTERS = (("cyc_lo", R.CYC_LO), ("cyc_hi", R.CYC_HI), ("str_lo", R.CYC_STREAM_LO), ("str_hi", R.CYC_STREAM_HI),
            ("rows", R.ROWS_DONE), ("sum_lo", R.SUM_LO), ("sum_hi", R.SUM_HI), ("xor", R.XOR),
            ("tuples", R.TUPLES), ("stall_in", R.STALL_IN), ("stall_out", R.STALL_OUT), ("err", R.ERRCNT))


def ops_readback(nports: int, nres: int) -> list:
    """STATUS, every counter, every per-port register, then nres words of the result window.
    -> [(op, key)]"""
    out = [(("r", R.STATUS), "status")] + [(("r", a), k) for k, a in COUNTERS]
    for p in range(nports):
        out += [(("r", R.port_reg(p, R.P_BEATS)), f"beats{p}"), (("r", R.port_reg(p, R.P_EMPTY)), f"empty{p}"),
                (("r", R.port_reg(p, R.P_FIFO_MAX)), f"fmax{p}")]
        out += [(("r", R.port_reg(p, R.P_XSUM + 4 * i)), f"xsum{p}_{i}") for i in range(4)]
    out += [(("r", R.RES_BASE + 4 * i), f"res{i}") for i in range(nres)]
    return out


# ----------------------------------------------------------------------------- simulation
@dataclass
class Stim:
    argap: int = 0              # % of cycles ARREADY is withheld
    rgap: int = 0               # % of cycles RVALID is withheld
    latmin: int = 20            # AR accept -> first beat, cycles
    latmax: int = 20
    seed: int = 1
    slowport: int = -1          # one port with its own R gap
    slowgap: int = 0
    errport: int = -1           # one SLVERR beat: port, beat index on that port
    errbeat: int = 0
    bfmrand: int = 0            # randomise the AXI-Lite master (AW/W skew, BREADY/RREADY delays)

    def tag(self):
        s = f"ar{self.argap}/r{self.rgap} lat{self.latmin}-{self.latmax}"
        if self.slowport >= 0:
            s += f" slow{self.slowport}@{self.slowgap}"
        if self.errport >= 0:
            s += f" err{self.errport}@{self.errbeat}"
        return s


@dataclass
class SimOut:
    ok: bool                    # the testbench finished (no timeout) and the slaves saw no protocol error
    reads: list                 # [(addr, data | None)] for every "r" / "poll" op, in order (None: X)
    polls: list                 # [reads taken] per poll op
    marks: list                 # [dict(tag, A, B, T, L, n_ar, n_beats)]
    slave_errors: int
    cycles: int
    stdout: str


_built = {}


def build(nports: int, res_lg: int, fifo_lg: int, rtl_dir: Path | None = None) -> Path:
    key = (nports, res_lg, fifo_lg, str(rtl_dir))
    if key in _built:
        return _built[key]
    SIM.mkdir(parents=True, exist_ok=True)
    d = Path(rtl_dir) if rtl_dir else HERE / "rtl"
    tagd = hashlib.md5(str(rtl_dir).encode()).hexdigest()[:6] if rtl_dir else "rtl"
    out = SIM / f"tb_n{nports}_r{res_lg}_f{fifo_lg}_{tagd}.vvp"
    cmd = [IVERILOG, "-g2012", f"-Ptb_gemv_axi.NPORTS={nports}", f"-Ptb_gemv_axi.GMAX={GMAX}",
           f"-Ptb_gemv_axi.RES_LG={res_lg}", f"-Ptb_gemv_axi.FIFO_LG={fifo_lg}",
           f"-Ptb_gemv_axi.MEM_LG={MEM_LG}", "-o", str(out)]
    cmd += [str(d / f) for f in RTL_FILES]
    cmd += [str(HERE / "tb" / "axi_rd_mem_slave.sv"), str(HERE / "tb" / "tb_gemv_axi.sv")]
    subprocess.run(cmd, check=True)
    _built[key] = out
    return out


OPC = {"w": 0, "r": 1, "poll": 2, "delay": 3, "mark": 4, "wq": 5}


def simulate_ops(ops, image, image_base: int, nports: int = 3, *, stim: Stim = Stim(), res_lg: int = 14,
                 fifo_lg: int = 11, rtl_dir=None, workdir=None, timeout_cycles: int = 0) -> SimOut:
    """Replay a register script on the simulated gemv_axi top.
    ops: sequence of ("w", addr, data) | ("r", addr[, ...]) | ("poll", addr, mask, value) |
         ("delay", cycles) | ("mark", tag) | ("wq", addr, data)   (lists from JSON work too; extra
         fields ignored; "wq" = a write the testbench does not treat as the start of a run)
    image: bytes / uint8 array served by the read slaves at byte address image_base (4 KB aligned).
    timeout_cycles: budget for the whole script; a poll gives up after about that many cycles (the
    script then carries on, so a hung run can still be read back) and the watchdog fires at 3x."""
    image = np.frombuffer(bytes(image), dtype=np.uint8) if not isinstance(image, np.ndarray) else image
    assert image.size % 16 == 0 and image_base % 4096 == 0 and image.size // 16 <= 1 << MEM_LG
    d = Path(workdir) if workdir else SIM / "adhoc"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "image.mem", "w") as f:
        f.write("\n".join(bytes(b[::-1]).hex() for b in image.reshape(-1, 16)) + "\n")
    lines = []
    for op in ops:
        k = op[0]
        a = [int(x) & M32 for x in op[1:4] if x is not None and not isinstance(x, str)]
        n = {"w": 2, "r": 1, "poll": 3, "delay": 1, "mark": 1, "wq": 2}[k]
        a = (a + [0, 0, 0])[:n] + [0] * (3 - n)
        lines.append(f"{OPC[k]:08x} {a[0]:08x} {a[1]:08x} {a[2]:08x}")
    (d / "script.mem").write_text("\n".join(lines) + "\n")
    (d / "out.txt").unlink(missing_ok=True)
    if not timeout_cycles:
        timeout_cycles = 40 * len(ops) + 2_000_000
    cmd = [VVP, "-n", str(build(nports, res_lg, fifo_lg, rtl_dir)), f"+script={d / 'script.mem'}",
           f"+nops={len(ops)}", f"+out={d / 'out.txt'}", f"+image={d / 'image.mem'}",
           f"+membase={image_base:x}", f"+membeats={image.size // 16}", f"+argap={stim.argap}",
           f"+rgap={stim.rgap}", f"+latmin={stim.latmin}", f"+latmax={stim.latmax}", f"+seed={stim.seed}",
           f"+slowport={stim.slowport}", f"+slowgap={stim.slowgap}", f"+errport={stim.errport}",
           f"+errbeat={stim.errbeat}", f"+bfmrand={stim.bfmrand}", f"+timeout={3 * timeout_cycles}",
           f"+polltimeout={max(200, timeout_cycles // 15)}"]     # a hung run: the poll gives up, the script reads on
    so = subprocess.run(cmd, capture_output=True, text=True).stdout
    md = re.search(r"AXI_TB_DONE ops=(\d+) slave_errors=(\d+) cycles=(\d+)", so)
    reads, polls, marks = [], [], []
    if (d / "out.txt").exists():
        for ln in (d / "out.txt").read_text().split("\n"):
            w = ln.split()
            if not w:
                continue
            if w[0] in ("R", "P"):
                reads.append((int(w[1], 16), None if "x" in w[2].lower() else int(w[2], 16)))
                if w[0] == "P":
                    polls.append(int(w[3]))
            elif w[0] == "M":
                v = [int(x) for x in w[1:]]
                marks.append(dict(tag=v[0], A=v[1], B=v[2], T=v[3], L=v[4], n_ar=v[5], n_beats=v[6]))
    serr = int(md.group(2)) if md else -1
    return SimOut(ok=bool(md) and serr == 0 and "AXI_TB_TIMEOUT" not in so, reads=reads, polls=polls,
                  marks=marks, slave_errors=serr, cycles=int(md.group(3)) if md else -1, stdout=so)


# ----------------------------------------------------------------------------- a plan = one sim
@dataclass
class Job:
    mat: int                    # index of the matrix in the packed image
    res_start: int = 0
    load_act: bool = True       # False: rely on the activation loaded by an earlier job
    maxout: int = 8
    go_twice: bool = False      # write GO again while busy (must be ignored)
    abort_after: int = -1       # >= 0: GO, wait this many cycles, SOFT_RST, then run normally
    abort_wait: bool = False    # after that SOFT_RST: True = poll STATUS.busy == 0 first (the documented
                                #   recipe); False = write the real GO during the drain (must be deferred)
    soft_rst: bool = False      # SOFT_RST (write 2, write 0) before this job
    exp_status: int = 2         # STATUS after DONE
    exp_err: int = 0            # ERRCNT


def port_beats(mem: np.ndarray, membase: int, cfg: dict, k: int) -> np.ndarray:
    """the beats port k reads in a run, straight from the address rule -> uint8 [beats, 16]."""
    cb = cfg["chunk"] * 16
    out = []
    for n in range(cfg["bursts"]):
        a = cfg["base"] + cfg["port_off"][k] + n * cfg["step"] - membase
        out.append(mem[a:a + cb])
    return np.concatenate(out).reshape(-1, 16)


def run_plan(name, runs, jobs, models, nports, chunk, *, slots=0, port_slots=(), stim=Stim(),
             res_lg=RES_LG_SIM, fifo_lg=11, ref_domain=True, membase=0x8_4000_0000, lead_stripes=3,
             rtl_dir=None, seed=0, want=None):
    """One simulation: pack `runs` (gs.Run) into one image, execute `jobs` back to back.
    -> (ok, line, info). `want(job_index, got: dict, cfg, fails: list)` adds case-specific checks."""
    slots, port_slots = (slots, port_slots) if slots and len(port_slots) == nports else (0, ())
    lay = pw.Layout(nports, chunk, slots, port_slots, "stripe")
    mats = [(f"{i}:{r.name}", (r.signs, r.scales)) for i, r in enumerate(runs)]
    img, index = pw.pack(mats, lay)
    rng = np.random.default_rng(1000 + seed)
    lead = rng.integers(0, 256, lead_stripes * lay.stripe_bytes, dtype=np.uint8)     # junk before the image
    tail = rng.integers(0, 256, lay.stripe_bytes, dtype=np.uint8)
    mem = np.concatenate([lead, img, tail])
    image_base = membase + lead.size
    res_words = 1 << res_lg

    ops, keys = [], []          # keys[i] = (job, key) for the i-th read/poll, in order

    def rd(addr, key, j=-1):
        ops.append(("r", addr))
        keys.append((j, key))

    # ---- identity + reset state
    rd(R.IDCODE, "idcode")
    rd(R.PARAMS, "params")
    rd(R.STATUS, "status0")
    for a in (0x0FC, 0x230, 0x400, R.port_reg(nports, R.P_BEATS) if nports < 8 else 0x0F8, R.CTRL):
        rd(a, "zero")
    cfgs, loaded = [], None
    for j, job in enumerate(jobs):
        r, e = runs[job.mat], index[job.mat]
        cfg = run_config(e, lay, image_base)
        cfgs.append(cfg)
        m = models[r.name]
        if job.soft_rst:
            ops += [("w", R.CTRL, 2), ("w", R.CTRL, 0)]
        if job.load_act:
            ops += ops_act(m["q"], m["e"])
            loaded = r.name
        assert loaded is not None
        if job.abort_after >= 0:                     # a run that is killed half way
            ops += ops_cfg(cfg, job.maxout, 0) + [("wq", R.CTRL, 1), ("delay", job.abort_after),
                                                  ("w", R.CTRL, 2), ("w", R.CTRL, 0)]
            rd(R.STATUS, "status_abort", j)
            if job.abort_wait:
                ops.append(("poll", R.STATUS, 1, 0))
                keys.append((j, "poll_drain"))
        ops += ops_cfg(cfg, job.maxout, job.res_start)
        if j == 0:                                   # RW registers read back what was written
            for k, a in (("rows", R.ROWS), ("groups", R.GROUPS), ("pad", R.PAD), ("bursts", R.BURSTS),
                         ("chunk", R.CHUNK), ("step", R.STEP), ("res_start", R.RES_START),
                         ("base_lo", R.BASE_LO), ("base_hi", R.BASE_HI), ("act_group", R.ACT_GROUP),
                         ("act_exp", R.ACT_EXP), ("mant0", R.ACT_MANT), ("mant39", R.ACT_MANT + 4 * 39)):
                rd(a, "rb_" + k, j)
            for k in range(8):
                rd(R.PORT_OFF + 4 * k, f"rb_off{k}", j)
        ops.append(("w", R.CTRL, 1))
        if job.go_twice:
            ops += [("wq", R.CTRL, 1), ("delay", 3), ("wq", R.CTRL, 1)]
        rd(R.STATUS, "status_run", j)
        ops.append(("poll", R.STATUS, 2, 2))
        keys.append((j, "poll"))
        ops.append(("mark", j))
        nres = max(0, min(e.rows - job.res_start, res_words))
        for op, key in ops_readback(nports, nres):
            ops.append(op)
            keys.append((j, key))

    d = SIM / f"{name}_n{nports}_c{chunk}_s{lay.slots}"
    beats_total = sum(c["bursts"] * c["chunk"] for c in cfgs)
    slow = 100 // max(1, 100 - max(stim.rgap, stim.slowgap if stim.slowport >= 0 else 0))
    so = simulate_ops(ops, mem, membase, nports, stim=stim, res_lg=res_lg, fifo_lg=fifo_lg, rtl_dir=rtl_dir,
                      workdir=d, timeout_cycles=16 * len(ops) + (4 + 3 * slow) * beats_total
                      + (stim.latmax + 600) * 4 * len(jobs) + 30000)
    fails = []
    if not so.ok:
        errs = sorted(set(re.findall(r"SLAVE_ERR ([^\n]*?) cyc=", so.stdout)))
        fails.append(f"sim: {'timeout' if 'TIMEOUT' in so.stdout else 'not finished' if so.cycles < 0 else ''} "
                     f"slave_errors={so.slave_errors} {errs[:3]}")
    got = [dict() for _ in jobs]
    glob = {}
    if len(so.reads) != len(keys):
        fails.append(f"read log {len(so.reads)}/{len(keys)}")
    for (j, key), (addr, val) in zip(keys, so.reads):
        (glob if j < 0 else got[j]).setdefault(key, []).append(val)

    def one(dct, key):
        v = dct.get(key, [None])
        return v[0]

    exp_params = (R.VERSION << 24) | (res_lg << 16) | (GMAX << 8) | nports
    if one(glob, "idcode") != R.IDCODE_VALUE or one(glob, "params") != exp_params or one(glob, "status0") != 0:
        fails.append(f"idcode/params/status0 {glob.get('idcode')} {glob.get('params')} {glob.get('status0')}")
    if any(v != 0 for v in glob.get("zero", [1])):
        fails.append(f"unmapped/WO registers do not read 0: {glob.get('zero')}")

    n_rows = n_ref = n_ind = 0
    info = dict(jobs=[])
    for j, job in enumerate(jobs):
        g, cfg, r, e = got[j], cfgs[j], runs[job.mat], index[job.mat]
        m = models[r.name]
        tag = f"job{j}({r.name})"
        if not g or "status" not in g:
            fails.append(f"{tag}: no readback")
            continue
        v = {k: one(g, k) for k in g}
        if any(x is None for k, x in v.items() if not k.startswith("res")):
            fails.append(f"{tag}: X in a register read")
            continue
        if j == 0:
            want_rb = dict(rows=cfg["rows"], groups=cfg["groups"], pad=cfg["pad"], bursts=cfg["bursts"],
                           chunk=cfg["chunk"], step=cfg["step"], res_start=job.res_start,
                           base_lo=cfg["base"] & M32, base_hi=cfg["base"] >> 32, act_group=e.groups - 1,
                           act_exp=int(m["e"][e.groups - 1]) & M32,
                           mant0=pack10(m["q"])[-1][0], mant39=pack10(m["q"])[-1][39])
            for k in range(8):
                want_rb[f"off{k}"] = cfg["port_off"][k] if k < nports else 0
            bad = [k for k, x in want_rb.items() if v.get("rb_" + k) != x]
            if bad:
                fails.append(f"{tag}: RW read-back {bad}")
        if job.abort_after >= 0 and (v.get("status_abort", 0) & 2):
            fails.append(f"{tag}: DONE set after SOFT_RST")
        if not v["status_run"] & 3:
            fails.append(f"{tag}: STATUS {v['status_run']:#x} right after GO (neither busy nor done)")
        if g["poll"][0] is None or (g["poll"][0] & 2) == 0:
            fails.append(f"{tag}: DONE never came (poll timed out)")
        if v["status"] != job.exp_status:
            fails.append(f"{tag}: STATUS {v['status']:#x} != {job.exp_status:#x}")
        if v["err"] != job.exp_err:
            fails.append(f"{tag}: ERRCNT {v['err']} != {job.exp_err}")
        ref = (m["ref"] if ref_domain else m["ind"]).astype(np.uint64)
        if v["rows"] != e.rows:
            fails.append(f"{tag}: ROWS_DONE {v['rows']} != {e.rows}")
        s = v["sum_lo"] | (v["sum_hi"] << 32)
        if s != int(ref.sum()) or v["xor"] != int(np.bitwise_xor.reduce(ref)):
            fails.append(f"{tag}: SUM/XOR")
        if v["tuples"] != e.tuples:
            fails.append(f"{tag}: TUPLES {v['tuples']} != {e.tuples}")
        nb = cfg["bursts"] * cfg["chunk"]
        for k in range(nports):
            pb = port_beats(mem, membase, cfg, k)
            xs = int.from_bytes(np.bitwise_xor.reduce(pb, axis=0).tobytes(), "little")
            gx = sum(v[f"xsum{k}_{i}"] << (32 * i) for i in range(4))
            if v[f"beats{k}"] != nb or gx != xs:
                fails.append(f"{tag}: port {k} BEATS {v[f'beats{k}']}/{nb} XSUM {'ok' if gx == xs else 'BAD'}")
            if v[f"fmax{k}"] > (1 << fifo_lg) or v[f"empty{k}"] > v["stall_in"]:
                fails.append(f"{tag}: port {k} FIFO_MAX {v[f'fmax{k}']} / P_EMPTY {v[f'empty{k}']} out of range")
        if v["stall_in"] > sum(v[f"empty{k}"] for k in range(nports)) or v["stall_in"] == 0:
            fails.append(f"{tag}: STALL_IN {v['stall_in']} inconsistent with P_EMPTY")
        # result window
        nres = max(0, min(e.rows - job.res_start, res_words))
        win = np.array([0xFFFF_FFFF_FFFF if x is None else x for x in (g.get(f"res{i}", [None])[0] for i in range(nres))],
                       dtype=np.uint64)
        n_rows += nres
        n_ref += int((win == m["ref"][job.res_start:job.res_start + nres].astype(np.uint64)).sum())
        n_ind += int((win == m["ind"][job.res_start:job.res_start + nres].astype(np.uint64)).sum())
        if not np.array_equal(win, ref[job.res_start:job.res_start + nres]):
            fails.append(f"{tag}: result window (RES_START {job.res_start}) differs from the "
                         f"{'spec' if ref_domain else 'independent model'}")
        # bus-side events
        cyc, cstr = v["cyc_lo"] | (v["cyc_hi"] << 32), v["str_lo"] | (v["str_hi"] << 32)
        mk = so.marks[j] if j < len(so.marks) else None
        if mk is None:
            fails.append(f"{tag}: no mark")
        else:
            exact = job.abort_after < 0 or job.abort_wait     # else: GO during the drain, bus log is mixed
            if exact and (mk["n_ar"] != nports * cfg["bursts"] or mk["n_beats"] != nports * nb):
                fails.append(f"{tag}: bus saw {mk['n_ar']} ARs / {mk['n_beats']} beats, want "
                             f"{nports * cfg['bursts']} / {nports * nb}")
            if exact and cstr != mk["B"] - mk["A"]:
                fails.append(f"{tag}: CYC_STREAM {cstr} != B-A {mk['B'] - mk['A']}")
            if exact and cyc != mk["L"] - mk["T"]:
                fails.append(f"{tag}: CYC {cyc} != L-T {mk['L'] - mk['T']}")
            if not exact and not (0 < cyc <= mk["L"] - mk["T"] and cstr > 0):
                fails.append(f"{tag}: CYC {cyc} / CYC_STREAM {cstr} vs L-T {mk['L'] - mk['T']} (deferred GO)")
        if want:
            want(j, v, cfg, fails)
        info["jobs"].append(dict(v, cyc=cyc, cyc_stream=cstr, cfg=cfg, nres=nres))
    ok = not fails
    line = (f"{'PASS' if ok else 'FAIL'} {name:22s} n={nports} chunk{chunk:<3d} "
            f"{'dense' if not slots else 'slots' + str(lay.slots) + str(list(lay.port_slots)):16s} {stim.tag():24s} "
            f"jobs={len(jobs):2d} rows={n_rows:4d} rtl==ref {n_ref}/{n_rows}{'' if ref_domain else ' (info)'} "
            f"rtl==indep {n_ind}/{n_rows}" + (f"  !! {'; '.join(fails[:4])}" + (f" (+{len(fails) - 4})" if len(fails) > 4 else "") if fails else ""))
    info.update(ok=ok, fails=fails, ops=ops, keys=keys, sim=so, mem=mem, membase=membase, image_base=image_base,
                index=index, lay=lay, cfgs=cfgs)
    return ok, line, info


# ----------------------------------------------------------------------------- special cases
def case_window(rng):
    return gs.Case("window", [gs.rand_run(rng, "tall", 5, 40), gs.rand_run(rng, "short", 3, 9)],
                   note="RES_START windowing, 16-row window, activation loaded once per matrix")


WINDOW_JOBS = [Job(0, 0), Job(0, 7, load_act=False), Job(0, 30, load_act=False), Job(1, 0),
               Job(0, 2), Job(0, 39, load_act=False), Job(0, 40, load_act=False), Job(0, 1000, load_act=False),
               Job(1, 8, load_act=True), Job(0, 24)]


def variants(nports, is_real, quick):
    """(chunk, slots, port_slots, stim, maxout) per simulation of a case."""
    if nports == 3:
        v = [(16, 0, (), Stim(seed=3), 8),
             (64, 4, (0, 1, 3), Stim(30, 35, 5, 60, seed=5, bfmrand=1), 3),
             (256, 0, (), Stim(60, 10, 1, 200, seed=7, bfmrand=1), 8),
             (256, 4, (0, 1, 3), Stim(0, 50, 30, 30, seed=9), 2)]
        return v[:2] if quick else (v[:3] if is_real else v)
    v = [(16, 0, (), Stim(25, 25, 3, 90, seed=11, bfmrand=1), 16),
         (64, 0, (), Stim(seed=13), 255),
         (256, 5, (0, 1, 3, 4), Stim(10, 40, 10, 40, seed=15, bfmrand=1), 1)]
    return v[:1] if is_real else v


MUTATIONS = [
    # (name, file, old, new, what)
    ("port_off", "gemv_axi.v", "{{(AW-32){1'b0}}, r_port_off[qk*32 +: 32]}", "{{(AW-32){1'b0}}, r_port_off[0 +: 32]}",
     "every port uses PORT_OFF[0]"),
    ("bursts+1", "axi_rd_stream.v", "bursts_left <= cfg_bursts;", "bursts_left <= cfg_bursts + 32'd1;",
     "one burst too many per port"),
    ("window-1", "gemv_axi.v", "skip_z <= (skip_left == 32'd1);", "skip_z <= (skip_left == 32'd2);",
     "capture window starts one row early"),
    ("credit-1", "axi_rd_stream.v", "(free >= chunk_w);", "((free + 1'b1) >= chunk_w);",
     "a burst is issued with one beat too little FIFO room"),
]


def mutate(name) -> Path:
    mname, fname, old, new, _ = next(m for m in MUTATIONS if m[0] == name)
    d = SIM / f"mut_{re.sub('[^a-z0-9]', '_', mname)}"
    d.mkdir(parents=True, exist_ok=True)
    for f in RTL_FILES:
        s = (HERE / "rtl" / f).read_text()
        if f == fname:
            assert s.count(old) == 1, f"mutation {mname}: pattern not found exactly once in {f}"
            s = s.replace(old, new)
        (d / f).write_text(s)
    return d


def starve_want(nports, slowport, fifo_lg, chunk):
    def want(j, v, cfg, fails):
        e_slow = v[f"empty{slowport}"]
        others = [v[f"empty{k}"] for k in range(nports) if k != slowport]
        fm = [v[f"fmax{k}"] for k in range(nports) if k != slowport]
        if cfg["bursts"] * cfg["chunk"] < 4 * (1 << fifo_lg):
            return
        if not (e_slow >= 0.9 * v["stall_in"] and max(others) <= 0.2 * v["stall_in"]):
            fails.append(f"job{j}: starvation not attributed: STALL_IN {v['stall_in']} P_EMPTY slow {e_slow} others {others}")
        if min(fm) < (1 << fifo_lg) - 2 * chunk:
            fails.append(f"job{j}: fast-port FIFOs never filled (FIFO_MAX {fm}): credits not exercised")
    return want


# ----------------------------------------------------------------------------- golden file
def write_golden(case, models, path: Path):
    """one real matrix, the driver's op sequence, expected read-back -> JSON (schema: module doc)."""
    nports, chunk, res_lg = 3, 256, 14
    ok, line, info = run_plan("golden_" + case.name, case.runs[:1], [Job(0)], models, nports, chunk,
                              stim=Stim(seed=1), res_lg=res_lg, membase=0x8_0000_0000, lead_stripes=2)
    if not ok:
        return False, line
    so, r, e, cfg = info["sim"], case.runs[0], info["index"][0], info["cfgs"][0]
    m = models[r.name]
    pinned = lambda key: key == "poll" or not (key in ("cyc_lo", "cyc_hi", "str_lo", "str_hi", "stall_in", "stall_out", "status_run")
                                               or key.startswith(("empty", "fmax")))
    ops, it = [], iter(zip(info["keys"], so.reads))
    for op in info["ops"]:
        if op[0] == "r":
            (j, key), (addr, val) = next(it)
            ops.append(["r", op[1], val if pinned(key) else None])
        elif op[0] == "poll":
            next(it)
            ops.append(["poll", op[1], op[2], op[3]])
        elif op[0] == "w":
            ops.append(["w", op[1], op[2]])
    v = info["jobs"][0]
    doc = dict(
        schema="gemv_axi_golden/1",
        note="Bonsai-1.7B l0.q rows 100..163 on attn_norm(embed row); written and replayed by fabric/big/run_gemv_axi.py",
        nports=nports, gmax=GMAX, res_lg=res_lg, fifo_lg=11,
        layout=dict(nports=nports, chunk=chunk, slots=info["lay"].slots, port_slots=list(info["lay"].port_slots), align="stripe"),
        matrix=e.as_dict(), image_base=info["membase"], image_hex=info["mem"].tobytes().hex(),
        base=cfg["base"], bursts=cfg["bursts"], maxout=8, res_start=0,
        activation=dict(q=np.asarray(m["q"]).reshape(-1, 128).tolist(), e=[int(x) for x in m["e"]]),
        ops=ops,
        expect=dict(rows_hex=[f"{int(x):08x}" for x in m["ref"]], sum=int(m["ref"].astype(np.uint64).sum()),
                    xor=int(np.bitwise_xor.reduce(m["ref"])), tuples=e.tuples, beats=cfg["bursts"] * chunk,
                    xsum_hex=[f"{sum(v[f'xsum{k}_{i}'] << (32 * i) for i in range(4)):032x}" for k in range(nports)]),
        measured_in_sim=dict(cyc=v["cyc"], cyc_stream=v["cyc_stream"], stall_in=v["stall_in"], stall_out=v["stall_out"],
                             p_empty=[v[f"empty{k}"] for k in range(nports)], p_fifo_max=[v[f"fmax{k}"] for k in range(nports)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return True, line


def replay(path: Path):
    """run a golden file's ops on the simulated top and check every pinned read. -> (ok, line)"""
    doc = json.loads(Path(path).read_text())
    assert doc["schema"] == "gemv_axi_golden/1"
    so = simulate_ops(doc["ops"], bytes.fromhex(doc["image_hex"]), doc["image_base"], doc["nports"],
                      res_lg=doc["res_lg"], fifo_lg=doc["fifo_lg"], workdir=SIM / "replay")
    want = [op for op in doc["ops"] if op[0] in ("r", "poll")]
    bad = [f"{op[1]:#x}: {got[1]} != {op[2]}" for op, got in zip(want, so.reads)
           if op[0] == "r" and op[2] is not None and got[1] != op[2]]
    rows = [got[1] for op, got in zip(want, so.reads) if op[0] == "r" and R.RES_BASE <= op[1]]
    exact = sum(int(a == int(b, 16)) for a, b in zip(rows, doc["expect"]["rows_hex"]) if a is not None)
    ok = so.ok and len(so.reads) == len(want) and not bad and exact == len(doc["expect"]["rows_hex"])
    return ok, (f"{'PASS' if ok else 'FAIL'} replay {Path(path).name}: {len(doc['ops'])} ops, {len(want)} reads, "
                f"{len(want) - len(bad)} match, result rows {exact}/{len(doc['expect']['rows_hex'])} bit-exact"
                + (f"  !! {bad[:3]} slave_errors={so.slave_errors}" if not ok else ""))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="adversarial subset, NPORTS 3, no mutations / real / golden")
    ap.add_argument("--nports", type=int, nargs="*", default=None)
    ap.add_argument("--only", default="", help="substring filter on case names")
    ap.add_argument("--replay", default="", help="replay a golden JSON through the simulated top and exit")
    ap.add_argument("--golden", default=str(GOLDEN), help="where the full gate writes the golden file")
    a = ap.parse_args()
    t_start = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if a.replay:
        ok, line = replay(Path(a.replay))
        print(line)
        print(f"GEMV_AXI_REPLAY_VERDICT {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    rng = np.random.default_rng(20261111)
    adv = [gs.case_maxinner(), gs.case_scales(rng), gs.case_negshift(rng), gs.case_bigshift(rng),
           gs.case_zero_rows(rng), gs.case_rne_ties(rng),
           gs.case_random(rng, "adv_random_small", [(1, 1), (1, 5), (2, 3), (7, 4), (8, 2), (9, 3), (15, 1), (17, 6), (1, 2)]),
           gs.case_random(rng, "adv_random_big", [(32, 6), (96, 4), (48, 3), (GMAX, 2), (95, 3), (16, 9)]),
           gs.case_ext_shift(rng),
           gs.case_random(rng, "adv_credits", [(1, 150), (2, 80), (8, 40), (1, 60)])]
    if a.quick:
        adv = [adv[0], adv[1], adv[6], adv[8], adv[9]]
    real = [] if a.quick else gs.cases_real()
    sweep = a.nports or ([3] if a.quick else [3, 4])
    ok_all, n_sims, n_rows = True, 0, 0
    all_models = {}

    def emit(ok, line, info=None):
        nonlocal ok_all, n_sims, n_rows
        ok_all &= ok
        n_sims += 1
        if info:
            n_rows += sum(j["nres"] for j in info["jobs"])
        print(line, flush=True)

    # ---- the cases x layouts x bus behaviours
    for ci, case in enumerate(adv + real):
        if a.only and a.only not in case.name:
            continue
        models, info_line, ind_ref = gs.model_case(case)
        all_models[case.name] = models
        print(info_line)
        if case.ref_domain and not ind_ref:
            ok_all = False
            print(f"FAIL {case.name}: independent model disagrees with ExactBfpGemv")
        is_real = case.name.startswith("real")
        for n in sweep:
            for vi, (chunk, slots, ps, stim, maxout) in enumerate(variants(n, is_real, a.quick)):
                order = list(range(len(case.runs)))
                if vi % 2:                      # every other sim runs the matrices out of image order
                    order = order[::-1]
                jobs = [Job(i, maxout=maxout) for i in order]
                ok, line, info = run_plan(case.name, case.runs, jobs, models, n, chunk, slots=slots, port_slots=ps,
                                          stim=stim, ref_domain=case.ref_domain, seed=ci * 10 + vi,
                                          membase=(0x8_4000_0000, 0x0_7FF0_0000, 0x10_0000_0000)[vi % 3])
                emit(ok, line, info)

    # ---- directed wrapper cases
    if not a.only or a.only in "window starve errbeat abort rate":
        wcase = case_window(rng)
        wmodels, info_line, _ = gs.model_case(wcase)
        print(info_line)
        for n in sweep:
            ok, line, info = run_plan("window", wcase.runs, WINDOW_JOBS, wmodels, n, 16, res_lg=4,
                                      stim=Stim(20, 20, 4, 40, seed=21, bfmrand=1))
            emit(ok, line, info)

        scase = gs.case_random(rng, "starve", [(32, 40), (9, 60), (96, 12)])
        smodels, info_line, _ = gs.model_case(scase)
        print(info_line)
        for n in sweep:
            sp = 1 if n == 3 else 3
            ok, line, info = run_plan("starve", scase.runs, [Job(i, maxout=16) for i in range(3)], smodels, n, 16,
                                      fifo_lg=8, stim=Stim(0, 0, 10, 10, seed=31, slowport=sp, slowgap=93),
                                      want=starve_want(n, sp, 8, 16))
            emit(ok, line, info)
            st = info["jobs"][0] if info["jobs"] else None
            if st:
                print(f"     starve n={n}: slow port {sp}: STALL_IN {st['stall_in']} P_EMPTY "
                      f"{[st[f'empty{k}'] for k in range(n)]} P_FIFO_MAX {[st[f'fmax{k}'] for k in range(n)]} "
                      f"(FIFO {1 << 8} beats) TUPLES {st['tuples']} CYC {st['cyc']}")

        ecase = gs.case_random(rng, "errbeat", [(8, 20), (5, 10)])
        emodels, info_line, _ = gs.model_case(ecase)
        print(info_line)
        jobs = [Job(0, exp_status=6, exp_err=1), Job(1, exp_status=6), Job(0, soft_rst=True), Job(1, load_act=True)]
        ok, line, info = run_plan("errbeat", ecase.runs, jobs, emodels, 3, 16,
                                  stim=Stim(10, 10, 5, 30, seed=41, errport=2, errbeat=37, bfmrand=1))
        emit(ok, line, info)

        acase = gs.case_random(rng, "abort", [(32, 60), (16, 30), (7, 25)])
        amodels, info_line, _ = gs.model_case(acase)
        print(info_line)
        for chunk, mo, after in ((16, 8, 150), (256, 8, 60), (64, 4, 400)):
            jobs = [Job(0, maxout=mo, go_twice=True), Job(1, maxout=mo, abort_after=after),
                    Job(2, maxout=mo), Job(0, maxout=mo, abort_after=5), Job(1, maxout=mo),
                    Job(2, maxout=mo, abort_after=after + 77, abort_wait=True), Job(0, maxout=mo)]
            ok, line, info = run_plan("abort", acase.runs, jobs, amodels, 3, chunk,
                                      stim=Stim(20, 30, 40, 120, seed=50 + chunk, bfmrand=1))
            emit(ok, line, info)

    # ---- rate with an ideal slave
    if not a.only or a.only in "rate":
        rcase = gs.case_random(rng, "rate", [(32, 512)] if not a.quick else [(32, 128)])
        rmodels, info_line, _ = gs.model_case(rcase)
        for n in sweep:
            for chunk in (256, 16):
                ok, line, info = run_plan("rate", rcase.runs, [Job(0)], rmodels, n, chunk, res_lg=10 if not a.quick else 8,
                                          stim=Stim(seed=61))
                emit(ok, line, info)
                if info["jobs"]:
                    v = info["jobs"][0]
                    act = v["tuples"] + v["cfg"]["pad"] + v["stall_in"] + v["stall_out"]
                    beats = v["cfg"]["bursts"] * chunk
                    print(f"GEMV_AXI_RATE n={n} chunk={chunk} maxout=8: {v['tuples'] + v['cfg']['pad']} tuples popped in "
                          f"{act} engine-active cycles = {(v['tuples'] + v['cfg']['pad']) / act:.4f} tuples/cycle "
                          f"(STALL_IN {v['stall_in']}, of which the wait for the first beat; STALL_OUT {v['stall_out']}); "
                          f"bus {beats} beats/port in CYC_STREAM {v['cyc_stream']} = {beats / v['cyc_stream']:.4f} beats/cycle/port; "
                          f"GO->last row CYC {v['cyc']} (MEASURED in sim, ideal slave: ARREADY=1, no R gaps, latency 20)")

    # ---- mutations: the gate must fail on each injected wrapper bug
    if not a.quick and not a.only:
        mcase = gs.case_random(rng, "mut", [(9, 30), (32, 40)])
        mmodels, _, _ = gs.model_case(mcase)
        checks = {
            "port_off": lambda d: run_plan("mut", mcase.runs, [Job(0), Job(1)], mmodels, 3, 16, rtl_dir=d),
            "bursts+1": lambda d: run_plan("mut", mcase.runs, [Job(0), Job(1)], mmodels, 3, 16, rtl_dir=d),
            "window-1": lambda d: run_plan("mut", mcase.runs, [Job(0, 5), Job(1, 11)], mmodels, 3, 16, res_lg=4, rtl_dir=d),
            "credit-1": lambda d: run_plan("mut", mcase.runs, [Job(1, maxout=16)], mmodels, 3, 16, fifo_lg=8, rtl_dir=d,
                                           stim=Stim(0, 0, 10, 10, seed=31, slowport=1, slowgap=97)),
        }
        base = {k: f(None) for k, f in checks.items()}
        for name, _, _, _, what in MUTATIONS:
            b_ok = base[name][0]
            m_ok, m_line, m_info = checks[name](mutate(name))
            caught = b_ok and not m_ok
            ok_all &= caught
            st = [f"{j['status']:#x}" for j in m_info["jobs"]]
            print(f"{'PASS' if caught else 'FAIL'} mutation {name:9s} ({what}): unmutated {'PASS' if b_ok else 'FAIL'}, "
                  f"mutated {'FAIL (caught)' if not m_ok else 'PASS (NOT caught)'} STATUS {st}: "
                  f"{'; '.join(m_info['fails'][:2])[:230]}", flush=True)

    # ---- golden file for the board driver
    if not a.quick and not a.only:
        gcase = next((c for c in real if c.name == "real_1p7B_l0.q_G16"), None)
        if gcase is None:
            print("NOTE no 1.7B GGUF: golden file not written")
        else:
            ok, line = write_golden(gcase, all_models[gcase.name], Path(a.golden))
            emit(ok, line)
            if ok:
                ok, line = replay(Path(a.golden))
                emit(ok, line)
                print(f"     wrote {a.golden} ({Path(a.golden).stat().st_size} bytes)")

    print(f"GEMV_AXI_VERDICT {'PASS' if ok_all else 'FAIL'} ({n_sims} sims, {n_rows} result rows read back through AXI-Lite, "
          f"NPORTS {sweep}, {time.time() - t_start:.0f} s)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
