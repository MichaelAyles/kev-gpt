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
NPORTS 3 and 4. Then a MUTATION check: wrapper bugs are injected into copies of the RTL and the
gate must fail on each. Finally the popped-tuples-per-cycle rate with an ideal slave.

REGISTER MAP VERSION 2 (aux DMA + argmax; `rtl/axi_aux_dma.v`, `gemv_argmax` in `rtl/gemv_axi.v`).
The version-1 scripts above are unchanged (MODE is never written, the aux slave has no memory, so
any aux access fails the sim) apart from reading ARGMAX / CYC_ACT / CYC_RES back; ARGMAX_IDX / VAL
are compared with numpy (lowest index of the IEEE maximum) in EVERY run. On top of them, every
case runs again with MODE bits: a memory-backed AXI4 read+write slave on aux_axi
(`tb/axi_rw_mem_slave.sv`; protocol checks: 4 KB rule, xVALID stability, WLAST position, RREADY /
BREADY never low, one burst at a time) holds the activation records (ignored bytes = junk) and
junk-filled result regions between junk guard pages, and afterwards

  * the whole aux memory must equal the initial image with exactly the result rows written
    (little endian, 4 per beat, last beat zero padded, ALL rows whatever RES_START says);
  * the aux bus must have carried exactly ceil(G/16) read bursts / 16 G beats (ACT_DMA) and
    ceil(R/4) write beats (RES_DMA), nothing in a run without the bit;
  * CYC_ACT == Ba - T - 4 and CYC_RES == Bw - L exactly (Ba = last R beat of the fetch, Bw = last
    BVALID), DONE rises after Bw; RES_NOCAP leaves the window as earlier runs wrote it;
  * a run with ACT_DMA = 0 after a fetch uses the fetched activation (q/k/v), with the ACT_*
    registers scribbled over in between.

MODE in {ACT, RES, ACT|RES, ACT|RES|NOCAP, RES|NOCAP} on the adversarial cases and the real rows
(8B: G = 32 and G = 96); planted argmax ties (`dma_tie`) and -0 / +0 (`argmax_unit`: gemv_argmax
alone, the engine never emits -0); a write slave slower than the engine (`dma_stall`: the result
FIFO back-pressures the core, STALL_OUT > 0, still bit-exact); RRESP / BRESP errors -> STATUS.aux_err
(`dma_err`); SOFT_RST during the fetch / the run / the write-back (`dma_abort`); the cycle cost with
ideal slaves (`dma_cost`, GEMV_AXI_DMA_COST lines).

    python -m fabric.big.run_gemv_axi                 # full gate -> GEMV_AXI_VERDICT
    python -m fabric.big.run_gemv_axi --quick         # adversarial subset, NPORTS 3, no mutations
    python -m fabric.big.run_gemv_axi --replay fabric/big/results/gemv_axi_golden.json

GOLDEN FILES (for unit-testing the board driver against the simulated top): the full gate
writes `results/gemv_axi_golden.json` (MODE 0) and `results/gemv_axi_dma_golden.json`
(MODE = ACT_DMA | RES_DMA) for one real case (Bonsai-1.7B l0.q, 64 rows, G = 16, NPORTS = 3,
CHUNK = 256, dense stripes) and immediately replays them. Schema
("gemv_axi_golden/2", all integers decimal unless the key says hex):

    schema, note, version, mode
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
    expect          {rows_hex: [R fp32 words], sum, xor, tuples, beats, xsum_hex: [per port],
                     argmax_idx, argmax_val_hex}
    measured_in_sim {cyc, cyc_stream, stall_in, stall_out, p_empty, p_fifo_max, cyc_act, cyc_res}
    MODE != 0 only:
    aux_base        byte address of aux_image_hex[0] on aux_axi
    aux_image_hex   the aux memory before the run: act_records(q, e) at act_addr, junk elsewhere
    aux_expect_hex  ... and after it: + res_image(rows) at res_addr
    act_addr, res_addr   what the ops write to ACT_ADDR / RES_ADDR

To test a driver: make it emit its op list for this matrix / activation / base and either
compare it with `ops`, or run it through the simulated top:

    from fabric.big.run_gemv_axi import simulate_ops
    out = simulate_ops(ops, image_bytes, image_base, nports=3)      # ops as above
    out.reads   # [(addr, data)] of every "r" / "poll" in order; out.ok, out.slave_errors
    out = simulate_ops(ops, image_bytes, image_base, nports=3, aux_image=aux_bytes, aux_base=aux_base)
    out.aux     # the aux memory after the script (uint8); act_records() / res_image() build the images
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
RTL_FILES = ("gemv_axi.v", "axi_rd_stream.v", "gemv_stream.v", "axi_aux_dma.v")
GOLDEN = HERE / "results" / "gemv_axi_golden.json"
GOLDEN_DMA = HERE / "results" / "gemv_axi_dma_golden.json"
GMAX, EXW = gs.GMAX, gs.EXW
M32 = 0xFFFFFFFF
RES_LG_SIM = 8                      # result window in most sims (256 rows); the RTL default is 14
MEM_LG = 18                         # beats of slave memory (4 MB)
AUX_LG = 15                         # beats of the aux slave's memory (512 KB)
AUX_BASE = 0x9_2000_0000            # where the aux slave's memory sits unless a caller says otherwise
ACT_DMA, RES_DMA, RES_NOCAP = 1, 2, 4      # MODE bits


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


def act_records(q, e, fill=None) -> np.ndarray:
    """MODE.ACT_DMA activation image: group g = one 256-byte record (gemv_axi_regs): bytes 0..159
    the 1280-bit mantissa word (little endian), byte 160 the int8 exponent. -> uint8 [G * 256].
    fill: uint8 [G, 256] for the bytes the fabric must ignore (default 0, what a driver writes)."""
    words = pack10(q)
    out = np.zeros((len(words), R.ACT_REC_BYTES), dtype=np.uint8) if fill is None else np.array(fill, dtype=np.uint8)
    for g, w in enumerate(words):
        out[g, :160] = np.array(w, dtype="<u4").view(np.uint8)
        assert -128 <= int(e[g]) <= 127, "the activation record holds an int8 exponent"
        out[g, 160] = int(e[g]) & 0xFF
    return out.reshape(-1)


def res_image(rows) -> np.ndarray:
    """MODE.RES_DMA result image: row r = little-endian fp32 word at 4r, zero padded to a 16-byte beat."""
    b = np.asarray(rows).astype("<u4").view(np.uint8)
    return np.concatenate([b, np.zeros((-b.size) % 16, dtype=np.uint8)])


def ops_mode(mode: int, act_addr: int, res_addr: int) -> list:
    """the version-2 part of a run's configuration."""
    return [("w", R.MODE, mode), ("w", R.ACT_ADDR_LO, act_addr & M32), ("w", R.ACT_ADDR_HI, act_addr >> 32),
            ("w", R.RES_ADDR_LO, res_addr & M32), ("w", R.RES_ADDR_HI, res_addr >> 32)]


def fits_i8(e) -> bool:
    return all(-128 <= int(x) <= 127 for x in e)


def argmax_ref(rows):
    """(index, raw word) of the maximum: IEEE order, -0 == +0, lowest index wins."""
    f = np.asarray(rows).astype(np.uint32).view(np.float32)
    assert not np.isnan(f).any()
    i = int(np.argmax(f))
    return i, int(np.asarray(rows)[i])


COUNTERS = (("cyc_lo", R.CYC_LO), ("cyc_hi", R.CYC_HI), ("str_lo", R.CYC_STREAM_LO), ("str_hi", R.CYC_STREAM_HI),
            ("rows", R.ROWS_DONE), ("sum_lo", R.SUM_LO), ("sum_hi", R.SUM_HI), ("xor", R.XOR),
            ("tuples", R.TUPLES), ("stall_in", R.STALL_IN), ("stall_out", R.STALL_OUT), ("err", R.ERRCNT),
            ("amax_idx", R.ARGMAX_IDX), ("amax_val", R.ARGMAX_VAL), ("cyc_act", R.CYC_ACT), ("cyc_res", R.CYC_RES))


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
    # ---- the aux slave (version 2)
    xargap: int = 0             # % of cycles ARREADY is withheld
    xrgap: int = 0              # % of cycles RVALID is withheld
    xlatmin: int = 20           # AR accept -> first beat
    xlatmax: int = 20
    awgap: int = 0              # % of cycles AWREADY is withheld
    wgap: int = 0               # % of cycles WREADY is withheld (99: a slave that stalls)
    blatmin: int = 2            # last W beat -> BVALID, cycles (0: the very next cycle)
    blatmax: int = 2
    rerr: int = -1              # aux R beat (counted from reset) that returns SLVERR
    berr: int = -1              # aux write burst (counted from reset) that returns SLVERR

    def tag(self):
        s = f"ar{self.argap}/r{self.rgap} lat{self.latmin}-{self.latmax}"
        if self.slowport >= 0:
            s += f" slow{self.slowport}@{self.slowgap}"
        if self.errport >= 0:
            s += f" err{self.errport}@{self.errbeat}"
        return s

    def xtag(self):
        s = (f"aux ar{self.xargap}/r{self.xrgap} lat{self.xlatmin}-{self.xlatmax} aw{self.awgap}/w{self.wgap} "
             f"b{self.blatmin}-{self.blatmax}")
        if self.rerr >= 0:
            s += f" rerr@{self.rerr}"
        if self.berr >= 0:
            s += f" berr@{self.berr}"
        return s


@dataclass
class SimOut:
    ok: bool                    # the testbench finished (no timeout) and the slaves saw no protocol error
    reads: list                 # [(addr, data | None)] for every "r" / "poll" op, in order (None: X)
    polls: list                 # [reads taken] per poll op
    marks: list                 # [dict(tag, A, B, T, L, n_ar, n_beats, Aa, Ba, Bw, x_ar, x_rbeats, x_aw, x_wbeats,
                                #        x_wzero, D)]  (tb_gemv_axi.sv: the aux_axi events of the run, DONE cycle)
    slave_errors: int
    cycles: int
    stdout: str
    aux: object = None          # uint8 array: the aux slave's memory after the script (None: no aux image)


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
    cmd.insert(2, f"-Ptb_gemv_axi.AUX_LG={AUX_LG}")
    cmd += [str(HERE / "tb" / "axi_rd_mem_slave.sv"), str(HERE / "tb" / "axi_rw_mem_slave.sv"),
            str(HERE / "tb" / "tb_gemv_axi.sv")]
    subprocess.run(cmd, check=True)
    _built[key] = out
    return out


OPC = {"w": 0, "r": 1, "poll": 2, "delay": 3, "mark": 4, "wq": 5}


def simulate_ops(ops, image, image_base: int, nports: int = 3, *, stim: Stim = Stim(), res_lg: int = 14,
                 fifo_lg: int = 11, rtl_dir=None, workdir=None, timeout_cycles: int = 0,
                 aux_image=None, aux_base: int = AUX_BASE) -> SimOut:
    """Replay a register script on the simulated gemv_axi top.
    ops: sequence of ("w", addr, data) | ("r", addr[, ...]) | ("poll", addr, mask, value) |
         ("delay", cycles) | ("mark", tag) | ("wq", addr, data)   (lists from JSON work too; extra
         fields ignored; "wq" = a write the testbench does not treat as the start of a run)
    image: bytes / uint8 array served by the read slaves at byte address image_base (4 KB aligned).
    aux_image: bytes / uint8 array = the memory behind aux_axi at byte address aux_base (4 KB aligned):
    activation records to fetch, room for the result rows. Its contents after the script come back
    in SimOut.aux. Without it any access of the aux master is a protocol error.
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
    (d / "auxout.mem").unlink(missing_ok=True)
    aux_args = []
    if aux_image is not None:
        aux_image = (np.frombuffer(bytes(aux_image), dtype=np.uint8) if not isinstance(aux_image, np.ndarray)
                     else aux_image)
        assert aux_image.size % 16 == 0 and aux_base % 4096 == 0 and aux_image.size // 16 <= 1 << AUX_LG
        with open(d / "auximage.mem", "w") as f:
            f.write("\n".join(bytes(b[::-1]).hex() for b in aux_image.reshape(-1, 16)) + "\n")
        aux_args = [f"+auximage={d / 'auximage.mem'}", f"+auxout={d / 'auxout.mem'}",
                    f"+auxbeats={aux_image.size // 16}"]
    if not timeout_cycles:
        timeout_cycles = 40 * len(ops) + 2_000_000
    cmd = [VVP, "-n", str(build(nports, res_lg, fifo_lg, rtl_dir)), f"+script={d / 'script.mem'}",
           f"+nops={len(ops)}", f"+out={d / 'out.txt'}", f"+image={d / 'image.mem'}",
           f"+membase={image_base:x}", f"+membeats={image.size // 16}", f"+argap={stim.argap}",
           f"+rgap={stim.rgap}", f"+latmin={stim.latmin}", f"+latmax={stim.latmax}", f"+seed={stim.seed}",
           f"+slowport={stim.slowport}", f"+slowgap={stim.slowgap}", f"+errport={stim.errport}",
           f"+errbeat={stim.errbeat}", f"+bfmrand={stim.bfmrand}", f"+timeout={3 * timeout_cycles}",
           f"+polltimeout={max(200, timeout_cycles // 15)}",     # a hung run: the poll gives up, the script reads on
           f"+auxbase={aux_base:x}", f"+auxargap={stim.xargap}", f"+auxrgap={stim.xrgap}",
           f"+auxlatmin={stim.xlatmin}", f"+auxlatmax={stim.xlatmax}", f"+awgap={stim.awgap}",
           f"+wgap={stim.wgap}", f"+blatmin={stim.blatmin}", f"+blatmax={stim.blatmax}",
           f"+rerrbeat={stim.rerr}", f"+berrburst={stim.berr}"] + aux_args
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
                marks.append(dict(tag=v[0], A=v[1], B=v[2], T=v[3], L=v[4], n_ar=v[5], n_beats=v[6], Aa=v[7], Ba=v[8],
                                  Bw=v[9], x_ar=v[10], x_rbeats=v[11], x_aw=v[12], x_wbeats=v[13], x_wzero=v[14],
                                  D=v[15]))
    aux_after = None
    if aux_image is not None and (d / "auxout.mem").exists():
        words = [ln.strip() for ln in (d / "auxout.mem").read_text().split("\n")
                 if ln.strip() and not ln.startswith(("//", "@"))]
        if len(words) == aux_image.size // 16 and not any("x" in w.lower() or "z" in w.lower() for w in words):
            aux_after = np.frombuffer(b"".join(bytes.fromhex(w)[::-1] for w in words), dtype=np.uint8).copy()
    serr = int(md.group(2)) if md else -1
    return SimOut(ok=bool(md) and serr == 0 and "AXI_TB_TIMEOUT" not in so, reads=reads, polls=polls,
                  marks=marks, slave_errors=serr, cycles=int(md.group(3)) if md else -1, stdout=so, aux=aux_after)


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
    mode: int = 0               # MODE bits (ACT_DMA | RES_DMA | RES_NOCAP). With ACT_DMA the activation comes
                                #   from the aux memory and the ACT_* registers are not written
    scribble: bool = False      # write junk into ACT_EXP / ACT_MANT WITHOUT committing before the run (a run
                                #   that re-uses the activation RAM must not see the registers)


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
             rtl_dir=None, seed=0, want=None, aux_base=AUX_BASE, act_fill="junk"):
    """One simulation: pack `runs` (gs.Run) into one image, execute `jobs` back to back.
    -> (ok, line, info). `want(job_index, got: dict, cfg, fails: list)` adds case-specific checks.

    A plan in which no job has a MODE bit is a version-1 script: MODE / ACT_ADDR / RES_ADDR are never
    written and the aux slave has no memory (any aux access is a protocol error). Otherwise every job
    writes them, and the aux memory is laid out page by page with junk guard pages in between: the
    activation records of each ACT_DMA matrix (ignored bytes = junk unless act_fill == "zero"), one
    junk-filled result region per RES_DMA job. The memory after the script must equal the initial
    image with exactly the result rows (+ zero padding of the last beat) written into those regions."""
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

    def rows_of(job):
        m = models[runs[job.mat].name]
        return np.asarray(m["ref"] if ref_domain else m["ind"]).astype(np.uint32)

    # ---- aux memory (version 2)
    v2 = any(job.mode for job in jobs)
    aux0 = aux_exp = None
    act_addr, res_addr, nowhere = {}, {}, 0
    if v2:
        arng = np.random.default_rng(7000 + seed)
        npages, act_page, res_page = 0, {}, {}

        def alloc(nbytes):
            nonlocal npages
            first = npages + 1                       # one guard page in front
            npages = first + -(-nbytes // 4096)
            return first

        for j, job in enumerate(jobs):
            if job.mode & ACT_DMA and job.mat not in act_page:
                act_page[job.mat] = alloc(index[job.mat].groups * R.ACT_REC_BYTES)
            if job.mode & RES_DMA:
                res_page[j] = alloc(index[job.mat].rows * 4)
        npages += 1
        assert npages * 256 <= 1 << AUX_LG, f"{name}: aux memory too small ({npages} pages)"
        aux0 = arng.integers(0, 256, npages * 4096, dtype=np.uint8)
        for mat, pg in act_page.items():
            m = models[runs[mat].name]
            G = index[mat].groups
            fill = None if act_fill == "zero" else aux0[pg * 4096:pg * 4096 + G * 256].reshape(G, 256)
            aux0[pg * 4096:pg * 4096 + G * 256] = act_records(m["q"], m["e"], fill)
            act_addr[mat] = aux_base + pg * 4096
        aux_exp = aux0.copy()
        for j, pg in res_page.items():
            ri = res_image(rows_of(jobs[j]))
            aux_exp[pg * 4096:pg * 4096 + ri.size] = ri
            res_addr[j] = aux_base + pg * 4096
        nowhere = aux_base + npages * 4096           # outside the memory: touching it is a protocol error

    ops, keys = [], []          # keys[i] = (job, key) for the i-th read/poll, in order

    def rd(addr, key, j=-1):
        ops.append(("r", addr))
        keys.append((j, key))

    def cfg_ops(j, job, cfg, res_start):
        o = ops_cfg(cfg, job.maxout, res_start)
        if v2:
            o += ops_mode(job.mode, act_addr[job.mat] if job.mode & ACT_DMA else nowhere,
                          res_addr[j] if job.mode & RES_DMA else nowhere)
        return o

    # ---- identity + reset state
    rd(R.IDCODE, "idcode")
    rd(R.PARAMS, "params")
    rd(R.STATUS, "status0")
    for a in (0x0FC, 0x240, 0x400, R.port_reg(nports, R.P_BEATS) if nports < 8 else 0x0F8, R.CTRL):
        rd(a, "zero")
    cfgs, loaded, nres_of = [], None, []
    for j, job in enumerate(jobs):
        r, e = runs[job.mat], index[job.mat]
        cfg = run_config(e, lay, image_base)
        cfgs.append(cfg)
        m = models[r.name]
        reg_load = job.load_act and not (job.mode & ACT_DMA)
        if job.soft_rst:
            ops += [("w", R.CTRL, 2), ("w", R.CTRL, 0)]
        if reg_load:
            ops += ops_act(m["q"], m["e"])
        if reg_load or job.mode & ACT_DMA:
            loaded = r.name
        assert loaded is not None
        if job.scribble:                             # registers changed, nothing committed
            ops += [("w", R.ACT_GROUP, 0), ("w", R.ACT_EXP, 0x55)]
            ops += [("w", R.ACT_MANT + 4 * i, (0x9E3779B9 * (i + 1 + j)) & M32) for i in range(R.ACT_MANT_WORDS)]
        if job.abort_after >= 0:                     # a run that is killed half way
            ops += cfg_ops(j, job, cfg, 0) + [("wq", R.CTRL, 1), ("delay", job.abort_after),
                                              ("w", R.CTRL, 2), ("w", R.CTRL, 0)]
            rd(R.STATUS, "status_abort", j)
            if job.abort_wait:
                ops.append(("poll", R.STATUS, 1, 0))
                keys.append((j, "poll_drain"))
        ops += cfg_ops(j, job, cfg, job.res_start)
        if j == 0:                                   # RW registers read back what was written
            rb = [("rows", R.ROWS), ("groups", R.GROUPS), ("pad", R.PAD), ("bursts", R.BURSTS),
                  ("chunk", R.CHUNK), ("step", R.STEP), ("res_start", R.RES_START),
                  ("base_lo", R.BASE_LO), ("base_hi", R.BASE_HI)]
            if reg_load and not job.scribble:
                rb += [("act_group", R.ACT_GROUP), ("act_exp", R.ACT_EXP), ("mant0", R.ACT_MANT),
                       ("mant39", R.ACT_MANT + 4 * 39)]
            if v2:
                rb += [("mode", R.MODE), ("act_addr_lo", R.ACT_ADDR_LO), ("act_addr_hi", R.ACT_ADDR_HI),
                       ("res_addr_lo", R.RES_ADDR_LO), ("res_addr_hi", R.RES_ADDR_HI)]
            for k, a in rb:
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
        # RES_NOCAP: the window must still hold what earlier runs left -> read its start back
        nres = min(res_words, 24) if job.mode & RES_NOCAP else max(0, min(e.rows - job.res_start, res_words))
        nres_of.append(nres)
        for op, key in ops_readback(nports, nres):
            ops.append(op)
            keys.append((j, key))

    d = SIM / f"{name}_n{nports}_c{chunk}_s{lay.slots}{'_v2' if v2 else ''}"
    beats_total = sum(c["bursts"] * c["chunk"] for c in cfgs)
    slow = 100 // max(1, 100 - max(stim.rgap, stim.slowgap if stim.slowport >= 0 else 0))
    budget = 16 * len(ops) + (4 + 3 * slow) * beats_total + (stim.latmax + 600) * 4 * len(jobs) + 30000
    if v2:                                           # the fetch, and a run that waits for its write slave
        pct = lambda g: 100 // max(1, 100 - g)
        for job in jobs:
            e = index[job.mat]
            n = 2 if job.abort_after >= 0 else 1
            if job.mode & ACT_DMA:
                budget += n * (2 * 16 * e.groups * pct(stim.xrgap) + (e.groups // 16 + 1) * (stim.xlatmax + 4 * pct(stim.xargap) + 20))
            if job.mode & RES_DMA:
                wb = -(-e.rows // 4)
                budget += n * (2 * wb * (2 + 2 * pct(stim.wgap)) + (wb // 16 + 1) * (stim.blatmax + 4 * pct(stim.awgap) + 20))
    so = simulate_ops(ops, mem, membase, nports, stim=stim, res_lg=res_lg, fifo_lg=fifo_lg, rtl_dir=rtl_dir,
                      workdir=d, timeout_cycles=budget, aux_image=aux0, aux_base=aux_base)
    fails = []
    if not so.ok:
        errs = sorted(set(re.findall(r"(?:SLAVE|AUX)_ERR ([^\n]*?) cyc=", so.stdout)))
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

    n_rows = n_ref = n_ind = n_ddr = n_ddr_ok = n_keep = 0
    info = dict(jobs=[])
    window = [None] * res_words                      # what the AXI-Lite window must hold (None: unknown)
    for j, job in enumerate(jobs):
        g, cfg, r, e = got[j], cfgs[j], runs[job.mat], index[job.mat]
        m = models[r.name]
        tag = f"job{j}({r.name}{',mode' + str(job.mode) if v2 else ''})"
        nres = nres_of[j]
        ref = rows_of(job).astype(np.uint64)
        if not job.mode & RES_NOCAP:                 # the model of the window, whatever the checks below say
            if job.abort_after >= 0:
                window = [None] * res_words          # the killed run (RES_START 0) may have written anything
            for i in range(nres):
                window[i] = int(ref[job.res_start + i])
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
                           base_lo=cfg["base"] & M32, base_hi=cfg["base"] >> 32)
            if "rb_act_group" in v:
                want_rb.update(act_group=e.groups - 1, act_exp=int(m["e"][e.groups - 1]) & M32,
                               mant0=pack10(m["q"])[-1][0], mant39=pack10(m["q"])[-1][39])
            if v2:
                aa = act_addr[job.mat] if job.mode & ACT_DMA else nowhere
                ra = res_addr[j] if job.mode & RES_DMA else nowhere
                want_rb.update(mode=job.mode, act_addr_lo=aa & M32, act_addr_hi=aa >> 32,
                               res_addr_lo=ra & M32, res_addr_hi=ra >> 32)
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
        if v["rows"] != e.rows:
            fails.append(f"{tag}: ROWS_DONE {v['rows']} != {e.rows}")
        s = v["sum_lo"] | (v["sum_hi"] << 32)
        if s != int(ref.sum()) or v["xor"] != int(np.bitwise_xor.reduce(ref)):
            fails.append(f"{tag}: SUM/XOR")
        if v["tuples"] != e.tuples:
            fails.append(f"{tag}: TUPLES {v['tuples']} != {e.tuples}")
        ai, av = argmax_ref(ref)                     # always on, any MODE
        if (v["amax_idx"], v["amax_val"]) != (ai, av):
            fails.append(f"{tag}: ARGMAX {v['amax_idx']} / {v['amax_val']:#010x} != {ai} / {av:#010x}")
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
        win = np.array([0xFFFF_FFFF_FFFF if x is None else x for x in (g.get(f"res{i}", [None])[0] for i in range(nres))],
                       dtype=np.uint64)
        if job.mode & RES_NOCAP:                     # nothing captured: the window still holds the older rows
            known = [i for i in range(nres) if window[i] is not None]
            n_keep += len(known)
            if not known:
                fails.append(f"{tag}: RES_NOCAP check is vacuous (no earlier capture in this plan)")
            if any(int(win[i]) != window[i] for i in known):
                fails.append(f"{tag}: RES_NOCAP but the result window changed")
            nres = 0
        else:
            n_rows += nres
            n_ref += int((win == m["ref"][job.res_start:job.res_start + nres].astype(np.uint64)).sum())
            n_ind += int((win == m["ind"][job.res_start:job.res_start + nres].astype(np.uint64)).sum())
            if not np.array_equal(win, ref[job.res_start:job.res_start + nres]):
                fails.append(f"{tag}: result window (RES_START {job.res_start}) differs from the "
                             f"{'spec' if ref_domain else 'independent model'}")
        # result rows in DDR (all rows, whatever RES_START says)
        if job.mode & RES_DMA:
            n_ddr += e.rows
            if so.aux is not None and so.aux.size == aux_exp.size:
                a0 = res_addr[j] - aux_base
                ddr = so.aux[a0:a0 + 4 * e.rows].view("<u4")
                n_ddr_ok += int((ddr == ref.astype(np.uint32)).sum())
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
            # aux master: exactly the fetch and the write-back of this MODE, nothing else
            wb = -(-e.rows // 4)
            x_want = ((-(-e.groups // 16), 16 * e.groups) if job.mode & ACT_DMA else (0, 0)) + \
                     ((-(-wb // 16), wb) if job.mode & RES_DMA else (0, 0))
            x_got = (mk["x_ar"], mk["x_rbeats"], mk["x_aw"], mk["x_wbeats"])
            if exact and (x_got != x_want or mk["x_wzero"] != 0):
                fails.append(f"{tag}: aux bus (ARs, R beats, AWs, W beats) {x_got} != {x_want}, "
                             f"{mk['x_wzero']} beats without strobes")
            if exact and v["cyc_act"] != (mk["Ba"] - mk["T"] - 4 if job.mode & ACT_DMA else 0):
                fails.append(f"{tag}: CYC_ACT {v['cyc_act']} != Ba-T-4 {mk['Ba'] - mk['T'] - 4}")
            if exact and v["cyc_res"] != (mk["Bw"] - mk["L"] if job.mode & RES_DMA else 0):
                fails.append(f"{tag}: CYC_RES {v['cyc_res']} != Bw-L {mk['Bw'] - mk['L']}")
            # DONE: after the last row, after the last write response (RES_DMA), and not later than
            # PAD + 30 cycles after the last bus event of the run (the masters must be idle and the
            # engine must have discarded the pad tuples first)
            last_ev = max(mk["L"], mk["B"], mk["Bw"] if job.mode & RES_DMA else -1)
            if job.mode & RES_DMA and not 0 <= mk["Bw"] < mk["D"]:
                fails.append(f"{tag}: DONE at {mk['D']} but the last write response came at {mk['Bw']}")
            if exact and not mk["L"] < mk["D"] <= last_ev + cfg["pad"] + 30:
                fails.append(f"{tag}: DONE at {mk['D']}; last row {mk['L']}, last R beat {mk['B']}, last B {mk['Bw']}")
        if want:
            want(j, v, cfg, fails)
        info["jobs"].append(dict(v, cyc=cyc, cyc_stream=cstr, cfg=cfg, nres=nres, mode=job.mode,
                                 done_after_last_row=(mk["D"] - mk["L"]) if mk else None))
    if v2:
        if so.aux is None or so.aux.size != aux_exp.size:
            fails.append("aux memory dump missing or X")
        else:
            bad = np.flatnonzero(so.aux != aux_exp)
            if bad.size:
                fails.append(f"aux memory: {bad.size} bytes differ from (initial image + result rows), first at "
                             f"page {int(bad[0]) // 4096} offset {int(bad[0]) % 4096:#x}")
    ok = not fails
    line = (f"{'PASS' if ok else 'FAIL'} {name:22s} n={nports} chunk{chunk:<3d} "
            f"{'dense' if not slots else 'slots' + str(lay.slots) + str(list(lay.port_slots)):16s} {stim.tag():24s} "
            f"jobs={len(jobs):2d} rows={n_rows:4d} rtl==ref {n_ref}/{n_rows}{'' if ref_domain else ' (info)'} "
            f"rtl==indep {n_ind}/{n_rows}"
            + (f" | v2 modes {''.join(str(job.mode) for job in jobs)} {stim.xtag()}: DDR rows {n_ddr_ok}/{n_ddr}, "
               f"window words kept {n_keep}" if v2 else "")
            + (f"  !! {'; '.join(fails[:4])}" + (f" (+{len(fails) - 4})" if len(fails) > 4 else "") if fails else ""))
    info.update(ok=ok, fails=fails, ops=ops, keys=keys, sim=so, mem=mem, membase=membase, image_base=image_base,
                index=index, lay=lay, cfgs=cfgs, n_ddr=n_ddr, aux0=aux0, aux_exp=aux_exp, aux_base=aux_base,
                act_addr=act_addr, res_addr=res_addr)
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


# ----------------------------------------------------------------------------- version 2 (aux DMA, argmax)
def dma_variants(nports, quick):
    """(chunk, slots, port_slots, stim, maxout) per MODE != 0 simulation of a case."""
    if nports == 3:
        v = [(16, 0, (), Stim(seed=71, xlatmin=1, xlatmax=1, blatmin=0, blatmax=0), 8),      # the tightest aux slave
             (64, 4, (0, 1, 3), Stim(30, 35, 5, 60, seed=73, bfmrand=1, xargap=30, xrgap=40, xlatmin=1, xlatmax=60,
                                     awgap=40, wgap=50, blatmin=0, blatmax=80), 3)]
        return v[1:] if quick else v
    return [(256, 0, (), Stim(10, 20, 3, 50, seed=75, bfmrand=1, xargap=50, xrgap=20, xlatmin=2, xlatmax=30,
                              awgap=20, wgap=30, blatmin=0, blatmax=20), 8)]


def dma_jobs(case, models, maxout=8):
    """every MODE on a case's matrices. A matrix whose exponents do not fit the record's int8 keeps the
    register load. Ends with the q/k/v pattern: fetch once, then runs with ACT_DMA = 0 that must use the
    fetched activation (the ACT_* registers are scribbled over, uncommitted, in between)."""
    same_x = len(case.runs) > 1 and all(np.array_equal(r.x, case.runs[0].x) for r in case.runs)
    jobs, last = [], None
    for i, r in enumerate(case.runs):
        mode = (ACT_DMA | RES_DMA, ACT_DMA, ACT_DMA | RES_DMA | RES_NOCAP, RES_DMA)[i % 4]
        if same_x and i:                             # one activation for all: fetched by the first run only
            jobs.append(Job(i, mode=(RES_DMA, RES_DMA | RES_NOCAP, 0)[(i - 1) % 3], load_act=False, maxout=maxout,
                            scribble=True))
            continue
        if not fits_i8(models[r.name]["e"]):
            mode &= ~ACT_DMA
        jobs.append(Job(i, mode=mode, maxout=maxout))
        if mode & ACT_DMA:
            last = i
    if last is not None:
        jobs += [Job(last, mode=ACT_DMA, maxout=maxout),
                 Job(last, mode=RES_DMA, load_act=False, scribble=True, maxout=maxout),
                 Job(last, mode=ACT_DMA | RES_DMA | RES_NOCAP, maxout=maxout),
                 Job(last, mode=0, load_act=False, scribble=True, maxout=maxout)]
    return jobs


def case_tie(rng):
    """planted argmax ties: the largest row is duplicated, the LOWEST index must be reported."""
    runs = []
    for name, G, Rr, first, dups in (("tie_later", 5, 30, 4, (11,)), ("tie_first0", 3, 40, 0, (39,)),
                                     ("tie_triple", 8, 24, 3, (9, 23)), ("tie_adjacent", 2, 12, 6, (7,))):
        r = gs.rand_run(rng, name, G, Rr)
        ref, _ = gs.ref_gemv(r.signs, r.scales, r.x)
        f = ref.view(np.float32)
        order = np.argsort(f, kind="stable")
        assert f[order[-1]] > f[order[-2]]           # a unique maximum to start from
        i = int(order[-1])
        for arr in (r.signs, r.scales):              # move the maximum to `first`, then copy it to the dups
            arr[[i, first]] = arr[[first, i]]
            for dd in dups:
                arr[dd] = arr[first]
        ref2, _ = gs.ref_gemv(r.signs, r.scales, r.x)
        f2 = ref2.view(np.float32)
        assert int(np.argmax(f2)) == first and int((f2 == f2.max()).sum()) == 1 + len(dups)
        runs.append(r)
    return gs.Case("dma_tie", runs, note="argmax ties: the maximum row duplicated at higher indices")


_am_built = None


def argmax_unit(rtl_dir=None):
    """gemv_argmax alone on directed vectors (the engine never emits -0). -> (ok, line)"""
    global _am_built
    SIM.mkdir(parents=True, exist_ok=True)
    d = Path(rtl_dir) if rtl_dir else HERE / "rtl"
    out = SIM / (f"tb_argmax_{hashlib.md5(str(rtl_dir).encode()).hexdigest()[:6]}.vvp" if rtl_dir else "tb_argmax.vvp")
    if rtl_dir or _am_built is None:
        subprocess.run([IVERILOG, "-g2012", "-s", "tb_gemv_argmax", "-o", str(out)] + [str(d / f) for f in RTL_FILES]
                       + [str(HERE / "tb" / "tb_gemv_argmax.sv")], check=True)
        if not rtl_dir:
            _am_built = out
    rng = np.random.default_rng(424)
    f32 = lambda *xs: np.array(xs, dtype=np.float32).view(np.uint32)
    NZ, PZ = 0x8000_0000, 0x0000_0000
    tests = [np.array(t, dtype=np.uint32) for t in (
        [NZ, PZ], [PZ, NZ], [NZ, NZ, PZ, NZ], [0xBF80_0000, NZ, PZ, NZ], [NZ], [PZ], [0xBF80_0000],
        [0xBF80_0000, 0xC000_0000, NZ], [0xC000_0000, 0xBF80_0000, 0xBF80_0000],          # all <= 0, ties
        [0x8000_0001, NZ, PZ, 0x0000_0001, 0x0000_0001],                                  # denormals around zero
        [0x0000_0001, 0x8000_0001, PZ], [0xFF7F_FFFF, 0xFF7F_FFFF], [0x7F7F_FFFF, 0xFF7F_FFFF, 0x7F7F_FFFF],
        [0xFF7F_FFFF, 0x8000_0001, 0xFF7F_FFFE], [0x3F80_0000] * 7, [NZ] * 5 + [PZ] * 3)]
    tests += [f32(*rng.standard_normal(n)) for n in (1, 2, 3, 17, 64, 200)]
    tests += [f32(*(-np.abs(rng.standard_normal(n)))) for n in (5, 40)]                    # all negative
    tests += [f32(*rng.integers(-3, 4, n).astype(np.float32)) for n in (30, 90, 90)]       # many ties, +0 only
    tests += [np.where(rng.random(60) < 0.5, f32(*rng.integers(-2, 1, 60).astype(np.float32)) | NZ,
                       f32(*rng.integers(-2, 1, 60).astype(np.float32))).astype(np.uint32)]  # ties among 0, -0, negatives
    tests += [rng.integers(0, 1 << 32, 300, dtype=np.uint64).astype(np.uint32) & np.uint32(0xFF7F_FFFF)
              for _ in range(3)]                                                          # any finite bit pattern
    words, want = [], []
    for k, t in enumerate(tests):
        for gaps in (0, 1):
            words += [t.size, gaps] + [int(x) for x in t]
            want.append(argmax_ref(t))
    (SIM / "argmax_in.mem").write_text("\n".join(f"{w:08x}" for w in words) + "\n")
    (SIM / "argmax_out.txt").unlink(missing_ok=True)
    so = subprocess.run([VVP, "-n", str(out), f"+in={SIM / 'argmax_in.mem'}", f"+out={SIM / 'argmax_out.txt'}",
                         f"+nw={len(words)}"], capture_output=True, text=True).stdout
    got = []
    if (SIM / "argmax_out.txt").exists():
        for ln in (SIM / "argmax_out.txt").read_text().split():
            got.append(ln)
    got = [(int(a), int(b, 16)) if "x" not in (a + b).lower() else None for a, b in zip(got[0::2], got[1::2])]
    nzero = sum(1 for t in tests if ((t & 0x7FFF_FFFF) == 0).any() and (t == NZ).any())
    good = sum(int(g == w) for g, w in zip(got, want))
    ok = "ARGMAX_TB_DONE" in so and len(got) == len(want) and good == len(want)
    return ok, (f"{'PASS' if ok else 'FAIL'} argmax_unit            gemv_argmax alone: {good}/{len(want)} vectors == numpy "
                f"(lowest index of the IEEE maximum; {nzero} vector sets with -0, 2 cycles after the last row, rows "
                f"back to back and with gaps)")


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
    # ---- version 2
    ("rec_stride", "axi_aux_dma.v", "f_addr    <= f_addr + 4096;", "f_addr    <= f_addr + 4096 + 256;",
     "activation records after the first burst are fetched one record (256 bytes) too far"),
    ("exp_byte", "axi_aux_dma.v", "wire signed [7:0] e8 = r1_d[7:0];", "wire signed [7:0] e8 = r1_d[15:8];",
     "the exponent is taken from byte 1 of beat 10"),
    ("exp_beat", "axi_aux_dma.v", "if (!f_drop && bcnt == 4'd10) begin", "if (!f_drop && bcnt == 4'd11) begin",
     "the exponent is taken from beat 11"),
    ("mant_order", "axi_aux_dma.v", "act_mant <= {r2_d, act_mant[1279:128]};", "act_mant <= {act_mant[1151:0], r2_d};",
     "the mantissa beats are assembled most significant first"),
    ("res_bytes", "axi_aux_dma.v", "2'd0: pk <= {96'd0, res_d};",
     "2'd0: pk <= {96'd0, res_d[7:0], res_d[15:8], res_d[23:16], res_d[31:24]};",
     "the first row of every beat is written big endian"),
    ("res_order", "axi_aux_dma.v", "2'd1: pk[63:32]  <= res_d;", "2'd1: pk[95:64]  <= res_d;",
     "rows 1 and 2 of a beat land in the same word"),
    ("res_pad", "axi_aux_dma.v", "2'd0: pk <= {96'd0, res_d};", "2'd0: pk[31:0] <= res_d;",
     "the last beat is not zero padded"),
    ("res_addr", "axi_aux_dma.v", "w_addr <= w_addr + {blen, 4'b0000};", "w_addr <= w_addr + {blen, 4'b0000} + 16;",
     "write bursts after the first land one beat too far"),
    ("res_start", "gemv_axi.v", ".res_v(c_v), .res_d(c_d),", ".res_v(c_v & skip_z), .res_d(c_d),",
     "RES_DMA skips the rows before RES_START"),
    ("nocap", "gemv_axi.v", " & ~clr_q & ~q_mode[2];", " & ~clr_q;", "RES_NOCAP still captures"),
    ("done_early", "gemv_axi.v", "& (~q_mode[1] | aux_res_done)};", "};",
     "DONE does not wait for the last write response"),
    ("fifo_room", "axi_aux_dma.v", "res_room <= ~res_en | (fill <= WFD - 4);", "res_room <= 1'b1;",
     "no back-pressure from the result FIFO"),
    ("argmax_tie", "gemv_axi.v", "if (k_v && (!have || k_key > best)) begin", "if (k_v && (!have || k_key >= best)) begin",
     "the highest index wins an argmax tie"),
    ("argmax_zero", "gemv_axi.v", "k_key <= d[31] ? (32'd0 - mag) : mag;", "k_key <= d[31] ? (32'd0 - mag - 32'd1) : mag;",
     "-0 orders below +0 (one's complement key)"),
    ("aux_err", "gemv_axi.v", "if (aux_err_evt) aux_err_st <= 1'b1;", "if (aux_err_evt) aux_err_st <= 1'b0;",
     "STATUS.aux_err never sets"),
    ("act_always", "gemv_axi.v", "act_go_r <= seq[7] & q_mode[0] & ~rst_h & ~srst_any;",
     "act_go_r <= seq[7] & ~rst_h & ~srst_any;", "the activation is fetched in every run, whatever MODE says"),
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
def write_golden(case, models, path: Path, mode: int = 0):
    """one real matrix, the driver's op sequence, expected read-back -> JSON (schema: module doc).
    mode != 0: the same run with MODE bits (the aux memory before / after goes into the file)."""
    nports, chunk, res_lg = 3, 256, 14
    ok, line, info = run_plan("golden_" + case.name, case.runs[:1], [Job(0, mode=mode)], models, nports, chunk,
                              stim=Stim(seed=1), res_lg=res_lg, membase=0x8_0000_0000, lead_stripes=2,
                              aux_base=0x8_7000_0000, act_fill="zero")
    if not ok:
        return False, line
    so, r, e, cfg = info["sim"], case.runs[0], info["index"][0], info["cfgs"][0]
    m = models[r.name]
    pinned = lambda key: key == "poll" or not (key in ("cyc_lo", "cyc_hi", "str_lo", "str_hi", "stall_in", "stall_out",
                                                       "status_run", "cyc_act", "cyc_res")
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
    ai, av = argmax_ref(m["ref"])
    doc = dict(
        schema="gemv_axi_golden/2",
        note="Bonsai-1.7B l0.q rows 100..163 on attn_norm(embed row); written and replayed by fabric/big/run_gemv_axi.py",
        version=R.VERSION, mode=mode,
        nports=nports, gmax=GMAX, res_lg=res_lg, fifo_lg=11,
        layout=dict(nports=nports, chunk=chunk, slots=info["lay"].slots, port_slots=list(info["lay"].port_slots), align="stripe"),
        matrix=e.as_dict(), image_base=info["membase"], image_hex=info["mem"].tobytes().hex(),
        base=cfg["base"], bursts=cfg["bursts"], maxout=8, res_start=0,
        activation=dict(q=np.asarray(m["q"]).reshape(-1, 128).tolist(), e=[int(x) for x in m["e"]]),
        ops=ops,
        expect=dict(rows_hex=[f"{int(x):08x}" for x in m["ref"]], sum=int(m["ref"].astype(np.uint64).sum()),
                    xor=int(np.bitwise_xor.reduce(m["ref"])), tuples=e.tuples, beats=cfg["bursts"] * chunk,
                    xsum_hex=[f"{sum(v[f'xsum{k}_{i}'] << (32 * i) for i in range(4)):032x}" for k in range(nports)],
                    argmax_idx=ai, argmax_val_hex=f"{av:08x}"),
        measured_in_sim=dict(cyc=v["cyc"], cyc_stream=v["cyc_stream"], stall_in=v["stall_in"], stall_out=v["stall_out"],
                             p_empty=[v[f"empty{k}"] for k in range(nports)], p_fifo_max=[v[f"fmax{k}"] for k in range(nports)],
                             cyc_act=v["cyc_act"], cyc_res=v["cyc_res"]))
    if mode:
        doc.update(aux_base=info["aux_base"], aux_image_hex=info["aux0"].tobytes().hex(),
                   aux_expect_hex=info["aux_exp"].tobytes().hex(),
                   act_addr=info["act_addr"].get(0), res_addr=info["res_addr"].get(0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return True, line


def replay(path: Path):
    """run a golden file's ops on the simulated top and check every pinned read (and, for a MODE != 0
    file, the aux memory afterwards). -> (ok, line)"""
    doc = json.loads(Path(path).read_text())
    assert doc["schema"] == "gemv_axi_golden/2"
    aux = bytes.fromhex(doc["aux_image_hex"]) if doc.get("mode") else None
    so = simulate_ops(doc["ops"], bytes.fromhex(doc["image_hex"]), doc["image_base"], doc["nports"],
                      res_lg=doc["res_lg"], fifo_lg=doc["fifo_lg"], workdir=SIM / "replay",
                      aux_image=aux, aux_base=doc.get("aux_base", AUX_BASE))
    want = [op for op in doc["ops"] if op[0] in ("r", "poll")]
    bad = [f"{op[1]:#x}: {got[1]} != {op[2]}" for op, got in zip(want, so.reads)
           if op[0] == "r" and op[2] is not None and got[1] != op[2]]
    rows = [got[1] for op, got in zip(want, so.reads) if op[0] == "r" and R.RES_BASE <= op[1]]
    nrows = len(doc["expect"]["rows_hex"])
    if doc.get("mode", 0) & RES_NOCAP:
        exact = nrows                                # nothing to read back through the window
    else:
        exact = sum(int(a == int(b, 16)) for a, b in zip(rows, doc["expect"]["rows_hex"]) if a is not None)
    ddr = ""
    aux_ok = True
    if aux is not None:
        aux_ok = so.aux is not None and so.aux.tobytes() == bytes.fromhex(doc["aux_expect_hex"])
        if doc["mode"] & RES_DMA and so.aux is not None:
            a0 = doc["res_addr"] - doc["aux_base"]
            got = so.aux[a0:a0 + 4 * nrows].view("<u4")
            ddr = (f", DDR rows {sum(int(int(x) == int(h, 16)) for x, h in zip(got, doc['expect']['rows_hex']))}/{nrows} "
                   f"bit-exact, aux memory {'==' if aux_ok else '!='} expected image")
    ok = so.ok and len(so.reads) == len(want) and not bad and exact == nrows and aux_ok
    return ok, (f"{'PASS' if ok else 'FAIL'} replay {Path(path).name}: MODE {doc.get('mode', 0)}, {len(doc['ops'])} ops, "
                f"{len(want)} reads, {len(want) - len(bad)} match, result rows {exact}/{nrows} bit-exact{ddr}"
                + (f"  !! {bad[:3]} slave_errors={so.slave_errors}" if not ok else ""))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="adversarial subset, NPORTS 3, no mutations / real / golden")
    ap.add_argument("--nports", type=int, nargs="*", default=None)
    ap.add_argument("--only", default="", help="substring filter on case names")
    ap.add_argument("--replay", default="", help="replay a golden JSON through the simulated top and exit")
    ap.add_argument("--golden", default=str(GOLDEN), help="where the full gate writes the golden file")
    ap.add_argument("--golden-dma", default=str(GOLDEN_DMA), help="where it writes the MODE = ACT_DMA|RES_DMA one")
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
    ok_all, n_sims, n_rows, n_ddr, n_v2 = True, 0, 0, 0, 0
    all_models = {}
    sel = lambda nm: not a.only or a.only in nm

    def emit(ok, line, info=None):
        nonlocal ok_all, n_sims, n_rows, n_ddr, n_v2
        ok_all &= ok
        n_sims += 1
        if info:
            n_rows += sum(j["nres"] for j in info["jobs"])
            n_ddr += info.get("n_ddr", 0)
            n_v2 += int(info.get("aux0") is not None)
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
        # version 2: the same matrices with the activation fetched from / the results written to "DDR"
        for n in sweep:
            for vi, (chunk, slots, ps, stim, maxout) in enumerate(dma_variants(n, a.quick)):
                jobs = dma_jobs(case, models, maxout)
                ok, line, info = run_plan(case.name, case.runs, jobs, models, n, chunk, slots=slots, port_slots=ps,
                                          stim=stim, ref_domain=case.ref_domain, seed=ci * 10 + vi + 500,
                                          membase=(0x8_4000_0000, 0x0_7FF0_0000)[vi % 2],
                                          aux_base=(AUX_BASE, 0x0_3F00_0000, 0x1F_FFF0_0000)[(ci + vi) % 3],
                                          act_fill=("junk", "zero")[ci % 2])
                emit(ok, line, info)
                if is_real and vi == 0 and info["jobs"]:
                    v = info["jobs"][0]
                    print(f"     {case.name} n={n}: G={v['cfg']['groups']} rows={v['cfg']['rows']}: CYC_ACT {v['cyc_act']} "
                          f"CYC_RES {v['cyc_res']} CYC {v['cyc']} ({stim.xtag()})")

    # ---- directed version-2 cases
    if sel("dma_tie"):
        tcase = case_tie(rng)
        tmodels, info_line, _ = gs.model_case(tcase)
        print(info_line)
        for n in sweep:
            jobs = [Job(i) for i in range(len(tcase.runs))]                       # version 1 script: argmax is always on
            ok, line, info = run_plan("dma_tie", tcase.runs, jobs, tmodels, n, 16, stim=Stim(seed=81))
            emit(ok, line, info)
            jobs = [Job(i, mode=(3, 5, 4, 6)[i % 4]) for i in range(len(tcase.runs))]    # incl. NOCAP without RES_DMA
            ok, line, info = run_plan("dma_tie", tcase.runs, jobs, tmodels, n, 16,
                                      stim=Stim(20, 20, 4, 40, seed=83, xrgap=30, wgap=40, blatmin=0, blatmax=30))
            emit(ok, line, info)
    if sel("argmax_unit"):
        ok, line = argmax_unit()
        emit(ok, line)

    if sel("dma_stall"):
        # a write slave slower than the engine: the result FIFO fills, the core is back-pressured through
        # m_ready (STALL_OUT counts it), and every row still lands, in the window and in DDR
        zcase = gs.case_random(rng, "dma_stall", [(1, 600), (2, 400), (32, 300), (96, 40)])
        zmodels, info_line, _ = gs.model_case(zcase)
        print(info_line)

        def stall_want(js):
            def want(j, v, cfg, fails):
                if j in js and v["stall_out"] == 0:
                    fails.append(f"job{j}: the write slave never back-pressured the engine (STALL_OUT 0): not exercised")
            return want

        for n, stim, js in ((3, Stim(seed=91, wgap=97, awgap=90, blatmin=0, blatmax=300), (0, 1)),
                            (3, Stim(seed=93, blatmin=1500, blatmax=3000, xrgap=20), (0, 1)),
                            (4, Stim(10, 10, 5, 30, seed=95, wgap=90, awgap=50, blatmin=200, blatmax=900, bfmrand=1), (0, 1))):
            if n not in sweep:
                continue
            jobs = [Job(0, mode=3), Job(1, mode=7), Job(2, mode=3), Job(3, mode=2), Job(0, mode=2, res_start=500)]
            ok, line, info = run_plan("dma_stall", zcase.runs, jobs, zmodels, n, 16, res_lg=7, stim=stim, want=stall_want(js))
            emit(ok, line, info)
            if info["jobs"]:
                print("     dma_stall: " + ", ".join(f"G={j['cfg']['groups']} rows={j['cfg']['rows']} STALL_OUT {j['stall_out']} "
                                                       f"CYC {j['cyc']} CYC_RES {j['cyc_res']}" for j in info["jobs"]))

    if sel("dma_err") and 3 in sweep:
        xcase = gs.case_random(rng, "dma_err", [(8, 20), (5, 300)])
        xmodels, info_line, _ = gs.model_case(xcase)
        print(info_line)
        # RRESP != OKAY on the 38th beat of the first fetch: aux_err, sticky until SOFT_RST; ERRCNT (weight ports) stays 0
        jobs = [Job(0, mode=3, exp_status=0x12), Job(1, mode=0, exp_status=0x12), Job(1, mode=3, exp_status=0x12),
                Job(0, mode=1, soft_rst=True), Job(1, mode=7)]
        ok, line, info = run_plan("dma_err_r", xcase.runs, jobs, xmodels, 3, 16,
                                  stim=Stim(10, 10, 5, 30, seed=101, bfmrand=1, xrgap=20, rerr=37))
        emit(ok, line, info)
        # BRESP != OKAY on the third write burst of the first run
        jobs = [Job(1, mode=3, exp_status=0x12), Job(0, mode=3, exp_status=0x12), Job(1, mode=2, soft_rst=True), Job(0)]
        ok, line, info = run_plan("dma_err_b", xcase.runs, jobs, xmodels, 3, 16,
                                  stim=Stim(10, 10, 5, 30, seed=103, bfmrand=1, wgap=20, blatmin=0, blatmax=40, berr=2))
        emit(ok, line, info)
        # both kinds of error in one run, together with an RRESP error on a weight port
        jobs = [Job(1, mode=3, exp_status=0x16, exp_err=1), Job(0, mode=3, soft_rst=True)]
        ok, line, info = run_plan("dma_err_all", xcase.runs, jobs, xmodels, 3, 16,
                                  stim=Stim(10, 10, 5, 30, seed=105, errport=1, errbeat=20, rerr=3, berr=1))
        emit(ok, line, info)

    if sel("dma_abort") and 3 in sweep:
        # SOFT_RST in the middle of the fetch / the run / the write-back, with aux bursts in flight
        bcase = gs.case_random(rng, "dma_abort", [(32, 60), (16, 130), (7, 25)])
        bmodels, info_line, _ = gs.model_case(bcase)
        print(info_line)
        for chunk, after, stim in ((16, 150, Stim(20, 30, 40, 120, seed=111, bfmrand=1, xargap=30, xrgap=50, xlatmin=20,
                                                  xlatmax=90, awgap=30, wgap=60, blatmin=10, blatmax=120)),
                                   (64, 420, Stim(seed=113, xlatmin=1, xlatmax=1, blatmin=0, blatmax=0)),
                                   (256, 700, Stim(10, 20, 10, 40, seed=115, bfmrand=1, xrgap=10, wgap=85, blatmin=50,
                                                   blatmax=400))):
            jobs = [Job(0, mode=3, go_twice=True), Job(1, mode=3, abort_after=after), Job(2, mode=3),
                    Job(0, mode=3, abort_after=5), Job(1, mode=7), Job(2, mode=3, abort_after=after + 77, abort_wait=True),
                    Job(0, mode=1), Job(1, mode=3, abort_after=after + 900, abort_wait=True),
                    Job(1, mode=2, abort_after=after + 1500), Job(2, mode=0, load_act=True),
                    Job(0, mode=3, abort_after=40), Job(0, mode=6, load_act=False, scribble=True)]
            ok, line, info = run_plan("dma_abort", bcase.runs, jobs, bmodels, 3, chunk, stim=stim)
            emit(ok, line, info)

    if sel("dma_cost"):
        # what the DMA costs with ideal slaves: the same run with MODE 0 / ACT_DMA / ACT+RES / ACT+RES+NOCAP
        ccase = gs.case_random(rng, "dma_cost", [(32, 64), (96, 64), (1, 256)])
        cmodels, info_line, _ = gs.model_case(ccase)
        for n in sweep:
            jobs = [Job(i, mode=md) for i in range(3) for md in (0, 1, 3, 7)]
            ok, line, info = run_plan("dma_cost", ccase.runs, jobs, cmodels, n, 256, stim=Stim(seed=121))
            js = info["jobs"]
            if len(js) == len(jobs):
                for i in range(3):
                    j0, j1, j3, j7 = js[4 * i:4 * i + 4]
                    if j1["cyc"] - j0["cyc"] != j1["cyc_act"] or j3["cyc"] - j0["cyc"] != j3["cyc_act"]:
                        ok = False
                        line += f"  !! G={j0['cfg']['groups']}: CYC(ACT_DMA) - CYC(MODE 0) != CYC_ACT"
                    if not (j3["stall_out"] == j7["stall_out"] == j0["stall_out"]):
                        ok = False
                        line += f"  !! G={j0['cfg']['groups']}: RES_DMA back-pressured the engine with an ideal slave"
            emit(ok, line.replace("FAIL", "PASS", 1) if ok else line.replace("PASS", "FAIL", 1), info)
            if len(js) == len(jobs):
                for i in range(3):
                    j0, j1, j3, j7 = js[4 * i:4 * i + 4]
                    print(f"GEMV_AXI_DMA_COST n={n} G={j0['cfg']['groups']} rows={j0['cfg']['rows']}: CYC {j0['cyc']} (MODE 0) -> "
                          f"{j1['cyc']} (ACT_DMA) = +CYC_ACT {j1['cyc_act']} for {16 * j0['cfg']['groups']} beats; "
                          f"RES_DMA: CYC {j3['cyc']}, CYC_RES {j3['cyc_res']}, DONE {j3['done_after_last_row']} cycles after the "
                          f"last row (MODE 0: {j0['done_after_last_row']}), STALL_OUT {j3['stall_out']} (MODE 0: {j0['stall_out']}) "
                          f"(MEASURED in sim, ideal slaves: xREADY=1, no gaps, read latency 20, B 2 cycles after the last W)")

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
    n_mut = 0
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
        # version 2: one MODE plan catches every data-path mutation; the others get their directed plan
        tcase2 = case_tie(np.random.default_rng(77))
        tmodels2, _, _ = gs.model_case(tcase2)
        v2jobs = [Job(1, mode=3, res_start=7), Job(0, mode=7), Job(0, mode=0, load_act=False, scribble=True),
                  Job(1, mode=1), Job(1, mode=2, load_act=False)]
        v2plan = lambda d: run_plan("mut", mcase.runs, v2jobs, mmodels, 3, 16, rtl_dir=d, res_lg=4,
                                    stim=Stim(seed=131, xrgap=20, wgap=20, blatmin=0, blatmax=10))
        for nm in ("rec_stride", "exp_byte", "exp_beat", "mant_order", "res_bytes", "res_order", "res_pad", "res_start",
                   "nocap", "act_always"):
            checks[nm] = v2plan
        zc = gs.case_random(np.random.default_rng(78), "mutz", [(1, 600), (5, 300)])
        zm, _, _ = gs.model_case(zc)
        slowplan = lambda d: run_plan("mutz", zc.runs, [Job(0, mode=3), Job(1, mode=3)], zm, 3, 16, rtl_dir=d,
                                      stim=Stim(seed=133, wgap=95, blatmin=100, blatmax=600))
        checks["res_addr"] = slowplan
        checks["fifo_room"] = slowplan
        checks["done_early"] = slowplan
        checks["argmax_tie"] = lambda d: run_plan("mutt", tcase2.runs, [Job(i) for i in range(len(tcase2.runs))],
                                                  tmodels2, 3, 16, rtl_dir=d)
        checks["argmax_zero"] = lambda d: argmax_unit(d) + (dict(jobs=[], fails=["gemv_argmax unit vectors"]),)
        checks["aux_err"] = lambda d: run_plan("mute", zc.runs, [Job(1, mode=3, exp_status=0x12)], zm, 3, 16, rtl_dir=d,
                                               stim=Stim(seed=135, berr=1))
        base = {}
        for name, _, _, _, what in MUTATIONS:
            if id(checks[name]) not in base:
                base[id(checks[name])] = checks[name](None)
            b_ok = base[id(checks[name])][0]
            m_ok, m_line, m_info = checks[name](mutate(name))
            caught = b_ok and not m_ok
            ok_all &= caught
            n_mut += 1
            st = [f"{j['status']:#x}" for j in m_info["jobs"]]
            print(f"{'PASS' if caught else 'FAIL'} mutation {name:11s} ({what}): unmutated {'PASS' if b_ok else 'FAIL'}, "
                  f"mutated {'FAIL (caught)' if not m_ok else 'PASS (NOT caught)'} STATUS {st}: "
                  f"{'; '.join(m_info['fails'][:2])[:230]}", flush=True)

    # ---- golden file for the board driver
    if not a.quick and not a.only:
        gcase = next((c for c in real if c.name == "real_1p7B_l0.q_G16"), None)
        if gcase is None:
            print("NOTE no 1.7B GGUF: golden file not written")
        else:
            for path, mode in ((Path(a.golden), 0), (Path(a.golden_dma), ACT_DMA | RES_DMA)):
                ok, line = write_golden(gcase, all_models[gcase.name], path, mode)
                emit(ok, line)
                if ok:
                    ok, line = replay(path)
                    emit(ok, line)
                    print(f"     wrote {path} ({path.stat().st_size} bytes)")

    print(f"GEMV_AXI_VERDICT {'PASS' if ok_all else 'FAIL'} ({n_sims} sims, {n_rows} result rows read back through AXI-Lite, "
          f"{n_v2} MODE != 0 sims with {n_ddr} result rows written to DDR by aux_axi and compared, {n_mut} mutations caught, "
          f"register map VERSION {R.VERSION}, NPORTS {sweep}, {time.time() - t_start:.0f} s)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
