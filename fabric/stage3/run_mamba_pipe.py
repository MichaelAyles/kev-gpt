"""run_mamba_pipe — the PIPELINED multi-stream mamba engine gate (Phase 2 Rung C).

Drives the wave-scheduled NC-stream engine (fabric/stage3/rtl/mamba_pipe.sv) and
compares EVERY stream's per-token logits + final residual, bit-exactly, against
the single-stream hardware oracle fabric/stage3/mamba_seq_ref.MambaSeqRef.step().
The wave schedule changes only the TIMING (which stream occupies which sub-core
each cycle), never the arithmetic, so the bar is bit-exact (0 max|diff|), and the
harness also prints the MEASURED per-stream cyc/token (total cyc / (NC*T)) — the
number the mamba_pipe_ref wave model predicts.

    python -m fabric.stage3.run_mamba_pipe --nc 2 --tokens 2
    python -m fabric.stage3.run_mamba_pipe --engine seq --nc 1 --tokens 2   # baseline

No torch: the reference is the pure-int MambaSeqRef (identical tables to the RTL).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

from fabric.stage3._simdir import kevbuild
from fabric.stage3.mamba_seq_ref import (
    MambaSeqRef, D, DIN, NST, H, L, V, INROWS, CONVD, REPO,
)

import json


def tool(name):
    t = shutil.which(name) or os.path.expanduser(f"~/.local/bin/{name}")
    if not os.path.exists(t):
        sys.exit(f"missing tool: {name}")
    return t


def rd16(path):
    return np.array([int(l, 16) for l in open(path) if l.strip()], dtype=np.int64)


def w32(path, vals):
    with open(path, "w") as f:
        for v in np.asarray(vals).ravel():
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")


def me_word(s):
    from model.mamba2_fixed import scale_me
    m, e = scale_me(s)
    return (e << 16) | m


def build_tables(sim, ex):
    """Write every ms_t*.mem / ms_cfg.mem exactly as run_mamba_seq does (torch-
    free part). Returns (gw, out_base, head_base, table_lengths)."""
    os.makedirs(sim, exist_ok=True)
    sj = json.load(open(f"{ex}/shifts.json"))

    inw = [rd16(f"{ex}/l{l}_in_proj_w4.mem") for l in range(L)]
    outw = [rd16(f"{ex}/l{l}_out_proj_w4.mem") for l in range(L)]
    hdw = rd16(f"{ex}/head_w4.mem")
    gw = np.concatenate(inw + outw + [hdw])
    out_base = sum(len(a) for a in inw)
    head_base = out_base + sum(len(a) for a in outw)
    w32(f"{sim}/ms_t0.mem", gw)

    def cat(fmt):
        return np.concatenate([rd16(f"{ex}/l{l}_{fmt}.mem") for l in range(L)])

    tables = {
        1: rd16(f"{ex}/tok_emb_w4.mem"),
        2: rd16(f"{ex}/tok_emb_scale.mem"),
        3: cat("conv_w"), 4: cat("conv_b"),
        5: rd16(f"{ex}/silu_lut.mem"),
        6: cat("alut"), 7: cat("dtlut"),
        8: cat("ln_g"), 9: cat("norm_g"),
        10: rd16(f"{ex}/normf_g.mem"),
        11: cat("in_proj_scale"), 12: cat("out_proj_scale"),
        13: rd16(f"{ex}/head_scale.mem"),
    }
    for s, v in tables.items():
        w32(f"{sim}/ms_t{s}.mem", v)

    # table 15 = rsqrt seed (WSEL_SEED), 64 x 20-bit — loaded over the write port
    # so nothing depends on the $readmemh init baking into a bitstream. Count
    # goes in cfg[17] (cfg[15] = gemv-weight count, cfg[16] = NC).
    from fabric.stage3.run_layernorm import seed_table
    seedv = np.asarray(list(seed_table()), dtype=np.int64)
    w32(f"{sim}/ms_t15.mem", seedv)

    consts = np.zeros(128, dtype=np.uint64)
    for l in range(L):
        lj = sj["layers"][l] if "layers" in sj else sj[str(l)]
        consts[l*16 + 0] = me_word(1.0 / lj["in_scale"])
        consts[l*16 + 1] = me_word(lj["in_scale"])
        consts[l*16 + 2] = me_word(1.0 / lj["out_scale"])
        consts[l*16 + 3] = me_word(lj["out_scale"])
        consts[l*16 + 4] = (lj["bX"] << 8) | (lj["bC"] << 4) | lj["bB"]
        Dq = [int(np.clip(round(d * 8192), -32768, 32767)) & 0xFFFF
              for d in lj["D"]]
        for k in range(4):
            consts[l*16 + 5 + k] = (Dq[2*k+1] << 16) | Dq[2*k]
    hs = sj.get("head_scale") or sj["layers"][0]["head_scale"]
    consts[112] = me_word(1.0 / hs)
    consts[113] = me_word(hs)
    consts[120] = out_base
    consts[121] = head_base
    w32(f"{sim}/ms_t14.mem", consts)

    lengths = {s: len(v) for s, v in tables.items()}
    lengths[15] = len(gw)
    lengths[14] = 128
    lengths[17] = len(seedv)   # seed count -> cfg[17]
    return gw, out_base, head_base, lengths


def reference(seqs, ex, qh=16):
    """N independent single-stream MambaSeqRef runs (the bit-exact oracle).

    qh<16 quantises the carried scan state to `qh` bits (the INT12 state-quant
    lever) — the oracle the qh-matched RTL is bit-exact to."""
    eng = MambaSeqRef(ex, qh=qh)
    gold = []  # gold[s][t] = (argmax, logits(list V), x_out(list D))
    for seq in seqs:
        st = eng.alloc_state()
        row = []
        for tok in seq:
            c = {}
            eng.step(int(tok), st, collect=c)
            row.append((c["argmax"], list(c["logits"]), list(c[f"l{L-1}.x_out"])))
        gold.append(row)
    return gold


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_mamba_pipe")
    ap.add_argument("--engine", choices=["seq", "pipe"], default="pipe")
    ap.add_argument("--nc", type=int, default=2)
    ap.add_argument("--scan-qh", type=int, default=16,
                    help="scan-h stored bits (16=INT16 bit-exact, 12=INT12 state-quant)")
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--export", default="data/fabric_mamba")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero-seed-init", action="store_true",
                    help="write seed.mem as all-zero (kills the $readmemh baked "
                         "init) while still loading the real seed via the write "
                         "port — proves the AXI seed-load path")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)

    NC, T, QH = args.nc, args.tokens, args.scan_qh
    sim = args.dir or kevbuild(f"stage3_mamba_pipe_nc{NC}_qh{QH}")
    os.makedirs(sim, exist_ok=True)
    ex = args.export

    # d_state (NST) drives the RTL geometry (CONVD/INROWS) — read it from the
    # export manifest so a config-B (d_state=32) export builds the RTL at NST=32.
    nst = 64
    try:
        _mf = json.load(open(os.path.join(REPO, ex, "manifest.json")
                             if not os.path.isabs(ex) else f"{ex}/manifest.json"))
        nst = int(_mf["cfg"]["d_state"])
    except (FileNotFoundError, KeyError):
        pass

    gw, out_base, head_base, lengths = build_tables(sim, ex)

    # ---- token streams: stream 0/1 = val.bin head; others deterministic RNG --
    rng = np.random.default_rng(args.seed)
    seqs = [[int(x) for x in rng.integers(0, V, T)] for _ in range(NC)]
    vb = f"{REPO}/data/bpe1024_chat6/val.bin"
    if os.path.exists(vb):
        val = np.fromfile(vb, dtype=np.uint16)
        seqs[0] = [int(x) for x in val[:T]]
        if NC > 1:
            seqs[1] = [int(x) for x in val[:T]]  # stream 1 also val (task 177/133)

    with open(f"{sim}/ms_tok.mem", "w") as f:
        for s in range(NC):
            for t in range(T):
                f.write(f"{seqs[s][t]:04x}\n")

    cfg = np.zeros(32, dtype=np.int64)
    cfg[0] = T
    for s, ln in lengths.items():
        cfg[s] = ln
    cfg[16] = NC
    w32(f"{sim}/ms_cfg.mem", cfg)

    # ---- reference ----------------------------------------------------------
    # gold is qh-matched (the RTL must be BIT-EXACT to it). gold16 is the INT16
    # truth used to MEASURE the INT12 fidelity hit (argmax agreement).
    gold = reference(seqs, ex, qh=QH)

    # board reference pack: pl_mamba_pipe replays these exact tokens on silicon
    # and compares the engine's per-(stream,token) argmax (dump_tok readback).
    np.savez(f"{sim}/mp_ref.npz",
             toks=np.asarray(seqs, dtype=np.int64),
             ref_argmax=np.asarray([[g[0] for g in row] for row in gold],
                                   dtype=np.int64),
             ref_logits=np.asarray([[g[1] for g in row] for row in gold],
                                   dtype=np.int64))

    # ---- simulate -----------------------------------------------------------
    # seed.mem feeds only the rmsnorm_gated $readmemh sim/fallback init; the
    # real seed loads over the write port (table 15). --zero-seed-init writes
    # zeros here to prove the loaded path (silicon behaves like zeros).
    from fabric.stage3.run_layernorm import seed_table
    with open(f"{sim}/seed.mem", "w") as f:
        for v in seed_table():
            f.write(f"{0 if args.zero_seed_init else int(v):05x}\n")

    if args.engine == "seq":
        src = [f"{REPO}/fabric/stage3/rtl/{m}.sv" for m in
               ("mamba_seq", "gemv_i4i8", "conv_silu", "ssm_scan_row",
                "rmsnorm_gated")]
        src.append(f"{REPO}/fabric/stage3/tb/tb_mamba_seq.sv")
        cfg2 = np.zeros(17, dtype=np.int64)
        cfg2[0] = T
        for s, ln in lengths.items():
            if s < 16:
                cfg2[s] = ln
        cfg2[14] = 128
        cfg2[15] = len(gw)
        cfg2[16] = lengths[17]   # seq cfg word16 = seed count (WSEL_SEED)
        w32(f"{sim}/ms_cfg.mem", cfg2)
        with open(f"{sim}/ms_tok.mem", "w") as f:
            for t in range(T):
                f.write(f"{seqs[0][t]:04x}\n")
    else:
        src = [f"{REPO}/fabric/stage3/rtl/{m}.sv" for m in
               ("mamba_pipe", "gemv_i4i8", "conv_silu", "ssm_scan_row",
                "rmsnorm_gated")]
        src.append(f"{REPO}/fabric/stage3/tb/tb_mamba_pipe.sv")

    subprocess.run([tool("iverilog"), "-g2012", f"-DNC_SIM={NC}",
                    f"-DT_SIM={T}", f"-DNST_SIM={nst}", f"-DQH_SIM={QH}",
                    "-o", "mp.vvp"] + src,
                   cwd=sim, check=True)
    out = subprocess.run([tool("vvp"), "mp.vvp"], cwd=sim, check=True,
                         capture_output=True, text=True, timeout=21600)
    tail = out.stdout[-9000:]   # must hold the whole STAGE_* block (was 1500: silently ate it)
    if "TB_MS_DONE" not in out.stdout:
        sys.exit(f"sim incomplete:\n{tail}")

    import re
    mcyc = re.search(r"TOTCYC\s+(\d+)", out.stdout)
    print(tail.strip())

    # ---- compare ------------------------------------------------------------
    if args.engine == "seq":
        got_l = rd16(f"{sim}/ms_logit.out")
        got_l = np.where(got_l >= 1 << 15, got_l - (1 << 16), got_l).reshape(-1, V)
        got_x = rd16(f"{sim}/ms_x.out")
        got_x = np.where(got_x >= 1 << 31, got_x - (1 << 32), got_x).reshape(-1, D)
        streams = [[(int(np.argmax(got_l[t])), got_l[t], got_x[t])
                    for t in range(T)]]
        ncc = 1
    else:
        got_l = rd16(f"{sim}/mp_logit.out")
        got_l = np.where(got_l >= 1 << 15, got_l - (1 << 16), got_l).reshape(NC, T, V)
        got_x = rd16(f"{sim}/mp_x.out")
        got_x = np.where(got_x >= 1 << 31, got_x - (1 << 32), got_x).reshape(NC, T, D)
        streams = [[(int(np.argmax(got_l[s, t])), got_l[s, t], got_x[s, t])
                    for t in range(T)] for s in range(NC)]
        ncc = NC

    bad = 0
    for s in range(ncc):
        for t in range(T):
            ga, gl, gx = gold[s][t]
            ca, cl, cx = streams[s][t]
            dl = int(np.abs(np.asarray(cl) - np.asarray(gl)).max())
            dx = int(np.abs(np.asarray(cx) - np.asarray(gx)).max())
            am = (ca == ga)
            ok = am and dl == 0 and dx == 0
            if not ok:
                bad += 1
            print(f"  s{s} tok#{t} ({seqs[s][t]}): argmax rtl {ca} ref {ga} "
                  f"{'MATCH' if am else 'DIFF'}  logit|d|={dl}  x|d|={dx} "
                  f"{'OK' if ok else 'FAIL'}")

    verdict = "PASS" if bad == 0 else "FAIL"
    print(f"\nMAMBA_PIPE_VERDICT: {verdict} (NC={ncc}, T={T}; bit-exact vs "
          f"MambaSeqRef qh={QH})  [{sim}]")

    # ---- INT12 fidelity: argmax + logit drift vs the INT16 truth ------------
    if QH < 16:
        gold16 = reference(seqs, ex, qh=16)
        tot_pred = 0; am_agree = 0; l1sum = 0.0; ltot = 0
        for s in range(ncc):
            for t in range(T):
                a12, l12, _ = gold[s][t]        # == RTL (bit-exact to qh gold)
                a16, l16, _ = gold16[s][t]
                tot_pred += 1
                am_agree += int(a12 == a16)
                d = np.abs(np.asarray(l12) - np.asarray(l16))
                l1sum += float(d.sum()); ltot += d.size
        hit = 100.0 * (tot_pred - am_agree) / tot_pred
        print(f"MAMBA_PIPE_FIDELITY qh={QH}: argmax agree {am_agree}/{tot_pred} "
              f"vs INT16 truth  (argmax hit {hit:.2f}%)  mean|logitΔ|={l1sum/ltot:.2f}")
    if mcyc:
        tot = int(mcyc.group(1))
        per = tot / (ncc * T)
        print(f"MAMBA_PIPE_CYC: total {tot:,} cyc  /  {ncc}x{T} = "
              f"{per:,.0f} cyc/token/stream  ({144246/per:.2f}x vs 144,246)")
        print(f"  => @200MHz {200e6/per:,.0f} tok/s   @250MHz {250e6/per:,.0f} tok/s"
              f"   (aggregate x{ncc})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
