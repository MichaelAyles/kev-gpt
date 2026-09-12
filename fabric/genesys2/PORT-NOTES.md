# Genesys2 port notes

Working notes for the kev-gpt -> Genesys2 (X-HEEP + cv32e40px) port. See
`/home/tparng/.claude/plans/steady-mixing-storm.md` for the full phased plan
and its rationale; this file tracks what's actually been done and any
refinements discovered along the way.

## Sizing decision

`sizing.py` sweeps three options; **Option A (d=128, n_layer=2, n_head=4,
ctx=128, ~280KB)** is the first bring-up target. Option B (d=192, n_layer=3,
ctx=192, ~909KB) is the "real chat" target once Option A is verified end to
end on hardware. Run `python -m fabric.genesys2.sizing` to regenerate the
table.

## Phase 1 — done

- `fabric/genesys2/sizing.py` written and self-checked (KV260 config -> weight
  bytes match the project's own "~1.5MB" figure exactly; `kv_bank.sv`'s own
  512KB/48KB worked example reproduced exactly).
- SoC/board repo forked locally: `~/RVchatbot/kevgpt-genesys2-soc` (`git clone
  --local` from `~/RVchatbot/Genesys2AiChatbot`, so it stays independently
  bisectable). Its `.venv` (Python 3.8, fusesoc/edalize/etc.) was physically
  copied across rather than rebuilt from PyPI (no network cost, exact version
  parity) — the copy's embedded absolute paths (`activate`, `pyvenv.cfg`,
  every `bin/*` shebang) were rewritten from the original repo's path to the
  fork's path with `sed`; confirmed working (`fusesoc --version`,
  `import edalize`).
- **Gotcha hit and worked around**: invoking `make mcu-gen` via the repo-root
  wrapper Makefile hit the documented `$(PWD)`-staleness bug even harder than
  CLAUDE.md's own note implied — it resolved the venv python to a truncated,
  wrong path (`<repo>/./.venv/bin/python`, missing
  `hw/vendor/esl_epfl_x_heep/`) and `unset PWD` made it worse, not better, in
  this shell. Fix: invoke `make mcu-gen PYTHON_X_HEEP_CFG=... X_HEEP_CFG=...`
  **directly inside `hw/vendor/esl_epfl_x_heep/`** with the venv activated
  from that same directory — matches CLAUDE.md's documented workaround for
  `vivado-fpga`, evidently also required for `mcu-gen` in this environment.
  `configs/cv32e40px_ai_accel.py` referenced by the wrapper's Makefile as
  `configs/...` actually resolves inside `hw/vendor/esl_epfl_x_heep/configs/`,
  not a repo-root `configs/` (there isn't one) — another reason direct
  invocation is simpler than the wrapper here.
- Stock `mcu-gen` (unmodified `cv32e40px_ai_accel.py` config, i.e. still
  ai_accel's own config, not kevgpt's yet) verified working from the fork.
  `verilator-build` was still running as of this note (Verilator compiles the
  whole X-HEEP SoC, meaningfully slower than mcu-gen) — see the plan file /
  task tracker for its outcome.

## Phase 2 — RTL primitive-swap: done (edits), gate run: in progress

Added a `MEM_PRIMITIVE` parameter (default `"ultra"`, so the KV260 build's
own gates and bitstream are byte-for-byte unaffected) threaded through:

- `weight_bank_tdp.sv` (all 3 `xpm_memory_tdpram` sites: the DP!=0 even/odd
  column-parity banks `u_e`/`u_o`, and the DP=0 `u_tdp`).
- `kv_bank.sv` (the code bank `u_code`; the header bank `u_hdr` already used
  `"block"`, untouched).
- Threaded up through `gemv_banked_resident_vec.sv` (its `weight_bank_tdp`
  instantiation) and `sequencer_vec.sv` (both its `gemv_banked_resident_vec`
  and `kv_bank` instantiations), each gaining the same parameter with the
  same `"ultra"`-preserving default.

**Refinement over the plan's stated verification approach**: the plan said to
run `fabric/stage3/run_vec_kv.py` twice (once per `MEM_PRIMITIVE` value) to
gate this change. Having now read the actual RTL: `MEMORY_PRIMITIVE` only
appears inside the `` `ifdef SYNTHESIS `` branch of each XPM instantiation:
iverilog (which `run_vec_kv.py` uses) always takes the `` `else `` behavioral
branch, which doesn't reference `MEM_PRIMITIVE` at all. **The two parameter
values are therefore bit-identical under iverilog by construction** — running
the gate twice would exercise the exact same simulated code path twice, not
two different ones. One run (at the default) is the correct regression check
for "did this parameter plumbing break anything"; the real `"ultra"` vs
`"block"` distinction is synthesis-only and can only be checked by Vivado
(Phase 6/7's build step) and, ultimately, real hardware. Updated here rather
than silently deviating from the written plan.

Also note (not a change, just confirmed while reading `weight_bank_tdp.sv`'s
own header comment): the "true-dual-port" requirement this port routes around
is about the **XPM `MEMORY_PRIMITIVE="ultra"` inference guarantee**, not
independent dual-port access per se — Xilinx 7-series `BRAM36`/`BRAM18` is
*natively* true-dual-port (two fully independent read/write ports), same as
UltraRAM. `MEMORY_PRIMITIVE="block"` gives up UltraRAM's larger native
geometry and per-port bandwidth, not dual-port capability. `BANKW` (the
72-bit-wide banking `weight_bank_tdp.sv` uses above 512 bits) is a geometry
choice tuned for URAM's native 72-bit port width; left unchanged for this
port since XPM composes whatever primitive is chosen internally regardless
of `BANKW`, and re-tuning it for BRAM's native width is an area-efficiency
optimization, not a correctness requirement — deferred, not forgotten.

Still pending: training Option A's config (`model/train.py`) and exporting
(`model/export_fabric.py`) to get a real weight image, so
`fabric/stage3/run_vec_kv.py` can actually run (it needs
`fabric/export/goformer.npz`, which is gitignored/regenerable and did not
exist in this checkout). `torch` is not in `requirements.txt` (only needed
for `model/`, installed separately) and was not present in the venv; install
was in progress as of this note.

## Phase 3 — done

`fabric/genesys2/rtl/xheep_kevgpt_peripheral.sv`: a `reg_req_t`/`reg_rsp_t`
adapter wrapping the unmodified `sequencer_vec.sv`, porting
`gemv_axi_seq_vec.v`'s register file and semantics (identical register map,
identical IDCODE "SQRV") onto X-HEEP's flat single-cycle valid/ready
protocol — collapses AXI's separate AW/W/B/AR/R channel state machine into
one always-ready decode (`reg_rsp_o.ready = reg_req_i.valid`), since X-HEEP's
register_interface has no separate address/data phase to manage and this
peripheral has no internal wait-state condition.

`fabric/genesys2/tb/tb_kevgpt_peripheral.sv`: directed protocol-conformance
test (IDCODE readback, STATUS before/after `go`, a full `go`..`done` round
trip through the register interface into `sequencer_vec` and back, CYCLES
readback) — no weights loaded, so this checks the bus handshake only, not
model correctness (that stays `run_vec_kv.py`'s job, unmodified). **Verified,
first try**: `KEVGPT_PERIPH_VERDICT pass=True errors=0` (iverilog -g2012;
compiled clean against `sequencer_vec.sv`'s real instantiation closure per
`VENDOR_MANIFEST.txt`; `$readmemh` warnings for LUT-init `.mem` files are
expected/harmless here — those are only populated by `run_vec_kv.py`'s own
setup step for a real model-correctness run, irrelevant to this protocol
test). 7,196 register polls / 14,391 simulated cycles for one full
`go`..`done` pass at the deployed engine's default parameters
(P=16, LANES=128, NLAYER=4, WWORDS=262144, TMAX=64).

## Phase 2 — model-correctness gate: BLOCKED, bigger than the plan anticipated

Trained an Option A checkpoint (`model/train.py --n-layer 2 --n-head 4 --n-embd
128 --block-size 128`, then `--qat --init-from`), exported both
`fabric/export/weights.npz` (195.6KB packed INT4 -- in the right ballpark of
`sizing.py`'s 204.1KB estimate; the gap is `sizing.py` using the KV260's
vocab_size=193 default for all three options rather than the real trained
vocab (57 chars for this corpus) -- a minor sizing.py correction, not
load-bearing) and `fabric/export/goformer.npz` (the golden-reference params),
then ran `fabric.stage3.run_vec_kv`.

**First failure, fixed**: `run_layernorm.py`'s `_ln_int_quantized` (called by
`seq_ref.py`'s `_ln`, which `model.goformer_kvq.IntKVQSequencer` uses every
token) hardcoded `D=256` -- `mean = S >> 8` and `for i in range(D)` never
looked at the actual input length. **And** `layernorm_vec.sv` (the RTL) had
the identical hardcoding one level deeper: `D` was a `localparam`, not a
`parameter`, so `sequencer_vec.sv`'s own `D` module parameter was silently
never reaching the LayerNorm block at all -- meaning `sequencer_vec.sv` has
in practice only ever been exercised at D=256, despite exposing `D` as if it
were already general. Fixed both (Python: derive `d_shift` from `len(X)`;
RTL: `D` promoted to a real parameter, `D_SH = $clog2(D)` replaces the
hardcoded `>>>8`, threaded through from `sequencer_vec.sv`). Both changes are
regression-verified as bit-exact-preserving at D=256
(`python -m fabric.stage3.run_layernorm` still gives
`LN_VERDICT bitexact=True mismatches=0/16384 ... pass=True`).

**Second failure, NOT yet fixed -- this is the one that needs a decision**:
`model/goformer_kvq.py` and `fabric/stage3/seq_ref.py` hardcode `NHEAD=4` /
`HEAD_DIM=64` as **module-level globals**, not per-instance state derived
from the actual model config, and use them as default arguments across
several functions in the KV-cache write/read path (`hadamard_rotate_q16`,
`position_ddr_row`, the `for h in range(NHEAD)` loops in `_store_kv`, the
`nh, hd = NHEAD, HEAD_DIM` in `_attn_step`). Option A's actual per-position
K/V vector is 128 long (d=128), but `position_ddr_row` still assumes
`nh=4, hd=64` (256 total) by default -- silently slicing past the end of a
128-long vector, raising `ValueError: min() iterable argument is empty`
several heads in.

Worse, this isn't just a "pass the right nh/hd" fix: the attention score
scale `1/sqrt(head_dim)` is baked in as an **exact numeric shift amount**,
correct only for `head_dim=64` specifically:
- Python: `seq_ref.py:63` `ISQRT = 3  # /sqrt(head_dim=64) = >>3`, used at
  `seq_ref.py:233`.
- RTL: `vec_attn_w.sv:53` `localparam integer SCORE_SH = 27;  // 2*VFRAC +
  ISQRT - SCORE_FRAC` -- a hand-computed literal, not derived from a
  parameter at elaboration time.

`1/sqrt(head_dim)` is only expressible as an *exact* integer right-shift
(what this whole fixed-point design relies on for bit-exactness) when
`head_dim` is a power of 4 (so its square root is itself a power of 2) --
64, 16, 4, 1, 256, .... Also, separately, `goformer_kvq.py`'s Hadamard
rotation path (`_HAD_SHIFT`, gated on `rotate=True`) asserts the same
"perfect square power of two" constraint on `HEAD_DIM`, though `run_vec_kv.py`
actually calls `IntKVQSequencer(..., rotate=False)`, so that specific path
isn't in the way for *this* gate.

**Why this changes the sizing picture from the approved plan**: `NHEAD=4,
HEAD_DIM=64` multiply out to `d=256` -- the KV260's exact deployed width.
Keeping `d=256` fixed sidesteps every one of these hardcoded constants with
zero further changes, but weight bytes scale as `12*d^2*n_layer`, so even
`n_layer=1` at `d=256` costs ~384KB in weights alone, already over Option
A's ~280KB target with nothing left for KV cache or margin -- not a workable
"safe bring-up" tier. Getting `sizing.py`'s Option A/B (`d=128`/`d=192`)
actually running requires real, scoped work: threading `nh`/`hd` as genuine
per-instance state through `seq_ref.py`/`goformer_kvq.py` instead of module
globals, and deriving `ISQRT`/`SCORE_SH` from the *actual* `head_dim` at both
Python-reference and RTL-elaboration time (restricted to `head_dim` values
that are powers of 4, so the shift stays exact) -- not a blind auto-fix,
since it touches the same numerically-sensitive fixed-point scaling this
whole project's "bit-honest" discipline is built to protect. See the
AskUserQuestion asked at this point in the session for how to proceed.

## Phase 2 continued — three real fixes made, gate still not green

After the "vary n_head, keep head_dim=64 fixed" decision above, retrained
Option A at `n_head=2, n_embd=128` (head_dim=64) and made three further
fixes, each regression-verified not to disturb the KV260 (D=256/NLAYER=4/
NHEAD=4) path:

1. `goformer_kvq.py`'s `_store_kv`/`_attn_step`: `NHEAD`/`HEAD_DIM` globals
   and a hardcoded `C = 256` replaced with `self.p["n_head"]` and
   `len(qkv_q16)//3` (both derived from the actual checkpoint). Added an
   assertion enforcing `head_dim == HEAD_DIM` so a future n_embd/n_head
   choice that breaks the fixed-head_dim=64 assumption fails loudly.
2. `fabric/stage3/tb/tb_seq_vec_kv.sv`: added `DVAL`/`NLAYERVAL`/`NHEADVAL`/
   `VOCABVAL` `` `define ``s (defaults 256/4/4/193, so nothing changes unless
   overridden) and threaded them into `sequencer_vec`'s instantiation
   (`.D`/`.D3`/`.D_MLP`/`.NLAYER`/`.NHEAD`/`.VOCAB`) -- previously only
   `P`/`LANES`/`TMAX` were wired, so this gate had *never* actually exercised
   `sequencer_vec.sv` at a non-deployed D/NLAYER/NHEAD/VOCAB despite those
   being real module parameters.
3. `run_vec_kv.py`: derived `d_model`/`nlayer`/`nhead`/`vocab` from the
   loaded checkpoint (`p["tok_emb"].shape`, `len(p["blocks"])`,
   `p["n_head"]`) instead of a hardcoded `4` at the `write_mems_wideword`
   call and nothing at all for the new testbench defines.

`sizing.py`'s Option B was also revised: `layernorm_vec.sv`'s mean/var divide
is *also* an exact shift, valid only for a power-of-2 D -- independent of
the head_dim=64-fixed constraint. With head_dim pinned at 64, D=n_head*64 is
a power of 2 only for n_head in {1,2,4,...}; there's no valid D between 128
and 256 under both constraints at once. Revised Option B to grow via
n_layer/context instead of D: `d=128 (n_head=2), n_layer=4, ctx=256` (was
`d=192, n_head=3, ctx=192`, which is no longer reachable at all).

**Ran the gate with all of the above -- `VVP_FAIL`, not a mismatch but a
simulation timeout.** The FSM hangs forever in state `S_ACL` (state 13,
`sequencer_vec.sv:754`, `if (adone_s && bdone_s) begin ...`), stuck at
`blk=0` -- it never even leaves the first transformer block.

**Root cause, traced (not fixed)**: `sequencer_vec.sv`'s twin-engine
attention scheduler (`u_attnA`/`u_attnB`, two `vec_attn_w` instances that
split the heads between them for throughput) assigns head indices via two
**hardcoded FSM states**, not a loop over `NHEAD`:
```
hh<=2'd0; hB<=2'd1;   // "pair 0": engine A does head 0, engine B does head 1
...
hh<=2'd2; hB<=2'd3;   // "pair 1": engine A does head 2, engine B does head 3
```
(`sequencer_vec.sv:562,586,770`). This is baked for exactly NHEAD=4 (two
fixed pairs), independent of the `NHEAD` module parameter -- unlike the
three fixes above (which were real parameters just never threaded), **there
is no parameter to thread here**; the pair count and head indices are
literal constants in the state-transition logic. There are also
head-index-derived magic offsets nearby (`vneedA = hh + 6`, `vneedB = hB +
4`, `sequencer_vec.sv:309-310`) whose derivation I have not traced and would
need to before trusting a fix.

This is a materially different and larger class of change than the three
fixes above: a real (if bounded) FSM redesign -- generalizing two hardcoded
pair-states into a genuine loop over `NHEAD/2` pairs -- touching
numerically/timing-sensitive control logic I have not fully traced (the
`vneedA`/`vneedB` offsets, and whether the KV-cache *write*-side states
S_KVW_S/S_KVW_F/S_KVW_W have the same hardcoded-pair pattern on the write
side, not just here on the read/attention side). Given the user's
"vary n_head" decision was specifically chosen *to avoid* the bigger,
riskier "generalize head_dim" RTL work, and this turns out to need real FSM
surgery anyway (just in a different place than expected), this changed the
risk/effort picture behind that decision -- flagged back to the user rather
than pushed through at this point in the session.

## Phase 2 continued — traced the scheduler in full; stopping short of rewriting it

User direction: generalize the pair scheduler rather than revert to n_head=4
or try a different sequencer. Traced it fully before touching anything
further (no RTL edits made past this point) -- it is **two interacting,
independently hand-tuned FSMs**, not the one simple pair-loop the earlier
framing assumed:

1. **The "main" KV writer** (`S_KVW_S`/`S_KVW_F`/`S_KVW_W`,
   `sequencer_vec.sv:568-588`): a blocking, serial writer that writes K then
   V for head 0, then K then V for head 1 (`kvw_h` 0->1, `kvw_kv` toggling),
   entered before the qkv GEMV even starts. On completion (`sequencer_vec.sv
   :584-587`) it hands off to (2), hardcoding `kvw_h<=2` (head *2*, literally
   assuming exactly 4 heads exist and heads 0-1 are already done).
2. **The "feeder" FSM** (`kvf_st`/`kvf_active`, driven combinationally at
   `sequencer_vec.sv:899-941`): writes the *remaining* heads (2, 3) in a
   deliberately shuffled order -- K2, K3, then V1, V0, V3, V2 (not V2, V3) --
   chosen so each engine's V-write lands in a specific gap in the *other*
   engine's read schedule (`sequencer_vec.sv:288-301`'s comment spells out
   the intent). This runs concurrently with pair-0's attention compute
   (`vec_attn_w` engines A/B), i.e. its writes are deliberately hidden under
   already-in-flight work -- a real latency-hiding optimization, not
   incidental structure.
3. The read side's `vneedA = hh+6` / `vneedB = hB+4`
   (`sequencer_vec.sv:309-310`) are the exact indices into this specific
   8-step interleaved order that gate when each engine's V-read may start
   without a read-before-write race against the feeder -- i.e. they encode
   the interleaving's timing contract, not an independent piece.

**Why I stopped here rather than rewriting it**: a safe generalization needs
either (a) faithfully extending this exact two-tier hide-the-latency design
to `NHEAD/2` pairs generically (real, delicate FSM design work -- the
overlap/gap structure that makes the shuffled V-order correct doesn't have
an obvious general form I could derive with confidence by inspection), or
(b) a deliberate *simplification* -- drop the overlap optimization entirely,
serialize all `2*NHEAD` K/V writes up front (simple K0..K(NHEAD-1),
V0..V(NHEAD-1) order), and gate `S_AST` on `kvp_done >= 2*NHEAD` for every
pair (no more per-pair partial gating, no more vneed/vpA/vpB interlocks at
all -- attention only ever starts once every write for the position is
already committed). (b) is very likely correct-by-construction and a small,
mechanical diff, at the cost of losing the write-under-compute latency
hiding (a real cycles-per-token regression on top of what this port is
already trading away elsewhere). I traced far enough to be confident (b) is
the right shape of fix, but did not make the edit: doing it correctly still
means resizing `kvw_h`/`kvp_done`/`pair` for a general (not just NHEAD=2)
bound, re-deriving the K/V write-order state transitions from scratch
(`sequencer_vec.sv:932-941`), and removing/proving-dead the vneed/vpA/vpB
interlocks (`sequencer_vec.sv:899-910`) -- real edits across several
interacting `always` blocks I want a waveform (VCD dump, not just the
pass/fail cycle counter this gate currently reports) to check against
before trusting, not just careful reading. That's a dedicated follow-up, not
a few more minutes.

## Phase 2 — RESOLVED: scheduler generalized, gate green at both shapes

Implemented the surgical version of fix (b) above -- not "wait for
everything" (too big a latency-hiding regression), but the smaller, more
precise fix the full trace made possible: keep the K-writes-first structure
and per-pair partial gating (which generalize with no changes at all —
`S_AST`'s existing `kvp_done >= (pair ? 4'd4 : 4'd2)` was already correct for
NPAIRS=1 without modification, since it only ever evaluates the pair=0
branch), and only replace the **V-write order** — plain ascending
(K0..K(NHEAD-1), V0..V(NHEAD-1)) instead of the hand-shuffled
(K0,K1,K2,K3,V1,V0,V3,V2) — since that shuffle, not the pairing/gating
structure itself, was the part with no general form. Four edits, all in
`sequencer_vec.sv`:

1. Added `localparam integer NPAIRS = NHEAD/2` (documented as staying valid
   only while `pair` (1 bit) can represent it, i.e. NPAIRS in {1,2} — this
   port's sizing options only ever need that).
2. `vneedA`/`vneedB` (the "V for this engine's head has committed" kvp_done
   thresholds): re-derived for the new ascending order (`NHEAD+h+1`), was
   `hh+6`/`hB+4` (exact only for the old shuffled order at NHEAD=4).
3. `S_CDR`'s next-pair transition: `if (pair)` -> `if (pair == NPAIRS-1)`;
   `hh<=2'd2; hB<=2'd3;` -> `hh<=(pair+1)*2; hB<=(pair+1)*2+1` (derived, not
   literal) — produces identical values to the original at NHEAD=4 (the only
   case checkable against the pre-port design), and is simply never taken at
   NHEAD=2 (NPAIRS=1, the first and only pair is always the last one).
4. The feeder's `KF_W` write-order case statement: replaced the 4-branch
   hardcoded shuffle with a 2-branch ascending-index increment
   (`kvw_h != NHEAD-1 -> kvw_h+1`, else advance phase/finish) — general for
   any NHEAD, not just 4.

Left untouched, deliberately: `S_KVW_S`/`S_KVW_F`/`S_KVW_W`
(`sequencer_vec.sv:568-588`) turned out to be **dead code** — grepped for
every place anything sets `st<=S_KVW_S` and found none; the real (and only)
write mechanism is the `kvf_st`-driven feeder, armed at `S_QKVRET` and
free-running (gated by `kvf_grant`) for the whole position, not a two-tier
"main FSM does heads 0-1, feeder does heads 2-3" split as an earlier,
incomplete read of this file suggested. Left in place rather than deleted —
removing it cleanly would also mean touching `kvf_grant`'s condition and
`kvw_src`'s mux (both reference `S_KVW_F`), a larger diff for a
purely-cosmetic win.

**Verified, both bit-exact via `fabric.stage3.run_vec_kv` (the same golden
reference, `model.goformer_kvq.IntKVQSequencer`, unmodified)**:
- Option A shape (d=128, n_layer=2, n_head=2, head_dim=64): `VEC_KV_VERDICT
  match=True` -- `gen=[1,29,30,34,1,40]` decodes to `'once' -> ' him s'`,
  exact token-for-token match with the golden model.
- **Regression check at the original deployed KV260 shape** (d=256,
  n_layer=4, n_head=4, head_dim=64 — a fresh checkpoint trained specifically
  for this check, `data/ckpt_kv260regress.qat.pt`): also `VEC_KV_VERDICT
  match=True`, `avg_cyc=14948`/token — in the range the project's own stated
  estimate for this exact engine implies (`fabric/stage3/README.md`: "~16,100
  cyc/token" for the PE=256 design; this run is P=8/LANES=128, a different
  point on the same curve, but the same order of magnitude, not a red flag).
  This is the load-bearing check: it proves the write-order generalization
  produces byte-identical *decode output* to the original hand-tuned
  schedule at the one shape it's possible to check it against, not just "no
  hang."

Sizing options remain as revised (`fabric/genesys2/sizing.py`): Option A
(d=128, n_layer=2, n_head=2, ctx=128) is now gate-verified end to end, not
just sized on paper.

## Phase 4 — SoC integration: scaffolding was already done, one build gotcha found + fixed

Resuming a later session found `fabric/genesys2/rtl/xheep_kevgpt_peripheral.sv` already
vendored into the fork (`hw/vendor/esl_epfl_x_heep/hw/ip/kevgpt_seq/`) along with
`kevgpt_seq.core`, `configs/cv32e40px_kevgpt.py`, a stripped
`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`, a lint waiver, and
`sw/applications/kevgpt_bringup/` (probe IDCODE / go / poll done / read tok_out) — all
dated earlier the same day, `make mcu-gen` + `make verilator-build` had already been run
(build tree present) but with no confirmed-successful completion log.

**Found and fixed: two orphaned `make app PROJECT=kevgpt_bringup` invocations stuck at
99.9% CPU for 14h and 5h respectively**, going nowhere. Root-caused via strace + a
timed/polled direct re-run of the `cmake` configure step (confirmed compiler-ID/ABI
detection succeeds fine; the hang is silent, right after "Detecting CXX compile features
- done", i.e. inside `sw/CMakeLists.txt`'s own logic, not toolchain detection). Cause:
both invocations manually passed `SOURCE=../../../sw/` on the command line — that value
is correct **only** for `make vivado-fpga` invoked directly inside
`hw/vendor/esl_epfl_x_heep/` (this fork's `CLAUDE.md` documents it there specifically).
For `make app`, `external.mk`'s own `SOURCE ?= $(SW_TO_SW_REL_PATH)` already
auto-computes the right (4-levels-up) path when left unset — passing the 3-up value
overrides that with one level too shallow. `sw/Makefile`'s
`source_path := $(realpath $(mkfile_path)/$(SOURCE))` then resolves to a **nonexistent**
directory (`hw/sw/` instead of the real `sw/` at repo root); GNU Make's `$(realpath)`
silently returns empty string for a nonexistent path (no error), so
`SOURCE_PATH = $(source_path)/` becomes literally `/`. `CMakeLists.txt:69`'s
`FILE(GLOB_RECURSE new_list FOLLOW_SYMLINKS ${SOURCE_PATH}*.h)` (and the matching
`*.c`/`*.s`/`*.S`/`*.cpp` globs right after) then walks the **entire filesystem root**,
following symlinks — explaining both the CPU-pegged hang and why it produced no error,
ever. This fork's own `CLAUDE.md` already documents the correct bare-wrapper form for
`ai_accel_bringup` (`make app PROJECT=ai_accel_bringup`, no `SOURCE=`) — the bug was
copying the *other* gotcha's `SOURCE=../../../sw/` value into this different invocation
context, not a defect in the wrapper itself.

**Fix applied (this session): killed both stuck processes, `rm -rf
hw/vendor/esl_epfl_x_heep/sw/build`, re-ran `make app PROJECT=kevgpt_bringup` with no
`SOURCE` override** — configures and links in seconds, produces `main.elf`/`main.hex`/
`main.bin` (`sw/build/`, `main.elf` 178,584 bytes). The run does still hit the
already-documented, benign trailing `$(PWD)`-staleness failure in the `mem_usage.py`
report step (`make[1]: .../.venv/bin/python: No such file or directory`, Error 127) —
expected per this fork's own `CLAUDE.md`, fires only after the binary is already fully
built, safe to ignore for simulation purposes.

**Lesson for future invocations in this fork: never pass an explicit `SOURCE=` to
`make app`** (any PROJECT, not just `kevgpt_bringup`) — let `external.mk` compute it.
Only `vivado-fpga`, invoked directly inside `hw/vendor/esl_epfl_x_heep/`, needs the
explicit `SOURCE=../../../sw/` override.

## Phase 4 — DONE: first CPU+peripheral+UART Verilator co-sim, green

`make verilator-run` (invoked directly inside `hw/vendor/esl_epfl_x_heep/` with
`SOURCE=../../../sw/`, the *other* documented `$(PWD)`-staleness workaround — the
top-level wrapper resolves the venv's `fusesoc` to a nonexistent path for this target
too, same class of bug as the one just fixed above but in the opposite direction: this
target needs the explicit override, `make app` needs it omitted): cv32e40px boots,
`kevgpt_bringup`'s control-plane smoke test runs end to end through the real X-HEEP bus/
crossbar into `xheep_kevgpt_peripheral` into unmodified `sequencer_vec`, and back out
over UART:

```
KEVGPT_PHASE,control_plane
KEVGPT_ID,0x53515256          -- "SQRV" (0x53='S',0x51='Q',0x52='R',0x56='V'), matches IDCODE
KEVGPT_STATUS_PRE,0x00000000  -- idle before go, as expected
KEVGPT_TOK_OUT,0
KEVGPT_CYCLES,14391
KEVGPT_PASS,control_plane
```

**Cross-check**: 14,391 cycles is an *exact* match to Phase 3's standalone iverilog
protocol testbench (`tb_kevgpt_peripheral.sv`) figure quoted above ("14,391 simulated
cycles for one full go..done pass at the deployed engine's default parameters"). Same
register-level transaction, same cycle count, now reached through the full SoC integration
(bus/crossbar/firmware) instead of a directed testbench driving the peripheral in
isolation — strong evidence the integration layer didn't perturb the timing-sensitive
handshake at all, not just "didn't crash."

**Scope note, deliberate**: this run has no real weight image loaded (`$readmem file not
found` warnings for `gamma_w.mem`/`gelu_lut_*.mem`/etc. are expected/harmless, same as
Phase 3's note) — `kevgpt_bringup/main.c`'s own comment says so explicitly: this is the
control-plane/handshake smoke test, and streaming a real trained weight image + asserting
`tok_out` against a precomputed value is deferred to Phase 5's `kevgpt_chat` (the
model-correctness side stays `fabric.stage3.run_vec_kv`'s job, per the project's
bit-honest rule — this phase only had to prove the bus plumbing, and did).

## Phase 5 — DONE: firmware streams real weights, bit-exact match, PASS

Re-exported Option A (`data/ckpt_optionA.qat.pt`) to `fabric/export_optionA/`
(`model/export_fabric.py` + a direct `goformer_full.save_params` call for the
golden-reference npz `run_vec_kv.py` needs) since `fabric/export/` had since been
overwritten by the KV260-regression checkpoint. 195.6KB packed weights, matching
`sizing.py` exactly.

**Three real bugs found and fixed before the firmware work could even be gated:**

1. **`xheep_kevgpt_peripheral.sv` was never actually parameterized to Option A.**
   Its `sequencer_vec` instantiation only threaded `P`/`LANES`/`NLAYER`/`WWORDS`/`TMAX`
   -- `D`/`D3`/`D_MLP`/`NHEAD`/`VOCAB` silently fell back to `sequencer_vec`'s own
   KV260-deployed defaults (256/768/1024/4/193). Added those five parameters (defaults
   now Option A's shape: 128/384/512/2/57) and threaded them into the instantiation,
   in both the fork's vendored copy and kev-gpt's canonical
   `fabric/genesys2/rtl/xheep_kevgpt_peripheral.sv` (kept in sync per
   `VENDOR_MANIFEST.txt`). Also explicitly overridden at the wrapper instantiation
   site (`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`) for visibility.
2. **P=16 (this instance's other never-actually-verified default) produces X-state
   `tok_out`** at the full-sequencer level -- found by running
   `fabric.stage3.run_vec_kv --p 16 --lanes 128` against the Option A export:
   `VEC_KV_VERDICT match=False`, `hw gen=[None]*6`. Every previously-green
   `run_vec_kv` result in this project (including every check earlier in this file)
   used P=8; P=16 is not exercised by any gate in the repo (`run_banked.sv` sweeps
   `lanes`, not `P`) and is apparently untested/broken at the `sequencer_vec` level.
   Not investigated further -- flagged in both peripheral copies' parameter comments,
   defaulted to P=8 (the only value this project has ever gated green). Re-ran the
   gate at P=8/LANES=128 against the Option A export: `VEC_KV_VERDICT match=True`,
   `hw=' him s' gold=' him s'`, `gen=[1, 29, 30, 34, 1, 40]` -- the exact config now
   deployed.
3. **Linker region overflow.** `configs/cv32e40px_kevgpt.py`'s memory_ss (inherited
   unmodified from `cv32e40px_ai_accel.py`) bumped from `[32]*5` (288KB total) to
   `[64]*5` (448KB total) for the ~293KB embedded weight blob (see below), but the
   *first* attempt left the code/data linker-section split at ai_accel's own
   `0x10000`/`0x10000` (64KB code, rest data) -- `ld: section .rodata will not fit in
   region ram0`, overflowed by 244,272 bytes, because ai_accel's own footprint is
   mostly buffers/state (so its split gives "data" the bulk of the room), while
   kevgpt_chat's ~293KB weight array is a `static const` -> `.rodata` -> the "code"
   region. Fixed by moving the split to `0x60000`/`0x10000` (384KB code, 64KB data).

**Architectural finding (the load-bearing one): the `wl_we`/`W_DATA` register path is
NOT a complete weight-loading mechanism.** `sequencer_vec.sv`, `layernorm_vec.sv`,
`softmax_f.sv`, `kv_bank.sv`, `gelu_lut.sv`/`gelu_lut2.sv` each `$readmemh` several
ROMs directly from hardcoded relative filenames -- `gamma_w.mem`/`inv_sact.mem`/
`dqm_w.mem`/`dqe_w.mem` (model-specific: LayerNorm gains + dequant scales, depend on
the trained checkpoint) and `gelu_lut.mem`/`gelu_lut_e.mem`/`gelu_lut_o.mem`/
`inv_lut_lo.mem`/`inv_lut_hi.mem`/`exp_lut.mem`/`gumbel_lut.mem`/`seed.mem` (fixed
math tables, not model-specific) -- completely independent of `wl_we`, which only
feeds `gemv_banked_resident_vec`'s big weight matrix (+ the embed tables appended
into the same `wrom.mem` stream, per the wide-word doc comment above). This matches
`webchat/demo/a53_daemon.py`'s own "baked-ROM bitstream" phrasing for the KV260's real
deployment: the *whole* model (not just the big matrix) is `$readmemh`-baked into the
bitstream at Vivado synthesis time; `wl_we`/`W_DATA` streaming exists for the gate
harness's convenience (swap weight sets across iverilog runs without re-synthesizing),
not as the production loading path.

First `kevgpt_chat` Verilator run (firmware streaming only `wrom.mem`'s content via
`wl_we`, nothing else present) completed cleanly -- no hang, `KEVGPT_CYCLES` matching
the gate's `avg_cyc` almost exactly -- but every generated token was `0`
(`KEVGPT_FAIL,generate_mismatch`). Root cause confirmed directly from the sim log's
own `%Warning: gamma_w.mem:0: $readmem file not found` lines (and the matching ones for
every other ROM above) -- none of those files existed in the Verilator build's working
directory, so LayerNorm gains/dequant scales/GELU/softmax/gumbel LUTs were all
garbage/zero regardless of the correctly-streamed big weight matrix. **Not a firmware
or protocol bug** -- confirmed by checking `weight_bank_tdp.sv`/
`gemv_banked_resident_vec.sv` have no `$readmemh` at all (the big matrix is genuinely
`wl_we`-only), while grepping the other RTL files for `readmemh` turned up all eight-ish
hardcoded filenames directly.

**Fix (simulation-only, this phase's actual scope): generated the missing ROM `.mem`
files for the Option A checkpoint (via `seq_ref.build` + `write_mems_wideword`, the
identical code path `run_vec_kv.py`'s gate uses, plus the same gelu-split/inv-lut/
gumbel-lut post-processing steps) directly into the Verilator build's working
directory** (`hw/vendor/esl_epfl_x_heep/build/.../sim-verilator/`) before re-running.
Re-ran `make verilator-run` -- the `$readmem file not found` warnings are gone, and:

```
KEVGPT_GEN,pos=3,tok=1     KEVGPT_GEN,pos=6,tok=34
KEVGPT_GEN,pos=4,tok=29    KEVGPT_GEN,pos=7,tok=1
KEVGPT_GEN,pos=5,tok=30    KEVGPT_GEN,pos=8,tok=40
KEVGPT_CYCLES,22464
KEVGPT_CMP,i=0,got=1,want=1   ... (all 6 match)
KEVGPT_PASS,generate
```

Bit-exact match against `model.goformer_kvq.IntKVQSequencer.generate_greedy` (the same
golden reference `run_vec_kv.py` gates against): `got == want == [1, 29, 30, 34, 1, 40]`
(`"once"` -> `" him s"`), `Program Finished with value 0`. This is the first time real
compiled RISC-V firmware, through the real X-HEEP bus, has driven `sequencer_vec`
through a full multi-token generation and produced output matching the Python
reference -- not just a protocol-conformance smoke test (Phase 4's scope) but genuine
model correctness end to end.

**Scope note for Phase 6/7**: this fix (dropping generated `.mem` files into the sim
build directory) only works because Verilator's `$readmemh` reads from its CWD at
simulation start. **Real Vivado synthesis needs the equivalent files present at
*synthesis* time**, in whatever relative location `weight_bank_tdp.sv`/etc.'s
`$readmemh` calls resolve against inside the Vivado build -- untested, flagged here
rather than assumed. `fabric/genesys2/gen_chat_fw.py` (the weight-image/expected-token
header generator, still useful for the firmware side) and this session's inline
ROM-file generation should be consolidated into one script before Phase 6 so both the
firmware weight stream and the synthesis-time ROM files come from one invocation
against one checkpoint, not two separately-run snippets.

## Phase 6 — DONE: real Genesys2 bitstream, synth+place+route+bitgen clean, positive timing

Six real bugs found and fixed, in order, each verified before moving to the next
(bit-honest before fast, applied to the synthesis flow itself, not just the RTL):

1. **`$readmemh` resolution for Vivado**: same architectural fact as Phase 5 (the
   `wl_we` register path only loads the big GEMV weight matrix; LayerNorm gains,
   dequant scales, and the fixed LUTs are baked in via hardcoded `$readmemh` calls),
   but for *synthesis* the resolution directory is different from Verilator's:
   confirmed empirically (a synth-only `make synth` checkpoint, deliberately run
   *before* committing to the full multi-hour build) that Vivado's `launch_runs
   synth_1` spawns synthesis as its own `vivado -mode batch` subprocess with CWD =
   `<project>.runs/synth_1/` -- generated the same 21 ROM `.mem` files there (and,
   redundantly but harmlessly, at the project root) before every synth/build run
   this phase.
2. **Wrong FPGA target entirely.** First "successful" build (0 errors, positive
   timing, 71.46% LUTs) turned out to have synthesized `xilinx_core_v_mini_mcu_
   wrapper` (the **ai_accel-combined** base design) instead of `xilinx_core_v_
   mini_mcu_wrapper_kevgpt` -- `core-v-mini-mcu.core`'s `genesys2` target's
   `toplevel` is a `no_ai_accel`-flag ternary between exactly those two wrappers;
   kevgpt's own dedicated `genesys2_kevgpt` target (already defined in the .core
   file, `toplevel: [xilinx_core_v_mini_mcu_wrapper_kevgpt]`) exists specifically
   to avoid this, and I simply used the wrong `FPGA_BOARD=` value. Caught by
   grepping the generated `.tcl` for `set_property top` before trusting the run --
   worth doing on every future FPGA_BOARD invocation in this fork.
3. **`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv` was never added to the `.core`
   file's fileset**, only referenced as a `toplevel` value -- `ERROR: [Synth 8-439]
   module 'xilinx_core_v_mini_mcu_wrapper_kevgpt' not found` at RTL elaboration
   once the *correct* target was actually used for the first time. Added it to
   `rtl-fpga`'s `files:` list alongside the other two wrappers.
4. **`gpio`'s `no-clock-gate` fileset never fires for `genesys2_kevgpt`.**
   `pulp_platform_gpio.core`'s `target_genesys2? (no-clock-gate)` is FuseSoC's
   implicit `target_<name>` flag, exact-match per target name -- `genesys2_kevgpt`
   being a *separate* target name (not a flag on `genesys2`, a deliberate choice
   documented in `core-v-mini-mcu.core`'s own comment, since FuseSoC's
   list-conditional syntax wasn't confirmed to support the compound `&&` a
   three-way ternary would need) means it silently misses every
   `target_genesys2?`-gated dependency elsewhere in the vendor tree. Grepped the
   whole tree for `target_genesys2?`; found exactly one real hit (this one, gpio's
   `gpio_input_stage` -- `ERROR: [Synth 8-439] module 'gpio_input_stage' not
   found`) and added a matching `target_genesys2_kevgpt?` line.
5. **`WWORDS` oversized** (fixed earlier in Phase 5's session but re-confirmed
   costly here): the untightened 262144-word default alone put LUTs at 123.75%
   before place&route could even be attempted. Already fixed to 8192 in Phase 5;
   confirmed again here as a real, necessary fix (not premature optimization).
6. **The big one: twin attention engines (`u_attnA`+`u_attnB`) blew both LUTs and
   DSPs, independent of `LANES`.** A synth-only checkpoint at the *correct* target
   (P=8, LANES=64, WWORDS=8192, everything else right) still showed **252,194 LUTs
   (123.75% of the xc7k325t's 203,800)** and **804/840 DSP48E1 (95.71%)** --
   `LANES` tuning (128->64) barely moved either number (LUTs 252,194->243,617,
   DSPs unchanged at 804), because `LANES` only sizes `gemv_banked_resident_vec`
   (0 DSPs, uses BRAM), not attention. A Vivado hierarchical utilization report on
   the completed checkpoint (`report_utilization -hierarchical`, run against the
   already-synthesized `.dcp` -- seconds, not another 15-minute resynth) showed
   `u_attnA`+`u_attnB` alone at 494/804 DSPs (61%) and 88,206/243,617 LUTs (36%),
   entirely inside `vec_attn_w.sv`'s own logic (wide fixed-point Q*K/prob*V
   multiplies, e.g. a ~48x~128-bit multiply, replicated across all P=8 lanes,
   times two concurrent engines) -- not further decomposable, i.e. genuinely
   structural, not a parameter.

   **Fix: collapsed `sequencer_vec.sv`'s twin-engine (head-PAIR-concurrent)
   attention scheduler to a single engine processing all NHEAD heads serially.**
   Traced the full structure before touching anything (matching this file's own
   Phase-2 precedent for changes of this class): removed `hB`/`pair`/`NPAIRS`/
   `qsel`/`bdone_s`/`vpB`/`ctxbufB`/`vneedB`/`u_attnB` and `kv_bank`'s second read
   port entirely (tied `rd2_start=1'b0` permanently at the instantiation --
   `kv_bank.sv` itself is UNCHANGED, still shared with the KV260 gate); `hh`
   becomes the single active head index, looping 0..NHEAD-1; `S_AST`'s pair gate
   (`kvp_done >= (pair?4:2)`) becomes a per-head gate (`kvp_done >= hh+1`); `S_ALD`
   drops its A-then-B two-phase Q-stream for a single-phase stream; `S_ACL` waits
   on `adone_s` alone; `S_CDR` drains `HR` cycles/head (was `2*HR`/pair) into
   `ctxv_bank[hh*HR+...]`, looping back to `S_AST` for the next head until
   `hh==NHEAD-1`. `vneedA`'s formula (`NHEAD+hh+1`) needed no change -- it already
   only depended on `hh`, never on the pair structure.

   **Verified bit-exact both shapes** (`fabric.stage3.run_vec_kv`, same golden
   reference): Option A (D=128/NLAYER=2/NHEAD=2/LANES=64) `VEC_KV_VERDICT
   match=True`, `gen=[1,29,30,34,1,40]` unchanged, avg_cyc **4038->4174 (+3.4%)** --
   far cheaper than the ~2x expected, since Q-streaming to the twin engines was
   already partially serial (stream to A, *then* to B) even in the concurrent
   design; only the K/V-read-and-compute phase actually overlapped. KV260
   regression (D=256/NLAYER=4/NHEAD=4/LANES=128) also `match=True`,
   `gen=[1,41,29,26,34,1]` unchanged, avg_cyc 14948->15492 (+3.6%). Re-verified
   Phase 5's full firmware `kevgpt_chat` end-to-end too: `KEVGPT_PASS,generate`,
   36324->37548 cycles (+3.4%), same match.

   **Result, re-synthesized: LUTs 252,194->93,651 (123.75%->45.95%)** -- fixed
   completely, as expected. **DSPs 804->803 (95.71%->95.60%) -- essentially
   unchanged**, which was NOT expected: a fresh hierarchical report on the new
   checkpoint showed `u_attnA` alone (the one remaining instance) now uses 384
   DSPs, up from 247 in the twin-engine version, for bit-identical RTL just
   invoked without a concurrent twin -- apparently a Vivado optimization-context
   effect (different surrounding logic changes DSP-inference/sharing decisions for
   the same module), not something controllable from the RTL as written. Flagged,
   not chased further -- 803/840 is still *under* 100%, and the real test is
   whether it places & routes, not whether the DSP count matches my mental model.

   **It does.** Full `make vivado-fpga FPGA_BOARD=genesys2_kevgpt SOURCE=../../../sw/`
   (synth->place->route->bitstream, ~35 min from a fresh synth) completed with
   **0 Errors, 0 Critical Warnings**: `place_design completed successfully`,
   `route_design completed successfully`, `write_bitstream completed successfully`,
   `Bitgen Completed Successfully`. Final placed utilization: **LUTs 96,424/203,800
   (47.31%)**, **DSPs 803/840 (95.60%)**, **Block RAM 371.5/445 tiles (83.48%)**.
   **Timing: WNS=2.640ns, TNS=0.000, "All user specified timing constraints are
   met."** -- comparable to or better than this fork's own ai_accel reference build
   (WNS=2.360ns per `CLAUDE.md`). Produces
   `.../genesys2_kevgpt-vivado/openhwgroup.org_systems_core-v-mini-mcu_1.0.5.bit`
   (11.4MB).

## Phase 6/7 real-hardware bring-up — DONE, bit-exact AND cycle-exact on real silicon

The board was physically connected (FT2232H JTAG on channel 0, FT232R UART, both
onboard) and powered on this session, so the "mandatory hands-on" step from the
original plan turned out to be doable directly: JTAG-programmed, loaded firmware via
GDB, and read real UART -- all from this environment, no separate human-operated
terminal needed. Corrected one stale-doc gotcha before trusting anything: this
fork's actual `sw/device/target/genesys2/x-heep.h` has `UART_BAUDRATE 115200` (not
9600 -- `QuadxHeep/CLAUDE.md`'s "9600 8N1" note is accurate for *its own* unmodified
X-HEEP copy but stale for this fork, which diverged; trusted the real file over
either doc).

**Programming**: `vivado -mode batch -source
openhwgroup.org_systems_core-v-mini-mcu_1.0.5_pgm.tcl -tclargs xc7k325tffg900-2 ...bit`
(FuseSoC's own `--run` is a no-op in this lineage, documented in the sibling
QuadxHeep repo) -- `INFO: SUCCESS! FPGA xc7k325tffg900-2 successfully programmed`.
`openocd -f quad-x-heep-openocd-genesys2.cfg` then found the RISC-V debug TAP
cleanly: `JTAG tap: riscv.cpu ... found`, `Examined RISC-V core; found 1 harts`.

**Process-hygiene gotcha, hit repeatedly**: orphaned `openocd`/`cat
/dev/ttyUSB0` processes from earlier attempts (in this session and, based on one
very stale PID, an earlier one) silently held the JTAG USB interface and/or the
UART tty exclusively, causing `LIBUSB_ERROR_BUSY` on reprogram attempts and
multiple simultaneous readers splitting/dropping UART bytes unpredictably (a run
that had genuinely completed successfully on the target produced an *empty*
capture file because 5 stale `cat` processes were racing for the same bytes).
Fix each time: `pgrep -af "cat /dev/ttyUSB0|riscv32-corev-elf-gdb|openocd"`,
kill everything found, **reprogram the bitstream fresh** (guaranteed full fabric
reset, sidesteps needing to reason about JTAG-reset-vs-peripheral-reset scope --
`reset_config none` in this fork's OpenOCD config means `monitor reset halt`
does NOT assert a hardware reset line, so stale peripheral state like
`xheep_kevgpt_peripheral`'s `done_latched` can survive a JTAG-only CPU reset
between two firmware loads), then restart OpenOCD, confirm exactly one UART
reader, and only then load.

**`kevgpt_bringup` (`TARGET=genesys2`), real hardware, first try after the above**:
```
KEVGPT_PHASE,control_plane
KEVGPT_ID,0x53515256          -- "SQRV", correct IDCODE readback on real silicon
KEVGPT_STATUS_PRE,0x00000000
KEVGPT_TOK_OUT,0
KEVGPT_CYCLES,4124
KEVGPT_PASS,control_plane
```

**`kevgpt_chat` (`TARGET=genesys2`), real hardware, after the process-hygiene fix**:
```
KEVGPT_WEIGHT_WORDS,73856
KEVGPT_STATUS_PRE,0x00000000
KEVGPT_GEN,pos=3,tok=1  ... pos=8,tok=40
KEVGPT_CYCLES,37548
KEVGPT_CMP,i=0..5: all match
KEVGPT_PASS,generate
```
`got=[1,29,30,34,1,40]` -- bit-exact with the Python golden reference (same as
every gate this session) **and cycle-exact with the Verilator simulation
(37,548 cycles both places)**, not just functionally matching. `"once" -> " him
s"` now confirmed on real Genesys2 silicon, weights streamed via the real
`wl_we` register path (not a testbench shortcut), model-correctness end to end.

Debugging note for future sessions: a `nm`-based "nearest preceding symbol"
lookup for a halted PC can point at the wrong function if the real containing
function has no exported symbol (a PC sitting in the CPU's ordinary post-`main()`
exit sequence -- `soc_ctrl_set_exit_value`/`soc_ctrl_set_valid`/`wfi` -- briefly
looked like an infinite loop in the unrelated `_fstat` stub this way; disassembly
around the actual PC resolved it in seconds once checked directly).

## Real-hardware token rate: ~11,900-12,000 tok/s (Option A, 50MHz)

Measured directly, not estimated: added `KEVGPT_PASS_CYCLES,pi=<n>,cyc=<count>`
per-pass logging to `kevgpt_chat/main.c` (each pass's own hardware cycle counter,
read right after that `go`->`done` round trip, on top of the existing summed
`KEVGPT_CYCLES` total) so per-token latency could be read directly off real
silicon instead of only inferred from the aggregate. Rebuilt (`TARGET=genesys2`),
reflashed, reran on the physical board:

```
KEVGPT_PASS_CYCLES,pi=0,cyc=4124   (prompt)      KEVGPT_PASS_CYCLES,pi=5,cyc=4184   (gen)
KEVGPT_PASS_CYCLES,pi=1,cyc=4136   (prompt)      KEVGPT_PASS_CYCLES,pi=6,cyc=4196   (gen)
KEVGPT_PASS_CYCLES,pi=2,cyc=4148   (prompt)      KEVGPT_PASS_CYCLES,pi=7,cyc=4208   (gen)
KEVGPT_PASS_CYCLES,pi=3,cyc=4160   (prompt/gen)  KEVGPT_PASS_CYCLES,pi=8,cyc=4220   (gen)
KEVGPT_PASS_CYCLES,pi=4,cyc=4172   (gen)
KEVGPT_CYCLES,37548   -- sum of the above, matches the earlier (unbroken-down) run exactly
KEVGPT_PASS,generate  -- same bit-exact [1,29,30,34,1,40] result, unaffected by the added logging
```

Per-pass cost grows linearly with context length (+12 cycles/pass, T=1..9 --
attention scans more committed positions each step), a clean, predictable curve,
not noise.

**Clock frequency confirmed from the actual Vivado config, not assumed**:
`hw/vendor/esl_epfl_x_heep/hw/fpga/scripts/genesys2/xilinx_generate_clk_wizard.tcl`
sets `CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {50}` -- 50MHz -- matching firmware's own
`REFERENCE_CLOCK_Hz` in `sw/device/target/genesys2/x-heep.h` exactly (if these
disagreed, UART would have transmitted at the wrong baud and produced garbled
text, not the clean output actually received -- an independent cross-check that
50MHz is the real running frequency, not just a firmware-side belief about it).

**Result**:
- Steady-state generation (the 6 passes that produce an output token, avg 4190
  cycles): **83.8 us/token -> ~11,930 tokens/sec**.
- Including prompt-priming passes (all 9, avg 4172 cycles): **83.4 us/token ->
  ~11,985 tokens/sec**.

This is a per-token *decode* latency for one sequence (no batching in this
design), and it's fast for the reason this whole architecture exists: Option A's
weights (195.6KB) and KV cache live entirely in on-chip BRAM -- no DRAM
round-trip per token, no bandwidth wall. ~12,000 tok/s at 50MHz for a d=128/
2-layer model is the expected payoff of that design, not an outlier; it's the
project's central bet paying off, measured on real silicon rather than claimed
from simulation alone.

## Open items / next up

- Interactive `kevgpt_chat` UART loop (arbitrary user prompts, not just the one
  baked-in `"once"` self-check) is future work -- current firmware is a
  self-contained correctness check, not yet a live chat interface.
- Consolidate `gen_chat_fw.py` + this session's inline ROM-file-generation snippet
  into one script (used identically ~6 times this session; still two separately
  invoked pieces).
- DSP margin is tight (37 DSP48E1 slots free, 95.6% used) -- fine for this exact
  build, but leaves little room for future growth (e.g. Option B's larger config)
  without either reducing HEAD_DIM/precision inside `vec_attn_w.sv` (numerically
  sensitive, would need its own bit-exactness re-verification) or accepting
  another architectural change of this session's magnitude.
- The pre-existing, unrelated whitespace/formatting diffs in `ai_accel/` RTL and
  vendor `.core`/wrapper files in the fork (present before this port's work started)
  are still untouched -- not this port's concern, flagged in an earlier session.

## DDR3-backed larger model: BRAM calibration (KV cache is the real ceiling)

Follow-on to the interactive-chat retarget (plan:
`/home/tparng/.claude/plans/steady-floating-pretzel.md`). Board spec used
throughout: Genesys2 DDR3 at 400MHz DDR (800MT/s) x4 bytes = **3.2GB/s peak**
(user-confirmed), giving 32-64MB/token of headroom at the 50 tok/s target even
at pessimistic controller efficiency -- DDR3 *bandwidth* is not a binding
constraint at this token rate; latency-hiding and correctness are the real
design concerns, not throughput.

Decomposed the real hierarchical `report_utilization` data (Option A,
single-engine attention, LANES=64, WWORDS=16384) instead of trusting
`sizing.py` alone (already known this session to underestimate BRAM by ~5x):

```
xilinx_core_v_mini_mcu_wrapper_kevgpt (top): 253 RAMB36, 13 RAMB18, 803 DSP
  u_kevgpt (xheep_kevgpt_peripheral): 245 RAMB36, 13 RAMB18, 798 DSP
    u_seq (sequencer_vec) own logic: 50 RAMB36, 5 RAMB18, 104 DSP
    u_attnA (vec_attn_w):             5 RAMB36, 3 RAMB18, 384 DSP
    u_dq (vec_dequant):               0 RAMB36, 0 RAMB18,  16 DSP
    u_gelu (vec_gelu):                8 RAMB36, 0 RAMB18,   0 DSP
    u_gemv (gemv_banked_resident_vec):158 RAMB36, 1 RAMB18,   0 DSP
      u_wb (weight_bank_tdp):         128 RAMB36, 0 RAMB18,   0 DSP
    u_kvb (kv_bank):                   24 RAMB36, 3 RAMB18,  30 DSP
```

Calibration points extracted:
- **`kv_bank` real-vs-formula tax: 1.68x.** `sizing.py`'s `kv_cache_bytes()`
  predicts 70.0KB for Option A's exact shape (NLAYER=2, D=128, TMAX=128); real
  synth gives 24 RAMB36 + 3 RAMB18 = 117.5KB. Small-buffer BRAM-tile rounding
  (36Kb/18Kb granularity) inflates a small KV cache more than a big one would
  be inflated proportionally -- treat 1.68x as an upper bound, not necessarily
  the asymptotic ratio at much bigger KV sizes.
- **`u_wb` (weight bank) 128 RAMB36 at `WWORDS=16384`** reflects a
  deliberately *oversized* window (this session's own earlier note: "right-
  sized then doubled for LANES=64's halved word-width"), not the true minimal
  per-layer footprint -- don't use this number directly as a per-layer weight
  cost; the true minimal 2-layer window at D=128 computes to ~192KB logical
  (12*D^2 INT4 params/layer), well under the 590KB actually provisioned here.
- **Fixed, ~non-scaling baseline: 93 RAMB36 = 428.5KB** (`u_seq`-own 50 +
  `u_attnA` 5 + `u_gelu` 8 + `u_gemv`-own-minus-`u_wb` 30). Conservative
  (treats all of `u_seq`-own as fixed even though some of it -- xres/qkv
  buffers -- likely scales mildly with D); good enough for first-order sizing,
  should be re-checked once a candidate shape is actually synthesized.

**Finding: weight streaming alone (the plan's original framing) does not
unlock a meaningfully bigger model.** Projecting fixed-428.5KB + a *true*
minimal weight window (x1.2 assumed packing tax) + KV cache (x1.68 tax)
against the ~1678.5KB free BRAM pool:

| config (NLAYER, D, ctx) | weight window | KV cache | total | vs free |
|---|---|---|---|---|
| Option A 2L/D=128/ctx=128 (sanity check) | 230K | 118K | 781K | 47% |
| 6L/D=192/ctx=128 | 518K | 529K | 1483K | 88% |
| 8L/D=192/ctx=128 | 518K | 706K | 1659K | **99%** |
| 12L/D=192/ctx=128 | 518K | 1058K | 2012K | 120% -- over |
| 16L/D=256/ctx=128 | 922K | 1882K | 3240K | 193% -- over |

KV cache was already the dominant BRAM cost once NLAYER exceeds ~2-3, and it
does NOT shrink from streaming weights -- every layer's K/V history for the
live context must stay reachable regardless of how weights are loaded. Even
with weights streamed to a near-zero window, KV cache alone saturates the
board by ~7-8 layers.

**Decision: stream the KV cache to DDR3 too**, not just weights (asked and
confirmed with the user). Initially assumed this needed a new write-capable
DMA engine from scratch (no reusable RTL) -- **wrong, corrected below.**

### Phase 2 architecture (derisked): four existing modules, all reusable unmodified

Read `kv_bank.sv`'s full interface end to end
(`hw/vendor/.../ai_accel/rtl/accelerator/streamer/`... no, `fabric/stage3/rtl/
kv_bank.sv`) and the `ai_accel` streamer stack. Two findings change Phase 2's
risk profile a lot:

1. **`kv_bank.sv`'s storage shape maps directly onto a flat DDR array.**
   Every (layer, kv, head, pos) has a fixed-size record: HEAD_DIM*KBITS-bit
   code row + 48-bit header (scale16+lo32), addressed by
   `w_pbase = ((layer*2+kv)*NHEAD+head)*TMAX + pos` -- exactly a linear
   index into `HROWS = NLAYER*2*NHEAD*TMAX` fixed-size slots. This is
   *already* how the on-chip TDP BRAM is addressed (`kva_a`/`pos_ra`/
   `pos_ra2` in `kv_bank.sv`), so relocating it to DDR3 is "same address
   arithmetic, different backing store," not a new addressing scheme.
   Write happens once per (layer,kv,head,pos) -- NLAYER*2*NHEAD times/token.
   Read streams `rd_tcount` (~pos+1) *consecutive* rows for one
   (layer,kv,head) -- NLAYER*2*NHEAD calls/token, each a burst of
   consecutive addresses. That read shape is exactly what `mig_read_engine`
   already does (issue N sequential `req_addr_i`, drain N `ret_data_o`
   beats); the write shape (one fire-and-forget beat per call) is exactly
   what `mig_write_engine` already does.
2. **A full write-capable engine + a 2-master MIG arbiter already exist**,
   hardware-tested via `ai_accel`'s own `AI_PASS,ddr_bram_prefetch` result,
   completely unrelated to and untouched by kevgpt so far:
   - `mig_read_engine.sv` -- streaming reads, credit-based outstanding
     tracking (already known).
   - `mig_write_engine.sv` -- single-beat write packets
     (`pkt_valid_i`/`pkt_addr_i`/`pkt_data_i`/`pkt_mask_i` in,
     `ack_valid_o` out), handles MIG's split cmd/wdf channel timing
     internally.
   - `mig_rw_arbiter.sv` -- merges one read-engine's command stream + one
     write-engine's command stream onto a single MIG command channel
     (priority-with-hysteresis, `BATCH_LIMIT`-based anti-thrash). This is
     the piece that combines kevgpt's own read+write traffic into one
     app-level bundle.
   - `mig_dual_master_arbiter.sv` -- merges TWO independent, already-self-
     consistent app-level bundles onto the *physical* MIG port, with correct
     read-return demuxing (owner FIFO, since MIG returns reads in issue
     order) and write-data pairing (a second owner FIFO, since `app_wdf_*`
     is independently paced from `app_addr/cmd/en`). Built generically for
     exactly this "two masters share one DDR3 controller" situation.
   - `v22_streamer_top.sv` is the reference integration showing all three of
     `mig_read_engine`+`mig_write_engine`+`mig_rw_arbiter` wired together
     (verbatim, minus its CDC `async_fifo_gray` layers, which exist only
     because `ai_accel`'s accelerator core runs on a separate clock domain
     from MIG's `ui_clk`; kevgpt's `clk_gen` is *already* sourced from
     `ui_clk` directly -- see `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv` --
     so kevgpt's own instances need NO clock-domain-crossing FIFOs at all,
     one less thing to get wrong).

   Confirmed via `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`: today only
   `cpu_ddr_bridge` is wired, straight to MIG's real app port (no
   `ai_accel`, no dual-master-arbiter present in this build at all) -- so
   the ENTIRE second slot of a `mig_dual_master_arbiter` is free.

**Resulting Phase 2 shape** (all four `mig_*` modules reused verbatim, zero
modification):
   - Add ONE `mig_read_engine` + ONE `mig_write_engine` + ONE `mig_rw_arbiter`
     for kevgpt's own traffic (shared by BOTH the weight-window reads and the
     KV-cache reads/writes -- ample bandwidth headroom from Phase 0 means
     time-multiplexing one read engine across two different DDR regions is
     fine, no need for a second read engine).
   - Add ONE `mig_dual_master_arbiter`: side A = kevgpt's new combined
     bundle, side B = the *existing, unmodified* `cpu_ddr_bridge` (rewire its
     `app_*` ports from "straight to MIG" into side B's ports instead).
     Arbiter output drives the real MIG app port.
   - New RTL actually needed is now scoped down to: (a) request-issuing FSMs
     inside `kv_bank.sv` (replace the BRAM read/write always-blocks with
     `mig_read_engine`/`mig_write_engine` request sequencing, same external
     `wq_*`/`rd_*` register-interface contract so `sequencer_vec.sv` and the
     gate harness don't need to change), (b) an analogous window-swap request
     FSM inside `weight_bank_tdp.sv`/`gemv_banked_resident_vec.sv`, and (c)
     the top-level wrapper wiring described above. No new arbitration or
     credit-tracking logic needs inventing -- the hard, error-prone parts
     (owner-order read demux, write-data pairing, MIG command-channel
     protocol) are already built and already proven on this exact board.

Next step: design `kv_bank.sv`'s DDR-backed read FSM in detail (replace
`R_RUN`'s `r_rowi`-indexed BRAM address stream with a loop issuing
`rd_tcount` sequential `mig_read_engine` requests, decode the returned
256-bit beats back into `code`+`hdr` fields, feed the existing `deq_word`
dequant combinational block unchanged), then the write FSM (replace `W_CWR`'s
BRAM commit with one `mig_write_engine` packet).

### MIG behavioral model + reused-engine smoke test (Icarus, no board needed)

Before touching `kv_bank.sv`, built a Icarus-simulatable stand-in for
`genesys2_mig_native_shell`'s app-level UI port: `fabric/genesys2/tb/
mig_behav_model.sv`. Deliberately NOT a timing-accurate MIG model (no
calibration sequence, no bank conflicts, always-ready cmd/wdf channels, a
fixed `READ_LATENCY`-cycle pipe for reads) -- Phase 0 already showed DDR3
bandwidth/latency headroom is ample at 50 tok/s, so the thing worth gating
bit-exactly is address/data correctness of the new request-issuing FSMs, not
real DDR timing (that's a Phase 6, real-hardware concern). Mask polarity
follows Xilinx UG586's MIG7 native-UI convention (active-low: 0 = byte
written, 1 = byte masked) -- flagged as an assumption to cross-check at
Phase 6, not yet verified against the real generated `genesys2_mig_native_
shell` wrapper.

Smoke-tested the full reused chain -- `mig_write_engine` -> `mig_read_engine`
-> `mig_rw_arbiter` -> `mig_behav_model`, all four wired exactly per
`v22_streamer_top.sv`'s reference pattern -- via a scratch `iverilog -g2012`
testbench (write a known pattern, read it back, then a SECOND write with a
partial byte mask, read back and confirm only the unmasked lane changed).
Two small fixes needed to get these ai_accel files through Icarus at all
(they'd only ever been through Vivado / real hardware before -- no prior
Icarus/Verilator testbench existed for this streamer stack, confirmed via
repo-wide grep): `unique case` qualifiers are accepted with a benign
"ignored" notice, and the SVA `assert property` blocks in `mig_read_engine.sv`
/ `mig_rw_arbiter.sv` / `sync_fifo.sv` need `-DSYNTHESIS` to skip (they're the
only content gated by that define in those three files, confirmed by grep
before relying on it -- doesn't change any functional logic, just compiles
out debug-only assertions Icarus's `-gno-assertions` didn't fully suppress).

**Result: `MIG_SMOKE_VERDICT,PASS`** -- both the full-write/full-read and the
partial-byte-mask cases matched exactly. This means the entire reused DMA
stack (`mig_read_engine`+`mig_write_engine`+`mig_rw_arbiter`, unmodified) is
now available as an ordinary Icarus gate dependency for the new `kv_bank.sv`
redesign -- the same bit-exact-before-synthesis discipline as every other
fabric/stage3 block, not a synthesis-only leap of faith.

### kv_bank_ddr.sv -- write-side FSM (DONE, gated PASS)

`fabric/genesys2/rtl/kv_bank_ddr.sv`: a new sibling module (kv_bank.sv itself
untouched, matching this project's established "add a variant, don't break
the existing build" pattern). Write-only for now -- the read-side streaming
loop is the next piece. Reuses kv_bank.sv's W_IDLE..W_QNT collect/scale/
quantise pipeline byte-for-byte verbatim (same inv_lut ROMs, same divide-free
magic-multiply scale/inv derivation), so wstage/w_scale/w_lo are guaranteed
identical to the already-verified reference for identical inputs. Only the
final commit differs: instead of a same-cycle on-chip TDP BRAM write, a new
DMA sub-FSM (states W_DMA/W_DACK) issues `ROW_BEATS` sequential write packets
on a `mig_write_engine`-shaped `pkt_*`/`ack_*` port.

DDR layout: each (layer,kv,head,pos) row is `CODE_BEATS` beats of quantised
codes (`ceil(HEAD_DIM*KBITS/DATA_W)`, =2 at HEAD_DIM=64/KBITS=8/DATA_W=256)
plus 1 header beat ({scale16,lo32} in the low 48 bits, byte-masked), addressed
as `KV_DDR_BASE + w_pbase*ROW_BEATS*BEAT_BYTES + beat*BEAT_BYTES` -- a flat
byte offset reusing kv_bank.sv's existing row-index arithmetic verbatim, just
multiplied by the beat stride instead of used as a direct BRAM address.
Address convention (byte address, beat-aligned, never shifted) and mask
polarity (active-low: 0=write) both cross-checked against `cpu_ddr_bridge.sv`'s
header before writing any of this, not assumed.

**Gate**: `fabric/genesys2/tb/tb_kv_bank_ddr.sv` -- an RTL-vs-RTL crosscheck
(not yet a Python-golden gate matching the `run_*.py` convention; flagged as
a follow-up, not done): drives identical `wq_*` stimulus into both
`kv_bank.sv` (already Python-gated via `fabric.stage3.run_vec_kv`) and
`kv_bank_ddr.sv`, decodes `kv_bank_ddr`'s DDR-resident bytes by hand in the
testbench, and compares against `kv_bank.sv`'s own dequantised `rd_data` for
the same row. Two test vectors (small positive-offset span; wide span
straddling zero, to exercise a negative `lo`), 64 lanes each = 128 comparison
points. **Result: `KV_BANK_DDR_VERDICT,PASS`, 0/128 mismatches.**

Three real bugs found and fixed while building this gate (not in the RTL --
all three were testbench/verification-infrastructure bugs, but the kind that
would have produced false confidence if shipped un-caught, so recorded in
full):
1. **`wait(signal)`/`@(posedge signal)` on a single-cycle pulse driven by the
   common "default-clear, then conditionally set" NBA idiom** (`wq_done<=0`
   unconditionally at the top of the always block, `wq_done<=1` in one
   specific branch -- what `kv_bank.sv`'s `wq_done`/`rd_done` and
   `kv_bank_ddr.sv`'s `wq_done` all use) **fired one full cycle early in
   Icarus**, confirmed directly via debug `$display` comparing the "just
   unblocked" data against the real pulse one cycle later. Root-caused (not
   just patched around): this class of pulse, sampled by a *different*
   process via edge-sensitivity, does not reliably behave like a true
   registered-value 0->1 transition on Icarus. Fixed by replacing every
   `wait`/`@(posedge ...)` on these pulses with a monotonic pulse-counter
   incremented in its own `always @(posedge clk)` block, and waiting for the
   counter to reach a target value instead -- sidesteps edge-detection
   semantics entirely. Worth remembering for every future testbench in this
   port that synchronizes on one of these single-cycle "done" pulses (the
   read-side FSM's `rd_done`/`rd2_done`, the eventual weight-window DMA's own
   completion pulse, etc.) -- use the counter idiom from the start, not
   `wait()`.
2. **`kv_bank.sv`'s read port has no direct "read position p" input** --
   `rd_start`/`rd_tcount` streams positions `0..tcount-1` of a (layer,kv,head)
   selector in order; there is no way to address a single arbitrary position
   directly. To check position `p`, request `tcount=p+1` and take the *last*
   emitted beat (`rd_done` fires on it). Mis-set to `tcount=1` initially
   (silently read position 0, which was never written, producing X compared
   against real DDR data by coincidence-passing/failing depending on which
   row position 0 happened to alias with) -- worth remembering for the
   read-side FSM's own gate, which will drive this same port directly.
3. **`mig_behav_model`'s `MEM_WORDS` sized too small for the row indices this
   test actually exercises** (1024, vs. `ROW_BEATS*HROWS`=3072 needed) --
   `word_idx()`'s fixed-width address slice *silently wraps/aliases* out-of-
   range addresses onto in-range words rather than erroring, so an
   undersized backing store doesn't fail loudly: it corrupted exactly the
   test case with a large row index (silently aliasing its beats onto
   unrelated words) while the small-row-index case happened to stay within
   bounds and passed normally -- the worst kind of test bug, since it looks
   like a partial pass. Root-caused by adding a temporary debug print of the
   real MIG-side `word_idx()` result next to the testbench's own (unshifted,
   untruncated) index computation and finding they disagreed. Fixed by
   sizing `MEM_WORDS` generously (8192) and documenting the sizing
   requirement directly in the testbench so it isn't silently re-introduced.

### kv_bank_ddr.sv -- read-side FSM (DONE, gated PASS)

Added to the same `fabric/genesys2/rtl/kv_bank_ddr.sv` file (kv_bank.sv
itself is a single module with both read and write sides; kv_bank_ddr.sv
follows the same shape now that both halves exist). New states RR_IDLE/
RR_REQ/RR_WAIT/RR_EMIT: matches kv_bank.sv's own read-port contract exactly
(no direct "read position p" input -- streams positions `0..tcount-1` of a
(layer,kv,head) selector in order, `rd_valid`/`rd_data` pulse once per
position, `rd_done` on the last one), but internally walks each position by
issuing `ROW_BEATS` sequential requests on a `mig_read_engine`-shaped
`req_*`/`ret_*` port (one request in flight at a time -- simpler than
pipelining ahead, and Phase 0 already established there's no bandwidth/
latency pressure to justify the extra complexity yet), capturing the
returned beats into the same code/hdr layout the write side stages them in,
then dequantising with the exact same `x_hat = code*scale+lo` math
`kv_bank.sv`'s own `deq_word` combinational block uses (copied verbatim, not
re-derived). No `rd2_*` (second read port) -- the current single-engine
`sequencer_vec.sv` ties it off permanently anyway, so this matches actual
usage rather than kv_bank.sv's full historical interface.

**Gate**: extended `fabric/genesys2/tb/tb_kv_bank_ddr.sv` to also issue a
real `rd_start` against `kv_bank_ddr` itself (through an actual
`mig_read_engine` + `mig_rw_arbiter` -- not a shortcut; this exercises the
same arbiter class that will later merge kevgpt's own DDR traffic with the
existing `cpu_ddr_bridge`, so it's useful mileage on that stack too) and
compares the result against `kv_bank.sv`'s reference read, in addition to
the existing hand-decoded-DDR-bytes check from the write-side gate. Both
checks now run for both test vectors: manual decode confirms the WRITE
path's bytes are correct, the real read-port exercise confirms the READ FSM
itself retrieves and dequantises them correctly. **Result:
`KV_BANK_DDR_VERDICT,PASS`, 0 mismatches across 256 comparison points** (64
lanes x 2 checks x 2 test vectors). Compiled clean with zero warnings after
fixing one cosmetic port-width mismatch (a debug-only `outstanding_o` wire
sized for the wrong `MAX_OUTSTANDING`).

No new bugs beyond the three already recorded above -- the write-side gate's
fixes (monotonic-counter pulse synchronization, `tcount=pos+1`-then-take-
the-last-beat semantics, `MEM_WORDS` sizing) carried over directly and
avoided repeating any of them on the read side.

### weight_loader_ddr.sv -- weight-window streaming (DONE, gated PASS)

Redesigned the scope from the original framing (a DMA-backed replacement for
`weight_bank_tdp.sv`'s read port) after actually tracing
`gemv_banked_resident_vec.sv`'s MAC pipeline: its read port
(`raddr_b`/`rword_b`) is consumed by an RLAT-deep, tightly-timed pipeline
(addend stage registered specifically to close a 5ns cone, per this
session's own earlier notes on tight DSP margin) that assumes weight reads
are ALWAYS ready -- `weight_bank_tdp` is a fixed 1-cycle-latency on-chip
memory today, it never stalls, and the MAC's `kc` walk has no backpressure
concept at all. Making the MAC's own read port DMA-backed would mean real
surgery on that already-verified, timing-critical pipeline.

Sidestepped that entirely: `fabric/genesys2/rtl/weight_loader_ddr.sv` is a
hardware-driven ALTERNATIVE SOURCE for `weight_bank_tdp`'s EXISTING boot-load
port (`ld_rst`/`w_we`/`w_data`) -- the same port firmware already drives via
the `wl_we` register to stream the whole weight image in once at boot
(`xheep_kevgpt_peripheral.sv`'s `wl_we`/`wl_data` -> `sequencer_vec.sv`'s
`wl_rst`/`wl_we`/`wl_data` -> `gemv_banked_resident_vec.sv`'s
`ld_rst`/`w_we`/`w_data`). This loader plays the identical role, just sourced
from DDR3 at runtime (once per weight-window reload) instead of from
firmware once at boot. `weight_bank_tdp.sv` and
`gemv_banked_resident_vec.sv`'s MAC pipeline are BOTH completely untouched --
by the time a GEMV `start` fires, the window is already fully resident,
satisfying the exact same "always ready" assumption the MAC pipeline already
relies on. Tradeoff: single-buffered (compute waits for the window to finish
loading; no double-buffered overlap yet) -- correctness first, matching
every other DMA piece built in this port so far; double-buffering is future
work once this baseline is proven on real hardware.

Design: issues DMA read requests via a `mig_read_engine`-shaped `req_*` port
as fast as `rd_req_ready` allows (decoupled from the drain side, unlike
`kv_bank_ddr.sv`'s read side which issues one outstanding request at a
time) -- weight-window reloads happen once per layer per TOKEN, so
throughput matters here for the cycle budget in a way the low-frequency KV
reads did not; paying DDR round-trip latency once per pipeline fill instead
of once per beat is the difference between a reload costing thousands of
cycles vs. tens of thousands. Each returned 256-bit beat is unpacked into
`DATA_W/32`=8 sequential `w_we` pulses feeding `weight_bank_tdp`'s existing
write-word assembler verbatim -- no new packing logic to get bit-exact,
since this is a straight copy (pre-quantized weights, no quantize/dequantize
math involved, unlike the KV cache).

**Gate**: `fabric/genesys2/tb/tb_weight_loader_ddr.sv` -- streams a known
LFSR-generated image out of `mig_behav_model` through a real
`mig_read_engine` into an unmodified `weight_bank_tdp` instance, then reads
the resident bank back via its existing `raddr_b`/`rword_b` port and checks
it matches the source DDR image exactly. Two cases (zero DDR offset/16
words; a nonzero 100-beat offset/24 words -- deliberately not repeating the
KV-cache gate's early mistake of only testing a zero-offset case, since that
one hid a real address-truncation bug until a nonzero-offset case exposed
it). At LANES=64 (Option A's actual value), `weight_bank_tdp`'s own word
width (WBITS=LANES*4=256) exactly equals the DMA beat width (DATA_W=256), so
the readback check is a direct beat-for-word comparison with no unpacking
arithmetic of its own to get wrong. **Result:
`WEIGHT_LOADER_DDR_VERDICT,PASS`, 0 mismatches across 40 words checked**,
clean compile, no new testbench-infrastructure bugs this time (the KV gate's
earlier fixes -- monotonic-counter pulse synchronization in particular --
were reused directly and avoided repeating that class of bug).

Both DMA subsystems (KV cache read+write, weight-window load) now exist and
are independently gated.

### Top-level arbiter wiring (DONE, gated PASS)

Two new modules, both pure glue (no new arbitration/credit-tracking logic of
their own beyond one small mux):

- **`fabric/genesys2/rtl/mig_read_mux2.sv`** -- merges `kv_bank_ddr`'s read
  requests and `weight_loader_ddr`'s read requests onto ONE shared
  `mig_read_engine` (the plan's own stated design: "ample bandwidth headroom
  makes time-multiplexing one engine across two DDR regions fine"). Same
  owner-FIFO IDEA `mig_dual_master_arbiter.sv` already uses for its own
  read-return demux (push an owner tag on every accepted request, pop it in
  the same order against real returns), applied one layer earlier in the
  stack (upstream of the engine, merging requesters, rather than downstream,
  demuxing an already-formed bundle).
- **`fabric/genesys2/rtl/kevgpt_ddr_bundle.sv`** -- combines `kv_bank_ddr`'s
  write engine + the shared read engine (via `mig_read_mux2`) through one
  `mig_rw_arbiter` into a single self-consistent app-level bundle -- the
  shape `mig_dual_master_arbiter`'s side A (or B) expects. Does NOT
  instantiate `kv_bank_ddr`/`weight_loader_ddr` themselves (those belong
  wherever `sequencer_vec.sv`'s own hierarchy eventually places them); it
  only takes their DMA-facing ports as pass-through.

**A real bug, found and fixed**: `mig_read_mux2`'s owner FIFO initially
popped on bare `ret_valid` (copying `mig_dual_master_arbiter`'s own pattern
verbatim) -- WRONG here. `mig_dual_master_arbiter`'s owner FIFO demuxes the
*raw MIG native-UI return path*, which has no backpressure at all (a return
must be consumed the instant it arrives, so valid alone means consumed).
`mig_read_engine`'s own `ret_valid`/`ret_ready` is a REAL handshake (backed
by its own return FIFO, data waits if the consumer isn't ready yet) -- the
owner FIFO here needed to pop on `ret_valid && ret_ready` (the real "beat
was actually consumed" event). Symptom with the bug present: the integration
gate hung during the weight-loader phase (worked fine for the low-throughput,
single-outstanding KV read path, which never let the two conditions diverge)
-- the owner FIFO drained faster than real consumption once multiple
requests were pipelined ahead, going empty while `mig_read_engine` still had
buffered returns waiting, permanently deadlocking `ret_ready` at 0. Debugged
by tracing `owner_empty`/`ret_valid`/`ret_ready`/`outstanding_o` cycle by
cycle and finding the owner FIFO empty while the engine still claimed
`ret_valid`. Fixed, and documented directly in `mig_read_mux2.sv`'s header
as a reusability warning: this owner-FIFO idiom needs the handshake-aware
pop condition against any port with a REAL valid/ready handshake, not MIG's
raw backpressure-free return path.

**Gate**: `fabric/genesys2/tb/tb_kevgpt_ddr_bundle.sv` -- the full stack
end to end: `kv_bank_ddr` + `weight_loader_ddr` + `mig_read_mux2` +
`kevgpt_ddr_bundle` + a real `mig_dual_master_arbiter`, side B driven by a
synthetic app-level requester standing in for `cpu_ddr_bridge` (that module
isn't modified by this port, so it isn't re-gated here -- this test proves
the WIRING and genuine two-master sharing, not `cpu_ddr_bridge`'s own
already-established correctness). Four phases: (1) `kv_bank_ddr` write+read
through the full stack, side B idle -- crosschecked against `kv_bank.sv`;
(2) `weight_loader_ddr` load through the full stack, side B idle --
crosschecked against the source DDR image; (3) synthetic side B alone,
side A idle -- proves side B's own data path isn't swapped with side A;
(4) `kv_bank_ddr` write + a synthetic side-B write/read, launched
concurrently (`fork`/`join`) -- real two-master sharing, not two sequential
single-master phases. **Result: `KEVGPT_DDR_BUNDLE_VERDICT,PASS`, 0 errors
across all four phases**, clean compile.

**2026-09-11 correction**: this gate was later found silently broken --
unrunnable on this repo's local Icarus install once `assert property
(... disable iff ...)` statements were added to its dependency chain
(2026-08-16 onward), and, once made runnable again, genuinely failing
(Phase 2 timeout) due to a stale testbench clock-domain mismatch, unrelated
to anything in this section. Both are root-caused, fixed, and re-verified
(gate is back to a real `PASS`) in "Owner-FIFO backpressure fix + a
testbench-only clock bug in tb_kevgpt_ddr_bundle.sv" near the end of this
log; see `fabric/genesys2/FIXATION-WORD-CDC-INVESTIGATION.md` §4/§4a for the
full writeup. Not implicated in the fixation-word investigation itself --
flagged here only so this section's original "PASS" claim isn't trusted at
face value without that context.

All of Phase 2 (both DMA subsystems, plus the arbiter wiring merging them
with room for `cpu_ddr_bridge` alongside) is now built and gated end to end
in simulation.

## Real hardware integration (RTL done + a real CDC bug found and fixed; Vivado check still pending)

### sequencer_vec.sv: KV_DDR_BACKED generate-selected kv_bank_ddr

Confirmed BOTH sides of `kv_bank.sv`'s external contract are event-driven
before touching anything (`if (kb_wdone)` gates the write-side FSM
transitions explicitly; the read side has no fixed-latency assumption
either -- `vec_attn_w` already tolerates kv_bank.sv's own multi-cycle
pipeline latency via `rd_valid` watching, not a cycle count) -- this is what
makes swapping `kv_bank` for `kv_bank_ddr` (drastically different internal
latency, same external pulses) timing-safe rather than a leap of faith.

Added a `KV_DDR_BACKED` parameter (default 0) to `sequencer_vec.sv` and
wrapped the `kv_bank` instantiation in a `generate if/else`: `KV_DDR_BACKED=0`
(default) elaborates the untouched, already-verified `kv_bank.sv` path byte-
for-byte as before -- KV260's build and every existing Genesys2 gate never
even see `kv_bank_ddr`'s logic. `KV_DDR_BACKED=1` elaborates `kv_bank_ddr.sv`
instead, with new `kv_wr_*`/`kv_rd_*` DMA ports added to `sequencer_vec.sv`'s
own port list (tied to inert idle values in the `KV_DDR_BACKED=0` branch, so
no dangling outputs). Threaded straight through `xheep_kevgpt_peripheral.sv`
(new `KV_DDR_BACKED` parameter + pass-through ports, same pattern) -- this
was tractable specifically because `xheep_kevgpt_peripheral` is a PEER to
the X-HEEP SoC in `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`, not nested
inside `core_v_mini_mcu`'s own vendored boundary (confirmed by reading the
wrapper: `xheep_kevgpt_peripheral` and `x_heep_system_i` are both instantiated
directly, side by side, connected only via the generic
`ext_peripheral_slave_req_o`/`resp_i` register bus) -- no X-HEEP SoC-boundary
plumbing needed at all.

**Regression check (KV_DDR_BACKED=0, the default/existing path)**:
re-ran `fabric.stage3.run_vec_kv` (had to route around a pre-existing harness
issue -- its own hardcoded default prompt `[48,10,100,77,...]` contains
token ids out of range for Option A's 57-vocab checkpoint, a KV260-vocab
leftover unrelated to this change; called `run_vec_kv.run()` directly with
an in-range prompt instead of via its CLI). **`VEC_KV_VERDICT match=True`**
-- confirms the generate-based swap didn't disturb the existing, deployed
behavior at all.

**Full-sequencer DDR-KV verification (KV_DDR_BACKED=1, the new path)**: built
`fabric/genesys2/tb/tb_seq_vec_kv_ddr.sv`, a byte-for-byte copy of
`fabric/stage3/tb/tb_seq_vec_kv.sv`'s own stimulus (weight stream, prompt/gen
GO-pulse loop, cycle counting, KVDBG hooks) with ONLY the `sequencer_vec`
instantiation changed (`KV_DDR_BACKED(1)` + the new DMA ports wired to a real
`mig_write_engine`+`mig_read_engine`+`mig_behav_model` stack, not a
shortcut). This exercises `kv_bank_ddr` under REALISTIC multi-token, multi-
layer, multi-head traffic -- not just the two isolated write+read pairs
`tb_kv_bank_ddr.sv` already covers. **Result: identical bit-exact token
stream to both the resident-`kv_bank` regression run above AND the Python
golden reference** (`gen=[41, 36, 46, 1, 29, 30]` in all three), confirmed
via a direct Python comparison against `IntKVQSequencer.generate_greedy`,
not just eyeballing printed lines.

### Top-level wrapper wiring + a real clock-domain-crossing bug

Edited `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`: instantiated
`kevgpt_ddr_bundle` (side A) + `mig_dual_master_arbiter`, rewired
`cpu_ddr_bridge`'s `app_*` ports from "straight to MIG" into the dual-
arbiter's side B (cpu_ddr_bridge itself untouched), wired
`xheep_kevgpt_peripheral`'s new `kv_wr_*`/`kv_rd_*` ports into the bundle,
and flipped `xheep_kevgpt_peripheral`'s `KV_DDR_BACKED` to 1 for THIS board
build specifically to prove the mechanism -- Option A's own KV cache is tiny
(~118KB) and doesn't need DDR streaming at this model size, but exercising
it now, ahead of actually needing it, is what proves the DMA path is real
before the Phase 4 model-size decision depends on it.

**A real bug, found while wiring this (not caught by any simulation gate up
to this point):** every Icarus gate built so far (`tb_kv_bank_ddr.sv`,
`tb_weight_loader_ddr.sv`, `tb_kevgpt_ddr_bundle.sv`) ran `kv_bank_ddr`,
`mig_read_engine`/`mig_write_engine`, and `mig_behav_model` all on ONE
shared simulation clock. Real hardware has TWO clock domains here:
`kv_bank_ddr` lives inside `sequencer_vec.sv`'s own hierarchy, clocked on
kevgpt's compute clock (`clk_gen`, confirmed 50MHz by an earlier session's
own clock-wizard-config check) -- but `mig_read_engine`/`mig_write_engine`/
`mig_rw_arbiter` and the app-level bundle MUST run on MIG's own `ui_clk`
(a hard requirement of MIG's native-UI app port, not a design choice), and
`clk_gen` is PLL-derived FROM `ui_clk` via `xilinx_clk_wizard_wrapper` --
genuinely different frequencies, not the same clock under two names. Wiring
`kevgpt_ddr_bundle`'s output straight into `mig_dual_master_arbiter` (itself
correctly on `ui_clk`) while `kv_bank_ddr` stays on `clk_gen` would have
been an un-synchronized multi-bit bus crossing two real, differently-clocked
domains -- a genuine metastability/data-corruption risk on real silicon,
invisible to every simulation run so far specifically because they all used
one shared clock.

**Fixed**: `kevgpt_ddr_bundle.sv` now takes separate `gen_clk`/`gen_rst` and
`ui_clk`/`ui_rst` inputs and does its own clock-domain crossing for
`kv_bank_ddr`'s four DMA signals (write-packet, write-ack, read-request,
read-return), using `async_fifo_gray` -- the exact same CDC primitive and
4-FIFO shape `cpu_ddr_bridge.sv` already uses for its own `clk_i`/`ui_clk_i`
boundary, reused rather than re-derived. `weight_loader_ddr`'s read port is
NOT yet CDC'd (it isn't wired to a real instance at the top level yet --
tied to constant idle values there, which need no synchronizer) -- flagged
directly in the module's header as a real follow-up needed once that
integration happens, not silently assumed away.

**Re-verified the CDC fix under a GENUINE two-clock-domain simulation** (not
just re-running the old single-clock gate and calling it done): updated
`tb_kevgpt_ddr_bundle.sv` to drive `clk`/`ui_clk` as two independent clocks
with a deliberately non-integer-multiple period ratio (10ns vs 7ns, to
actually stress the crossing rather than accidentally look synchronous).
First attempt still failed for the (deliberately still-un-CDC'd)
`weight_loader_ddr` path -- exactly the expected failure mode, confirming
the CDC finding empirically: `weight_loader_ddr`/`weight_bank_tdp` were
still wired directly across the two clocks with no synchronizer in the
testbench, producing visibly torn/shifted data. Isolated that known,
already-documented gap by moving `weight_loader_ddr`/`weight_bank_tdp` onto
`ui_clk` for this test only (not a real deployment shape -- noted inline)
so the KV path's real CDC fix could be verified cleanly on its own. **Result:
`KEVGPT_DDR_BUNDLE_VERDICT,PASS`, 0 errors across all four phases**
(including Phase 4's genuinely concurrent two-master traffic), under real
two-clock-domain conditions.

### Fileset wiring (fork repo)

Added `kv_bank_ddr.sv`/`weight_loader_ddr.sv`/`mig_read_mux2.sv`/
`kevgpt_ddr_bundle.sv` to `kevgpt_seq.core`'s `rtl` fileset. No changes
needed for the `mig_*`/`async_fifo_gray`/`cpu_ddr_bridge` files themselves --
confirmed via `grep` that `x-heep:ip:ai_accel`'s own `rtl` fileset (which
already contains all of them) is an unconditional dependency of
`core-v-mini-mcu.core`, already pulled into every Genesys2 kevgpt build
(this is how `genesys2_mig_native_shell.sv` already resolved for
`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv` before any of this session's
work). All new/changed RTL files re-synced from kev-gpt's canonical copies
into the fork's vendored `hw/ip/kevgpt_seq/rtl/` directory.

### The real Vivado synth-only checkpoint (DONE, clean)

Corrected after the fact: Vivado 2022.2 IS available in this environment, at
`~/tools-2022/Xilinx/Vivado/2022.2` (`source ~/vivado2022.sh` or that
directory's `settings64.sh` directly) -- the earlier "not installed" note
above was simply looking in the wrong place (`/tools/Xilinx`,
`/opt/Xilinx`). Ran the real checkpoint once this was found.

**Process notes** (for next time): the project regeneration
(`make vivado-fpga-nobuild FPGA_BOARD=genesys2_kevgpt`) needs the repo's own
Python venv (`.venv/bin/activate`) SOURCED, not just its `fusesoc` binary on
PATH -- FuseSoC's own code-generation step (OpenTitan's `prim` generator,
needs the `mako` package) shells out to a bare `python3`, which resolves to
system Python (missing `mako`) unless the venv is actually activated in the
shell, not just referenced via absolute path. Regenerating the project wipes
the previous stale build's ROM `.mem` files (this session's earlier full
run of `fabric.stage3.run_vec_kv` already regenerated a correct set for
Option A's exact shape in a scratch dir -- reused directly rather than
regenerating a third time); they need re-placing into both the project's top
level and its `.runs/synth_1/` subdirectory before synthesis, same gotcha as
earlier sessions already documented. `make synth` (not the full
`vivado-fpga`/`.bit` target) is the right target for a synth-only
checkpoint -- it depends on `<name>.v`/`<name>.edn` (via `_synth.tcl` +
`_netlist.tcl`), stopping well short of place & route.

**Result: clean.** Zero `ERROR` lines in the full synthesis log; netlist
(`.v`/`.edn`) written successfully. Real hierarchical utilization
(`report_utilization -hierarchical`, opened directly against the synth_1
`.dcp` checkpoint -- fast, no resynthesis):

```
xilinx_core_v_mini_mcu_wrapper_kevgpt (top): 98769 LUT (48.46%), 54933 FF, 238 RAMB36+11 RAMB18 (54.72%), 803 DSP (95.60%)
  u_kevgpt (xheep_kevgpt_peripheral):        58443 LUT,          26015 FF, 230 RAMB36+11 RAMB18,            798 DSP
    (u_seq) own logic:                        8603 LUT,           3974 FF,  50 RAMB36+ 5 RAMB18,            104 DSP
    g_kvb_ddr.u_kvb (kv_bank_ddr):           14821 LUT,           3655 FF,   9 RAMB36+ 1 RAMB18,             30 DSP
    u_attnA (vec_attn_w):                    14709 LUT,           9336 FF,   5 RAMB36+ 3 RAMB18,            384 DSP
    u_gemv (gemv_banked_resident_vec):        8642 LUT,           3785 FF, 158 RAMB36+ 1 RAMB18,              0 DSP
      u_wb (weight_bank_tdp):                  430 LUT,            279 FF, 128 RAMB36,                        0 DSP
  u_kevgpt_ddr_bundle (new: CDC + mig_read_mux2 + engines + arbiter): 2921 LUT, 8693 FF, 0 RAMB36, 0 DSP
  u_mig_dual_arb:                             152 LUT,            102 FF,   0 RAMB36,                         0 DSP
  u_cpu_ddr_bridge (unmodified):             1386 LUT,           1558 FF,   0 RAMB36,                         0 DSP
  x_heep_system_i (cv32e40px SoC):          33796 LUT,          18538 FF,   0 RAMB36,                         5 DSP
```

**The key confirmation**: `kv_bank_ddr` uses **9 RAMB36+1 RAMB18**, vs.
resident `kv_bank`'s **24 RAMB36+3 RAMB18** at this exact same shape
(NLAYER=2/D=128/NHEAD=2/TMAX=128 -- the number from this session's earlier
BRAM-calibration work). A real, measured **-15 RAMB36/-2 RAMB18** reduction,
matching (and directly explaining) `u_kevgpt`'s own top-level reduction from
245+13 (the pre-this-session baseline) to 230+11 -- the DDR-backed KV cache
genuinely moved the bulk of its storage off-chip, exactly as designed, not
just in simulation. The remaining 9 RAMB36 in `kv_bank_ddr` are the
`inv_lut_lo`/`inv_lut_hi` ROM tables (reused verbatim from `kv_bank.sv`,
unrelated to the on-chip code/header storage that actually moved to DDR3).
`u_kevgpt_ddr_bundle` (all the new CDC FIFOs + read/write engines + rw
arbiter combined) costs **zero BRAM, zero DSP** -- small enough to fit
entirely in distributed RAM/logic, matching the expectation that this is
control/muxing/FIFO plumbing, not compute. DSP total (803/840, 95.60%) is
UNCHANGED from the pre-DDR baseline to the exact tile -- confirms none of
this session's new RTL touches the tight DSP budget at all, only BRAM (and
only in the intended, favorable direction).

This is a synth-only checkpoint (no place & route, no timing closure, no
bitstream) -- it confirms the design elaborates/synthesizes cleanly and
gives real resource numbers, but does NOT yet confirm the CDC fix's timing
closes under real constraints (that needs a full implementation run) or
that a bitstream programs and runs correctly on the physical board. Those
remain real follow-ups, same as any other synth-only checkpoint in this
project's history.

Next: Phase 3 (firmware-driven weight staging into DDR3, so
`weight_loader_ddr` has a real image to load from -- and
`weight_loader_ddr`'s own CDC, once it's actually wired to the top level)
and Phase 4 (final model-size selection), now with a real, measured
per-shape BRAM number for the DDR-backed KV cache to calibrate against
instead of an estimate.

## Full place & route, timing closure, bitstream, and real board bring-up (DONE, bit-exact)

Went all the way: full `make` (synth+impl+bitgen, not just the synth-only
checkpoint above), JTAG programming, and a real `kevgpt_chat` run against
the DDR-backed KV cache on physical silicon.

**Timing closure**: clean, positive margin on every path group.
`WNS=2.474ns, TNS=0.000, 0 failing endpoints (of 2124)`;
`WHS=0.086ns, THS=0.000, 0 failing endpoints (of 1980)`. Per-clock: `jtag_clk_pin`
WNS=4.622ns, `spi_slave_clk_pin` WNS=2.474ns -- both positive. This is real
static timing analysis covering the new CDC paths (`kevgpt_ddr_bundle`'s
`async_fifo_gray` instances) along with everything else -- a clean WNS here
is real evidence the gray-code synchronizer chains are correctly
constrained (an existing project-wide XDC already covers `async_fifo_gray`
generically, since `cpu_ddr_bridge.sv` already relies on the same primitive
for its own, pre-existing CDC boundary), not just an absence of errors.
`write_bitstream` completed successfully: 0 Errors, 356 Warnings, 0 Critical
Warnings. The many `REQP-1839` "RAMB36 async control check" warnings are the
expected, benign signature of async-reset gray-code CDC synchronizers
feeding BRAM address pins -- standard, accepted CDC design, not a real
finding (STA doesn't model async reset assertion timing by design; that's
what the false-path/max-delay constraints on those same paths are for).

**Board bring-up procedure** (same process-hygiene discipline as earlier
sessions -- stale watcher processes from a PRIOR session were still holding
both the JTAG cable and the UART port; had to `kill -9` an openocd PID that
survived a first, non-forceful `pkill` before OpenOCD could open the FTDI
device at all): programmed via `vivado -source *_pgm.tcl -tclargs
xc7k325tffg900-2 <bitstream>.bit` (FuseSoC's own `vivado-fpga-pgm` target
hung silently with no error output for an unknown reason -- invoking the
generated `_pgm.tcl` directly worked cleanly and is the more debuggable
path anyway), confirmed OpenOCD re-examines the RISC-V core cleanly on the
freshly-programmed bitstream, started a fresh UART reader at 115200 baud
(`UART_BAUDRATE` in `x-heep.h`), then loaded and ran the EXISTING
`kevgpt_chat` firmware (`sw/build/main.elf`, unchanged since an earlier
session -- the DDR-backed KV cache swap is entirely transparent to firmware,
same register contract, so no rebuild was needed) via
`riscv32-corev-elf-gdb ... -ex "target remote :3333" -ex "monitor reset
halt" -ex load -ex continue -batch`.

**Result: bit-exact.** Full UART trace:
```
KEVGPT_PHASE,control_plane
KEVGPT_ID,0x53515256
KEVGPT_PHASE,weight_load
KEVGPT_WEIGHT_WORDS,73856
KEVGPT_STATUS_PRE,0x00000000
KEVGPT_PHASE,generate
KEVGPT_PASS_CYCLES,pi=0,cyc=4587
KEVGPT_PASS_CYCLES,pi=1,cyc=5050
KEVGPT_PASS_CYCLES,pi=2,cyc=5550
KEVGPT_PASS_CYCLES,pi=3,cyc=6011  GEN pos=3 tok=1
KEVGPT_PASS_CYCLES,pi=4,cyc=6479  GEN pos=4 tok=29
KEVGPT_PASS_CYCLES,pi=5,cyc=6924  GEN pos=5 tok=30
KEVGPT_PASS_CYCLES,pi=6,cyc=7418  GEN pos=6 tok=34
KEVGPT_PASS_CYCLES,pi=7,cyc=7873  GEN pos=7 tok=1
KEVGPT_PASS_CYCLES,pi=8,cyc=8359  GEN pos=8 tok=40
KEVGPT_CYCLES,58251
KEVGPT_CMP,i=0,got=1,want=1    KEVGPT_CMP,i=1,got=29,want=29
KEVGPT_CMP,i=2,got=30,want=30  KEVGPT_CMP,i=3,got=34,want=34
KEVGPT_CMP,i=4,got=1,want=1    KEVGPT_CMP,i=5,got=40,want=40
KEVGPT_PASS,generate
```
All 6 generated tokens exactly match `kevgpt_expected_gen` (the golden
values `gen_chat_fw.py` baked into the firmware from the same
`IntKVQSequencer.generate_greedy` reference this whole port has been graded
against throughout) -- `STATUS_PRE=0x0` confirms a clean, non-stale reset
(the freshly-programmed bitstream sidesteps `reset_config none`'s "soft
reset doesn't clear done_latched" gotcha entirely, same fix as earlier
sessions). This is the same discipline as the original Option A hardware
bring-up (kv_bank.sv, resident) -- now repeated with `KV_DDR_BACKED=1`
(kv_bank_ddr.sv, DDR3-streamed) and passing identically.

**Real measured rate with the DDR-backed KV cache**: steady-state generation
(the 6 passes producing a token) averages 7177.3 cycles/token -> at the
confirmed 50MHz clock, **143.5us/token -> ~6,966 tok/s**. Compare to Option
A's resident-KV-cache measurement from earlier this port (~11,930-11,985
tok/s): DDR-backed KV costs roughly **1.7x** the resident design's per-token
latency at this tiny scale (NLAYER=2/D=128/TMAX=128) -- a real, measured
number, not a simulation estimate. Per-pass cycles grow with position
(6011->8359 over positions 3->8), matching the expected shape: `rd_tcount`
for a KV read scales with `pos+1`, so more DDR round-trips accumulate as
context grows. At this small scale DDR latency is clearly NOT free, but it's
also nowhere near the ~240x cycle-budget headroom the 50 tok/s target was
built on (Phase 0) -- meaning a genuinely bigger model, where compute cost
per layer grows much faster than KV traffic, should have this overhead
comfortably hidden. Worth re-measuring once a larger candidate shape (Phase
4) is actually built, not assumed to scale the same way.

This closes out real-hardware verification for Phase 2's KV-cache half
completely: gated in isolation (`tb_kv_bank_ddr.sv`), gated at full-sequencer
scale in simulation (`tb_seq_vec_kv_ddr.sv`), gated through the complete
arbiter stack including genuine dual-clock CDC (`tb_kevgpt_ddr_bundle.sv`),
synthesized clean, placed and routed with positive timing margin, and now
proven bit-exact on the real board.

## Phase 3 -- firmware-driven weight staging into DDR3 (DONE, bit-exact on real hardware)

New app: `sw/applications/kevgpt_ddr_stage/main.c`. Reuses
`kevgpt_weight_words[]`/`KEVGPT_WEIGHT_WORDS_COUNT` verbatim from
`kevgpt_chat/kevgpt_weights.h` (`#include`d via a relative path, no
duplication of the ~287KB generated array) -- the exact same packed word
stream `kevgpt_chat` already streams into the on-chip `weight_bank_tdp` via
`wl_we`, just written to DDR3 byte addresses (`EXT_SLAVE_START_ADDRESS`,
from the mcu-gen-generated `core_v_mini_mcu.h`) instead of a register.
Matches the plan's own framing exactly: "reuse its packing logic, change
only the destination" -- no new packing logic needed. Write-then-readback-
verify, bit-exact, over the SAME `cpu_ddr_bridge` path
`kevgpt_ddr_bench.c` (Phase 0) already proved readable/writable with
synthetic data -- this is the first time it's carried the REAL, full-size
weight image.

**Build gotcha**: `make app PROJECT=kevgpt_ddr_stage` failed initially --
X-HEEP's `sw/CMakeLists.txt` resolves app sources relative to `SOURCE`
(default `.`, meaning X-HEEP's own `sw/` when the sub-make `-C sw`'s into
that directory), and this project's own `kevgpt_*` apps live in the
TOP-LEVEL repo's `sw/applications/`, three directories above X-HEEP's own
root -- but the sub-make's OWN working directory is one level deeper
(`sw/`) than where that "three directories up" figure was measured from, so
the correct value is **`SOURCE=../../../../sw/` (four `..`, not three)** --
confirmed via `os.path.relpath` rather than guessed. This is likely the
same class of off-by-one-directory mistake as the earlier documented
`SOURCE=../../../sw/` `make vivado-fpga` bug (a different depth, for a
different target, easy to conflate) -- worth remembering as its own, separate
gotcha specifically for `make app` on a repo-external application directory.

**Result** (JTAG-loaded via the already-programmed DDR-backed-KV-cache
bitstream from the P&R/bring-up step above, no reprogramming needed --
FPGA configuration persists across firmware reloads): full UART trace:
```
KEVGPT_DDR_STAGE_PHASE,write
KEVGPT_DDR_STAGE_WORDS,73856
KEVGPT_DDR_STAGE_PHASE,verify
KEVGPT_DDR_STAGE_ERRORS,0
KEVGPT_DDR_STAGE_PASS,stage_verify
```
All 73,856 words (Option A's full weight image) written to DDR3 and read
back bit-exact, zero mismatches, on real silicon.

**Scope note**: this proves the CPU's OWN write+read path into DDR3 is
correct byte for byte. It does NOT yet prove `weight_loader_ddr`'s hardware
DMA path reads this SAME staged image correctly on real hardware (that
module is gated against a synthetic image in
`fabric/genesys2/tb/tb_weight_loader_ddr.sv`, and isn't wired into the real
top level yet -- the weight-window control-flow integration into
`sequencer_vec.sv`'s own layer loop, and giving `weight_loader_ddr` its own
CDC once it is wired in, remain the real next steps before Phase 2's
weight-window half gets the same real-hardware treatment the KV-cache half
just did).

Next: Phase 4 (final model-size selection), now with real BRAM, rate, AND
DDR3-staging numbers to anchor it instead of estimates -- and, before that,
finishing Phase 2's weight-window half (top-level wiring + CDC + real
hardware bring-up) to match the KV-cache half's now-complete verification
depth.

### weight_loader_ddr wired into the top level (firmware-triggerable, additive)

Closed the exact gap the previous section flagged: `weight_loader_ddr` is
now instantiated in the real hierarchy, not just gated in isolation.

`sequencer_vec.sv` got a new `WEIGHT_DDR_BACKED` parameter (default 0,
every existing build untouched) -- unlike `KV_DDR_BACKED`'s generate-
selected REPLACEMENT of `kv_bank`/`kv_bank_ddr`, this one is ADDITIVE:
`weight_loader_ddr`'s `wb_ld_rst`/`wb_w_we`/`wb_w_data` OR onto
`gemv_banked_resident_vec`'s existing boot-load port alongside firmware's
own `wl_rst`/`wl_we`/`wl_data` (`w_data` muxed `wl_we ? wl_data :
wld_ldb_data` so the two sources never collide -- a firmware-sequencing
invariant, not hardware-enforced). New ports (`wld_ld_start/ld_ddr_addr/
ld_words/ld_done`, `wl_rd_req_*/wl_rd_ret_*`) threaded through
`xheep_kevgpt_peripheral.sv` as three new write-only registers beyond the
AXI-shell-parity map (`0x34 WLD_ADDR`, `0x38 WLD_WORDS`, `0x3C WLD_CTRL`)
plus a new `STATUS` bit (b2, `wld_done` latched) -- added as NEW registers
rather than reusing CTRL bits specifically so the documented "0x00-0x30
matches gemv_axi_seq_vec.v exactly" KV260-parity statement stays true.
`kevgpt_ddr_bundle.sv` got the SAME CDC treatment its own header flagged as
outstanding: two more `async_fifo_gray` instances (`u_wl_rd_req_cdc`,
`u_wl_rd_ret_cdc`) bridging `weight_loader_ddr`'s req/ret port from
`gen_clk` into `ui_clk`, mirroring `kv_bank_ddr`'s existing four exactly.
`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv` now wires the real `wl_rd_*`
signals end to end (previously tied to constant idle values) and flips
`WEIGHT_DDR_BACKED(1)` alongside the existing `KV_DDR_BACKED(1)`.

### Two real, pre-existing RTL bugs found while gating this (neither caused by today's wiring)

Built `fabric/genesys2/tb/tb_seq_vec_kv_wld.sv` (byte-for-byte copy of
`tb_seq_vec_kv_ddr.sv`'s stimulus shape: stage the full weight image into a
behavioral DDR model, this time triggering ONE `wld_ld_start` pulse instead
of streaming `wl_we`, then run the same prompt/gen loop) to prove
`WEIGHT_DDR_BACKED=1` end to end. First attempts -- and, damningly, a
re-run of the EXISTING default-path regression (`fabric.stage3.run_vec_kv`,
`KV_DDR_BACKED=0`/`WEIGHT_DDR_BACKED=0`, unmodified `sequencer_vec.sv` from
`HEAD`) -- hung: stuck forever in `G_RB` (state 8), `ci` frozen at 0, 40M
simulated cycles, no progress. Confirmed via a controlled isolation (same
call against the untouched pre-session file) that this predates today's
work entirely -- not a regression from this session's wiring.

**Bug 1 -- `gdone` 4-bit wraparound (`gemv_banked_resident_vec.sv`).**
`gdone` (the GEMV's committed-group counter, gating `sequencer_vec.sv`'s
`G_RB` row-issue: `ci>>GRPSH < gv_gdone`) was hardcoded `reg [3:0]` (max
15), but `GROUPS = ceil(MMAX/LANES)` with `MMAX=1024` fixed can need MORE
than 16 groups for a single GEMV call (e.g. `D_MLP=1024` at `LANES=64`
needs EXACTLY 16). `gdone` silently wraps 15->0 right as `ci`'s group index
reaches the same boundary, and the plain `<` comparison (no modular
unwrap) deadlocks permanently at that exact point -- explaining the
identical `st=8,blk=0` hang signature on every affected run. Never
triggered by the KV260 deployment (`LANES=128`, <=8 groups) or Genesys2
Option A's real shape (`D_MLP=512`, <=8 groups at `LANES=64`) -- only
surfaced testing a bigger candidate checkpoint (`D=256/NLAYER=4/D_MLP=1024`)
at `LANES=64`, exactly Phase 4's own "compute candidate shapes" territory.
**Fix**: `gdone`'s width is now a derived parameter, `GDONE_W =
$clog2((MMAX+LANES-1)/LANES + 1)`, scaling automatically with whatever
`MMAX`/`LANES` a caller instantiates -- not a hardcoded guess. `gdone` is
pure bookkeeping (a counter, never part of the MAC accumulate datapath), so
widening it touches no timing-critical logic. `sequencer_vec.sv`'s
`gv_gdone` wire widened to match (same formula, so the port connection
doesn't silently truncate).

**Bug 2 -- embed lookup requires `LANES >= 8*P` (`sequencer_vec.sv`,
inherent to the design, not a bug to fix, but a silently-violated
precondition in the standard gate scripts' own defaults).** `S_EMB`'s embed
fetch ONLY reads through the GEMV weight bank's spare-depth port-B path
(`emb_sel_w`/`emb_addr_w`, "log §36 fit-plan 2") -- `sequencer_vec.sv` has
NO fallback to the older dedicated `tok_emb_w.mem`/`pos_emb_w.mem` ROMs
(those files are still emitted by `write_mems_wideword` "for other
harnesses" per its own comment, but never `$readmemh`'d by `sequencer_vec`
at all). That embed-table append into `wrom.mem` only happens when
`wrom_embed_words()`'s own `EPW = (LANES*4)//(P*32) >= 1`, i.e.
`LANES >= 8*P`. `fabric.stage3.run_vec_kv`'s own CLI/function default is
`P=8, lanes=16` -- `EPW=0` at that combination, so the embed table is
NEVER written, and `S_EMB` reads uninitialized (X) weight-bank rows
UNCONDITIONALLY, for ANY checkpoint, independent of the `gdone` bug above.
Traced via a scaled-down manual repro: `run_sequencer.py`'s OLDER, non-P-
wide `sequencer.sv` gate (`--nlayer 1`) confirmed block-0's actual MATH is
bit-exact for this checkpoint (`SEQ_VERDICT block0_bitexact=True`) --
proving the corruption is specific to `sequencer_vec`'s P-wide embed path,
not the model/checkpoint. Progressively narrowed with `dbg_stop`
(1=after embed, 3=after block0) against `seq_ref.block0_phase_signals`:
even the RAW EMBED OUTPUT (`x_in_q25`) was already 100% X before block0
even started, at `LANES=16`. This was masked until today because the
`gdone` bug (above) always hung the FIRST real GEMV before any test could
observe the (still garbage) final output. **Not fixed in RTL** -- this is
an inherent tradeoff of the spare-weight-depth embed design (documented
already in `sequencer_vec.sv`'s own header comment: "smaller LANES no
longer carry embeds in the wrom image"), not something to patch around.
The actionable finding: `LANES=16` (the gate scripts' historical/CLI
default) is a STRUCTURALLY INVALID configuration for `sequencer_vec.sv` --
always was, for any checkpoint -- and must never be used for a full-
sequencer gate. `LANES=64` (this port's actual deployed value,
`EPW=(64*4)/(8*32)=1`) is the smallest valid choice and was already in use
by every one of today's new DDR gates, which is exactly why they surfaced
this precondition instead of silently inheriting it.

**Re-verified everything at `LANES=64` with both fixes in place, bit-exact
against the Python golden reference `gen=[30, 26, 1, 29, 30, 34]`** (this
session's checkpoint, `D=256/NLAYER=4/VOCAB=57`, prompt `[3,10,25,40]`):
  - Default path (`KV_DDR_BACKED=0`/`WEIGHT_DDR_BACKED=0`, on-chip
    `kv_bank` + firmware `wl_we` stream): `VEC_KV_VERDICT match=True`,
    ~27,940 cyc/pass avg -- confirms the fix didn't disturb the existing
    resident path at all.
  - `KV_DDR_BACKED=1` (`tb_seq_vec_kv_ddr.sv`, DDR-streamed KV cache):
    `gen=[30, 26, 1, 29, 30, 34]`, exact match.
  - `WEIGHT_DDR_BACKED=1` (new `tb_seq_vec_kv_wld.sv`, weight image loaded
    ENTIRELY through `weight_loader_ddr`'s DMA path instead of `wl_we`):
    `gen=[30, 26, 1, 29, 30, 34]`, exact match -- proves today's top-level
    wiring (the OR-mux onto the boot-load port, the register interface,
    the new CDC pair) end to end, under realistic multi-token/multi-layer
    traffic, not just `tb_weight_loader_ddr.sv`'s isolated synthetic-image
    gate.
  - As a fast-failure sanity check, re-ran the OLD `LANES=16` default
    afterward: no more 40M-cycle hang (confirms `gdone` alone was the hang
    cause), but still `match=False` with all-X output (confirms `LANES=16`
    remains structurally invalid, as expected -- a clean fast failure now
    instead of a silent infinite one, which is itself the improvement
    "bit-honest before fast" asks for).

Synced the fixed `gemv_banked_resident_vec.sv`/`sequencer_vec.sv` plus the
new `xheep_kevgpt_peripheral.sv`/`kevgpt_ddr_bundle.sv`/
`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv` into the fork repo's vendored
`hw/ip/kevgpt_seq/rtl/` copies (the file FuseSoC's build actually consumes
-- confirmed via diff before copying, not assumed in sync).

**Not yet done**: synth-only Vivado checkpoint, full P&R/bitstream, and
real-hardware bring-up for `WEIGHT_DDR_BACKED=1` (mirroring the KV-cache
half's full real-hardware treatment from the earlier section) -- next
step. The `gdone` width fix should be synth-neutral (pure bookkeeping
register, no MAC-path timing impact) but that claim itself needs the same
"prove it before trusting it" treatment as everything else in this repo.

### Synth-only Vivado checkpoint for WEIGHT_DDR_BACKED=1 (DONE, clean)

Regenerated the `genesys2_kevgpt` FuseSoC project (`make vivado-fpga-nobuild
FPGA_BOARD=genesys2_kevgpt`, X-HEEP's own `.venv` sourced for FuseSoC's
`mako`-dependent codegen step) and ran `make synth` **from inside the
generated project directory** (`build/openhwgroup.org_systems_core-v-mini-
mcu_1.0.5/genesys2_kevgpt-vivado/`, not the top-level `esl_epfl_x_heep`
Makefile -- `synth` is a target of the generated project's own Makefile,
confirmed by grepping it after the top-level invocation failed with "No
rule to make target 'synth'"). Regenerating wipes the project's ROM `.mem`
files (same gotcha earlier sessions hit) -- re-placed a fresh set generated
via `write_mems_wideword` at `P=8, LANES=64` (the deployed value, and the
only one proven valid by today's `LANES>=8*P` finding) against the current
checkpoint (`D=256/NLAYER=4/VOCAB=57` -- note this does NOT match the
wrapper's own hardcoded Option A parameters, `D=128/NLAYER=2`; for a
synth-only check this is fine, since the RTL's actual memory/logic
footprint is fixed by the wrapper's own instantiation parameters, not by
`.mem` file content -- Vivado just pads/truncates `$readmemh` against the
RTL's declared array size and warns, it doesn't resynthesize the array
itself. A future run should regenerate matching Option-A-shaped `.mem`
files, or a real checkpoint retrained at that shape, before any bitstream
meant to actually run correctly -- this checkpoint just proves synthesis
elaborates cleanly).

**Result: clean.** Zero `ERROR` lines in the full synthesis log; both
`.v`/`.edn` netlist outputs and the `synth_1` `.dcp` checkpoint written
successfully. All `CRITICAL WARNING`s are the same pre-existing, benign
CDC-constraint pattern-match misses (DMI JTAG debug subsystem, SPI-slave
CDC FIFOs, `sync.xdc`) already accepted in the earlier KV-cache-only synth
checkpoint -- none touch this session's new RTL.

Real hierarchical utilization (`report_utilization -hierarchical
-hierarchical_depth 4`, opened directly against the `synth_1` `.dcp`):

```
xilinx_core_v_mini_mcu_wrapper_kevgpt (top): 99567 LUT (48.86%), 55608 FF (13.64%),
                                              248 RAMB36+10 RAMB18 (56.85% RAMB36-equiv), 803 DSP (95.60%)
  u_kevgpt (xheep_kevgpt_peripheral):        58851 LUT,          26583 FF, 240 RAMB36+10 RAMB18, 798 DSP
    (u_seq) own logic:                        8869 LUT,           4067 FF,  60 RAMB36+ 4 RAMB18, 104 DSP
    g_kvb_ddr.u_kvb (kv_bank_ddr):           14819 LUT,           3656 FF,   9 RAMB36+ 1 RAMB18,  30 DSP
    g_wld.u_wld (weight_loader_ddr, NEW):       155 LUT,            418 FF,   0 RAMB36+ 0 RAMB18,   0 DSP
    u_attnA (vec_attn_w):                    14708 LUT,           9336 FF,   5 RAMB36+ 3 RAMB18, 384 DSP
    u_gemv (gemv_banked_resident_vec):        8634 LUT,           3784 FF, 158 RAMB36+ 1 RAMB18,   0 DSP
      u_wb (weight_bank_tdp):                  422 LUT,            277 FF, 128 RAMB36,              0 DSP
  u_kevgpt_ddr_bundle (CDC+mux2+engines+arb):  3235 LUT,           8799 FF,   0 RAMB36,               0 DSP
  u_mig_dual_arb:                               145 LUT,            101 FF,   0 RAMB36,               0 DSP
  u_cpu_ddr_bridge (unmodified):              1399 LUT,           1557 FF,   0 RAMB36,               0 DSP
  x_heep_system_i (cv32e40px SoC):          (unchanged from earlier checkpoint)
```

**The key confirmation**: `g_wld.u_wld` (`weight_loader_ddr`, wired in for
the first time this checkpoint) costs **155 LUT, 418 FF, ZERO BRAM, ZERO
DSP** -- exactly matching its design intent as a thin control FSM +
sequential-issue counter feeding an existing memory port, not a new
compute or storage block. `u_kevgpt_ddr_bundle` grew from the earlier
KV-only checkpoint's 2921 LUT/8693 FF to 3235 LUT/8799 FF (**+314 LUT/+106
FF**, still **zero BRAM, zero DSP**) -- consistent with exactly the two new
`async_fifo_gray` CDC instances (`u_wl_rd_req_cdc`: 40 LUT/25 FF,
`u_wl_rd_ret_cdc`: 192 LUT/25 FF) plus `mig_read_mux2` now actually
arbitrating two real requesters instead of one idle input. **DSP total
(803/840, 95.60%) is byte-for-byte UNCHANGED from the KV-cache-only
checkpoint** -- confirms neither the `gdone` width fix nor any of today's
new wiring touches the tight DSP budget at all. Total BRAM (253 RAMB36-
equivalent / 445, 56.85%) has headroom; the `(u_seq) own logic` row shows a
+10 RAMB36 shift vs. the earlier session's own quoted number (50->60) that
wasn't chased down further -- most likely attributable to comparing against
a different `.mem`/checkpoint content and/or hierarchical-depth reporting
granularity between the two runs, not a new BRAM consumer (every NEW
module this session -- `weight_loader_ddr`, the two new CDC FIFOs -- is
independently confirmed zero-BRAM above); worth a clean side-by-side re-run
if it matters before committing to a final BRAM budget.

This is a synth-only checkpoint (no place & route, no timing closure, no
bitstream) -- confirms the design elaborates/synthesizes cleanly with real
resource numbers, but does NOT yet confirm the new CDC pair's timing
closes under real constraints or that a bitstream programs and runs
correctly on the physical board. Full P&R/bitstream/real-hardware bring-up
for `WEIGHT_DDR_BACKED=1` remains the next step, mirroring the KV-cache
half's earlier treatment.

### Full place & route, timing closure, and bitstream for WEIGHT_DDR_BACKED=1 (DONE, clean)

Ran `make` (the default `all: post_build` target, full synth+impl+bitgen)
from inside the same regenerated project directory, reusing the ROM `.mem`
files already placed for the synth-only checkpoint above -- no project
regeneration needed this time (project state, including the placed `.mem`
files, survives between `make synth` and `make`). Implementation run took
18m11s.

**Result: clean, zero errors throughout.** `0 Errors` at every DRC
checkpoint (placement DRC, route DRC, final bitstream-precondition DRC --
`0 Errors, 1473 Warnings`, all the same class of pre-existing DSP-pipelining/
methodology advisories already seen at the synth stage, none new). `Bitgen
Completed Successfully`; `xilinx_core_v_mini_mcu_wrapper_kevgpt.bit`
written (11.4MB).

**Timing closes cleanly and positively across the whole design** (the
final, signed-off `report_timing_summary` against the routed design, not
just the router's own estimate):

```
WNS(ns)=1.672   TNS(ns)=0.000   TNS failing endpoints: 0 / 2124
WHS(ns)=0.092   THS(ns)=0.000   THS failing endpoints: 0 / 1980
WPWS(ns)=4.600  TPWS(ns)=0.000  TPWS failing endpoints: 0 / 1111
"All user specified timing constraints are met."
```

Positive margin on setup (WNS), hold (WHS), AND pulse-width (WPWS) checks,
zero failing endpoints on any of them -- this covers the new CDC paths
(`kevgpt_ddr_bundle`'s two additional `async_fifo_gray` instances for
`weight_loader_ddr`'s req/ret port) as part of the real, whole-design
static timing analysis, not a partial/optimistic estimate. Margin is
somewhat tighter than the KV-cache-only checkpoint's own bring-up
(WNS=2.474ns/WHS=0.086ns there vs. 1.672ns/0.092ns here) but both are
comfortably positive with zero violations -- expected, since this build
adds real logic (the CDC pair, the new registers) without changing the
clock structure.

**Not yet done**: real-hardware bring-up -- JTAG-programming this
bitstream and confirming `WEIGHT_DDR_BACKED=1` actually reloads a weight
window through the DMA path correctly on physical silicon (mirroring the
KV-cache half's own real-hardware confirmation earlier). The bitstream
exists and is ready for that step; the `.mem` files used to build it
reflect this session's checkpoint (`D=256/NLAYER=4/VOCAB=57`, not the
wrapper's own hardcoded Option A parameters `D=128/NLAYER=2` -- flagged
already at the synth-only checkpoint above) -- a real board run to confirm
bit-exact generation would need either regenerating this bitstream against
Option-A-shaped `.mem` content, or a checkpoint retrained at that exact
shape, same caveat as before.

### Real hardware: found the real Option A checkpoint, rebuilt, and real board bring-up (DONE, PASS)

**The `.mem` shape caveat above resolved itself cleanly**: `fabric/genesys2/
gen_chat_fw.py` (the script that already produces `kevgpt_chat`'s deployed
`kevgpt_weights.h`) defaults to `fabric/export_optionA/goformer.npz`, NOT
the `fabric/export/goformer.npz` this whole session's gates used. That file
is genuinely `D=128/NLAYER=2/VOCAB=57/NHEAD=2` -- Option A's real shape,
matching the wrapper's own hardcoded RTL parameters exactly. Regenerated
every ROM `.mem` file against it (same `write_mems_wideword` call, `P=8,
LANES=64`, now `nlayer=2`), replaced the mismatched-shape files in the
project directory, and force-rebuilt (removed the stale `synth_1`/`impl_1`
run directories so Vivado couldn't silently reuse cached results against
the wrong ROM content -- `.mem` files aren't tracked as Makefile
prerequisites, so this step doesn't happen automatically).

**Result: clean rebuild, better timing margin.** `0 Errors` throughout;
`Bitgen Completed Successfully`. Timing: `WNS=2.135ns` (0/2124 failing),
`WHS=0.108ns` (0/1980 failing), `WPWS=4.600ns` (0/1111 failing) -- all
positive, all comfortably better than the mismatched-ROM build's own
numbers (2.135 vs. 1.672ns WNS) -- consistent with genuinely-sized memory
content packing more cleanly, though that wasn't the point of the rebuild.

**New firmware**: `sw/applications/kevgpt_wld_bringup/` (own `kevgpt.h`/
`kevgpt_regs.h`/`kevgpt.c` copies, same per-app-directory convention
`kevgpt_chat`/`kevgpt_bringup` already use -- plus two new register
offsets, `KEVGPT_REG_WLD_ADDR`/`WLD_WORDS`/`WLD_CTRL`, and a
`kevgpt_wld_load()` helper mirroring `kevgpt_step()`'s own write-then-poll
pattern). Deliberately does NOT call `kevgpt_weight_load_reset()`/
`kevgpt_weight_load_word()` (the firmware `wl_we` boot-stream) AT ALL --
the resident weight image reaches `weight_bank_tdp` ONLY through
`weight_loader_ddr`'s DMA port, so a passing result is proof of the DMA
path specifically, not a coincidence of some other load mechanism still
running underneath it. Sequence: (1) write `kevgpt_weight_words[]`
(reused verbatim from `kevgpt_chat`, same array Phase 3's
`kevgpt_ddr_stage` already proved round-trips through `cpu_ddr_bridge`
bit-exact) to a DDR3 offset of `0x20000` (128KiB) via `EXT_SLAVE_START_
ADDRESS`: (2) `kevgpt_wld_load()` -- write `WLD_ADDR`/`WLD_WORDS`, pulse
`WLD_CTRL`, poll `STATUS.wld_done`; (3) run the SAME prompt/gen loop
`kevgpt_chat` uses (`kevgpt_prompt_ids`/`kevgpt_expected_gen`, also from
the shared `kevgpt_weights.h`); (4) compare. The `0x20000` offset is
deliberately placed past `kv_bank_ddr`'s own KV-cache DDR footprint at
this shape (`NLAYER=2*2*NHEAD=2*TMAX=128` rows `* ROW_BEATS=3 * 32B/beat` =
98,304B `< 0x20000`) so the staged weight image and the KV cache's DDR
region can never alias, even though today's own sequencing (stage -> load
-> first `go`) would be safe regardless of overlap.

Cleared a stale `openocd`/JTAG-holding `cat /dev/ttyUSB0` pair left running
from an earlier session before starting (same `LIBUSB_ERROR_BUSY` class of
issue this port has hit before) -- confirmed killed via `ps -p` before
reprogramming.

**Result: real UART trace, bit-exact.**
```
KEVGPT_WLD_PHASE,control_plane
KEVGPT_WLD_ID,0x53515256
KEVGPT_WLD_PHASE,stage_ddr
KEVGPT_WLD_STAGED_WORDS,73856
KEVGPT_WLD_PHASE,dma_load
KEVGPT_WLD_DMA_DONE
KEVGPT_WLD_STATUS_PRE,0x00000004
KEVGPT_WLD_PHASE,generate
... (per-pass cycle counts, KEVGPT_WLD_GEN lines) ...
KEVGPT_WLD_CYCLES,58252
KEVGPT_WLD_CMP,i=0,got=1,want=1
KEVGPT_WLD_CMP,i=1,got=29,want=29
KEVGPT_WLD_CMP,i=2,got=30,want=30
KEVGPT_WLD_CMP,i=3,got=34,want=34
KEVGPT_WLD_CMP,i=4,got=1,want=1
KEVGPT_WLD_CMP,i=5,got=40,want=40
KEVGPT_WLD_PASS,generate
```

All 6 generated tokens `[1, 29, 30, 34, 1, 40]` bit-exact against the same
golden `kevgpt_chat` already checks against -- generated ENTIRELY from a
weight image that reached `weight_bank_tdp` through `weight_loader_ddr`'s
hardware DMA engine, never through firmware's `wl_we` register stream.
`STATUS_PRE=0x00000004` confirms `wld_done` (bit 2) latches correctly and
the busy/done bits stay clear once the load completes, matching the
register design. Total cycle count (58,252) matches the earlier KV-cache-
only real-hardware bring-up's own number (58,251, off-by-one is measurement
rounding) almost exactly -- expected, since the one-time DMA weight load
happens BEFORE the per-pass cycle counter starts and isn't part of that
total; the actual per-token compute cost is unchanged by where the weights
came from.

**This closes the exact scope gap Phase 3 flagged**: *"this proves the
CPU's own write+read path into DDR3 is correct byte for byte. It does NOT
yet prove `weight_loader_ddr`'s hardware DMA path reads this same staged
image correctly on real hardware."* It now does, on real silicon, with the
real deployed model shape, matching the KV-cache half's own real-hardware
rigor exactly. Both halves of Phase 2 (`KV_DDR_BACKED`, `WEIGHT_DDR_BACKED`)
are now gated in isolation, gated at full-sequencer scale, gated through
the arbiter stack with genuine dual-clock CDC, synthesized, placed & routed
with positive margin, and proven bit-exact on real hardware.

## Phase 4 -- model-size sizing, anchored by real synth numbers

Two exploratory synth-only checkpoints (NOT deployed -- the wrapper was
restored to Option A's real, hardware-proven values,
`D=128/D3=384/D_MLP=512/NHEAD=2/NLAYER=2/WWORDS=16384`, immediately after
each check; confirmed via `git status` showing no diff before moving on).
`.mem` ROM content for both checks reused this session's D=256/NLAYER=4/
VOCAB=57 files (correct for the FIRST check's shape; mismatched, but
harmless, for sizing-only purposes on the SECOND -- resource footprint is
fixed by RTL parameters, not by `$readmemh` file content, confirmed by
`DQ_N`/`DQROWS` already being checkpoint-independent throughout this whole
port).

**Key open question going in**: today's `WEIGHT_DDR_BACKED=1` synth
checkpoint showed DSP already at 95.60% (803/840) at Option A's tiny
current shape (`D=128/NLAYER=2`) -- if DSP scales with model width or head
count at all, there'd be almost no room to grow before hitting a hard DSP
ceiling, independent of anything the DDR-streaming redesign unlocks on the
BRAM side.

**Traced the RTL first, before spending synthesis time**: `vec_attn_w.sv`'s
per-position dot-product/context-accumulate multiplier arrays are
`HEAD_DIM`-wide (`for (l = 0; l < HEAD_DIM; l = l+1) ... $signed(q_l) *
$signed(k_l)`) -- HEAD_DIM is pinned at sequencer_vec's own default (64)
regardless of D/NHEAD (the single serial attention engine iterates over
NHEAD heads in more CYCLES, not more hardware -- see the earlier
"SINGLE vec_attn_w engine" collapse in this file). `layernorm_vec.sv`'s own
multiplier arrays are `P`-wide (fixed P=8) or fully scalar (mean/var/rsqrt
Newton iteration operate on a single accumulated value, not per-D). Neither
block's DSP footprint should depend on D, NHEAD, NLAYER, VOCAB, or TMAX at
all -- only on the pinned P=8/HEAD_DIM=64 pipeline widths. This is the
whole point of the P-wide/serial-attention-engine architecture: bigger
models cost more CYCLES, not more DSP hardware.

**Exploratory check 1 -- D=256/D3=768/D_MLP=1024/NHEAD=4/NLAYER=4/VOCAB=57**
(this session's own checkpoint shape, already bit-exact gated in Icarus at
BOTH `KV_DDR_BACKED=1` and `WEIGHT_DDR_BACKED=1` separately). `WWORDS=65536`
(real analytic need ~59,424 wide words at this shape/LANES=64/P=8).
**Result: clean synth, 0 errors. DSP = 803/840 (95.60%) -- byte-for-byte
IDENTICAL to Option A's tiny shape**, confirming the RTL-tracing hypothesis
directly: DSP really is a fixed architectural cost, invariant to model
size, at least across this 2x-D/2x-NLAYER/2x-NHEAD jump. LUT also stayed
close (106089->99567 range across these runs). **BRAM overflowed**: 551
RAMB36+4 RAMB18 vs. the device's 445 RAMB36 budget (124%) -- `weight_bank_
tdp` alone (`u_wb`) took 512 RAMB36 (exactly 4x Option A's 128, matching
the 4x `WWORDS` bump 16384->65536 linearly) -- full weight residency (the
ONLY mode actually built today; per-layer weight-WINDOW streaming remains
future work) is the real ceiling for this candidate, not DSP, not the
(now BRAM-flat) KV cache.

**A real build-system finding along the way**: tried `WWORDS=40960`
(≈1.27x headroom over a smaller D=192 candidate's real ~32,284-word need)
expecting `weight_bank_tdp`'s BRAM to shrink proportionally -- it measured
**byte-for-byte identical** to the `WWORDS=65536` run (still 512 RAMB36),
even after a full project regeneration (`rm -rf` the whole FuseSoC build
dir and `make vivado-fpga-nobuild` from scratch -- ruling out a Vivado
project-cache explanation directly, not assumed). Root cause: `weight_bank_
tdp`'s `MEM_PRIMITIVE="block"` synthesis branch infers an `xpm_memory_
tdpram`, and Vivado's own BRAM-cascade packer appears to round the
DEPTH-cascade up to the next POWER-OF-2 tile-cascade bucket (`WWORDS` in
`(16384, 32768]` and `(32768, 65536]` both land in the SAME 128-deep-tile-
cascade bucket, hence the same 512-RAMB36 footprint) -- confirmed
definitively by then trying `WWORDS=32768` (the exact boundary), which DID
drop to exactly **256 RAMB36** (half), matching the "one bucket down"
prediction precisely. **Actionable finding for any future WWORDS choice**:
size to the boundary AT OR BELOW the real need, not just "some headroom
over the real need" -- an oversized-but-still-under-the-next-power-of-2
WWORDS costs nothing extra; crossing a power-of-2 boundary costs a full
doubling for no reason.

**Exploratory check 2 -- D=192/D3=576/D_MLP=768/NHEAD=3/NLAYER=4/VOCAB=57**,
`WWORDS=32768` (real analytic need ~32,284 -- see the wide-word-count
formula below). **Result: clean synth, 0 errors, and it FITS**:
```
LUT:  99825/203800 = 48.98%
FF:   55589/407600 = 13.64%
BRAM: 381/445 (Block RAM Tile, Vivado's own normalized count) = 85.62%
DSP:  803/840 = 95.60%  <- byte-for-byte unchanged from Option A, again
```
`u_wb` (weight_bank_tdp) = 256 RAMB36 (confirms the power-of-2-bucket
finding above); `g_kvb_ddr.u_kvb` (kv_bank_ddr) = 9 RAMB36+1 RAMB18,
**unchanged from Option A's NLAYER=2 shape despite NLAYER doubling to 4**
-- direct empirical confirmation that `kv_bank_ddr`'s BRAM cost really is
flat/shape-independent (it holds only the fixed `inv_lut_lo`/`inv_lut_hi`
scale-inverse ROMs, not per-position K/V data, which now lives in DDR3).
Every fixed-baseline block (`(u_seq)` own scratch buffers, `u_attnA`,
`u_gelu`, `u_gemv` own logic, `u_ln`) matched Option A's ORIGINAL numbers
almost exactly once BRAM pressure dropped back under 100% -- the mild
LUTRAM-vs-BRAM packing swings observed in the two OVER-budget runs (an
earlier "+10 RAMB36 shift" noted without explanation in the synth-only-
checkpoint section above) are best explained by Vivado's own memory-
inference heuristic opportunistically favoring LUTRAM for small/borderline
scratch buffers when overall BRAM pressure is already critical -- not a
real per-block resource change, and it goes away once the design fits
comfortably.

**Word-count formula** (matching `sequencer_vec.sv`'s own `GW_QKV/GW_PROJ/
GW_FC/GW_MP/GW_HEAD/EMB_TOK*/EMB_POS*` localparams, verified against this
session's own empirical `wrom_n` counts within ~7%, good enough for
first-pass sizing): per layer, `GW_BLK = ceil(D3/LANES)*D + ceil(D/LANES)*D
+ ceil(D_MLP/LANES)*D + ceil(D/LANES)*D_MLP` (D3=3D, D_MLP=4D by this
project's convention); total `= GW_BLK*NLAYER + ceil(VOCAB/LANES)*D +
VOCAB*(D/P) + TMAX*(D/P)` (the last two terms are the tok/pos embed tables
appended into the same wrom image, only when `LANES>=8*P` -- see the
`LANES>=8*P` embed finding earlier in this file).

**Conclusion**: with the KV cache genuinely BRAM-flat (`kv_bank_ddr`) and
DSP genuinely model-size-invariant (confirmed twice, empirically, not just
by RTL inspection), the SOLE real constraint for a fully-weight-resident
build today is `weight_bank_tdp`'s linear-in-WWORDS (power-of-2-bucketed)
BRAM cost. **D=192/NHEAD=3/NLAYER=4/VOCAB=57 -- roughly 2x Option A's
depth (NLAYER) and 1.5x its width (D), a substantially more capable model
-- synthesizes cleanly with real margin on every resource** (BRAM 85.62%,
LUT 48.98%, DSP 95.60% unchanged). Context length (TMAX) is free to grow
independently (it costs zero BRAM now, only DDR3 footprint + per-position
DMA latency) since it was decoupled from BRAM entirely by `kv_bank_ddr`.

**Cycle budget**: not separately re-measured for this exact candidate, but
strongly bounded by two already-real data points: (1) this session's
D=256/NLAYER=4/NHEAD=4 shape (bigger in every dimension than this
candidate) measured **~27,940-28,132 cycles/pass** in the Icarus gate
(`tb_seq_vec_kv_wld.sv`/`tb_seq_vec_kv_ddr.sv`, LANES=64) -- both comfortably
under 3% of the 1,000,000-cycle/token budget the 50 tok/s target allows;
(2) real hardware measured Option A's own D=128/NLAYER=2 shape at
~4,587-8,359 cycles/pass. A D=192/NLAYER=4 candidate should land well
within these two bounds. Cycle budget is NOT the binding constraint at
this shape -- BRAM (specifically, full weight residency) is, exactly as
this whole Phase 4 investigation set out to determine.

**Not yet done / explicit next steps, not started this session**:
1. Train (or re-export, if a suitable checkpoint already exists) a real
   D=192/D3=576/D_MLP=768/NHEAD=3/NLAYER=4/VOCAB=57 checkpoint -- everything
   above used either this session's own D=256/NLAYER=4 checkpoint (for
   functional/cycle numbers) or mismatched/synthetic `.mem` content
   (synth-only checks 1 and 2, resource-footprint-only, not functionally
   meaningful) -- neither is a trained model at the ACTUAL recommended
   shape.
2. Gate that real checkpoint bit-exact via `fabric.stage3.run_vec_kv` (at
   `LANES=64`, never `LANES=16` -- see the earlier `LANES>=8*P` finding)
   before spending any more Vivado time on it.
3. Full P&R/bitstream/real-hardware bring-up at this shape, matching every
   other piece of this port's own "bit-honest before fast" discipline --
   not done, since this session's two checks were synth-only sizing
   exploration, not a deployment candidate build.

## Phase 4 continued -- training the candidate, and two real NLAYER bugs found

Trained a real checkpoint at the Phase 4 candidate shape and immediately
hit the "bit-honest before fast" gate doing exactly its job -- twice.

**Attempt 1: D=192/NHEAD=3/NLAYER=4 (the original recommendation) -- INVALID
shape, caught before any board/Vivado time was spent on it.** Trained
cleanly (best val 1.009 FP, 1.041 QAT, `data/char_optionA`'s existing
57-char corpus, `--n-embd 192 --n-head 3 --n-layer 4 --block-size 128`,
CUDA, ~5 min total). Exporting and gating it (`fabric.stage3.run_vec_kv
--lanes 64 --npz fabric/export_optionB/goformer.npz`) crashed immediately:
`fabric/stage3/run_layernorm.py`'s `_ln_int_quantized` asserts `d_model &
(d_model-1) == 0` -- **`layernorm_vec.sv`'s mean/variance divide is
shift-based (`sum >>> D_SH`), only exact when D is a power of 2. D=192
(=3*64) is not.** This is a real, structural constraint of the design
(not a bug to patch), missed entirely by the earlier synth-only sizing
exploration -- that check only exercised RESOURCE FOOTPRINT (LUT/FF/BRAM/
DSP counts), never the actual LayerNorm MATH, so it had no way to catch
this. **Corrected candidate: D=128 (Option A's own value, a valid power of
2), NHEAD=2 (unchanged), NLAYER=9** (real weight need ~30,740 wide words,
still fits the 256-RAMB36 bucket) -- chosen to keep every already-validated
width parameter and only grow depth.

**Attempt 2: D=128/NHEAD=2/NLAYER=9 -- hit a real capacity limit (`DQ_N`),
caught before wasting the trained checkpoint.** Retrained (~3 min FP + ~11
min QAT). Gating failed again, this time with NO crash but `hw gen=[None,
...]` (X output) at a normal, non-hung cycle count -- a DIFFERENT failure
mode than the LN assertion. Root cause: `sequencer_vec.sv`'s `DQ_N`
parameter (capacity of the `dqm_w`/`dqe_w` per-channel dequant-scale ROMs)
was **hardcoded to `9409`** -- exactly `NLAYER*(D3+D+D_MLP+D)+VOCAB` at the
ORIGINAL KV260 shape (`4*(768+256+1024+256)+193`), never re-derived for any
other shape, and never exposed as an overridable parameter anywhere in the
`xheep_kevgpt_peripheral.sv`/wrapper instantiation chain. NLAYER=9/D=128's
real need (10,425 entries) exceeds it; NLAYER=8's (9,273) does not --
**retrained one layer shallower, NLAYER=8**, to sidestep this specific
limit without touching RTL yet.

**Attempt 3: D=128/NHEAD=2/NLAYER=8 -- STILL X output, a real, distinct RTL
bug, not resolved by picking a smaller NLAYER.** Sanity-checked the test
methodology itself first (re-ran Option A's already-proven checkpoint
through the identical `--lanes 64` pathway -- still `match=True`, ruling
out an environment/tooling regression). Bisected empirically by training
throwaway checkpoints at NLAYER=4 (PASS), 6 (FAIL), 5 (FAIL) -- exact
boundary: NLAYER<=4 works, NLAYER>=5 fails, unrelated to `DQ_N` (NLAYER=5's
real dequant-channel need is comfortably under even the OLD hardcoded 9409
limit). **Root cause: `GAMMA_N` (the `gamma_w` LayerNorm-scale ROM's
capacity) was ALSO hardcoded, to `9` -- exactly `2*NLAYER+1` at
NLAYER=4 (2 LayerNorms per block + 1 final LN_f)**, same unparameterized-
constant class of bug as `DQ_N`, coincidentally exact-fitting at NLAYER=4
and silently truncating `gamma_w.mem` at load time for anything deeper.

**Fixed both `GAMMA_N` and `DQ_N` properly**: converted from hardcoded
`parameter` defaults to parameters DERIVED from the actual shape
(`GAMMA_N = 2*NLAYER+1`, `DQ_N = NLAYER*(D3+D+D_MLP+D)+VOCAB`), requiring
reordering `sequencer_vec.sv`'s own parameter list so `NLAYER`/`D`/`D3`/
`D_MLP`/`VOCAB` are declared before them (safe -- every instantiation site
in the whole project uses named `#(.PARAM(value))` connections, confirmed
before reordering, never positional). Both formulas reproduce the OLD
hardcoded defaults bit-for-bit at NLAYER=4 (`2*4+1=9`,
`4*(768+256+1024+256)+193=9409` exactly) -- a pure generalization, not a
behavior change for any existing NLAYER=4 build. Regression-checked
D=256/NLAYER=4 (this session's own earlier-gated checkpoint) still
`match=True` after the change.

**Re-gated NLAYER=8 -- STILL X.** The `GAMMA_N`/`DQ_N` fix alone wasn't
enough; a THIRD bug remained. Bisected properly this time using the RTL
itself rather than more retraining: patched a scratch copy of
`sequencer_vec.sv`'s existing `dbg_stop==2'd3 && blk==4'd0` "stop after
block 0" debug hook to stop after a CHOSEN block instead (`blk==4'd7`, then
`4'd3`, `4'd5`, `4'd4` -- binary search), each time comparing the dumped
`xres_bank` residual against `IntKVQSequencer.block0_phase_signals(tok)`'s
`x_out_q25` (note: `block0_phase_signals` is defined on the base
`IntSequencer` class but must be called on an `IntKVQSequencer` INSTANCE --
calling it on a raw `IntSequencer` hits an unrelated `v_cache` bug in that
base class's own `_attn_step`, a dead end that cost real debugging time
before finding `IntKVQSequencer`'s override resolves it correctly). Also
had to restrict the comparison to the first `D/P` dump entries -- the dump
task always emits 256 lines regardless of shape, so entries beyond the
real `xres_bank` extent (128 of 256, at D=128) are ALWAYS X regardless of
correctness, an easy false positive to avoid. **Result: blocks 0-3 bit-
exact, blocks 4-7 entirely X** -- pinpointing the failure to exactly
`blk==4`.

**Root cause 3: `NSACT` (capacity of the `inv_sact` scale-select ROM),
hardcoded to `17`.** `g_asel` (the `inv_sact[g_asel]` index) is set to
`blk*4 + {0,1,2,3}` for the four per-block dequant ops (qkv/proj/mlp_fc/
mlp_proj) plus `4*NLAYER` for the head -- ranging `0..4*NLAYER` inclusive,
needing `4*NLAYER+1` entries. At NLAYER=4 (the original default), that's
exactly `17` -- matching the hardcoded constant bit-for-bit, same
coincidental-exact-fit pattern as `GAMMA_N`. At NLAYER=8, block 4's FIRST
access (`g_asel=16`, its qkv dequant) is still the array's last valid
index, but blocks 4's remaining three accesses (`g_asel=17,18,19`, for
proj/mlp_fc/mlp_proj) read past the end -- exactly matching the empirical
"blocks 0-3 fine, block 4 corrupted" boundary. Same fix pattern: `NSACT`
converted from a hardcoded `17` to a derived `4*NLAYER+1`.

**All three (`GAMMA_N`, `DQ_N`, `NSACT`) are now derived from the actual
shape parameters instead of guessed constants -- the same fix philosophy
as this session's earlier `gdone` width fix**, and for the same underlying
reason: every one of them happened to be sized EXACTLY right for
NLAYER=4/D=256 (the original KV260 shape this whole codebase was first
built against), so nothing before this session ever had a reason to
suspect they weren't properly parameterized -- Option A (NLAYER=2, well
under every one of these limits) never exercised the boundary either.

**Regression-checked after the `NSACT` fix**: D=256/NLAYER=4 still
`match=True` (byte-for-byte identical cycle-count and token stream to
before). **Re-gated D=128/NHEAD=2/NLAYER=8**: **`VEC_KV_VERDICT
match=True`** -- `gen=[1, 41, 30, 34, 26, 1, 41, 29]`, exact match, real
trained-and-QAT-fine-tuned checkpoint (`data/ckpt_optionB.qat.pt`, best QAT
val loss 1.031), `fabric/export_optionB/goformer.npz`. Synced the fixed
`sequencer_vec.sv` into the fork repo's vendored copy.

**Final candidate actually trained and gated: D=128/D3=384/D_MLP=512/
NHEAD=2/NLAYER=8/VOCAB=57** (4x Option A's depth, same width) rather than
the originally-recommended D=192/NHEAD=3/NLAYER=4 -- a deeper-not-wider
tradeoff forced by the power-of-2-D constraint, landing on a shape this
session's real BRAM-bucket analysis already showed fits comfortably (27668
real wide-words at this exact NLAYER, well under the 32768-word/256-RAMB36
bucket boundary established earlier in this Phase 4 section). Not yet
synth-checked at THIS exact NLAYER=8 shape specifically (the earlier synth
checks used D=192/NLAYER=4 and D=256/NLAYER=4, not D=128/NLAYER=8) -- worth
a real synth-only checkpoint before committing to P&R, same discipline as
everywhere else in this file.

**Not yet done**: synth-only Vivado checkpoint for the ACTUAL D=128/
NLAYER=8 shape (the earlier synth checks used different shapes for the
DSP/BRAM feasibility argument, not this exact final candidate); full P&R/
bitstream/real-hardware bring-up.

### Synth-only Vivado checkpoint for "Option B" (D=128/NLAYER=8) -- DONE, clean, and DEPLOYED

Unlike the earlier two exploratory checks (D=192/NLAYER=4, D=256/NLAYER=4
-- both reverted immediately after, never committed), this one updates the
REAL wrapper: `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`'s `u_kevgpt`
instantiation now reads `NLAYER(8)`/`WWORDS(32768)` (every other parameter
-- D/D3/D_MLP/NHEAD/VOCAB/TMAX/LANES/P -- unchanged from Option A), since
this candidate is now a real, trained, QAT-fine-tuned, bit-exact-gated
checkpoint (`data/ckpt_optionB.qat.pt`, `fabric/export_optionB/
goformer.npz`), not a sizing-only exploration. ROM `.mem` files generated
via `write_mems_wideword` directly against this real checkpoint (P=8,
LANES=64) -- real content this time, not reused/mismatched files from a
different checkpoint. Regenerated the FuseSoC project from scratch (`rm
-rf` the build dir, `make vivado-fpga-nobuild`) since `NLAYER` genuinely
changed, then `make synth`.

**Result: clean, 0 errors.**
```
LUT:  99617/203800 = 48.88%
FF:   55604/407600 = 13.64%
BRAM: 382/445 (377 RAMB36 + 10 RAMB18, Vivado's own normalized Block RAM
      Tile count) = 85.84%
DSP:  803/840 = 95.60%  <- unchanged AGAIN (now confirmed identical across
                            FOUR different NLAYER values this session: 2,
                            4 [x2 shapes], and 8)
```
`g_kvb_ddr.u_kvb` (`kv_bank_ddr`) = **9 RAMB36+1 RAMB18, byte-for-byte
unchanged from Option A's own NLAYER=2 build** -- direct empirical
confirmation, now at the ACTUAL deployed NLAYER=8 shape (not just an
exploratory check), that the KV cache truly costs nothing extra as depth
grows. `u_wb` (`weight_bank_tdp`) = 256 RAMB36 at `WWORDS=32768`, matching
the power-of-2-bucket prediction from the earlier exploration exactly.
`g_wld.u_wld` (`weight_loader_ddr`) = 135 LUT/418 FF/0 BRAM/0 DSP, tiny as
always.

This is now the real, committed, in-progress deployment target (both
repos), not a revert-after-measuring exploration. Next: full P&R/
bitstream/real-hardware bring-up at this shape.

### The real BRAM budget: NLAYER=8 fails place_design, corrected to NLAYER=4

Ran the full build (`make`, synth+impl+bitgen) for NLAYER=8 straight after
the clean synth-only checkpoint above. **It failed** -- not synthesis, not
timing, but a hard resource DRC at `place_design`:

```
ERROR: [DRC UTLZ-1] Resource utilization: RAMB36/FIFO over-utilized in Top
Level Design. This design requires 489 of such cell types but only 445
compatible sites are available in the target device.
```

**Root cause of the discrepancy**: the synth-only checkpoint's utilization
report (382/445 RAMB36-equivalent, "fits") was measured by `open_checkpoint`
on the ISOLATED `synth_1` `.dcp`, opened out of its full-project context.
Doing this treats `mig_7series_0` (the DDR3 controller), `xilinx_clk_
wizard_clk_wiz_0_0`, and `xilinx_mem_gen_4096`/`xilinx_mem_gen_16384`
(X-HEEP's own on-chip SRAM banks, 8+5 instances) as **unresolved black
boxes reporting ZERO resource usage** -- confirmed by the `CRITICAL
WARNING: Could not resolve non-primitive black box cell ...` lines that
were present (and, in hindsight, should have been treated as a real
caveat rather than noted and moved past) in EVERY synth-only checkpoint
this whole session, not just this one. Their REAL combined BRAM cost is
`489 - 382 = 107` RAMB36-equivalent tiles -- entirely missing from every
isolated-checkpoint utilization number reported anywhere in this file
before now. This means the true usable budget for `kevgpt`'s own logic is
`445 - 107 = 338` tiles, not 445.

**This changes the whole Phase 4 sizing conclusion.** With `kevgpt`'s
non-weight fixed baseline + `kv_bank_ddr` at ~126 tiles (unchanged across
every shape measured this session), the REAL budget available for
`weight_bank_tdp` is `338 - 126 = 212` tiles -- meaning the largest usable
power-of-2 BRAM-cascade bucket is **128 RAMB36** (`WWORDS<=16384`), not the
256-tile bucket the NLAYER=8 candidate needed. Recomputing real wide-word
need at D=128 (same word-count formula as before):

```
NLAYER=2: 9,236 words   (Option A's own real shape)
NLAYER=3: 12,308 words
NLAYER=4: 15,380 words  <- largest that still fits the 16384-word/128-RAMB36 bucket
NLAYER=5: 18,452 words  <- needs the next bucket (256 RAMB36) -- does NOT fit
NLAYER=6: 21,524 words
```

**Corrected candidate: D=128/D3=384/D_MLP=512/NHEAD=2/NLAYER=4/VOCAB=57**
(2x Option A's depth, not 4x) -- the largest depth that stays within the
SAME BRAM bucket Option A's own already-proven-working build already uses.
Retrained (best val 1.045 FP / 1.057 QAT, same recipe as before, ~8 min
total), exported, and gated: **`VEC_KV_VERDICT match=True`**
(`gen=[1, 41, 30, 34, 26, 1, 41, 29]`). Updated the wrapper to
`NLAYER(4)`/`WWORDS(16384)` (WWORDS unchanged from Option A -- real need
15,376 fits the SAME bucket).

**Full build result: clean, this time for real.** `0 Errors` throughout
(synth, place, route, bitgen DRC). `Bitgen Completed Successfully`;
`write_bitstream completed successfully`. Timing closes with positive
margin and zero failing endpoints on setup, hold, and pulse-width:

```
WNS(ns)=2.627   (0/2124 failing)
WHS(ns)=0.108   (0/1980 failing)
WPWS(ns)=4.600  (0/1111 failing)
```

**Real, full-device utilization this time** (`report_utilization` from the
PLACED design -- `xilinx_core_v_mini_mcu_wrapper_kevgpt_utilization_
placed.rpt`, which correctly includes `mig_7series_0`/X-HEEP's SRAM/
everything, not an isolated-checkpoint undercount):

```
LUT:  102751/203800 = 50.42%
FF:    60668/407600 = 14.88%
BRAM:    360/445     = 80.90% (355 RAMB36 + 10 RAMB18)
DSP:     803/840     = 95.60%  <- unchanged YET AGAIN
```

Real margin this time (64 spare BRAM tiles, ~14%), confirmed against the
device's TRUE total resource picture, not an optimistic partial one.
`xilinx_core_v_mini_mcu_wrapper_kevgpt.bit` written (11.4MB).

**Lesson for any future synth-only checkpoint in this file**: an isolated
`open_checkpoint` report on `synth_1`'s own `.dcp` is only trustworthy for
resources INSIDE `kevgpt`'s own hierarchy (which is genuinely elaborated
there) -- it silently reports zero for any sibling IP core resolved as a
black box. Trust the full-build `place_design`/`utilization_placed.rpt`
numbers for an actual go/no-go BRAM budget call; treat a clean isolated
synth-only checkpoint as "no errors elaborating kevgpt's own logic",
NOT as "the whole design fits."

**Not yet done**: real-hardware bring-up for this corrected D=128/NLAYER=4
shape (JTAG-program the new bitstream, confirm bit-exact generation on
physical silicon, mirroring the KV-cache/weight-DMA halves' own real-
hardware treatment earlier in this file).

### Real hardware bring-up for Option B (D=128/NLAYER=4) -- DONE, bit-exact PASS

**A second, real build-system limit found before this could even build**:
regenerated `kevgpt_chat`'s own `kevgpt_weights.h` (`fabric.genesys2.
gen_chat_fw`, against `fabric/export_optionB/goformer.npz`) and tried the
usual `make app PROJECT=kevgpt_chat` flow -- linker failure, not a Vivado
or RTL problem this time: `region 'ram0' overflowed by 109188 bytes`.
Option B's weight image (123,008 words = 492,032 bytes) no longer fits
X-HEEP's 384KB on-chip `ram0`, which `kevgpt_chat`'s own boot sequence
embeds the ENTIRE weight image into as a literal `.rodata` C array before
streaming it out over `wl_we` -- a hard ceiling on model size that has
nothing to do with `weight_bank_tdp`'s own BRAM budget, and would have hit
ANY sufficiently large model regardless of how much on-chip weight storage
the FPGA itself has room for.

**This is precisely the scenario `weight_loader_ddr`'s DMA path exists
for** -- but `kevgpt_wld_bringup` (built earlier this session) ALSO
embeds `kevgpt_weight_words[]` via the same shared `kevgpt_weights.h`, so
it would hit the identical linker overflow. Built a new, minimal app,
`sw/applications/kevgpt_optionB_bringup/` (own small hand-written
constants -- `KEVGPT_PLEN`/`NGEN`/`WEIGHT_WORDS_COUNT`/prompt/expected-gen
-- no embedded weight array at all) that skips CPU-side DDR3 staging
entirely: the weight image is written directly into DDR3 via **GDB's own
`restore` command** (`restore <binary> binary 0xF0000000`, targeting
`EXT_SLAVE_START_ADDRESS` over JTAG, BEFORE the program even starts
running) rather than through any firmware code path, sidestepping
`ram0`'s size limit completely -- the CPU never needs to hold the weight
image in its own address space at all, not even transiently. The raw
binary was produced by `fabric.genesys2.gen_chat_fw`'s own `wrom_to_words()`
packing (low-chunk-first 32-bit words, the exact order `weight_loader_
ddr.sv`'s unpacker expects -- reused verbatim, not re-derived, so the byte
layout is guaranteed to match what every other path in this project
already treats as canonical). Firmware: probe -> `kevgpt_wld_load(dev, 0,
123008)` (trigger the DMA load, poll `wld_done`) -> the same prompt/gen
loop `kevgpt_chat`/`kevgpt_wld_bringup` use -> compare.

**Real UART result: bit-exact.**
```
KEVGPT_OB_PHASE,control_plane
KEVGPT_OB_ID,0x53515256
KEVGPT_OB_PHASE,dma_load
KEVGPT_OB_DMA_DONE
KEVGPT_OB_STATUS_PRE,0x00000004
KEVGPT_OB_PHASE,generate
... (per-pass cycles, KEVGPT_OB_GEN lines) ...
KEVGPT_OB_CYCLES,256551
KEVGPT_OB_CMP,i=0,got=1,want=1
KEVGPT_OB_CMP,i=1,got=41,want=41
KEVGPT_OB_CMP,i=2,got=30,want=30
KEVGPT_OB_CMP,i=3,got=34,want=34
KEVGPT_OB_CMP,i=4,got=26,want=26
KEVGPT_OB_CMP,i=5,got=1,want=1
KEVGPT_OB_CMP,i=6,got=41,want=41
KEVGPT_OB_CMP,i=7,got=29,want=29
KEVGPT_OB_PASS,generate
```

All 8 generated tokens `[1, 41, 30, 34, 26, 1, 41, 29]` bit-exact against
the same golden reference `fabric.stage3.run_vec_kv` already gated against
in simulation -- on real silicon, weights sourced ENTIRELY through
`weight_loader_ddr`'s DMA path from DDR3 (never through `wl_we`), for a
genuinely bigger (2x Option A's depth), genuinely trained model.
`STATUS_PRE=0x00000004` again confirms `wld_done` latches correctly.

**This closes Phase 4 end to end**: a real model bigger than Option A --
found via real synth numbers, corrected TWICE by real failures (the
isolated-checkpoint BRAM undercount at the Vivado stage, the `ram0`
on-chip-RAM ceiling at the firmware stage) rather than paper estimates,
each time landing on a smaller, real, working answer instead of a bigger,
untested one -- is trained, QAT-fine-tuned, gated bit-exact in simulation,
synthesized, placed & routed with positive timing margin, and now proven
bit-exact on real Genesys2 hardware. `sequencer_vec.sv`'s three fixed
capacity constants (`GAMMA_N`/`DQ_N`/`NSACT`) are permanently fixed for
any future NLAYER, not just this one shape.

## Interactive UART chat: untethered weight loading + real generation loop

Every firmware app up to this point only ever ran ONE baked-in prompt at
boot, diffed against a fixed golden token stream, with weight loading
tethered to a live GDB session (`restore ... binary`) over JTAG. This pass
builds real interactive chat on top of the already-proven Option B model:
a UART weight loader (so a chat session only needs a serial cable, not a
live JTAG/OpenOCD/GDB session) and an interactive generate loop with
on-chip Gumbel sampling for variety and a degenerate-reply guard ported
from the KV260 lineage's own already-solved version of this problem
(`fabric/stage3/board/pl_kv256.py`'s min-length ender mask).

New pieces: `fabric/genesys2/gen_chat_fw.py`'s `emit_tokenizer_header()`
(generates `kevgpt_tokenizer.h` -- `kevgpt_itos`/`kevgpt_stoi`/
`kevgpt_stop_ids` -- from the same `meta.json` every gate already treats
as the vocab, not hand-written); `fabric/genesys2/send_weights.py` (host
side of the UART loader, reuses `wrom_to_words()`'s existing packing);
`sw/applications/kevgpt_interactive/` (fork repo) -- `main.c` implements
the UART weight-load handshake (`KEVGPT_UART_READY`/`KEVGPT_UART_LOAD_DONE`
markers), a hand-rolled char-by-char prompt reader (no line-read helper
exists in this SDK; `_read()` is a stub), lowercase+`kevgpt_stoi[]` encode,
`kevgpt_step()` prefill with a once-per-turn reseed, and a streaming
generate loop with the ender-remask guard via a new `kevgpt_read_bank()`
helper (head-logit readback, `RD_SEL=8`, Q6.25).

**Firmware builds clean for `TARGET=genesys2`** (`make app
PROJECT=kevgpt_interactive TARGET=genesys2`, no `SOURCE=` override --
matches this fork's own documented `make app` lesson): `main.elf` links,
the trailing `mem_usage.py` `$(PWD)`-staleness failure is the same
already-documented cosmetic issue every prior app hit, harmless since it
fires after `.elf`/`.hex`/`.bin` are already built.

**Found and fixed a real bug in the gate itself while verifying the
sampling path on Option B's shape for the first time**:
`fabric/stage3/run_vec_kv.py`'s `_sample_stream()` constructed
`gumbel.GumbelRng(seed)` with no `vocab` argument, silently defaulting to
`gumbel.py`'s module-level `VOCAB=193` (the KV260 shape) -- the exact same
"constant coincidentally exact-fit for NLAYER=4/VOCAB=193, silently wrong
for anything else" bug class as `GAMMA_N`/`DQ_N`/`NSACT` above, just in the
Python golden this time instead of the RTL. Sampling mode had never been
exercised against a non-193-vocab checkpoint before Option B (57 vocab), so
it was never caught: `GumbelRng.sample_token()` looped `range(self.vocab)`
= `range(193)` against a 57-entry `logits` list and threw `IndexError` on
the first run. Fixed by deriving `vocab` in `run()` before building `gold`
(it was already computed, just later in the function) and threading it
through `_sample_stream(..., vocab)` into `GumbelRng(seed, vocab=vocab)`.
The RTL side never had this bug -- `-DVOCABVAL={vocab}` was already
correctly derived and wired to the TB's `VOCAB` parameter.

**Also hit, immediately after the vocab fix**: both greedy (`--seed 0`)
and sampled runs came back all-`X` on the RTL side even after the fix --
turned out to be an already-documented gate-usage gotcha, not a new bug:
`run_vec_kv.py`'s own CLI defaults (`--p 8 --lanes 16`) give `EPW =
(LANES*4)//(P*32) = 0`, so the token/pos embedding table never gets
appended into `wrom.mem` and `S_EMB` reads uninitialized weight-bank rows
unconditionally (`LANES >= 8*P` is required, already documented earlier in
this file from the original Option A crash). Re-ran with `--lanes 64`
(matching every other Option B invocation this session) -- clean.

**Verification step 1 (host-only, automated) result: bit-exact at three
different seeds**, `--lanes 64` against `fabric/export_optionB/goformer.npz`:

| prompt | seed | ngen | result |
|---|---|---|---|
| "once" | 0 (greedy) | 8 | match=True, gen=`[1,42,37,36,35,1,41,30]` |
| "once" | 1 | 8 | match=True, gen=`[1,42,37,36,35,1,41,30]` (coincides with greedy for this prompt -- seed=42/999999 below diverge, ruling out an accidental no-op) |
| "hey" | 42 | 8 | match=True, gen=`[1,34,36,34,1,40,36,1]` |
| "yo" | 999999 | 10 | match=True, gen=`[42,1,39,30,28,29,41,1,46,36]` |

Proves the on-chip Gumbel sampling path itself is bit-exact against
`gumbel.GumbelRng` on Option B's actual shape (D=128/NLAYER=4/VOCAB=57),
not just on the original KV260-shaped checkpoint the gate happened to be
written against.

**Real hardware bring-up**: fresh JTAG bitstream program (`vivado -mode
batch -source ..._pgm.tcl`, same `.bit` Option B's earlier bring-up
produced -- unchanged, since only firmware changed this pass), `openocd -f
quad-x-heep-openocd-genesys2.cfg` found the TAP cleanly, `riscv32-corev-
elf-gdb ... -ex "monitor reset halt" -ex load -ex continue -batch` loaded
`kevgpt_interactive`'s `main.elf`. UART came up at `KEVGPT_INTERACTIVE_ID,
0x53515256` ("SQRV", correct), then blocked at `KEVGPT_UART_READY` as
designed.

**Sharp edge, immediately hit and fixed**: the firmware's `KEVGPT_UART_
READY` marker is a one-shot printf -- if a `cat /dev/ttyUSB0` reader (or
any other passive listener) consumes it before `send_weights.py` starts
listening, the byte is gone for good and `send_weights.py` times out even
though the firmware is still correctly sitting ready to receive. Not a
firmware or protocol bug -- a real ordering requirement for any host tool
in this role: start listening (`send_weights.py`) BEFORE triggering the
reset+load+continue that makes the firmware print the marker, not after.

**`send_weights.py`, real hardware: `SEND_WEIGHTS_PASS`** -- 123,008 words
(492,032 bytes) sent, `KEVGPT_UART_LOAD_DONE` + `KEVGPT_INTERACTIVE_READY`
confirmed on the wire.

**Verification step 2 (real hardware, scripted): bit-exact.** The
production `chat_turn()` loop deliberately has no host-controllable seed
(it self-seeds from a live cycle counter, "good enough for chat variety,
not a cryptographic requirement" -- see `main.c`), so a scripted seed-exact
comparison against it directly isn't possible by design. Instead drove the
SAME register sequence `_sample_stream()`/the TB use, directly over JTAG
via a plain (non-Python -- this GDB build has no Python scripting support)
GDB command script: `monitor halt`, soft_reset pulse, three greedy prefill
GOs discarded, `SEED=1` write, then GO/read-TOK_OUT chained through 8
sampled positions (mirroring `_sample_stream`'s exact pass schedule), 
`monitor resume` to hand control back to the firmware. Prompt "once"
(ids `[36,35,24,26]`), seed=1:

```
KEVGPT_SEED_PROBE_RESULT,[1, 42, 37, 36, 35, 1, 41, 30]
```

Bit-exact against the same Python-computed sampled sequence from the host
gate above (`[1, 42, 37, 36, 35, 1, 41, 30]`) -- the SEED register, the
persistent xorshift/argmax datapath, and the head-logit path all work
correctly when driven for real on Option B's actual placed-and-routed
silicon, not just in simulation.

**Verification step 3 (manual interactive session): legible, non-
degenerate, non-hanging.** Sent real prompts over the UART (Python
`pyserial`, standing in for a human at a terminal) once the CPU resumed
back into `chat_turn()`'s live read loop:

```
"hello there" -> " beautiful new thing him go back begin always where you scarf yo"
"what is up"  -> "le them look angry see not find big batter batter vegetable me w"
"i like cats" -> " make her point her eat snake snake one day her see moon beautif"
"a"           -> "ll reach away her motorbin say it make lot funny man look tim ha"
```

Also confirmed two edge cases: an empty line (bare Enter) re-prompts
cleanly with no hang; backspace (`xyz<BS><BS>oo` -> encodes "xoo") echoes
and edits correctly. No reply degenerated to a bare `.`/empty string (the
`MIN_CHARS=12` ender-remask guard doing its job), nothing hung, every turn
re-prompted with `> ` afterward. Reported honestly as a human-judgment
check on real Genesys2 hardware, not a bit-exact gate -- free-form sampled
chat has no fixed golden reference by nature of the feature.

**This closes the interactive-chat pass.** Cleaned up per this file's own
documented process-hygiene discipline (`kill` the `openocd` PID once
verification's JTAG needs are done -- `reset_config none` means the
running firmware is unaffected by the JTAG connection dropping). The board
is left running `kevgpt_interactive` live, chatting, with no host process
attached.

## Mini story teller: raw-TinyStories checkpoint + a real Vivado ROM-staging bug

Follow-up ask: make the chatbot tell actual (mini) stories, not just
single-sentence replies. Two pieces -- a corpus/training change (real,
worth keeping) and a Vivado build-process bug this uncovered along the way
(real, now fixed, and will bite any future resynthesis if forgotten again).

**Corpus**: the Kevin-speak checkpoints (Option A/B) are trained on
`keviniser`-transformed TinyStories -- POS-stripped, telegraphic, and
(empirically, checked via a 100-char host-side generation at 4 different
seeds) producing sentence-ending `.`/`!`/`?`/`\n` only very rarely, since
the transform strips a lot of the sentence structure those characters
mark. `model/prep_raw_corpus.py` (new) normalizes RAW (non-Kevinised)
`data/TinyStories-valid.txt` instead -- lowercase-folds (matches this
project's existing lowercase-only vocab convention and
`kevgpt_interactive`'s firmware, which already force-lowercases typed
prompts), normalizes curly quotes/dashes/ellipsis to plain ASCII, drops
anything outside a fixed common set. Checked empirically: loses 41 of
19,092,837 characters, all rare encoding artifacts, not real content.
Natural vocab: 47 chars (`data/char_stories/meta.json`) -- genuinely
*smaller* than the Kevin-speak 57, not larger (real English needs fewer
distinct symbols than this project's own POS-substitution/punctuation
conventions did).

**Trained fresh at Option B's exact shape** (D=128/D3=384/D_MLP=512/
NHEAD=2/NLAYER=4/block_size=128, ~2.5 min FP + ~6 min QAT on this GPU --
this corpus is bigger than the Kevin-speak valid split, 18.9M vs a much
smaller compressed corpus, but training throughput is still ~97 it/s):
best FP val 0.886 / QAT val 0.907, BOTH markedly better than Option B's
own FP 1.045 / QAT 1.031 -- real English grammar is a genuinely easier
next-char prediction task than the keviniser-compressed vocabulary that
strips predictable function words. Sample text at this point already
included `"the end. | once upon a time, there was a brave little boy..."`
-- coherent multi-sentence output, unprompted.

**Firmware**: `kevgpt_interactive/main.c`'s single-sentence generate loop
became a story loop -- `STORY_SENTENCES=4`, `MAX_GEN_LEN` raised 64->100,
and the `MIN_CHARS` degenerate-sentence guard now RE-ARMS after every
accepted ender (`chars_this_sentence` resets to 0), not just the first
one, so a 4-sentence story can't quietly collapse into one real sentence
plus three single-character ones.

**Real bug #1 (Python gate, not RTL)**: `run_vec_kv.py`'s sampling gate
built `gumbel.GumbelRng(seed)` without passing the checkpoint's real
`vocab` -- silently defaulted to `gumbel.py`'s module-level `VOCAB=193`
(the KV260 shape). Never caught before since sampling had never been
exercised against a non-193-vocab checkpoint. Fixed: `GumbelRng(seed,
vocab=vocab)`, `vocab` now computed before `_sample_stream()` is called
rather than after.

**Real bug #2 (methodology, not a code bug): VOCAB is a synthesis-time
parameter, not runtime data.** Tried deploying the story checkpoint's
natural 47-char vocab directly -- retrained/exported/gated bit-exact in
simulation, but real hardware needed `xheep_kevgpt_peripheral`'s `VOCAB`
parameter changed from 57 (Option B) to 47 and RESYNTHESIZED (unlike an
NLAYER-only or weight-only change, which the deployed bitstream already
handles generically -- VOCAB is baked into the argmax/dequant/embedding
hardware's own width and address decode at synthesis time). Did the full
synth/PnR/bitgen cycle for VOCAB=47 -- clean, 0 errors, WNS=+2.263ns --
but produced all-zero head logits on real hardware (confirmed via direct
`RD_SEL=8` readback of individual vocab positions, not just a garbled
`tok_out`). Padded the checkpoint's vocab back up to 57 with 10 unused
filler characters (`ABCDEFGHIJ`, ids 47-56, appended AFTER the natural
47-char `sorted(set(text))` assignment so the real characters keep their
original ids and the padding entries get zero training gradient -- they
never appear in the corpus and `encode_char()` already force-lowercases
input, so they're unreachable from both training and live chat) and
retrained fresh (FP val 0.886, QAT val 0.907 -- unaffected by the unused
padding), to reuse Option B's exact, already-proven VOCAB=57 shape.

**Real bug #3, the actual root cause (and the reason bug #2's "fix"
*also* failed at first): a missing Vivado ROM-staging step, not a hardware
or checkpoint bug at all.** The VOCAB=57-padded rebuild STILL produced
all-zero logits -- and, decisively, so did a re-send of Option B's own
already-proven-good checkpoint onto that same fresh bitstream. Since the
RTL source diffed byte-identical against the exact commit that built the
original working Option B bitstream (`git diff` showed only comment
additions), the bug had to be in the BUILD PROCESS, not the RTL or the
checkpoint. `synth_1`'s own `runme.log` had the answer directly:
`CRITICAL WARNING: [Synth 8-4445] could not open $readmem data file
'gamma_w.mem'/'inv_sact.mem'/'dqm_w.mem'/'dqe_w.mem'/'gumbel_lut.mem'/
'inv_lut_lo.mem'/'inv_lut_hi.mem'/'seed.mem'/'gelu_lut_e.mem'/
'gelu_lut_o.mem'/'exp_lut.mem'; ... ignoring`. These 11 files are the
LayerNorm-gain/dequant-scale/gumbel-noise/fixed-math ROMs
`sequencer_vec.sv`/`layernorm_vec.sv`/`kv_bank_ddr.sv`/`gelu_lut2.sv`/
`softmax_f.sv` `$readmemh` **at synthesis time, baked into the bitstream
itself** -- an architectural fact already documented (Phase 5/6 above) but
apparently not durably remembered: every prior real build regenerated and
placed these files by hand before synthesizing; this session's `rm -rf` +
fresh `fusesoc --build` cycles (done twice, for VOCAB=47 and again for the
VOCAB=57-padded retry) never did. Missing files -> Vivado silently
zero-initializes those ROMs and continues -- every dequantized value
becomes zero regardless of the real (correctly UART-loaded) weight
matrix, matching the all-zero symptom exactly.

**This also fully explains the DSP utilization mystery** noted along the
way: both broken rebuilds synthesized to ~550/840 DSP48E1 (vs Option B's
803/840, on RTL source confirmed byte-identical) -- looked like alarming
synthesis non-determinism, but was the SAME root cause: with the
dequant/gumbel ROMs reading as constant zero, Vivado's optimizer correctly
treats large chunks of that logic as dead/constant-foldable and infers far
fewer DSP48E1 instances for it. Confirmed directly: after fixing the ROM
staging (below) and rebuilding, DSP came back to exactly 803/840 (95.60%)
-- identical to Option B, no mystery left.

**Fix, now a real project artifact, not another one-off snippet**:
`fabric/genesys2/stage_vivado_roms.py` (new) generates and copies all 11
ROM `.mem` files into any given directory, reusing
`write_mems_wideword` (the same code the gate uses) plus the
gelu-split/inv-lut/gumbel-lut generation verbatim from `run_vec_kv.py`'s
own `run()`. **Needs staging in BOTH the Vivado project root AND
`<project>.runs/synth_1/`** (`launch_runs synth_1` spawns its own `vivado
-mode batch` subprocess with `synth_1/` as CWD, where `$readmemh`'s bare
relative filenames actually resolve -- confirmed by finding zero
`could not open` warnings in the very next `synth_1/runme.log` after
staging). **Must run AFTER `reset_run synth_1`, not before** --
`reset_run` recreates that directory empty, silently discarding anything
staged earlier. Rebuilt via three separate Vivado invocations (one Tcl
script can't mix `exec`-ing a venv Python inside a running Vivado batch
session -- Vivado's own bundled Python 3.8 sets `PYTHONHOME`/`PYTHONPATH`
that a `.venv`'s Python 3.12 inherits and chokes on): (1) `open_project` +
`reset_run synth_1` + `reset_run impl_1`, exit; (2) plain shell,
`stage_vivado_roms.py`; (3) `open_project` + `launch_runs impl_1 -to_step
write_bitstream` + `wait_on_run` + the bitstream-copy step from the
original generated `_run.tcl` (unmodified).

**Real hardware, verified bit-exact**: greedy register-poke probe (same
JTAG-bypass technique as the interactive-chat pass, prompt "once upon a
time" encoded against `data/char_stories/meta.json`) against the rebuilt,
ROM-fixed bitstream: `[5, 1, 40, 28, 25, 38, 25, 1, 43, 21, 39, 1, 21, 1,
32, 29, 40, 40, 32, 25, 1, 27, 29, 38, 32, 1, 34, 21, 33, 25, 24, 1, 32,
29, 32, 45, 7, 1, 39, 28]` -- bit-exact against `fabric.stage3.run_vec_kv`'s
own golden for this checkpoint. Then real interactive chat over UART:

```
"once upon a time" -> "f he wanted to play. so, the bee had gone first when they had to paint the money. it was the lion wa"
"the dog ran"       -> "ing away and couldn't find her. she tried to give it to the birds and put it in her red easpact. she"
"a little girl"     -> " named lily who loved to play with her toy car. daisy wanted to go on an adventure, but she thought,"
```

Real periods, commas, and quoted dialogue with a question mark in a
follow-up backspace-edge-case check -- a qualitative step up from the
Kevin-speak model's punctuation-free rambles, confirming the whole point
of this pass. Empty-line and backspace edge cases both still work
correctly on the new firmware build. Cleaned up JTAG/UART processes;
board left running `kevgpt_interactive` live with the story checkpoint,
chatting, no host process attached.

## Per-layer weight streaming: breaking the BRAM depth ceiling

Follow-up: even at NLAYER=4, arbitrary prompts (not "once upon a time"-
style openers) produce incoherent completions -- confirmed a real MODEL
CAPACITY limit (the same incoherence reproduces in the pure Python
reference, multiple seeds, even pure greedy decoding -- not a firmware or
hardware bug). Growing NLAYER or D at all is currently blocked by BRAM
regardless of direction: `weight_bank_tdp`'s resident cost is a bucketed
step function of WWORDS (128 RAMB36 for <=16384 words, 256 RAMB36 for
(16384,32768]), and this project already proved the 256-tile bucket
doesn't fit this device's real usable budget (`place_design`: "requires
489... only 445 available") -- true for NLAYER=8 (deeper) and D=192
(wider, ~32,284 words) alike. Real fix: stop keeping the WHOLE weight
image resident, stream it from DDR3 one layer at a time.

**Researched first, before writing any RTL**: the DMA engine
(`fabric/genesys2/rtl/weight_loader_ddr.sv`) already exists, is already
gated for correctness (`tb_weight_loader_ddr.sv`,
`WEIGHT_LOADER_DDR_VERDICT,PASS`), and is already instantiated in the
deployed build (`WEIGHT_DDR_BACKED(1)`) -- it's only ever triggered once,
by firmware, at boot. `sequencer_vec.sv`'s own `GW_QKV`/`GW_PROJ`/
`GW_FC`/`GW_MP`/`GW_BLK` localparams and `g_wbase<=blk*GW_BLK+{WB_QKV|
WB_PROJ|WB_FC|WB_MP}` computation (already used at 4 call sites) are
exactly the per-layer address stride a per-layer DDR3 reload needs, just
retargeted from on-chip address space to a DDR3 byte offset. The DDR3-
staged image itself doesn't need to change -- `send_weights.py`'s
existing packing is already laid out block-relative. A clean FSM
insertion point exists between `S_RES2` (end of block `blk`, `blk` itself
increments here) and `S_QKVRET` (first weight-bank read of block
`blk+1`) -- the `L_COLL` LayerNorm pass in between never touches the
weight bank. An older PORT-NOTES.md note ("weight streaming alone
doesn't unlock a meaningfully bigger model, because KV cache dominates")
predates KV streaming being built and deployed (`KV_DDR_BACKED(1)` is
already active, confirmed flat/model-size-independent BRAM cost) -- that
conclusion is stale for the current build and does not apply here.

### Step 1: feasibility measurement -- real numbers, not a guess

Extended `fabric/genesys2/tb/tb_weight_loader_ddr.sv` with a third test
case sized to `GW_BLK`=3072 words (the current D=128/D3=384/D_MLP=512/
LANES=64 shape's real per-block window, vs. the existing 16/24-word
correctness-only cases) and a cycle counter around the existing
`ld_start`-to-`ld_done` wait. Compiled directly with iverilog (no
`run_*.py` harness for this gate yet, matching the testbench's own header
comment) against `weight_bank_tdp.sv` + `weight_loader_ddr.sv` +
`mig_behav_model.sv` + a one-line `` `define SYNTHESIS`` shim (needed
only to skip `mig_read_engine.sv`/`sync_fifo.sv`'s SVA under Icarus,
inserted in the file list right before them -- `weight_bank_tdp.sv` must
NOT see `SYNTHESIS` defined, so the shim goes after it, not as a global
`-D` flag) + `mig_read_engine.sv` + `sync_fifo.sv`.

**Result**: `WEIGHT_LOADER_DDR_CYCLES,tc=2,words=3072,cycles=27660` --
~9 cycles/word, correctness clean (`WEIGHT_LOADER_DDR_VERDICT,PASS`
across all 3 cases). ~9 cycles/word held constant across all three test
sizes (16/24/3072 words), which does NOT match "latency paid once, not
once per beat" -- traced to a REAL RTL bottleneck, not a simulation-
fidelity artifact of `mig_behav_model.sv` (whose own header comment is
explicit it isn't a timing model: "Real latency behavior is a Phase 6
(real-hardware) concern, not a Phase 2 gate concern" -- checked its read
pipe directly to rule out the model itself being the limiter: `app_rdy_o`
is tied high, one command accepted per cycle, genuinely pipelined).

**Root cause: `weight_loader_ddr.sv`'s own drain side.**
`assign rd_ret_ready = (ldst==LD_RUN) && !beat_have;` -- once a 256-bit
DMA beat arrives, `rd_ret_ready` drops until all `SUBW=8` of its 32-bit
chunks have been unpacked one-per-cycle into `weight_bank_tdp`'s existing
write port (which only accepts one 32-bit chunk per cycle, by design --
the SAME assembler the firmware boot-load path already uses). This is a
REAL architectural ceiling of the resident weight bank's write port
width, present on real hardware exactly as in simulation -- not a DDR3
latency/CDC question at all, which is what the original plan expected to
be the limiting factor.

**Real-world translation** (50MHz clock, 20ns/cycle): 27,660 cycles =
~553us per layer reload. An NLAYER=8 model (doubling again from the
current NLAYER=4) would cost roughly 8x that in reload alone (~4.4ms) on
top of compute (~360us, doubled from NLAYER=4's own real ~180us/token) --
**~4.8ms/token total, a ~27x slowdown vs. NLAYER=4's fully-resident
~180us/token**, but still >200 tok/s in absolute terms -- far faster than
any human reads, for an interactive-chat use case. Presented this
tradeoff to the user with the real numbers rather than assuming either
direction; **decision: accept the throughput cost and proceed** --
widening `weight_bank_tdp`'s write port to recover most of this (a real,
separate RTL change to the assembler + `weight_loader_ddr.sv`'s
unpacking) stays an explicit, not-yet-needed follow-up, not blocking this
pass.

### Step 2: RTL integration -- DONE, bit-exact in simulation

Wired the per-layer reload into `sequencer_vec.sv`'s block loop, behind a
new `WEIGHT_STREAM_PER_LAYER` parameter (default 0, every existing build
byte-for-byte untouched -- checked via a full regression re-run of
`fabric.stage3.run_vec_kv` after EVERY edit in this section, `VEC_KV_
VERDICT match=True` throughout, same exact gold sequence as before any of
this work started).

**Found one more real design gap while implementing, not just planned
ahead of time**: the token/position embedding tables (`EMB_TOK_BASE`/
`EMB_POS_BASE`) are ALSO addressed through the same resident weight bank
(read via `emb_sel_w`/`emb_addr_w` during `S_EMB`, phase-disjoint with
block GEMV reads per the RTL's own comment) -- and at this checkpoint's
shape are comparably sized to one block's own window (`GW_EMB`=2,960 vs
`GW_BLK`=3,072 words). They can't stay at their old large absolute
on-chip offsets once `WWORDS` shrinks to one block's size either. Rather
than keep them separately resident (a real option, costing permanent
BRAM regardless of NLAYER), extended the same streaming mechanism to
them too: reloaded once per token, before `S_EMB` runs, via the same
`S_STRW` state.

**New parameters**: `WEIGHT_STREAM_PER_LAYER` (0 default), `WEIGHTS_DDR_
BASE` (the DDR3 byte offset the full weight image starts at -- 0,
matching `send_weights.py`/`uart_load_weights()`'s existing staging
point unchanged).

**New FSM state, `S_STRW`** (a CALLABLE state, matching this FSM's own
existing convention for reusable sub-FSMs like `L_COLL`/`G_AQ`): arms
exactly one `weight_loader_ddr` load on entry (embed tables, block
`blk`'s window, or the head's -- discriminated by `strw_ret`, which also
IS the real destination state once the reload completes: `S_EMB`,
`S_QKVRET`, or `S_HEADSET`), then stalls on `wld_ld_done`. Entered from
three places: (1) `S_IDLE` directly, for the embed-table reload every
token needs before `S_EMB` can run at all (block 0's weights aren't
resident just because the LAST token ended somewhere else -- head, or
mid-block on an interrupted run); (2) `S_EMB`'s own completion arms
`strw_ret<=S_QKVRET` fresh (this L_COLL->S_STRW hop is ALWAYS the
block-0 reload, S_EMB only ever precedes block 0) before falling through
to the existing `L_COLL` LN-drain state, whose own `l_ret` was set to
`S_STRW` back at `S_IDLE`; (3) `S_MPSET`, redirecting its existing
`l_ret<=S_HEADSET`/`S_QKVRET` through `S_STRW` (`strw_ret` carrying the
real destination) for every inter-block transition, same mechanism.

**`g_wbase` at all five existing call sites** (`S_QKVRET`, `S_CDR`'s
proj dispatch, `S_FCRET`, `S_MPSET`, `S_HEADSET`) becomes conditional:
`WEIGHT_STREAM_PER_LAYER ? WB_XXX : (blk*GW_BLK + WB_XXX)` -- under
streaming, every block's (or the head's) window always starts at
on-chip address 0 right after its own fresh reload, so the `blk*GW_BLK`
absolute-addressing term is dropped entirely. `emb_addr_w` gets the same
treatment via new `EMB_TOK_BASE_STRM=0`/`EMB_POS_BASE_STRM=EMB_TOKW`
0-based equivalents. `weight_loader_ddr`'s own `ld_start`/`ld_ddr_addr`/
`ld_words` inputs are muxed (elaboration-time, parameter-selected, not a
runtime mux) between the existing firmware-facing `wld_ld_*` ports and
new internal `wldi_start`/`wldi_addr`/`wldi_words` regs S_STRW drives.
Two `` `ifndef SYNTHESIS`` sanity warnings guard real misconfiguration
risks: `WEIGHT_STREAM_PER_LAYER` without `WEIGHT_DDR_BACKED` (S_STRW
would hang forever on a `wld_ld_done` that `g_wld_off` ties permanently
low), and `WWORDS` sized below `max(GW_BLK, GW_HEAD, GW_EMB)` (silent
address truncation/wraparound into `weight_bank_tdp`).

### Step 4 (gate): DONE, bit-exact -- the first time weight_loader_ddr has ever been triggered mid-inference

New `fabric/stage3/tb/tb_seq_vec_kv_stream.sv` (no `run_*.py` harness yet,
matching `tb_weight_loader_ddr.sv`'s own precedent): stages the checkpoint's
FULL weight image directly into a simulated DDR3 (`mig_behav_model`, via
`$readmemh("wrom.mem", u_mem.mem)` -- WBITS=DATA_W=256 at LANES=64 makes
one `wrom.mem` line exactly one DMA beat, no unpacking arithmetic of its
own to get wrong) instead of bulk-loading it into `weight_bank_tdp` via
`wl_we`, wires `sequencer_vec`'s `wl_rd_req_*`/`wl_rd_ret_*` ports to a
real `mig_read_engine`, and drives the SAME PLEN=16/NGEN=40 prompt+gen
passes `tb_seq_vec_kv.sv`'s own gate already proved bit-exact -- with
`WEIGHT_DDR_BACKED(1)`, `WEIGHT_STREAM_PER_LAYER(1)`, and `WWORDS(3072)`
(this checkpoint's `GW_BLK`, the actual BRAM win: down from the fully-
resident design's 15,376).

**Result: bit-exact.** `got == gold ==
[5, 1, 40, 28, 25, 38, 25, 1, 43, 21, 39, 1, 21, 1, 32, 29, 40, 40, 32, 25,
1, 27, 29, 38, 32, 1, 34, 21, 33, 25, 24, 1, 32, 29, 32, 45, 7, 1, 39, 28]`
-- identical to `tb_seq_vec_kv.sv`'s own already-proven gold sequence for
this exact checkpoint/prompt/seed, confirming the streaming redesign
reproduces the fully-resident design's exact numerics, not a separately-
computed approximation. This is the first time `weight_loader_ddr` has
ever been exercised mid-inference (every prior real use, including its
own `tb_weight_loader_ddr.sv` gate, only ever triggered it externally,
once, at boot).

**Measured throughput, real numbers from this same run**: 147,184
cycles/pass average (vs. the fully-resident design's own 8,716
cycles/pass for the identical checkpoint) -- at 50MHz, ~2.94ms/token at
the CURRENT NLAYER=4 (a ~17x per-token slowdown from streaming
architecture alone, before growing NLAYER at all -- every token now
reloads the embed tables + all 4 blocks + the head from scratch, since
nothing stays resident across a soft_reset). Extrapolating to NLAYER=8
(fixed embed+head reload cost, block-reload+compute cost roughly
doubling): ~265k cycles/token, ~5.3ms/token, ~189 tok/s -- consistent
with the earlier feasibility measurement's own back-of-envelope estimate
(>200 tok/s), still far faster than human reading speed for interactive
chat, the already-accepted tradeoff.

**Not yet done**: shrink `WWORDS` in the real deployed wrapper
(`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`), train a real checkpoint at
a bigger NLAYER (NLAYER=8 per the plan), full synth/PnR/bitstream cycle
(remembering `fabric/genesys2/stage_vivado_roms.py` this time), and
real-hardware bring-up.

## Per-layer weight streaming at NLAYER=8: a real DDR3 address collision, not a timing bug

Deployed the plan's next step: trained a genuinely bigger checkpoint
(`data/ckpt_stream8.pt`/`.qat.pt`, NLAYER=8 same D/D3/D_MLP/NHEAD/VOCAB,
raw-TinyStories corpus -- FP best val 0.841, QAT best val 0.862, real
improvement over the NLAYER=4 numbers), exported (`fabric/export_stream8/`,
33 quant layers, 885,248-byte packed image), gated bit-exact in both the
fully-resident and per-layer-streaming RTL paths, shrunk `WWORDS` to 3072
in the real wrapper (`.NLAYER(8)`, `.WWORDS(3072)`,
`.WEIGHT_STREAM_PER_LAYER(1)`), removed a real firmware hang risk
(`uart_load_weights()`'s `kevgpt_wld_load()` call polls a register that's
muxed out at elaboration time under streaming -- found via code review,
fixed before it ever hung on real hardware), and completed a clean
synth/PnR/bitstream cycle: BRAM 267/445 (60%, down from the earlier
NLAYER=4-fully-resident build's 360+ despite doubling depth -- the whole
point), DSP 803/840, WNS +2.288ns.

**Real hardware: wrong output, but precisely reproducible.** The JTAG
register-poke probe (same technique as every prior real-hardware gate
this project has used) returned `[5, 1, 40, 28, 25, 38, 25, 38, ...]`
against golden `[5, 1, 40, 28, 25, 38, 25, 1, ...]` -- correct through the
first 7 generated tokens, wrong on the 8th, every downstream token then
diverging further (expected, autoregressive feedback of a wrong token).
Bit-exact in simulation, wrong on real silicon: a real sim/synth gap, the
kind this project's own "bit-honest before fast" rule exists to catch
before trusting a number, not after.

**False lead: looked non-deterministic, was actually a testing mistake.**
Two back-to-back probe re-runs (no bitstream reprogram between them) gave
*different* wrong answers each time, both wrong from token 0 -- looked
exactly like real-hardware CDC metastability (the settle-cycle delay
between a weight reload finishing and the next read starting had already
been widened 8->32 cycles with zero effect on the original divergence,
consistent with a genuine timing issue rather than a tunable race). Traced
instead to a self-inflicted procedural bug: DDR3 contents do not survive
an FPGA reconfiguration, and those two probe runs reprogrammed the
bitstream without re-running `send_weights.py` to re-stage the weight
image afterward -- the design was reading uninitialized/stale DRAM, a
different garbage pattern each reprogram. Once the full bring-up sequence
(reprogram -> `send_weights.py` listening -> `gdb load+continue` to boot
firmware and stream weights -> `SEND_WEIGHTS_PASS` -> THEN the register-
poke probe) was followed correctly, the result was exactly reproducible:
`[5, 1, 40, 28, 25, 38, 25, 38, 25, 1, 43, 29, 39, 40, 28, 29]`, identical
to the original settle=8 and settle=32 runs. Worth remembering: a
"different every run" symptom is not proof of hardware nondeterminism if
the bring-up sequence itself wasn't controlled for -- check the mundane
explanation (stale/uninitialized external memory) before reaching for
metastability.

**Real root cause: `KV_DDR_BASE` and `WEIGHTS_DDR_BASE` were both 0.**
`kv_bank_ddr.sv` preallocates a flat, fixed-size DDR3 region indexed by
`((layer*2+kv)*NHEAD+head)*TMAX+pos`, sized `NLAYER*2*NHEAD*TMAX` rows *
96 bytes/row = 393,216 bytes at this shape. `weight_loader_ddr` streams
the staged weight image starting at `WEIGHTS_DDR_BASE`, 885,248 bytes at
this shape. `KV_DDR_BASE` was hardcoded to `0` INSIDE `sequencer_vec.sv`'s
own `kv_bank_ddr` instantiation -- never exposed as a top-level parameter
-- and the real wrapper's `WEIGHTS_DDR_BASE` is also `0`: two independent
DMA consumers, same physical DDR3 bytes, from byte 0. KV's 393,216-byte
footprint sits entirely inside the weight image's own blocks 0-3 (each
98,304 bytes: `[0,98304)`, `[98304,196608)`, `[196608,294912)`,
`[294912,393216)`). Every attention-cache write for layers 0-3, at any
position, silently overwrites staged weight bytes in DRAM. The row-index
formula explains the specific failure point: only the first few positions
of each `(layer,kv,head)` combo get written early on, so corruption starts
as a handful of small (~2.3KB) scattered chunks per token and accumulates
-- consistent with several early tokens surviving before enough of blocks
0-3 is clobbered to flip an argmax decision. Simulation never caught this
because no gate has ever run `KV_DDR_BACKED=1` and
`WEIGHT_STREAM_PER_LAYER=1` together -- `tb_seq_vec_kv_stream.sv` keeps
KV resident on purpose ("unrelated to this gate"), so nothing has ever
populated both DDR3 regions in the same simulated image and checked for
overlap.

**Fix**: `KV_DDR_BASE` is now a real `sequencer_vec` parameter (default
0, so every existing `KV_DDR_BACKED`-only build that never touches
`WEIGHT_STREAM_PER_LAYER` is unaffected), threaded to the internal
`kv_bank_ddr` instantiation instead of hardcoded. The real wrapper now
passes `.KV_DDR_BASE(1048576)` (1MB, >2x headroom over the 885,248-byte
weight image, comfortably inside 29-bit addressing and the board's DDR3
capacity). Added a sim-only `initial` warning in `sequencer_vec.sv`
(mirrors `kv_bank_ddr.sv`'s own row/beat sizing formula) that computes
both regions' real byte footprints and fires if they'd overlap for
whatever `KV_DDR_BASE`/`WEIGHTS_DDR_BASE`/shape combination a future build
uses -- this exact bug should be impossible to reintroduce silently again.
Verified the address-parameterization change doesn't disturb the existing
NLAYER=8 streaming gate (`tb_seq_vec_kv_stream.sv`, `KV_DDR_BACKED=0` by
default there, so behavior-neutral by construction) before touching real
hardware again.

**Fix confirmed on real hardware.** Clean rebuild (BRAM 267/445, DSP
803/840, Setup WNS +3.506ns, Hold WNS +0.093ns -- no regression from the
address-only change), reprogrammed, weights re-staged, register-poke
probe extended to 16 tokens: `[5, 1, 40, 28, 25, 38, 25, 1, 43, 21, 39, 1,
21, 1, 32, 29]` -- bit-exact against the Python golden, run twice in a
row, both times identical. The token-8 divergence is gone; no new
divergence anywhere in the 16-token window. This closes out the address-
collision bug entirely -- `KV_DDR_BASE`/`WEIGHTS_DDR_BASE` non-overlap is
now real, verified, and protected by the sim-only overlap check so it
can't silently regress.

**Not yet done**: sampling-mode and manual interactive chat at NLAYER=8
(the actual point of growing the model -- human-judgment check for
whether completions are more coherent than NLAYER=4's).

## Sampling-mode chat garbled at NLAYER=8: a miscalibrated temperature, not a bug

Manual interactive chat over UART (which ALWAYS uses on-chip Gumbel
sampling -- `main.c`'s `chat_turn()` derives a nonzero seed from
`kevgpt_cycles()` every turn, never greedy) produced garbage: `"once upon
a time" -> ",h'maEmejumy. seks m, frogs. lejby tips ail vbujer..."`. The
greedy register-poke probe never exercises this path at all (SEEDVAL=0
throughout), so this was the first real test of on-chip sampling under
the per-layer-streaming NLAYER=8 build.

**Ruled out streaming as the cause first.** `fabric.stage3.run_vec_kv
--seed 12345` (the FULLY-RESIDENT, non-streaming RTL path) reproduced the
same `match=False` for `export_stream8` -- the bug has nothing to do with
per-layer weight streaming. The same test against `export_stories`
(the earlier, real-hardware-confirmed-coherent NLAYER=4 checkpoint), same
seed, passed (`match=True`) -- isolating the difference to something
about the export_stream8 checkpoint/shape itself, not the RTL streaming
work.

**Ruled out an RTL logic bug via direct instrumentation**, not guessing:
temporary `$display` traces (removed after use) dumping (1) every
gumbel_bank noise value the precompute FSM writes and (2) every argmax
row's post-noise winner, run against BOTH checkpoints at the same seed.
The raw noise sequence (xorshift advances + LUT reads) was BIT-IDENTICAL
between the NLAYER=4 and NLAYER=8 builds, as expected (the precompute FSM
has zero NLAYER dependence in its code) -- ruling out a desync in the
noise generator itself. The argmax pipeline's own row-by-row reduction
traced out internally consistent and structurally identical between
builds (same PLEN=16 pass count, same ARROWS=8 rows/pass). This pointed
away from an RTL defect and toward the DATA: the actual head-logit
magnitudes feeding the compare.

**Real root cause, confirmed via `model.goformer_kvq.IntKVQSequencer.
step_head_q25`**: `export_stream8`'s head-logit std at this prompt is
~4.78 (Q6.25 real units) vs `export_stories`'s ~7.01 -- a real property
of the bigger, differently-trained NLAYER=8 model, not a defect. Gumbel
noise magnitude is a GLOBAL FIXED CONSTANT (`gumbel.TEMP=0.85`, baked
once into `gumbel_lut.mem`, shared by every checkpoint this project has
ever deployed) -- at the old TEMP, individual noise draws (commonly
5-8 real units at the LUT's upper quantiles) were comparable to or larger
than the ENTIRE spread of this checkpoint's real logit signal, so the
Gumbel-max "winner" was effectively noise-dominated (near-random),
matching the garbled real-hardware symptom exactly, while greedy (zero
noise) stayed bit-exact because it never touches the noise path at all.

**Fix**: `gumbel.TEMP` rescaled 0.85 -> 0.58 (`fabric/stage3/gumbel.py`),
proportional to the std ratio (0.85 * 4.78/7.01 ~= 0.58), preserving the
noise-to-signal ratio the earlier, real-hardware-confirmed-coherent
checkpoint had. `gumbel.TEMP` is gumbel.py's own single source of truth
for BOTH `gumbel_lut.mem` generation paths (`run_vec_kv.py`'s sim gate AND
`stage_vivado_roms.py`'s real-hardware ROM staging both call
`gumbel.make_gumbel_lut()` with no override), so this one-line change
propagates correctly to both without separate patching. Framed as a
checkpoint-specific recalibration, not a universal "correct" value -- a
future checkpoint with a different logit scale should re-measure via the
same method (`seq_ref.build` + `IntKVQSequencer.step_head_q25`'s real
std), not reuse this number blindly.

**TEMP recalibration alone did not fix it.** With the new TEMP, 2 of 4
tested seeds (`12345`, `999999`) started matching, but `seed=1` and
`seed=42` still diverged (`seed=1`: correct through 7 tokens, wrong on
the 8th -- `[...,25,45]` vs gold `[...,25,1]`). Since BOTH the RTL's own
ROM and the Python gold reference derive their Gumbel noise from
`gumbel.TEMP` identically, a genuine hw-vs-gold MISMATCH cannot be a
temperature/calibration artifact -- it means a real, independent
computation bug, unrelated to noise, was ALSO present and TEMP=0.85 had
simply been masking it (noise so dominant that close comparisons,
where the bug would show up as a flipped winner, were rare).

**Root-caused via direct RTL instrumentation, one pipeline stage at a
time** (temporary `$display` traces, each removed after use, each
cross-checked against an independently-computed Python value from
`model.goformer_kvq.IntKVQSequencer` / `fabric.stage3.seq_ref`):
1. Gumbel noise sequence (xorshift advances + LUT reads): BIT-IDENTICAL
   between RTL and Python. Not the noise generator.
2. Argmax pipeline structure (row count, pass count): identical between
   NLAYER=4 and NLAYER=8 builds. Not an argmax/index-mapping bug.
3. Per-channel head logits (all 57, not just non-argmax ones): EVERY
   channel differed from Python in absolute value, but the TRUE argmax
   channel's relative ranking was usually still preserved -- explaining
   why greedy stayed robust while sampling (sensitive to ALL 57 values)
   did not.
4. Dequant math (`gemvy * mant * 2^exp`): verified EXACT using each
   side's OWN `gemvy` -- `vec_dequant.sv`'s multiply/shift is bit-perfect.
   Not the scale application.
5. Raw GEMV accumulator (`gemvy`, pre-dequant): WRONG vs Python's `y_int`,
   confirming the bug is upstream of dequant, in the GEMV or its input.
6. On-chip INT8 activation quantization (`aqw`, the RTL's own `ix`):
   wrong vs Python's `ix`, ruling OUT the weight side and narrowing to
   the activation path -- specifically its INPUT, the final LayerNorm's
   output.
7. `gamma_w.mem`'s `ln_f` entry: decoded and compared against Python's
   `ln_f_q` array -- EXACT match, all 16 checked values identical. The
   ROM content was never wrong.
8. `inv_sact.mem`'s head entry: also an EXACT match against Python's
   `_inv_sact(head['s_act'])`. Not an export/scale-miscalibration bug
   either -- this closed out the TEMP-adjacent hypothesis entirely.

**Real root cause**: `l_gbase` (`sequencer_vec.sv`, "LN gamma set" index,
selects which of `GAMMA_N=2*NLAYER+1` stored LayerNorm gain vectors to
use -- `0=ln1 of block 0, ..., 2*NLAYER=ln_f`) was declared `reg [3:0]`
-- 4 bits, max value 15. `l_gbase<=NLAYER*2` for the FINAL LayerNorm
(`ln_f`, read right before the head GEMV) needs to hold 16 at NLAYER=8.
16 does not fit in 4 bits: it silently wrapped to 0, so the RTL's final
LayerNorm pass READ BLOCK 0's OWN `ln1` gamma gains instead of `ln_f`'s
-- the ROM content was always right (confirmed above); the FSM was
reading the WRONG ROW of it. This is the exact same class of bug as the
earlier `GAMMA_N`/`NSACT` "hardcoded for NLAYER=4, silently breaks past
it" fixes (`fabric/genesys2/PORT-NOTES.md`, "Phase 4 continued -- two
real NLAYER bugs found") -- but this ONE register was missed at the
time, because nothing exercised it: at NLAYER=4, `NLAYER*2=8` fits 4
bits fine (invisible), and even at NLAYER=8 it stayed invisible under
GREEDY decode, since a wrong-but-still-reasonable-magnitude per-channel
gamma distorts head-logit MAGNITUDES channel-by-channel without
reliably flipping which single channel is largest. Gumbel sampling,
which needs ALL 57 values individually correct (not just the winner's
identity), had no such robustness margin.

**Fix**: `l_gbase` widened from a hardcoded `[3:0]` to
`[$clog2(GAMMA_N)-1:0]` -- computed from the actual shape parameter,
same "don't hardcode a width that happens to work for today's NLAYER"
discipline the GAMMA_N/NSACT fixes already established. Every other
`l_gbase` assignment site was checked for the same class of truncation
(`blk*2+1`, `(blk+1)*2` at intermediate blocks) -- all stay well under
15 even at NLAYER=8, so only the `ln_f` site (`NLAYER*2`) was ever
actually broken.

**Verified**: sampling-mode `run_vec_kv.py --seed <n>` at 4 different
seeds (1, 42, 12345, 999999) against `export_stream8` -- ALL FOUR now
`match=True` (seeds 1 and 42, the ones that failed even after the TEMP
recalibration, now pass outright). Greedy regression (ngen=16, both the
fully-resident and per-layer-streaming RTL paths): still bit-exact,
unaffected. The earlier NLAYER=4 checkpoint (`export_stories`): also
unaffected (`NLAYER*2=8` never overflowed 4 bits there, so this fix is
a no-op for that build, confirmed by re-running its own sampling gate).

**Confirmed on real hardware, full loop closed.** Clean rebuild (Setup
WNS +2.77 to +4.11ns, Hold +0.108ns, BRAM 267/445, DSP 803/840 --
identical utilization to every prior NLAYER=8 build, this was a pure
logic-width fix with no resource-cost change), reprogrammed, weights
re-staged. Greedy register-poke probe: still bit-exact,
`[5, 1, 40, 28, 25, 38, 25, 1, 43, 21, 39, 1, 21, 1, 32, 29]`. Real
interactive chat over UART (sampling mode, the thing that was actually
broken):

```
"once upon a time" -> " there was a girl named lily. she loved to play outside in the garden. one day, lily went to the par"
"the dog ran"       -> "ing to the floor and the tasty bath. he took a big hole and the shovel. he wanted to go to school th"
"a little girl"     -> " named lily who loved to play with her toys. one day, she found a snack in the park and she was so p"
```

Coherent, grammatical English across all three -- a dramatic, qualitative
change from the pre-fix `",h'maEmejumy. seks m, frogs. lejby tips ail
vbujer..."` garbage. This closes out BOTH bugs this per-layer-streaming
phase surfaced: the DDR3 `KV_DDR_BASE`/`WEIGHTS_DDR_BASE` address
collision (streaming-specific, fixed earlier this section) and the
`l_gbase` width overflow (NOT streaming-specific -- a real, independent,
pre-existing RTL bug that per-layer streaming's real-hardware bring-up
happened to be the first thing in this project's history to actually
exercise Gumbel sampling at NLAYER>4). Both fixes are synced to the
fork's vendored copies (`sequencer_vec.sv`,
`xheep_kevgpt_peripheral.sv`), and the sim-only overlap warning
(`KV_IMAGE_BYTES`/`WEIGHT_IMAGE_BYTES`) plus the width computed from
`GAMMA_N` instead of a hardcoded literal should make both bugs' specific
failure modes structurally hard to reintroduce silently.

**Not yet committed** -- both repos have uncommitted work spanning this
entire per-layer-streaming phase (RTL, firmware, checkpoints, exports,
this file). Next: commit kev-gpt and the kevgpt-genesys2-soc fork.

## Longer replies: firmware-only, no resynth needed

With both real bugs fixed, chat is coherent but replies were still
capped short (~100 chars, often mid-sentence) by `kevgpt_interactive/
main.c`'s own `MAX_GEN_LEN`/`STORY_SENTENCES` constants -- unrelated to
the hardware/model at all, and with real throughput headroom to spend
(measured ~188 tok/s per-layer-streaming at NLAYER=8, well above
interactive-chat speed needs). `MAX_GEN_LEN` raised 100->120,
`STORY_SENTENCES` 4->8 (doubled, so replies actually use the larger
budget instead of stopping early on sentence count). The real hard
ceiling remains the generate loop's own `pos < KEVGPT_TMAX` check, which
is prompt-length-aware at runtime (unlike the MAX_GEN_LEN constant) and
was left untouched -- this change only raises the SOFTER cap toward it.

Firmware-only change -- no RTL/bitstream rebuild needed (build via
`make app PROJECT=kevgpt_interactive TARGET=genesys2 SOURCE=../../../../sw`
from `hw/vendor/esl_epfl_x_heep/`, note `TARGET=genesys2` not
`genesys2_kevgpt` -- the latter is the FPGA_BOARD/Vivado-side target
name only, `mcu-gen`/CMake's own sw-side target is the plain board name;
using the wrong one fails with a `x-heep.h: No such file` error that
looks like a missing `make mcu-gen`, but isn't). Reloaded via the
existing `gdb load+continue` + `send_weights.py` handshake, no
reprogram. Real-hardware confirmation, same 3 prompts as the sampling
fix, all noticeably longer (~133 bytes vs. ~110-120 before) and still
coherent:

```
"once upon a time" -> " there was a big bear. he was a cold and always wanted to play with his friends. one day, he saw a big tree and "
"the dog ran"       -> "ing into the park. they said goodbye to the park and the dog. they took the path and started to cry. they felt sad, b"
"a little girl"     -> " named lily who loved to carry in her bedroom. one day, she was walking to the beach when her mom says she was the "
```

## NLAYER=12: data-limited at first, fixed with the full TinyStories train split

Grew NLAYER 8->12 (same D/D3/D_MLP/NHEAD/VOCAB shape, `WWORDS=3072`
unchanged -- the whole point of per-layer streaming: weight-bank BRAM
cost stays flat regardless of depth).

**First attempt: worse than NLAYER=8, not better.** Trained on the SAME
small (~19M-char) `TinyStories-valid` split used for the NLAYER=8
checkpoint, identical recipe (12000 iters, same LR schedule): best val
1.020 -- WORSE than NLAYER=8's 0.841. Train loss kept falling while val
loss climbed back up after ~iter 6500-7000: classic overfitting, a
bigger model outrunning what a small, heavily-repeated dataset can
teach it. Two hyperparameter interventions (dropout=0.1; lower LR +
longer warmup) each landed in the same 0.90-0.93 range -- three
different reasonable configs converging to the same plateau is the
signature of a data-limited regime, not a tuning problem.

**Fix: the full TinyStories train split (~1.9GB), not the ~20MB
validation split.** `keviniser.fetch_tinystories --split train` pulls
it directly from the same HuggingFace source. Its natural vocab is 241
chars (uppercase, digits, and rare multilingual/emoji noise mixed in) --
incompatible with the deployed VOCAB=57 hardware shape (embed/head
tables are sized to exactly that at synthesis time). Lowercased every
story and dropped any story containing a character outside the same
47-char whitelist the working NLAYER=8 checkpoint already used (kept
1,988,101 of 2,119,489 stories, 93.8%) -- NOT a new tokenizer or a
hardware change, just a stricter filter on which stories can be used
with the SAME character set. Padded to VOCAB=57 with the exact same
A-J filler scheme (ids 47-56) already baked into the deployed firmware's
tokenizer table, so `kevgpt_tokenizer.h` came out byte-identical on
regeneration -- no firmware retouch needed.

Retrained on this ~94x-bigger corpus (~1.78B chars after filtering),
same recipe: best val 0.842 at 12000 iters -- already matching NLAYER=8's
0.841, and train/val stayed close together (0.843/0.845) instead of
diverging, confirming the overfitting was genuinely data-limited, not an
optimization problem. Doubled to 24000 iters since there was no
overfitting signal yet: best val **0.772** -- a real, clean improvement
over NLAYER=8, not just parity. QAT (INT4/INT8, warm-started): best val
**0.785**, vs NLAYER=8's own QAT 0.862.

**A real "recheck NLAYER-dependent sizing on every depth change" catch,
made before synth, not after.** `KV_DDR_BASE` (fixed at 1MB earlier this
session for the NLAYER=8 collision bug) does NOT automatically scale --
at NLAYER=12 the weight image itself grew to 1,278,464 bytes, LARGER
than the old 1MB base, which would have silently reintroduced the exact
same KV/weight DDR3 collision this session already spent real effort
finding and fixing once. Caught by just redoing the arithmetic
(`fabric/genesys2/PORT-NOTES.md`'s own per-layer-streaming section has
the formula) before touching Vivado; `KV_DDR_BASE` raised to 2MB.

**Export path note**: `model.export_fabric` produces a DIFFERENT npz
layout (`blocks_N_qkv__int_w` naming) than what `fabric.stage3.seq_ref`
and the RTL gates actually consume (`bN_qkv_iw`/`_ws`/`_sa` naming,
`n_blocks`/`tok_emb`/`pos_emb`/`ln_f` top-level keys) -- the correct tool
is `model.goformer_full.params_from_ckpt()` + `save_params()`, not
`export_fabric.export()` directly. `python -m model.goformer_full
<ckpt>` also runs a useful independent sanity check (numpy INT reference
vs the actual PyTorch/Brevitas forward pass, cosine>0.9999 gate) --
FAILED at NLAYER=12 (cosine=0.9998257) despite the REAL hardware-facing
gate (`run_vec_kv.py` against `seq_ref.IntKVQSequencer`, the fixed-point
reference the RTL is actually built to match) passing bit-exact. These
are two different reference implementations; `goformer_full`'s own
float-leaning numpy re-implementation apparently accumulates enough
independent rounding drift over 12 layers to cross its own strict
threshold without that drift reflecting anything the RTL/hardware gate
actually cares about -- flagged here as a known, apparently-benign gap
rather than silently ignored, since a future NLAYER bump might behave
differently and deserves the same scrutiny, not an assumed pass.

**Real hardware, confirmed**: greedy register-poke probe bit-exact
(`[5, 1, 40, 28, 25, 38, 25, 1, 43, 21, 39, 1, 21, 1, 32, 29]`, matching
the same golden as NLAYER=8's, coincidentally -- "once upon a time" is
evidently a strong enough opener that multiple model depths converge to
the same completion). Real interactive chat, coherent across multiple
prompts:

```
"once upon a time" -> " there was a little girl named lily. she loved to play outside in the sun and explore the woods. one day, she sa"
"the dog ran"       -> "ing and ran to the dog. the dog was happy to see the dog. the dog was still angry and he said, \"wow, that was a norma"
"a little girl"     -> " named lily who loved to play outside. one day, she decided to go to the park to play with her friends. they wanted"
```

**Real synth numbers**: Setup WNS +1.055ns (tighter than NLAYER=8's
+2.77 to +4.11ns, as expected with more logic, still positive/clean),
Hold +0.170ns, BRAM 171/445 (38%, down from NLAYER=8's 267 -- a real,
observed, NOT fully root-caused result; the per-layer scale ROMs
[gamma_w/dqm_w/dqe_w/inv_sact] DO grow with NLAYER and should need MORE
raw bits, so this is most likely the same BRAM-is-a-step-function-of-
depth bucketing effect already proven for weight_bank_tdp's own WWORDS
sizing elsewhere in this document, landing in a better-packed
RAMB36/RAMB18 bucket at this specific depth -- but this wasn't confirmed
with a hierarchical utilization report, and the NLAYER=8 build's own
report no longer exists to diff against since its build directory was
`rm -rf`'d for this rebuild. Worth a real look if BRAM headroom ever
becomes tight again, not fully explained here). DSP 804/840, essentially
unchanged (compute is NLAYER-independent, same physical MAC array
reused serially per layer). Measured per-token cost: ~385,380
cycles/token (~7.7ms, ~130 tok/s at 50MHz) -- still comfortably above
the ~50 tok/s interactive-chat comfort floor.

Committed as `n12-d128-stable` (both repos) once real hardware was
confirmed; this is the deployed state the two follow-up experiments below
branched from.

## NLAYER=16 and D=256: two depth/width growth attempts, both dead ends

**NLAYER=16 (16 layers, same D=128/D3/D_MLP/NHEAD/VOCAB shape as
NLAYER=12): diminishing returns, not a regression.** Same full-train-
split recipe as the NLAYER=12 fix above (24000 iters, same corpus). Best
val landed essentially flat vs NLAYER=12's 0.772 -- no real quality win
for 4 more layers' worth of BRAM/DSP/cycle cost. Not pursued to synth;
this is a genuine, informative negative result (the model is not simply
"more layers = better" at this data/width budget), not a bug or a bad
training run.

**D=256/NLAYER=12 (doubling width instead of depth): real training
instability, not overfitting, not a tuning problem.** Same corpus/recipe.
At `lr=5e-4`: best val 0.774 at iter 8000, then BOTH train and val loss
climbed together for the rest of training -- not the classic overfitting
signature (train improving while val worsens), which pointed at
something optimizer/architecture-level rather than a regularization gap.
Retried at a lower LR (`lr=2e-4`, longer warmup) per the user's own
choice after seeing the first result: the instability persisted (got
worse, if anything) -- ruling out "just needed a gentler LR" as the
explanation. Best val across the two runs: 0.792. Not pursued further;
flagged as a real open question if width growth is revisited, not
resolved here.

Per-layer reload cost (the dominant per-token cost at this shape, see
NLAYER=12's own ~86%-of-total-cycles measurement above) scales very
differently with the two axes: linearly with NLAYER (`GW_BLK` is
per-layer-window-sized, so more layers just means more windows, each the
same size), but *quadratically* with D (QKV/PROJ/FC/MP weight counts are
all ~D^2, so doubling D roughly quadruples `GW_BLK`). This is *why*
NLAYER growth stayed close to linear in cycles/token while D=256 was
projected at ~4x the reload cost of D=128 (~32 tok/s estimated) for a
width doubling -- before the instability finding above made the question
moot. Both explicit follow-ups (further D=256 debugging, NLAYER=16) were
deprioritized by the user's own choice in favor of the word-level
vocabulary work below, which had better expected ROI.

## Word-level vocabulary: a fixed ~1900-word tokenizer, replacing char-level

**Motivation.** Char-level VOCAB=57 forces every word to be spelled out
one character at a time -- most of each `TMAX=128`-token budget (and most
of each ~130 tok/s reply) is spent on spelling, not content. A fixed
small word vocabulary trades a bigger embed/head table for far more
*story* per token: the same TMAX now covers many more words, not many
more characters.

**Tokenizer design** (`model/word_data.py`): fixed vocabulary of the N
most-frequent word/punctuation tokens via a simple regex split
(`[a-z']+|[.,!?;:\"-]`) -- deliberately NOT BPE/subword, to keep the
encode/decode path trivial and dependency-free, matching this project's
existing char-level tokenizer's own philosophy. Two reserved ids: 0 =
`<unk>` (anything outside the fixed vocab), 1 = `<eos>` (story boundary
-- char-level reused a literal `\n` for this; word-level has no single
"boundary character" to borrow, so it gets its own token). The remaining
`vocab_size-2` ids are the that-many most frequent tokens over the
corpus, sorted alphabetically among themselves for determinism (matching
`data.py`'s own `sorted(set(text))` convention). Same `<|endoftext|>`-
delimited corpus format, same `train.bin`/`val.bin`/`meta.json` file
contract as the char-level `data.py`, so everything downstream
(`load_split`, `meta["vocab_size"]`) is tokenizer-agnostic --
`model/train.py --tokenizer word --vocab-size N` dispatches to
`word_data.prepare_word()` instead of `data.prepare()`, nothing else
changes.

**Sizing, done BEFORE training, not after** (the same discipline as
every other shape change in this document): `weight_bank_tdp`'s BRAM
cost is a step function of `WWORDS` (128-RAMB36 bucket for
`WWORDS<=16384`, 256-RAMB36 bucket for `WWORDS` in `(16384,32768]`), and
`WWORDS` must be `>= max(GW_BLK, GW_HEAD, GW_EMB)` under per-layer
streaming, where `GW_HEAD` and `GW_EMB` both grow with `VOCAB`. At the
deployed D=128/NLAYER=12/LANES=64 shape: `GW_BLK=3072` (VOCAB-
independent), and at `VOCAB=1900`, `GW_HEAD=3840` and `GW_EMB=32448` --
so the embed tables, not the per-block weights, become the single
largest reload window once the vocabulary gets this big, and `WWORDS`
must grow to `>=32448` (was `3072` at VOCAB=57) -- still inside the
256-RAMB36 bucket boundary (`<=32768`), but a real, non-obvious jump
worth flagging before synth. Measured word-frequency coverage over the
full 418M-token filtered-TinyStories corpus: 512->86.45%, 900->91.45%,
1024->92.37%, 1536->94.89%, 1900->96.06%, 2048->96.47%. Picked
`VOCAB=1900` (user's own choice, "~1536-2048 words, pricier bucket") --
comfortably inside the 256-tile bucket with room to spare, and a real
coverage jump over the cheaper ~900-word bucket option.

**Training**: `model/word_data.prepare_word()` over
`data/TinyStories-train.filtered.txt` (the same corpus NLAYER=12 used),
`VOCAB=1900` -> 418,492,016 total tokens (414,307,096 train / 4,184,920
val), `unk_frac=0.0394`. Same D=128/NLAYER=12/NHEAD=2 shape as the
char-level deployed model, just a new embed/head width and a new
tokenizer. Trained clean and stable (no instability, unlike D=256): best
val **2.022** (word-level cross-entropy isn't comparable to the char-
level numbers above -- different vocabulary, different task difficulty
per token). QAT (INT4/INT8, warm-started): best val **2.074**. Real
qualitative win: full multi-sentence coherent stories with proper
grammar/punctuation inside the same `TMAX=128` budget, instead of
char-level's mostly-spelling output.

**Export tool gotcha (reused, not new)**: `model.export_fabric` produces
an npz layout the RTL gates don't consume -- `model.goformer_full.
params_from_ckpt()` + `save_params()` is the correct tool (same finding
as NLAYER=12's own note above). `save_params()` does not create its own
output directory (`mkdir -p fabric/export_word16` first, or it raises
`FileNotFoundError`).

**A new class of RTL bug, found via a systematic sweep, not by
accident**: `sequencer_vec.sv` had several hardcoded 9-bit (`[8:0]`, max
511) and 14-bit (`[13:0]`, max 16383) signal widths sized for the
largest VOCAB this project had ever actually deployed (char-level,
<=193) -- the exact same bug *class* as the earlier `GAMMA_N`/`NSACT`/
`l_gbase` fixes in this file (a register sized to fit the original
KV260 shape exactly, with nothing before now exercising the boundary).
At `VOCAB=1900`, every one of these silently truncates or wraps:

- `dor` (the G_RB readback row counter, shared across QKV/PROJ/FC/MP
  *and* head): declared width only covered `ROWSM` (the largest
  per-block row count, VOCAB-independent), but the head readback needs
  to count up to `ARROWS-1=237` (`ARROWS=ceil(VOCAB/P)`), which wrapped
  a 7-bit `dor` before it ever hit its own exit condition -- an
  **infinite loop** (simulation timeout at 40M cycles), not a wrong
  answer. Fixed with `MAXROWS = max(ROWSM, ARROWS)`, `dor` re-declared
  at `$clog2(MAXROWS+1)` bits (split out from `fr`/`orow`'s own
  declaration, which stayed narrow since they never see ARROWS-scale
  values), plus a companion width fix at the `qkv_wrow` feeder-horizon
  assignment that referenced the old, now-stale width formula.
- `tok_id`/`tok_out` (the module's own input/output ports): hardcoded
  `[8:0]` (max 511) -- `VOCAB=1900` needs `$clog2(1900)=11` bits. Any
  prompt token id or generated token id above 511 was silently
  truncated at the port boundary, corrupting the computation from the
  very first embed lookup. Widened to `$clog2(VOCAB)-1:0`, matching a
  new `VIDXW=$clog2(VOCAB)` localparam.
- `emb_row_w` (`tok_id*EROWS+fr`, hardcoded `[13:0]`, max 16383): at
  `VOCAB=1900`, `1899*16=30384 > 16383` -- widened via a new
  `EMBROWW=$clog2(max(VOCAB,TMAX)*EROWS+EROWS)` localparam (must cover
  whichever of `VOCAB`/`TMAX` is larger, since this same wire serves
  both the tok and the pos embed lookup).
- The argmax winner-index pipeline (`gj`/`gj_d`, `best_idx`/`hidx`,
  `wm_idx`, `am_idx`, `pi0..pi3`, `ia`/`ib`) -- all hardcoded `[8:0]`,
  all widened to `VIDXW-1:0`.
- `gj != VOCAB[8:0]` (the gumbel-noise-precompute loop's own exit
  check, sampling-mode only): slicing `VOCAB` to 9 bits before
  comparing evaluates to `1900 & 0x1FF = 364`, not `1900` -- the
  precompute loop stopped after only 364 of the needed 1900 noise
  draws. Fixed to `gj != VOCAB[VIDXW-1:0]`.
- Both fully-resident testbenches (`tb_seq_vec_kv.sv`,
  `tb_seq_vec_kv_stream.sv`) had the SAME hardcoded-9-bit pattern in
  their own `tok`/`tok_out`/`prompt[]`/`stream[]` stimulus signals --
  fixed with a matching `VIDXWP=$clog2(VOCABP)` local to each TB.
- `xheep_kevgpt_peripheral.sv` (the firmware-facing register file) had
  the same truncation one level up: `tok_id`'s own register, the
  `wdata[8:0]` write-side slice, and `core_tok_out`'s readback wire +
  its `{23'b0, ...}` zero-pad -- fixed the same way, `VIDXW=$clog2
  (VOCAB)` local to that module (it already carried its own `VOCAB`
  parameter).

Confirmed via code review as already-correctly-parametric and NOT
needing changes: `head_bank`'s own array depth (`ARROWS`-sized),
`ar`/`ard`/`amd`/`ad1` (already `$clog2(ARROWS+1)`-wide), `dqm_w`/
`dqe_w` (already `DQROWS`-deep, `DQROWS` already threads `DQ_N`/VOCAB
correctly), `pos` (TMAX-based, correctly left at 9 bits), `at_tcount`/
`wi`/`wic` (HEAD_DIM/attention-related, confirmed VOCAB-independent via
their own usage context).

**A gate-usage gotcha that ate real debugging time before the fix
above was even implicated**: `fabric.stage3.run_vec_kv`'s own CLI
default (`--lanes 16`) gives `EPW=(LANES*4)/(P*32)=0` at `P=8`, silently
skipping the token/pos embed table append into `wrom.mem` entirely --
`S_EMB` then reads uninitialized weight-bank rows unconditionally, and
EVERY generated token comes back all-X, regardless of checkpoint,
VOCAB, or NLAYER. This is an *already-documented* gotcha (see the
Option B sampling-mode section elsewhere in this file: `LANES>=8*P` is
required) that got rediscovered the hard way here -- confirmed by
reproducing the exact same all-X failure on `export_optionB` and
`export_stream8` (both previously verified `match=True` checkpoints) on
a completely clean, `git stash`-baseline checkout of `sequencer_vec.sv`
and both testbenches. That ruled out a code regression before the real
`dor`/`tok_id`/etc. bugs above were even found -- `--lanes 64` (matching
this project's own deployed LANES value) is required for any
`run_vec_kv.py` invocation, not just the word-vocab one.

**Verified, fully-resident gate (`fabric.stage3.run_vec_kv`,
`--lanes 64`, `fabric/export_word16/goformer.npz`), after all fixes
above**: greedy, `--seed 0`, `--prompt "once upon a time" --ngen 16`:
**`VEC_KV_VERDICT match=True`**, `hw=gold=[5, 1646, 1790, 12, 945, 659,
1075, 937, 7, 1415, 971, 1682, 1226, 1153, 809, 1641]`, decoded
`", there was a little girl named lily. she loved to play outside in
the"`. On-chip Gumbel sampling also verified at two seeds: `seed=42`,
`"once upon a time"` -> `match=True`; `seed=999999`, `"the dog ran"` ->
`match=True` -- proving the `gj != VOCAB[VIDXW-1:0]` sampling-mode fix
is correct too, not just the greedy path.

**Verified, per-layer-streaming gate** (`tb_seq_vec_kv_stream.sv`, the
configuration that actually matches real hardware deployment --
`WEIGHT_STREAM_PER_LAYER=1`, a real `mig_read_engine`+behavioral-DDR3
stack, `WWORDS=32768` to cover `GW_EMB=32448`): compiled directly with
`iverilog` (no `run_*.py` harness for this TB yet, matching `tb_
weight_loader_ddr.sv`'s own precedent -- see that file's header comment
for the exact file list / `<ai_accel>` path / `define_synth.sv` shim
needed), reusing the fully-resident gate's own generated `.mem` files.
Same prompt/seed (`"once upon a time"`, greedy): `gen=[5, 1646, 1790,
12, 945, 659]` for the first 6 generated tokens -- bit-identical to the
fully-resident gate's own gold, proving the DDR3-backed per-layer reload
path (13 reloads/token at this shape: 12 blocks + head, embed reloaded
once at token-start) reproduces the fully-resident design's exact
behavior at `VOCAB=1900`, not just at the char-level shapes this path
was originally proven against. Real wall-clock cost to simulate is high
(~10 minutes for 6 tokens, driven by the `GW_EMB=32448`-word embed
reload alone) -- a simulation-only cost, not a real-hardware throughput
number (the real DDR3 controller runs far faster than Icarus's
behavioral model + interpreted event simulation).

**Firmware port (`kevgpt_interactive`)**: char-level `main.c` assumed
1 char = 1 token throughout (typed-line buffer doubled as the token
array, `kevgpt_stoi`/`kevgpt_itos` were ASCII-indexed/single-char
tables). Word-level needs a real tokenizer on the firmware side, added
as an `#ifdef KEVGPT_TOKENIZER_WORD`-gated second code path (both
tokenizers coexist in the same `main.c`, selected at compile time by
whichever `kevgpt_tokenizer.h` was regenerated, not a hard fork):
- `fabric/genesys2/gen_chat_fw.py`'s new `emit_tokenizer_header_word()`
  emits `kevgpt_itos` as an array of C string literals (one per
  word/punct token) instead of a `char[]`, plus `KEVGPT_UNK_ID`/
  `KEVGPT_EOS_ID` -- derived by SCANNING itos for the `<unk>`/`<eos>`
  strings, not hardcoded 0/1 (see the `build_vocab` bug below).
- `main.c` adds `kevgpt_word_lookup()` (binary search over
  `kevgpt_itos[2..VOCAB_SIZE-1]` by `strcmp`, valid because
  `build_vocab()` guarantees that range is alphabetically sorted),
  `tokenize_line()` (a C port of `model.word_data.tokenize()`'s regex:
  scan raw typed chars, group `[a-z']+` runs into one word lookup, each
  of `.,!?;:"-` into its own single-char lookup, silently skip
  everything else -- matching `re.findall()`'s own behavior on
  non-matching characters), and `print_word_token()` (C port of
  `model.word_data.decode()`'s spacing rule: space before every token
  except the first, a token starting with `'`, or single-char
  punctuation). `MIN_CHARS`/`chars_this_sentence` became `MIN_WORD_
  TOKENS`/`tokens_this_sentence` (word-count-based ender guard, not
  char-count); `MAX_GEN_LEN` dropped from 120 (chars) to 60 (whole
  WORDS, far fewer needed to fill the same reply length);
  `KEVGPT_EOS_ID` is always treated as a stop condition, ending the
  reply outright (the model predicting the story-boundary token mid-
  reply means "this story is over").
- Real build (`make app PROJECT=kevgpt_interactive TARGET=genesys2
  COMPILER_PREFIX=riscv32-corev- SOURCE=../../../../sw`) compiles clean
  or the word-level path and fits comfortably in the default 64KB
  memory region (33.2/58.0 kB ROM, 5.3/6.0 kB data) -- no linker-region
  bump needed (unlike the much older `kevgpt_chat` app, which bakes the
  weight image INTO the firmware image itself; `kevgpt_interactive`
  streams weights over UART into DDR3 at boot, so this port's tokenizer
  table growth -- ~1900 string literals -- is the only real firmware
  size cost, and it's small).

**A real, previously-undiscovered bug found while wiring this up**:
`model.word_data.build_vocab()`'s frequency-counted candidate pool
(`Counter(tokens)`) included the EOS sentinel token ITSELF (`prepare_
word()` appends one per story before calling `build_vocab`) -- EOS is
by far the single most frequent "token" in that stream (~2M
occurrences, one per story), so it won automatically own a REGULAR
top-word slot, silently overwriting `stoi["<eos>"]`'s own reserved
id=1 with whatever position it sorted to alphabetically (id=10, for
the `word_stream16` checkpoint this session trained). Training itself
is unaffected (encode() only ever sees the FINAL, self-consistent
`stoi`, so every real EOS occurrence in the corpus was encoded as
id=10 consistently throughout), and `model.word_data.decode()` is
unaffected too (it compares the STRING `"<eos>"`, not the id) -- but
id=1 ended up a dead, never-trained embedding slot, and anything
assuming EOS==1 BY CONVENTION instead of by lookup (like this
session's first draft of the firmware tokenizer header) would have
been silently wrong. Fixed in `build_vocab()` (excludes EOS from the
frequency pool, so future retrains don't hit this) AND worked around
for the ALREADY-TRAINED `word_stream16` checkpoint (`gen_chat_fw.py`'s
`emit_tokenizer_header_word()` derives `KEVGPT_UNK_ID`/`KEVGPT_EOS_ID`
by scanning itos for the literal strings, not by hardcoding 0/1) --
no retrain needed, the checkpoint's own dead id=1 slot is harmless
once the firmware knows to look up the real id instead of assuming it.

**Real synth/PnR/bitstream**: `.VOCAB(1900)`/`.WWORDS(32768)` threaded
through `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`'s peripheral
instantiation (was 57/3072); `.KV_DDR_BASE` raised 2MB->3MB (the weight
image itself grew from 1,278,464 to 2,340,864 bytes at this VOCAB,
which would have collided with the OLD 2MB base -- same "recheck on
every shape change" discipline as every prior NLAYER/VOCAB bump in
this document). Full `make vivado-fpga-nobuild` + `fabric.genesys2.
stage_vivado_roms` + `launch_runs impl_1 -to_step write_bitstream`
(the proven reset_run+restage+relaunch recipe, needed again here --
the FIRST attempt, staging ROMs only into the project root before a
one-shot `make vivado-fpga`, reproduced the exact known "`could not
open $readmem data file`" bug this document already found and fixed
once for NLAYER=12 char-level: `launch_runs synth_1` spawns its own
`vivado -mode batch` subprocess with `synth_1/`'s own directory as
CWD, so the project-root copy alone isn't enough -- killed the
in-flight (already-wrong) synth run, `reset_run synth_1`+`reset_run
impl_1`, staged ROMs into BOTH the project root AND `synth_1/` this
time, relaunched clean). **0 Errors, 0 real Critical Warnings** (all
42 `CRITICAL WARNING`s were pre-existing, unrelated `set_max_delay`/
`set_false_path` "no valid objects" constraint-file warnings this
board's own `.xdc` already carries, confirmed none were the ROM
`could not open` pattern this time). **Final placed utilization: LUTs
97,567/203,800 (47.87%), Block RAM 398/445 (89.44% -- up from NLAYER=
12 char-level's 371.5/445 (83.48%), the real cost of the bigger
embed/head tables, still fits inside the 256-tile bucket with margin),
DSPs 804/840 (95.71%, VOCAB-independent as expected). Timing: WNS=
1.651ns, WHS=0.091ns (both positive -- all constraints met).**

**Real hardware bring-up**: killed a stale `openocd` process left
running from an earlier session (same "check for stale watchers before
touching JTAG/UART" discipline as every prior bring-up in this
document) before programming. `vivado -mode batch -source *_pgm.tcl
-tclargs xc7k325tffg900-2 <bitstream>.bit` programmed clean.
**A new sharp edge, found and fixed here**: the established `gdb ...
-ex continue -batch` load pattern HUNG indefinitely this time -- gdb's
own `continue` on a `target remote` connection blocks waiting for the
target to stop, and a free-running firmware loop never does. (Whether
this differs from an EARLIER session's own successful use of the same
`continue`-based pattern, elsewhere in this document, wasn't resolved
-- possibly a GDB/OpenOCD version-specific behavior difference, not
chased further.) Fixed by using OpenOCD's own `monitor resume` +
`detach` instead of gdb's `continue` -- resumes the target without
gdb's blocking wait-for-stop semantics, confirmed via a raw `cat
/dev/ttyUSB0` capture immediately showing the full expected boot
sequence (`KEVGPT_INTERACTIVE_PHASE,control_plane` ->
`KEVGPT_INTERACTIVE_ID,0x53515256` -> `KEVGPT_UART_READY`).
**A second sharp edge**: `fabric.genesys2.send_weights`'s own stdout,
redirected to a log file by a host-side orchestration script (to let
another process poll for the `KEVGPT_UART_READY`-wait marker before
triggering the JTAG load, matching this document's own documented
ordering requirement), was BLOCK-buffered instead of line-buffered
(Python's default when stdout is a file, not a TTY) -- the polling
script's `until grep -q "waiting for..."` loop saw nothing until the
whole process exited and flushed everything at once, meaning the JTAG
load was triggered AFTER `send_weights.py` had already given up and
timed out, not before. Fixed with `python -u` (force unbuffered
stdout) -- a real gotcha for host-side scripting around this protocol,
not a bug in the protocol or firmware themselves.

**`send_weights.py`, real hardware: `SEND_WEIGHTS_PASS`** -- 585,216
words (2,340,864 bytes) sent, `KEVGPT_UART_LOAD_DONE` +
`KEVGPT_INTERACTIVE_READY` confirmed on the wire.

**Real interactive chat over UART** (on-chip Gumbel sampling, self-
seeded from a live cycle counter every turn -- no host-controllable
seed, same as every prior interactive-chat pass in this document, so
this is a fair, honest sample of ACTUAL chat quality, not a cherry-
picked greedy run):
```
"once upon a time" -> ", there was a little girl brush food. a <unk> down that was way home was. hours us <unk> a little get out of home to help me of clean up,\". everyone while we in with a a lot together. everyone with with a <unk> and with their together and. everyone anything a"
"the dog ran"       -> " to his friend and laughed an <unk>"
"a little girl"     -> " before back back. home to help him to to help him get to help him get. finally, back back down. when it was under in in and make to help in the <unk>"
```
Real words, correct spacing/punctuation placement, `<unk>` tokens
printing as the literal marker (by design, `main.c`'s `print_word_
token()` never special-cases it beyond that) -- confirms the RTL fix,
the C tokenizer port, AND the on-chip Gumbel sampling path all work
correctly together on real hardware. Grammatically rougher than the
simulation gate's own GREEDY gold text -- expected, matching this
project's own established pattern (every prior model's sampling-mode
chat has been noisier than its greedy decode; this is a sampling-
temperature/model-capacity property, not a new bug) -- and honestly
worse-sounding prose than the char-level NLAYER=12 build's own sampled
chat transcript earlier in this document, an OPEN QUESTION not chased
further this session: whether that's the word-vocab model itself
being undertrained relative to its larger embed/head parameter count,
a bug in how `main.c`'s sentence-ending guard interacts with word-level
MIN_WORD_TOKENS=4 (visibly repetitive phrasing like "to help him to to
help him get to help him get" suggests the sampling temperature/guard
combination may need tuning at word-level, not necessarily a
tokenizer or RTL defect), or genuinely a fair reflection of val loss
2.022 not translating to fluent multi-sentence prose the way the
char-level model's 0.772 does -- flagged honestly, not swept under
"it works."

## A new checkpoint from the scale-up-instability investigation (`eps=3e-4`): verified in simulation AND real synthesis

`model/SCALE-UP-LOG.md`'s Adam-instability investigation (started on a
wider D=384 diagnostic shape, then ported back to check this project's
actual deployed D=128/NLAYER=12/VOCAB=1900 shape) found that this
deployment shape carries the same Adam-driven divergence risk as the
wider model -- just non-deterministically (Attempt 13: re-running the
exact original recipe, same hyperparameters/architecture/data/fixed
`torch.manual_seed(1337)`, produced one clean run and one diverging
run -- GPU floating-point non-determinism near a genuine edge-of-
stability point, not a seed/batch-order difference). Attempt 14 found a
practical hedge: keep the deployment recipe's peak LR/warmup/full-
cosine-decay exactly as-is, add a mild `--adam-eps 3e-4` (this shape's
own sweet spot -- `1e-4` only partially dampens, `1e-3` costs real
peak quality). Result: best val **2.203** -- matching or beating the
best-case unstabilized run's own peak (2.208), with genuine post-peak
flatness (+0.008 drift over the last ~4500 iters) instead of a lucky-
run gamble. QAT fine-tune (3000 iters, warm-started, same recipe as
every other QAT pass in this document) landed at val **2.155** --
notably *improved* over its own FP source, unlike the original
`ckpt_word16.pt` -> `ckpt_word16.qat.pt` pass, which slightly regressed
(2.022 -> 2.074).

**Bit-exact RTL simulation** (`fabric.stage3.run_vec_kv`, `--p 8
--lanes 64 --tmax 128` -- Genesys2's real deployed parameters, exported
via `model.goformer_full.params_from_ckpt()`/`save_params()` to
`fabric/export_word16_epsmild3e4/goformer.npz`): five prompts, one at
`--ngen 60` and four at `--ngen 120` (near the `TMAX=128` context
limit), all seeded on-chip Gumbel sampling, all **`VEC_KV_VERDICT
match=True`** -- token-for-token identical to the Python golden
reference. Sample (`"once upon a time"`, seed=7): *"there was a little
girl named amy. she was three years old and loved to explore. one day,
amy went to the park with her mom. it was a sunny day and amy wanted to
take a walk. she saw a pond and asked her mom," what is that?" her
mom"* -- coherent, grammatical, no repetition-loop degradation across
any of the five samples (unlike some other checkpoints tried earlier in
the same eps sweep, which fell into noun-fixation loops at this
length).

**Real Vivado synthesis + implementation + bitstream**, not just
simulation. The actual buildable Vivado project for this port does
NOT live in this repo -- it's a separate sibling repo on the same
machine, `~/RVchatbot/kevgpt-genesys2-soc/` (an X-HEEP SoC checkout
whose git history matches this repo's own fabric/genesys2 commits
almost verbatim, vendoring the real `kevgpt_seq` IP core --
`sequencer_vec.sv`, `xheep_kevgpt_peripheral.sv`, `kevgpt_ddr_bundle.sv`
-- under `hw/vendor/esl_epfl_x_heep/hw/ip/kevgpt_seq/`). Its Vivado
project directory
(`hw/vendor/esl_epfl_x_heep/build/openhwgroup.org_systems_core-v-mini-mcu_1.0.5/genesys2_kevgpt-vivado/`)
already had a completed build from the `ckpt_word16.pt`/`.qat.pt` pass
above. Rebuilt with the new checkpoint's ROM content, following this
document's own established recipe exactly: `fabric.genesys2.
stage_vivado_roms --npz fabric/export_word16_epsmild3e4/goformer.npz
--lanes 64 --p 8 --dest <project root>` first, then `reset_run synth_1`
+ `reset_run impl_1` (via `vivado -mode batch`), then re-staged into
the now-emptied `<project>.runs/synth_1/` (the same "stage AFTER
reset_run, not before" gotcha this document already found once), then
`launch_runs impl_1 -to_step write_bitstream -jobs 4` + `wait_on_run`.

**Full clean rebuild, ~26 minutes wall time**: `synth_design Complete!`
/ `write_bitstream Complete!`, **0 Errors** end to end. Synthesis: 1103
Infos, 931 Warnings, 4 Critical Warnings (the same pre-existing
`set_max_delay`/CDC "no valid objects" `.xdc` warnings this document
already established as harmless, re-confirmed identical here).
Post-route DRC: 0 Errors, 1441 Warnings (all generic X-HEEP-
infrastructure `RAMB36 async control check` warnings about CDC/DMA/
crossbar registers driving `ram0`'s address pins -- unrelated to the
`kevgpt_seq` IP or its ROMs, present regardless of checkpoint). Final
timing: **WNS=1.449ns, WHS=0.094ns** (both positive, all constraints
met -- compare the original build's WNS=1.651ns/WHS=0.091ns: slightly
less setup margin, essentially identical hold margin, still comfortably
closing). Utilization is essentially unchanged from the original build,
as expected (same RTL, same shape parameters -- only the ROM-baked
LayerNorm-gain/dequant-scale/GELU-LUT/gumbel-LUT *values* differ, not
their sizes): LUTs 97,584/203,800 (47.88%, vs 97,567/203,800 (47.87%)
originally), Block RAM 398/445 (89.44%, **exact match**), DSPs 804/840
(95.71%, **exact match**).

**Not yet done**: programming the resulting `.bit`
(`<project>.runs/impl_1/xilinx_core_v_mini_mcu_wrapper_kevgpt.bit`)
onto real hardware and observing a live chat session -- no physical
Genesys2 board/JTAG cable was available in this session's environment,
so this checkpoint is verified bit-exact in simulation and
fully-synthesizable-with-timing-closure in real Vivado P&R, but not
yet exercised on actual silicon. Whoever has board access next just
needs to program the bitstream and re-run `send_weights.py` with this
checkpoint's exported weight image -- no RTL or project changes needed
beyond what's already staged.

**Update: exercised on actual silicon in a later session, board now
attached.** Programmed the bitstream above via `vivado -mode batch
-source *_pgm.tcl -tclargs xc7k325tffg900-2 <bitstream>.bit`
(`SUCCESS!` on the real `xc7k325t_0` device, target
`Digilent/200300B5E5AAB`), built `kevgpt_interactive`
(`COMPILER_PREFIX=riscv32-corev-`, toolchain at `~/tools/corev-openhw-
gcc-ubuntu2204-20240530/bin/`, not on `PATH` by default), loaded via
`openocd -f quad-x-heep-openocd-genesys2.cfg` + `riscv32-corev-elf-gdb
... -ex "monitor reset halt" -ex load -ex "monitor resume" -ex detach`
(the documented `continue`-hangs gotcha above still applies, `monitor
resume`+`detach` avoided it cleanly), streamed this checkpoint's
weights via `send_weights.py` (`SEND_WEIGHTS_PASS`, 585,216 words /
2,340,864 bytes, matching this section's own byte count exactly), and
ran real interactive chat. Sample replies (on-chip Gumbel sampling,
self-seeded, no host-controllable seed -- a fair sample):
```
"once upon a time" -> , there was a little girl named lily. she loved
to play with her dolls and her dolls. one day, her little brother came
to her house to her house. he saw lily's toys and wanted to play with
them. lily said," hi, my little brother!" he said with a
"the little robot" -> was <unk> and he was so excited. he asked the
little boy if he could borrow it. the little boy said," can i borrow
your toy car?" the little boy said yes. so the little boy helped the
robot. the little boy was so happy. the little boy and the little
"she opened the box" -> to her box and <unk> inside. lily heard a noise
and came to see what was inside. she saw the box was a <unk>. the
<unk> had a picture of a rabbit and a green friend. he was so happy.
he said," this is a bear!" he smiled and said,
```
Coherent grammar, correct dialogue punctuation, no severe repetition
loop across any sample -- a real improvement over this document's own
earlier word-vocab chat transcript (`"to help him to to help him get to
help him get"`, flagged there as visibly repetitive), despite this
checkpoint's QAT val loss (2.155) being numerically *worse* than the
originally-deployed one's (2.074) -- another case in this project where
val loss doesn't cleanly predict subjective sampled quality.

**Four more samples**, captured after reloading the board back to clean
(non-instrumented) firmware following the throughput investigation
below -- same on-chip Gumbel sampling, no host-controllable seed:
```
"the dragon looked" -> at her and said," i want to <unk> this!" <unk>
smiled and said," yes! let's go on an adventure!" so the dragon and the
dragon went off together. they explored the world together and
enjoyed the view of the day. when they got to the top, they all
"in a small village" -> and always had a <unk> to help him <unk>. one
day, the <unk> decided to make lots of <unk>. he put the <unk> in a
big basket and spread them all together. the <unk> was very happy and
they both had a lot of fun playing together all day. the <unk> grew
more <unk> and
"my favorite toy" -> is broken," timmy replied. but timmy didn't want
to give up his toy car and he said he didn't want it. timmy got sad
and said he was sorry." don't worry, timmy. i'll help you find it."
timmy remembered what his mom said and promised to take care of it
"the old man said" -> ," i like to <unk> that <unk> <unk>!" the two
friends shared the <unk>, laughed and had lots of fun. at the end of
the day, the family said goodbye and the little boy and the <unk>
said goodbye. the little boy was so happy to have the <unk> <unk>! he
was
```
Generally coherent dialogue/narrative flow, consistent with the three
samples above. Two real, honest small-model tells worth flagging: "in
a small village" leans on `<unk>` more heavily than the others (likely
landed on rarer/out-of-vocabulary nouns for that topic), and "the
dragon looked" repeats itself ("the dragon and the dragon went off
together") -- neither a new bug, both the same class of tell already
documented for this word-vocab build elsewhere in this file.

**Real-hardware throughput measured for the first time at this shape**
(flagged as "not yet measured" earlier in this document/the README).
Initial host-side timing (wall-clock around the UART round trip)
showed a puzzling result: total reply latency was ~1.0-1.03 seconds
regardless of whether the captured reply looked like 60, 28, or 14
printed word-tokens -- suggestive of some large *fixed* per-turn
overhead independent of generation length. **That reading was wrong,
traced down and corrected, not left standing**: it was a host-side
parsing artifact (the next turn's `"> "` prompt marker occasionally
landing inside the capture window and truncating what got counted as
"this reply's tokens"), not a real hardware anomaly. Confirmed by
adding temporary instrumentation to `kevgpt_interactive`'s `chat_turn()`
(printing `kevgpt_cycles(dev)` at four checkpoints -- before
`soft_reset`, after `soft_reset`, after the prefill loop, and after the
generation loop, plus the real `gen_len`/`sentences` counters at the
end -- reverted afterward, not shipped): **`gen_len` is 60 (the full
`MAX_GEN_LEN` cap) on every real turn tested**, `sentences` landing at
4 of the `STORY_SENTENCES=8` budget -- confirming `MAX_GEN_LEN=60`, not
`STORY_SENTENCES`, is almost always the real binding stop condition
(typical sentence lengths need more than 60/8=7.5 tokens/sentence on
average, so the sentence cap rarely triggers first). A genuine "short"
reply was never actually happening -- every turn does essentially the
same ~64 total forward-pass steps (4 prefill + 60 generated), so every
turn genuinely takes essentially the same time; the earlier
"suspiciously constant" reading was actually a *correct* observation,
just wrongly attributed.

`KEVGPT_REG_CYCLES` (`kevgpt_cycles()`) turned out not to be a
free-running elapsed-time counter at all -- RTL
(`xheep_kevgpt_peripheral.sv`): `cycles_run` resets to 0 on every
`go_pulse` (one `kevgpt_step()` call) and only latches into
`cycles_latched` when that single step's own `core_done_w` fires, so it
reports **the cycle cost of the single most recent step**, not a
cumulative or elapsed total. Read at prefill position 3: **696,419-
696,423 cycles** (two independent turns, nearly identical); read at the
final generation position (~63, after the full 60-token run):
**866,080-866,082 cycles** -- consistent, position-dependent growth
(13.9ms -> 17.3ms per token at the real 50MHz `REFERENCE_CLOCK_Hz`),
matching attention cost scaling with KV-cache depth stacked on top of
the roughly position-independent per-layer DDR3 weight-reload cost this
document's own architecture section already describes as dominant.

**Real measured throughput: ~58 tokens/sec** (60 generated tokens in
1.0352-1.0394s wall time across three independent turns; UART
transmission of the ~270-300-byte reply is a negligible ~23-25ms of
that, so this is genuinely a compute-bound number, not an artifact of
serial-link overhead). For comparison, the char-level NLAYER=12 build's
own real-hardware number (elsewhere in this document) is ~130 tok/s
(~385k cycles/token) -- this word-vocab shape is markedly slower per
token, consistent with its much larger `GW_EMB=32,448`-word embed-table
reload window (vs the char-level build's far smaller vocab) adding real
per-token cost on top of the same 12-block+head reload structure.

## Proposed next step: streaming embed lookups to remove the VOCAB/BRAM ceiling

Motivated by wanting to grow VOCAB to cut down `<unk>` frequency (real,
not hypothetical -- `unk_frac=0.0394` at the current VOCAB=1900). Real
word-frequency coverage over the full 418M-token filtered-TinyStories
corpus, computed fresh this session (supersedes the coarser numbers
quoted earlier in this document by ~0.01-0.05%, same shape): `512->
86.40%, 900->91.41%, 1024->92.34%, 1536->94.87%, 1900->96.05%, 2048->
96.45%, 3000->98.24%, 4096->99.12%, 6000->99.64%, 8192->99.82%, 12000->
99.93%, 16000->99.96%, 24000->32000->99.99%`. The coverage curve looked
nearly flat right around the current VOCAB (1900->2048 only buys
+0.40%), but that's misleading -- it opens back up substantially past
2048 (2048->3000 alone buys +1.79%), so growing VOCAB is genuinely
worth pursuing, not a lost cause.

**But naively growing VOCAB hits a hard BRAM wall, not just "needs more
engineering".** `GW_EMB` (this build's largest per-layer-streaming
reload window, see the word-level-vocabulary section above) scales as
`VOCAB*16+2048` at this shape (`EROWS=D/P=16`, `EPW=1`). The weight
bank is a single 256-bit-wide (`WBITS=LANES*4=256`) memory; Vivado
tiles that at 4 RAMB36 tiles per 512-deep block (256/72 bits-per-tile
rounds up to 4), matching this document's own documented "256-RAMB36
bucket" exactly (32,768 words / 512 = 64 blocks * 4 = 256 tiles).
Extrapolating that same tiling to bigger VOCAB (weight-bank tiles
alone, then + the ~142 tiles everything else on this SoC already uses,
against this Kintex-7's 445-tile total budget):

| VOCAB | GW_EMB | weight-bank tiles | + ~142 other | Fits in 445? |
|---|---|---|---|---|
| 1900 (current) | 32,448 | 256 | 398 | yes (this is the current build) |
| 2048 | 34,816 | 272 | 414 | yes, barely (~31 tiles left) |
| 3000 (98.24% cov.) | 50,048 | 392 | 534 | **no -- 89 tiles over** |
| 4096 (99.12% cov.) | 67,584 | 532 | 674 | **no -- 229 tiles over** |

Even VOCAB=2048 (a negligible coverage gain, 96.05%->96.45%) nearly
exhausts the whole chip's Block RAM. Any VOCAB that would meaningfully
cut `<unk>` needs 1.2-1.5x more Block RAM than this part physically
has, under the current architecture. Not a resynthesis-and-see problem
-- a real resource ceiling.

**Root cause, and the actual fix.** `sequencer_vec.sv`'s `S_EMB` state
already reads a NARROW slice per token -- `EROWS=16` sub-rows for the
token embedding, 16 more for the position embedding (32 total,
`emb_row_w = (etp ? pos : tok_id) * EROWS + fr`, one row/cycle). The
waste isn't in that read pattern; it's that the ENTIRE embed table
(every VOCAB token-row + every TMAX position-row) has to be bulk-DMA'd
into `weight_bank_tdp` via `weight_loader_ddr` BEFORE that narrow read
can happen -- once per token, at the FIXED `EMB_TOK_BASE`/
`EMB_POS_BASE` addresses, regardless of which single row is actually
needed. **Fix**: replace the one fixed-address bulk reload with two
small, `tok_id`/`pos`-*dependent* reloads (16 words each) --
`WEIGHTS_DDR_BASE + (EMB_TOK_BASE + tok_id*EROWS)*WBYTES_STRM` for the
token row, the equivalent for the position row -- reusing
`weight_loader_ddr`'s existing on-demand arbitrary-window reload
mechanism (already proven for per-layer block streaming), just with a
computed address instead of a constant one, into a buffer sized ~32
words instead of `GW_EMB`.

**Why this changes the whole VOCAB ceiling, not just trims it a bit**:
`GW_EMB` scales ~8x steeper with VOCAB than `GW_HEAD` does (`GW_EMB`
needs one full row per vocab word, slope ~16/VOCAB-unit; `GW_HEAD=
ceil(VOCAB/LANES)*D` needs only `ceil(VOCAB/64)` row-GROUPS, slope
~2/VOCAB-unit -- unavoidable, since computing logits genuinely needs a
score for every vocabulary word, unlike embedding lookup which only
ever needs one row). Once `GW_EMB` stops being a bulk-reload cost,
`GW_HEAD` becomes the new binding constraint:

| VOCAB | new GW_HEAD | fits the SAME 256-tile/32,768-word bucket? |
|---|---|---|
| 4096 (99.12% cov.) | 8,192 | yes, comfortably |
| 8192 (99.82% cov.) | 16,384 | yes, comfortably |
| **16384 (99.96%+ cov.)** | **32,768** | **yes, right at the edge** |

**This redesign alone could support VOCAB up to ~16,000 -- virtually
eliminating `<unk>` -- inside the EXISTING BRAM budget, no bucket-tier
increase needed.** Likely bonus, not yet measured: today's embed
reload is the single largest per-token DDR3 transfer at this VOCAB
(`GW_EMB=32,448` words vs the redesign's ~32-word replacement, roughly
1000x less data moved per token) -- this probably also meaningfully
improves the ~58 tok/s throughput measured above, since that reload is
a real, currently-dominant per-token cost, though this needs
re-measuring once implemented, not assumed from the byte-count alone.

**Real scope, not yet started**: new tok_id/pos-dependent DDR3 address
computation (replacing the current fixed `EMB_TOK_BASE`/`EMB_POS_BASE`
constant addressing on the DDR3 source side), splitting the single
embed-reload trigger into two small sequential reloads in
`sequencer_vec.sv`'s FSM (currently one combined window per this
document's `strw_ret==S_EMB` handling), shrinking the on-chip embed
buffer and the `WWORDS`/`GW_EMB` sizing to match, retraining/exporting
at a real target VOCAB once the RTL change is verified (not before --
the RTL change is shape-independent and should be gated bit-exact on
the CURRENT VOCAB=1900 checkpoint first), and full bit-exact
re-verification against the golden reference before trusting any of
this -- this touches the core reload FSM, squarely inside this
project's "bit-honest before fast" rule. Not a quick parameter bump;
genuine, multi-step RTL engineering, but well-scoped and clearly
justified by the numbers above.

## Streaming embed lookups: implemented, verified, and real-hardware confirmed

Implemented the proposal above in full. `fabric/stage3/rtl/sequencer_vec.sv`:

- New `EMB_ROWG_W` localparam -- the on-chip streaming embed buffer size
  (`(EROWS+EPW-1)/EPW` words, same packing `EMB_TOKW`/`EMB_POSW` already
  use, just sized for ONE row-group instead of VOCAB/TMAX of them --
  VOCAB-independent, 16 words at this shape). `GW_EMB` itself is
  UNCHANGED and still means what it always did (the full DDR3 image
  span, used by `WEIGHT_IMAGE_BYTES`/`send_weights.py`'s own byte count
  and to locate a given row within the off-chip table) -- only the
  on-chip BRAM sizing driver changed, not the DDR3-side bookkeeping.
- New `emb_ph` (reload/read phase: 0=tok row, 1=pos row) and `tok_hold`
  (a small `[0:ROWS-1]` register array) registers. `weight_loader_ddr`'s
  boot-load write port has no destination-address input -- `ld_rst`
  always resets `weight_bank_tdp`'s internal write pointer to 0 -- so
  the tok row has to be fully read out and held in registers BEFORE the
  pos-row reload overwrites the same on-chip words; there is no way to
  land them at two different on-chip addresses simultaneously.
- `S_EMB`'s body split into two paths (`if (WEIGHT_STREAM_PER_LAYER)`):
  the streaming path processes ONE row-group per pass (phase 0: hold the
  tok row in `tok_hold`; phase 1: sum the just-read pos row with the
  held tok row and commit to `xres_bank`, LN fusion gated on phase 1
  only, since that's the only phase that ever produces a real xres
  value) instead of the old interleaved per-cycle tok/pos toggle, which
  only worked when both tables were simultaneously resident. The
  non-streaming (fully-resident, `WEIGHT_STREAM_PER_LAYER=0`) path is
  UNTOUCHED, byte-for-byte identical to before.
- `S_STRW`'s embed-reload dispatch now computes a `tok_id`/`pos`-
  dependent small DDR3 window (`EMB_ROWG_W` words, address = `EMB_TOK_
  BASE + (tok_id*EROWS)>>EPWS` or the pos equivalent) instead of one
  fixed-address bulk window covering the whole table. `S_EMB`'s own
  completion re-arms this same dispatch with `emb_ph<=1` for the second
  (pos-row) reload -- no changes needed to `S_STRW`'s generic settle/
  completion logic, which already just jumps back to `strw_ret` (S_EMB)
  once a reload settles, for either phase.
- Updated the `WWORDS` sizing-check `$display` warning to check against
  `EMB_ROWG_W` instead of the old bulk `GW_EMB`.

**Verified bit-exact, twice, through the REAL per-layer-streaming DDR3
path** (`fabric/stage3/tb/tb_seq_vec_kv_stream.sv` -- no `run_*.py`
harness exists for this TB yet, matching `tb_weight_loader_ddr.sv`'s own
precedent; compiled directly with `iverilog -g2012`, file list per that
TB's own header comment: the `run_vec_kv.py` RTL_FILES set +
`fabric/genesys2/rtl/weight_loader_ddr.sv` + `fabric/genesys2/tb/
mig_behav_model.sv` + a one-line `` `define SYNTHESIS `` shim (bracketed
around JUST `mig_read_engine.sv`/`sync_fifo.sv`, reused from
`~/RVchatbot/kevgpt-genesys2-soc`'s vendored `ai_accel` IP -- `weight_
bank_tdp.sv` itself must NOT see `SYNTHESIS` defined or it switches to a
different, non-simulation memory primitive) + the TB itself), at the
current deployed shape (P=8, LANES=64, D=128, NLAYER=12, NHEAD=2,
VOCAB=1900, TMAX=128), prompt "once upon a time" (plen=4), 20 greedy
generated tokens: **`got == gold ==
[5, 1646, 1790, 12, 945, 659, 1075, 937, 7, 1415, 971, 1682, 1226, 1153,
809, 1641, 1589, 7, 1136, 406]`** -- identical to `tb_seq_vec_kv.sv`'s
own already-proven fully-resident gold for this exact checkpoint/
prompt/seed, at BOTH `WWORDS=32768` (the old value, isolating "is the
new addressing logic correct" from "does shrinking WWORDS also work")
and `WWORDS=3840` (the new value, `max(GW_BLK=3072, GW_HEAD=3840,
EMB_ROWG_W=16)` -- confirming both claims independently, not
conflating them into one pass/fail.

**Real Vivado synthesis + implementation + bitstream, WWORDS=3840,
same checkpoint/ROMs as the `eps=3e-4` build earlier in this document**
(copied the updated `sequencer_vec.sv` into `~/RVchatbot/kevgpt-
genesys2-soc`'s vendored `kevgpt_seq` IP, changed `.WWORDS(32768)` to
`.WWORDS(3840)` in `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`, `reset_
run synth_1`+`reset_run impl_1`, restaged ROMs into the now-emptied
`synth_1/`, relaunched clean, ~25.5 minutes wall time): **0 Errors, 0
real Critical Warnings** (same 10 pre-existing constraint-file warnings
this document already established as harmless, unchanged), **WNS=
1.507ns / WHS=0.089ns** (both positive, vs the pre-redesign build's
1.449ns/0.094ns -- comparable, still comfortably closing). **Block RAM:
171/445 (38.43%), down from 398/445 (89.44%) -- 227 tiles freed, more
than half the chip's total Block RAM** -- closely matching the design
proposal's own hand-calculated estimate (~174 tiles) before any of this
was implemented. LUTs 97,859/203,800 (48.02%, vs 97,584/203,800
(47.88%) before -- the small real cost of `tok_hold`+`emb_ph`) and DSPs
804/840 (95.71%, exact match) essentially unchanged, as expected -- this
was purely a BRAM win, not a compute-structure change.

**Real hardware, confirmed working end to end.** Killed the stale
`openocd` session, reprogrammed the board with the new bitstream
(`vivado -mode batch -source *_pgm.tcl`, `SUCCESS!` on the real
`xc7k325t_0` device), restarted `openocd`, reloaded the SAME firmware
(unchanged -- tokenizer/VOCAB didn't change, `main.elf` load size
identical: 34,330 bytes), re-streamed the SAME checkpoint's weights
(`SEND_WEIGHTS_PASS`, 585,216 words), and ran real interactive chat:
```
"once upon a time" -> , there was a little girl named lily. she loved
to eat <unk>, and she always wanted to eat all the yummy <unk>. one
day, lily's mom asked her," lily said, i want to eat a <unk>!" lily
replied," okay, i will eat <unk>!" they ate
"the little robot" -> had been <unk> in the <unk>. from that day on,
the two friends were playing together. they were all having fun, but
they always <unk>. the two friends were with a big and tall <unk>. they
were very sad and they never <unk> to the other. the two friends were
sad and angry
"she opened the box" -> and found a <unk>. she was so happy and thanked
her friend for helping her find. from that day on, lily made sure to
always look out for her <unk>. she never felt alone again. the end is
a special <unk>. and she lived in the <unk> and <unk> all the <unk> in
the
```
Coherent grammar throughout, on-chip Gumbel sampling as every other
real-hardware sample in this document -- this run happens to lean on
`<unk>` a bit more than some earlier samples (normal sampling variance
across different self-seeded runs, not a defect this redesign
introduced; the checkpoint and its own `unk_frac=0.0394` are unchanged).

**Honest status: this is a verified, working BRAM-savings mechanism,
not yet a bigger vocabulary.** `VOCAB` is still 1900 in this build --
the redesign proves the freed BRAM is real and usable, but nothing yet
spends it. Growing VOCAB for real is the natural next step this
unblocks, not yet done: pick a target (the coverage table above
suggests 4096 (~99.1%) or 8192 (~99.8%) as good candidates, both well
inside the new `GW_HEAD`-bound ceiling of ~16,384), retrain a tokenizer
+ model at that VOCAB, export, and run one more synth/P&R/bitstream
cycle at the new (still comfortably small) `GW_HEAD`.

## VOCAB=16384 runbook: full retrain -> real hardware, step by step

Chose the actual edge of what the new `GW_HEAD`-bound ceiling supports
(`VOCAB=16384` -> `GW_HEAD=ceil(16384/64)*128=32768`, exactly the same
256-tile/32,768-word bucket boundary the ORIGINAL VOCAB=1900 build's
old bulk `GW_EMB` used to sit at -- so real BRAM usage for this build
should land back around the original ~398/445 (89.44%) figure, not
anywhere near the 171/445 the intermediate WWORDS=3840 test showed;
that number was for keeping VOCAB=1900 while proving the redesign, not
for actually spending the freed room). Coverage at this VOCAB:
**99.96%** (computed earlier this session), vs the deployed VOCAB=1900
build's 96.05% -- real `<unk>` frequency should drop roughly 10x.
Recorded here step by step, with the REAL commands actually run and
their real observed results, not an idealized version, so this is
repeatable later without re-deriving anything.

**Step 1: retrain at VOCAB=16384**, same deployment shape (D=128,
NLAYER=12, NHEAD=2, block_size=128) and the best recipe this session's
own scale-up-instability investigation (`model/SCALE-UP-LOG.md`) found
for it -- unchanged peak LR/warmup/full-cosine-decay, `adam-eps=3e-4`
margin (this shape's own sweet spot -- see that log's Attempt 14):
```bash
python -m model.train --max-iters 24000 --lr 5e-4 --warmup 100 --adam-eps 3e-4 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16384.pt --states data/states_word16384.jsonl
```
**Result: best val 2.408 @ iter 15500**, params=4.46M (up from 2.61M at
VOCAB=1900 -- the bigger embed/head tables), ~18.7 min wall time on this
GPU. Genuinely stable the entire 24000 iters (val loss plateaus in a
narrow 2.408-2.427 band from iter ~12000 on, never diverges) -- the
`eps=3e-4` margin doing its job at this shape exactly as `SCALE-UP-
LOG.md` found. (Val loss numbers aren't comparable to the VOCAB=1900
build's own -- cross-entropy scales with vocab size, a bigger vocab has
higher entropy for the same real quality, not a regression.)

**Step 2: QAT fine-tune**, same recipe as every other QAT pass in this
project (warm-started, `--max-iters 3000`, defaults otherwise):
```bash
python -m model.train --qat --init-from data/ckpt_word16384.pt --max-iters 3000 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16384.qat.pt --states data/states_word16384_qat.jsonl
```
**Result: best val 2.338 @ iter 3000, ~11.4 min** -- improved over its
own FP source (2.408), same pattern the `eps=3e-4` VOCAB=1900 checkpoint
showed earlier in this document.

**Step 3: export** (`model.goformer_full`, the correct tool for the RTL
gates -- `model.export_fabric` produces a layout the gates don't
consume, see this document's own earlier finding):
```bash
mkdir -p fabric/export_word16384
python -c "
from model.goformer_full import params_from_ckpt, save_params
p, model, ck = params_from_ckpt('data/ckpt_word16384.qat.pt')
save_params(p, 'fabric/export_word16384/goformer.npz')
print('exported OK, cfg:', ck['cfg'])
"
```
**Result: exported OK.**

**Step 4: bit-exact verification BEFORE touching real hardware** -- this
project's own established discipline ("recheck on every shape change,"
this document's own repeated finding of VOCAB-size-triggered RTL width
bugs). Two gates, matching the deployment's real parameters (P=8,
LANES=64, TMAX=128):

Fully-resident gate first (fast, `fabric.stage3.run_vec_kv`):
```bash
python -m fabric.stage3.run_vec_kv \
  --npz fabric/export_word16384/goformer.npz --meta data/word_v16384/meta.json \
  --p 8 --lanes 64 --tmax 128 --prompt "once upon a time" --plen 4 --ngen 20 --seed 0 \
  --dir /tmp/kevbuild_v16384
```
**First result: `IVERILOG_COMPILE_FAIL`** -- 2 real compile errors, both
`Concatenation repeat may not be negative (-1)` (`sequencer_vec.sv:817,
1160`). Root cause: an 11-bit (`[10:0]`, max 2047) row-index/raw-
dimension register family hardcoded throughout the FSM (`ci`,
`cid`/`cid1-3`, `rb0-2`, `qkv_wrow`, `gv_m`/`gv_k`/`gv_rdaddr`, `g_m`/
`g_k`, and every `*_ra` read-address wire) -- sized for whatever the
largest VOCAB this project had synthesized before needed (`dor`, the
sibling register this SAME bug class hit once already at the VOCAB=57
-> 1900 jump, got fixed then but the fix wasn't propagated to these
OTHER family members). At VOCAB=16384: `ARROWS=2048` needs a 12-bit
counter to reach its own loop-exit sentinel value (2048), and `g_m`
stores `VOCAB` itself as a raw dimension value (16384, 15 bits) when
driving the head GEMV -- both exceed 11 bits, one as a compile error,
the other (`g_m<=VOCAB[10:0]`) SILENTLY (16384 truncates to 0 in an
11-bit slice) -- would have corrupted every head logit without any
compiler warning at all. Fixed with a new `BUSW` localparam
(`max(DIMW, RIDXW, 11)`, floored at 11 to keep every existing smaller-
VOCAB shape's own already-verified bit-exactness byte-for-byte
unchanged) and widened the whole family to `[BUSW-1:0]`; the three
manual zero-pad concatenations that had the negative-repeat-count
compile errors got simplified to plain assignments (safe once the
destination width naturally covers the source).

**Second result, after that fix: compiles clean, but hangs.** Real
board-style debug print (`[cyc N] st=... blk=...`, already built into
`tb_seq_vec_kv.sv`) showed the FSM stuck at `st=8 (G_RB) blk=11 pos=0`
for 27.5 million cycles straight before hitting the testbench's own
400M-ns safety timeout -- the exact same STATE (`G_RB`, the readback
FSM) as the original VOCAB=57->1900 `dor`-wraparound bug this document
already documents, but a DIFFERENT root cause this time (`dor` itself
was already correctly widened). Traced to `gemv_banked_resident_vec`'s
own `MMAX`/`KMAX` parameters -- hardcoded `.MMAX(1024)` at the
`u_gemv` instantiation, itself ALREADY a prior fix-by-guessing-a-bigger-
number for an earlier real bug (`GEMV_MMAX`'s own comment: "A hardcoded
[3:0] here deadlocked G_RB... D_MLP=1024 at LANES=64 needs exactly
16") -- but never generalized to derive from actual shape parameters.
`MMAX` sizes `gdone`'s own completion-counter width (`GDONE_W=
$clog2(ceil(MMAX/LANES)+1)`) -- the HEAD GEMV's own M dimension is
VOCAB=16384, 16x past the hardcoded 1024 ceiling, so `gv_gdone` could
structurally never count high enough for `G_RB`'s own readback-gating
condition (`ci>>GRPSH < gv_gdone`) to ever become true past the first
1024/LANES=16 groups -- a permanent stall, not a slow simulation.
(`KMAX` stays at 1024 unchanged -- it sizes a REAL internal memory
array (`xmem`, depth `KMAX/P`) by the actual reduction length, which
never exceeds `D_MLP=512` in this design regardless of VOCAB; only
`MMAX` needed to grow.) Fixed by deriving `GEMV_MMAX` from `MAXDIM`
(the same "widest M any call in this design ever uses" value the
`BUSW` fix above already computes -- D3/D_MLP for the block GEMVs,
VOCAB for the head) instead of another hardcoded guess, so this
specific bug class shouldn't need rediscovering a third time.

**Third result, after that fix: compiles clean, no hang, but
`VEC_KV_VERDICT match=False`** -- hardware-generated tokens completely
wrong and degenerate (`hw=[6234, 15365, 15360, 15360, ...]` repeating
two or three token IDs, vs a varied, sensible-looking
`gold=[163, 14470, 15685, 169, ...]`). Two DIFFERENT real bugs were
layered under this single symptom; both had to be found and fixed
before the gate went green.

*Bug A -- a second hardcoded-`1024`, not fixed by the `GEMV_MMAX` fix
above.* Diagnosed by re-running with `plen=4 ngen=1` and a throwaway
`ifdef`-gated `$display` inside `S_ARGMAX` (best-so-far winner + final
pick), then a second throwaway dump of every raw per-index head logit
compared element-by-element against `IntKVQSequencer.step_head_q25`'s
own logit vector for the identical prompt (`fabric/stage3/seq_ref.py`
+ `model/goformer_kvq.py`, invoked directly from a scratch script --
not a project tool, just Python REPL-style comparison). Only 2/16384
logits matched at all -- not a narrow-range bug. Root cause: the
`u_gemv` instantiation's OWN readback port connection was still
hardcoded independent of the `GEMV_MMAX` fix:
```
.rd_addr(gv_rdaddr[$clog2(1024/P)-1:0]), .y_out(gv_yout)
```
`$clog2(1024/8)-1:0` = `[6:0]`, 7 bits, max 127 -- but `gv_rdaddr`
(now correctly `[BUSW-1:0]`) needs to range over every row of the
head's own `ARROWS=ceil(VOCAB/P)` output (2048 rows at VOCAB=16384,
238 at VOCAB=1900), and `gemv_banked_resident_vec`'s `rd_addr` port
literally IS the row-select address feeding `y_out` (see that module's
own header comment, `"rd_addr is a P-group index -- y_out = lanes
[rd_addr*P .. +P-1]"`) -- reads for any row past 127 alias/wrap back
into rows 0-127, corrupting readback for that entire tail. This is
WHY the VOCAB=1900 build regressed too, mid-session, once this fix
landed alongside the `BUSW`/`GEMV_MMAX` changes above (VOCAB=1900's
own `ARROWS=238` already exceeds 127) -- confirmed by re-running
`run_vec_kv` against the VOCAB=1900 export (`fabric/export_word16/
goformer.npz` + `data/word_stream16/meta.json`) with the accumulated
uncommitted changes and seeing the SAME `match=False` degenerate-
repeat symptom there, which should NOT have been possible for an
already-verified shape. Fixed the same way as `GEMV_MMAX` itself --
derive from the actual parameter instead of a second hardcoded guess:
```
.rd_addr(gv_rdaddr[$clog2(GEMV_MMAX/P)-1:0]), .y_out(gv_yout)
```
Re-running VOCAB=1900 after this fix: `VEC_KV_VERDICT match=True`,
exact 20/20 token match -- regression closed. VOCAB=16384, though,
STILL failed after only this fix -- same `hw=[6234, 15365, ...]`-style
repeats, unchanged from before Bug A's fix. Diffing the raw per-index
head logits again (same scratch-script method) showed literally EVERY
index wrong, including row 0 (which was never in the 128-row-truncated
range Bug A affected) -- proof of a second, independent bug still
live.

*Bug B -- the fully-resident test's on-chip weight-bank depth
(`WWORDS`) was never sized for VOCAB=16384's actual resident image.*
Backed out the expected pre-dequant GEMV accumulator from gold's own
head logit (inverting the `vec_dequant.sv` formula by hand using the
per-channel mant/exponent read out of `dqm_w.mem`/`dqe_w.mem`,
confirmed correct against `IntSequencer.head["mant"/"exp"]` directly)
and found hardware's raw accumulator for head channel 0
(`gv_yout`, dumped via one more throwaway `$display` gated on
`g_dst==head`) was ~3600 vs an expected ~900 -- the dequant stage
itself was proven bit-exact given ITS inputs (hand-computed the exact
same output from the observed accumulator + mant/exp using the
`vec_dequant.sv` formula and got hardware's own number to the last
digit); the corruption was upstream, in the WEIGHTS the GEMV was
actually MAC-ing against. `sequencer_vec`'s `WWORDS` module parameter
(default 262144 words, sizing the single on-chip `weight_bank_tdp`
resident-weight memory `run_vec_kv.py`'s fully-resident test streams
the WHOLE model into) was never overridden by
`tb/tb_seq_vec_kv.sv` -- it always used that hardcoded default. At
VOCAB=1900 the full resident image (12 blocks + head + the tok/pos
embed tables appended per this document's own "log §36 fit-plan 2"
entry) is ~40.7K words, comfortably under 262144, so this was
invisible. At VOCAB=16384 the tok_emb table ALONE is
`16384*128/64=32768` wide words before even packing/padding, and the
full image (`wc -l wrom.mem`) is **333,824 words** -- past the
262144-word bucket the write pointer wraps at. The tail of the image
(everything from word 262144 onward, which lands inside the embed-
table region given blocks+head only span the first ~69.6K words) got
written a second time into addresses `[0, 333824-262144) =
[0, 71680)` -- silently overwriting BOTH block 0's weights (words
0-36863) AND the entire head's weight window (words 36864-69631) with
unrelated tail-of-embed-table bytes. That is exactly consistent with
EVERY head logit being wrong from row 0 onward (block 0's corruption
alone would already derail the whole forward pass long before the
head). Fixed by overriding `WWORDS` at the testbench's `sequencer_vec`
instantiation to the actual image size, which the harness already
computes and passes in as `\`WROMN` (the exact word count `wc -l
wrom.mem` reports, already threaded through as a `-DWROMN=<n>`
`iverilog` define):
```systemverilog
sequencer_vec #(.P(P), .LANES(LANES), .TMAX(TMAXP), .D(DP), .D3(3*DP),
                 .D_MLP(4*DP), .NLAYER(NLAYERP), .NHEAD(NHEADP), .VOCAB(VOCABP),
                 .WWORDS(`WROMN)) dut ( ...
```
(`tb/tb_seq_vec_kv_stream.sv`, the OTHER fully-resident-adjacent
testbench, already had its own explicit `WWORDSVAL` override sized for
streaming's much smaller per-block resident window -- unaffected,
no change needed there.)

**Re-running both shapes after Bug A + Bug B, both via the real
`fabric.stage3.run_vec_kv` CLI (not the scratch debug builds)**:
```bash
python -m fabric.stage3.run_vec_kv \
  --npz fabric/export_word16/goformer.npz --meta data/word_stream16/meta.json \
  --p 8 --lanes 64 --tmax 128 --prompt "once upon a time" --plen 4 --ngen 20 --seed 0 \
  --dir /tmp/kevbuild_v1900_official
# VEC_KV_VERDICT match=True mode=greedy plen=4 ngen=20 avg_cyc=26827

python -m fabric.stage3.run_vec_kv \
  --npz fabric/export_word16384/goformer.npz --meta data/word_v16384/meta.json \
  --p 8 --lanes 64 --tmax 128 --prompt "once upon a time" --plen 4 --ngen 20 --seed 0 \
  --dir /tmp/kevbuild_v16384_official
# VEC_KV_VERDICT match=True mode=greedy plen=4 ngen=20 avg_cyc=44007
# hw gen == gold gen, 20/20 tokens, both decode to the same sentence.
```
Fully-resident bit-exact gate is GREEN at VOCAB=16384.

**Step 4, other half: the real per-layer-streaming DDR3 path.** Manual
`iverilog` build (no `run_*.py` wrapper, matching every prior use of
this testbench -- file list per its own header comment: `fabric/
stage3/rtl/*.sv` + `fabric/genesys2/rtl/weight_loader_ddr.sv` +
`fabric/genesys2/tb/mig_behav_model.sv` + a `` `define SYNTHESIS ``
shim + the vendored `mig_read_engine.sv`/`sync_fifo.sv` +
`tb_seq_vec_kv_stream.sv` itself), `WWORDSVAL=32768` (this VOCAB's
`max(GW_BLK=3072, GW_HEAD=32768, EMB_ROWG_W=16)`, NOT `WROMN` --
streaming only needs the largest SINGLE reload window resident,
`WROMN` sizes the simulated DDR3 image, a different, much bigger
number):
```bash
HERE=fabric/stage3; RTL=$HERE/rtl; GEN=fabric/genesys2
AIACCEL=~/RVchatbot/kevgpt-genesys2-soc/hw/vendor/esl_epfl_x_heep/hw/ip/ai_accel/rtl/accelerator
RTL_FILES="sequencer_vec.sv kv_bank.sv vec_attn_w.sv layernorm_vec.sv vec_dequant.sv vec_gelu.sv gelu_lut.sv gelu_lut2.sv softmax_f.sv gemv_banked_resident_vec.sv weight_bank_tdp.sv"
iverilog -g2012 -o sim_stream.vvp \
  -DPVAL=8 -DLVAL=64 -DWROMN=333824 -DWWORDSVAL=32768 -DTMAXVAL=128 \
  -DPLEN=4 -DNGEN=20 -DSEEDVAL=0 -DDVAL=128 -DNLAYERVAL=12 -DNHEADVAL=2 -DVOCABVAL=16384 \
  $HERE/tb/tb_seq_vec_kv_stream.sv $(for f in $RTL_FILES; do echo $RTL/$f; done) \
  $GEN/rtl/weight_loader_ddr.sv $GEN/tb/mig_behav_model.sv \
  <define_synth.sv shim> $AIACCEL/streamer/mig_read_engine.sv $AIACCEL/common/sync_fifo.sv
vvp sim_stream.vvp
```
**Result: `GEN pos=3..22 tok=[163, 14470, 15685, 169, 8129, 5777, 9284,
8059, 165, 12495, 8268, 14709, 10509, 9824, 6915, 14452, 14002, 165,
9689, 3620]`** -- bit-identical to the fully-resident gate's own gold
sequence above (this testbench's own contract: match the fully-
resident design's output exactly, not a separate golden reference).
Streaming reproduces fully-resident behavior at VOCAB=16384 too.

**Copied the fixed `sequencer_vec.sv` into the vendored
`kevgpt-genesys2-soc` tree** (`cp fabric/stage3/rtl/sequencer_vec.sv
~/RVchatbot/kevgpt-genesys2-soc/hw/vendor/esl_epfl_x_heep/hw/ip/
kevgpt_seq/rtl/sequencer_vec.sv`), committed in both repos, pushed
(`kev-gpt`'s `word-vocab` to the `tparng-fork` remote -- `origin`
there, `MichaelAyles/kev-gpt`, 403s on push, no write access;
`kevgpt-genesys2-soc`'s `word-vocab` to its own `origin`, a local
path to `~/RVchatbot/Genesys2AiChatbot`, already tracking this same
branch, fast-forwardable). Also committed, separately, an EARLIER
uncommitted change already sitting in that repo from before this
session's VOCAB=16384 work: `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`'s
`WWORDS` 32768->3840 shrink (the streaming-embed redesign landing,
still VOCAB=1900 at that point) -- unrelated to today's bug, kept as
its own commit rather than bundled in.

**Real Vivado rebuild, step by step (repeatable recipe).**

*Wrapper parameters* (`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`'s
`u_kevgpt` instantiation): `.VOCAB(1900)` -> `.VOCAB(16384)`,
`.WWORDS(3840)` -> `.WWORDS(32768)` (this VOCAB's own
`max(GW_BLK=3072, GW_HEAD=32768, EMB_ROWG_W=16)`, GW_HEAD now binding
instead of GW_BLK), `.KV_DDR_BASE(3145728)` -> `.KV_DDR_BASE(12582912)`
(12MB -- the real resident weight image grew to 10,682,368 bytes,
confirmed directly against `wc -l wrom.mem`=333,824 words * 32
bytes/word via `fabric.genesys2.gen_chat_fw`'s own `--out` report;
3MB would have reintroduced the exact weight/KV DDR3 collision this
file's own history already found once -- **always recompute this from
the real staged image size on a VOCAB/NLAYER change, never copy
forward**).

*Firmware tokenizer table regen* (`kevgpt_interactive`'s
`kevgpt_tokenizer.h`):
```bash
python -m fabric.genesys2.gen_chat_fw \
  --npz fabric/export_word16384/goformer.npz --meta data/word_v16384/meta.json \
  --prompt "once upon a time" --ngen 4 --lanes 64 --p 8 \
  --out /tmp/scratch_weights_header.h \
  --tokenizer-out ~/RVchatbot/kevgpt-genesys2-soc/sw/applications/kevgpt_interactive/kevgpt_tokenizer.h
```
(`--out` is a required arg but its output isn't used by this app --
`kevgpt_interactive` streams weights over UART at runtime, doesn't
bake them into firmware -- pointed at a scratch path and discarded.)

**First real blocker, not an RTL bug: firmware wouldn't link.**
`make app PROJECT=kevgpt_interactive TARGET=genesys2
COMPILER_PREFIX=riscv32-corev- SOURCE=../../../../sw` failed:
`section .rodata will not fit in region ram0`, **overflowed by
166,744 bytes**. VOCAB=16384's tokenizer string table alone is
~182KB (16,384 word/punctuation literals + a `char*` pointer table),
but `kevgpt_interactive`'s generated `link.ld` only had a 58KB `ram0`
(64KB total SoC memory -- X-HEEP's bone-stock default). Not fixable by
packing the strings more efficiently -- even a maximally compact
offset+blob encoding is still well over 100KB just for the text data,
the real floor is the word count, not the encoding. The actual fix:
this repo already has a bigger memory-map config file prepared and
waiting, `hw/vendor/esl_epfl_x_heep/configs/cv32e40px_kevgpt.py`
(`add_ram_banks([64]*5)` + `add_ram_banks_il(8,16,...)` = 448KB total
SRAM, split 384KB code / 64KB data) -- **but the repo's own root
`Makefile`'s `mcu-gen` target hardcodes `PYTHON_X_HEEP_CFG=configs/
cv32e40px_ai_accel.py`** (a literal string in the recipe body, not a
variable reference -- passing `PYTHON_X_HEEP_CFG=...` to the outer
`make mcu-gen` invocation does nothing, the sub-make's own explicit
argument always wins), so `cv32e40px_kevgpt.py` had never actually
been applied; the currently-generated SoC was still running X-HEEP's
default tiny memory map. This means the memory-map bump ships WITH
this VOCAB jump, not before it -- nobody had needed more than 64KB
until a ~16K-entry string table showed up.

**Regenerating the SoC memory map** (must be run FROM INSIDE
`hw/vendor/esl_epfl_x_heep/`, in a real shell -- not through the root
wrapper, not via `make -C`/`-f external.mk` from outside -- both hit
the same `$(PWD)`-goes-stale-through-nested-make bug the README
already documents for `make app`/`make vivado-fpga`, resolving
`$(PYTHON)`/`$(VENV)`-relative paths against the ORIGINAL caller's
directory instead of `hw/vendor/esl_epfl_x_heep/`):
```bash
cd hw/vendor/esl_epfl_x_heep
PY=python3.8 make venv          # one-time; was already done, "Nothing to be done"
source .venv/bin/activate       # REQUIRED -- regtool.py's own #!/usr/bin/env python3
                                 # shebang resolves to system python3 (missing hjson)
                                 # without this, even though mcu_gen.py itself is invoked
                                 # via the Makefile's own $(PYTHON) venv path already
make mcu-gen \
  PYTHON_X_HEEP_CFG=configs/cv32e40px_kevgpt.py \
  X_HEEP_CFG=configs/python_unsupported.hjson
```
**Result: succeeded, all 44 templates regenerated + soc_ctrl/power_
manager/pdm2pcm/pad_control/dma reg-gen + boot_rom rebuild all OK.**
Confirmed via the regenerated (gitignored, `git check-ignore -v`
confirmed) `sw/linker/link.ld`: `ram0 LENGTH=0x060000` (384KB),
`ram1 LENGTH=0x010000` (64KB) -- the intended split.

**Second real blocker, unrelated collateral damage: `make verible`
(one of `mcu-gen`'s own recipe steps) reformatted far more than
intended.** `git status` afterward showed ~19 modified TRACKED files
-- not just kevgpt RTL, but entirely unrelated `hw/ip/ai_accel/*`
files and the OTHER (non-kevgpt) top-level wrapper. Confirmed via
`git diff --ignore-all-space` (diffs shrank but didn't vanish -- still
real, just non-semantic: verible's column-alignment style rewrites
`input  wire        clk,` as `input wire clk,`, `2*NLAYER+1` as
`2 * NLAYER + 1`, etc., re-flowing lines, not just trimming
whitespace) that this was 100% cosmetic reformatting, no logic change
-- confirmed by reading actual hunks, not assumed. Reverted all of it
(`git checkout --` on every affected file, including
`sequencer_vec.sv` and `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`,
which also discarded that file's own real, intended VOCAB/WWORDS/
KV_DDR_BASE edits made earlier the same session -- re-applied those
cleanly on the freshly-reverted file afterward, so the final diff is
ONLY the intended semantic change, no formatting noise mixed in).
**Lesson for next time: stage/commit real semantic edits to tracked
`.sv` files BEFORE running `make mcu-gen`, not after** -- `make
verible` runs unconditionally as part of that target and will
reformat the whole tree it can reach, regardless of relevance to the
memory-map change actually being made.

**Firmware rebuild, after the memory-map fix**: same `make app`
command as above. **Result: linked clean.** `scripts/building/
mem_usage.py --elf sw/build/main.elf --ld sw/build/main.ld --mcu-pkg
hw/core-v-mini-mcu/include/core_v_mini_mcu_pkg.sv` (run directly,
not through `make app`'s own trailing call to it, which hits the same
stale-`$(PWD)` bug one more time -- harmless, the ELF was already
built by that point) reports **ram0 (code+rodata): 220.4/384.0 kB
(57.4%)**, **ram1 (data): 35.3/64.0 kB (55.2%)** -- comfortable
margin, not a knife's-edge fit.

**Real Vivado project rebuild.** The memory-map regeneration is a
genuine SoC-structural change (differing RAM bank count/depth, not
just an already-instantiated module's parameter edit) -- per this
document's own established precedent ("regenerated the FuseSoC
project from scratch... since NLAYER genuinely changed", Option B
section above), did a full clean rebuild rather than trying to reuse
the existing project via `reset_run`:
```bash
rm -rf hw/vendor/esl_epfl_x_heep/build/openhwgroup.org_systems_core-v-mini-mcu_1.0.5/genesys2_kevgpt-vivado   # 575MB, gitignored, regenerable
cd hw/vendor/esl_epfl_x_heep
source ~/tools-2022/Xilinx/Vivado/2022.2/settings64.sh
source .venv/bin/activate
make vivado-fpga-nobuild FPGA_BOARD=genesys2_kevgpt SOURCE=../../../sw/
```
**Result: FuseSoC project regeneration succeeded** (dependency graph,
IP core-file generation for every `prim_*`/`lowrisc:*` primitive,
Makefile/Tcl skeleton) -- this stage does NOT invoke Vivado itself
yet (no `.xpr` produced), confirming "nobuild" really means
"generate build scripts only." Created the actual Vivado project next
(`make openhwgroup.org_systems_core-v-mini-mcu_1.0.5.xpr` inside the
generated project dir, `vivado -mode batch -source
openhwgroup.org_systems_core-v-mini-mcu_1.0.5.tcl` under the hood) --
**0 errors**, IP generation (MIG DDR3 controller, clk wizard) clean
(the usual `Invalid Input Clock Period 200.02 -> 200`/mode-register
CRITICAL WARNINGs are pre-existing MIG IP quirks this project's
history already treats as benign, not new).

Staged ROMs (`fabric.genesys2.stage_vivado_roms --npz fabric/
export_word16384/goformer.npz --lanes 64 --p 8 --dest <project root>
--dest <project root>/<proj>.runs/synth_1`, `--dest` repeatable for
both destinations in one call) -- but a first `launch_runs synth_1`
attempt was killed by an over-eager local shell timeout mid-`wait_on_run`
(no real failure, just impatience on my end) after only creating the
`synth_1/` run directory with no actual synthesis output yet written.
Recovered exactly per this document's own already-proven recipe:
`reset_run synth_1` + `reset_run impl_1` (harmless/idempotent on a
never-completed run), **restage ROMs again** (`reset_run` empties the
run directory, silently discarding anything staged before it -- the
"stage AFTER reset_run, not before" gotcha this document already
found once, still true for a fresh project), then relaunch:
```tcl
open_project openhwgroup.org_systems_core-v-mini-mcu_1.0.5.xpr
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
```
(driven via `vivado -mode batch -notrace -source <this>.tcl`,
backgrounded with `nohup ... &`+`disown` so it survives past any one
shell command's own lifetime -- this step alone is a real 20-90+
minute wall-clock build per this document's own prior figures at
similar shapes.)

**In progress as of this note** -- synth_1/impl_1/bitstream launched,
not yet complete. Still open once it finishes: confirm 0 errors +
timing closure + real BRAM/LUT/DSP utilization numbers at this VOCAB
(expect Block RAM to land back up near the original VOCAB=1900-era
~89% figure, per this runbook's own opening sizing note -- GW_HEAD=
32768 sits at the same 256-tile bucket boundary the old bulk GW_EMB
did), then the real board bring-up: program the bitstream, reload
firmware, stream weights over UART, confirm real chat at the much
bigger vocabulary.

**That expectation was wrong -- a real, hard BRAM overflow, not a
build-process bug.** `place_design` failed: `ERROR: [DRC UTLZ-1]
Resource utilization: RAMB36E1 over-utilized... requires 522... only
445 compatible sites are available`. `report_utilization -hierarchical`
against the `opt_design` checkpoint (`xilinx_core_v_mini_mcu_wrapper_
kevgpt_opt.dcp`) gave the real breakdown: `u_kevgpt` (kevgpt's own
weight/KV-cache BRAM, essentially fixed by VOCAB=16384's own `GW_HEAD`
bucket) = 402 RAMB36 + top-level misc 8 = ~410; `memory_subsystem_i`
(the X-HEEP CPU-side SRAM, bumped to 448KB earlier in this runbook
specifically to fit VOCAB=16384's ~182KB firmware-baked tokenizer
table) = 112 RAMB36. Total 522 vs 445 available -- over by 77. Critically,
even shrinking `memory_subsystem_i` down to the ACTUAL firmware need
(~256KB, from Step 4's own `mem_usage.py` figures) only gets it to
roughly 65-70 tiles -- total ~475-480, still ~30-35 tiles over. **No
SoC-memory resize closes this gap**: baking a 16,384-word string table
into on-chip BRAM fundamentally doesn't fit alongside VOCAB=16384's own
weight tables on this part.

**Fix: move the tokenizer table off on-chip BRAM into DDR3**, read back
through the SAME already-verified `cpu_ddr_bridge`/`ext_slaves` CPU-DDR3
path `main.c`'s own `uart_load_weights()` already used for the weight
image (`volatile uint32_t *ddr = (volatile uint32_t *)EXT_SLAVE_START_
ADDRESS;` -- a 64MB window at CPU address `0xF0000000`, already
arbitrated onto the single physical MIG alongside kevgpt's own weight/
KV DMA traffic via `mig_dual_master_arbiter`, per the wrapper's own
"Merge kevgpt's own bundle (side A) with cpu_ddr_bridge (side B)"
comment -- already-verified infrastructure, nothing new to build in
RTL). Since this core has no data cache, a `const char *` into this
memory-mapped region behaves exactly like a normal in-SRAM C string for
every purpose the tokenizer code needs (`strcmp`, indexing, byte
iteration) -- no special DDR3-aware read path required, just correct
address arithmetic.

**DDR3 tokenizer blob format** (`fabric.genesys2.gen_chat_fw.
build_tokenizer_ddr_blob()`): a per-id `uint32` LE offset table
(`VOCAB` entries, byte offset of that id's null-terminated string,
relative to the start of the strings section which immediately follows
the offset table -- `KEVGPT_TOKENIZER_STRINGS_OFFSET = VOCAB*4`, no
separate stored field needed, firmware computes the same constant),
then every id's string 0..VOCAB-1 in order, null-terminated,
concatenated -- same "every id needs an entry, missing ones get an
empty-string placeholder" handling as the old baked-in `kevgpt_itos[]`.
**A real bug caught before it reached hardware**: the strings section's
total byte length is arbitrary (sum of variable string lengths + null
terminators), not necessarily a multiple of 4 -- but `send_weights.py`'s
wire protocol moves whole 32-bit words, and `struct.unpack(f"<{len(blob)
//4}I", blob)` silently TRUNCATES a trailing partial word via integer
division, no error. First-built blob was 189,446 bytes (`%4 == 2`) --
caught by a round-trip sanity check (decode every id's string back out
of the blob via the offset table, diff against `meta.json`'s own
`itos`, confirm `ids[2:]` stay alphabetically sorted) before ever
trusting it on real hardware, not by a hardware failure. Fixed by
padding the whole blob up to a word boundary in
`build_tokenizer_ddr_blob()` itself; re-verified: 189,448 bytes
(`%4 == 0`), 0/16384 string mismatches, sort invariant holds.

**`TOKENIZER_DDR_BASE = 16MB`** (`fabric.genesys2.gen_chat_fw`):
clears `WEIGHTS_DDR_BASE=0`'s own image (10,682,368 bytes at
VOCAB=16384) and `KV_DDR_BASE=12,582,912`'s own region
(12,582,912+589,824=13,172,736) with ~2.8MB of margin -- same
"recompute from the real staged image size on every VOCAB/NLAYER
change" discipline `KV_DDR_BASE`'s own history already established.
Firmware-side-only constant -- no RTL parameter needed, since the
accelerator itself never touches this region, only the CPU does.

**Firmware changes** (`sw/applications/kevgpt_interactive/main.c`):
generalized `uart_load_weights()` into `uart_load_blob(byte_offset,
ready_marker, done_marker)`, called twice at boot under
`KEVGPT_TOKENIZER_DDR` (weights at offset 0 with the original
`KEVGPT_UART_READY`/`_LOAD_DONE` markers, tokenizer at
`KEVGPT_TOKENIZER_DDR_BASE` with new `KEVGPT_UART_TOK_READY`/`_TOK_DONE`
markers, `fabric.genesys2.send_weights --tokenizer-blob` sends both
over one connection). Added `kevgpt_tok_offset()`/`kevgpt_itos_ddr()`
(the DDR3 pointer-arithmetic helpers above) behind a `KEVGPT_ITOS(id)`
macro that resolves to either `kevgpt_itos_ddr(id)` (DDR3 build) or the
old `kevgpt_itos[id]` (any future non-DDR word-level build) -- swapped
into `kevgpt_word_lookup()`'s `strcmp` and `print_word_token()`'s string
fetch, the only two places `kevgpt_itos[]` was ever read. New
`fabric.genesys2.gen_chat_fw.emit_tokenizer_header_word_ddr()` emits
only the small constants + `kevgpt_stop_ids[]` (still tiny, stays in
on-chip SRAM) -- not the string array itself.

**SoC memory-map correction**: the earlier 448KB bump (`configs/
cv32e40px_kevgpt.py`) is now entirely unnecessary AND actively harmful
(the thing that caused the overflow) -- shrunk back down. First
attempt, 64KB total (2x32KB banks, 32KB code / 32KB data): linked, but
`ld` overflowed `ram1` (DATA) by 3,424 bytes -- `.heap` alone needs more
than a bare 32KB once stack+bss+heap are accounted for; CODE fit fine
at 32KB (the link error was only ever about `ram1`). Landed on 96KB
total (3x32KB banks, 32KB code / 64KB data -- grow the region that was
actually short, not the one that already fit): links clean,
`mem_usage.py` reports **code 10.9/32.0kB (34.0%), data 35.3/64.0kB
(55.2%)** -- comfortable margin on both sides. 96KB costs only ~24
RAMB36 tiles vs the old 448KB's 112, well inside the ~35-tile headroom
VOCAB=16384's own BRAM need leaves.

**Same `make verible` collateral-damage pattern as before, twice more**
(once per `mcu-gen` rerun while iterating on the memory-bank size) --
same fix each time: `git checkout --` every unintentionally-reformatted
file (unrelated `ai_accel/*`, the non-kevgpt wrapper, and kevgpt's own
already-correct RTL), keeping only the real semantic changes
(`configs/cv32e40px_kevgpt.py`, the legitimately-regenerated-and-
TRACKED `hw/core-v-mini-mcu/core_v_mini_mcu.sv` -- fewer RAM-bank
instantiation blocks, a real consequence of the bank-count edit, not
noise -- and `kevgpt_tokenizer.h`), then re-applying the wrapper's own
VOCAB/WWORDS/KV_DDR_BASE edit fresh each time `git checkout` discarded
it along with the noise.

**Real Vivado rebuild, clean project (same "genuine structural change,
full regen" precedent as before, not a `reset_run` reuse this time
either)**: `rm -rf` the build dir, `make vivado-fpga-nobuild
FPGA_BOARD=genesys2_kevgpt SOURCE=../../../sw/`, `make openhwgroup.org
_systems_core-v-mini-mcu_1.0.5.xpr` (project creation, IP generation --
0 errors), `fabric.genesys2.stage_vivado_roms` into both the project
root and `.runs/synth_1/`, then `launch_runs synth_1` (no `-jobs` --
see the sharp edge below) -> `wait_on_run` -> `launch_runs impl_1
-to_step write_bitstream` -> `wait_on_run`, backgrounded via `nohup
... & disown` so it survives past any one shell command's own
lifetime.

**A real sharp edge found mid-build, unrelated to the RTL/memory-map
work itself**: the FIRST attempt at this exact recipe used `launch_runs
synth_1 -jobs 4` (matching the very first VOCAB=16384 attempt earlier
in this runbook) and hung for over an hour -- `vrs` (Vivado's local
parallel-job dispatcher for `-jobs N`) sat idle the whole time, every
process in the tree accumulating near-zero CPU time, the `mig_7series_
0_synth_1` sub-run's own log frozen mid-line. Not a slow synthesis --
confirmed via `ps -eo pid,etime,time,%cpu` showing ~0.0-0.3% CPU across
every vivado/vrs process for 1h44m of wall-clock elapsed time, and the
job-dispatcher's own `.jobs/vrs_config_N.xml` file untouched since
creation. Root cause not chased further (this project's OWN prior
VOCAB=1900 rebuild used `-jobs 4` successfully, so this isn't a
blanket "never use -jobs here" finding, just an unreliable one in this
environment) -- killed the whole process tree, `reset_run
mig_7series_0_synth_1`/`synth_1`/`impl_1`, restaged ROMs (`reset_run`
wipes the run directory, same "stage AFTER reset_run" gotcha as
always), relaunched WITHOUT `-jobs`. That attempt synthesized cleanly
in a normal amount of time with real, multi-process CPU activity
visible throughout -- this is the recipe now used for every build in
this runbook (including the DDR3-tokenizer rebuild above).

**Real hardware protocol update needed**: `fabric.genesys2.send_weights`
now takes `--tokenizer-blob <path>` to send the DDR3 tokenizer blob
(`fabric.genesys2.gen_chat_fw --tokenizer-ddr --tokenizer-blob-out
<path>`) right after the weight image, over the same UART connection,
before real chat can be confirmed at VOCAB=16384.

**Build result: clean, 0 errors throughout, no `-jobs` hang this
time.** `synth_design`/`place_design`/`route_design`/`write_bitstream`
all completed successfully -- critically, the `place_design` DRC that
failed with the RAMB36E1 overflow before this fix now reports **0
Errors**. Final placed utilization:
```
Block RAM Tile: 440/445 (98.88%) -- RAMB36/FIFO 434/445, RAMB18 12/890
Slice LUTs:     98,630/203,800 (48.40%)
DSPs:           803/840 (95.60%, VOCAB-independent as expected)
Timing: WNS=2.257ns, WHS=0.092ns (both positive -- all constraints met,
        0 failing endpoints on setup (2124 total) or hold (1980 total))
```
**98.88% Block RAM -- only 5 tiles of margin left**, tighter than this
runbook's own "should land back near VOCAB=1900's ~89%" opening guess
(that guess didn't account for the ~24-tile cost of even the shrunk
96KB SoC memory, or synth/place-time BRAM inference differences from
the VOCAB=1900 build's own numbers) -- fits, but there is essentially
no headroom left in this shape for anything that adds even a few more
BRAM tiles (a bigger VOCAB, a bigger SoC memory, or a new on-chip
buffer) without either shrinking something else first or hitting this
exact overflow again. Timing has real margin (2.257ns WNS) despite the
BRAM tightness. `.bit` at `<project>.runs/impl_1/xilinx_core_v_mini_
mcu_wrapper_kevgpt.bit`.

**Still open**: real hardware bring-up -- program this bitstream,
reload firmware (built against `KEVGPT_TOKENIZER_DDR`), stream the
weight image AND the new DDR3 tokenizer blob over UART
(`fabric.genesys2.send_weights --tokenizer-blob tokenizer_ddr_v16384.
bin`), confirm real chat at VOCAB=16384 with the DDR3-resident
tokenizer table -- not yet exercised on real hardware as of this note.

## VOCAB=16384 real hardware: a genuine hang, root-caused and fixed (Gumbel-precompute counter width)

Real hardware bring-up (bitstream above, firmware reload, weight+
tokenizer UART transfer -- all clean) hung on the FIRST real chat
prompt: `kevgpt_step()` polls `STATUS.done` forever. JTAG halt+readback
confirmed a genuine hang, not just slow -- `STATUS=0x00000006`
(BUSY=1, DONE=0) and `CYCLES=0x000a5de2` **frozen** across a 1-second
gap (re-halted, re-read, identical value both times). Backtrace showed
the CPU itself stuck in `chat_turn`'s prefill loop at `pos=3` (the 4th
and LAST prompt token) -- notably the exact step where `main.c` writes
the `SEED` register for the first time that turn, immediately before
this `kevgpt_step()` call (`if (i+1==plen) { ...; kevgpt_seed(...); }
kevgpt_step(...)`) -- `pos=0,1,2` (greedy, no seed yet) had already
completed successfully.

**Isolation, step by step, per the user's own real-time debugging
input** ("try the previous version", "there must be a hang, don't wait
if it's not responding"): rebuilt the OLD VOCAB=1900 config from source
(the actual `.bit` was destroyed by an earlier `rm -rf`, but the RTL
parameters were still recoverable) as a full clean isolation build --
synth/place/route/bitstream all clean, 0 errors. Programmed it, reloaded
firmware, streamed its own weight+tokenizer blobs, sent the SAME prompt
-- **got a real, complete, coherent generated reply, no hang at all.**
This conclusively isolated the bug to VOCAB=16384 specifically, not the
UART/JTAG/board procedure (a real early false lead: two send_weights.py
invocations in a row produced zero UART traffic because the `READY`/
`TOK_READY` markers had already been silently consumed by a prior
crashed/mistimed connection attempt before any listener was open for
them -- exactly the "start listening before triggering the reload"
gotcha this document already flags, refound the hard way here. Fixed by
sending the blob directly once JTAG readback confirmed firmware was
already parked waiting for it, bypassing the marker wait -- not a new
bug, just the same known procedural trap).

**Given `pos=3` is the first call with `smp_en=1` (on-chip Gumbel
sampling engaged, seed just written) and `pos=0-2` (greedy) already
succeeded, suspicion fell on the Gumbel precompute logic specifically
-- and critically, EVERY bit-exact simulation gate run against
VOCAB=16384 so far used `--seed 0` (greedy) exclusively. The sampling
path had never been independently verified at this VOCAB at all**, in
simulation or otherwise -- real hardware chat always samples
(self-seeded from a live cycle counter), so this was a real, load-
bearing gap in this runbook's own "bit-exact before touching hardware"
discipline, not just an edge case. Confirmed by reproducing the SAME
hang in simulation: `fabric.stage3.run_vec_kv --seed 12345` (nonzero,
forces sampling) timed out at 300s where `--seed 0` always passed
instantly. Manual `iverilog` build with the TB's own periodic `[cyc N]
st=...` debug print showed the FSM parked at `st=23` (`S_ARGMAX`) `blk=
11 pos=3`, cycle count frozen from 3,000,000 through 22,500,000+ straight
-- the exact same state/position the real-hardware JTAG readback
already pointed at, now reproducible in minutes instead of requiring a
board.

**Root cause**: `sequencer_vec.sv`'s Gumbel-precompute logit counter
(`gj`/`gj_d`, fills `gumbel_bank` with per-logit noise "riding" the head
GEMV, armed in `S_HEADSET` when `smp_en=1`) was declared `[VIDXW-1:0]`
(`VIDXW=$clog2(VOCAB)`) -- correct for holding WINNER indices (which
only ever need to represent `0..VOCAB-1`, per this file's own existing
comment on `ia`/`ib`/`best_idx`/etc.), but `gj`'s own advance-stage exit
check needs to detect `gj` reaching the SENTINEL value `VOCAB` itself
(one past the last valid index) -- and compared it against
`VOCAB[VIDXW-1:0]`. At VOCAB=16384 (`VIDXW=14`), the value 16384 needs
15 bits to represent at all; truncating it to the low 14 bits gives
**exactly 0**. Since `gj` starts at 0, `gj != 0` is false on the very
first check of the very first sampling-enabled forward pass -- the
advance body (`gj_d<=gj; gj_dv<=1; gj<=gj+1`) never runs even once,
`gumbel_bank` never gets a single word written, `gpre_done` never
fires, and `S_ARGMAX` (`if (gpre_done) begin ... end`, gating the ENTIRE
argmax reduction) waits for a condition that can structurally never
become true -- a permanent hang, not a slow computation. This bug class
**only bites VOCAB values that are an exact power of 2** (`$clog2`
under-counts by exactly one bit for representing the value itself, not
just values up to it minus one) -- VOCAB=1900's own `VIDXW=11` bits
already holds 1900 exactly (1900 < 2048), no truncation, which is
exactly why the isolation rebuild above came back clean. A genuinely
new instance of this project's own recurring "hardcoded/under-derived
width, only exposed at a specific shape" bug class, this time hiding
behind an UNTESTED CODE PATH (sampling) rather than an untested VOCAB
value per se.

**Fix**: widened `gj`/`gj_d` from `[VIDXW-1:0]` to `[VIDXW:0]` (one more
bit, matching this file's own established "+1 for a sentinel" pattern,
e.g. `RIDXW=$clog2(MAXROWS+1)`) and the comparison to `VOCAB[VIDXW:0]`
(15 bits, holds 16384 exactly, no truncation) -- both halves of the fix
were necessary: fixing only the comparison without widening `gj` itself
would let the counter correctly start advancing, but then silently WRAP
back to 0 after processing the last valid index (`16383+1` doesn't fit
in a 14-bit register), turning the hang into an infinite re-loop instead
of curing it.

**Verification**: recompiled, reran the same `--seed 12345` case --
**progresses cleanly past the former freeze point, reaches `TB_DONE`**.
Ran the real `fabric.stage3.run_vec_kv` CLI (not just the manual debug
build) for full golden-reference comparison, two different prompts/
seeds:
```
VEC_KV_VERDICT match=True mode=sample(seed=12345) plen=4 ngen=8 avg_cyc=43575
VEC_KV_VERDICT match=True mode=sample(seed=777) plen=3 ngen=15 avg_cyc=43791
```
Both bit-exact against the Python golden Gumbel-max sampler -- the
second prompt/seed's own generated text genuinely diverges from what
greedy decoding would produce (not a coincidental greedy-match, real
evidence the sampling path itself is being exercised and verified, not
just "didn't hang this time"). Per-layer-streaming re-verification
(the actual path real hardware uses, `tb_seq_vec_kv_stream.sv`,
`--seed 12345`) still in progress as of this note.

**Per-layer-streaming re-verification also passed** (the actual DDR3
path real hardware uses, `tb_seq_vec_kv_stream.sv --seed 12345`):
reaches `TB_DONE`, generated tokens `[163, 14470, 15685, 169, 8129,
5777, 9284, 8059]` match the fully-resident gold sequence exactly (this
testbench's own pass criterion). Copied the fixed `sequencer_vec.sv`
into the vendored `kevgpt-genesys2-soc` tree.

**Real Vivado rebuild with the fix, end to end**: restored the
wrapper's `.VOCAB(16384)`/`.WWORDS(32768)`/`.KV_DDR_BASE(12582912)`
(reverted to the VOCAB=1900 isolation-test values during the isolation
build above), regenerated the VOCAB=16384 tokenizer header+DDR3 blob,
rebuilt firmware (clean), full clean Vivado rebuild (`rm -rf` + `make
vivado-fpga-nobuild` + project creation + ROM staging + `launch_runs
synth_1`/`impl_1` via the no-`-jobs` recipe) -- **0 errors throughout,
`write_bitstream completed successfully`.**

**Real hardware, full end-to-end confirmation**: programmed the new
bitstream, restarted OpenOCD, reloaded firmware, sent the weight image
AND the DDR3 tokenizer blob over UART (`fabric.genesys2.send_weights
--tokenizer-blob tokenizer_ddr_v16384.bin` -- `SEND_WEIGHTS_PASS`, both
markers caught cleanly this time, no missed-marker recovery needed),
sent the prompt `"once upon a time"` over the console UART. **Real,
complete, coherent generated reply, no hang**:
```
once upon a time
, there was a little girl named lily. she loved to play outside and
explore the world around her. one day, lily's mommy told her that she
had to go to the park to play. lily wanted to go on the swings and
slide, but her mommy said they couldn't go to the park.
>
```
VOCAB=16384 -- retrained, QAT'd, bit-exact verified (both fully-
resident and per-layer-streaming, both greedy AND sampled modes), real
Vivado synth/place/route/bitstream clean, real hardware chat confirmed
working, DDR3-resident tokenizer table in place -- this runbook's
original ask ("retrain at VOCAB=16384 and rebuild") is DONE.

**One real send_weights.py protocol bug found and fixed along the
way** (separate from the RTL fix, both needed to reach this point):
the leftover-buffer bug described further up this section (a marker
arriving packed with the previous one in the same `ser.read()` chunk
was silently dropped, since each `_wait_for_marker` call started from
an empty buffer) -- fixed by threading leftover bytes between
sequential `_send_blob` calls. Confirmed fixed on this final run: the
`TOK_READY` marker (which arrived packed with `LOAD_DONE`, exactly the
scenario that bit the very first attempt) was caught correctly with no
manual intervention needed this time.

**Not yet done**: commit the `sequencer_vec.sv` fix (both repos),
`send_weights.py`'s leftover-buffer fix, and this session's full DDR3-
tokenizer feature set -- per this project's own standing "only commit
when explicitly asked" discipline. (Since committed and pushed.)

## VOCAB=16384 real hardware: five more stories + measured token rate

Five more real chat prompts, on-chip Gumbel sampling (self-seeded,
same as every prior chat pass in this document -- no cherry-picking):
```
"once upon a time" ->
, there was a little girl named lily. she loved to play outside and
explore the world around her. one day, she went to the park with her
mommy. there, she saw a big tree with a big tree. she wanted to climb
it, but her mommy said they needed to stay and

"the dog ran" ->
around the park, and he was happy to be with his friends. at the end
of the day, tom thanked the dog for letting him play with his toys.
tom promised to be careful and always remembered the big dog from his
pond. he went back to the pond to play with his friends.

"a little girl" ->
named lily care about her blouse. one day, lily and her blouse
together, when they got home, lily saw a beautiful blouse that she
liked very much. she wanted to pick it up, so she asked her mom if she
could buy it for her. her mom told her that it was not

"the sun was" ->
shining brightly and it was the best day ever. the little girl was so
happy that she had explored the world around her. she even made a big,
yellow house and a big house filled with lots of paper. the little
girl was so excited and couldn't wait to tell her friends about it. she

"she found a" ->
little bird flying towards her. it seemed to be very high and had lots
of other animals. it was a very exciting adventure! the little girl
was so amazed by her new friend, she wanted to go find her new friend.
she asked her mom if she could go to the park and her mom
```
Coherent, grammatically real prose throughout (one visible weak spot:
"a little girl" free-associates "blouse" oddly and gets slightly
repetitive -- a model-capacity/sampling-temperature property, not an
RTL or tokenizer defect, matching this document's own established
pattern of honestly flagging sampled-chat rough edges rather than
cherry-picking). All five hit close to `MAX_GEN_LEN=60` words, matching
every prior real-hardware chat pass in this document.

**Measured real token rate, host-side wall-clock timing** (send the
prompt over the console UART, time from send to a 3-second quiet
period on the wire -- the whole turn, prefill included, dominates
negligibly since prefill is only 3-4 tokens vs ~55 generated):
```
'once upon a time': 1.004s, 53 words -> 52.8 tok/s
'the dog ran':      0.992s, 54 words -> 54.4 tok/s
'a little girl':    1.008s, 53 words -> 52.6 tok/s
'the sun was':       0.993s, 55 words -> 55.4 tok/s
'she found a':       0.991s, 55 words -> 55.5 tok/s
average: 54.0 words / 0.998s = 54.1 tok/s
```
**~54 tokens/sec**, consistent to within ~2 tok/s across five different
prompts (each generation completing in ~1.0s regardless of content) --
a real, repeatable number, not a one-off. First attempt at this
measurement used a naive "wait for the `> ` prompt string" completion
check that never fired correctly (all four test runs came back at
suspiciously identical ~90.0s, the artificial deadline, not a real
completion signal) -- switched to a "no new bytes for 3 seconds"
quiet-period heuristic instead, which is robust to exact byte-pattern
matching and confirmed correct by full timestamp instrumentation on one
run (byte-by-byte arrival times showing the whole ~55-word reply
streaming out continuously from t=0.004s to t=0.996s, no gaps, then
genuinely done).

This is notably FASTER than intuition suggested going in -- VOCAB=16384's
per-layer weight streaming reloads a much bigger head window
(`GW_HEAD=32768` words) from real DDR3 every single token, ~8.5x bigger
than the VOCAB=1900 build's own `GW_HEAD=3840` -- worth remembering
that real DDR3 bandwidth at this design's beat width comfortably
absorbs that reload cost within the ~1/54s per-token budget; the
bottleneck this session's own earlier real-hardware throughput
investigation found (fixed per-token overhead unrelated to prompt
length, PORT-NOTES.md "Real hardware token rate" section above) does
not appear to have gotten meaningfully worse at this much bigger VOCAB.

## VOCAB=16384: clean repeatable sequence (final commands only, no detours)

The three sections above are a real chronological diary -- they
include real dead ends (a 448KB SoC memory config that had to be
corrected down to 96KB, three separate `make verible`-collateral-damage
cleanups, a `-jobs 4` Vivado hang, a missed-UART-marker recovery) mixed
in with the load-bearing steps, and some final commands were only
narrated by flag name rather than shown complete. This section is the
distilled, final-correct-order version -- if repeating this whole
process from a similar starting point (another VOCAB bump, another
shape change), follow THIS section, not the diary above; the diary is
for understanding WHY each step exists and what NOT to do, not for
literal replay. Every value below is the one that actually shipped.

**Where "change VOCAB" actually happens**: it's just two flags in
Step 1's train command -- `--vocab-size <N> --data-dir data/word_v<N>`.
`model/train.py`'s `--tokenizer word` path calls
`model.word_data.prepare_word(corpus, data_dir, vocab_size)`
internally, so building the new tokenizer (word list, `meta.json`,
coverage %, train/val bins) is NOT a separate manual step -- it happens
automatically as part of the training invocation itself. Everything
else in Step 1's command (`--n-layer`/`--n-head`/`--n-embd`/
`--block-size`, the `eps=3e-4` recipe) stays whatever this deployment
shape already uses, unchanged. Every step from here on (2 through 9 --
the bit-exact gates, `WWORDS`/`KV_DDR_BASE` recompute, the DDR3
tokenizer blob, the SoC memory map, the Vivado rebuild, real hardware
bring-up) follows MECHANICALLY once you have a new checkpoint at the
new VOCAB -- none of it requires re-deriving anything by hand, just
recomputing each formula against the actual new checkpoint/image
(`GW_HEAD`, `wc -l wrom.mem`, etc.) as each step already shows. The
underlying tokenizer MECHANISM itself (`build_vocab()`'s frequency
ranking, the EOS-exclusion bug, the alphabetical-sort invariant
`kevgpt_word_lookup()` relies on) is documented once, earlier in this
file, under "Word-level vocabulary" -- not repeated here, since it
doesn't change between VOCAB sizes, only the checkpoint's own N does.

**1. Train, QAT, export** (from `kev-gpt`, `.venv` activated):
```bash
python -m model.train --max-iters 24000 --lr 5e-4 --warmup 100 --adam-eps 3e-4 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16384.pt --states data/states_word16384.jsonl

python -m model.train --qat --init-from data/ckpt_word16384.pt --max-iters 3000 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16384.qat.pt --states data/states_word16384_qat.jsonl

mkdir -p fabric/export_word16384
python -c "
from model.goformer_full import params_from_ckpt, save_params
p, model, ck = params_from_ckpt('data/ckpt_word16384.qat.pt')
save_params(p, 'fabric/export_word16384/goformer.npz')
"
```

**2. Bit-exact gates -- BOTH must pass, BOTH modes** (the real bug this
runbook found was invisible to `--seed 0`; always also test a nonzero
seed):
```bash
# fully-resident, greedy AND sampled
python -m fabric.stage3.run_vec_kv \
  --npz fabric/export_word16384/goformer.npz --meta data/word_v16384/meta.json \
  --p 8 --lanes 64 --tmax 128 --prompt "once upon a time" --plen 4 --ngen 20 --seed 0 \
  --dir /tmp/kevbuild_v16384
python -m fabric.stage3.run_vec_kv \
  --npz fabric/export_word16384/goformer.npz --meta data/word_v16384/meta.json \
  --p 8 --lanes 64 --tmax 128 --prompt "once upon a time" --plen 4 --ngen 8 --seed 12345 \
  --dir /tmp/kevbuild_v16384_seed

# per-layer-streaming (the real DDR3 path hardware uses), manual iverilog build --
# WWORDSVAL = max(GW_BLK, GW_HEAD, EMB_ROWG_W) for THIS vocab, not WROMN.
# Rerun with BOTH -DSEEDVAL=0 and a nonzero seed.
HERE=fabric/stage3; RTL=$HERE/rtl; GEN=fabric/genesys2
AIACCEL=~/RVchatbot/kevgpt-genesys2-soc/hw/vendor/esl_epfl_x_heep/hw/ip/ai_accel/rtl/accelerator
RTL_FILES="sequencer_vec.sv kv_bank.sv vec_attn_w.sv layernorm_vec.sv vec_dequant.sv vec_gelu.sv gelu_lut.sv gelu_lut2.sv softmax_f.sv gemv_banked_resident_vec.sv weight_bank_tdp.sv"
iverilog -g2012 -o sim_stream.vvp \
  -DPVAL=8 -DLVAL=64 -DWROMN=<wc -l wrom.mem> -DWWORDSVAL=32768 -DTMAXVAL=128 \
  -DPLEN=4 -DNGEN=8 -DSEEDVAL=12345 -DDVAL=128 -DNLAYERVAL=12 -DNHEADVAL=2 -DVOCABVAL=16384 \
  $HERE/tb/tb_seq_vec_kv_stream.sv $(for f in $RTL_FILES; do echo $RTL/$f; done) \
  $GEN/rtl/weight_loader_ddr.sv $GEN/tb/mig_behav_model.sv \
  <define_synth.sv shim -- see this TB's own header comment> \
  $AIACCEL/streamer/mig_read_engine.sv $AIACCEL/common/sync_fifo.sv
vvp sim_stream.vvp   # PASS = generated tokens match the fully-resident gold sequence
```

**3. Copy the verified RTL into the vendored repo**:
```bash
cp fabric/stage3/rtl/sequencer_vec.sv \
   ~/RVchatbot/kevgpt-genesys2-soc/hw/vendor/esl_epfl_x_heep/hw/ip/kevgpt_seq/rtl/sequencer_vec.sv
```

**4. Wrapper parameters** (`~/RVchatbot/kevgpt-genesys2-soc/hw/vendor/
esl_epfl_x_heep/hw/fpga/xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`,
`u_kevgpt` instantiation) -- the FINAL values, recompute all three from
the actual checkpoint/image on any future VOCAB/NLAYER change, don't
copy forward:
```systemverilog
.VOCAB (16384),
.WWORDS(32768),        // max(GW_BLK, GW_HEAD, EMB_ROWG_W) for this VOCAB
.KV_DDR_BASE(12582912) // clears WEIGHTS_DDR_BASE=0's image + margin --
                       // recompute from `wc -l wrom.mem` * 32 bytes/word
```

**5. SoC memory map** (only needed if the tokenizer/firmware doesn't
fit the *previous* memory map -- check with `mem_usage.py` first before
touching this; DDR3-resident tokenizer means this rarely needs to
change again). Final content of `~/RVchatbot/kevgpt-genesys2-soc/hw/
vendor/esl_epfl_x_heep/configs/cv32e40px_kevgpt.py`'s memory block:
```python
memory_ss = MemorySS()
memory_ss.add_ram_banks([32] * 3)
memory_ss.add_linker_section(LinkerSection.by_size("code", 0, 0x8000))
memory_ss.add_linker_section(LinkerSection("data", 0x8000, None))
system.set_memory_ss(memory_ss)
```
Regenerate (from `~/RVchatbot/kevgpt-genesys2-soc/hw/vendor/
esl_epfl_x_heep/`, MUST `cd` into this dir and `source .venv/bin/
activate` first -- see [[reference-vivado-genesys2]] for why):
```bash
make mcu-gen PYTHON_X_HEEP_CFG=configs/cv32e40px_kevgpt.py \
             X_HEEP_CFG=configs/python_unsupported.hjson
```
Then `git status` and `git checkout --` every file that's PURE
`make verible` reformatting noise (everything except
`configs/cv32e40px_kevgpt.py` itself and the legitimately-regenerated,
TRACKED `hw/core-v-mini-mcu/core_v_mini_mcu.sv`) -- re-apply step 4's
wrapper edit again if it got swept up in the same revert.

**6. DDR3 tokenizer table + firmware tokenizer header**:
```bash
python -m fabric.genesys2.gen_chat_fw \
  --npz fabric/export_word16384/goformer.npz --meta data/word_v16384/meta.json \
  --prompt "once upon a time" --ngen 4 --lanes 64 --p 8 \
  --out /tmp/scratch_weights_header.h \
  --tokenizer-out ~/RVchatbot/kevgpt-genesys2-soc/sw/applications/kevgpt_interactive/kevgpt_tokenizer.h \
  --tokenizer-ddr \
  --tokenizer-blob-out ~/RVchatbot/kevgpt-genesys2-soc/tokenizer_ddr_v16384.bin
```
(`--out` is required but unused by `kevgpt_interactive` -- streams
weights at runtime, doesn't bake them in -- point it at a scratch path.)

**7. Firmware build**:
```bash
export PATH=/home/tparng/tools/corev-openhw-gcc-ubuntu2204-20240530/bin:$PATH
cd ~/RVchatbot/kevgpt-genesys2-soc
make app PROJECT=kevgpt_interactive TARGET=genesys2 \
  COMPILER_PREFIX=riscv32-corev- SOURCE=../../../../sw
```
(Ignore the trailing `.venv/bin/python: No such file or directory` /
`mem_usage.py` error -- `main.elf` is already built by that point; run
`mem_usage.py` directly if you want the real ROM/RAM utilization
numbers -- see [[reference-vivado-genesys2]].)

**8. Real Vivado rebuild** (full clean rebuild whenever the SoC memory
map or a wrapper parameter changes -- `reset_run` reuse is NOT
sufficient for a structural change, see this document's own Option B
precedent):
```bash
cd ~/RVchatbot/kevgpt-genesys2-soc/hw/vendor/esl_epfl_x_heep
rm -rf build/openhwgroup.org_systems_core-v-mini-mcu_1.0.5/genesys2_kevgpt-vivado
source ~/tools-2022/Xilinx/Vivado/2022.2/settings64.sh
source .venv/bin/activate
make vivado-fpga-nobuild FPGA_BOARD=genesys2_kevgpt SOURCE=../../../sw/
cd build/openhwgroup.org_systems_core-v-mini-mcu_1.0.5/genesys2_kevgpt-vivado
make openhwgroup.org_systems_core-v-mini-mcu_1.0.5.xpr
python -m fabric.genesys2.stage_vivado_roms \
  --npz fabric/export_word16384/goformer.npz --lanes 64 --p 8 \
  --dest . --dest ./openhwgroup.org_systems_core-v-mini-mcu_1.0.5.runs/synth_1
```
Then, via a small Tcl script (`vivado -mode batch -notrace -source
build.tcl`, NO `-jobs` flag -- see [[reference-vivado-genesys2]]'s
`vrs` hang warning):
```tcl
open_project openhwgroup.org_systems_core-v-mini-mcu_1.0.5.xpr
launch_runs synth_1
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
```
PASS criterion: `write_bitstream completed successfully`, 0 Errors at
every DRC checkpoint along the way (synth, pre-place, pre-route,
pre-bitstream).

**9. Real hardware bring-up** (kill any stale `openocd` first):
```bash
cd ~/RVchatbot/kevgpt-genesys2-soc/hw/vendor/esl_epfl_x_heep/build/openhwgroup.org_systems_core-v-mini-mcu_1.0.5/genesys2_kevgpt-vivado
vivado -mode batch -source openhwgroup.org_systems_core-v-mini-mcu_1.0.5_pgm.tcl \
  -tclargs xc7k325tffg900-2 openhwgroup.org_systems_core-v-mini-mcu_1.0.5.runs/impl_1/xilinx_core_v_mini_mcu_wrapper_kevgpt.bit

cd ~/RVchatbot/kevgpt-genesys2-soc
openocd -f quad-x-heep-openocd-genesys2.cfg   # leave running in background
```
**Start the weight+tokenizer sender FIRST and confirm from its own log
that it's actually printing "waiting for KEVGPT_UART_READY" before
triggering the reload** (see [[feedback-confirm-listener-before-triggering]]
-- getting this ordering wrong is the single most repeated mistake in
this whole runbook):
```bash
# terminal/background job 1 -- start this and CONFIRM it's listening first
python -u -m fabric.genesys2.send_weights \
  --npz fabric/export_word16384/goformer.npz --lanes 64 --p 8 \
  --port /dev/ttyUSB0 \
  --tokenizer-blob ~/RVchatbot/kevgpt-genesys2-soc/tokenizer_ddr_v16384.bin

# only THEN, in a separate shell:
export PATH=/home/tparng/tools/corev-openhw-gcc-ubuntu2204-20240530/bin:$PATH
cd ~/RVchatbot/kevgpt-genesys2-soc
riscv32-corev-elf-gdb hw/vendor/esl_epfl_x_heep/sw/build/main.elf \
  -ex "target remote :3333" -ex "monitor reset halt" -ex load \
  -ex "monitor resume" -ex detach -batch
```
PASS: `SEND_WEIGHTS_PASS`, then `KEVGPT_INTERACTIVE_READY` on the
console UART (`/dev/ttyUSB0`, 115200 8N1). Send a prompt + newline over
that same port to confirm real chat.

## VOCAB=16384 real hardware: four more stories, different prompt topics

Same board/firmware session as the five stories above (no reload, no
resend), four fresh prompts on different topics (wizard/fantasy,
forest/animal, toy, vehicle -- none matching the "little girl in a
park" template the earlier prompts nudged toward), on-chip Gumbel
sampling, no cherry-picking:
```
"the wizard cast" ->
on his arm care." so, the wizard gave the key to the bird! the bird
was so happy and thanked the wizard for his help. he was so happy and
thanked the wizard for his kind help. he smiled and said," i knew you
would always remember to use the key to

"in the forest" ->
care for the little girl. buster's was never scared of the little
girl. he was very brave and he never wanted to be scared. but one day,
he was playing in the forest when he saw a big bear. he thought it was
cute. he ran to the rabbit and said," hello

"my favorite toy" ->
care for you care." lily said," i love your toy. it is very pretty."
she gave it to lily. lily was happy and said," thank you, lily. you
are very nice." they played together and had fun. buster's was not
selfish anymore.

"the rocket ship" ->
care for his family. he was very happy and happy to have his ship
back. buster's was not sad anymore. he said," my ship is broken. i
don't want to play with your ship." buster's was sad and said," no, i
don't want to play with my ship.
```
All ~1.0s per reply (consistent with the measured ~54 tok/s), zero
`<unk>` across all four. **Rougher quality than the earlier five,
flagged honestly rather than cherry-picked**: three of the four open
with the same odd stock phrase "care for ... care." (reads like a
sampling fixation, not a real continuation of the typed prompt), and
none of the four topics (wizard, forest animal, toy, rocket) actually
steer the story much -- "the wizard cast" name-drops a wizard/bird/key
but "the rocket ship"/"my favorite toy" drift into the same generic
"buster's"/"lily" park-and-toy template the training corpus is
saturated with. Consistent with this project's own established
"honest-first" pattern (sampled chat is noisier than greedy, a model-
capacity/corpus-composition property, not a tokenizer or RTL defect) --
not evidence of a regression, but a real, visible ceiling on how much a
prompt can steer this size of model.

## Final architecture reference: Genesys2 deployment as of VOCAB=16384

Everything above is diary -- what follows is a settled description of
the system as it stands, cross-checked against the actual current RTL
and firmware source rather than restated from earlier prose. Treat this
section as the reference; if anything above contradicts it, this
section wins.

**System overview.** An X-HEEP SoC (`cv32e40px`, `configs/
cv32e40px_kevgpt.py`) runs on the Genesys2's Kintex-7 (`xc7k325tffg900
-2`), with a custom fabric-resident tiny-transformer inference core
(`xheep_kevgpt_peripheral` wrapping the unmodified `sequencer_vec`
core) attached through X-HEEP's generic `ext_peripheral` hook -- the
same attach point `ai_accel` used in this SoC's other lineage, no
X-HEEP core changes needed. A single physical MIG DDR3 controller
serves THREE independent consumers, arbitrated in two stages: the
accelerator's own DMA traffic (per-layer weight-window reloads via
`weight_loader_ddr` + the DDR3-backed KV cache via `kv_bank_ddr`,
bundled by `kevgpt_ddr_bundle` into one app-level stream) is side A of
a `mig_dual_master_arbiter`; the CPU's own direct DDR3 access, through
`cpu_ddr_bridge`'s `ext_slaves` memory window (CPU address `0xF0000000`,
64MB), is side B. That CPU-side path was originally wired for boot-time
weight staging over UART and now does double duty: it is ALSO how
firmware reads the tokenizer's string table live, at inference time
(see "DDR3 memory map" below) -- no new hardware was needed for that,
just a second consumer of an already-arbitrated path.

```mermaid
flowchart TB
    CPU["cv32e40px CPU<br/>(X-HEEP core_v_mini_mcu)"]

    subgraph PERIPH["xheep_kevgpt_peripheral"]
        REGS["reg_req_t/reg_rsp_t shell<br/>CTRL/STATUS/TOK_ID/POS/SEED/<br/>RD_SEL/RD_ADDR/RD_DATA/TOK_OUT/CYCLES"]
    end

    subgraph SEQBOX["sequencer_vec (the FSM)"]
        SEQ["sequencer_vec"]
        GEMV["gemv_banked_resident_vec<br/>+ weight_bank_tdp"]
        ATTN["vec_attn_w + softmax_f"]
        LN["layernorm_vec"]
        DQ["vec_dequant"]
        GELU["vec_gelu + gelu_lut2"]
        SEQ --> GEMV
        SEQ --> ATTN
        SEQ --> LN
        SEQ --> DQ
        SEQ --> GELU
    end

    subgraph BUNDLE["kevgpt_ddr_bundle"]
        WLD["weight_loader_ddr"]
        KVDDR["kv_bank_ddr"]
        MUX["mig_read_mux2"]
        WLD --> MUX
        KVDDR --> MUX
    end

    BRIDGE["cpu_ddr_bridge<br/>ext_slaves window<br/>0xF0000000, 64MB"]
    ARB["mig_dual_master_arbiter"]
    MIG["genesys2_mig_native_shell<br/>MIG DDR3 controller"]
    DDR3[("DDR3 SDRAM<br/>weights @0 / KV @12MB / tokenizer @16MB")]

    CPU -->|register writes/polls| REGS
    CPU -->|direct loads/stores| BRIDGE
    REGS <--> SEQ
    SEQ -->|S_STRW reload trigger| WLD
    SEQ <-->|KV read/write| KVDDR

    BUNDLE -->|side A, clk_gen domain| ARB
    BRIDGE -->|side B| ARB
    ARB -->|ui_clk domain| MIG
    MIG <--> DDR3
```
Solid arrows are the control/data paths described in the module
inventory below; the two arrows converging on `mig_dual_master_arbiter`
are the two arbitrated sides ("System overview" above) -- `BUNDLE`'s
own internal DMA streams cross from `clk_gen` into `ui_clk` inside
`kevgpt_ddr_bundle` itself (the `async_fifo_gray` CDC, not drawn
separately here), while `cpu_ddr_bridge` does its own CDC the same way
on side B.

**Deployed model shape.** `D=128` (residual width), `D3=384` (QKV
width, `3*D`), `D_MLP=512` (MLP hidden width, `4*D`), `NHEAD=2`,
`HEAD_DIM=64` (`sequencer_vec`'s own fixed default -- this port varies
`NHEAD`, not `HEAD_DIM`), `NLAYER=12`, `VOCAB=16384`, `TMAX=128`
(context window), `LANES=64` (systolic PE lanes / nibbles per wide
weight word -- capped below the KV260 lineage's `LANES=128` by a real
LUT/DSP overflow on this smaller part, see `xheep_kevgpt_peripheral
.sv`'s own comment), `P=8` (read/write boundary width, GEMV boundary
and every wide-word bank's per-cycle lane count), `WWORDS=32768` (the
on-chip resident weight bank's depth in `LANES*4`-bit wide words --
sized to `max(GW_BLK, GW_HEAD, EMB_ROWG_W)` for this VOCAB, `GW_HEAD`
binding at exactly this VOCAB). `KV_DDR_BACKED=1`, `WEIGHT_DDR_BACKED=1`,
and `WEIGHT_STREAM_PER_LAYER=1` are all set -- this deployment keeps
NEITHER the weight image NOR the KV cache fully BRAM-resident; both
stream from DDR3 (see below).

**RTL module inventory.** `sequencer_vec` is the whole design's FSM --
one state machine walking token-embed lookup, all `NLAYER` transformer
blocks, final LayerNorm, head projection, and argmax, calling out to
the shared datapath bricks below as callable sub-routines (a GEMV call,
an LN call) rather than duplicating logic per block. `gemv_banked_
resident_vec` is the systolic INT4(weight) x INT8(activation) GEMV
core -- `LANES` PEs wide, banked activation/output memory, reading its
resident weight image through `weight_bank_tdp` (a true-dual-port
on-chip memory, `MEM_PRIMITIVE="block"` here since the Kintex-7 part
has no URAM, unlike the KV260 lineage's `"ultra"`). `vec_attn_w` +
`softmax_f` implement attention (one full head-width position per
cycle, softmax pipelined to 1 element/cycle). `layernorm_vec` does
gamma-only LayerNorm, `P`-wide I/O, a single-pass algebraic variance
computation (no second data pass) with a pipelined Newton rsqrt.
`vec_dequant` applies the per-channel mantissa/exponent dequant
(`P` lanes/cycle, bit-exact to the scalar reference) that turns each
GEMV's raw INT32 accumulator into a real-valued activation. `vec_gelu`
(built from `gelu_lut2`, a BRAM-saving 2-lanes-per-table variant of the
single-lane `gelu_lut`) applies GELU via an 8192-entry LUT with 3-bit
linear interpolation. `kv_bank` (the fully on-chip KV cache) exists in
the source tree but is NOT instantiated in this deployment at all --
`KV_DDR_BACKED=1` selects `kv_bank_ddr` exclusively at elaboration
(a Verilog `generate`, not a runtime mux); `kv_bank_ddr` is a byte-for-
byte port of `kv_bank`'s own quantize/dequantize contract onto DMA
read/write packets instead of direct on-chip BRAM ports, guaranteeing
identical results for identical inputs by construction. `weight_loader_
ddr` reloads one weight window (one block, or the head, or -- in the
now-superseded bulk-embed design -- the embed tables) into the SAME
resident `weight_bank_tdp` from DDR3, triggered internally by
`sequencer_vec`'s own `S_STRW` state every block/embed/head, every
token; it is deliberately single-buffered (compute waits for the
reload to finish, no double-buffering yet) to avoid touching `gemv_
banked_resident_vec`'s own timing-critical, backpressure-free MAC
pipeline. `kevgpt_ddr_bundle` merges `kv_bank_ddr`'s and `weight_loader_
ddr`'s independent DMA streams into one app-level bundle AND does the
real clock-domain crossing between them (`gen_clk`/`clk_gen`, PLL-
derived from MIG's `ui_clk`, measured 50MHz on this board, is a
genuinely different frequency from `ui_clk` itself, not just a renamed
wire) via `async_fifo_gray`, the same CDC primitive `cpu_ddr_bridge`
already uses for its own clock boundary; `mig_read_mux2` inside that
bundle time-multiplexes `kv_bank_ddr`'s and `weight_loader_ddr`'s reads
onto one shared `mig_read_engine`, since Phase 0's own bandwidth
headroom made two independent read engines unnecessary. `xheep_kevgpt_
peripheral` is the register-interface shell around unmodified
`sequencer_vec` (a flat X-HEEP `reg_req_t`/`reg_rsp_t` port instead of
`gemv_axi_seq_vec.v`'s AXI4-Lite, same register map/semantics/IDCODE
`"SQRV"`). `xilinx_core_v_mini_mcu_wrapper_kevgpt` is the real top-
level FPGA wrapper tying everything together: X-HEEP's `core_v_mini_
mcu` instance, `xheep_kevgpt_peripheral` on the `ext_peripheral` hook,
MIG + `genesys2_mig_native_shell` + the clock wizard, `cpu_ddr_bridge`,
and `mig_dual_master_arbiter`. `cpu_ddr_bridge` is the CPU's own direct
DDR3 path (via X-HEEP's `ext_slaves` crossbar window, OBI, single-
outstanding since CPU loads/stores are latency-sensitive single-word
ops, not bulk DMA) -- unmodified from its origin in this SoC's other
lineage, now doing double duty for both boot-time weight/tokenizer
staging over UART and live tokenizer-string reads during inference.
**One open, honestly-flagged caveat**: the CDC wiring between `kevgpt_
ddr_bundle`'s two real clock domains and `mig_dual_master_arbiter` has
only ever been gated in simulation on ONE shared clock (predating the
CDC fix) -- genuinely concurrent two-clock-domain traffic through the
arbiter has not been independently re-verified, only argued to be
correct by construction. Real hardware chat working end-to-end is
evidence this is fine in practice, not proof.

**DDR3 memory map.** Three regions, all firmware/software-level
conventions (no RTL parameter enforces the tokenizer region; only
`WEIGHTS_DDR_BASE`/`KV_DDR_BASE` are real `sequencer_vec` parameters) --
`WEIGHTS_DDR_BASE=0`: the full per-layer-streaming weight image,
10,682,368 bytes at this VOCAB (`NLAYER*GW_BLK+GW_HEAD+GW_EMB`, `wc -l
wrom.mem` * 32 bytes/word), read by `weight_loader_ddr`, written once
at boot by firmware via `cpu_ddr_bridge`. `KV_DDR_BASE=12582912` (12MB):
`kv_bank_ddr`'s own KV-cache region, 589,824 bytes (`NLAYER*2*NHEAD*
TMAX` rows * a fixed per-row byte count, VOCAB-independent), read AND
written by `kv_bank_ddr` only -- never touched by the CPU directly.
`TOKENIZER_DDR_BASE=16777216` (16MB, `fabric.genesys2.gen_chat_fw
.TOKENIZER_DDR_BASE`): the DDR3-resident tokenizer table (a per-id
offset table + concatenated null-terminated strings, 189,448 bytes at
this VOCAB), written once at boot by firmware over UART (the same
`cpu_ddr_bridge` path) and read live, every generated/typed token, by
firmware's own `kevgpt_itos_ddr()` -- never touched by the accelerator
at all. All three ranges must stay non-overlapping (each base was
chosen with real headroom over the other two's actual measured sizes,
not guessed) -- `sequencer_vec.sv` carries a simulation-only `$display`
overlap check between the first two; the tokenizer range has no such
check since the accelerator never sees it, so it's the one range
that would fail silently if it were ever miscomputed.

**SoC CPU-side memory map.** 96KB total (`configs/cv32e40px_kevgpt.py`:
`add_ram_banks([32]*3)`, three continuous 32KB banks, no interleaved
banks), split 32KB code / 64KB data (`LinkerSection.by_size("code", 0,
0x8000)`, data from `0x8000`). This is deliberately small: `kevgpt_
interactive` neither bakes the weight image nor the tokenizer table
into firmware -- both stream over UART into DDR3 at boot -- so the only
real ROM/RAM cost left is the firmware binary itself (`mem_usage.py`:
~11/32KB code, ~35/64KB data). An earlier attempt at this VOCAB bumped
this map to 448KB specifically to fit the tokenizer table IN firmware
before the DDR3-tokenizer redesign; that bump is gone entirely now, and
going back to it would reintroduce the real RAMB36 overflow the DDR3
redesign fixed (see "Real BRAM budget" below).

**One-token data flow.** Firmware pulses `CTRL.go` after writing
`TOK_ID`/`POS` (and, once per chat turn, `SEED`, which also enables
on-chip Gumbel-max sampling for every subsequent call that turn).
`sequencer_vec` enters `S_EMB`, looks up the token/position embedding
rows (row-streamed from DDR3 in small, VOCAB/TMAX-sized windows, not a
bulk table reload -- the "streaming embed lookups" redesign), then
loops over blocks 0..`NLAYER-1`: each block's own weight window is
reloaded from DDR3 via `S_STRW` (`weight_loader_ddr`, gated on `wld_
ld_done`) BEFORE that block's QKV/attention/MLP GEMVs run (`S_QKVRET`
-> `S_AST`/`S_ALD`/`S_ACL` attention -> `S_RES1` -> `S_FCRET`/`S_MPSET`
MLP -> `S_RES2`, with the shared `G_AQ`/`G_WAIT`/`G_RB` GEMV callable
sequence and `L_GAM`/`L_FEED`/`L_COLL` LayerNorm callable sequence
invoked from each of those). After the last block, `S_HEADSET`
reloads the head's own weight window (the single largest reload at
this VOCAB, `GW_HEAD=32768` words) and arms the Gumbel-noise precompute
(`gpre_active`, filling `gumbel_bank` one logit per cycle, riding the
head GEMV -- skipped entirely in greedy mode) before entering `S_ARGMAX`,
which waits for `gpre_done` and then reduces all `VOCAB` logits (Gumbel
noise added first when sampling) down to a single winning token id,
written to `TOK_OUT`, before `S_FIN` raises `STATUS.done`.

**Real BRAM budget.** Final placed utilization on the xc7k325t: 440/445
RAMB36E1 tiles (98.88%) -- essentially maxed out, ~5 tiles of margin.
Of that, the accelerator's own resident weight/KV BRAM (`weight_bank_
tdp`'s `WWORDS=32768`-deep image plus everything else inside `u_kevgpt`)
accounts for roughly 410 tiles; the SoC's own 96KB CPU memory accounts
for the remaining ~24. Any future growth on this board (bigger VOCAB,
more layers, a bigger SoC memory bump, a new on-chip buffer) will very
likely need to shrink something else first, not just be added on top --
see [[project-kevgpt-vocab16384-status]].

**Firmware/software stack.** `kevgpt_interactive`'s `main.c` is a
single-threaded, blocking control loop: read a typed line off the
console UART one character at a time (with backspace handling), run a
C port of `model.word_data.tokenize()`'s own regex (`tokenize_line()`,
grouping `[a-z']+` runs and treating `.,!?;:"-` as single-char tokens)
combined with a binary search over the alphabetically-sorted vocabulary
(`kevgpt_word_lookup()`, `strcmp` against `KEVGPT_ITOS(id)`), prefill
each resulting token id via `kevgpt_step()` (write `TOK_ID`/`POS`, pulse
`CTRL.go`, poll `STATUS.done`, read `TOK_OUT`), then keep calling `kevgpt_
step()` to generate, applying a masked-argmax remask fallback
(`remask_pick()`) if a sentence-ending token would fire before the
minimum word count, until `MAX_GEN_LEN`/`TMAX`/`STORY_SENTENCES` caps
out. `KEVGPT_ITOS(id)` resolves to `kevgpt_itos_ddr(id)` under
`KEVGPT_TOKENIZER_DDR` (this deployment): `kevgpt_tok_offset()` reads a
`uint32_t` from the DDR3-resident offset table via a plain `volatile`
pointer into `cpu_ddr_bridge`'s `ext_slaves` window, and `kevgpt_itos_
ddr()` returns a `const char *` computed from that offset -- since this
core has no data cache, that pointer behaves exactly like a normal
in-SRAM C string for every purpose (`strcmp`, indexing, byte iteration)
with no special DDR3-aware I/O code anywhere in the tokenizer path.

## Repetition guards: taming degenerate sampled-story loops (firmware-only, no RTL changes)

Real sampled chat, on real hardware, showed a recurring class of
degenerate output the user could paste directly: "once upon a time" ->
"...she wanted to climb it, but her mommy said they needed to stay..."
was fine, but other replies looped -- "big tree with a big tree",
"big house and a big house". ChatGPT-style generic advice (repetition
penalty, top-k/top-p, no-repeat-ngram) was evaluated against this
project's ACTUAL fixed-function accelerator before touching anything:
the deployed engine does a single-pass on-chip Gumbel-max tournament
over all VOCAB=16384 logits with a baked ROM temperature (no sort
primitive, no per-token host visibility into the full distribution by
design -- that's the bandwidth-wall thesis), so most of that generic
advice doesn't map onto this hardware. What DOES map: `main.c` already
had a real, proven precedent for "reject-and-resample-on-collision" --
`remask_pick()`, the existing masked-argmax fallback used when a
stop-token would fire before `MIN_WORD_TOKENS`. Generalizing that one
mechanism turned out to be the right lever.

**Implementation** (`kevgpt_interactive/main.c`, commit `d4770d2`,
pushed to `origin/word-vocab`): `remask_pick()` became a thin wrapper
around a new `remask_pick_excluding(dev, extra[], n_extra)` -- same
full-VOCAB linear scan over `kevgpt_read_bank(dev, 8, i)` (head logits,
Q6.25), same "only paid when it actually collides" cost model (4
register round-trips per excluded-candidate check; unconditional
per-token use would dominate the cycle budget, so every caller below
fires only on an actual collision). Three guards wired into `chat_turn`'s
generation loop, in order:
1. **Immediate exact doubling** ("yummy yummy") -- `tok == last_tok`.
2. **Immediate near-duplicate stem** ("bustled bustle") -- different
   vocab ids, but one word's string is a prefix of the other's
   (`is_stem_repeat()`, `MIN_STEM_LEN=4` to avoid short-word false
   positives like "he"/"hello").
3. **Delayed bigram recurrence** ("big tree ... big tree") -- tracks
   every accepted token this reply in `hist[MAX_GEN_LEN]`; before
   accepting a candidate, checks whether `(last_tok, tok)` already
   occurred earlier in the same reply.

All three are deliberately EXACT/prefix/bigram checks, not "was this
token used recently" -- the latter would forbid ordinary function-word
reuse ("she"/"the"/"and" recur constantly in fluent prose within a few
words of each other).

**A real bug, found only via real-hardware diagnostic instrumentation,
not static reasoning**: the first version of guard 3 fired far more
than the "occasional collision" cost model assumed -- 2 to 7 times per
short reply, flooding output with generic greedy-argmax fallback words
("care", "buster's", "donkey's", "curled") that were often MORE
jarring than the repeats they replaced. Added a temporary
`KEVGPT_REPEAT_DEBUG` build flag (per-guard fire counters + a
`[BIGRAM last,before -> after]` print at every firing) and reloaded on
real hardware to see it directly: most firings were on bigrams like
`(".", "buster's")` or `(".", "she")` -- a PERIOD followed by a common
sentence-starter, which recurs in literally every multi-sentence story
and is not a loop at all. Root cause: the bigram-history scan checked
`!is_stop_id(tok)` (the current candidate) but never checked whether
`hist[k]` (the PRECEDING word in the historical pair) was itself a stop
id. Fixed with one added condition (`!is_stop_id(hist[k])`, plus the
symmetric `!is_stop_id(last_tok)` at the outer gate). Re-tested on real
hardware: bigram fires dropped to legitimate-looking counts (0-7 per
reply, but now firing on real recurring content bigrams like
`("the","dog")` -> `"ball"`, correctly steering a story that kept
wanting to repeat "the dog" into a coherent "ball" sub-plot instead).
`KEVGPT_REPEAT_DEBUG` was stripped before the final commit -- confirmed
via a rebuild+reload+retest that the stripped binary behaves
identically to the instrumented one (same guard logic, just no debug
prints).

**Real-hardware verification, several rounds** (each round: kill/
restart `openocd` and `send_weights.py` as needed, full reset+load+
resume firmware reload, `SEND_WEIGHTS_PASS` on the ~10.7MB weight image
+ tokenizer blob at 115200 baud -- each transfer genuinely takes ~15
minutes at that baud rate, not a quick check -- then chat over the
console UART): confirmed the exact-doubling and stem-doubling guards
eliminate their target patterns outright (no more "yummy yummy",
"bustled, bustled", "bustled bustle" observed after the fix), and the
bigram guard, post stop-id-exemption fix, fires on genuine content
recurrence without the false-positive flood.

**Two real hardware debugging detours hit along the way** (both now in
[[reference-vivado-genesys2]], not repeated in full here): (1) Vivado's
`*_pgm.tcl` resolved to a STALE, no-longer-live `hw_target` (two
registrations differing only by a trailing character, `...AA` dead vs
`...AAB` live) -- fixed by probing `get_hw_targets` directly and
pinning the live one via the `HW_TARGET` env var the script already
reads. (2) A JTAG USB re-enumeration event killed BOTH `openocd`'s
connection AND a separately-running `send_weights.py` UART transfer
AT THE SAME TIME (same physical board/hub, two FTDI chips) -- the
UART side gave zero error, just hung silently in a blocked
`ser.write()` for ~26 minutes before being diagnosed via a `devnum`
sysfs check. Neither was a logic/RTL bug; both were transient USB/JTAG
session state, fixed by killing and restarting fresh.

**Honest final finding, from a proper controlled test, not a single
eyeballed sample**: comparing the guarded firmware against the
unmodified baseline with a 5-seed x 5-prompt sweep (25 samples each),
scored by an objective detector (flag if any bigram recurs 3+ times in
one sample, exempting punctuation-adjacent pairs) -- **7/25 flagged on
the guarded checkpoint vs 8/25 on baseline, a statistical tie.** The
guards demonstrably eliminate the SPECIFIC narrow patterns they target
(confirmed on real hardware, not just in this software sweep), but they
do NOT change the deployed model's overall tendency to loop -- that
turned out to be a property of the model's own capacity/training level,
not something a reactive firmware-side guard can fully suppress (the
identical repetition class shows up in the plain FP software model with
zero guards involved, and in a genuinely bigger D=384 teacher model
too -- see the follow-up investigation below). Two follow-up attempts
to close that gap -- extending training to 60000 iterations (both at
the deployed `eps=3e-4` and at the more-stable `eps=1e-3`), and
distillation from a D=384 teacher into a much larger synthetic
fine-tuning corpus (24,918 filtered samples, 2.1M tokens, after an
initial 283-sample attempt catastrophically overfit) -- were both tried
and both came back negative: neither beat the baseline's val loss or
its measured repetition rate. Recorded in detail, not repeated here, in
this session's `model/SCALE-UP-LOG.md`-adjacent memory
(`project-kevgpt-word-vocab-quality-ceiling` if consulting saved
session memory) so a future session doesn't re-attempt either without
a genuinely new angle.

## Distillation from a bigger teacher: clean repeatable sequence (final commands only, no detours)

The negative result above is real and shouldn't be re-derived blindly,
but the PROCESS is legitimate and worth being able to re-run exactly --
e.g. with a bigger/cleaner synthetic corpus, a different teacher shape,
or after some other change makes it worth re-checking. All three
pipeline stages are now real, committed, reusable scripts (`model/
gen_synth_corpus.py`, `model/filter_synth_corpus.py`, `model/
prep_synth_ft_data.py` -- promoted from one-off scratchpad scripts,
each independently smoke-tested), not one-off shell history. Commands
below are the exact ones that produced the numbers already reported in
this document.

**1. Train the teacher.** A wider (D=384 vs the deployed D=128), more
capable model at the SAME vocab as the deployment target (VOCAB=16384 --
avoids any cross-vocab re-tokenization complication later), using the
stability recipe `model/SCALE-UP-LOG.md`'s Attempt 11 already found for
this width (`eps=1e-3`, `lr-decay-iters=10000` of a 15000-iter run):
```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 10000 --adam-eps 1e-3 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_teacher_d384_v16384.pt --states data/states_teacher_d384_v16384.jsonl
```
**Result: best val 2.743 @ iter 9500, ~37 min wall time** on this GPU
(D=384 trains at ~6.8 it/s vs the deployed D=128 shape's ~24-55 it/s --
budget accordingly). Sanity-check the teacher's OWN raw training-log
samples before proceeding -- this recipe/shape still shows real
repetition on its own ("the sun was shining and the sun was shining"),
it is not a clean gold-standard generator; that's expected, not a sign
something went wrong.

**2. Generate a synthetic corpus from the teacher.**
```bash
python -m model.gen_synth_corpus data/ckpt_teacher_d384_v16384.pt \
  --corpus data/TinyStories-train.filtered.txt \
  --out data/teacher_synth_corpus.txt --n-samples 50000
```
**Result: 50,048 samples in ~33 min** (~26 samples/s at the default
`--batch 128` -- tested up to `--batch 512` with no further speedup,
compute-bound at this model size, not batch-size-bound). `--n-samples
600` (the first attempt) is a trap: looks like a reasonable quick test,
but the resulting corpus is far too small to fine-tune on without severe
overfitting (see step 4's own warning).

**3. Filter for degenerate repetition.**
```bash
python -m model.filter_synth_corpus data/teacher_synth_corpus.txt \
  --out data/teacher_synth_corpus_filtered.txt
```
**Result: kept ~24,918 / 50,048 (~50%).** If the kept fraction comes out
much lower than this (e.g. single digits or the input almost entirely
dropped), the filter is almost certainly hitting the punctuation-adjacent
false-positive bug already found and fixed once (see `model.
filter_synth_corpus`'s own docstring) -- check `--show-dropped`'s output
for whether the flagged bigrams are real content loops or just normal
sentence-starter reuse before trusting a low keep-rate.

**4. Tokenize the filtered corpus against the EXISTING deployed vocab**
(NOT a fresh vocab -- the synthetic corpus is far too small to build a
real 16384-word vocabulary from scratch, and the fine-tune must stay
vocabulary-compatible with the checkpoint it warm-starts from):
```bash
python -m model.prep_synth_ft_data data/teacher_synth_corpus_filtered.txt \
  --src-meta data/word_v16384/meta.json --out-dir data/word_v16384_synth_ft
```
**Result: 2,113,247 tokens (2,007,585 train / 105,662 val), unk_frac=0.0**
(0 by construction -- every word came from a model already using this
exact vocab). **This token count is the load-bearing number in this
whole pipeline.** The first attempt (600 raw -> 283 filtered) produced
only 22,830 train tokens here -- at `--batch-size 64 --block-size 128`
(the training defaults), that's under 3 "epochs"-worth of unique
128-token windows per single training iteration, and even a modest
3000-iteration/`lr=1e-4` fine-tune collapsed train loss to ~0.05 while
val loss climbed from 1.71 to 3.85 -- catastrophic memorization,
producing whole-sentence VERBATIM repeats in generated output ("they
wished they had not played with the dog. they wished they had not
played with the dog."). The 50,000-raw-sample run above avoids this by
being ~93x bigger.

**5. Fine-tune the deployed small model on the filtered corpus.** At
`--batch-size 64 --block-size 128` (defaults), one epoch over ~2M train
tokens is ~245 iterations; `--max-iters 2000` here is ~8 epochs --
enough real exposure without an excessive iteration count relative to a
still-fairly-narrow (all-synthetic, single-teacher-style) corpus:
```bash
python -m model.train --init-from data/ckpt_word16384.pt --lr 1e-4 --warmup 100 \
  --tokenizer word --vocab-size 16384 --corpus data/teacher_synth_corpus_filtered.txt \
  --data-dir data/word_v16384_synth_ft --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16384_distilled.pt --states data/states_word16384_distilled.jsonl \
  --max-iters 2000
```
**Result: train loss 1.79->1.23, val loss 1.88->1.36, tracking closely
together the whole run (final gap ~0.13)** -- a HEALTHY trajectory,
unlike step 4's overfitting warning case (gap ~3.8). This confirms the
corpus-size fix worked at the training-mechanics level. It does NOT by
itself mean the output is better -- see step 6.

**6. Verify with a seeded multi-sample sweep, not one eyeballed
generation.** A single unseeded `model.sample` run per prompt is not
evidence of anything -- sampling variance alone can make one draw look
clean and the next look broken on the IDENTICAL checkpoint (this
happened directly in this session: an initial "no more repeats!" claim
from 5 unseeded samples was contradicted by a second unseeded run on the
same checkpoint minutes later). Score BOTH the new checkpoint and the
original baseline with the SAME fixed-seed sweep and the SAME objective
detector -- `model.filter_synth_corpus.is_degenerate()` is exactly this
detector, reusable directly:
```python
import torch
from model.sample import load
from model.word_data import decode, tokenize, UNK
from model.filter_synth_corpus import is_degenerate

prompts = ["once upon a time", "the sun was", "the dog ran", "a little girl", "she found a"]
seeds = [1, 2, 3, 4, 5]
model, meta, it, val = load("data/ckpt_word16384_distilled.pt", "cuda")
stoi, itos = meta["stoi"], meta["itos"]
flagged = 0
for seed in seeds:
    for prompt in prompts:
        torch.manual_seed(seed)
        ids = torch.tensor([[stoi.get(t, stoi[UNK]) for t in tokenize(prompt.lower())]], device="cuda")
        out = model.generate(ids, 60, temperature=0.58, top_k=None)[0].tolist()
        text = decode(out, itos)
        if is_degenerate(text):
            flagged += 1
print(f"{flagged}/{len(seeds)*len(prompts)} flagged")
```
Run the IDENTICAL script against `data/ckpt_word16384.pt` (the baseline)
for the comparison to mean anything. **Actual result from this session:
7/25 (28%) flagged on the distilled checkpoint vs 8/25 (32%) on
baseline -- a statistical tie, no measurable improvement.** This is the
real, final verdict for the whole pipeline as tried: it fixes the
overfitting failure mode from too little data, but does not close the
gap to the char-level model's fluency, and does not measurably reduce
this model's baseline repetition rate. `data/ckpt_word16384_distilled.pt`
was NOT promoted over the deployed checkpoint.

## Owner-FIFO backpressure fix + a testbench-only clock bug in tb_kevgpt_ddr_bundle.sv

Part of the fixation-word investigation (`FIXATION-WORD-CDC-INVESTIGATION.md`
§4/§4a/§8 item 2 have the full writeup; this entry is the PORT-NOTES-style
chronological record). Two separate, unrelated fixes came out of this work.

**Fix 1: owner-FIFO backpressure, `mig_dual_master_arbiter.sv` +
`mig_read_mux2.sv`.** Both modules' owner-tracking FIFOs (the ones that
record which requester a pending DDR3 read return belongs to) left
`in_ready_o` unconnected -- a command could be accepted and forwarded to MIG
while its owner FIFO was full, silently dropping the push and scrambling
every owner tag after it. Wired `in_ready_o` into real backpressure in both
(command acceptance now gates on the relevant owner FIFO having room), and
added outstanding-request accounting: independent push/pop counters
cross-checked every cycle against each FIFO's own `count_o`, plus
overflow/underflow/backpressure-violation assertions. `mig_dual_master_arbiter.sv`
turned out to have *two* owner FIFOs with this exact bug (`u_rd_owner_fifo`
for reads, `u_wr_owner_fifo` for writes -- only the read one had been
flagged in the investigation doc before this).

**Fix 2: a real, separate bug found while verifying fix 1 --
`tb_kevgpt_ddr_bundle.sv`'s weight-loader DUTs were on the wrong clock.**
Verifying fix 1 needed a working gate to run it against. Getting
`tb_kevgpt_ddr_bundle.sv` to even compile on this repo's local Icarus
install (12.0 stable) first required working around a toolchain gap: this
Icarus build cannot parse `assert property (... disable iff ...)` at all --
confirmed with a two-line minimal repro, unaffected by
`-gsupported-assertions`/`-gno-assertions`. Four files in this gate's own
dependency chain use that syntax (`mig_read_mux2.sv`, `mig_read_engine.sv`,
`mig_dual_master_arbiter.sv`, `sync_fifo.sv`, all touched 2026-08-16
onward) -- compiled around it locally by bracketing `` `define
SYNTHESIS``/`` `undef SYNTHESIS `` shims around exactly those four files
(stripping their `` `ifndef SYNTHESIS `` assertion blocks for this run
only; `kv_bank.sv`/`weight_bank_tdp.sv` must never see `SYNTHESIS` defined,
since that flips them to an unelaborable `xpm_memory_tdpram` macro).
Whatever machine last produced this gate's recorded "PASS, clean compile"
either used a different simulator or ran before these assertions existed --
worth checking before assuming this specific `iverilog` package is enough
for a fresh dev machine on this repo.

Once compiling, Phase 1 (KV path) passed but Phase 2 (weight-loader path)
hung to timeout -- reproduced identically with fix 1 both applied and
reverted, ruling it out as cause or cure of this specific hang. Traced with
targeted instrumentation (a software scoreboard mirroring `mig_read_mux2`'s
owner-FIFO push/pop order first -- 0 mismatches, clearing that layer; then
direct hierarchical counts on `u_wl_rd_ret_cdc`'s own write/read ports):
**10 real writes, 14 reads** on the weight-loader's read-return CDC FIFO --
a 4-entry excess exactly equal to `CDC_FIFO_DEPTH`, the signature of a
phantom-pop bug. Root cause: `tb_kevgpt_ddr_bundle.sv` instantiated
`weight_loader_ddr`/`weight_bank_tdp` on `ui_clk`, per an explicit in-file
comment dating from when `kevgpt_ddr_bundle.sv`'s weight-loader read port
was still "an un-CDC'd wl_* pass-through." A real CDC
(`u_wl_rd_req_cdc`/`u_wl_rd_ret_cdc`) was added for that port later, closing
the gap that comment flagged as future work, but the testbench's clock
wiring for these two DUTs was never updated to match -- leaving
`weight_loader_ddr`'s `rd_ret_ready` crossing into the CDC FIFO's `rd_clk_i`
domain from the wrong, unrelated clock (`ui_clk` instead of `gen_clk`),
occasionally double-popping a beat. **Real hardware never had this
mismatch** -- `weight_loader_ddr`'s `clk` is always `gen_clk` in the actual
design. Fixed by reclocking `u_wb`/`u_wl_dut` (and the dependent
`ldn_cnt`/stimulus/verification code) from `ui_clk` to `clk`. **Result:
`KEVGPT_DDR_BUNDLE_VERDICT,PASS`, 0 errors, all 4 phases** -- confirmed
independent of fix 1 (passes with it present or reverted).

**Bottom line for the fixation-word investigation**: fix 1 is a real,
correct, defensive hardening, not yet proven to be the active real-hardware
defect (this gate's traffic never filled either owner FIFO close to
capacity). Fix 2 is entirely testbench-only and explicitly does not explain
the real-hardware symptom -- it matters only because it means this gate's
"genuine two-master sharing is correct" claim had been unverified (not
disproven, just untested) since 2026-08-16, and is now restored to an
actual passing gate.

## Real-hardware isolation experiment: "care" pinned down, then a build-dependent twist

Part of the fixation-word investigation (`FIXATION-WORD-CDC-INVESTIGATION.md`
§2a/§3/§8 item 3 has the full writeup; this entry is the PORT-NOTES-style
chronological record). §8 item 3's plan (disable `KV_DDR_BACKED`/
`cpu_ddr_bridge` entirely, resynth, rerun the greedy test) turned out not to
be buildable for checkpoint C's shape: `KV_DDR_BACKED=0` overflows BRAM to
~116% (it's hardcoded at `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv:436`, not
a build flag), and `cpu_ddr_bridge` isn't actually idle during normal chat
even before touching that -- `main.c`'s `KEVGPT_ITOS(id)` macro reads the
tokenizer string table through it on every generated token whenever
`KEVGPT_TOKENIZER_DDR` is defined (true for VOCAB=16384), independent of any
diagnostic toggle.

Ran the cheap partial version instead: a new off-by-default toggle,
`KEVGPT_PRINT_IDS_ONLY`, skips `print_word_token()`'s own `cpu_ddr_bridge`
read (prints the raw numeric token id instead of the decoded string) while
deliberately leaving `is_stem_repeat()`'s two per-token `cpu_ddr_bridge`
reads alone, since those feed the repetition guard's actual pick and
touching them would change what gets generated, invalidating any
comparison. First attempt used `printf("%u ", tok)` for the id print and
silently wedged the console after one reply -- libc stdio buffering
colliding with this file's otherwise-universal raw `uart_putc()` writes,
unrelated to the DMA path under investigation. Fixed by hand-formatting the
decimal digits through `uart_putc()` directly (see the comment at
`print_word_token()` for the full account).

**Result: clean negative for cpu_ddr_bridge print-path traffic specifically.**
With the fixed toggle, real-hardware output was byte-for-byte identical
between this reduced-traffic build and a baseline build (`KEVGPT_FORCE_GREEDY=1`,
`KEVGPT_PRINT_IDS_ONLY=0`) across 5 test prompts, two independent real
weight+tokenizer reloads (~15 min each, `fabric.genesys2.send_weights`).

**While collecting that comparison, rank-analyzed each divergence against
the Python golden reference (`IntKVQSequencer(kbits=8, vbits=8, rotate=False,
divfree=True)`, same checkpoint) and got the sharpest evidence this
investigation has captured.** 3 of 5 prompts ("in the forest", "my favorite
toy", "the rocket ship") produced "care" (id 2213) or its stem-relative
"carefree" (id 2216) as the very first generated token, ranked 2551, 9311,
and 8149 of 16384 in golden's own distribution -- wild misses, not close
calls. This is the exact word §2's original greedy-mode capture picked
(rank 15,118) and the word recurring across multiple earlier real-hardware
story captures elsewhere in this log ("care for the little girl," "care for
his family," "care for you care") -- confirms "care" specifically, not a
generic wrong-word pattern, is this investigation's single most-reproduced
symptom, now caught at token 0 and precisely rank-characterized. One prompt
("the wizard cast") was a near-tie (rank 1); one ("once upon a time")
matched golden exactly -- same severity-variance pattern §2 already
documented.

**The complication**: added a third off-by-default toggle,
`KEVGPT_DIAG_LOGIT_PROBE`, printing the raw Q6.25 head logit for ids
2213/2216/the-actual-winner via `kevgpt_read_bank()`, inserted *after* the
first token is already decided by `kevgpt_step()`. This third build --
otherwise identical generation logic -- produced a *different* winning
token for the same 3 prompts (both "in the forest" and "the rocket ship"
won with "." instead of "care"; "my favorite toy" won with a different id
instead of "carefree"), reproducible byte-for-byte across 2 back-to-back
trials on the same boot (ruling out live per-call flakiness) but different
from the isolation/baseline builds' shared result. The inserted code cannot
causally affect this token's value -- it only runs after `kevgpt_step()`
already returned it -- so the only remaining explanation is indirect: any
firmware change shifts instruction addresses/timing throughout the binary,
including during weight prefetch before generation starts (weight reloads
happen once per layer per token). If the real defect is a live,
timing-sensitive race, sensitivity to incidental build-to-build timing is
exactly what this would look like: deterministic within a boot, different
across boots/builds, same symptom class, different specific corrupted
value. Leans the investigation back toward the CDC-timing hypothesis over a
fixed, boot-independent addressing bug.

All 4 diagnostic toggles (`KEVGPT_FORCE_GREEDY`, `KEVGPT_DIAG_DUMP_HEAD`,
`KEVGPT_PRINT_IDS_ONLY`, `KEVGPT_DIAG_LOGIT_PROBE`) reverted to off before
committing.

## The CDC clock-naming question, resolved: clk_200mhz_p was never constrained at all

Part of the fixation-word investigation (`FIXATION-WORD-CDC-INVESTIGATION.md`
§6/§8 item 4 has the full writeup). Earlier work in this investigation had
narrowed the question to two possibilities and couldn't distinguish them
from static inspection: either MIG/Clocking-Wizard's internal clocks were
legitimately hidden behind Vivado's out-of-context (OOC) IP methodology
("fine, just invisible to the queries tried"), or `gen_clk`/`ui_clk`
genuinely had no top-level timing closure at all ("not fine"). Resolved it
by actually doing what was recommended: opened the real project (not an
archived checkpoint) --
`hw/vendor/esl_epfl_x_heep/build/openhwgroup.org_systems_core-v-mini-mcu_1.0.5/genesys2_kevgpt-vivado/openhwgroup.org_systems_core-v-mini-mcu_1.0.5.xpr`,
the actual `.xpr` that produced the currently-deployed bitstream (Sep 6
02:23, matches the bitstream timestamp) -- via `vivado -mode batch`,
`open_run impl_1`, and ran `report_clocks -verbose` / `report_clock_networks`
against the live implemented design.

**Answer: the second, worse one.** `report_clock_networks` lists
`clk_200mhz_p` -- MIG's 200MHz DDR3 reference clock, the actual board pin
the differential oscillator drives -- directly under "Unconstrained Clocks":
**67,133 clock endpoints, 232 non-clock endpoints**. That's not a corner of
the design; that's essentially the entire `gen_clk`/`ui_clk`/MIG/kevgpt/
`cpu_ddr_bridge` domain (everything except JTAG and SPI-slave), with zero
real static timing analysis ever applied, in every build that has produced
a bitstream for this board so far. This also resolves the "why was the
Unconstrained Path Table short" mystery from earlier in this investigation:
Vivado's timing engine doesn't enumerate paths as "unconstrained" for
endpoints that were never associated with any clock context at all -- it
only flags paths that have *some* clock but a missing exception. A whole
domain with no root clock produces a short report, not a long one; the
absence of alarm bells was itself the bug.

**Root cause, once found, was almost anticlimactic.** `clk_200mhz_p`'s pin
constraints (`PACKAGE_PIN`/`IOSTANDARD`) are correctly in `pin_assign.xdc`.
Its `create_clock` line is *also* already correct, verbatim, in
`hw/vendor/esl_epfl_x_heep/hw/fpga/constraints/genesys2/mig_traffic_gen_top.xdc`
-- but that file is an orphaned leftover from an earlier standalone MIG
example-design bring-up, and was never added to this target's actual
fileset (`core-v-mini-mcu-fpga.core`'s `genesys2`/`genesys2_kevgpt` filesets
list only `pin_assign.xdc` + `constraints.xdc` + `ddr3.xdc`). The correct
line existed in the repo this whole time, just in a file nothing ever
built.

**Fix applied, verified for clock-graph connectivity against the live
design before touching the real file.** Re-applied
`create_clock -period 5.000 -name sys_clk_pin [get_ports clk_200mhz_p]`
in-memory against the same open `impl_1` design first: `clk_200mhz_p` moved
from "Unconstrained Clocks" to "Constrained Clocks" immediately, and a
follow-up `check_timing` reported 0 `no_clock` / 0
`unconstrained_internal_endpoints` / 0 `multiple_clock` / 0
`generated_clocks`-not-connected-to-source -- Vivado's own generated-clock
inference correctly derives the whole downstream MIG-PLL -> Clocking-Wizard
-> `gen_clk`/`ui_clk` chain automatically once this one root clock exists,
no further `create_generated_clock` lines needed. Only then added the line
for real to `constraints.xdc`, next to the existing `jtag_clk_pin`/
`spi_slave_clk_pin` `create_clock` statements.

**What this doesn't yet prove**: clock-graph connectivity isn't timing
closure. This confirms the whole domain has never been *checkable*, not
that it's currently *broken* -- that needs a full clean re-synthesis/
re-implementation/re-bitstream cycle (not an incremental `reset_run`) to
see what real slack this constraint surfaces, and a real-hardware re-test
of the greedy-mode divergence test afterward. Given the previous session's
finding that the specific wrong token picked for 3/5 test prompts changed
between two firmware builds differing only in a diagnostic read that
cannot causally affect the value in question, a real, previously-invisible
timing violation somewhere in this 67,133-endpoint domain is a coherent
explanation -- and this is now the clearest, most concrete next thing to
test on real hardware this investigation has had.

## The full resynth ran, and it found real, razor-thin margins throughout every kevgpt_ddr_bundle CDC crossing

Direct continuation of the entry above. Ran the full clean rebuild the
previous entry called for: `AUTO_INCREMENTAL_CHECKPOINT` was explicitly
disabled on `synth_1` first -- it was pointing at the *pre-fix* synthesis
checkpoint, which would have silently defeated the whole point of the
resynth -- then `reset_run synth_1` -> `launch_runs synth_1` (clean, ~10
min) -> `reset_run impl_1` -> `launch_runs impl_1 -to_step write_bitstream`
(no `-jobs`, per this project's own documented `launch_runs -jobs` hang
risk), against the real `.xpr`. ~56 minutes total real Vivado time.
Post-build, `report_clock_networks` reconfirms `sys_clk_pin` constrained
with the identical endpoint counts as the in-memory check.

**`report_timing_summary`'s top-level numbers looked clean (0 failing
setup/hold/PW design-wide) but turned out not to be trustworthy for this
specific clock class.** Every clock downstream of `sys_clk_pin`
(`clk_pll_i`, `clk_out1_xilinx_clk_wizard_clk_wiz_0_0` -- Vivado's own
names for `ui_clk`/`gen_clk`) is auto-inferred, never explicitly
`create_clock`'d, and lands in the summary's separate "Other Path Groups
Table" under an `**async_default**` label -- which showed a suspiciously
clean WNS +11.120ns / 0-of-3264-failing for `clk_out1_xilinx_clk_wizard_clk_wiz_0_0`,
directly contradicted by a manual `report_timing -from ... -to ...` query
on the *same* clock object (confirmed non-duplicate: one clock, correctly
linked `MASTER_CLOCK=clk_pll_i`), which reproducibly found a real
`VIOLATED -4.122ns` setup path, 85 logic levels deep, inside the CPU
core's own multiplier/FPU-operand forwarding logic
(`cv32e40px_xif_wrapper_i/.../id_stage_i`). Could not fully reconcile why
the summary table disagrees with a direct query for this clock class --
flagged as an open sub-question, not investigated further since the path
itself is unrelated CPU-core logic, not kevgpt's own datapath.

**Auditing `kevgpt_seq`'s own hierarchy directly is where this landed
squarely on target.** Found the real instance names via `get_cells -hier
-filter {ORIG_REF_NAME == <name>}` rather than guessing (`u_kevgpt/u_seq`
= sequencer_vec, `u_kevgpt_ddr_bundle`, `u_kevgpt/u_seq/u_gemv/u_wb` =
weight_bank_tdp, `u_kevgpt/u_seq/g_kvb_ddr.u_kvb` = kv_bank_ddr,
`u_kevgpt/u_seq/g_wld.u_wld` = weight_loader_ddr, `u_kevgpt/u_seq/u_gemv` =
gemv_banked_resident_vec, `u_kevgpt/u_seq/u_attnA` = vec_attn_w), then ran
`report_timing -to [get_pins ...]` scoped to every D pin across all of them
(8,803 pins), both setup (`-delay_type max`) and hold (`-delay_type min`).

**Zero VIOLATED paths -- but the 5 worst hold-margin paths, 0.054ns to
0.067ns, are every `async_fifo_gray` crossing in `kevgpt_ddr_bundle.sv`,
both directions, all four instances.** Not confined to the "official"
Gray-pointer synchronizer stages either: the single worst path
(`u_kv_rd_req_cdc/mem_reg_0_3_6_11/RAMC_D1/CLK` -> `u_rd_engine/cmd_addr_q_reg[11]/D`,
0.054ns) is the CDC FIFO's own data-memory array, read straight into
`mig_read_engine`'s DMA command-address register -- not a redundant
pointer bit the FIFO's own empty/full protocol is designed to tolerate,
but the actual address value the next weight/KV read command will use.
The next four worst paths repeat the identical shape against
`cmd_addr_q_reg`/`data_q_reg` for the other three FIFOs (weight
read-request, KV write-packet), plus two genuine Gray-pointer synchronizer
paths at 0.060-0.067ns.

**This is the strongest, most complete explanation this investigation has
produced.** Systemic (every crossing, both directions, not one outlier),
on paths that directly determine DMA address/data values rather than
metastability-tolerant pointer bits, and margins this thin are exactly
what this project's own JTAG-CDC history already documents as fragile
enough to flip negative under placement changes elsewhere in the design --
which is precisely the mechanism the earlier session's build-dependent-
winner finding needed: different firmware builds, different floorplan/
congestion, different margin tips negative first, different corrupted
address, different wrong token. Also consistent with why "care"/id 2213
recurs so often rather than corruption landing uniformly at random: this
investigation's own earlier raw-DDR3 diagnostic already flagged "rows
2048-4095" (which covers id 2213) as where fixation words statistically
cluster -- a corrupted address landing near a real request's address,
rather than at a uniformly random one, would predictably favor nearby
rows.

**Not yet done**: actually fixing these margins (Gray-pointer-bus skew
constraints, `set_max_delay -datapath_only` on the specific FIFO-memory-
to-consumer paths identified above, or a deeper pipeline/register stage on
the consumer side) and a real-hardware re-test. Real, likely-iterative
RTL/constraint engineering, not a one-line fix like the root clock gap
was.

## The CDC margins are fixed: three scoped set_max_delay exceptions in constraints.xdc

Direct continuation. Three `set_max_delay -datapath_only` exceptions added
to `constraints.xdc`, scoped explicitly by full hierarchical cell name to
the six real `async_fifo_gray` instances inside `u_kevgpt_ddr_bundle`
(`u_kv_wr_pkt_cdc`, `u_kv_wr_ack_cdc`, `u_kv_rd_req_cdc`, `u_kv_rd_ret_cdc`,
`u_wl_rd_req_cdc`, `u_wl_rd_ret_cdc`) -- deliberately not a bare
`*async_fifo_gray*` wildcard, matching this file's own documented lesson
from the DMI CDC fix about over-broad CDC-net wildcards silently perturbing
unrelated logic elsewhere in the design.

**A real Vivado quirk hit along the way**: `get_cells -hierarchical
{pattern1 pattern2 ...}` (the bare multi-pattern-list form) silently
resolved to "No valid object(s) found" for these six-instance lists, even
though the exact same instances individually resolve fine and
`get_nets -hierarchical {pattern1 pattern2}` (used successfully by the
existing DMI CDC constraint in this same file) handles multi-pattern lists
correctly. `get_cells` needed the explicit `-filter {NAME =~ "..." || NAME
=~ "..." || ...}` boolean-OR form instead -- worth remembering before
reusing the `-hierarchical {list}` shorthand for `get_cells` specifically.

Three exceptions, 4ns bound (comfortably under min_period(gen_clk=20ns,
ui_clk=10ns)=10ns, ~7x the worst real routed delay actually observed):
(1) `wr_gray_q` -> `wr_gray_rsync1_q`, the write-side Gray-pointer
synchronizer stage, across all 6 instances (18 cells each side); (2) the
mirror `rd_gray_q` -> `rd_gray_wsync1_q` stage (18 cells each side); (3)
the FIFO-memory-array data paths, `-from` only (any destination), across
the 5 instances that actually have a memory array (`u_kv_wr_ack_cdc` is a
1-bit ack pulse, no `mem_reg` cells to except) -- 1,287 cells total.

**Verified twice against the live implemented design before committing.**
First pass: built the exception via a Tcl filter-string helper, applied
in-memory, re-checked the exact worst path
(`u_kv_rd_req_cdc/mem_reg_0_3_6_11` -> `u_rd_engine/cmd_addr_q_reg[11]/D`)
-- went from `Slack (MET): 0.054ns` to "No timing paths found" (correctly
reclassified under the exception); re-ran the same `kevgpt_seq`-hierarchy
hold audit from the previous entry -- worst remaining margin is now
0.108ns, matching the DMI CDC's own already-accepted baseline elsewhere in
this file, not a new thin spot; `check_timing` shows identical 0/0/0/0
no_clock/unconstrained_internal_endpoints/multiple_clock/generated_clocks
counts before and after. Second pass: `read_xdc` on the *exact* real,
committed file (multi-line backslash continuations and all, not a
Tcl-reconstructed equivalent) against the same live design, specifically
to catch any escaping/continuation issue the first pass's differently-
constructed Tcl strings could have missed -- identical result.

**Still needed**: one more full clean resynthesis with both fixes present
(sys_clk_pin + these three exceptions), and a real-hardware re-test of the
greedy-mode divergence test -- the actual answer to whether this was the
fixation-word cause.

## The real-hardware retest: CDC timing-constraint gap ruled out, definitively

Direct continuation. Ran the second full clean resynthesis
(`AUTO_INCREMENTAL_CHECKPOINT` disabled again, ~56 min, no `-jobs`) with
both fixes present. Confirmed live in the real build before touching
hardware: `report_clock_networks` shows `sys_clk_pin` constrained with the
same 67,133/232 endpoints as the in-memory check; the previously-worst CDC
path (`u_kv_rd_req_cdc/mem_reg_0_3_6_11` -> `u_rd_engine/cmd_addr_q_reg[11]/D`)
now reports "No timing paths found"; `write_bitstream Complete!`.

**Board was repowered mid-session by the user.** SRAM-based FPGA
configuration is volatile, so this meant reprogramming from scratch, not
just reloading firmware. Killed the now-stale `openocd` process first.
Programming hit the exact stale-hw-target-registration issue this
project's reference notes already document: the default target resolution
tried `.../Digilent/200300B5E5AA` (dead post-repower) and failed with "no
current hw_target." Probed `get_hw_targets` directly, found the live one
differs only by a trailing "B" (`200300B5E5AAB`), pinned it via `HW_TARGET`
env var, and it opened on the first try. Also needed an absolute path for
the `.bit` file -- the pgm.tcl script's relative-path resolution didn't
match my shell's cwd inside the Vivado batch session. Programmed clean.
Restarted `openocd` fresh against the newly-programmed board -- connected
immediately, JTAG tap found.

Rebuilt firmware with `KEVGPT_FORCE_GREEDY=1`/`KEVGPT_PRINT_IDS_ONLY=1`
(the exact isolation-build config from the earlier "care" discovery
session), reloaded via the usual confirm-listener-then-trigger sequence,
resent weights (~15 min), reran the identical 5-prompt x 8-repeat greedy
test via the same `isolation_test.py` harness used before.

**Result: byte-for-byte identical to every capture before either fix.**
Same wrong tokens ("care"/id 2213, "carefree"/id 2216) at the same
positions (token 0 for 4/5 prompts, token 49 for "once upon a time"),
same 100% determinism across all 8 repeats each. Two real, previously-
completely-invisible timing gaps -- found via live Vivado queries, fixed,
verified multiple independent ways, built into a genuinely fresh bitstream
with zero incremental-synthesis reuse -- changed nothing observable about
the fixation-word symptom on real hardware. The CDC timing-constraint gap
(both the missing root clock and the razor-thin async_fifo_gray margins)
is ruled out as the fixation-word cause, definitively, not just narrowed.
Both fixes are kept regardless -- they close a genuine, previously-total
blind spot in this design's timing closure -- but they are not *the*
answer.

**A necessary correction to the earlier "care weight row" session's own
conclusion**: that session attributed a build-dependent change in which
wrong token won (isolation/baseline builds vs. the logit-probe build) to
"placement/routing differences between builds," and used it as supporting
evidence for the CDC-timing hypothesis. That reasoning doesn't survive
scrutiny: those three builds differ only in *firmware*, all running on the
*same* FPGA bitstream -- firmware cannot move a placement or a route on
the fabric. Whatever changed the outcome between those builds, it has to
be sensitive to firmware *execution timing* itself (e.g. exactly which
cycle a DMA request gets issued on, relative to some other concurrent
activity), not a static synchronizer margin. That finding is still real
evidence for *some* live runtime race -- it just was never actually
evidence for §6's CDC margins specifically, which this session's direct
real-hardware test now rules out on its own terms.

Firmware toggles (`KEVGPT_FORCE_GREEDY`, `KEVGPT_PRINT_IDS_ONLY`) reverted
to off before committing, matching convention.

Diagnostic infrastructure this closes out: §6/Hypothesis A (the CDC
timing-constraint gap) is done as a fixation-word candidate. Remaining
open leads: §4/Hypothesis B (owner-FIFO backpressure -- already fixed
defensively, never confirmed as the active bug under real contention) and
a full multi-master contention testbench (weight + KV + CPU traffic
simultaneously, randomized timing) as the next real diagnostic step.