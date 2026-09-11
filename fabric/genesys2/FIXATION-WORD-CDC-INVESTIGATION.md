# Fixation-word investigation: real-hardware-only text corruption in the multi-master DDR read path

Status as of 2026-09-11: **open, not root-caused, not fixed.** This document is
the standalone reference for the whole investigation — everything needed to
either continue it or hand it off, without reconstructing the trail from
`model/SCALE-UP-LOG.md`'s chronological entries (which have the full blow-by-blow
if this summary needs expanding).

**Revision note:** this document's first version over-weighted the CDC
timing-constraint gap (§6) as *the* leading hypothesis. An external review of
that draft (summarized in §6a) made a strong case that an unsafe owner-tracking
FIFO in the multi-master DMA path is at least as likely a cause, is more
consistent with the *severity variance* between the two captured divergences
(§2), and is far cheaper to test. §§3, 6, 7, and 8 below have been corrected
and re-prioritized accordingly; nothing was deleted, only re-weighted.

## 1. Original task goal

Checkpoint C (`data/ckpt_stepC_d128_v16384.qat.pt`, D=128/n_layer=12/n_head=2/
VOCAB=16384, the checkpoint actually deployed to the real Genesys2 board) was
producing occasional non-sequitur words as story subjects/objects on real
hardware — "cardinal," "buster's," "cube," and similar — words that never
appear in software-side generation of the identical checkpoint. The ask was
two-part: **(1) first make the real FPGA and software produce the same
stories** (a deterministic, verifiable baseline), **(2) then fix the
fixation-word pattern.** Neither has been achieved; the investigation instead
spent its effort ruling out almost everything *except* the actual cause,
landing on a real but unconfirmed hardware/toolchain gap.

## 2. The problem, precisely stated

Real-hardware text generation diverges from the Python golden reference
(`model.goformer_kvq.IntKVQSequencer`, the same algorithm the RTL is designed
to implement bit-exactly) at a specific decode step, substituting a token the
golden reference ranks nowhere near competitive. Two concrete, captured
instances:

- **Sampled mode**, real seed `0x42da8a1f`: golden and hardware agree for 22
  tokens, then diverge — golden picks "went" (rank 1, logit 415,276,205),
  hardware picks "saw" (rank 2, logit 414,740,811). A 0.13% margin — a
  near-tie, plausibly explained by a tiny numerical discrepancy.
- **Greedy mode** (no sampling noise at all), same checkpoint: diverges at
  generated token ~50 — golden picks `.` (logit 11.95), hardware picks "care"
  (rank **15,118 of 16,384**, logit -8.48). Not a near-tie. A gross,
  unambiguous wrong answer.

Both are **fully deterministic**: 40 repeated trials (5 prompts × 8 repeats,
greedy mode) produced byte-identical output every single time, fixation words
included. This is not timing noise or metastability-flavored randomness in
the colloquial sense — it is a reproducible function of (checkpoint, prompt,
seed, *this specific bitstream*).

## 3. What's ruled out, with direct evidence (in the order it was checked)

| Candidate | Verdict | Evidence |
|---|---|---|
| Checkpoint/weights | Clean | Golden-reference logits for all fixation words rank ~3,000–16,000/16,384 across test prompts — never competitive under correct computation. |
| RTL compute logic, fully-resident config | Clean | Bit-exact vs. golden reference, greedy and sampled, arbitrary seeds. |
| RTL compute logic, streaming config | Clean | Bit-exact vs. golden reference using the *exact real captured seed* from a hardware run that produced "care," extended to 53 generated tokens (previously untested that deep) — simulation predicts "went"/"."/whatever golden predicts, not what hardware produced. |
| Weight-packing pipeline (`write_mems_wideword`/`wrom_to_words`) | Clean | `send_weights.py`'s transmitted word list is byte-for-byte identical (2,670,592/2,670,592 words) to the RTL simulation's own `wrom.mem`. |
| UART reception → **DDR3 storage** (`uart_load_blob` in `main.c`) | Clean, but narrower than first claimed | Built a raw-DDR3-readback diagnostic (`KEVGPT_DIAG_DUMP_HEAD`, off by default in `kevgpt_interactive/main.c`) that reads the suspect address range straight from DDR3 via a plain CPU load. Zero mismatches across all 8,192 dumped words. **Correction: this reads DDR3 via a plain CPU load, which bypasses `weight_loader_ddr`, the CDC crossing, `mig_read_mux2`, and `mig_dual_master_arbiter` entirely.** It proves the bytes UART wrote into DDR3 are correct. It proves *nothing* about whether those bytes come back correctly through the real streaming-read path into `weight_bank_tdp` — which is exactly the path under suspicion in §4 and §6a. This was originally written up as "the write side is clean," which overstated what was actually tested. |
| Tokenizer ID→string table | Clean | The DDR3-resident tokenizer blob on the board is byte-identical to a fresh build from `meta.json`; decodes every suspicious ID correctly (id 2048 genuinely is "buster," etc. — the words themselves are real, unremarkable vocabulary entries). |
| `async_fifo_gray.sv` (the CDC primitive itself) | Clean | Audited directly against Cummings' canonical async-FIFO design: Gray-code math, 2-FF `ASYNC_REG` synchronizer structure, and the full/empty detection formulas are all textbook-correct. The one deliberate deviation (registered `wr_full` instead of combinational, to break a real Vivado DRC LUTLP-1 loop) was hand-traced through a worked example and confirmed not to cause overflow. This clears the FIFO's own logic; it says nothing about physical placement of the synchronizer flops or the actual clock relationship feeding them (§6a). |
| Sampling-methodology mismatch (my own earlier test artifact) | Ruled out | Reran with the *exact* algorithm the RTL implements (`gumbel.GumbelRng`), not an approximate PyTorch proxy: 1,500 tokens, zero fixation-word hits. |
| Marginal/random real-silicon timing noise | Narrowed, not ruled out | The 40-trial repeated-greedy-decode test (above) is 100% deterministic. **Correction: this only rules out *pure random/probabilistic* noise, not CDC as a mechanism generally.** `gen_clk` is PLL-derived from `ui_clk`, so their relative phase can be extremely repeatable across power-on/reconfiguration — a synchronizer sampling too close to a transition on one specific, fixed phase relationship would reproduce the *same* deterministic failure every time on a given bitstream. Determinism narrows which CDC mechanisms are plausible; it does not clear CDC as a category. |

## 4. What IS implicated: modules, signals, and the traffic path

Weight reads do not go directly from `weight_loader_ddr.sv` to the physical
MIG. The real deployed path (never exercised by any simulation gate,
including this session's own `tb_seq_vec_kv_stream.sv`, which ties
`KV_DDR_BACKED=0` and has no second master at all):

```
weight_loader_ddr.sv (gen_clk, ~50MHz — instantiated inside sequencer_vec.sv's
                       own hierarchy, PLL-derived from ui_clk)
    |
    |  CDC crossing: async_fifo_gray x2 (read-request, read-return),
    |  inside kevgpt_ddr_bundle.sv
    v
mig_read_mux2.sv (ui_clk)   -- merges weight_loader_ddr's reads vs. kv_bank_ddr's
    v
mig_read_engine.sv (ui_clk)
    v
mig_dual_master_arbiter.sv (ui_clk)  -- merges kevgpt's own bundle (side A)
    |                                    vs. cpu_ddr_bridge (side B)
    v
physical MIG (genesys2_mig_native_shell) / real DDR3
```

Relevant files, all in `kevgpt-genesys2-soc` (the vendored X-HEEP SoC repo,
**separate from the `kev-gpt` repo** — `~/RVchatbot/kevgpt-genesys2-soc`):

- `hw/vendor/esl_epfl_x_heep/hw/ip/kevgpt_seq/rtl/weight_loader_ddr.sv` —
  issues DMA read requests for the head/block weight windows, drains
  returned beats into `weight_bank_tdp`'s boot-load port. Lives on `gen_clk`.
- `hw/vendor/esl_epfl_x_heep/hw/ip/kevgpt_seq/rtl/kevgpt_ddr_bundle.sv` —
  the CDC crossing itself. Six `async_fifo_gray` instances (KV write-packet,
  KV write-ack, KV read-request, KV read-return, weight read-request, weight
  read-return), all `gen_clk`↔`ui_clk`.
- `hw/vendor/esl_epfl_x_heep/hw/ip/kevgpt_seq/rtl/mig_read_mux2.sv` —
  merges weight and KV read streams onto one `mig_read_engine`, `ui_clk`
  domain, post-CDC. Uses an owner-tracking FIFO to route returns back to the
  correct requester.
- `hw/vendor/esl_epfl_x_heep/hw/ip/ai_accel/rtl/accelerator/streamer/mig_dual_master_arbiter.sv`
  — merges kevgpt's bundle with `cpu_ddr_bridge`'s traffic onto the physical
  MIG. Same owner-FIFO idiom as `mig_read_mux2`.
- `hw/vendor/esl_epfl_x_heep/hw/ip/ai_accel/rtl/accelerator/common/async_fifo_gray.sv`
  — the CDC primitive (audited clean, see §3).
- `hw/vendor/esl_epfl_x_heep/hw/fpga/xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`
  — top-level wrapper; instantiates everything above, generates `clk_gen`
  from `mig_ui_clk` via `xilinx_clk_wizard_wrapper_i`.
- `hw/vendor/esl_epfl_x_heep/hw/fpga/constraints/genesys2/constraints.xdc` —
  the timing-constraint file with the gap described in §6.

**Hypothesis B, and a finding that deserves equal billing with §6's CDC gap
(Hypothesis A), not a footnote to it:** both `mig_dual_master_arbiter.sv`'s
`u_rd_owner_fifo` and
`mig_read_mux2.sv`'s `u_owner_fifo` — the FIFOs that track which master a
pending DDR3 read return belongs to, so a returned beat routes back to
whoever actually asked for it — leave `in_ready_o` unconnected:
```systemverilog
sync_fifo #(...) u_owner_fifo (
    .in_valid_i(owner_push_valid),
    .in_ready_o(),   // never checked
    ...
```
Hand-checked the sizing (`MAX_OUTSTANDING=16` vs. 32/64-deep owner FIFOs) and
it looks adequate *under normal operation*, so this was not confirmed active
as originally written up. An external review of this document (§6a) made a
concrete case for why it deserves to be the lead suspect: if any single
`owner_push` is ever dropped or misordered, ownership tracking shifts by one
entry from that point on — DDR3 data itself stays perfectly correct, but
request N's *return* gets attributed to request N+1, silently, with no error
signal (no assertion here catches this — only underflow is checked, not a
push/pop count mismatch). That produces exactly this investigation's evidence
profile: DDR3 storage correct (per the corrected row in §3), RTL compute
logic correct in every simulation, and a **wrong word that can be arbitrarily
wrong** (not clustered near a numerical near-tie) — because it's not a
numerical error on the *intended* row's data at all, it's the *entirely
unrelated* row from a neighboring, misrouted request. This also naturally
explains why the two captured divergences have such different severity
(§2: a 0.13% near-tie once, a rank-15,118 miss another time) — a shifted
ownership index doesn't correlate with any numerical closeness between the
correct and substituted values, unlike a marginal-timing bit-flip theory,
which has no obvious reason to sometimes be tiny and sometimes enormous.

## 5. What's impacted

- **Real-hardware chat quality on the deployed Genesys2 board.** The
  Gumbel-noise TEMP recalibration earlier in this investigation measurably
  reduced the flagged/degenerate rate (32%→12%), but that fix addressed a
  *different*, real problem (miscalibrated noise magnitude for NLAYER=12) —
  it's plausible it also partially masked this defect's visibility without
  touching its cause (lower temperature → model picks its own high-confidence
  token more often → fewer chances for this corruption to become the winner).
- **Any future scale-up.** DMA traffic through this exact boundary has grown
  ~8.5x since the first time it was exercised (`GW_HEAD` = 3,840 words at
  VOCAB=1900 vs. 32,768 words at today's VOCAB=16384). Both live hypotheses
  (§6's CDC gap, §4/§6a's owner-FIFO backpressure gap) scale with traffic
  volume the same way — either predicts growing the model further makes the
  symptom *more* frequent, not less.
- **Trust in "PASS" real-hardware verdicts for this whole class of
  deployment.** Every prior "real hardware confirmed working" milestone for
  per-layer weight streaming (NLAYER=8 onward, 2026-08-26+) was verified with
  small sample sweeps and RTL-simulation bit-exactness — neither of which
  would have caught this (see §7 and the "why didn't this show up before"
  analysis in `SCALE-UP-LOG.md`'s corresponding section).

## 6. Hypothesis A: the CDC timing-constraint gap

**Important correction, made after external review (§6a): `set_clock_groups
-asynchronous` is not itself a hardware fix.** It only changes what static
timing analysis checks — it cannot improve a synchronizer or resolve
metastability that's already physically present. If `gen_clk` truly is
MMCM/PLL-derived from `ui_clk`, blindly declaring the pair asynchronous can
*hide* a real, deterministic timing relationship rather than illuminate it.
The right first move is determining what Vivado actually believes the clock
relationship is — via `report_clocks -verbose`, `report_clock_networks`, and
specifically `get_clocks -of_objects [get_pins <sync_ff>/C]` on the actual
synchronizer flip-flops in `kevgpt_ddr_bundle.sv` — not guessing a
constraint and hoping. The rest of this section is the evidence for why a gap
exists at all; §8 has the corrected, safer procedure for closing it.

`gen_clk` is PLL-derived from `ui_clk` — a real, computable frequency
relationship. Grepped every `.xdc` file in the project: **no
`set_clock_groups`/`create_generated_clock` declares this pair as
asynchronous, or adds a per-path CDC exception for any of `kevgpt_ddr_bundle.sv`'s
six `async_fifo_gray` crossings.** This project's own constraints file
documents hitting exactly this failure mode once before, for a *different*
clock pair (`jtag_clk_pin` vs. everything else) — Vivado can derive an
"implicit" synchronous relationship between related clocks from their
periods' least-common-multiple beat pattern and report a build as
timing-clean while the real CDC synchronizer margin is thin or negative.
That fix (`set_clock_groups -asynchronous` for JTAG) is already in the file,
at `hw/vendor/esl_epfl_x_heep/hw/fpga/constraints/genesys2/constraints.xdc`
line 18. No analogous line exists for `gen_clk`/`ui_clk`.

**This was going to be a one-line fix. It is not.** Attempting to add it
(`set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins
u_mig/ui_clk_o]] -group [get_clocks -of_objects [get_pins
xilinx_clk_wizard_wrapper_i/clk_out1_0]]`) and then *verifying* it against
the real implemented design (`open_run impl_1` on the exact `.xpr`/checkpoint
that produced the bitstream currently on the board) revealed it doesn't
resolve — neither guessed pin path returns a clock object. This is itself
an instance of the mistake the external review calls out: I guessed
hierarchy-name-based pin paths (`u_mig/ui_clk_o`) instead of querying the
actual synchronizer register's clock pin directly. Digging further with what
I had:

```
report_clocks  ==>  only jtag_clk_pin and spi_slave_clk_pin exist.
```

Confirmed this isn't a query artifact by independently checking the actual
build's own output report
(`.../impl_1/xilinx_core_v_mini_mcu_wrapper_kevgpt_timing_summary_routed.rpt`,
"Clock Summary" section) from the real Vivado run that produced the current
bitstream (Sep 6 02:24) — same result. **`gen_clk` and `ui_clk` do not
appear anywhere in this design's user-visible clock list at all**, which is
either:

- Vivado's out-of-context (OOC) IP methodology handling MIG's and the
  Clocking Wizard's internal timing entirely inside their own
  pre-characterized black-box models (`mig_7series_0_ooc.xdc` and
  `xilinx_clk_wizard_ooc.xdc` both do have their own internal `create_clock`
  statements, on their own OOC-local port names) — in which case the actual
  CDC-relevant clock relationship is invisible to the queries I know how to
  run without deeper Vivado-OOC-methodology expertise, or
- there genuinely is no root `create_clock` for the 200MHz DDR3 reference
  clock anywhere in this project's *top-level* XDC, and the entire
  `gen_clk`/`ui_clk` domain — which is to say, essentially the entire design
  outside the debug/SPI paths — has never had top-level timing closure
  checked at all.

I could not distinguish between these two from static inspection. The
"Unconstrained Path Table" in the real build's own timing report is short
(not the flood of thousands of paths you'd expect if truly *nothing* else in
the design were clocked), which argues against the second, more alarming
reading — but I don't have a confirmed explanation for why it's short if
`gen_clk` genuinely isn't a recognized clock. §8 has the correct procedure
(querying synchronizer pins directly, plus `report_cdc` if licensed) for
resolving this cleanly instead of guessing again.

## 6a. External review: corrections and a co-equal hypothesis

A review of this document's first draft (2026-09-11, pasted into the
`kev-gpt` session, full text not reproduced here) made several corrections,
summarized and credited here since they materially changed this document:

1. **`set_clock_groups -asynchronous` is not a fix** — folded into §6 above.
2. **Determinism doesn't clear CDC as a category** — folded into §3's table.
3. **The "write side is clean" claim overstated what was tested** — folded
   into §3's table; the CPU-readback diagnostic bypasses the entire streaming
   path under suspicion.
4. **The unconnected `in_ready_o` owner-FIFO gap (§4) deserves to be a
   co-equal leading hypothesis, not a subordinate footnote** — folded into
   §4, with a mechanistic explanation for why it fits the evidence (severity
   variance between the two captured divergences) at least as well as CDC
   metastability does, and is far cheaper to test or fix.
5. A concrete, prioritized action plan — CRC-based hardware diagnostics, a
   weight-traffic-only isolation experiment, transaction-ID tracing, a
   proper multi-master contention testbench, Gray-bus skew constraints, and
   architectural-invariant ILA triggers rather than probing synchronizer
   metastability directly — folded into the rewritten §8.

My own assessment, for whoever reads this next: points 1–4 are corrections I
accept without reservation — each identifies a real gap in the original
reasoning. Point 5's plan is sound and better-sequenced than what this
document had; §8 now reflects it with one addition — `report_cdc` is a
licensed Vivado ML Enterprise feature and its availability on this
installation is unverified, so it's listed as "try this, fall back to direct
pin queries if unavailable" rather than assumed to work. I'd also keep the
CDC-constraint question (§6) and the owner-FIFO question (§4) running in
parallel rather than fully deprioritizing either — they are not mutually
exclusive, and the "clocks don't appear in the design's clock list at all"
finding in §6 is strange enough on its own to be worth resolving regardless
of what the owner-FIFO experiments show.

## 7. Why this investigation cannot go further from here — and what actually can

Not everything below needs a human driving Vivado. Splitting this explicitly,
since conflating them in the original draft made the whole remaining task
look more blocked than the owner-FIFO half of it actually is:

**Tractable as normal RTL/firmware work, no interactive Vivado archaeology
needed** (see §8, items 1–4):
- Wiring the owner FIFOs' `in_ready_o` into real backpressure and adding the
  outstanding-request/response/owner-occupancy accounting assertions (§4,
  §6a) — a contained RTL change to two files plus new `assert property`
  statements.
- A weight-bank CRC diagnostic that verifies data through the *real* full
  path (`weight_loader_ddr` → CDC → mux → arbiter → MIG → CDC → weight bank),
  closing the gap the corrected §3 table now flags.
- The weight-traffic-only isolation experiment (disable `KV_DDR_BACKED`/
  `cpu_ddr_bridge` traffic, rerun the same greedy test) — a config/parameter
  change plus a resynth, no new logic.
- A proper `tb_kevgpt_ddr_bundle_full.sv` contention testbench exercising
  weight + KV + CPU traffic simultaneously with randomized MIG latency —
  real engineering effort, but self-contained simulation work.

These are the right *next* things to do, precisely because they don't require
resolving the clock-naming mystery first, and several of them (the isolation
experiment especially) can independently falsify or confirm the CDC
hypothesis in §6 as a side effect.

**Genuinely blocked without a human at the Vivado controls**:
- **Distinguishing "OOC-hidden but fine" from "genuinely unconstrained"**
  requires interactively driving Vivado (`report_clock_networks`, direct
  `get_clocks -of_objects [get_pins <sync_ff>/C]` queries on the real
  synchronizer cells, `report_cdc` if licensed, or a fresh from-scratch
  synthesis with the OOC methodology deliberately disabled to see what
  surfaces) — exploratory, judgment-driven work, not a lookup.
- **Writing a constraint against the wrong theory is worse than writing
  none.** A `set_clock_groups` line that silently resolves to an empty
  `get_clocks` result doesn't error the build — it just does nothing, while
  looking exactly like a real fix in a diff. I caught my own first attempt
  doing exactly this by verifying against the live implemented design before
  committing it; that verification step is not optional for whoever
  continues this.
- **If real negative timing slack is eventually confirmed**, closing it needs
  a full re-synthesis/re-implementation/re-bitstream cycle and real-hardware
  re-test after the correct constraint is in place — genuinely
  time-consuming, real-tool work — or, per the external review (§6a, §8),
  physical placement checks and Gray-bus skew constraints if the
  synchronizer stages turn out to be routed with uncontrolled skew.
- If it comes to observing metastability directly, this project has working
  precedent (the `ai_accel` CDC investigation on 2026-08-16) for doing it via
  ILA — but per §6a/§8, probing architectural invariants (owner-FIFO
  occupancy mismatches, request/response counters) is a more tractable ILA
  strategy than trying to catch metastability on the synchronizer flops
  themselves directly.

None of this is a dead end — it's a well-scoped, concrete set of next tasks,
most of which don't require the part that's actually blocked.

## 8. Recommended next steps, in priority order (revised per §6a)

Ordered by diagnostic value per unit of implementation effort, per the §6a
review. Items 1–4 don't require resolving §6's clock-naming question first;
item 4 can independently shed light on it as a side effect.

1. **Add a weight-bank CRC diagnostic that exercises the real full path.**
   Software computes the expected CRC32 over each packed weight block/head
   image; a debug firmware mode triggers a real hardware load through the
   *actual* `weight_loader_ddr → CDC → mux → arbiter → MIG → CDC →
   weight_bank_tdp` path (not the CPU-bypass readback from §3) and reports
   the CRC after each block. `HEAD PASS / BLOCK0 PASS / BLOCK1 FAIL` localizes
   the defect immediately, far faster than waiting 50 generated tokens for
   a fixation word to appear. This is the single most direct fix for the
   corrected §3 claim ("write side is clean" never actually covered this
   path) and should come first.
2. **Make the owner FIFOs' `in_ready_o` real backpressure, in both
   `mig_dual_master_arbiter.sv` and `mig_read_mux2.sv`:**
   ```systemverilog
   // before
   .in_valid_i(owner_push_valid), .in_ready_o(),
   // after
   .in_valid_i(owner_push_valid), .in_ready_o(owner_ready),
   ...
   assign upstream_ready = downstream_ready && owner_ready;
   ```
   Add outstanding-request accounting as a running assertion in both modules
   (`req_count - ret_count == owner_fifo_level`, checked every cycle) plus:
   ```systemverilog
   assert property (@(posedge ui_clk) request_accepted |-> owner_ready);
   assert property (@(posedge ui_clk) owner_pop |-> !owner_empty);
   ```
   Cheap, safe, and closes the exact hole §4/§6a identifies as never having
   been closed. If this assertion ever fires — in simulation or on real
   hardware via ILA — that is the confirmed defect, not an inference from
   generated text.
3. **Run the weight-traffic-only isolation experiment on real hardware.**
   Disable `KV_DDR_BACKED`/`cpu_ddr_bridge` traffic (config + resynth, no new
   RTL) so only `weight_loader_ddr → CDC → mig_read_engine → MIG` is active,
   then rerun the same greedy-mode repeated-trial test from §2. If the
   fixation-word pattern disappears, the defect is in `mig_read_mux2` /
   `mig_dual_master_arbiter` / the owner-FIFO path specifically, not the base
   CDC crossing — a fast, high-value bisection. A further bypass of just
   `mig_dual_master_arbiter` (kevgpt's bundle wired directly to the physical
   MIG, no `cpu_ddr_bridge` contention) or just `mig_read_mux2` (weight reads
   wired directly to `mig_read_engine`, no KV contention) sharpens this
   further if needed — classic divide-and-conquer.
4. **In parallel, resolve §6's clock-naming question properly.** Open the
   project interactively (not via `open_run` on an archived checkpoint) and
   query the actual synchronizer flip-flops directly rather than guessing
   hierarchy paths:
   ```tcl
   report_clocks -verbose
   report_clock_networks
   get_cells -hier -filter {NAME =~ *kevgpt_ddr_bundle*async_fifo_gray*}
   # for each representative sync register on both sides of a crossing:
   get_clocks -of_objects [get_pins <sync_ff_name>/C]
   report_property [get_cells <sync_ff_name>]   ;# then check LOC placement,
                                                  ;# the two sync stages should
                                                  ;# be physically close
   ```
   Try `report_cdc -details` / `report_cdc -verbose` if licensed (Vivado ML
   Enterprise CDC methodology — availability on this installation is
   unverified) for a direct classification of each synchronizer as safe/
   unsafe. If clocks are confirmed genuinely unconstrained, add the correct
   `create_clock`/`set_clock_groups -asynchronous` pair (matching the JTAG
   precedent at `constraints.xdc` line 18) *and* Gray-pointer-bus skew
   constraints (`set_max_delay -datapath_only` and, where supported,
   `set_bus_skew`, from the real synthesized Gray-register names — a 2-FF
   synchronizer alone doesn't protect a multi-bit bus if routing skew between
   bits is uncontrolled), then re-synthesize/re-implement from scratch (not
   an incremental `reset_run`) and check whether previously-"passing" paths
   now report real, possibly negative, slack.
5. **If items 1–4 don't localize the defect**, build the full
   `tb_kevgpt_ddr_bundle_full.sv` contention testbench (weight + KV + CPU
   traffic simultaneously, randomized MIG return latency within whatever
   ordering guarantee the native UI actually provides, randomized `app_rdy`/
   `app_wdf_rdy` stalls, varied `gen_clk`/`ui_clk` phase) — this project's
   biggest missing regression, and likely to expose owner-FIFO/backpressure
   bugs quickly if any remain after item 2's fix.
6. **Only after 1–5 are clean should real ILA time be spent hunting
   metastability directly**, and even then, per §6a: trigger on architectural
   invariants (`owner_fifo_level != req_count - ret_count`, `owner_push &&
   !owner_ready`, `owner_pop && owner_empty`) rather than trying to observe
   the synchronizer flip-flops' metastable behavior itself, which is
   difficult to catch and rarely conclusive.
7. **Longer-term, once fixed**: keep a small permanent hardware health
   monitor (owner-FIFO overflow/underflow counters, request/response
   counters per master, a sticky `DDR_PROTOCOL_ERROR` bit) so any future
   regression of this class surfaces as an explicit error rather than a
   silently wrong generated word. Useful precedent for future scale-up,
   independent of how this specific investigation resolves.

## 9. Evidence trail / artifacts

- `model/SCALE-UP-LOG.md` — full chronological narrative, every command run,
  every intermediate result, going back to the original checkpoint C vs.
  D=384 quality-gap question that started this. This document is a
  structural summary of that log's later sections; the log has strictly
  more detail if anything here needs expanding.
- Real captured seed used for the sampled-mode divergence: `0x42da8a1f`,
  prompt "once upon a time," checkpoint `fabric/export_stepC_d128_v16384/goformer.npz`.
- `kevgpt_interactive/main.c`'s `KEVGPT_DEBUG_SEED` (always on) and
  `KEVGPT_FORCE_GREEDY`/`KEVGPT_DIAG_DUMP_HEAD` (both off by default,
  documented in-place) diagnostic instrumentation — reusable for any
  follow-up real-hardware capture.
- `model/tinystories_hf_repro/hw_vs_sw_report.html` (published as the
  "Silicon Fidelity" artifact) — the story-by-story sample evidence behind
  the fixation-word pattern, with real captured seeds shown per sample.
