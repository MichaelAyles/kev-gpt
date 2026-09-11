# Fixation-word investigation: real-hardware-only text corruption, narrowed to an unverified CDC boundary

Status as of 2026-09-11: **open, not root-caused, not fixed.** This document is
the standalone reference for the whole investigation — everything needed to
either continue it or hand it off, without reconstructing the trail from
`model/SCALE-UP-LOG.md`'s chronological entries (which have the full blow-by-blow
if this summary needs expanding).

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
| UART reception → DDR3 write (`uart_load_blob` in `main.c`) | Clean | Built a raw-DDR3-readback diagnostic (`KEVGPT_DIAG_DUMP_HEAD`, off by default in `kevgpt_interactive/main.c`) that reads the suspect address range straight from DDR3 via a plain CPU load, bypassing the streaming FSM entirely. Zero mismatches across all 8,192 dumped words against the known-correct reference. |
| Tokenizer ID→string table | Clean | The DDR3-resident tokenizer blob on the board is byte-identical to a fresh build from `meta.json`; decodes every suspicious ID correctly (id 2048 genuinely is "buster," etc. — the words themselves are real, unremarkable vocabulary entries). |
| `async_fifo_gray.sv` (the CDC primitive itself) | Clean | Audited directly against Cummings' canonical async-FIFO design: Gray-code math, 2-FF `ASYNC_REG` synchronizer structure, and the full/empty detection formulas are all textbook-correct. The one deliberate deviation (registered `wr_full` instead of combinational, to break a real Vivado DRC LUTLP-1 loop) was hand-traced through a worked example and confirmed not to cause overflow. |
| Sampling-methodology mismatch (my own earlier test artifact) | Ruled out | Reran with the *exact* algorithm the RTL implements (`gumbel.GumbelRng`), not an approximate PyTorch proxy: 1,500 tokens, zero fixation-word hits. |
| Marginal/random real-silicon timing noise | Ruled out | The 40-trial repeated-greedy-decode test (above) is 100% deterministic — a genuinely random/probabilistic mechanism would show *some* variation and didn't. |

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

**A concrete code smell found along the way (not confirmed as the active
bug):** both `mig_dual_master_arbiter.sv`'s `u_rd_owner_fifo` and
`mig_read_mux2.sv`'s `u_owner_fifo` — the FIFOs that track which master a
pending DDR3 read return belongs to — leave `in_ready_o` unconnected:
```systemverilog
sync_fifo #(...) u_owner_fifo (
    .in_valid_i(owner_push_valid),
    .in_ready_o(),   // never checked
    ...
```
Hand-checked the sizing (`MAX_OUTSTANDING=16` vs. 32/64-deep owner FIFOs) and
it looks adequate under normal operation, so this is not confirmed active —
but it's a real, fragile pattern with zero overflow detection, sitting in
exactly the subsystem implicated by everything else.

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
  VOCAB=1900 vs. 32,768 words at today's VOCAB=16384). If the mechanism is
  what §6 hypothesizes, growing the model further will make the symptom
  *more* frequent, not less.
- **Trust in "PASS" real-hardware verdicts for this whole class of
  deployment.** Every prior "real hardware confirmed working" milestone for
  per-layer weight streaming (NLAYER=8 onward, 2026-08-26+) was verified with
  small sample sweeps and RTL-simulation bit-exactness — neither of which
  would have caught this (see §7 and the "why didn't this show up before"
  analysis in `SCALE-UP-LOG.md`'s corresponding section).

## 6. Leading hypothesis, and exactly where it stands

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
resolve — neither guessed pin path returns a clock object. Digging further:

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
`gen_clk` genuinely isn't a recognized clock.

## 7. Why this investigation cannot go further from here

Everything up to this point was resolvable through simulation, real-hardware
behavioral comparison (RTL sim vs. golden reference vs. real chat output),
and static file/log inspection — all things achievable without a human at
the controls. This last step is not:

- **Distinguishing "OOC-hidden but fine" from "genuinely unconstrained"**
  requires interactively driving Vivado (`report_clock_networks`, possibly
  `report_timing -through` specific `async_fifo_gray` synchronizer cells,
  or a fresh from-scratch synthesis with the OOC methodology deliberately
  disabled to see what surfaces) — exploratory, judgment-driven Vivado work,
  not a lookup.
- **Writing a constraint against the wrong theory is worse than writing
  none.** A `set_clock_groups` line that silently resolves to an empty
  `get_clocks` result doesn't error the build — it just does nothing, while
  looking exactly like a real fix in a diff. I caught my own first attempt
  doing exactly this by verifying against the live implemented design before
  committing it; that verification step is not optional for whoever
  continues this.
- **If the root cause is real CDC metastability**, closing it needs either
  the correct constraint (once the clock-naming question above is resolved)
  followed by a full re-synthesis/re-implementation/re-bitstream cycle and
  real-hardware re-test — genuinely time-consuming, real-tool work — or,
  if metastability is confirmed via `report_timing` to actually be present
  with negative margin, real signal-level instrumentation (an ILA on the
  synchronizer flip-flops) to observe it directly, which this project has
  working precedent for (the `ai_accel` CDC investigation on 2026-08-16 used
  exactly this technique successfully).

None of this is a dead end — it's a well-scoped, concrete next task. It's
just not one further code/log reading can finish.

## 8. Recommended next steps, in order

1. **Resolve the clock-naming question first.** Open the project
   interactively (not via `open_run` on an archived checkpoint) and run
   `report_clock_networks` plus `report_property [get_cells
   xilinx_clk_wizard_wrapper_i/xilinx_clk_wizard_i/clk_wiz_0/inst/mmcm_adv_inst]`
   to find whatever `gen_clk`/`ui_clk` are actually named (if anything) once
   OOC block clocks are correctly surfaced at the top level.
2. **If they're unconstrained**: add the correct `create_clock`/
   `set_clock_groups -asynchronous` pair, matching the JTAG precedent at
   `constraints.xdc` line 18, then re-synthesize/re-implement from scratch
   (not an incremental `reset_run`, since this is a constraint-set change
   that could shift placement/routing everywhere) and check whether
   previously-"passing" CDC paths in `kevgpt_ddr_bundle.sv` now report real
   (possibly negative) slack.
3. **If real negative slack shows up**: that's the confirmed root cause.
   Fixing it is a genuine synchronizer-design problem (more synchronizer
   stages, a different CDC scheme, or slowing the crossing down) — a
   separate, follow-on piece of work.
4. **If timing is clean even after proper declaration**: the CDC hypothesis
   is falsified, and the investigation should return to the two remaining
   candidates from §4 — the unconnected `in_ready_o` owner-FIFO backpressure
   gap, or something not yet considered — with a real ILA capture on the
   physical board as the next diagnostic (this project's own precedent for
   exactly this class of problem, `ai_accel`'s 2026-08-16 investigation, is
   worth reading directly before repeating that work from scratch).

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
