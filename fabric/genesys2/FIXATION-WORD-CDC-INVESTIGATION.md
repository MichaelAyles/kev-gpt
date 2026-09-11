# Fixation-word investigation: real-hardware-only text corruption in the multi-master DDR read path

Status as of 2026-09-12: **open, not confirmed-fixed on real hardware, but
the CDC timing-constraint gap is now root-caused down to specific paths with
razor-thin real margin.** Four concrete, verified pieces of progress landed
since the previous revision: (1) §8 item 2's owner-FIFO backpressure gap is
now actually fixed in both `mig_dual_master_arbiter.sv` and
`mig_read_mux2.sv`, not just proposed — see §4's status update; (2) while
building the gate to verify that fix, a second, *separate* real bug was found
and fixed — a testbench-only clock-domain mismatch that had been silently
breaking `tb_kevgpt_ddr_bundle.sv` for weeks (see §4a); (3) §8 item 3's
weight-traffic-only isolation experiment was run on real hardware (partial/
cheap version — see §2a) and, in the process, produced the most precise
real-hardware evidence this investigation has yet captured: a specific,
repeatable wrong-token pick ("care"/"carefree") with a real-hardware/build-
dependent twist pointing at a timing mechanism; (4) **§6's clock-naming
question is resolved, and it's now the leading hypothesis by a wide
margin**: `clk_200mhz_p`, MIG's DDR3 reference clock and the root of the
entire `gen_clk`/`ui_clk` domain, was confirmed via a live query against the
real implemented design to have **never been given a `create_clock`
anywhere in this target's actual XDC fileset** — 67,133 clock endpoints,
effectively the whole non-JTAG/SPI portion of the chip, with zero real
static timing analysis ever applied, in any build that has run on this
board. Fixed with one line (a correct constraint already existed, orphaned,
in a sibling file never added to the build), and a full clean resynthesis
+ reimplementation + rebitstream cycle run against it (~56 minutes) found
**every `async_fifo_gray` CDC crossing inside `kevgpt_ddr_bundle.sv` — both
directions, all four instances — sitting at 0.054–0.067ns hold margin**, on
paths that include the crossing FIFOs' own data-memory output feeding
directly into the DMA engines' command-address/data registers, not just
the Gray-pointer synchronizer stages. This is systemic (every crossing, not
one outlier), on paths that directly determine DMA address/data values, and
margins this thin are exactly what this project's own JTAG-CDC history
(§6) already documents as fragile enough to flip negative under placement
changes from unrelated parts of the design — a coherent, physical
explanation for §2a's build-dependent-winner finding. Not yet fixed or
verified on real hardware — see §6/§8 item 4. §4a is explicitly ruled out
as a hardware explanation, since it lives entirely in test code. This
document is the
standalone reference for the whole investigation — everything needed to
either continue it or hand it off, without reconstructing the trail from
`model/SCALE-UP-LOG.md`'s chronological entries (which have the full
blow-by-blow if this summary needs expanding).

**Revision note:** this document's first version over-weighted the CDC
timing-constraint gap (§6) as *the* leading hypothesis. An external review of
that draft (summarized in §6a) made a strong case that an unsafe owner-tracking
FIFO in the multi-master DMA path is at least as likely a cause, is more
consistent with the *severity variance* between the two captured divergences
(§2), and is far cheaper to test. §§3, 6, 7, and 8 below have been corrected
and re-prioritized accordingly; nothing was deleted, only re-weighted. This
revision adds §4a and updates §4/§8 item 2 with the results of actually doing
that work.

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

## 2a. Real-hardware isolation experiment: "care" identified precisely, then a build-dependent twist

§8 item 3 called for disabling `KV_DDR_BACKED`/`cpu_ddr_bridge` traffic
entirely and re-running the greedy test — infeasible as scoped (checked
before spending real Vivado time: `KV_DDR_BACKED=0` overflows BRAM to
~116% for checkpoint C's shape, and `cpu_ddr_bridge` turns out to carry
traffic on *every* generated token even outside any diagnostic, via
`KEVGPT_ITOS`'s tokenizer-string read in `main.c`). Ran the cheap partial
version instead: a new off-by-default firmware toggle,
`KEVGPT_PRINT_IDS_ONLY` (`main.c`), skips `print_word_token()`'s own
`cpu_ddr_bridge` read (prints the raw numeric id instead of the decoded
string) while deliberately leaving `is_stem_repeat()`'s two per-token
`cpu_ddr_bridge` reads untouched, since those feed the repetition guard's
actual pick and touching them would change what gets generated. Net effect:
2 `cpu_ddr_bridge` reads/token instead of 3, not full elimination.

**First attempt at this toggle used `printf()` for the id print and silently
wedged the console after one reply** — libc stdio buffering vs. this file's
otherwise-universal raw `uart_putc()` — nothing to do with the DMA path
under investigation; fixed by hand-formatting the decimal digits through
`uart_putc()` directly, matching the rest of the file's convention (see the
comment at `print_word_token()` in `main.c` for the full account, kept in
place as a warning against mixing `printf()` into this file's hot per-token
path again).

**Result — clean negative for the print-path hypothesis specifically**:
with the fixed toggle, real-hardware output is **byte-for-byte identical**
between this reduced-traffic build and a baseline build (same
`KEVGPT_FORCE_GREEDY=1`, `KEVGPT_PRINT_IDS_ONLY=0`) across all 5 test
prompts, confirmed via two independent real weight+tokenizer reloads
(~15 minutes each). Reducing `cpu_ddr_bridge` print-path traffic did not
change the generated tokens at all. §3's table gets a new row for this.

**But rank-analyzing each divergence against the Python golden reference
turned up the sharpest evidence this investigation has captured**. Using
`IntKVQSequencer`'s own real-valued logits at the first generated position
for 5 prompts ("the wizard cast", "in the forest", "my favorite toy", "the
rocket ship", "once upon a time"):

| prompt | hardware's 1st token | golden's rank for it |
|---|---|---|
| "the wizard cast" | "the" | rank 1 of 16384 (golden's #2, logit 8.74 vs 9.05 — a near-tie) |
| "in the forest" | **"care"** (id 2213) | rank **2551** of 16384 |
| "my favorite toy" | **"carefree"** (id 2216) | rank **9311** of 16384 |
| "the rocket ship" | **"care"** (id 2213) | rank **8149** of 16384 |
| "once upon a time" | exact match | rank 0 |

Three of five prompts converge on the *same specific token* — "care" or its
stem-relative "carefree" — as the very first generated word, with massive
rank misses, not close calls. This is not a new phenomenon: it's the exact
word §2's own original greedy-mode capture picked (rank 15,118, quoted
above) and the word that recurs across multiple earlier real-hardware story
captures in `PORT-NOTES.md` ("care for the little girl," "care for his
family," "care for you care"). This confirms "care"/id 2213 specifically —
not a generic "wrong word sometimes wins" pattern — is the investigation's
single most-reproduced symptom, now caught at the earliest possible decode
position (token 0) and precisely rank-characterized for the first time.
Confirmed byte-identical across both the isolation and baseline builds
above (independent evidence the wrong pick isn't itself print-path-related).

**The complication**: a third build — identical generation logic, plus a new
`KEVGPT_DIAG_LOGIT_PROBE` toggle that reads the raw Q6.25 head logit for
ids 2213/2216/the-actual-winner via `kevgpt_read_bank()`, inserted *after*
the first token is already decided — produced a **different** winning token
for the same 3 prompts ("in the forest" and "the rocket ship" both won with
"." instead of "care"; "my favorite toy" won with a different id instead of
"carefree"). "the wizard cast" and "once upon a time" were unaffected. This
new build's own result was perfectly reproducible (byte-identical across 2
back-to-back trials on the same boot, ruling out live per-call flakiness),
but differs from the isolation/baseline builds' shared result.

The inserted diagnostic code cannot causally affect *this* token's value —
it only executes after `kevgpt_step()` has already returned it. The
remaining explanation is indirect: any firmware change shifts instruction
addresses and timing throughout the whole binary, including during prefill/
weight-prefetch *before* generation starts, which happens on every layer
per token (`PORT-NOTES.md`, "weight-window reloads happen once per layer
per TOKEN"). If the real defect is a **live, timing-sensitive race**
(matching §6's CDC-gap hypothesis, or contention this investigation's
existing simulation gates never exercise since they're single-master or
lightly-loaded — see §4's status update), its outcome being sensitive to
incidental build-to-build timing shifts is exactly what would produce this:
deterministic within a boot, different across boots/builds, same class of
symptom, different specific corrupted value depending on exact timing. That
would make this evidence lean back toward §6 (timing) over a fixed,
boot-independent addressing bug, which should have reproduced identically
regardless of unrelated code elsewhere in the binary.

The raw Q6.25 magnitudes captured from the third build are not treated as
reliable on their own given the winner reassignment — worth re-collecting
against a build whose winner is independently confirmed stable first,
per §8's updated priority list.

## 3. What's ruled out, with direct evidence (in the order it was checked)

| Candidate | Verdict | Evidence |
|---|---|---|
| Checkpoint/weights | Clean | Golden-reference logits for all fixation words rank ~3,000–16,000/16,384 across test prompts — never competitive under correct computation. |
| RTL compute logic, fully-resident config | Clean | Bit-exact vs. golden reference, greedy and sampled, arbitrary seeds. |
| RTL compute logic, streaming config | Clean | Bit-exact vs. golden reference using the *exact real captured seed* from a hardware run that produced "care," extended to 53 generated tokens (previously untested that deep) — simulation predicts "went"/"."/whatever golden predicts, not what hardware produced. |
| Weight-packing pipeline (`write_mems_wideword`/`wrom_to_words`) | Clean | `send_weights.py`'s transmitted word list is byte-for-byte identical (2,670,592/2,670,592 words) to the RTL simulation's own `wrom.mem`. |
| UART reception → **DDR3 storage** (`uart_load_blob` in `main.c`) | Clean, but narrower than first claimed | Built a raw-DDR3-readback diagnostic (`KEVGPT_DIAG_DUMP_HEAD`, off by default in `kevgpt_interactive/main.c`) that reads the suspect address range straight from DDR3 via a plain CPU load. Zero mismatches across all 8,192 dumped words. **Correction: this reads DDR3 via a plain CPU load, which bypasses `weight_loader_ddr`, the CDC crossing, `mig_read_mux2`, and `mig_dual_master_arbiter` entirely.** It proves the bytes UART wrote into DDR3 are correct. It proves *nothing* about whether those bytes come back correctly through the real streaming-read path into `weight_bank_tdp` — which is exactly the path under suspicion in §4 and §6a. This was originally written up as "the write side is clean," which overstated what was actually tested. |
| Tokenizer ID→string table | Clean | The DDR3-resident tokenizer blob on the board is byte-identical to a fresh build from `meta.json`; decodes every suspicious ID correctly (id 2048 genuinely is "buster," etc. — the words themselves are real, unremarkable vocabulary entries). |
| `cpu_ddr_bridge` print-path traffic (§2a) | Ruled out | Real-hardware output byte-identical between a build with `print_word_token()`'s per-token `cpu_ddr_bridge` read removed (`KEVGPT_PRINT_IDS_ONLY`) and a baseline build with it present, across all 5 test prompts, two independent hardware reloads. Reducing this specific traffic source changed nothing. Does not clear `cpu_ddr_bridge`/`mig_dual_master_arbiter` contention generally — only this one traffic source (print-path reads); `is_stem_repeat()`'s own per-token reads were deliberately left in place (§2a) and remain untested in isolation. |
| `async_fifo_gray.sv` (the CDC primitive itself) | Clean | Audited directly against Cummings' canonical async-FIFO design: Gray-code math, 2-FF `ASYNC_REG` synchronizer structure, and the full/empty detection formulas are all textbook-correct. The one deliberate deviation (registered `wr_full` instead of combinational, to break a real Vivado DRC LUTLP-1 loop) was hand-traced through a worked example and confirmed not to cause overflow. This clears the FIFO's own logic; it says nothing about physical placement of the synchronizer flops or the actual clock relationship feeding them (§6a). |
| Sampling-methodology mismatch (my own earlier test artifact) | Ruled out | Reran with the *exact* algorithm the RTL implements (`gumbel.GumbelRng`), not an approximate PyTorch proxy: 1,500 tokens, zero fixation-word hits. |
| Marginal/random real-silicon timing noise | Narrowed, not ruled out — new evidence points back toward a timing mechanism | The 40-trial repeated-greedy-decode test (above) is 100% deterministic *within one build*. **Correction: this only rules out *pure random/probabilistic* noise, not CDC as a mechanism generally.** `gen_clk` is PLL-derived from `ui_clk`, so their relative phase can be extremely repeatable across power-on/reconfiguration — a synchronizer sampling too close to a transition on one specific, fixed phase relationship would reproduce the *same* deterministic failure every time on a given bitstream. Determinism narrows which CDC mechanisms are plausible; it does not clear CDC as a category. §2a adds a new data point in the same direction: the specific wrong token picked for 3/5 test prompts changed between two firmware builds whose only difference (a diagnostic read) cannot causally affect the value in question — consistent with a race whose outcome depends on incidental build-to-build timing, not a fixed boot-independent corruption. |

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

**Status: FIXED (defensive), not confirmed as the active bug.** Both
`u_rd_owner_fifo`/`u_wr_owner_fifo` in `mig_dual_master_arbiter.sv` and
`u_owner_fifo` in `mig_read_mux2.sv` now wire `in_ready_o` into real
backpressure — command acceptance (`app_en_o` / `req_valid`) is gated on the
relevant owner FIFO actually having room, so a command can no longer be
presented downstream that this arbiter/mux can't track the return of. Added
outstanding-request counters (independent push/pop accounting, cross-checked
every cycle against each FIFO's own `count_o`) plus `overflow_o`/`underflow_o`
assertions in both files, matching the shape §8 item 2 specified. Diffs:
`mig_dual_master_arbiter.sv` (`kevgpt-genesys2-soc` repo only — this module
is `ai_accel`-owned, not mirrored into `kev-gpt`) and `mig_read_mux2.sv`
(present in both repos, kept byte-identical).

Verified via `fabric/genesys2/tb/tb_kevgpt_ddr_bundle.sv` (the actual gate
that already existed for this — see §4a for why it needed real repair
first): Phase 1 (KV read/write through the full stack) passes identically
before and after this fix, 0 errors. The fix could not be positively
confirmed as *the* active defect, because — with `cpu_ddr_bridge` idle and
only one requester active per phase in this testbench — none of its owner
FIFOs ever came close to filling under this test's traffic pattern (hand-
checked: peak occupancy stayed in single digits against 32/64-deep FIFOs).
§8 item 5's full contention testbench is what would actually stress this
path enough to prove or disprove it as the real-hardware cause; this fix is
correct and cheap regardless of that answer, so it's in either way.

## 4a. A second, real bug found while repairing the verification gate itself

Starting §8 item 2's work required first *running* `tb_kevgpt_ddr_bundle.sv`
to have something to verify the fix against — and it turned out this gate
had been silently broken for weeks, independent of anything in §4:

**Toolchain gap.** This Icarus Verilog install (12.0 stable,
`iverilog -V`) cannot parse this codebase's `assert property (... disable
iff ...)` concurrent-assertion syntax at all — not a flag issue
(`-gsupported-assertions`/`-gno-assertions` make no difference), a flat
parser limitation, confirmed with a two-line minimal repro. Four files in
this gate's own dependency chain use that syntax (`mig_read_mux2.sv`,
`mig_read_engine.sv`, `mig_dual_master_arbiter.sv`, `sync_fifo.sv`) — all
added between 2026-08-16 and this session. Compiling this gate at all
required bracketing `` `define SYNTHESIS ``/`` `undef SYNTHESIS `` shims
around exactly those four files (stripping their `` `ifndef SYNTHESIS ``
assertion blocks for this local run only — real files untouched, and
`kv_bank.sv`/`weight_bank_tdp.sv` must never see `SYNTHESIS` defined, since
that flips them to a Xilinx `xpm_memory_tdpram` macro Icarus can't
elaborate). This means: **whatever machine last reported this gate's own
"PASS, clean compile" result did not use this Icarus install**, or used it
before these assertions existed. Worth flagging for whoever sets up CI or a
fresh dev machine for this repo — the gate harnesses assume Icarus SVA
support this specific package build doesn't have.

**The real bug, once the gate could actually run.** With the toolchain gap
worked around, Phase 1 (KV path) passed but Phase 2 (weight-loader path)
hung to timeout — reproduced identically with §4's fix both applied and
reverted, ruling that out as cause or cure. Traced precisely:
`weight_loader_ddr` correctly issued all 16 needed DMA beat requests
(`issue_cnt` reached `total_beats`), but `drain_cnt` permanently stalled at
112/128 words. Direct instrumentation of `u_wl_rd_ret_cdc` (the
`async_fifo_gray` CDC instance for the weight-loader's read-return path,
inside `kevgpt_ddr_bundle.sv`) at its own ports showed **10 real writes but
14 reads** — a 4-entry excess exactly equal to `CDC_FIFO_DEPTH`, the
textbook signature of a phantom-pop bug, not a pointer-math defect. (A
software scoreboard mirroring `mig_read_mux2`'s owner-FIFO push/pop order
was built first and found 0 mismatches across the whole run, ruling that
layer out before chasing this further downstream.)

**Root cause: stale testbench clock wiring, not a production RTL bug.**
`tb_kevgpt_ddr_bundle.sv` instantiated `weight_loader_ddr`/`weight_bank_tdp`
on `ui_clk`, with an explicit comment explaining why: at the time that
choice was made, `kevgpt_ddr_bundle.sv`'s weight-loader read port was "an
un-CDC'd wl_* pass-through" (the comment's own words), so any clock choice
for the far side was harmless. Sometime after that comment was written, a
real CDC (`u_wl_rd_req_cdc`/`u_wl_rd_ret_cdc`, `async_fifo_gray`) was added
for exactly this port — closing the gap that comment flagged as future work
— but the testbench's clock wiring for these two DUTs was never updated to
match. The result: `weight_loader_ddr`'s `rd_ret_ready` became an
unsynchronized signal crossing into `u_wl_rd_ret_cdc`'s `rd_clk_i` domain
from the *wrong* clock (`ui_clk`, 7ns, instead of `gen_clk`/`clk`, 10ns, with
no relationship between them) — occasionally causing the FIFO to register a
pop twice for what should have been one logical beat. **Real hardware never
had this mismatch**: `weight_loader_ddr`'s `clk` port is always `gen_clk` in
the actual deployed design, inside `sequencer_vec.sv`'s own hierarchy — this
was purely a testbench artifact.

**Fix and verification.** Reclocked `u_wb`/`u_wl_dut` (and the `ldn_cnt`
counter and Phase 2's stimulus/verification `@(posedge ...)` waits that
interact with them) from `ui_clk` to `clk`, matching real deployment.
Result: `KEVGPT_DDR_BUNDLE_VERDICT,PASS`, 0 errors across all 4 phases.
Confirmed independent of §4's fix (passes with that fix present or reverted
— the two bugs are unrelated). Diff: `fabric/genesys2/tb/tb_kevgpt_ddr_bundle.sv`
only (test code, `kev-gpt` repo).

**Why this matters for the investigation despite fixing nothing on real
hardware:** this gate exists specifically to prove "genuine two-master DMA
sharing through the real arbiter/mux/CDC stack is correct" — precisely the
claim §4's traffic-path diagram depends on. It had been unrunnable-or-failing
for an unknown but non-trivial stretch of time, meaning that claim was
unverified (not disproven, just untested) for as long as this gate was
broken. It's now restored to a real, passing gate, which is worth something
independent of whether either bug found here turns out to be the real
fixation-word cause — but it should not be read as evidence *toward* either
hypothesis; it neither confirms nor rules out §6 (the CDC constraint gap) or
the still-open question of whether §4's owner-FIFO gap was ever actually hit
on real hardware.

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

**Status update (2026-09-12): resolved, and it's the second, more alarming
reading.** Ran §8 item 4's procedure for real — `open_project`/`open_run
impl_1` on the actual live `.xpr` that produced the currently-deployed
bitstream (not an archived checkpoint), `report_clock_networks`, direct
pin/cell discovery instead of guessed hierarchy paths. Result:
**`clk_200mhz_p` — MIG's 200MHz DDR3 reference clock, the actual pin the
board's differential oscillator drives — sits under `report_clock_networks`'s
"Unconstrained Clocks" heading, with 67,133 clock endpoints and 232
non-clock endpoints.** That single missing root clock explains everything
this section couldn't previously resolve: it's not OOC-hidden-but-fine, it's
not a query artifact (confirmed via `check_timing`, which independently
reported the same 0-clock-source-for-this-domain picture before the fix),
and the "short Unconstrained Path Table" red herring makes sense now too —
Vivado's timing engine doesn't enumerate "unconstrained" paths for endpoints
that were never associated with any clock context at all; it only flags
paths that have *some* clock but a missing exception. `gen_clk`/`ui_clk`
never showing up as named clock objects was because they don't exist as
*top-level* clocks at all — the OOC IPs' own internal clock objects
(`clk_pll_i`, `clk_out1_xilinx_clk_wizard_clk_wiz_0_0` — confirmed via
direct query on `kevgpt_ddr_bundle`'s own sync-register cells) exist as
bookkeeping inside their own OOC scope, but were never linked to a real,
analyzed top-level clock tree because nothing upstream of them was ever
constrained.

**Root cause, and it's almost embarrassingly simple**: `clk_200mhz_p`'s
`create_clock` was never added to this target's actual XDC fileset. A
correct line already exists, verbatim, in an *orphaned* file in the same
directory (`mig_traffic_gen_top.xdc`, a leftover from an earlier standalone
MIG example-design bring-up) — but that file was never referenced by
`core-v-mini-mcu-fpga.core`'s `genesys2`/`genesys2_kevgpt` filesets (only
`pin_assign.xdc` + `constraints.xdc` + `ddr3.xdc` are). The pin-level
`PACKAGE_PIN`/`IOSTANDARD` constraints made it into `pin_assign.xdc`
correctly; the `create_clock` line got left behind.

**Fix applied and verified for clock-graph connectivity** (not yet for
timing closure or on real hardware): added
`create_clock -period 5.000 -name sys_clk_pin [get_ports clk_200mhz_p]` to
`constraints.xdc` (matching `mig_traffic_gen_top.xdc`'s already-correct
line exactly). Verified by applying it in-memory against the same live
implemented design and re-querying: `clk_200mhz_p` moved from
"Unconstrained Clocks" to "Constrained Clocks" immediately, and a follow-up
`check_timing` reported 0 `no_clock` / 0 `unconstrained_internal_endpoints`
/ 0 `multiple_clock` / 0 `generated_clocks`-not-connected-to-source —
Vivado's own generated-clock inference correctly derives the entire
downstream MIG-PLL → Clocking-Wizard → `gen_clk`/`ui_clk` chain
automatically once this one root clock exists; no additional
`create_generated_clock` lines were needed.

**What this does and doesn't prove**: this confirms the *entire*
`gen_clk`/`ui_clk`/MIG/kevgpt/`cpu_ddr_bridge` portion of the chip — which
is to say, essentially the whole design outside JTAG/SPI-slave — has never
had real static timing analysis applied to it in any build that has ever
run on this board.

**Update: the full clean rebuild is done, and it found real, systemic,
razor-thin CDC margins throughout `kevgpt_ddr_bundle.sv`'s entire DMA
crossing scheme.** Ran the full clean re-synthesis/re-implementation/
re-bitstream cycle this required (`AUTO_INCREMENTAL_CHECKPOINT` disabled
first — `synth_1` had it pointing at the *pre-fix* checkpoint, which would
have silently defeated the point; ~56 minutes real Vivado time, no `-jobs`
per this project's own known `launch_runs -jobs` hang risk). Post-build,
`sys_clk_pin` is confirmed constrained with the same 67,133/232 endpoint
counts, and the top-level `report_timing_summary` shows 0 failing setup/
hold/pulse-width endpoints design-wide.

That top-level "0 failing" number is misleading on its own, though — it's
computed per named "Path Group," and every clock downstream of `sys_clk_pin`
(`clk_pll_i`, `clk_out1_xilinx_clk_wizard_clk_wiz_0_0` — Vivado's own names
for the MIG-PLL and Clocking-Wizard outputs, i.e. `ui_clk`/`gen_clk`) is an
*auto-inferred* generated clock, never explicitly `create_clock`'d, and
lands in `report_timing_summary`'s separate "Other Path Groups Table" under
an `**async_default**` label showing a suspiciously clean aggregate (WNS
+11.120ns, 0/3264 failing) that a direct, explicit `report_timing -from
... -to ...` query on the *same* clock (confirmed non-duplicate: one
`clk_out1_...` clock object, `IS_GENERATED=1`, correctly linked
`MASTER_CLOCK=clk_pll_i`) flatly contradicts — reproducibly, two different
query constructions, same result: a real, VIOLATED -4.122ns setup path, 85
logic levels deep, inside the CPU core's own multiplier/FPU-operand
forwarding logic (`cv32e40px_xif_wrapper_i/.../id_stage_i`). This
discrepancy between the summary table and a direct path query was not fully
reconciled — worth understanding properly before trusting `report_timing_summary`'s
top-line numbers for this class of auto-inferred clock again — but it's not
load-bearing for what follows, because that specific violated path is
unrelated CPU-core logic, not `kevgpt_seq`'s own datapath.

**Directly auditing `kevgpt_seq`'s own hierarchy is where this lands
squarely on target.** `report_timing -to [get_pins ...]` scoped to every D
pin inside `sequencer_vec`/`kevgpt_ddr_bundle`/`weight_bank_tdp`/
`kv_bank_ddr`/`weight_loader_ddr`/`gemv_banked_resident_vec`/`vec_attn_w`
(8,803 pins) found **zero VIOLATED paths** — but the worst-margin paths,
setup and hold both, are all inside `kevgpt_ddr_bundle.sv`, and the hold
margins are startlingly thin: the 5 worst hold paths, **0.054ns to
0.067ns**, are *every* `async_fifo_gray` instance in the bundle (KV
read-request, weight read-request, KV write-packet, KV write-ack), in both
directions, and — critically — not confined to the "official" Gray-pointer
synchronizer stages. The single worst path
(`u_kv_rd_req_cdc/mem_reg_.../RAMC_D1/CLK` → `u_rd_engine/cmd_addr_q_reg[11]/D`,
0.054ns) is the CDC FIFO's own **data memory array**, read straight into
`mig_read_engine`'s **DMA command-address register** — not a redundant
pointer bit protected by the FIFO's own empty/full logic, but the actual
address value a weight/KV read command will use. The next four worst paths
repeat the same shape against `cmd_addr_q_reg`/`data_q_reg` for the other
three FIFOs, plus two genuine Gray-pointer synchronizer paths
(`wr_gray_q_reg` → `wr_gray_rsync1_q_reg`) at 0.060–0.067ns.

**This is the strongest, most complete explanation this investigation has
produced.** It's systemic (every crossing, both directions, not one
outlier), it's on paths that directly determine DMA address/data values
(not just synchronizer metastability that the FIFO's own protocol is
designed to tolerate), and margins this thin are exactly what the
project's own JTAG-CDC history (this same file, above) already documents
as fragile enough to flip negative under placement changes from *unrelated*
parts of the design — which is precisely the mechanism §2a's build-
dependent-winner finding needed: different firmware builds, different
overall floorplan/congestion, different specific margin that tips negative
first, different specific corrupted address, different specific wrong
token. It's also consistent with why "care"/id 2213 recurs so often rather
than corruption landing uniformly at random — this investigation's own
earlier raw-DDR3 diagnostic work already flagged "rows 2048–4095" (which
covers id 2213) as where fixation words statistically cluster; a corrupted
DMA address landing near a real request's address, rather than at a
uniformly random one, would predictably favor nearby rows.

**Not yet done**: actually fixing these margins (Gray-pointer-bus skew
constraints, `set_max_delay -datapath_only` on the specific FIFO-memory-to-
consumer paths identified above, or a deeper pipeline stage) and a
real-hardware re-test. This needs real RTL/constraint engineering, likely
iterative, and is the natural continuation of this section rather than a
one-line fix like the root-clock gap was. Raw report files (`timing_summary_after_fix.rpt`,
`kevgpt_setup_audit.rpt`, `kevgpt_hold_audit.rpt`, `timing_new_domain.rpt`)
are session scratch files, not committed — §9 has the exact queries to
reproduce them against the now-fixed `constraints.xdc`.

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
- ~~Wiring the owner FIFOs' `in_ready_o` into real backpressure and adding
  the outstanding-request/response/owner-occupancy accounting assertions
  (§4, §6a)~~ **done** — see §4/§4a.
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
item 4 can independently shed light on it as a side effect. **Item 2 is now
done** (see §4's status update and §4a) — left in place below, unrenumbered,
as the historical record of the plan and because item 5's full contention
testbench is still the right way to actually stress-test it. **Item 3 is
partially done** (§2a) — its own result (a build-dependent shift in which
wrong token wins, for a code change that can't causally affect the value)
is itself evidence favoring item 4 over item 5 as the next move: it points
at *timing*, which item 4 investigates directly, rather than at contention
volume/ordering, which item 5's testbench is built to stress. Left in
original order below since item 4 was already next regardless.

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
2. ~~Make the owner FIFOs' `in_ready_o` real backpressure, in both
   `mig_dual_master_arbiter.sv` and `mig_read_mux2.sv`~~ **DONE** — see §4's
   status update and §4a for the fix, the gate it was verified against, and
   the second bug found along the way. Outstanding-request accounting is in
   place in both files as independent push/pop counters cross-checked
   against each FIFO's own `count_o`, plus overflow/underflow/backpressure-
   violation assertions. Not yet exercised under real two-master contention
   (this gate's traffic pattern never filled either owner FIFO close to
   capacity) — that's item 5 below.
3. ~~Run the weight-traffic-only isolation experiment on real hardware.~~
   **Partially done, see §2a.** The full version (disable `KV_DDR_BACKED`)
   is not buildable for checkpoint C's shape (~116% BRAM); ran the cheap
   partial version instead (`cpu_ddr_bridge` print-path traffic removed via
   `KEVGPT_PRINT_IDS_ONLY`) — clean negative, byte-identical hardware output
   with and without it (§3's new table row). In the process, precisely
   characterized the "care"/"carefree" fixation pattern via rank analysis
   AND found that a third build's diagnostic-only change shifted which
   token wins for 3/5 prompts — see §2a for the full account and why that
   favors re-prioritizing item 4 below. Remaining unexplored: `is_stem_repeat()`'s
   own 2 reads/token were deliberately left untouched (decision-relevant,
   can't be removed without changing what gets generated) — full print+guard
   `cpu_ddr_bridge` elimination, and the `mig_dual_master_arbiter`/
   `mig_read_mux2` bypass bisections originally proposed here, remain open
   if the timing-hypothesis work below doesn't localize it first.
4. ~~In parallel, resolve §6's clock-naming question properly.~~ **Done —
   root cause found and a fix applied, not yet verified on real hardware.**
   `clk_200mhz_p` (MIG's 200MHz DDR3 reference clock) was confirmed via a
   live query against the real implemented design (`report_clock_networks`
   on `open_run impl_1` of the actual `.xpr`) to be genuinely unconstrained
   — 67,133 clock + 232 non-clock endpoints, essentially the entire
   `gen_clk`/`ui_clk`/MIG/kevgpt/`cpu_ddr_bridge` domain, never covered by
   real STA in any build. Root cause: the `create_clock` line for it exists,
   correctly, in an orphaned sibling file (`mig_traffic_gen_top.xdc`) that
   was never added to this target's actual XDC fileset. See §6's status
   update for the full account. Added
   `create_clock -period 5.000 -name sys_clk_pin [get_ports clk_200mhz_p]`
   to `constraints.xdc`, verified in-memory (moved to "Constrained Clocks",
   `check_timing` clean) against the live design before committing it to the
   file. **The full clean re-synthesis/re-implementation/re-bitstream cycle
   is now also done** (~56 minutes, `AUTO_INCREMENTAL_CHECKPOINT` explicitly
   disabled first) **and found the real thing**: every `async_fifo_gray`
   crossing inside `kevgpt_ddr_bundle.sv` — both directions, all four
   instances — has razor-thin (0.054–0.067ns) hold margin, on paths that
   include the FIFO's own data-memory output feeding straight into
   `mig_read_engine`/`mig_write_engine`'s command-address/data registers,
   not just the Gray-pointer synchronizer stages. See §6's status update for
   the full account, including a not-fully-reconciled discrepancy between
   `report_timing_summary`'s top-line numbers and a direct `report_timing`
   query for this class of auto-inferred clock (not load-bearing for the
   `kevgpt_seq`-hierarchy audit that found the thin margins, since that used
   direct queries throughout). **Now needed**: fix these margins (Gray-
   pointer-bus skew constraints, `set_max_delay -datapath_only` on the
   specific FIFO-memory-to-consumer paths identified in §6, or a deeper
   pipeline/register stage on the consumer side) and a real-hardware
   re-test of §2/§2a's greedy-mode divergence test. This is real,
   iterative RTL/constraint engineering, not a one-line fix like the root
   clock gap was.
   One adjacent question not investigated here: `spi_slave_clk_pin` has no
   `set_clock_groups` of its own (only `jtag_clk_pin` does, line ~18) — now
   that `sys_clk_pin` and its derived clocks are real, Vivado derives *some*
   implicit relationship between spi_slave and that whole domain too, the
   same class of gap this section's own header comment already documents
   for JTAG. Worth checking alongside the margin fix above.
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
  `KEVGPT_FORCE_GREEDY`/`KEVGPT_DIAG_DUMP_HEAD`/`KEVGPT_PRINT_IDS_ONLY`/
  `KEVGPT_DIAG_LOGIT_PROBE` (all off by default, documented in-place)
  diagnostic instrumentation — reusable for any follow-up real-hardware
  capture. The latter two are new this revision (§2a).
- §2a's real-hardware captures: 3 independent weight+tokenizer reloads
  (~15 min each) across 3 firmware builds (isolation, baseline, logit-probe)
  for the same 5 prompts ("the wizard cast", "in the forest", "my favorite
  toy", "the rocket ship", "once upon a time"), checkpoint C. Golden-
  reference rank/logit comparison computed via `IntKVQSequencer(kbits=8,
  vbits=8, rotate=False, divfree=True)` against `fabric/export_stepC_d128_v16384/goformer.npz`.
  Raw captured token-id streams, golden comparisons, and per-build results
  not committed to the repo (session scratch files) — rerun from the
  firmware toggles above plus the same prompts to reproduce.
- `model/tinystories_hf_repro/hw_vs_sw_report.html` (published as the
  "Silicon Fidelity" artifact) — the story-by-story sample evidence behind
  the fixation-word pattern, with real captured seeds shown per sample.
- `fabric/genesys2/tb/tb_kevgpt_ddr_bundle.sv` — the gate for §4/§4a's work.
  Verdict now `KEVGPT_DDR_BUNDLE_VERDICT,PASS`, 0 errors, all 4 phases.
- Diffs from this revision's work: `mig_dual_master_arbiter.sv` and
  `mig_read_mux2.sv` (owner-FIFO backpressure, §4/§8 item 2) in
  `kevgpt-genesys2-soc`; `mig_read_mux2.sv` (kept in sync) and
  `tb_kevgpt_ddr_bundle.sv` (clock fix, §4a) in `kev-gpt`.
- §6's clock-graph queries (Vivado 2022.2, batch-mode TCL against
  `hw/vendor/esl_epfl_x_heep/build/openhwgroup.org_systems_core-v-mini-mcu_1.0.5/genesys2_kevgpt-vivado/openhwgroup.org_systems_core-v-mini-mcu_1.0.5.xpr`,
  `open_run impl_1` — the real implemented design, Sep 6 02:23 build, matching
  the currently-deployed bitstream): `report_clock_networks` before the fix
  listed `clk_200mhz_p` under "Unconstrained Clocks" (67,133 clock + 232
  non-clock endpoints); after applying `create_clock -period 5.000 -name
  sys_clk_pin [get_ports clk_200mhz_p]` in-memory, it moved to "Constrained
  Clocks" and `check_timing` reported 0 `no_clock` / 0
  `unconstrained_internal_endpoints` / 0 `multiple_clock` / 0
  `generated_clocks`-not-connected. Query scripts and raw logs are session
  scratch files, not committed — rerun the same `open_project`/`open_run
  impl_1`/`report_clock_networks` sequence to reproduce.
- Diff: `constraints.xdc` (the `create_clock` fix, §6/§8 item 4) in
  `kevgpt-genesys2-soc`.
- The full clean rebuild that found the thin CDC margins: `reset_run
  synth_1` (with `AUTO_INCREMENTAL_CHECKPOINT` explicitly disabled first —
  it was pointing at the pre-fix checkpoint) → `launch_runs synth_1` →
  `reset_run impl_1` → `launch_runs impl_1 -to_step write_bitstream`, no
  `-jobs` (this project's own known hang risk with `launch_runs -jobs`),
  against the same `.xpr` as above. ~56 minutes real Vivado time. Post-build
  `report_clock_networks` reconfirms `sys_clk_pin` constrained with the same
  endpoint counts; `report_timing_summary`'s top-level numbers show 0
  failing setup/hold/PW design-wide but are not fully trustworthy for
  auto-inferred clocks (see §6's status update for the unreconciled
  `**async_default**`-table-vs-direct-query discrepancy). The `kevgpt_seq`-
  hierarchy-specific audit that found the real thin margins used direct
  `report_timing -to [get_pins ...]` queries scoped to every D pin inside
  `sequencer_vec`/`kevgpt_ddr_bundle`/`weight_bank_tdp`/`kv_bank_ddr`/
  `weight_loader_ddr`/`gemv_banked_resident_vec`/`vec_attn_w` (8,803 pins;
  real hierarchy paths discovered via `get_cells -hier -filter
  {ORIG_REF_NAME == <name>}`, not guessed), both `-delay_type max` (setup)
  and `-delay_type min` (hold), `-max_paths 15 -sort_by slack`. Worst hold
  path: `u_kevgpt_ddr_bundle/u_kv_rd_req_cdc/mem_reg_0_3_6_11/RAMC_D1/CLK`
  (clocked by `clk_out1_xilinx_clk_wizard_clk_wiz_0_0`, i.e. `gen_clk`) →
  `u_kevgpt_ddr_bundle/u_rd_engine/cmd_addr_q_reg[11]/D` (clocked by
  `clk_pll_i`, i.e. `ui_clk`), slack 0.054ns. Query scripts and raw report
  files are session scratch files, not committed — rerun the same
  `open_project`/`open_run impl_1`/`get_cells -hier`/`report_timing`
  sequence against the now-fixed `constraints.xdc` to reproduce (no need to
  redo the full resynthesis if the bitstream from this session is still
  the one loaded/available).
