# The Fixation-Word Investigation: a reassessment

A reassessment of six weeks chasing a real-hardware-only text corruption bug in
the Genesys2 board's multi-master DDR3 path — what was tested, what was
actually found, and where the evidence points next. Companion document to
`FIXATION-WORD-CDC-INVESTIGATION.md` (the full blow-by-blow this report
synthesizes) and `model/SCALE-UP-LOG.md` (the longer chronological narrative).

**Status: open, not root-caused. Reassessed 2026-09-13.**

## Executive summary

Checkpoint C's real Genesys2 deployment occasionally substitutes a nonsense
word — "care," "cardinal," "chug" — for what the model actually computed,
sometimes derailing an entire reply into a repetition loop. It's fully
deterministic per build (same checkpoint, prompt, seed, bitstream →
byte-identical output every time) but disagrees with the Python golden
reference the RTL is supposed to implement bit-exactly.

Six weeks of investigation have ruled out the checkpoint, the RTL's own
compute logic, the weight-packing pipeline, the UART→DDR3 write path, the
tokenizer table, and — after real setbacks and real fixes — both of the two
leading electrical hypotheses (a CDC timing-constraint gap, and an
owner-tracking FIFO race). Along the way, two genuine, previously-invisible
hardware bugs were found and fixed, and are worth keeping regardless. **The
root cause remains unknown.**

| | |
|---|---|
| Hypotheses / traffic sources tested and ruled out or fixed defensively | **9** |
| Confirmed as the actual cause | **0** |
| Full clean resynth → bitstream → real-hardware rebuild cycles (~56 min each) | **2** |
| Real, previously-invisible production RTL bugs found and fixed | **2** |

The strongest lead in the whole file — that fixation words statistically
cluster in one specific weight-bank row range — has never been directly
pursued. That's this reassessment's headline recommendation (§7).

## 1. The symptom

Real hardware diverges from software at a specific decode step. Two captured
instances show the same defect at wildly different severities — itself a
clue.

- **Sampled mode, seed `0x42da8a1f`:** golden picks "went" (rank 1); hardware
  picks "saw" (rank 2). Logit margin 0.13% — a near-tie, plausibly a tiny
  numerical discrepancy.
- **Greedy mode, no sampling noise:** golden picks "." (logit 11.95); hardware
  picks "care." Golden rank of "care": **15,118 of 16,384** — not a near-tie,
  a gross, unambiguous wrong answer.

Both are 100% reproducible within one build — 40 repeated trials (5 prompts ×
8 repeats, greedy) produced byte-identical output every time, fixation words
included, across three separate builds tested this way. This rules out
random/probabilistic timing noise in the colloquial sense: whatever this is,
it's a deterministic function of (checkpoint, prompt, seed, *this specific
bitstream*).

**The recurring word.** Rank-analyzing five fresh first-token divergences
against the golden reference found that three of five converge on the exact
same token — `id 2213`, "care" (or its stem-relative "carefree") — with
massive rank misses, not close calls:

| Prompt | Hardware's 1st token | Golden's rank for it |
|---|---|---|
| "the wizard cast" | "the" | 1 of 16,384 (near-tie) |
| "in the forest" | "care" (id 2213) | 2,551 of 16,384 |
| "my favorite toy" | "carefree" (id 2216) | 9,311 of 16,384 |
| "the rocket ship" | "care" (id 2213) | 8,149 of 16,384 |
| "once upon a time" | exact match | 0 (correct) |

This isn't a new phenomenon — it's the same word real story captures have
shown for weeks ("care for the little girl," "care for you care"). "care"/id
2213 is the single most-reproduced symptom in this entire investigation, not
a generic "wrong word sometimes wins" pattern.

## 2. The traffic path under suspicion

Weight reads don't go straight from the loader to DDR3 — they cross two
clock domains and share the physical memory controller with the CPU's own
traffic:

```
weight_loader_ddr.sv (gen_clk, ~50MHz)
    |  CDC crossing: async_fifo_gray x2 (read-request, read-return)
    v
mig_read_mux2.sv (ui_clk)   -- merges weight vs. KV reads
    v
mig_read_engine.sv (ui_clk)
    v
mig_dual_master_arbiter.sv (ui_clk)  -- merges kevgpt vs. cpu_ddr_bridge
    v
Physical MIG / real DDR3
```

Two structural facts made this path suspicious from the start: it crosses
clock domains through six `async_fifo_gray` instances no simulation gate
exercised for months, and its "owner" FIFOs (which track which requester a
returned DDR3 beat belongs to) had an unconnected backpressure signal —
either one, if broken, could silently misroute or corrupt a read without
ever touching the correctly-written DDR3 bytes underneath it.

## 3. Everything ruled out

In the order it was checked. Nine independent candidates, each closed with
direct evidence — not by elimination alone.

| Candidate | Verdict | Evidence |
|---|---|---|
| Checkpoint / trained weights | Clean | Golden-reference logits for fixation words rank ~3,000–16,000/16,384 — never competitive under correct computation. |
| RTL compute logic (both configs) | Clean | Bit-exact vs. golden, greedy and sampled, arbitrary seeds — including the exact real seed that produced a captured "care," extended 53 tokens deep. |
| Weight-packing pipeline | Clean | Transmitted word list byte-for-byte identical (2,670,592/2,670,592 words) to simulation's own ROM image. |
| UART → DDR3 storage | Clean (narrower than first claimed) | Zero mismatches across 8,192 words read straight from DDR3 — but via a CPU load that bypasses the entire CDC/arbiter path under suspicion. Proves the write, not the read-back. |
| Tokenizer ID→string table | Clean | DDR3-resident table byte-identical to a fresh build; decodes every suspicious id correctly. |
| `cpu_ddr_bridge` print-path traffic | Ruled out | Byte-identical hardware output with this traffic source removed vs. present, across 5 prompts, two independent hardware reloads. |
| `async_fifo_gray` (the CDC primitive itself) | Clean | Audited against the canonical async-FIFO design — Gray-code math, synchronizer structure, full/empty formulas all textbook-correct. |
| Sampling-methodology mismatch (own test artifact) | Ruled out | Rerun with the RTL's exact Gumbel algorithm, 1,500 tokens, zero fixation-word hits. |
| Pure random / probabilistic silicon noise | Narrowed | 100% deterministic within one build across three separate 40-trial sweeps — rules out randomness, not a live race. |

## 4. The two hypotheses chased hard

Everything above was cheap to rule out. These two took real Vivado time,
real hardware rebuilds, and real new testbenches — and both came back
negative.

### A. CDC timing-constraint gap — ruled out, definitively

The investigation's original lead hypothesis. Found two real,
previously-invisible bugs: MIG's 200MHz reference clock had no top-level
`create_clock` anywhere in the actual build fileset — the correct line
existed, verbatim, in an orphaned file nothing referenced — meaning the
entire `gen_clk`/`ui_clk`/MIG/kevgpt domain had never had real static timing
analysis applied, in any build, ever. Once fixed, a full clean resynthesis
found every one of the six `async_fifo_gray` crossings running with
razor-thin **0.054–0.067ns hold margin** — on the data path feeding a DMA
command-address register directly, not just the Gray-pointer synchronizer.

Both were fixed, verified against the live implemented design, built into a
genuinely fresh bitstream (no incremental synthesis reuse, ~56 minutes
each), and re-tested on real hardware against the same 5-prompt greedy
sweep.

**Result: byte-for-byte identical wrong tokens, at identical positions, with
identical determinism, before and after the fix.** Fixing two real timing
gaps changed nothing observable. The fixes are still correct and worth
keeping — they close a genuine blind spot in this design's timing closure —
but they are not the answer.

### B. Owner-tracking FIFO race — fixed defensively, never confirmed active

The co-equal hypothesis raised by external review: two owner-tracking FIFOs
(which route a returned DDR3 beat back to whoever requested it) left their
`in_ready_o` backpressure signal unconnected. If a push were ever dropped,
ownership would silently shift by one entry — DDR3 data itself stays
correct, but a later request's *return* gets misattributed to the wrong
requester. This mechanism naturally explains why the two captured
divergences (§1) are such different sizes: a shifted index doesn't correlate
with numerical closeness the way a bit-flip theory would.

Wired real backpressure into both FIFOs, added independent push/pop
accounting with overflow/underflow assertions. Then stress-tested three
separate ways:

1. A from-scratch three-master contention testbench with randomized MIG
   latency and backpressure — found and fixed a real bug, but in the
   testbench's own read-completion detector, not the design; 11 seed sweeps
   clean afterward.
2. A real-hardware ILA watching all 8 owner-FIFO invariant flags across 111
   generations while the fixation-word symptom fired pervasively in the
   replies.
3. A permanent always-on hardware health monitor (no ILA/JTAG needed)
   watching the same invariants across 96 more real generations.

**Result: zero violations, across all three independent checks.** The fix is
real and worth keeping, but nothing confirms it was ever the active defect
on real hardware.

### C. Weight-bank CRC diagnostic, real DMA path — built, deployed, comparison inconclusive

The most direct test on paper: tap a CRC32 checksum onto the actual
`weight_loader_ddr` DMA write port (not the CPU-bypass read used elsewhere),
so a real hardware capture can be diffed against an independently-computed
expected value. Built, verified standalone against `zlib.crc32()`,
resynthesized clean, deployed to real hardware — captured `0x0086427c` for a
fixed prompt in forced-greedy mode.

The matching simulation run initially looked like a 1-token divergence from
a previously-documented reference sequence — investigated carefully rather
than taken at face value: a byte-identical control build (no CRC tap at all)
reproduced the exact same divergent token, which rules out the new
diagnostic and simulator nondeterminism as the cause. Traced instead to the
reference sequence simply predating a real checkpoint swap six days earlier
— same model shape, different trained weights, enough to flip one
near-tied decision. **No bug found here.** The real hardware-vs-simulation
CRC comparison itself is still open — blocked on a generation-length
mismatch (free-running counter, different reply lengths) that hasn't been
resolved yet.

## 5. Real bugs found along the way

Neither explains the symptom, but both are genuine, previously-invisible
defects worth having fixed regardless of how this investigation ends.

- **Missing root clock constraint for the DDR3 reference clock** — fixed &
  verified on real hardware. `clk_200mhz_p`, the root of the entire
  MIG/kevgpt clock tree, had never been given a `create_clock` in this
  target's real XDC fileset. Net effect: essentially the whole chip outside
  JTAG/SPI had zero real static timing analysis applied, in any build, on
  this board, ever.
- **Razor-thin CDC hold margins across the entire weight/KV crossing
  scheme** — fixed & verified on real hardware. Every `async_fifo_gray`
  instance in the DMA bundle, both directions, showed 0.054–0.067ns hold
  margin against a 10ns `ui_clk` period — including on the FIFO's own
  data-memory path feeding straight into a DMA command-address register.
  Closed with three scoped `set_max_delay -datapath_only` exceptions
  (worst remaining margin after the fix: 0.108ns).

Smaller, but worth knowing about:

- **Testbench clock-domain mismatch in the DDR-bundle gate** (test-code
  only) — a CDC was added to the real design for the weight-loader's
  read-return path; the verification testbench's own clock wiring was
  never updated to match, occasionally double-popping the model's CDC FIFO.
  Real hardware never had this mismatch, but the gate proving "two-master
  DMA sharing is correct" had been silently unrunnable for an unknown
  stretch of time.
- **Stale-counter race in the contention testbench's own completion
  detector** (test-code only) — a level-check against a repeatable counter
  could see a stale match left over from a previous read and sail past one
  that had barely started, hanging the testbench forever on certain random
  seeds. Fixed by switching to an edge-triggered done pulse.
- **Icarus Verilog can't parse this codebase's concurrent-assertion
  syntax** (toolchain gap) — a flat parser limitation in the local Icarus
  install, affecting four files added in the weeks before this
  investigation. Whatever last reported these gates "PASS, clean compile"
  did not use this toolchain, or predates those assertions.

## 6. Reassessment: where this actually stands

Read plainly, the pattern across §3 and §4 is not "one hypothesis
confirmed, effort well spent" — it's **nine consecutive negative results**,
two of them expensive (full clean FPGA rebuilds, real-hardware ILA
bring-up, a from-scratch contention testbench). That's a legitimate,
rigorous process — every negative closed with direct evidence, not
assumption, and it produced two real fixes worth keeping. But it also means
the investigation has now exhausted every hypothesis anyone has proposed
for *how traffic moves through the DMA path*, and none of them explain the
symptom.

That's worth sitting with rather than immediately generating a tenth
timing/ordering hypothesis. Three observations stand out on rereading the
whole trail together:

1. **The symptom is oddly specific for an electrical bug.** A CDC
   metastability event or a misrouted FIFO entry should land on an
   arbitrary row, roughly uniformly, with severity uncorrelated to
   anything. Instead, one specific token (id 2213, "care") recurs across
   independent captures spanning weeks, and rank-misses cluster in one
   address range. Electrical races don't usually have a favorite word.
2. **The one build-dependent result left standing points at firmware
   timing, not fabric.** §2a's own finding — a diagnostic-only firmware
   change shifted which wrong token won, across builds sharing one
   identical bitstream — survived the CDC ruling-out (it can't be a
   placement/routing effect by construction) but was never chased on its
   own terms. It's real evidence for *some* live sensitivity to execution
   timing; nothing since has gone back to isolate what specifically.
3. **Every test so far assumed the bug is in how requests move, not in
   what address gets computed.** The CDC and owner-FIFO hypotheses are
   both about *ordering/routing* of DMA transactions. Nothing in this
   investigation has yet directly audited the *address arithmetic* that
   decides which weight-bank row a given block/head/layer combination
   reads from — the one part of the real path that would produce exactly
   this profile: a specific, reproducible, wrong-but-plausible row, not
   scattered noise.

## 7. The clue nobody's chased

An early raw-DDR3 diagnostic, run to check the UART write path,
incidentally flagged that fixation words statistically cluster in
**weight-bank rows 2048–4095** — a range that covers id 2213 ("care")
itself. This was noted once, in passing, while confirming an unrelated
claim, and has never been the actual subject of an experiment.

Every hypothesis tested since has been about *transport* — CDC timing,
FIFO ownership, contention. None has asked the more basic question this
clue actually points at: **what is structurally special about that specific
row range in `weight_bank_tdp`'s address space?** A page/burst boundary in
the DDR3 controller, a modulo pattern in how block/head/layer indices
compute a row address, an off-by-N in `weight_loader_ddr`'s drain-counter
logic that happens to alias into that range under some but not all (block,
layer) combinations — any of these would produce exactly this
investigation's evidence profile: a specific, reproducible, non-random
wrong row, invisible to every timing/ordering check run so far because it
isn't a timing or ordering bug.

## 8. Recommended next steps

1. ~~**Audit the row-2048–4095 address computation directly.**~~ **Done,
   2026-09-13 — result: clean, direct negative for corrupted weight
   data.** Reading turned up that `weight_loader_ddr.sv` has no per-load
   address arithmetic at all in the deployed streaming mode (every
   reload target is a fixed compile-time constant) — so instead of
   auditing arithmetic that doesn't exist, built a direct readback tap
   on `weight_bank_tdp`'s own otherwise-unused port A, verified it in
   simulation (catching and fixing a real bug in the tap itself — an
   unhandled DP=1 column-parity split that returned the wrong row's data
   for every odd address), then deployed to real hardware. Captured
   vocab id 2213's ("care") real, DMA-streamed weight row immediately
   after a live reproduction of the fixation symptom itself — **exact
   bit-for-bit match** against the known-correct source. The wrong
   weights are not the mechanism, at least for this row, at this moment.
   A full clean rebuild for this also surfaced a new, previously-uncaught
   setup-timing violation (4 paths, worst −0.255ns) on the KV-cache
   read-return path — unrelated to this tap, not yet chased, worth its
   own follow-up. Full account in `FIXATION-WORD-CDC-INVESTIGATION.md`
   §8 item 8.
2. **Isolate §2a's firmware-timing sensitivity on its own terms.** The one
   still-unexplained build-dependent result (a diagnostic-only firmware
   change shifting which wrong token wins, same bitstream) was folded into
   the now-ruled-out CDC hypothesis and never independently chased. Bisect
   exactly which instruction/timing change between the isolation and
   logit-probe builds caused the shift.
3. **Close the CRC volume mismatch for a true end-to-end check.** Item 1's
   real-vs-simulation CRC comparison is still blocked on a
   generation-length mismatch between the free-running hardware counter
   and a short simulation run. Either cap generation length in firmware for
   this diagnostic specifically, or run a much longer matching-length
   simulation.
4. **Finish the print+guard `cpu_ddr_bridge` elimination.**
   `is_stem_repeat()`'s own two per-token DDR3 reads were deliberately left
   untouched in the earlier isolation experiment (§2a) because removing
   them changes what gets generated. Worth auditing that function's own
   logic directly for a bug, independent of timing — it's the one
   remaining untested consumer on the traffic path.

## 9. Evidence & artifacts

Full blow-by-blow, every command run, every intermediate result:
`fabric/genesys2/FIXATION-WORD-CDC-INVESTIGATION.md` (this report's source)
and `model/SCALE-UP-LOG.md` (longer chronological narrative) in the
`kev-gpt` repo.

Real-hardware verification volume tallied across this investigation:

| | |
|---|---|
| Independent 40-trial (5 prompt × 8 repeat) greedy determinism sweeps | **3×**, byte-identical every time |
| Real generations monitored for owner-FIFO invariants (111 via ILA, 96 via permanent monitor) | **207**, zero violations |
| Full clean synth→impl→bitstream rebuilds | **2×**, ~56 real Vivado minutes each |
| Degenerate-reply rate after an earlier, related Gumbel-noise temperature recalibration | **32% → 12%** (masks visibility, doesn't fix the cause) |

Diffs referenced in this report live across two repositories: `kev-gpt`
(docs, golden references, mirrored RTL/testbenches) and the vendored
X-HEEP SoC project `kevgpt-genesys2-soc` (real Vivado build, owner-FIFO
fixes, ILA/health-monitor RTL, firmware).

- kev-gpt — `fabric/genesys2/FIXATION-WORD-CDC-INVESTIGATION.md`
- kev-gpt — `fabric/genesys2/rtl/crc32_word.sv`, `fabric/stage3/rtl/sequencer_vec.sv`
- kevgpt-genesys2-soc — `hw/vendor/.../mig_dual_master_arbiter.sv`, `mig_read_mux2.sv`
- kevgpt-genesys2-soc — `hw/vendor/.../constraints/genesys2/constraints.xdc`

---

*Published version (with diagrams and status visuals):
https://claude.ai/code/artifact/f0b95221-de1e-430f-a3a8-9d6e6115702c*
