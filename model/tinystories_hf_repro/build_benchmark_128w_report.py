"""Renders benchmark_128w_hw.json + benchmark_128w_reference.json (produced
by benchmark_hw_stories_128w.py / benchmark_reference_stories_128w.py) into
benchmark_128w_report.html: kev-gpt's real Genesys2 board vs. the published
SauravP97/tiny-stories-19M reference, both generating ~128-word stories.

    python model/tinystories_hf_repro/build_benchmark_128w_report.py
"""
import json

HW_PATH = "/home/tparng/kev-gpt/model/tinystories_hf_repro/benchmark_128w_hw.json"
REF_PATH = "/home/tparng/kev-gpt/model/tinystories_hf_repro/benchmark_128w_reference.json"
REPORT_PATH = "/home/tparng/kev-gpt/model/tinystories_hf_repro/benchmark_128w_report.html"

hw = json.load(open(HW_PATH))
ref = json.load(open(REF_PATH))

hw_avg = sum(s["word_count"] for s in hw) / len(hw)
ref_avg = sum(s["word_count"] for s in ref) / len(ref)
hw_flagged = sum(1 for s in hw if s["reason"])
ref_flagged = sum(1 for s in ref if s["reason"])
hw_rate_avg = sum(s["rate_tok_s"] for s in hw) / len(hw)

payload = {"hardware": hw, "reference": ref}
json_blob = json.dumps(payload, separators=(",", ":")).replace("</script", "<\\/script")

HTML = r"""<title>Long-Form Fidelity</title>
<style>
:root{
  --bg:#F5F6F8; --surface:#FFFFFF; --surface-2:#EEF0F3;
  --ink:#1C2129; --ink-muted:#5B6472; --ink-faint:#88909B;
  --border:#DDE1E7; --border-soft:#E8EAED;
  --accent-hw:#1E7F72; --accent-hw-bg:rgba(30,127,114,0.10); --accent-hw-bg-strong:rgba(30,127,114,0.16);
  --accent-ref:#5B4B8A; --accent-ref-bg:rgba(91,75,138,0.09); --accent-ref-bg-strong:rgba(91,75,138,0.15);
  --flag-mark-bg:rgba(201,138,31,0.28); --flag-mark-fg:#5C3F06;
  --fix-mark-fg:#8B4B2E; --fix-underline:#C48257;
  --clean:#2E7D4F;
  --shadow: 0 1px 2px rgba(28,33,41,0.05), 0 4px 14px rgba(28,33,41,0.05);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#14171C; --surface:#1B1F26; --surface-2:#20252D;
    --ink:#E7EAEF; --ink-muted:#9AA3B2; --ink-faint:#6C7684;
    --border:#2A2F38; --border-soft:#242932;
    --accent-hw:#4FD1C0; --accent-hw-bg:rgba(79,209,192,0.12); --accent-hw-bg-strong:rgba(79,209,192,0.20);
    --accent-ref:#B7A3EC; --accent-ref-bg:rgba(183,163,236,0.12); --accent-ref-bg-strong:rgba(183,163,236,0.20);
    --flag-mark-bg:rgba(240,180,41,0.30); --flag-mark-fg:#FCE3A6;
    --fix-mark-fg:#E3A177; --fix-underline:#C48257;
    --clean:#5FBE86;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 20px rgba(0,0,0,0.35);
  }
}
:root[data-theme="dark"]{
  --bg:#14171C; --surface:#1B1F26; --surface-2:#20252D;
  --ink:#E7EAEF; --ink-muted:#9AA3B2; --ink-faint:#6C7684;
  --border:#2A2F38; --border-soft:#242932;
  --accent-hw:#4FD1C0; --accent-hw-bg:rgba(79,209,192,0.12); --accent-hw-bg-strong:rgba(79,209,192,0.20);
  --accent-ref:#B7A3EC; --accent-ref-bg:rgba(183,163,236,0.12); --accent-ref-bg-strong:rgba(183,163,236,0.20);
  --flag-mark-bg:rgba(240,180,41,0.30); --flag-mark-fg:#FCE3A6;
  --fix-mark-fg:#E3A177; --fix-underline:#C48257;
  --clean:#5FBE86;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 20px rgba(0,0,0,0.35);
}

*{box-sizing:border-box;}
body{ background:var(--bg); color:var(--ink); font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif; line-height:1.5; }
::selection{ background:var(--flag-mark-bg); }
.wrap{ max-width:1180px; margin:0 auto; padding:56px 28px 80px; }

header.top{ margin-bottom:40px; }
.eyebrow{ font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:0.12em; text-transform:uppercase; color:var(--ink-faint); margin:0 0 10px; }
h1{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:clamp(32px,4.2vw,46px); margin:0 0 14px; letter-spacing:-0.01em; text-wrap:balance; color:var(--ink); }
h2.sec{ font-family:'Fraunces',serif; font-size:22px; font-weight:600; margin:0 0 14px; }
.dek{ font-size:16px; color:var(--ink-muted); max-width:70ch; margin:0 0 18px; }
.provenance{ font-family:'IBM Plex Mono',monospace; font-size:12.5px; color:var(--ink-faint); display:flex; flex-wrap:wrap; gap:6px 18px; border-top:1px solid var(--border-soft); padding-top:16px; }
.provenance b{ color:var(--ink-muted); font-weight:500; }

.caveat{
  background:var(--surface-2); border:1px solid var(--border); border-left:3px solid var(--accent-ref);
  border-radius:8px; padding:14px 18px; font-size:13.5px; color:var(--ink-muted); margin-bottom:32px; max-width:80ch;
}
.caveat b{ color:var(--ink); }
.caveat + .caveat{ margin-top:-20px; }

.modeltag{ display:inline-flex; align-items:center; gap:6px; font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; padding:3px 9px; border-radius:5px; letter-spacing:0.01em; }
.modeltag.hw{ background:var(--accent-hw-bg); color:var(--accent-hw); }
.modeltag.ref{ background:var(--accent-ref-bg); color:var(--accent-ref); }
.dot{ width:7px; height:7px; border-radius:50%; display:inline-block; }
.dot.hw{ background:var(--accent-hw); }
.dot.ref{ background:var(--accent-ref); }

.stats-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:12px; overflow:hidden; margin-bottom:18px; box-shadow:var(--shadow); }
@media (max-width:820px){ .stats-grid{ grid-template-columns:repeat(2,1fr); } }
.stat-tile{ background:var(--surface); padding:22px 20px; }
.stat-tile .sweep-label{ font-family:'IBM Plex Mono',monospace; font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:var(--ink-faint); margin-bottom:10px; }
.stat-tile .num{ font-family:'Fraunces',serif; font-size:30px; font-weight:600; line-height:1; font-variant-numeric:tabular-nums; margin-bottom:6px; }
.stat-tile .num small{ font-size:14px; font-weight:500; color:var(--ink-faint); }
.stat-tile.hw .num{ color:var(--accent-hw); }
.stat-tile.ref .num{ color:var(--accent-ref); }
.stat-tile .pct{ font-size:12.5px; color:var(--ink-muted); font-family:'IBM Plex Mono',monospace; }

.runbook{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px 26px; margin-bottom:44px; box-shadow:var(--shadow); }
.runbook ol{ margin:0; padding-left:22px; }
.runbook li{ margin-bottom:18px; font-size:14px; color:var(--ink-muted); }
.runbook li:last-child{ margin-bottom:0; }
.runbook li b{ color:var(--ink); }
.runbook pre{
  background:var(--surface-2); border:1px solid var(--border-soft); border-radius:8px; padding:12px 14px;
  overflow-x:auto; font-family:'IBM Plex Mono',monospace; font-size:12px; line-height:1.55; color:var(--ink);
  margin:8px 0 0;
}
.runbook code{ font-family:'IBM Plex Mono',monospace; font-size:0.93em; background:var(--surface-2); padding:1px 5px; border-radius:4px; }

.controls{ display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px; margin-bottom:22px; position:sticky; top:0; background:var(--bg); padding:14px 0; border-bottom:1px solid var(--border-soft); z-index:5; }
.toggle{ display:inline-flex; align-items:center; gap:8px; font-size:13.5px; color:var(--ink-muted); cursor:pointer; user-select:none; }
.toggle input{ accent-color:var(--fix-mark-fg); width:15px; height:15px; }
.legend-strip{ display:flex; gap:16px; align-items:center; font-size:12.5px; color:var(--ink-faint); flex-wrap:wrap; }
.legend-strip mark.flag{ background:var(--flag-mark-bg); color:var(--flag-mark-fg); padding:0 2px; border-radius:2px; }
.legend-strip mark.fix{ background:transparent; color:var(--fix-mark-fg); text-decoration:underline; text-decoration-style:dashed; text-decoration-color:var(--fix-underline); text-underline-offset:3px; }

.rowgroup{ margin-bottom:14px; }
.rowmeta{ font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--ink-faint); margin:22px 0 8px; display:flex; gap:14px; align-items:baseline; }
.rowmeta .seedno{ color:var(--ink-muted); font-weight:600; }

.pair{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--border-soft); border:1px solid var(--border-soft); border-radius:10px; overflow:hidden; }
@media (max-width:760px){ .pair{ grid-template-columns:1fr; } }

.panel{ background:var(--surface); padding:18px 20px 20px; display:flex; flex-direction:column; }
.panel-head{ display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; gap:10px; }
.flagpill{ font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600; letter-spacing:0.02em; padding:3px 8px; border-radius:20px; white-space:nowrap; }
.flagpill.clean{ background:rgba(46,125,79,0.12); color:var(--clean); }
.flagpill.flagged{ background:var(--flag-mark-bg); color:var(--flag-mark-fg); }
.reason{ font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--flag-mark-fg); margin-bottom:10px; line-height:1.4; }
.story{ font-family:'Literata',Georgia,serif; font-size:14.5px; line-height:1.68; color:var(--ink); flex:1; }
.story mark.flag{ background:var(--flag-mark-bg); color:var(--flag-mark-fg); padding:0 1px; border-radius:2px; }
.story mark.fix{ background:transparent; color:var(--fix-mark-fg); font-weight:600; text-decoration:underline; text-decoration-style:dashed; text-decoration-color:var(--fix-underline); text-underline-offset:3px; }
.panel-foot{ margin-top:12px; padding-top:10px; border-top:1px solid var(--border-soft); font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--ink-faint); display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px; }

footer.endnote{ margin-top:56px; padding-top:20px; border-top:1px solid var(--border-soft); font-size:12.5px; color:var(--ink-faint); max-width:74ch; }
#empty-state{ display:none; padding:60px 0; text-align:center; color:var(--ink-faint); font-size:14px; }
</style>

<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Literata:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">

<div class="wrap">

  <header class="top">
    <p class="eyebrow">Genesys2 &middot; ~128-word stories, real hardware vs. published reference</p>
    <h1>Long-Form Fidelity</h1>
    <p class="dek">kev-gpt's actual deployed weights on the
      (<span class="modeltag hw"><span class="dot hw"></span>real Genesys2 board</span>), 2-4M params, INT4/INT8
      quantized, streamed over UART, generating stories near the model's own 128-token context limit &mdash;
      set against <span class="modeltag ref"><span class="dot ref"></span>SauravP97/tiny-stories-19M</span>, a
      published 19M-parameter FP reference trained on the same TinyStories-style corpus. Every draw, not a
      curated subset.</p>
    <div class="provenance">
      <span><b>Detector</b> model.filter_synth_corpus.is_degenerate, min_bigram_recurrence=6</span>
      <span><b>Draws</b> 5 &times; 5 prompts, 25 total each side</span>
      <span><b>Hardware config</b> on-chip Gumbel-max, TEMP=0.45, MAX_GEN_LEN=124/STORY_SENTENCES=20 (benchmark
        override, see runbook)</span>
      <span><b>Reference config</b> temp=0.7, top_k=50, max_new_tokens=180</span>
      <span><b>Board weights</b> data/ckpt_stepC_d128_v16384.qat.pt (INT4/INT8, 2-4M params)</span>
      <span><b>Reference weights</b> SauravP97/tiny-stories-19M (FP, 19M params)</span>
    </div>
  </header>

  <div class="caveat">
    <b>This dataset caught two consecutive real bugs, both now fixed (2026-09-15).</b> An earlier capture
    (prompt "a little girl", real_seed <code>0x2d7acf1a</code>) read "...lily loved her <b>new new new new
    new</b> magnet..." &mdash; a repetition guard's own substitution logic never excluded <code>last_tok</code>,
    so it kept landing right back on the word it was trying to avoid. Fixed in <code>57ce25a</code> &mdash; but
    a second capture (prompt "the dog ran", real_seed <code>0x4361ec7e</code>) immediately surfaced a subtler
    version of the same class of bug: "the dog" recurring 10&times; because the FIRST fix's own re-validation
    loop could <b>oscillate</b> between two already-rejected candidates ("dog" &#8596; "cat") in stories with
    multiple genuinely recurring nouns, landing back on the original word by pass-count parity. Fixed in
    <code>4c3a398</code> by accumulating every rejected candidate across passes, not just the single most
    recent one. Both root-caused via a real-hardware guard trace (not RTL simulation, which structurally
    can't see this: the guards are pure firmware, never touch RTL) &mdash; full write-ups in
    <code>fabric/genesys2/PORT-NOTES.md</code>. The hardware panels below are a
    <b>fresh recapture after both fixes</b> (same 5&times;5 methodology): <b>0/25 flagged</b> at any
    <code>min_bigram_recurrence</code> threshold, down from 4-6/25 on the same methodology before this second
    fix. The original pre-fix samples remain preserved in <code>PORT-NOTES.md</code> and git history, not
    silently dropped.
  </div>
  <div class="caveat">
    <b>The 0/25-vs-5/25 gap is not a capability win.</b> It does not mean kev-gpt's 2-4M-parameter model is
    more fluent than the 19M-parameter reference. Three things stack in kev-gpt's favor here, and none of
    them speak to underlying model quality: (1) kev-gpt's output is actively repaired at inference time by
    three firmware repetition guards (mechanism and cost in the panel below) &mdash; the reference's output
    is raw, unedited HF <code>generate()</code> sampling with no equivalent correction; the fair comparison
    would be guards-on vs. guards-on, and the reference has no guards to turn on. (2) Length isn't matched
    (hw averages 108.6 words, reference 145.8 &mdash; more words means more chances for the detector to
    trip). (3) <code>min_bigram_recurrence=6</code> was calibrated by inspecting THIS benchmark's own
    hardware distribution (see the threshold caveat below), not chosen independently for both sides. A
    held-out check backs this up directly: this project's own earlier 5&times;5 sweep on ~60-word samples
    found the guarded firmware and an UNGUARDED baseline build statistically tied (7/25 vs 8/25 flagged)
    &mdash; the guards eliminate the specific narrow patterns they target, but do not change kev-gpt's
    underlying tendency to loop, which two separate follow-up attempts (60,000-iteration extended training,
    distillation from an 8.7&times;-larger D=384 teacher) also failed to close. Full record in
    <code>PORT-NOTES.md</code>'s "Repetition guards" section and this project's
    <code>project-kevgpt-word-vocab-quality-ceiling</code> history.
  </div>
  <div class="caveat">
    <b>Not an apples-to-apples parameter comparison.</b> kev-gpt is a 2-4M-parameter model that lives entirely
    in FPGA on-chip memory by design (README's own "compression is the joke and the optimization"); the
    reference is a conventional 19M-parameter FP model with no such constraint. This benchmark measures how
    each behaves at a length near <i>kev-gpt's own</i> hard context ceiling (<code>KEVGPT_TMAX=128</code>
    tokens) &mdash; not a fair fight on model capacity, a check of what kev-gpt's real hardware output looks
    like at the length it's actually built to produce.
  </div>
  <div class="caveat">
    <b>Threshold raised from 3 to 6 for this length.</b> The bigram-recurrence detector's usual
    <code>min_bigram_recurrence=3</code> (calibrated for ~60-word samples elsewhere in this directory)
    saturates at ~110-150 words &mdash; already documented in <code>quality_sweep_C_vs_reference.py</code>
    ("detector saturates here per the original investigation" for its own 250-token Sweep 2). At threshold=3
    this benchmark's first pass flagged 14/25 hardware and a full 25/25 reference samples, most on ordinary
    function-word reuse ("the girl" 3&times;, "it was" 3&times;), not real loops. 6 was picked by inspecting
    this benchmark's own actual max-bigram-count distribution: hardware samples split cleanly into a 2-4 bulk
    (incidental reuse) and a 6/8/10/12 tail (genuine repetition, e.g. "the dog" recurring 12&times; in one
    reply) &mdash; 6 is the natural cut between them, and it brings both sides down to a comparable, no-longer-
    saturated rate (below). Fixation-term counts (kev-gpt-specific non-sequitur words this project has tracked
    since the fixation-word investigation) remain the more targeted signal regardless of threshold.
  </div>

  <div class="stats-grid" id="stats-grid"></div>

  <div class="runbook">
    <h2 class="sec">Reproduce this benchmark</h2>
    <ol>
      <li><b>Build a benchmark firmware image.</b> kevgpt_interactive's normal chat config caps replies at
        <code>MAX_GEN_LEN=60</code> word tokens (~45-55 words) &mdash; too short for a 128-word target. Both
        constants are <code>#ifndef</code>-guarded in <code>kevgpt_interactive/main.c</code> specifically for
        this (search "BENCHMARK OVERRIDE"):
        <pre>\
# in ~/RVchatbot/kevgpt-genesys2-soc/sw/applications/kevgpt_interactive/main.c:
#define MAX_GEN_LEN 124u      // was 60u -- headroom under KEVGPT_TMAX=128 for a short prompt
#define STORY_SENTENCES 20u   // was 8u -- generous, so the token budget (not sentence count) binds

cd ~/RVchatbot/kevgpt-genesys2-soc
make app PROJECT=kevgpt_interactive TARGET=genesys2</pre>
      </li>
      <li><b>Reload the board.</b> Start the weight+tokenizer listener <i>first</i>, wait for it to print
        "waiting for KEVGPT_UART_READY", <i>then</i> trigger the GDB reload &mdash; reversing this order
        causes <code>SEND_WEIGHTS_FAIL</code>. Regenerate the DDR3 tokenizer blob first if it doesn't already
        exist:
        <pre>\
python -m fabric.genesys2.gen_chat_fw \
  --npz fabric/export_word16384/goformer.npz --meta data/word_v16384/meta.json \
  --prompt "once upon a time" --ngen 4 --lanes 64 --p 8 \
  --out /tmp/scratch_weights_header.h --tokenizer-ddr \
  --tokenizer-blob-out ~/RVchatbot/kevgpt-genesys2-soc/tokenizer_ddr_v16384.bin

python -m fabric.genesys2.send_weights --port /dev/ttyUSB0 \
  --npz fabric/export_stepC_d128_v16384/goformer.npz \
  --tokenizer-blob ~/RVchatbot/kevgpt-genesys2-soc/tokenizer_ddr_v16384.bin
# once that prints "waiting for KEVGPT_UART_READY", in another terminal:
riscv32-corev-elf-gdb -batch -ex "target remote :3333" \
  -ex "monitor reset halt" -ex "load" -ex "monitor resume" -ex "detach" \
  ~/RVchatbot/kevgpt-genesys2-soc/hw/vendor/esl_epfl_x_heep/sw/build/main.elf</pre>
      </li>
      <li><b>Capture the hardware side</b> &mdash; 25 real generations (5 prompts &times; 5 repeats), on-chip
        Gumbel-max sampling, no seed control (each turn's real captured seed is recorded):
        <pre>python model/tinystories_hf_repro/benchmark_hw_stories_128w.py \
  --out model/tinystories_hf_repro/benchmark_128w_hw.json</pre>
      </li>
      <li><b>Capture the reference side</b> &mdash; independent of the board, GPU-only, can run in parallel
        with steps 1-3:
        <pre>python model/tinystories_hf_repro/benchmark_reference_stories_128w.py \
  --out model/tinystories_hf_repro/benchmark_128w_reference.json</pre>
      </li>
      <li><b>Build this report</b> from the two JSON files:
        <pre>python model/tinystories_hf_repro/build_benchmark_128w_report.py</pre>
      </li>
      <li><b>Restore the board to normal chat.</b> Revert <code>MAX_GEN_LEN</code>/<code>STORY_SENTENCES</code>
        to <code>60u</code>/<code>8u</code> in <code>main.c</code>, then repeat step 1 (rebuild) and step 2
        (reload) to put the board back in its normal-chat configuration before real interactive use.</li>
    </ol>
  </div>

  <div class="runbook">
    <h2 class="sec">How the repetition guards work, and what they cost</h2>
    <p style="font-size:14px;line-height:1.6;color:var(--ink-muted);margin:0 0 14px;max-width:80ch;">
      kev-gpt's accelerator generates each token with a single-pass on-chip Gumbel-max tournament over all
      16,384 vocab logits &mdash; a fixed-function hardware primitive with no sort, no top-k/top-p, and no
      per-token host visibility into the full distribution by design (the same bandwidth-wall constraint
      that keeps weights resident in on-chip memory in the first place). Firmware can't re-rank or resample
      the distribution the way typical decoding tricks (repetition penalty, no-repeat-ngram) assume. What it
      CAN do, reusing a mechanism that already existed for a different purpose (a stop-token remask), is
      detect a collision after the token comes back and ask the hardware for a fresh answer with the
      offending candidate excluded.
    </p>
    <p style="font-size:14px;line-height:1.6;color:var(--ink-muted);margin:0 0 14px;max-width:80ch;">
      Three guards run in <code>chat_turn()</code>'s generation loop, checked in order after each token
      comes back: exact doubling (<code>tok == last_tok</code>), near-duplicate word stems
      (<code>is_stem_repeat()</code>), and delayed bigram recurrence (checked against the full history of
      the reply so far). On a hit, <code>remask_pick_excluding()</code> reruns a full linear scan over all
      16,384 head logits (<code>kevgpt_read_bank(dev, 8, i)</code>, Q6.25 fixed-point) to find the
      best-scoring token NOT in the exclusion set, then re-checks the result against every other guard again
      &mdash; a bounded 4-pass re-validation loop (fixed in commit <code>4c3a398</code> to accumulate every
      rejected candidate across passes, not just the most recent one, closing an oscillation bug the first
      fix, <code>57ce25a</code>, had itself introduced).
    </p>
    <p style="font-size:14px;line-height:1.6;color:var(--ink-muted);margin:0 0 14px;max-width:80ch;">
      <b style="color:var(--ink);">Not pronoun-aware, or part-of-speech-aware at all.</b> It's easy to read
      the "dog"/"cat"/"owner" example below as the guard deliberately swapping a repeated noun for a pronoun
      or a topically-sensible alternative &mdash; it isn't. <code>remask_pick_excluding()</code> is a plain
      masked-argmax over the SAME full 16,384-logit distribution the hardware already computed for that
      position: whatever scores highest among the ids not yet excluded wins, with no notion of grammatical
      role or narrative coherence. In the traces captured for both bugs above, the actual substitutions were
      other concrete nouns, not pronouns &mdash; "penny" &rarr; "magnet" in the first case, and "dog"
      &harr; "cat" (cycling between two other cast members) in the second, before either fix. When that
      second-best logit happens to be topically sensible, the output reads as a deliberate edit; when it
      isn't, the guard has no way to tell &mdash; it's picking a runner-up from the model's own distribution,
      not exercising editorial judgment.
    </p>
    <p style="font-size:14px;line-height:1.6;color:var(--ink-muted);margin:0;max-width:80ch;">
      <b style="color:var(--ink);">Cost model: only paid on an actual collision.</b> An unconditional
      full-vocab rescan on every token would dominate the cycle budget, so each guard does nothing on the
      common case and only triggers the extra scan when it actually fires. That shows up directly in this
      benchmark's own token rates: <b style="color:var(--ink);">41.8 tok/s average, ranging 36.9-46.1 tok/s
      (a 23% spread) across otherwise-identical prompts and settings.</b> The slowest prompt group, "the dog
      ran" (39.9 tok/s avg) &mdash; the same prompt family behind both bugs fixed above, whose stories tend
      to revolve around a small set of recurring nouns ("dog"/"cat"/"owner") that trip the bigram guard
      repeatedly &mdash; runs about 8% slower than the fastest group, "she found a" (43.2 tok/s avg),
      consistent with more guard firings on that prompt, not a hardware slowdown.
    </p>
  </div>

  <div class="controls">
    <div style="width:1px"></div>
    <label class="toggle"><input type="checkbox" id="flagged-only"> flagged or fixation-word only</label>
    <div class="legend-strip">
      <span><mark class="flag">flag</mark> = detector-flagged span</span>
      <span><mark class="fix">fixation</mark> = known kev-gpt non-sequitur token</span>
    </div>
  </div>

  <div id="rows"></div>
  <div id="empty-state">No samples match this filter.</div>

  <footer class="endnote">
    Hardware panels: <code>fabric.genesys2.chat_over_uart</code>'s own send/receive path, against the board's
    live post-fix weights (<code>fabric/export_stepC_d128_v16384/</code>) &mdash; see
    <code>fabric/genesys2/FIXATION-WORD-CDC-INVESTIGATION.md</code> for the RDADDRW fix this data postdates.
    Reference panels: <code>transformers.AutoModelForCausalLM</code> against the published
    <code>SauravP97/tiny-stories-19M</code> checkpoint. Rows pair the Nth hardware draw against the Nth
    reference draw of the same prompt-theme for reading convenience only, not a claim of equivalent sampling
    conditions.
  </footer>

</div>

<script id="report-data" type="application/json">__JSON_DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('report-data').textContent);
const FIXATION_TERMS = ['cardinal', "buster's", 'buster', 'bustled', "spidey's", 'spidey', 'contains', 'cube', 'care', 'chug', 'carefree'];

function escapeHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function flagSpan(text, reason){
  if (!reason) return null;
  let m = reason.match(/^exact doubling: (['"])(.*?)\1 (['"])(.*?)\3$/);
  if (m) return {w1: m[2], w2: m[4]};
  m = reason.match(/^bigram recurred \d+x: \((['"])(.*?)\1,\s*(['"])(.*?)\3\)$/);
  if (m) return {w1: m[2], w2: m[4]};
  return null;
}

function findSpans(text, reason, trackFixation){
  const spans = [];
  const fs = flagSpan(text, reason);
  if (fs){
    const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(esc(fs.w1) + '\\s?' + esc(fs.w2), 'gi');
    let mm;
    while ((mm = re.exec(text))){
      spans.push({start: mm.index, end: mm.index + mm[0].length, type: 'flag'});
      if (mm[0].length === 0) re.lastIndex++;
    }
  }
  if (trackFixation){
    FIXATION_TERMS.forEach(term => {
      const re = new RegExp('\\b' + term.replace(/'/g, "['’]") + '\\b', 'gi');
      let mm;
      while ((mm = re.exec(text))){
        spans.push({start: mm.index, end: mm.index + mm[0].length, type: 'fix'});
        if (mm[0].length === 0) re.lastIndex++;
      }
    });
  }
  spans.sort((a, b) => a.start - b.start || b.end - a.end);
  const merged = [];
  let lastEnd = -1;
  for (const s of spans){
    if (s.start < lastEnd) continue;
    merged.push(s);
    lastEnd = s.end;
  }
  return merged;
}

function highlight(text, reason, trackFixation){
  const spans = findSpans(text, reason, trackFixation);
  if (spans.length === 0) return escapeHtml(text);
  let out = '', last = 0;
  spans.forEach(s => {
    out += escapeHtml(text.slice(last, s.start));
    out += `<mark class="${s.type}">${escapeHtml(text.slice(s.start, s.end))}</mark>`;
    last = s.end;
  });
  out += escapeHtml(text.slice(last));
  return out;
}

function hasFixation(text){
  return FIXATION_TERMS.some(term => new RegExp('\\b' + term.replace(/'/g, "['’]") + '\\b', 'i').test(text));
}

function computeStats(samples){
  const total = samples.length;
  const flagged = samples.filter(s => s.reason).length;
  const avgWords = samples.reduce((a,s) => a + s.word_count, 0) / total;
  return {total, flagged, avgWords};
}

function renderStats(){
  const hwStats = computeStats(DATA.hardware);
  const refStats = computeStats(DATA.reference);
  const tiles = [
    {label:'Real Genesys2 board', cls:'hw', stats: hwStats},
    {label:'SauravP97/tiny-stories-19M', cls:'ref', stats: refStats},
  ];
  let html = tiles.map(t => `<div class="stat-tile ${t.cls}">
      <div class="sweep-label">${t.label}</div>
      <div class="num">${t.stats.avgWords.toFixed(0)}<small>avg words</small></div>
      <div class="pct">${t.stats.flagged}/${t.stats.total} flagged (see caveat)</div>
    </div>`).join('');
  const hwFix = DATA.hardware.filter(s => hasFixation(s.text)).length;
  html += `<div class="stat-tile hw">
      <div class="sweep-label">kev-gpt fixation words</div>
      <div class="num">${hwFix}<small>/${hwStats.total} samples</small></div>
      <div class="pct">care/cardinal/chug/&hellip;</div>
    </div>`;
  const avgRate = DATA.hardware.reduce((a,s) => a + (s.rate_tok_s||0), 0) / DATA.hardware.length;
  html += `<div class="stat-tile hw">
      <div class="sweep-label">Hardware throughput</div>
      <div class="num">${avgRate.toFixed(1)}<small>tok/s avg</small></div>
      <div class="pct">real UART-measured</div>
    </div>`;
  document.getElementById('stats-grid').innerHTML = html;
}

let flaggedOnly = false;

function panelHtml(sample, cls, label, trackFixation){
  const flagged = !!sample.reason;
  const fix = trackFixation && hasFixation(sample.text);
  const badge = flagged ? `<span class="flagpill flagged">flagged</span>` : `<span class="flagpill clean">clean</span>`;
  const reasonHtml = flagged ? `<div class="reason">${escapeHtml(sample.reason)}</div>` : '';
  const hide = (flaggedOnly && !flagged && !fix) ? ' hidden' : '';
  const rateTag = (sample.rate_tok_s !== undefined) ? `<span>${sample.rate_tok_s} tok/s</span>` : '';
  const seedTag = sample.real_seed ? `<span>seed ${sample.real_seed}</span>` : (sample.seed !== undefined ? `<span>seed ${sample.seed}</span>` : '');
  return `<div class="panel${hide}">
    <div class="panel-head">
      <span class="modeltag ${cls}"><span class="dot ${cls}"></span>${label}</span>
      ${badge}
    </div>
    ${reasonHtml}
    <div class="story">${highlight(sample.text, sample.reason, trackFixation)}</div>
    <div class="panel-foot">
      <span>prompt: &ldquo;${escapeHtml(sample.prompt)}&rdquo;</span>
      <span style="display:flex;gap:10px">${seedTag}<span>${sample.word_count} words</span>${rateTag}</span>
    </div>
  </div>`;
}

function render(){
  const hw = DATA.hardware, ref = DATA.reference;
  const container = document.getElementById('rows');
  let html = '', visibleRows = 0;
  for (let i = 0; i < hw.length; i++){
    const h = hw[i], r = ref[i];
    const anyMatch = h.reason || r.reason || hasFixation(h.text);
    if (flaggedOnly && !anyMatch) continue;
    visibleRows++;
    html += `<div class="rowgroup">
      <div class="rowmeta"><span class="seedno">draw ${i+1}</span><span>condition ${i+1} of ${hw.length}</span></div>
      <div class="pair">
        ${panelHtml(h, 'hw', 'real board', true)}
        ${panelHtml(r, 'ref', 'reference 19M', false)}
      </div>
    </div>`;
  }
  container.innerHTML = html;
  document.getElementById('empty-state').style.display = visibleRows === 0 ? 'block' : 'none';
}

document.getElementById('flagged-only').addEventListener('change', e => {
  flaggedOnly = e.target.checked;
  render();
});

renderStats();
render();
</script>
"""

HTML = HTML.replace("__JSON_DATA__", json_blob)

with open(REPORT_PATH, "w") as f:
    f.write(HTML)

print(f"wrote {REPORT_PATH}")
print(f"hw: avg {hw_avg:.1f} words, {hw_flagged}/{len(hw)} flagged, {hw_rate_avg:.1f} tok/s avg")
print(f"reference: avg {ref_avg:.1f} words, {ref_flagged}/{len(ref)} flagged")
