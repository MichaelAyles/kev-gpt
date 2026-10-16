"""Renders guard_fix_pairs.json + guard_fix_stages.json into
guard_fix_diff_report.html: same real_seed replayed across three firmware
build stages (KEVGPT_FORCE_SEED) to show exactly what token-level effect
the two repetition-guard fixes (57ce25a, 4c3a398) had on kev-gpt's real
Genesys2 output, plus the full 25-sample sweep at each build stage.

    python model/tinystories_hf_repro/build_guard_fix_diff_report.py
"""
import json

PAIRS_PATH = "/home/tparng/kev-gpt/model/tinystories_hf_repro/guard_fix_pairs.json"
STAGES_PATH = "/home/tparng/kev-gpt/model/tinystories_hf_repro/guard_fix_stages.json"
REPORT_PATH = "/home/tparng/kev-gpt/model/tinystories_hf_repro/guard_fix_diff_report.html"

pairs = json.load(open(PAIRS_PATH))
stages = json.load(open(STAGES_PATH))


def common_prefix_words(a, b):
    aw, bw = a.split(), b.split()
    n = 0
    while n < len(aw) and n < len(bw) and aw[n] == bw[n]:
        n += 1
    return n


for p in pairs.values():
    p["before"]["prefix_words"] = common_prefix_words(p["before"]["text"], p["after"]["text"])
    p["after"]["prefix_words"] = p["before"]["prefix_words"]

payload = {"pairs": pairs, "stages": stages}
json_blob = json.dumps(payload, separators=(",", ":")).replace("</script", "<\\/script")

STAGE_ORDER = ["pre_any_fix", "post_fix1_only", "post_both_fixes"]

HTML = r"""<title>Same Seed, Different Guard</title>
<style>
:root{
  --bg:#F5F6F8; --surface:#FFFFFF; --surface-2:#EEF0F3;
  --ink:#1C2129; --ink-muted:#5B6472; --ink-faint:#88909B;
  --border:#DDE1E7; --border-soft:#E8EAED;
  --accent-before:#A6491F; --accent-before-bg:rgba(166,73,31,0.09); --accent-before-bg-strong:rgba(166,73,31,0.16);
  --accent-after:#1E7F72; --accent-after-bg:rgba(30,127,114,0.10); --accent-after-bg-strong:rgba(30,127,114,0.16);
  --before-mark-bg:rgba(201,90,41,0.22); --before-mark-fg:#7A3413;
  --after-mark-bg:rgba(30,127,114,0.16); --after-mark-fg:#155048;
  --flag-mark-bg:rgba(201,138,31,0.28); --flag-mark-fg:#5C3F06;
  --clean:#2E7D4F;
  --shadow: 0 1px 2px rgba(28,33,41,0.05), 0 4px 14px rgba(28,33,41,0.05);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#14171C; --surface:#1B1F26; --surface-2:#20252D;
    --ink:#E7EAEF; --ink-muted:#9AA3B2; --ink-faint:#6C7684;
    --border:#2A2F38; --border-soft:#242932;
    --accent-before:#E08A56; --accent-before-bg:rgba(224,138,86,0.12); --accent-before-bg-strong:rgba(224,138,86,0.20);
    --accent-after:#4FD1C0; --accent-after-bg:rgba(79,209,192,0.12); --accent-after-bg-strong:rgba(79,209,192,0.20);
    --before-mark-bg:rgba(224,138,86,0.24); --before-mark-fg:#F3C9AC;
    --after-mark-bg:rgba(79,209,192,0.22); --after-mark-fg:#A9EEE3;
    --flag-mark-bg:rgba(240,180,41,0.30); --flag-mark-fg:#FCE3A6;
    --clean:#5FBE86;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 20px rgba(0,0,0,0.35);
  }
}
:root[data-theme="dark"]{
  --bg:#14171C; --surface:#1B1F26; --surface-2:#20252D;
  --ink:#E7EAEF; --ink-muted:#9AA3B2; --ink-faint:#6C7684;
  --border:#2A2F38; --border-soft:#242932;
  --accent-before:#E08A56; --accent-before-bg:rgba(224,138,86,0.12); --accent-before-bg-strong:rgba(224,138,86,0.20);
  --accent-after:#4FD1C0; --accent-after-bg:rgba(79,209,192,0.12); --accent-after-bg-strong:rgba(79,209,192,0.20);
  --before-mark-bg:rgba(224,138,86,0.24); --before-mark-fg:#F3C9AC;
  --after-mark-bg:rgba(79,209,192,0.22); --after-mark-fg:#A9EEE3;
  --flag-mark-bg:rgba(240,180,41,0.30); --flag-mark-fg:#FCE3A6;
  --clean:#5FBE86;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 20px rgba(0,0,0,0.35);
}

*{box-sizing:border-box;}
body{ background:var(--bg); color:var(--ink); font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif; line-height:1.5; }
::selection{ background:var(--flag-mark-bg); }
.wrap{ max-width:1180px; margin:0 auto; padding:56px 28px 80px; }

header.top{ margin-bottom:36px; }
.eyebrow{ font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:0.12em; text-transform:uppercase; color:var(--ink-faint); margin:0 0 10px; }
h1{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:clamp(32px,4.2vw,46px); margin:0 0 14px; letter-spacing:-0.01em; text-wrap:balance; color:var(--ink); }
h2.sec{ font-family:'Fraunces',serif; font-size:22px; font-weight:600; margin:0 0 14px; }
h3.sub{ font-family:'Fraunces',serif; font-size:17px; font-weight:600; margin:0 0 4px; color:var(--ink); }
.dek{ font-size:16px; color:var(--ink-muted); max-width:74ch; margin:0 0 18px; }
.provenance{ font-family:'IBM Plex Mono',monospace; font-size:12.5px; color:var(--ink-faint); display:flex; flex-wrap:wrap; gap:6px 18px; border-top:1px solid var(--border-soft); padding-top:16px; }
.provenance b{ color:var(--ink-muted); font-weight:500; }

.caveat{
  background:var(--surface-2); border:1px solid var(--border); border-left:3px solid var(--accent-after);
  border-radius:8px; padding:14px 18px; font-size:13.5px; color:var(--ink-muted); margin-bottom:32px; max-width:82ch;
}
.caveat b{ color:var(--ink); }
.caveat + .caveat{ margin-top:-20px; }

.modeltag{ display:inline-flex; align-items:center; gap:6px; font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; padding:3px 9px; border-radius:5px; letter-spacing:0.01em; }
.modeltag.before{ background:var(--accent-before-bg); color:var(--accent-before); }
.modeltag.after{ background:var(--accent-after-bg); color:var(--accent-after); }
.dot{ width:7px; height:7px; border-radius:50%; display:inline-block; }
.dot.before{ background:var(--accent-before); }
.dot.after{ background:var(--accent-after); }

.explainer{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px 26px; margin-bottom:40px; box-shadow:var(--shadow); }
.explainer p{ font-size:14px; line-height:1.6; color:var(--ink-muted); margin:0 0 14px; max-width:82ch; }
.explainer p:last-child{ margin-bottom:0; }
.explainer code{ font-family:'IBM Plex Mono',monospace; font-size:0.93em; background:var(--surface-2); padding:1px 5px; border-radius:4px; }
.explainer b{ color:var(--ink); }
.legend-inline{ display:flex; gap:18px; align-items:center; font-size:12.5px; color:var(--ink-faint); flex-wrap:wrap; margin-top:4px; }
.legend-inline mark.before{ background:var(--before-mark-bg); color:var(--before-mark-fg); padding:0 2px; border-radius:2px; }
.legend-inline mark.after{ background:var(--after-mark-bg); color:var(--after-mark-fg); padding:0 2px; border-radius:2px; }

.diffcard{ background:var(--surface); border:1px solid var(--border); border-radius:14px; margin-bottom:28px; overflow:hidden; box-shadow:var(--shadow); }
.diffcard-head{ padding:20px 24px 16px; border-bottom:1px solid var(--border-soft); }
.diffcard-head .commit{ font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--ink-faint); }
.diffcard-head .commit code{ background:var(--surface-2); padding:1px 6px; border-radius:4px; color:var(--ink-muted); }
.diffmeta{ font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--ink-faint); display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; }
.diffpair{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--border-soft); }
@media (max-width:760px){ .diffpair{ grid-template-columns:1fr; } }
.diffpanel{ background:var(--surface); padding:18px 22px 20px; display:flex; flex-direction:column; }
.diffpanel-head{ display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; gap:10px; }
.flagpill{ font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600; letter-spacing:0.02em; padding:3px 8px; border-radius:20px; white-space:nowrap; }
.flagpill.clean{ background:rgba(46,125,79,0.12); color:var(--clean); }
.flagpill.flagged{ background:var(--flag-mark-bg); color:var(--flag-mark-fg); }
.reason{ font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--flag-mark-fg); margin-bottom:10px; line-height:1.4; }
.story{ font-family:'Literata',Georgia,serif; font-size:14.5px; line-height:1.7; color:var(--ink); flex:1; }
.story .prefix{ color:var(--ink-muted); }
.story mark.before{ background:var(--before-mark-bg); color:var(--before-mark-fg); padding:0 1px; border-radius:2px; font-weight:600; }
.story mark.after{ background:var(--after-mark-bg); color:var(--after-mark-fg); padding:0 1px; border-radius:2px; font-weight:600; }
.diffpanel-foot{ margin-top:12px; padding-top:10px; border-top:1px solid var(--border-soft); font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--ink-faint); display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px; }
.note{ margin:0 24px 24px; background:var(--surface-2); border:1px solid var(--border-soft); border-radius:10px; padding:18px 20px; }
.note-head{ font-family:'IBM Plex Mono',monospace; font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:var(--ink-faint); margin-bottom:10px; }
.note p{ font-size:13.5px; line-height:1.6; color:var(--ink-muted); margin:0 0 12px; }
.note p:last-child{ margin-bottom:0; }
.note b{ color:var(--ink); }
.note code{ font-family:'IBM Plex Mono',monospace; font-size:0.92em; background:var(--surface); border:1px solid var(--border-soft); padding:1px 5px; border-radius:4px; }
.eventgrid{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }
@media (max-width:640px){ .eventgrid{ grid-template-columns:1fr; } }
.eventcol{ background:var(--surface); border:1px solid var(--border-soft); border-radius:8px; padding:12px 14px; }
.eventcol .evlabel{ font-family:'IBM Plex Mono',monospace; font-size:10.5px; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px; }
.eventcol.before .evlabel{ color:var(--accent-before); }
.eventcol.after .evlabel{ color:var(--accent-after); }
.evchain{ font-family:'IBM Plex Mono',monospace; font-size:12px; line-height:1.9; color:var(--ink-muted); }
.evchain .arrow{ color:var(--ink-faint); margin:0 4px; }
.evchain .raw{ color:var(--ink-muted); }
.evchain .rej{ color:var(--before-mark-fg); text-decoration:line-through; text-decoration-color:var(--before-mark-fg); opacity:0.75; }
.evchain .final{ font-weight:700; }
.eventcol.before .evchain .final{ color:var(--accent-before); }
.eventcol.after .evchain .final{ color:var(--accent-after); }
.evsource{ font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--ink-faint); margin-top:8px; }

.stats-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:12px; overflow:hidden; margin-bottom:44px; box-shadow:var(--shadow); }
@media (max-width:820px){ .stats-grid{ grid-template-columns:1fr; } }
.stat-tile{ background:var(--surface); padding:22px 20px; }
.stat-tile .sweep-label{ font-family:'IBM Plex Mono',monospace; font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:var(--ink-faint); margin-bottom:4px; }
.stat-tile .sweep-desc{ font-size:12px; color:var(--ink-faint); margin-bottom:12px; }
.stat-tile .num{ font-family:'Fraunces',serif; font-size:30px; font-weight:600; line-height:1; font-variant-numeric:tabular-nums; margin-bottom:6px; }
.stat-tile .num small{ font-size:14px; font-weight:500; color:var(--ink-faint); }
.stat-tile:nth-child(3n+1) .num{ color:var(--accent-before); }
.stat-tile:nth-child(3n+2) .num{ color:#9A7A1E; }
.stat-tile:nth-child(3n+3) .num{ color:var(--accent-after); }
.stat-tile .pct{ font-size:12.5px; color:var(--ink-muted); font-family:'IBM Plex Mono',monospace; }

.archive-head{ display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px; margin-bottom:22px; position:sticky; top:0; background:var(--bg); padding:14px 0; border-bottom:1px solid var(--border-soft); z-index:5; }
.toggle{ display:inline-flex; align-items:center; gap:8px; font-size:13.5px; color:var(--ink-muted); cursor:pointer; user-select:none; }
.toggle input{ accent-color:var(--accent-before); width:15px; height:15px; }

.stage-section{ margin-bottom:40px; }
.stage-title{ display:flex; align-items:center; gap:12px; margin-bottom:16px; }
.stage-title h3{ font-family:'Fraunces',serif; font-size:19px; font-weight:600; margin:0; }
.stage-title .stage-sub{ font-size:12.5px; color:var(--ink-faint); font-family:'IBM Plex Mono',monospace; }
.cardgrid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; }
.card{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 18px 18px; box-shadow:var(--shadow); display:flex; flex-direction:column; }
.card-head{ display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; gap:8px; }
.card-head .seedtag{ font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--ink-faint); }
.card .story{ font-size:13.5px; flex:1; }
.card-foot{ margin-top:10px; padding-top:8px; border-top:1px solid var(--border-soft); font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--ink-faint); display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px; }

footer.endnote{ margin-top:56px; padding-top:20px; border-top:1px solid var(--border-soft); font-size:12.5px; color:var(--ink-faint); max-width:76ch; }
footer.endnote a{ color:var(--accent-after); }
#empty-state{ display:none; padding:40px 0; text-align:center; color:var(--ink-faint); font-size:14px; }
</style>

<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Literata:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">

<div class="wrap">

  <header class="top">
    <p class="eyebrow">Genesys2 &middot; KEVGPT_FORCE_SEED replay across three firmware builds</p>
    <h1>Same Seed, Different Guard</h1>
    <p class="dek">Two consecutive real bugs turned up in kev-gpt's firmware repetition guards this session
      (commits <code>57ce25a</code> and <code>4c3a398</code>). Rather than describe the fixes in prose, this
      replays the <b>exact same real-hardware seed</b> through three firmware builds &mdash; before either
      fix, after the first, and after both &mdash; so the story text itself shows what each fix actually
      changed. A second section below shows the full 25-sample sweep at each build stage, the way the
      companion <a href="https://claude.ai/code/artifact/b1997b8c-0994-4bc6-8b93-12514f6e120e" target="_blank" rel="noopener">Long-Form Fidelity</a> report does for hardware vs. reference.</p>
    <div class="provenance">
      <span><b>Method</b> KEVGPT_FORCE_SEED replays one real captured seed deterministically across builds</span>
      <span><b>Detector</b> model.filter_synth_corpus.is_degenerate, min_bigram_recurrence=6</span>
      <span><b>Board weights</b> data/ckpt_stepC_d128_v16384.qat.pt (INT4/INT8, 2-4M params) &mdash; unchanged
        across all three builds; only firmware changed</span>
      <span><b>Full writeups</b> fabric/genesys2/PORT-NOTES.md</span>
    </div>
  </header>

  <div class="explainer">
    <h2 class="sec">How to read these</h2>
    <p>
      Each diff card below pairs the <b>same <code>real_seed</code></b> captured on real hardware, replayed
      through two different firmware builds. Because token generation is autoregressive, the two outputs are
      identical word-for-word up to whatever position the fixed guard logic first behaves differently &mdash;
      then every token after that point can diverge, since each new token conditions on everything before it.
      The shared run of words is shown in muted ink; the first point of divergence onward is highlighted,
      <mark class="before">before</mark> in rust and <mark class="after">after</mark> in teal.
    </p>
    <p>
      <b>The divergence point is not always exactly where the named bug fires.</b> Both fixes changed
      <code>remask_pick_excluding()</code>/<code>remask_pick()</code> themselves, which is shared machinery
      used by <i>every</i> guard, including the unrelated stop-token (<code>MIN_WORD_TOKENS</code>) remask
      that can fire much earlier in a reply. In the first diff card below, the visible "new new new new new"
      loop starts around word 72, but the two outputs already diverge at word 65 ("mom." vs. "mommy.")
      &mdash; an earlier, unrelated guard firing that the same code change also touched. That earlier
      divergence is expected, not a discrepancy: once one token differs, nothing downstream is directly
      comparable anymore, which is exactly why the visible effect of a one-line fix can be a completely
      different second half of the story, not a single swapped word.
    </p>
    <p>
      <b>The substitutions are not pronoun- or grammar-aware.</b> <code>remask_pick_excluding()</code> is a
      plain masked-argmax over the same full 16,384-logit distribution the hardware already computed for
      that position &mdash; whatever scores highest among the ids not yet excluded wins, with no notion of
      grammatical role. Below, the actual substitutions are other concrete nouns ("penny" &rarr; "magnet"),
      not pronouns.
    </p>
    <p>
      <b>Where the candidates in each event chain actually come from.</b> Every forward pass genuinely computes
      a score for all 16,384 vocab ids, not just the winner &mdash; that's inherent to how the model's output
      layer works. But kev-gpt's accelerator normally exposes only ONE of those: its fast path runs a
      single-pass Gumbel-max tournament in hardware and returns just the winning token, by design (no sort
      primitive, no per-token host visibility into the full ranking). When a guard fires,
      <code>remask_pick_excluding()</code> does not trigger a new forward pass or ask the model a fresh
      question &mdash; it uses a separate, slower diagnostic readback port
      (<code>kevgpt_read_bank(dev, 8, i)</code>) to pull the raw logit for each of the 16,384 ids, one at a
      time, from that SAME already-computed distribution for that exact token position, then does a masked
      argmax over it in software, skipping whatever's already in the exclusion set. So every "raw pick
      &rarr; rejected &rarr; rejected &rarr; final" chain in the event grids below is firmware replaying a
      masked version of one static, already-finished hardware computation &mdash; not the model being asked
      again and again. It's also why guard firings cost real cycles without costing a new inference: the
      O(16,384) rescan is comparatively slow firmware work layered on top of a computation the accelerator
      already did once.
    </p>
  </div>

  <div id="diffcards"></div>

  <h2 class="sec" style="margin-top:8px;">The 25-sample sweep, at each build stage</h2>
  <p class="dek" style="margin-bottom:24px;">Not seed-matched like the pairs above &mdash; each sweep is an
    independent 5&times;5 real-hardware capture, run once per build stage as that stage was deployed. Shown
    here to size the aggregate effect, not to compare individual stories 1:1.</p>

  <div class="stats-grid" id="stats-grid"></div>

  <div class="archive-head">
    <div style="width:1px"></div>
    <label class="toggle"><input type="checkbox" id="flagged-only"> flagged only, all stages</label>
    <div class="legend-inline">
      <span><mark class="before" style="background:var(--flag-mark-bg);color:var(--flag-mark-fg);">flag</mark> = detector-flagged span</span>
    </div>
  </div>

  <div id="stage-archive"></div>
  <div id="empty-state">No samples match this filter.</div>

  <footer class="endnote">
    Matched-seed replays captured via the <code>KEVGPT_FORCE_SEED</code>/<code>KEVGPT_DIAG_GUARD_TRACE</code>
    firmware diagnostics added this session (left in the tree, off by default). Sweep data for
    <code>pre_any_fix</code> and <code>post_fix1_only</code> are historical snapshots from
    <code>kev-gpt</code> git history (commits <code>838058e</code> and <code>d53fbd3</code>) &mdash; those
    firmware builds are superseded and not separately reproducible; <code>post_both_fixes</code> is the
    currently deployed build and reproducible via the companion Long-Form Fidelity report's own runbook.
    Full technical writeups: <code>fabric/genesys2/PORT-NOTES.md</code>'s "Guard-substitution repetition
    bug" and "Guard-substitution repeat, round two" sections.
  </footer>

</div>

<script id="report-data" type="application/json">__JSON_BLOB__</script>
<script>
const DATA = JSON.parse(document.getElementById('report-data').textContent);
const STAGE_ORDER = __STAGE_ORDER__;

function escapeHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function flagSpan(text, reason){
  if (!reason) return null;
  let m = reason.match(/^exact doubling: (['"])(.*?)\1 (['"])(.*?)\3$/);
  if (m) return {w1: m[2], w2: m[4]};
  m = reason.match(/^bigram recurred \d+x: \((['"])(.*?)\1,\s*(['"])(.*?)\3\)$/);
  if (m) return {w1: m[2], w2: m[4]};
  return null;
}

function highlightFlag(text, reason){
  const fs = flagSpan(text, reason);
  if (!fs) return escapeHtml(text);
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(esc(fs.w1) + '\\s?' + esc(fs.w2), 'i');
  const mm = re.exec(text);
  if (!mm) return escapeHtml(text);
  return escapeHtml(text.slice(0, mm.index)) +
    `<mark class="before" style="background:var(--flag-mark-bg);color:var(--flag-mark-fg);">${escapeHtml(mm[0])}</mark>` +
    escapeHtml(text.slice(mm.index + mm[0].length));
}

function diffHtml(text, prefixWords, markClass){
  const words = text.split(' ');
  const prefix = words.slice(0, prefixWords).join(' ');
  const tail = words.slice(prefixWords).join(' ');
  if (!tail) return `<span class="prefix">${escapeHtml(prefix)}</span>`;
  return `<span class="prefix">${escapeHtml(prefix)}</span> <mark class="${markClass}">${escapeHtml(tail)}</mark>`;
}

function diffPanelHtml(sample, cls, label){
  const flagged = !!sample.reason;
  const badge = flagged ? `<span class="flagpill flagged">flagged</span>` : `<span class="flagpill clean">clean</span>`;
  const reasonHtml = flagged ? `<div class="reason">${escapeHtml(sample.reason)}</div>` : '';
  const rateTag = sample.rate_tok_s ? `<span>${sample.rate_tok_s} tok/s</span>` : `<span>rate n/a (diagnostic replay)</span>`;
  return `<div class="diffpanel">
    <div class="diffpanel-head">
      <span class="modeltag ${cls}"><span class="dot ${cls}"></span>${label}</span>
      ${badge}
    </div>
    ${reasonHtml}
    <div class="story">${diffHtml(sample.text, sample.prefix_words, cls)}</div>
    <div class="diffpanel-foot">
      <span>${sample.build}</span>
      <span style="display:flex;gap:10px"><span>${sample.word_count} words</span>${rateTag}</span>
    </div>
  </div>`;
}

function chainHtml(ev){
  const parts = [`<span class="raw">raw pick: <b>${escapeHtml(ev.raw_word)}</b> (${ev.raw_tok})</span>`];
  ev.fired.forEach((f, i) => {
    const isLast = i === ev.fired.length - 1;
    const cls = (f.sub_tok === ev.final_tok && isLast) ? 'final' : 'rej';
    parts.push(`<span class="arrow">&rarr;</span><span class="${cls}">${escapeHtml(f.sub_word)} (${f.sub_tok})</span>`);
  });
  if (ev.fired.length === 0){
    // guard never fired at this position in this build; raw pick was accepted as-is
  }
  return parts.join('');
}

function noteHtml(p){
  const m = p.mechanism;
  const wi = m.divergence_word_index;
  const beforeChain = chainHtml(m.before_event);
  const afterChain = chainHtml(m.after_event);
  const samePick = m.before_event.raw_tok === m.after_event.raw_tok;
  return `<div class="note">
    <div class="note-head">First token that changes &mdash; word ${wi + 1} of the reply</div>
    <p>
      Both builds' models make the <b>identical raw pick</b> at this exact position (word ${wi + 1}):
      &ldquo;<b>${escapeHtml(m.before_event.raw_word)}</b>&rdquo; (token ${m.before_event.raw_tok}). Everything up to
      here is byte-identical between builds because the accelerator draws the same Gumbel sample either way &mdash;
      only what the firmware does with a colliding pick differs. Here, that candidate collides with the
      bigram-recurrence guard (it already appeared in this exact position relative to the word before it, earlier
      in the same reply), so both builds intervene. What happens next is where they diverge:
    </p>
    <div class="eventgrid">
      <div class="eventcol before">
        <div class="evlabel">before fix</div>
        <div class="evchain">${beforeChain}</div>
        <div class="evsource">${escapeHtml(m.before_source)}, pos=${m.before_event.pos}</div>
      </div>
      <div class="eventcol after">
        <div class="evlabel">after fix</div>
        <div class="evchain">${afterChain}</div>
        <div class="evsource">${escapeHtml(m.after_source)}, pos=${m.after_event.pos}</div>
      </div>
    </div>
    <p>${p.note}</p>
  </div>`;
}

function renderDiffCards(){
  const container = document.getElementById('diffcards');
  let html = '';
  ['pair1', 'pair2'].forEach(key => {
    const p = DATA.pairs[key];
    html += `<div class="diffcard">
      <div class="diffcard-head">
        <h3 class="sub">${escapeHtml(p.title)}</h3>
        <div class="commit">fix: <code>${escapeHtml(p.fix_commit)}</code></div>
        <div class="diffmeta">
          <span>prompt: &ldquo;${escapeHtml(p.prompt)}&rdquo;</span>
          <span>real_seed: ${escapeHtml(p.real_seed)}</span>
          <span>common prefix: ${p.before.prefix_words} words</span>
        </div>
      </div>
      <div class="diffpair">
        ${diffPanelHtml(p.before, 'before', 'before fix')}
        ${diffPanelHtml(p.after, 'after', 'after fix')}
      </div>
      ${noteHtml(p)}
    </div>`;
  });
  container.innerHTML = html;
}

function computeStats(samples){
  const total = samples.length;
  const flagged = samples.filter(s => s.reason).length;
  const avgWords = samples.reduce((a,s) => a + s.word_count, 0) / total;
  const avgRate = samples.reduce((a,s) => a + (s.rate_tok_s||0), 0) / total;
  return {total, flagged, avgWords, avgRate};
}

function renderStats(){
  let html = '';
  STAGE_ORDER.forEach(key => {
    const meta = DATA.stages.meta[key];
    const stats = computeStats(DATA.stages.stages[key]);
    html += `<div class="stat-tile">
      <div class="sweep-label">${escapeHtml(meta.label)}</div>
      <div class="sweep-desc">${escapeHtml(meta.desc)} &middot; <code>${escapeHtml(meta.commit)}</code></div>
      <div class="num">${stats.flagged}<small>/${stats.total} flagged</small></div>
      <div class="pct">${stats.avgWords.toFixed(1)} avg words &middot; ${stats.avgRate.toFixed(1)} tok/s avg</div>
    </div>`;
  });
  document.getElementById('stats-grid').innerHTML = html;
}

let flaggedOnly = false;

function cardHtml(sample){
  const flagged = !!sample.reason;
  const badge = flagged ? `<span class="flagpill flagged">flagged</span>` : `<span class="flagpill clean">clean</span>`;
  const reasonHtml = flagged ? `<div class="reason">${escapeHtml(sample.reason)}</div>` : '';
  return `<div class="card">
    <div class="card-head">
      <span class="seedtag">${escapeHtml(sample.real_seed)}</span>
      ${badge}
    </div>
    ${reasonHtml}
    <div class="story">${highlightFlag(sample.text, sample.reason)}</div>
    <div class="card-foot">
      <span>&ldquo;${escapeHtml(sample.prompt)}&rdquo;</span>
      <span>${sample.word_count}w &middot; ${sample.rate_tok_s} tok/s</span>
    </div>
  </div>`;
}

function renderArchive(){
  const container = document.getElementById('stage-archive');
  let html = '', visible = 0;
  STAGE_ORDER.forEach(key => {
    const meta = DATA.stages.meta[key];
    const samples = DATA.stages.stages[key];
    const shown = flaggedOnly ? samples.filter(s => s.reason) : samples;
    visible += shown.length;
    html += `<div class="stage-section">
      <div class="stage-title">
        <h3>${escapeHtml(meta.label)}</h3>
        <span class="stage-sub">${escapeHtml(meta.desc)} &middot; <code>${escapeHtml(meta.commit)}</code></span>
      </div>
      <div class="cardgrid">${shown.map(s => cardHtml(s)).join('')}</div>
    </div>`;
  });
  container.innerHTML = html;
  document.getElementById('empty-state').style.display = visible === 0 ? 'block' : 'none';
}

document.getElementById('flagged-only').addEventListener('change', e => {
  flaggedOnly = e.target.checked;
  renderArchive();
});

renderDiffCards();
renderStats();
renderArchive();
</script>
"""

HTML = HTML.replace("__JSON_BLOB__", json_blob).replace("__STAGE_ORDER__", json.dumps(STAGE_ORDER))

with open(REPORT_PATH, "w") as f:
    f.write(HTML)

print(f"wrote {REPORT_PATH}")
for key in STAGE_ORDER:
    samples = stages["stages"][key]
    flagged = sum(1 for s in samples if s["reason"])
    avg_words = sum(s["word_count"] for s in samples) / len(samples)
    avg_rate = sum(s["rate_tok_s"] for s in samples) / len(samples)
    print(f"{key}: {flagged}/{len(samples)} flagged, {avg_words:.1f} avg words, {avg_rate:.1f} tok/s avg")
