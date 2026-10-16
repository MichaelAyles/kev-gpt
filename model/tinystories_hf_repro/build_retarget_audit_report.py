"""Renders quality_sweep_results.json (produced by
quality_sweep_C_vs_reference.py) into repetition_audit_report.html: every
generated sample, paired by matching (seed, prompt-theme) across checkpoint
C and the reference model, with the exact span that tripped
model.filter_synth_corpus.is_degenerate highlighted inline.

    python model/tinystories_hf_repro/build_repetition_audit_report.py
"""
import json

RESULTS_PATH = "/home/tparng/kev-gpt/model/tinystories_hf_repro/quality_sweep_d128_results.json"
REPORT_PATH = "/home/tparng/kev-gpt/model/tinystories_hf_repro/retarget_audit_report.html"

data = json.load(open(RESULTS_PATH))

payload = {
    "sweep1": {"D128": data["sweep1"]["D128"], "reference": data["sweep1"]["reference"]},
    "sweep2": {"D128": data["sweep2"]["D128"], "reference": data["sweep2"]["reference"]},
}
json_blob = json.dumps(payload).replace("</script", "<\\/script")

HTML = r"""<title>Retarget Audit</title>
<style>
:root{
  --bg:#F5F6F8; --surface:#FFFFFF; --surface-2:#EEF0F3;
  --ink:#1C2129; --ink-muted:#5B6472; --ink-faint:#88909B;
  --border:#DDE1E7; --border-soft:#E8EAED;
  --accent-c:#1E7F72; --accent-c-bg:rgba(30,127,114,0.10); --accent-c-bg-strong:rgba(30,127,114,0.16);
  --accent-r:#5B4B8A; --accent-r-bg:rgba(91,75,138,0.09); --accent-r-bg-strong:rgba(91,75,138,0.15);
  --flag:#9A6B0C; --flag-mark-bg:rgba(201,138,31,0.28); --flag-mark-fg:#5C3F06;
  --clean:#2E7D4F;
  --shadow: 0 1px 2px rgba(28,33,41,0.05), 0 4px 14px rgba(28,33,41,0.05);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#14171C; --surface:#1B1F26; --surface-2:#20252D;
    --ink:#E7EAEF; --ink-muted:#9AA3B2; --ink-faint:#6C7684;
    --border:#2A2F38; --border-soft:#242932;
    --accent-c:#4FD1C0; --accent-c-bg:rgba(79,209,192,0.12); --accent-c-bg-strong:rgba(79,209,192,0.20);
    --accent-r:#B7A3EC; --accent-r-bg:rgba(183,163,236,0.12); --accent-r-bg-strong:rgba(183,163,236,0.20);
    --flag:#F0B429; --flag-mark-bg:rgba(240,180,41,0.30); --flag-mark-fg:#FCE3A6;
    --clean:#5FBE86;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 20px rgba(0,0,0,0.35);
  }
}
:root[data-theme="dark"]{
  --bg:#14171C; --surface:#1B1F26; --surface-2:#20252D;
  --ink:#E7EAEF; --ink-muted:#9AA3B2; --ink-faint:#6C7684;
  --border:#2A2F38; --border-soft:#242932;
  --accent-c:#4FD1C0; --accent-c-bg:rgba(79,209,192,0.12); --accent-c-bg-strong:rgba(79,209,192,0.20);
  --accent-r:#B7A3EC; --accent-r-bg:rgba(183,163,236,0.12); --accent-r-bg-strong:rgba(183,163,236,0.20);
  --flag:#F0B429; --flag-mark-bg:rgba(240,180,41,0.30); --flag-mark-fg:#FCE3A6;
  --clean:#5FBE86;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 20px rgba(0,0,0,0.35);
}

*{box-sizing:border-box;}
body{
  background:var(--bg); color:var(--ink);
  font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif;
  line-height:1.5;
}
::selection{ background:var(--flag-mark-bg); }

.wrap{ max-width:1180px; margin:0 auto; padding:56px 28px 80px; }

/* ---------- Header ---------- */
header.top{ margin-bottom:40px; }
.eyebrow{
  font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:0.12em;
  text-transform:uppercase; color:var(--ink-faint); margin:0 0 10px;
}
h1{
  font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:clamp(32px,4.2vw,46px);
  margin:0 0 14px; letter-spacing:-0.01em; text-wrap:balance; color:var(--ink);
}
.dek{
  font-size:16px; color:var(--ink-muted); max-width:66ch; margin:0 0 18px;
}
.provenance{
  font-family:'IBM Plex Mono',monospace; font-size:12.5px; color:var(--ink-faint);
  display:flex; flex-wrap:wrap; gap:6px 18px; border-top:1px solid var(--border-soft); padding-top:16px;
}
.provenance b{ color:var(--ink-muted); font-weight:500; }

/* ---------- Legend chips (models) ---------- */
.modeltag{
  display:inline-flex; align-items:center; gap:6px;
  font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600;
  padding:3px 9px; border-radius:5px; letter-spacing:0.01em;
}
.modeltag.c{ background:var(--accent-c-bg); color:var(--accent-c); }
.modeltag.r{ background:var(--accent-r-bg); color:var(--accent-r); }
.dot{ width:7px; height:7px; border-radius:50%; display:inline-block; }
.dot.c{ background:var(--accent-c); }
.dot.r{ background:var(--accent-r); }

/* ---------- Stat dashboard ---------- */
.stats-grid{
  display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:var(--border); border:1px solid var(--border); border-radius:12px;
  overflow:hidden; margin-bottom:18px; box-shadow:var(--shadow);
}
.stat-tile{ background:var(--surface); padding:22px 20px; }
.stat-tile .sweep-label{
  font-family:'IBM Plex Mono',monospace; font-size:11px; text-transform:uppercase;
  letter-spacing:0.1em; color:var(--ink-faint); margin-bottom:10px;
}
.stat-tile .num{
  font-family:'Fraunces',serif; font-size:34px; font-weight:600; line-height:1;
  font-variant-numeric:tabular-nums; margin-bottom:6px;
}
.stat-tile .num small{ font-size:16px; font-weight:500; color:var(--ink-faint); }
.stat-tile.c .num{ color:var(--accent-c); }
.stat-tile.r .num{ color:var(--accent-r); }
.stat-tile .pct{ font-size:13px; color:var(--ink-muted); font-family:'IBM Plex Mono',monospace; }

.metrics-row{
  display:grid; grid-template-columns:repeat(3,1fr); gap:1px;
  background:var(--border); border:1px solid var(--border); border-radius:12px;
  overflow:hidden; margin-bottom:44px;
}
.metric-cell{ background:var(--surface-2); padding:16px 20px; }
.metric-cell .mlabel{ font-size:12.5px; color:var(--ink-muted); margin-bottom:8px; }
.metric-cell table{ width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:12.5px; }
.metric-cell td{ padding:2px 0; }
.metric-cell td.k{ color:var(--ink-faint); }
.metric-cell td.v{ text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); }

/* ---------- Controls ---------- */
.controls{
  display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px;
  margin-bottom:22px; position:sticky; top:0; background:var(--bg); padding:14px 0;
  border-bottom:1px solid var(--border-soft); z-index:5;
}
.seg{ display:inline-flex; border:1px solid var(--border); border-radius:8px; overflow:hidden; background:var(--surface); }
.seg button{
  font-family:'IBM Plex Sans',sans-serif; font-size:13.5px; font-weight:500; color:var(--ink-muted);
  background:transparent; border:none; padding:8px 16px; cursor:pointer; border-right:1px solid var(--border);
}
.seg button:last-child{ border-right:none; }
.seg button.active{ background:var(--surface-2); color:var(--ink); }
.seg button:focus-visible{ outline:2px solid var(--accent-c); outline-offset:-2px; }

.toggle{
  display:inline-flex; align-items:center; gap:8px; font-size:13.5px; color:var(--ink-muted); cursor:pointer;
  user-select:none;
}
.toggle input{ accent-color:var(--flag); width:15px; height:15px; }

.legend-strip{
  display:flex; gap:16px; align-items:center; font-size:12.5px; color:var(--ink-faint); flex-wrap:wrap;
}
.legend-strip mark{ background:var(--flag-mark-bg); color:var(--flag-mark-fg); padding:0 2px; border-radius:2px; }

/* ---------- Rows / comparison grid ---------- */
.rowgroup{ margin-bottom:14px; }
.rowmeta{
  font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--ink-faint);
  margin:22px 0 8px; display:flex; gap:14px; align-items:baseline;
}
.rowmeta .seedno{ color:var(--ink-muted); font-weight:600; }

.pair{
  display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--border-soft);
  border:1px solid var(--border-soft); border-radius:10px; overflow:hidden;
}
@media (max-width:760px){ .pair{ grid-template-columns:1fr; } }

.panel{ background:var(--surface); padding:18px 20px 20px; display:flex; flex-direction:column; }
.panel.hidden{ display:none; }
.panel-head{
  display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; gap:10px;
}
.flagpill{
  font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600; letter-spacing:0.02em;
  padding:3px 8px; border-radius:20px; white-space:nowrap;
}
.flagpill.clean{ background:rgba(46,125,79,0.12); color:var(--clean); }
.flagpill.flagged{ background:var(--flag-mark-bg); color:var(--flag-mark-fg); }
.reason{
  font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--flag); margin-bottom:10px;
  line-height:1.4;
}
.story{
  font-family:'Literata',Georgia,serif; font-size:14.5px; line-height:1.68; color:var(--ink);
  flex:1;
}
.story mark{
  background:var(--flag-mark-bg); color:var(--flag-mark-fg); padding:0 1px; border-radius:2px;
  box-shadow:0 0 0 1px rgba(0,0,0,0.02) inset;
}
.panel-foot{
  margin-top:12px; padding-top:10px; border-top:1px solid var(--border-soft);
  font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--ink-faint);
  display:flex; justify-content:space-between;
}

footer.endnote{
  margin-top:56px; padding-top:20px; border-top:1px solid var(--border-soft);
  font-size:12.5px; color:var(--ink-faint); max-width:70ch;
}

#empty-state{ display:none; padding:60px 0; text-align:center; color:var(--ink-faint); font-size:14px; }
</style>

<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Literata:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">

<div class="wrap">

  <header class="top">
    <p class="eyebrow">Repetition detector &middot; seeded sweep</p>
    <h1>Retarget Audit</h1>
    <p class="dek">Every generated story from a two-sweep, multi-seed comparison of
      <span class="modeltag c"><span class="dot c"></span>checkpoint C, D=128</span> (kev-gpt,
      the actual retrained-fresh-at-D=128 shape flashed on Genesys2 &mdash; NOT the D=384
      checkpoint the ten-step ablation walk originally evaluated) against the published
      <span class="modeltag r"><span class="dot r"></span>reference</span>
      (SauravP97/tiny-stories-19M), run through the project's own bigram/doubling
      detector. Flagged spans are marked inline in the text below &mdash; this is
      every sample, not a curated subset.</p>
    <div class="provenance">
      <span><b>Detector</b> model.filter_synth_corpus.is_degenerate, min_bigram_recurrence=3</span>
      <span><b>Sweep 1</b> 5 seeds &times; 5 prompts, 60 new tokens</span>
      <span><b>Sweep 2</b> 3 seeds &times; 4 prompts, 250 new tokens</span>
      <span><b>D=128 sampling</b> temp=0.8, top_k=40</span>
      <span><b>Reference sampling</b> temp=0.7, top_k=50</span>
    </div>
  </header>

  <div class="stats-grid" id="stats-grid"></div>
  <div class="metrics-row" id="metrics-row"></div>

  <div class="controls">
    <div class="seg" id="sweep-seg">
      <button data-sweep="sweep1" class="active">Sweep 1 &middot; 60 tok</button>
      <button data-sweep="sweep2">Sweep 2 &middot; 250 tok</button>
    </div>
    <label class="toggle"><input type="checkbox" id="flagged-only"> flagged only</label>
    <div class="legend-strip">
      <span><mark>highlight</mark> = the exact span that tripped the detector</span>
      <span><span class="flagpill clean" style="margin-right:4px">clean</span>no repeated bigram &ge;3&times;, no doubling</span>
    </div>
  </div>

  <div id="rows"></div>
  <div id="empty-state">No samples match this filter.</div>

  <footer class="endnote">
    Generated fresh via <code>data/ckpt_stepC_d128_v16384.pt</code> (iter 8000,
    D=128/n_layer=12/n_head=2 &mdash; a separate, fresh training run at D=128 with Model C's
    exact recipe; its weights share no relationship with the D=384 checkpoint, confirmed by
    the saved checkpoint's own cfg and tensor shapes) vs <code>SauravP97/tiny-stories-19M</code>,
    same objective detector, same sampling convention as the original D=384 comparison. Rows
    pair identical (seed, prompt-theme) conditions across models for direct comparison.
  </footer>

</div>

<script id="report-data" type="application/json">__JSON_DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('report-data').textContent);

function escapeHtml(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function highlight(text, reason){
  if (!reason) return escapeHtml(text);
  let w1, w2;
  let m = reason.match(/^exact doubling: (['"])(.*?)\1 (['"])(.*?)\3$/);
  if (m){ w1 = m[2]; w2 = m[4]; }
  else {
    m = reason.match(/^bigram recurred \d+x: \((['"])(.*?)\1,\s*(['"])(.*?)\3\)$/);
    if (m){ w1 = m[2]; w2 = m[4]; }
  }
  if (w1 === undefined) return escapeHtml(text);
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(esc(w1) + '\\s?' + esc(w2), 'gi');
  let out = '', last = 0, mm;
  while ((mm = re.exec(text))){
    out += escapeHtml(text.slice(last, mm.index));
    out += '<mark>' + escapeHtml(mm[0]) + '</mark>';
    last = mm.index + mm[0].length;
    if (mm[0].length === 0) re.lastIndex++;
  }
  out += escapeHtml(text.slice(last));
  return out;
}

function wordCount(t){ return t.trim().split(/\s+/).length; }

function reasonKind(reason){
  if (!reason) return null;
  return reason.startsWith('exact doubling') ? 'doubling' : 'bigram';
}

function computeStats(samples){
  const total = samples.length;
  const flagged = samples.filter(s => s.reason).length;
  const avgWords = samples.reduce((a,s) => a + wordCount(s.text), 0) / total;
  const endCount = samples.reduce((a,s) => a + ((s.text.toLowerCase().match(/the end/g) || []).length), 0);
  const restartCount = samples.reduce((a,s) => a + ((s.text.toLowerCase().match(/once upon a time/g) || []).length), 0);
  const kinds = {bigram:0, doubling:0};
  samples.forEach(s => { const k = reasonKind(s.reason); if (k) kinds[k]++; });
  return {total, flagged, avgWords, endCount, restartCount, kinds};
}

function renderStats(){
  const groups = [
    {key:'sweep1', model:'D128', label:'Sweep 1 &middot; D=128', cls:'c'},
    {key:'sweep1', model:'reference', label:'Sweep 1 &middot; reference', cls:'r'},
    {key:'sweep2', model:'D128', label:'Sweep 2 &middot; D=128', cls:'c'},
    {key:'sweep2', model:'reference', label:'Sweep 2 &middot; reference', cls:'r'},
  ];
  const grid = document.getElementById('stats-grid');
  grid.innerHTML = groups.map(g => {
    const s = computeStats(DATA[g.key][g.model]);
    const pct = Math.round(100 * s.flagged / s.total);
    return `<div class="stat-tile ${g.cls}">
      <div class="sweep-label">${g.label}</div>
      <div class="num">${s.flagged}<small>/${s.total} flagged</small></div>
      <div class="pct">${pct}% of samples</div>
    </div>`;
  }).join('');

  const s1c = computeStats(DATA.sweep1.D128), s1r = computeStats(DATA.sweep1.reference);
  const s2c = computeStats(DATA.sweep2.D128), s2r = computeStats(DATA.sweep2.reference);
  const row = (label, c, r, fmt) => `<tr><td class="k">${label}</td><td class="v" style="color:var(--accent-c)">${fmt(c)}</td><td class="v" style="color:var(--accent-r)">${fmt(r)}</td></tr>`;
  const fmt1 = n => n.toFixed(2);
  document.getElementById('metrics-row').innerHTML = `
    <div class="metric-cell">
      <div class="mlabel">Sweep 1 (60 tok) &middot; per-sample rate, D=128 vs reference</div>
      <table>
        <tr><td class="k"></td><td class="v" style="color:var(--accent-c)">D128</td><td class="v" style="color:var(--accent-r)">ref</td></tr>
        ${row('avg word count', s1c.avgWords, s1r.avgWords, n=>n.toFixed(1))}
        ${row('&ldquo;the end&rdquo; / sample', s1c.endCount/s1c.total, s1r.endCount/s1r.total, fmt1)}
        ${row('story-restarts / sample', s1c.restartCount/s1c.total, s1r.restartCount/s1r.total, fmt1)}
      </table>
    </div>
    <div class="metric-cell">
      <div class="mlabel">Sweep 2 (250 tok) &middot; per-sample rate, D=128 vs reference</div>
      <table>
        <tr><td class="k"></td><td class="v" style="color:var(--accent-c)">D128</td><td class="v" style="color:var(--accent-r)">ref</td></tr>
        ${row('avg word count', s2c.avgWords, s2r.avgWords, n=>n.toFixed(1))}
        ${row('&ldquo;the end&rdquo; / sample', s2c.endCount/s2c.total, s2r.endCount/s2r.total, fmt1)}
        ${row('story-restarts / sample', s2c.restartCount/s2c.total, s2r.restartCount/s2r.total, fmt1)}
      </table>
    </div>
    <div class="metric-cell">
      <div class="mlabel">Flag kind breakdown (both sweeps combined)</div>
      <table>
        <tr><td class="k"></td><td class="v" style="color:var(--accent-c)">D128</td><td class="v" style="color:var(--accent-r)">ref</td></tr>
        ${row('bigram recurrence', s1c.kinds.bigram+s2c.kinds.bigram, s1r.kinds.bigram+s2r.kinds.bigram, n=>n)}
        ${row('exact doubling', s1c.kinds.doubling+s2c.kinds.doubling, s1r.kinds.doubling+s2r.kinds.doubling, n=>n)}
      </table>
    </div>
  `;
}

function pairRows(sweepKey){
  const cList = DATA[sweepKey].D128, rList = DATA[sweepKey].reference;
  const rows = [];
  for (let i = 0; i < cList.length; i++){
    rows.push({c: cList[i], r: rList[i]});
  }
  return rows;
}

let currentSweep = 'sweep1';
let flaggedOnly = false;

function panelHtml(sample, cls, label){
  const flagged = !!sample.reason;
  const badge = flagged
    ? `<span class="flagpill flagged">flagged</span>`
    : `<span class="flagpill clean">clean</span>`;
  const reasonHtml = flagged ? `<div class="reason">${escapeHtml(sample.reason)}</div>` : '';
  const hide = (flaggedOnly && !flagged) ? ' hidden' : '';
  return `<div class="panel${hide}">
    <div class="panel-head">
      <span class="modeltag ${cls}"><span class="dot ${cls}"></span>${label}</span>
      ${badge}
    </div>
    ${reasonHtml}
    <div class="story">${highlight(sample.text, sample.reason)}</div>
    <div class="panel-foot">
      <span>prompt: &ldquo;${escapeHtml(sample.prompt)}&rdquo;</span>
      <span>${wordCount(sample.text)} words</span>
    </div>
  </div>`;
}

function render(){
  const rows = pairRows(currentSweep);
  const container = document.getElementById('rows');
  let html = '';
  let visibleRows = 0;
  rows.forEach((pair, idx) => {
    const cFlagged = !!pair.c.reason, rFlagged = !!pair.r.reason;
    if (flaggedOnly && !cFlagged && !rFlagged) return;
    visibleRows++;
    html += `<div class="rowgroup">
      <div class="rowmeta"><span class="seedno">seed ${pair.c.seed}</span><span>condition ${idx+1} of ${rows.length}</span></div>
      <div class="pair">
        ${panelHtml(pair.c, 'c', 'checkpoint C, D=128')}
        ${panelHtml(pair.r, 'r', 'reference')}
      </div>
    </div>`;
  });
  container.innerHTML = html;
  document.getElementById('empty-state').style.display = visibleRows === 0 ? 'block' : 'none';
}

document.getElementById('sweep-seg').addEventListener('click', e => {
  const btn = e.target.closest('button');
  if (!btn) return;
  currentSweep = btn.dataset.sweep;
  document.querySelectorAll('#sweep-seg button').forEach(b => b.classList.toggle('active', b === btn));
  render();
});

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
print("wrote", len(HTML), "bytes")
