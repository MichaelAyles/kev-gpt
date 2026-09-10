# Word-vocab scale-up: does a bigger model fix story quality, and what breaks when you try

## Motivation

The deployed Genesys2 word-vocab checkpoint (D=128, NLAYER=12, NHEAD=2,
VOCAB=1900, val=2.022 — see `fabric/genesys2/PORT-NOTES.md`, "Word-level
vocabulary") produces real-hardware sampled chat that is rougher and more
repetitive than the char-level build's own sampled output ("to help him to
to help him get to help him get"). The open question: is that a model-
capacity problem (fixable by making the model bigger), or something else
(sampling, data, tokenizer)? This log is a pure PyTorch/GPU investigation —
no RTL, no synthesis, no Genesys2 BRAM constraints — meant to answer that
before spending any more real-hardware effort.

**Bottom line up front**: a bigger model (D=384, VOCAB=4096, ~8.7x the
deployed model's params) genuinely produces better prose when it trains
well — but this codebase's default training recipe (AdamW) has a real,
reproducible instability at this width that caps effective training to
roughly the first 3500-4500 iterations of any AdamW run, surviving lower
LR, longer warmup, GPT-2 residual-projection init scaling, excluding
LayerNorm gains from weight decay, logit-scale regularization at two
strengths, and even a completely frozen floor LR held for 10,000
iterations. **Root cause, confirmed in Attempt 7: it's specific to Adam's
optimizer dynamics, not the model, data, or width** — the identical setup
trained with plain SGD+momentum instead shows no divergence at all, just
a (much worse) stable plateau. The key finding for "what does it take to
get acceptable stories": **training stability at width is the actual
bottleneck right now, and it's fixable in principle by tuning Adam's own
hyperparameters (`eps`, `beta2`) rather than needing a different
optimizer or architecture — just not yet tried.**

## Environment / how to reproduce

```bash
cd kev-gpt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All runs below used an NVIDIA RTX 4060 Laptop GPU (8GB), `torch.cuda`
available, bf16 autocast (`model.train.amp_dtype` auto-selects bf16 on
CUDA when supported). Throughput at the D=384/VOCAB=4096 shape: ~12.3
it/s, ~100k tokens/s (`--smoke 50` on `model.train`). A 24000-iter run
takes ~48 minutes at this shape; a 10000-iter diagnostic probe ~19-21
minutes; a 6000-iter probe ~12 minutes.

**Corpus**: `data/TinyStories-train.filtered.txt` — the same filtered/
lowercased full TinyStories train split the deployed models use (see
`fabric/genesys2/PORT-NOTES.md`'s NLAYER=12 section for how this was
built). Not checked into the repo (`data/` is gitignored); regenerate via
`keviniser.fetch_tinystories --split train` + the filtering step described
there if starting from scratch.

**Baseline being compared against**: the deployed
D=128/NLAYER=12/NHEAD=2/VOCAB=1900 checkpoint, val=2.022, trained via:
```bash
python -m model.train --max-iters 24000 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 \
  --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16.pt --states data/states_word16.jsonl
```

## Step 0: data prep at the bigger vocab

```bash
python -m model.word_data data/TinyStories-train.filtered.txt data/word_big4096 4096
```
```
vocab_size=4096  train_tokens=414,307,096  val_tokens=4,184,920  unk_frac=0.0087
```
Real, useful data point on its own: `unk_frac` (fraction of tokens that
fall outside the vocab and get mapped to `<unk>`) drops from **0.0394 at
VOCAB=1900** to **0.0087 at VOCAB=4096** — under 1% out-of-vocabulary vs.
~4%. `train_tokens`/`val_tokens` are identical to the VOCAB=1900 prep
(same corpus, same tokenizer regex — only the vocab cutoff changes which
words get their own id vs. collapse to `<unk>`).

## Experiment 1: naive combined scale-up (D=384, NLAYER=12, NHEAD=6, VOCAB=4096)

Chosen deliberately as ONE big combined jump (not a careful one-axis-at-a-
time ablation) to answer "is this fixable by scale at all" fastest before
spending runs isolating which axis matters — `NHEAD=6` keeps head_dim=64,
matching this project's own established convention.

```bash
python -m model.train --max-iters 24000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 \
  --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096.pt --states data/states_word_big4096.jsonl
```

`params=22.82M` (~8.7x the deployed 2.6M model — computed directly by
`GPT.num_params()`, printed at startup).

| iter | train | val | lr |
|---|---|---|---|
| 500 | 2.829 | 2.815 | 1.0e-03 |
| 1000 | 2.495 | 2.481 | 1.0e-03 |
| 1500 | 2.370 | 2.358 | 9.9e-04 |
| 2000 | 2.327 | 2.312 | 9.9e-04 |
| **2500** | **2.322** | **2.309 (best)** | 9.8e-04 |
| 3000 | 2.343 | 2.330 | 9.7e-04 |
| 5000 | 2.523 | 2.510 | 9.1e-04 |
| 10000 | 3.262 | 3.239 | 6.7e-04 |
| 15000 | 4.162 | 4.143 | 3.8e-04 |
| 20000 | 4.934 | 4.911 | 1.6e-04 |
| 24000 | 5.634 | 5.608 | 1.0e-04 |

Train and val loss climb **together** for the entire back 21,500 iterations
of a 24000-iter run — not the "train keeps improving, val worsens"
signature of overfitting, but genuine instability (the model gets worse at
everything). `model.train`'s own auto-rollback correctly saved the
iter-2500 checkpoint (`data/ckpt_word_big4096.pt`, val=2.309) — nothing is
lost, but that checkpoint only saw ~10% of the intended schedule.

**Context**: this exact signature (train+val climbing together after an
early bottom) was already seen once before this investigation, at
D=256/NLAYER=12 **char-level** (see `fabric/genesys2/PORT-NOTES.md`'s
"NLAYER=16 and D=256" section) — where a lower LR was tried once and made
it *worse*. Two independent configs (different width, different vocab
scheme, different corpus tokenization) hitting the same signature is what
motivated treating this as a real, general phenomenon worth root-causing
rather than a one-off fluke.

## Diagnosing the instability

### Attempt 1: lower peak LR + longer warmup

Hypothesis: default `--lr 1e-3 --warmup 100` is too aggressive for a wider
model (a well-known phenomenon without width-aware LR scaling / muP).

**Short probe first (6000 iters)** — looked like it worked:
```bash
python -m model.train --max-iters 6000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_lowlr.pt --states data/states_word_big4096_lowlr.jsonl
```
Best val **2.291 @ iter 4000**, only a small uptick by iter 6000 (2.343).
Looked stable — **this was a false positive**, caught only by re-running
the full schedule (see below). Lesson embedded in every subsequent
diagnostic in this log: **a short probe is not sufficient evidence of
stability** — always extend to at least iter 10000 before trusting a fix.

**Same recipe, full 24000 iters**:
```bash
python -m model.train --max-iters 24000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_stable.pt --states data/states_word_big4096_stable.jsonl
```
Best val **2.332 @ iter 4000**, then the SAME climbing shape all the way to
**val=5.624 @ iter 24000**. The fix only delayed the onset (iter 2500 with
the default recipe -> iter 4000 here) without changing its character.
**Verdict: LR/warmup alone does not fix it.**

### Attempt 2: GPT-2 residual-projection init scaling

Hypothesis: this codebase's `GPT._init()` (`model/gpt.py`) applies a flat
`std=0.02` to every `nn.Linear`, including the two matrices that write
DIRECTLY into the residual stream (`attn.proj`, the second MLP `Linear`).
The original GPT-2 paper scales those specifically by
`1/sqrt(2*n_layer)` to keep residual-stream variance from growing
unboundedly with depth — this codebase never had that. Plausible root
cause since residual-variance blowup compounds with BOTH depth and width.

**Code change** (`model/gpt.py`, in `GPT.__init__`, after `self.apply(self._init)`):
```python
for pn, p in self.named_parameters():
    if pn.endswith("attn.proj.weight") or pn.endswith("mlp.2.weight"):
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
```
This is now a permanent, unconditional part of `GPT.__init__` — it affects
every future training run (no flag to disable), since it's a strict
correctness improvement to the init scheme regardless of this
investigation's outcome. Existing already-trained checkpoints are
unaffected (init only matters at construction time).

**Short probe (6000 iters, same LR=3e-4/warmup=2000)**:
```bash
python -m model.train --max-iters 6000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_initfix.pt --states data/states_word_big4096_initfix.jsonl
```
Best val **2.235 @ iter 4500** — slightly better than the non-init-fix
probe, but per the lesson above, not trusted on its own.

**Full 24000 iters**:
```bash
python -m model.train --max-iters 24000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_initfix_full.pt --states data/states_word_big4096_initfix_full.jsonl
```
Best val **2.271 @ iter 3500**, then the same climb to **val=5.531 @ iter
24000**. **Verdict: the init fix does not resolve the instability either**
(kept as a permanent improvement regardless — it's still correct practice
— but it isn't the fix).

**Weight-norm comparison** (best checkpoint, iter 3500, vs. the diverged
end, iter 24000 — both pulled from `data/ckpts/ckpt_03500.pt` and
`data/ckpts/ckpt_24000.pt`, the per-eval snapshot dir; **note**: this
snapshot dir is shared across ALL runs in this log since none of the
commands above override `--snapshot-dir`, so files get overwritten by
whichever run last wrote that iter number — only trust these snapshots
immediately after the run you care about, or pass a distinct
`--snapshot-dir` per run if repeating this):
```python
import torch
for f in ["data/ckpts/ckpt_03500.pt", "data/ckpts/ckpt_24000.pt"]:
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    print(f, "iter", ckpt["iter"], "val", ckpt["val"])
    print("  tok_emb norm", sd["tok_emb.weight"].norm().item())
    print("  block0 qkv norm", sd["blocks.0.attn.qkv.weight"].norm().item())
    print("  block11 mlp.2 norm", sd["blocks.11.mlp.2.weight"].norm().item())
    print("  ln_f weight mean/std", sd["ln_f.weight"].mean().item(), sd["ln_f.weight"].std().item())
```
| | iter 3500 (best) | iter 24000 (diverged) |
|---|---|---|
| val | 2.271 | 5.531 |
| tok_emb norm | 48.79 | 59.03 (+21%) |
| block0 qkv norm | 15.44 | 24.93 (+61%) |
| block11 mlp.2 norm | 9.97 | 25.93 (+160%) |
| ln_f weight mean | 1.45 | 2.81 (+93%) |

Weight norms, especially the final LayerNorm's own gain, grow
substantially over the divergent period — the seed for the logit-scale
hypothesis tested next.

**Qualitative check**: samples from the iter-3500 checkpoint (val=2.271,
"worse" by raw val loss than the deployed model's val=2.022) are
noticeably MORE coherent than the deployed model's real-hardware sampled
output:
```python
import torch
from model.gpt import GPT, GPTConfig
from model.train import sample, pick_device

device = pick_device("auto")
ck = torch.load("data/ckpts/ckpt_03500.pt", map_location=device, weights_only=False)
model = GPT(GPTConfig(**ck["cfg"])).to(device)
model.load_state_dict(ck["model"]); model.eval()
for p in ["once upon a time", "the dog ran", "a little girl"]:
    print(p, "->", sample(model, ck["meta"], device, n_tokens=80, prompt=p, seed=42))
```
```
"once upon a time" -> ", there was a girl named lily. she loved to play outside and pick
  flowers. one day, she found a box that was very dirty. she wanted to find a way to make it
  look nice. but, she had an idea. she saw a big jar of orange, a shiny <unk>. it was yellow,
  and her mommy gave her a bottle. lily was excited and started to help."
"the dog ran"       -> " to the tree and gave it a hug. the squirrel said, 'you are very
  brave and strong. you can help me and the dog.' they both got a band-aid from the tree and
  helped the dog. the dog felt proud and happy. tim and his friend played with the dog every
  day. they were happy and cozy."
"a little girl"     -> " named lily. one day, lily and her mommy went to the park. there,
  the park and the swings. it was so pretty! there, they saw many kids playing with a ball."
```
Coherent, grammatical, far less repetitive than the real-hardware
"to help him to to help him get to help him get" pattern — despite a
nominally worse val loss (not directly comparable across different vocab
sizes: cross-entropy over 4096 classes isn't the same scale as over 1900)
and despite far fewer total training iterations. **This is the strongest
evidence that capacity genuinely helps quality when the model actually
gets to train** — the practical blocker is training length, not
architecture capacity.

### Attempt 3: exclude LayerNorm gains/biases from weight decay + add logit-scale instrumentation

Hypothesis refinement: `AdamW(model.parameters(), ..., weight_decay=0.1)`
applies decay uniformly to EVERY parameter, including 1-D LayerNorm gains
and biases — standard GPT-2/nanoGPT practice explicitly excludes those
(decay only >=2D weight matrices). Also added direct instrumentation
(logit std/max, `ln_f` gain norm) to every eval step instead of inferring
weight-scale drift from snapshot diffs after the fact.

**Code change** (`model/train.py`, optimizer construction):
```python
decay, no_decay = [], []
for p in model.parameters():
    if not p.requires_grad:
        continue
    (decay if p.dim() >= 2 else no_decay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95))
```
Also permanent/unconditional — a strict correctness improvement kept
regardless of outcome.

**Code change** (`model/train.py`, eval block — logit-scale diagnostic):
```python
model.eval()
with torch.no_grad():
    xs, _ = get_batch(splits["val"], block, batch, device)
    with ctx:
        logits_dbg, _ = model(xs, xs)
    logit_std = logits_dbg.float().std().item()
    logit_max = logits_dbg.float().abs().max().item()
lnf_norm = model.ln_f.weight.norm().item()
model.train()
```
Printed alongside the usual `iter/train/val/lr` line as
`logit_std {..} logit_max {..} lnf_norm {..}`.

**Probe, this time 10000 iters** (lesson from the false-positive 6000-iter
probes earlier — 10000 gives enough runway past the ~4000-iter onset
point to actually see whether a fix changes the trajectory):
```bash
python -m model.train --max-iters 10000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_wdfix.pt --states data/states_word_big4096_wdfix.jsonl
```
| iter | val | logit_std | logit_max | lnf_norm |
|---|---|---|---|---|
| 0 | 8.400 | 0.39 | 2.5 | 19.60 |
| 2000 | 2.529 | 1.68 | 15.9 | 24.61 |
| **3500** | **2.262 (best)** | 1.87 | 19.6 | 29.98 |
| 5000 | 2.342 | 2.03 | 21.2 | 34.20 |
| 10000 | 3.221 | 2.39 | 31.8 | 41.24 |

Confirms the mechanism directly: `logit_std`, `logit_max`, and `lnf_norm`
all grow **smoothly and continuously from iteration 0** — they do NOT
spike specifically at the point training turns bad. Growing logit scale
is actually *helping* early (sharper, more confident correct predictions
lower loss) and only becomes harmful once the underlying ranking accuracy
stops improving, at which point the same growing confidence starts
amplifying wrong predictions instead. **Verdict: real, measurable
mechanism, but not itself the root trigger** — something else causes the
ranking-accuracy plateau/reversal around iter 3500-4000; unbounded logit
growth (nothing in this architecture caps it) then makes the consequences
worse than they'd otherwise be. Val at iter 10000 here (3.221) is close
to, not clearly better than, the earlier un-decayed-LN-gain runs (3.066
"stable", 3.085 "initfix_full") — the weight-decay fix alone made no
clear difference either, kept as a correctness improvement regardless.

### Attempt 4: z-loss (logit-scale penalty), weak then strong

Directly targets the confirmed-real mechanism from Attempt 3: penalize
`logsumexp(logits)^2` (the log-normalizer cross-entropy already computes
internally) so logit scale is capped by the loss function itself instead
of hoping weight decay controls it indirectly. Standard technique from
PaLM/ST-MoE, coefficient ~1e-4 there.

**Code change** (`model/gpt.py`, `GPTConfig`):
```python
z_loss_coef: float = 0.0     # 0.0 = off (unchanged default behavior)
```
**Code change** (`model/gpt.py`, `GPT.forward()`):
```python
flat_logits = logits.view(-1, logits.size(-1))
loss = F.cross_entropy(flat_logits, targets.view(-1))
if self.cfg.z_loss_coef > 0:
    z = torch.logsumexp(flat_logits, dim=-1)
    loss = loss + self.cfg.z_loss_coef * z.pow(2).mean()
```
**Code change** (`model/train.py`, new CLI flag `--z-loss-coef`, default
`0.0`, threaded into `GPTConfig`). **Also**: `estimate_loss()` was patched
to temporarily zero `model.cfg.z_loss_coef` during eval so reported `val`
stays pure cross-entropy and comparable across z-loss on/off runs (z-loss
is a training regularizer, not part of the tracked quality metric):
```python
z_loss_coef = model.cfg.z_loss_coef
model.cfg.z_loss_coef = 0.0
# ... compute val loss as before ...
model.cfg.z_loss_coef = z_loss_coef
```

**Weak (1e-4, the PaLM default), 10000 iters**:
```bash
python -m model.train --max-iters 10000 --lr 3e-4 --warmup 2000 --z-loss-coef 1e-4 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_zloss.pt --states data/states_word_big4096_zloss.jsonl
```
Best val **2.261 @ iter 3500**, then the same climb to val=3.167 @ iter
10000 (logit_max=30.1, lnf_norm=41.14) — **nearly identical to the
no-z-loss run** (val=3.221, logit_max=31.8, lnf_norm 41.24) at every
tracked iter. At this scale (logit_max~30), the z-loss gradient
contribution is roughly `2 * 1e-4 * 30 ≈ 0.006` per logit — tiny next to
cross-entropy's own gradient (order 0.1-1) — too weak to move anything
measurably.

**Strong (1e-2, 100x), 10000 iters**:
```bash
python -m model.train --max-iters 10000 --lr 3e-4 --warmup 2000 --z-loss-coef 1e-2 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_zloss2.pt --states data/states_word_big4096_zloss2.jsonl
```
| iter | val | logit_std | logit_max | lnf_norm |
|---|---|---|---|---|
| **4000** | **2.244 (best)** | 2.86 | 27.8 | 31.69 |
| 10000 | 2.931 | 4.05 | 37.2 | 40.91 |

Counterintuitively, `logit_max` grew to **37.2** — HIGHER than either
weaker run (30.1-31.8) — yet the val-loss trajectory shows the exact same
shape and timing as every other run (best at iter 4000, climbing
afterward). **Verdict: z-loss, at 100x the standard strength, changed
logit growth (made it worse, if anything) but had zero measurable effect
on when or how severely the divergence happens.** This rules out
logit-scale growth as causal — it's a correlated symptom, not the
trigger.

### Attempt 5: per-layer gradient-norm logging

The next step flagged in the original write-up: instead of another
loss-function-level guess, instrument WHERE in the network the gradients
change character, to see if one specific layer/component is the trigger.

**Code change** (`model/train.py`): a new `layer_grad_norms(model)`
helper groups every parameter's gradient L2 norm by `blockN.{ln1,ln2,
qkv,proj,mlp_fc,mlp_proj}`, plus `tok_emb` (tied with `head`), `pos_emb`,
and `ln_f` — called right after `scaler.unscale_(opt)` and before
`clip_grad_norm_` (clipping is a single global rescale, so it doesn't
change the RELATIVE proportions between layers, but pre-clip keeps the
raw numbers meaningful on their own). New CLI flags: `--layer-grad-log
PATH` (JSONL output, one line per logged step) and `--layer-grad-interval
N` (default 100).

```bash
python -m model.train --max-iters 8000 --lr 3e-4 --warmup 2000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --layer-grad-log data/gradlog_word_big4096.jsonl --layer-grad-interval 50 \
  --out data/ckpt_word_big4096_gradlog.pt --states data/states_word_big4096_gradlog.jsonl
```
Same val trajectory as every other run at this recipe (best val=2.257 @
iter 3500, climbing to 2.720 @ iter 8000) — confirms this run is
representative before trusting the gradient data.

**Per-block totals** (sum of that block's 6 component norms), sampled
every 500 iters:
```
iter   tok_emb  pos_emb  ln_f    blk0   blk1   blk2   blk3  ...  blk9  blk10  blk11
   0     1.58    0.55   0.03    7.64   7.67   7.00   6.39  ...  4.48   4.49   4.26
2500     0.39    0.22   0.01    1.04   0.57   0.64   0.66  ...  0.30   0.31   0.33
3500     0.39    0.23   0.01    1.06   0.50   0.62   0.64  ...  0.35   0.35   0.37
5000     0.41    0.30   0.01    1.38   0.65   0.78   0.85  ...  0.53   0.49   0.46
7500     0.47    0.39   0.01    1.90   0.94   1.24   1.31  ...  0.84   0.76   0.65
```
(full file: `data/gradlog_word_big4096.jsonl`, gitignored, one JSON object
per logged iter with all 75 per-component keys — regenerate via the
command above.)

**Finding: EVERY block bottoms out and re-grows in lockstep, uniformly
across depth and across attention/MLP sub-components** — this is a
network-wide, synchronized phenomenon, not one layer misbehaving:

| group | grad-norm minimum | @ iter | value at iter 7950 | relative growth |
|---|---|---|---|---|
| tok_emb (tied w/ head) | 0.356 | 2100 | 0.459 | **+29%** |
| block0 (total) | 0.942 | 2900 | 1.954 | +107% |
| block5 (total) | 0.446 | 2950 | 1.121 | +151% |
| block11 (total) | 0.305 | 2350 | 0.654 | +115% |

Every transformer block's gradient norm bottoms out at almost exactly the
same iteration (2350-2950) that val loss bottoms (3500) and grows by a
similar, large relative amount (107-151%) regardless of depth — checked
down to the qkv/proj/mlp_fc/mlp_proj component level (block0 vs block11
compared directly), same synchronized bottom-then-grow shape everywhere.
**tok_emb/head is the one clear outlier: its gradient norm stays far more
stable (+29% vs +107-151% for the transformer stack).**

**Interpretation**: this rules out "one specific layer is the trigger" —
whatever is happening, it's not localized to a particular depth or to
attention vs. MLP specifically. The one real asymmetry (embedding/head
comparatively stable, the full 12-layer transformer stack uniformly
unstable) also weakens a "big VOCAB=4096 tied embedding/head is the
problem" hypothesis, since that's exactly the layer that stays *most*
stable. The synchronized, network-wide growth is consistent with a
genuine optimization feedback loop rather than a localized bug: cross-
entropy gradients scale with prediction error, so once the model drifts
even slightly worse from its iter-3500 optimum (for whatever underlying
reason — plausibly just normal SGD noise failing to stay inside a narrow
minimum at this width, with LR still only partially decayed at that
point), every block's gradients grow together, reinforcing the drift
further. This is a plausible mechanism, not a proven one — it wasn't
tested directly (e.g., by comparing gradient-noise magnitude/variance
across widths, or checking if a much longer, gentler LR decay avoids ever
leaving the iter-3500 basin in the first place). **Root cause is still
open**; this rules out where it ISN'T (a single bad layer, the embedding/
head) more than it pins down where it IS.

### Attempt 6: longer/gentler LR decay — get to a low LR fast, then hold it there

The next step flagged above: instead of stretching decay across the WHOLE
run (so LR is still ~80-90% of peak right when the model usually leaves
its optimum, iter ~3500-4000), decouple the decay horizon from total
training length — ramp down to the LR floor FAST, then just keep training
at that tiny, fixed LR for a long time. Tests directly: does the model
stay near its optimum once it's off the high-LR plateau, or does it drift
away regardless of LR?

**Code change** (`model/train.py`): new `--lr-decay-iters` flag,
decoupled from `--max-iters` — `cosine_lr`'s own `total` parameter (which
governs when LR reaches its floor and holds there, via its existing
`if it > total: return lr * min_lr_frac` branch) now defaults to
`--max-iters` (old behavior, unchanged) but can be set independently.

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 5000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_gentledecay.pt --states data/states_word_big4096_gentledecay.jsonl
```
LR ramps over 0-2000 (warmup), decays over 2000-5000 (reaching floor
`0.1*3e-4=3e-5` right at iter 5000), then **holds flat at 3.0e-05 for all
10,000 remaining iters (5000-15000)**.

| iter | train | val | lr |
|---|---|---|---|
| 3500 | 2.254 | 2.231 | 1.6e-04 |
| 4000 | 2.231 | 2.209 | 9.8e-05 |
| **4500** | **2.225** | **2.203 (best of the ENTIRE investigation)** | 4.8e-05 |
| 5000 | 2.242 | 2.219 | 3.0e-05 (floor reached) |
| 6000 | 2.304 | 2.282 | 3.0e-05 (holding) |
| 8000 | 2.496 | 2.475 | 3.0e-05 (holding) |
| 10000 | 2.814 | 2.792 | 3.0e-05 (holding) |
| 12500 | 3.365 | 3.344 | 3.0e-05 (holding) |
| 15000 | 3.598 | 3.570 | 3.0e-05 (holding) |

**This is a decisive negative result: the model diverges even at a
completely frozen, near-zero learning rate held for two-thirds of the
run.** Train and val loss both climb continuously and substantially
(val +61% from iter 5000 to 15000) with LR literally unchanged the whole
time. **This conclusively rules out the LR-schedule hypothesis
entirely** — every variant tried (lower peak, longer warmup, faster
decay, and now a fully pinned floor LR) shows the same divergence, so it
was never really about LR magnitude or shape.

**What this points to instead**: Adam's per-parameter update is
variance-normalized (`lr * m_hat / (sqrt(v_hat) + eps)`), not simply
proportional to raw gradient magnitude — so even at `lr=3e-5`, the
*normalized* step size per parameter doesn't shrink to zero, and
compounded over 10,000 steps that's still a meaningful cumulative drift
independent of the nominal LR value. This would explain why NO amount of
LR tuning fixes it: the mechanism producing the drift isn't LR-gated at
all. Not tested directly in this pass (candidates for whoever picks this
up next): plain SGD instead of AdamW (removes the per-parameter variance
normalization entirely, a clean way to test this theory); an even more
extreme floor LR (e.g. 1e-6) to see if the drift ever actually stops;
`weight_decay=0` to rule out decay-driven erosion; or replaying a FIXED
batch sequence across runs to rule out data-order effects instead of
optimizer dynamics.

### Attempt 7: plain SGD instead of AdamW — the direct test, and the answer

The most direct test of Attempt 6's leading hypothesis: SGD's update is
directly proportional to the raw gradient (with momentum), with no
per-parameter variance normalization at all. If the divergence still
happens with SGD, Adam's normalization is cleared. If it doesn't, that's
the root cause.

**Code change** (`model/train.py`): new `--optimizer {adamw,sgd}` flag
(default `adamw`, unchanged behavior). SGD uses `momentum=0.9`, the same
decay/no_decay parameter grouping as AdamW (weight decay only on >=2D
matrices).

**LR calibration first** (SGD needs a much higher LR than Adam — no
adaptive per-parameter scaling to compensate for raw gradient magnitude).
Three short 1000-iter probes at `--warmup 200`:

| lr | val @ iter 1000 |
|---|---|
| 0.03 | 4.549 |
| 0.1 | 4.753 (worse) |
| 0.3 | 5.743 (worse, visibly noisy: train loss went UP from iter 200→600) |

`0.03` was the best of the three (higher values were actively worse, not
just slower) — used for the real run.

**Real run**, same ramp/decay/hold structure as Attempt 6 (swap only the
optimizer), given a longer decay horizon since SGD converges slower:
```bash
python -m model.train --max-iters 15000 --optimizer sgd --lr 0.03 --warmup 2000 --lr-decay-iters 6000 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_sgd.pt --states data/states_word_big4096_sgd.jsonl
```
| iter | train | val | lr | logit_max | lnf_norm |
|---|---|---|---|---|---|
| 2000 (peak lr) | 4.752 | 4.743 | 3.0e-02 | 9.6 | 22.74 |
| 4000 | 4.517 | 4.501 | 1.7e-02 | 11.0 | 24.43 |
| **6000 (floor reached)** | 4.421 | **4.409 (best)** | 3.0e-03 | 10.8 | 24.99 |
| 8000 | 4.427 | 4.415 | 3.0e-03 (holding) | 10.8 | 25.20 |
| 10000 | 4.426 | 4.414 | 3.0e-03 (holding) | 10.8 | 25.39 |
| 12500 | 4.418 | 4.406 | 3.0e-03 (holding) | 10.9 | 25.63 |
| 15000 | 4.414 | **4.401** | 3.0e-03 (holding) | 10.9 | 25.85 |

**No divergence.** Val loss settles at iter ~6000 (right as LR reaches
its floor) and then stays completely flat — fluctuating in a narrow
4.401-4.416 band — for the remaining 9000 iterations, the exact same
hold-period length where AdamW (Attempt 6, identical model/data/LR-decay
structure) diverged by +61%. `logit_max` and `lnf_norm` plateau too
(logit_max barely moves at all, 10.8→10.9 across 9000 iters, vs AdamW's
continuous climb).

**This is conclusive: the instability is specific to Adam's optimizer
dynamics, not the model, the data, or the width itself.** The obvious
caveat: SGD's absolute loss (4.401) is far worse than AdamW's best
(2.203) — SGD is much less sample-efficient for transformers at this
scale, well-documented in the broader literature and confirmed here, not
a surprise. This isn't a usable fix on its own (nobody wants a model
stuck at val=4.4 when val=2.2 is reachable), but it decisively answers
the ROOT CAUSE question and points at a much narrower, more promising
next step than anything tried before: **target Adam's specific
mechanism directly rather than abandoning Adam** — most directly, try a
much larger `eps` in `AdamW(..., eps=...)` (default `1e-8`; something
like `1e-3` or `1e-2` would make the `sqrt(v_hat)+eps` denominator less
aggressive at small gradient-variance estimates, closer to un-normalized
SGD-like behavior while keeping Adam's momentum/adaptivity elsewhere).
Also worth trying: lower `beta2` (default `0.95` here; a value like
`0.9` or lower makes the variance estimate track recent gradients more
responsively instead of averaging over a longer window that may be
slow to "notice" it should shrink the effective step). Neither was
tried in this pass — this is the concrete next step, not yet taken.

## Summary: every intervention tried, and what it did

| Intervention | Best val | Divergence onset | Changed the core instability? |
|---|---|---|---|
| Default (LR=1e-3, warmup=100) | 2.309 @ iter 2500 | ~iter 2500 | baseline |
| LR=3e-4, warmup=2000 | 2.332 @ iter 4000 | ~iter 4000 | delayed onset only |
| + GPT-2 residual-projection init | 2.271 @ iter 3500 | ~iter 3500 | no |
| + weight decay excludes LN gains/biases | 2.262 @ iter 3500 | ~iter 3500 | no |
| + z-loss @ 1e-4 | 2.261 @ iter 3500 | ~iter 3500 | no (too weak to engage) |
| + z-loss @ 1e-2 (100x) | 2.244 @ iter 4000 | ~iter 4000 | no (engaged, still no effect) |
| per-layer gradient logging (no fix, pure diagnostic) | 2.257 @ iter 3500 | ~iter 3500 | n/a — localizes WHERE, not a fix: uniform across all 12 blocks, embedding/head comparatively stable |
| decay-iters=5000, held flat past that (fully pinned floor LR) | **2.203 @ iter 4500 (best overall)** | ~iter 4500, THEN CONTINUES for 10,000 more iters at frozen LR | **no — rules out LR entirely** |
| plain SGD+momentum (same ramp/decay/hold structure) | 4.409 @ iter 6000 | ~iter 6000, then FLAT for 9,000 more iters (4.401-4.416 band) | **N/A — no divergence at all; identifies Adam as the cause** |
| AdamW, `adam-beta2=0.9` (lower variance-EMA decay) | 2.218 @ iter 4500 | ~iter 4500, tracks the AdamW-default divergence almost exactly | **no — beta2 isn't the mechanism** |
| AdamW, `adam-eps=1e-3` | **2.743 @ iter 11500-13500 (genuinely flat, not a snapshot)** | none — holds flat for the entire 10,000-iter window | **yes — fixes it, efficiency mostly retained** |
| AdamW, `adam-eps=1e-2` | 3.919 @ iter 15000 (still slowly falling) | none | **yes — fixes it, but efficiency mostly lost (~SGD territory)** |
| AdamW, `adam-eps=3e-4` + gentler decay (decay-iters=8000) | 2.365 @ iter 7500 | ~iter 7500-8000, then climbs 3-4x slower than default AdamW | **partial — dampens but doesn't eliminate; best peak of any config, but not stable** |
| AdamW, `adam-eps=3e-4` at the *original* decay length (decay-iters=5000, filling Attempt 8's gap) | 2.426 @ iter 7500 | ~iter 8000, drifts +0.111 to iter 15000 | **partial — meaningfully better peak than eps=1e-3 at the same decay length, some residual drift** |
| AdamW, `adam-eps=3e-4` + even gentler decay (decay-iters=10000) | 2.391 @ iter 6500 — worse than decay-iters=8000's 2.365 | drifts +0.262 to iter 15000, worst of the three eps=3e-4 decay lengths | **worse on both axes — eps=3e-4's decay-length optimum sits at ~8000, unlike eps=1e-3's which improves up to ~10000** |
| AdamW, `adam-eps=1e-3` + gentler decay (decay-iters=10000) | **2.606 @ iter 9500** | ~iter 9500-10000, then drifts +0.052 over 5000 iters (~4x gentler than the eps=3e-4 combo) | **yes, close to fully — best practical tradeoff of the sweep** |
| AdamW, `adam-eps=1e-3` + even gentler decay (decay-iters=13000) | 2.607 @ iter 10500 (plateau, no improvement over decay-iters=10000) | ~iter 11000 (starts before the floor), drifts +0.058 over 4500 iters (same magnitude as decay-iters=10000) | **plateau — recipe already found its ceiling at decay-iters~10000** |
| Same recipe (`eps=1e-3`, decay@10000) ported to the deployment shape (D=128/NLAYER=12/VOCAB=1900) | 2.672 @ iter 23000, still improving | none — this run didn't diverge (see below: not guaranteed) | **N/A — negative transfer: worse than that shape's own existing recipe (val 2.022 @ iter 23000)** |
| Reproducing the deployment shape's *original* recipe (lr=5e-4, warmup=100, full cosine, default eps) | 2.208 @ iter 7500, then climbs to 2.398 @ iter 24000 | ~iter 7500-8000, continues the whole rest of the run — same shape of divergence as the wide model, milder | **N/A — a second run of the exact original recipe diverges; the clean 2.022 result wasn't guaranteed by the recipe alone** |
| Same original recipe + mild `adam-eps=1e-5` margin | 2.208 @ iter 7500 (same peak), 2.394 @ iter 24000 | same as unstabilized — no measurable change | **no — dose too small to matter at this width** |
| Same original recipe + mild `adam-eps=1e-4` margin | 2.213 @ iter 10000-10500 (~same peak, negligible cost), 2.310 @ iter 24000 | still drifts, but ~half the rate (+0.097 vs +0.186-0.190) and starting later | **yes — meaningfully reduces divergence risk at negligible peak-quality cost** |
| Same original recipe + `adam-eps=3e-4` margin | **2.203 @ iter 19500-20000 (best of all deployment-shape runs tried), 2.211 @ iter 24000** | none — genuinely flat (+0.008 over the last ~4500 iters) | **yes, best result — beats the unstabilized peak and is stable, superseding the eps=1e-4 recommendation** |

Every AdamW configuration bottoms at essentially the same *relative*
point — roughly 1500-2500 iterations past wherever warmup ends —
regardless of peak LR, warmup length, init scaling, weight-decay
grouping, or logit-scale regularization strength. Attempt 6 went further
and pinned LR completely flat at its floor for two-thirds of an entire
run, and the model STILL diverged just as much, ruling out the entire
LR-schedule axis. **Attempt 7 then closed the loop**: the exact same
model/data/ramp-decay-hold structure, with plain SGD instead of AdamW,
shows NO divergence at all — it settles and stays flat. Combined with
Attempt 5's finding (growth synchronized across the whole network, not
localized to one layer), **the root cause is now identified as Adam's
optimizer dynamics specifically** (most likely the variance-normalized
per-parameter step size not actually shrinking at low nominal LR), not
the model, the data, the width, or the LR schedule. **Attempt 8 then
found the actual fix**: a much larger `adam-eps` (`1e-3` or `1e-2`)
eliminates the divergence entirely, while a lower `adam-beta2` (`0.9`)
does not — pinning the mechanism specifically to the `sqrt(v_hat)+eps`
denominator floor, not the variance EMA's window length.

### Attempt 8: target Adam's mechanism directly — `eps` fixes it, `beta2` doesn't

Attempt 7 narrowed the fix to two candidates on Adam's own denominator
(`lr * m_hat / (sqrt(v_hat) + eps)`): a much larger `eps`, or a lower
`beta2`. This attempt tests both, isolated, against the exact same
ramp/decay/hold structure as Attempts 6 and 7 (`--lr 3e-4 --warmup 2000
--lr-decay-iters 5000 --max-iters 15000`, floor `3.0e-05` held from
iter 5000 to 15000).

**Code change** (`model/train.py`): new `--adam-eps` (default `1e-8`,
torch's own default) and `--adam-beta2` (default `0.95`, this codebase's
prior default) flags, both wired into the `AdamW(...)` constructor.
`--optimizer sgd` is unaffected (still fixed `momentum=0.9`).

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 5000 --adam-eps 1e-2 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps1e2.pt --states data/states_word_big4096_eps1e2.jsonl
# repeat with --adam-eps 1e-3 --out ..._eps1e3.pt --states ..._eps1e3.jsonl
# repeat with --adam-beta2 0.9 (eps left at default) --out ..._beta2_0.9.pt --states ..._beta2_0.9.jsonl
```

| iter | eps=1e-2 val | eps=1e-3 val | eps=3e-4 val | beta2=0.9 val | baseline (Attempt 6) val |
|---|---|---|---|---|---|
| 4000 | 4.095 | 2.876 | 2.509 | 2.225 | 2.209 |
| **4500** | 4.069 | 2.830 | 2.461 | **2.218 (best)** | **2.203 (best)** |
| 5000 (floor reached) | 4.056 | 2.808 | 2.442 | 2.236 | 2.219 |
| 6000 | 4.038 | 2.791 | 2.432 | 2.298 | 2.282 |
| **7500** | 4.014 | 2.772 | **2.426 (best)** | 2.429 | (not logged at this iter) |
| 8000 | 4.006 | 2.766 | 2.427 | 2.486 | 2.475 |
| 10000 | 3.978 | 2.751 | 2.437 | 2.802 | 2.792 |
| 12500 | 3.947 | **2.743 (best, flat 11500-13500)** | 2.473 | 3.237 | 3.344 |
| 15000 | **3.919 (best, still slowly falling)** | 2.746 | 2.537 | 3.527 | 3.570 |

(`eps=3e-4`'s own command: identical flags to `eps=1e-3`'s above except
`--adam-eps 3e-4 --out data/ckpt_word_big4096_eps3e4_decay5000.pt
--states data/states_word_big4096_eps3e4_decay5000.jsonl` — run
separately, later, to fill in this exact gap; see the addendum after
Attempt 9 below.)

**`--adam-beta2 0.9` does not fix it.** Its whole trajectory tracks the
AdamW-default baseline almost exactly, iter for iter (2.218 vs 2.203
best, 2.298 vs 2.282 at iter 6000, 3.527 vs 3.570 at iter 15000) —
lowering beta2 alone leaves the instability fully intact. This rules
out the "variance EMA window too slow to react" half of Attempt 7's
hypothesis.

**Both `--adam-eps` values fix it — no divergence, in either case, for
the entire 10,000-iteration hold period.** `eps=1e-2` is stable but
pays a steep sample-efficiency cost, settling near val=4.0 (closer to
SGD's 4.4 than to AdamW's 2.2 — a large `eps` mostly cancels out Adam's
variance normalization, which is exactly the mechanism responsible, so
this is the expected trade). `eps=1e-3` is the interesting result: it
converges smoothly to **val=2.743 and then genuinely holds flat**
(2.743-2.746 across iters 11500-15000, not still falling and not
climbing) — a real converged optimum, not a lucky early snapshot ahead
of divergence. That number (2.743) is numerically worse than the
default-AdamW/beta2=0.9 *best-checkpoint* value (2.203/2.218), but that
comparison is misleading: 2.203 was never a stable point the model
settled at — it was the single best moment glimpsed on the way to a run
that eventually reached val=3.57. `eps=1e-3` gives up some of that peak
but is a genuine floor, reachable without needing the auto-rollback to
gamble-catch it.

**This closes the loop Attempt 7 opened**: the fix is `eps`, not
`beta2`. `eps=1e-3` is the practical choice — meaningfully more
efficient than `1e-2` and, unlike default `1e-8`/`beta2=0.9`, converges
to a value it actually stays at rather than diverging away from. Not
yet tried: whether combining a moderate `eps` increase (e.g. `3e-4` to
`1e-3`) with a longer/gentler decay recovers closer to the ~2.2 peak
while keeping the flat-hold stability — the natural next calibration
step for anyone extending this further.

### Attempt 9: `eps=3e-4` + a gentler decay — dampens the divergence, doesn't eliminate it

Attempt 8's closing question: does a moderate `eps` (between the
default `1e-8` and the fix-strength `1e-3`) combined with a
longer/gentler LR decay recover closer to the ~2.2 peak while keeping
`eps=1e-3`'s flat-hold stability?

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 8000 --adam-eps 3e-4 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps3e4_decay8000.pt --states data/states_word_big4096_eps3e4_decay8000.jsonl
```
`lr-decay-iters=8000` (vs 5000 in every prior attempt) stretches the
decay from 2000-8000 instead of 2000-5000, reaching the same floor
(`3.0e-05`) later and holding it for 7000 iters instead of 10,000.

| iter | train | val | lr |
|---|---|---|---|
| 4000 | 2.541 | 2.523 | 2.3e-04 |
| 6000 | 2.409 | 2.390 | 9.8e-05 |
| **7500** | **2.384** | **2.365 (best)** | 3.5e-05 |
| 8000 (floor reached) | 2.389 | 2.369 | 3.0e-05 |
| 10000 | 2.427 | 2.408 | 3.0e-05 (holding) |
| 12500 | 2.497 | 2.479 | 3.0e-05 (holding) |
| 15000 | 2.597 | 2.580 | 3.0e-05 (holding) |

**Partial result: this beats `eps=1e-3`'s peak (2.365 vs 2.743) but
does not reproduce its flat-hold stability.** Val bottoms right around
where LR reaches its floor (iter 7500-8000, matching the pattern of
every earlier AdamW run) and then climbs steadily for the remaining
7000 iters — a real divergence, not noise, but a much gentler slope
than any other AdamW configuration tried (2.365→2.580 over 7000 iters
here, vs default AdamW's 2.203→3.570 over 10,000 iters in Attempt 6, or
`beta2=0.9`'s 2.218→3.527 in Attempt 8 — roughly 3-4x slower per-iter
growth). So `eps=3e-4` measurably dampens the instability relative to
default `eps=1e-8`, but doesn't cross the threshold `eps=1e-3` and
`eps=1e-2` do into genuine flatness — this is a continuous effect (more
`eps` = less divergence), not a hard on/off switch, and `3e-4` sits
partway across it. The best-val checkpoint (2.365 @ iter 7500) is
still, like the pre-Attempt-8 runs, a lucky early snapshot the
auto-rollback has to catch rather than a point the model settles at and
stays.

**Answer to Attempt 8's open question**: no, this combo doesn't get
both the ~2.2-ish peak and flat-hold stability simultaneously — there's
an efficiency/stability tradeoff along the `eps` axis, and `1e-3` (not
`3e-4`) is the practical operating point already found in Attempt 8 if
genuine stability (not just a better rollback target) is the goal.
Not yet tried: an even gentler decay (e.g. `lr-decay-iters` close to
`max-iters`, more like a standard single cosine sweep) at `eps=1e-3`,
which might let the model spend more time at higher effective LR before
eps's stabilizing effect has to do its work at the floor.

**Addendum, filling a gap in Attempt 8's own sweep**: the table above
pairs `eps=3e-4` with the *gentler* decay (`lr-decay-iters=8000`) — it
was never tested at Attempt 8's original `lr-decay-iters=5000`
structure, the same one `eps=1e-3`/`1e-2` were tested against, so the
two weren't cleanly comparable on decay length alone. Filling that gap:

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 5000 --adam-eps 3e-4 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps3e4_decay5000.pt --states data/states_word_big4096_eps3e4_decay5000.jsonl
```
Result: best val **2.426 @ iter 7500**, final 2.537 @ iter 15000, drift
**+0.111**. This slots cleanly between the two extremes already in
Attempt 8's table, at the *same* decay length: much better peak than
`eps=1e-3` (2.426 vs 2.743) at the cost of real but modest drift (+0.111
vs `eps=1e-3`'s ~0), and vastly better than default `eps=1e-8`'s peak
(2.203, but +1.367 drift to full divergence). Confirms `eps` behaves as
a genuinely continuous dial on this shape, independent of the decay-
length axis: `1e-8` (divergent) → `3e-4` (much dampened, some drift) →
`1e-3` (flat, worse peak) → `1e-2` (flat, much worse peak) — the same
efficiency/stability tradeoff curve Attempt 9's decay-8000 pairing
showed, just cleanly isolated from the decay-length variable this time.
Practically: `eps=3e-4` at this decay length is a genuinely attractive
middle option if a small residual drift (auto-rollback still catches
the 2.426 peak regardless) is acceptable in exchange for meaningfully
better quality than `eps=1e-3`'s full-flatness guarantee.

**Second addendum: does `eps=3e-4` keep improving with an even gentler
decay, the way `eps=1e-3` does (Attempt 10)?** Tested
`lr-decay-iters=10000` (same as Attempt 10's best `eps=1e-3` config,
just with `eps=3e-4` instead):

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 10000 --adam-eps 3e-4 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps3e4_decay10000.pt --states data/states_word_big4096_eps3e4_decay10000.jsonl
```

| `eps=3e-4`, `lr-decay-iters=` | peak val | drift, best→final (iter 15000) |
|---|---|---|
| 5000 | 2.426 @ iter 7500 | +0.111 |
| **8000** | **2.365 @ iter 7500 (best peak)** | +0.211 |
| 10000 | 2.391 @ iter 6500 (before the floor) | **+0.262 (worst drift)** |

**No — the answer is no, and it's non-monotonic, unlike `eps=1e-3`'s
own curve.** Going from `decay-iters=5000` to `8000` improved the peak
(2.426→2.365); going further to `10000` made it *worse* on both counts
at once (peak regresses to 2.391, and drift keeps climbing to +0.262,
its largest yet). This is a genuinely different shape from `eps=1e-3`'s
own decay-length sweep, where stretching from 5000→10000 kept improving
the peak with only a small, roughly-constant drift cost (Attempts 8→10),
before plateauing at 13000 (Attempt 11). `eps=3e-4`'s sweet spot for
decay length sits around `8000`, not further out — pushing past it just
gives the model more high-LR time to drift away in, without a
compensating peak-quality gain. Practical implication: the eps/decay-
length interaction isn't a single shared curve across `eps` values —
each `eps` choice has its own decay-length optimum, and it has to be
found per-`eps` rather than assumed to transfer (the wide model's
`eps=1e-3` optimum at decay~10000 does not predict `eps=3e-4`'s optimum,
which is lower, at decay~8000).

### Attempt 10: `eps=1e-3` + a gentler decay — the best combo so far

Direct test of Attempt 9's closing question, on the *other* eps value:
same `eps=1e-3` that gave genuine flatness at `lr-decay-iters=5000`, now
with a gentler decay (`lr-decay-iters=10000` instead of 5000 — decay
stretches 2000-10000, floor holds only 10000-15000).

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 10000 --adam-eps 1e-3 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps1e3_decay10000.pt --states data/states_word_big4096_eps1e3_decay10000.jsonl
```

| iter | train | val | lr |
|---|---|---|---|
| 4000 | 2.889 | 2.868 | 2.6e-04 |
| 6000 | 2.706 | 2.686 | 1.6e-04 |
| 8000 | 2.642 | 2.622 | 7.0e-05 |
| **9500** | **2.625** | **2.606 (best)** | 3.3e-05 |
| 10000 (floor reached) | 2.627 | 2.607 | 3.0e-05 |
| 12500 | 2.648 | 2.629 | 3.0e-05 (holding) |
| 15000 | 2.677 | 2.659 | 3.0e-05 (holding) |

**This is the best result of the whole `eps` sweep.** Best val 2.606 —
beats `eps=1e-3`/decay-5000's flat floor (2.743) by a clear margin, and
close to `eps=3e-4`/decay-8000's peak (2.365, still the single best) but
without that config's real divergence: the post-floor drift here is
tiny (2.607→2.659 over 5000 iters, +0.052) compared to `eps=3e-4`'s
2.369→2.580 over roughly the same span (+0.211) — about 4x gentler. Not
perfectly flat like `eps=1e-3`/decay-5000 (which had ~0 net drift over
its whole hold), but close enough that it reads as "slowly settling",
not "diverging" — a genuinely better practical tradeoff point than
either Attempt 9 config.

**Confirms the tradeoff is continuous along two axes together** (`eps`
magnitude and decay gentleness), not a single knob: at fixed `eps`,
stretching the decay recovers peak quality at a small, controlled cost
to hold-flatness, rather than either being free or reproducing the
uncontrolled divergence a small `eps` shows under the same gentler
decay (Attempt 9). Since `model.train.py`'s auto-rollback always saves
the best-val checkpoint regardless of what happens afterward, this
config's practical deployment value is closer to its peak (2.606) than
its end-of-run number (2.659) — the residual drift matters for trusting
*how far past the peak* it's safe to keep training, not for the
checkpoint itself. Not yet tried: pushing `lr-decay-iters` even closer
to `max-iters` (e.g. 13000-14000) to see whether the drift keeps
shrinking toward zero, or whether it plateaus once eps's stabilizing
margin is fully substituted for a specific decay length.

### Attempt 11: `eps=1e-3` + `lr-decay-iters=13000` — the plateau

Direct test of Attempt 10's closing question: does stretching the decay
even further than 10000 keep buying back peak quality / shrinking the
residual drift, or does it plateau?

```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --lr-decay-iters 13000 --adam-eps 1e-3 \
  --tokenizer word --vocab-size 4096 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_big4096 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_word_big4096_eps1e3_decay13000.pt --states data/states_word_big4096_eps1e3_decay13000.jsonl
```
Only a 2000-iter hold this time (13000-15000) instead of the prior
attempts' 5000-10000-iter holds, since decay itself now eats most of the
run.

| iter | train | val | lr |
|---|---|---|---|
| 4000 | 2.887 | 2.866 | 2.8e-04 |
| 8000 | 2.647 | 2.629 | 1.5e-04 |
| **10500** | **2.626** | **2.607 (best)** | 6.3e-05 |
| 13000 (floor reached) | 2.647 | 2.629 | 3.0e-05 |
| 15000 | 2.683 | 2.665 | 3.0e-05 (holding) |

**Plateau confirmed — no further improvement.** Best val 2.607 is
statistically identical to `lr-decay-iters=10000`'s 2.606 (Attempt 10):
stretching the decay another 3000 iters bought nothing. The end-of-run
drift (2.607→2.665, +0.058 over 4500 iters) is also essentially the
same magnitude as Attempt 10's (+0.052 over 5000 iters) — not smaller.
One new wrinkle: here the climb starts noticeably *before* the LR floor
is reached (val already rising from iter 11000, ~2000 iters before
`lr-decay-iters=13000`'s floor at iter 13000), where in every prior
attempt the climb began right at or after the floor. This is consistent
with the divergence being tied to an *absolute* LR threshold being
crossed (however low, still nonzero this early) rather than strictly
"whenever LR stops moving" — worth keeping in mind for anyone tuning
further.

**Answer to Attempt 10's open question: it plateaus, doesn't keep
improving.** `lr-decay-iters=10000` was already at (or past) the
efficient point on this curve — `eps=1e-3` combined with *some*
long-ish decay recovers about as much peak quality as this axis alone
is going to give (~2.6), with a small, roughly constant residual drift
regardless of exactly how much further the decay is stretched past
~10000. This closes out the eps/decay-length exploration for this model
size: **`eps=1e-3`, `lr-decay-iters≈10000` (of 15000) is the
recommended practical recipe** from this whole investigation — best
achievable peak without genuine, unbounded divergence.

### Attempt 12: porting the recipe to the deployment shape — a negative transfer result

Every attempt above trained the wide D=384/NLAYER=12/VOCAB=4096
diagnostic shape used to find and reproduce the instability. This
attempt asks the practical question: does the fix recipe (`eps=1e-3`,
`lr-decay-iters≈10000`) help the model actually deployed on real
hardware — D=128/NLAYER=12/NHEAD=2/VOCAB=1900 (`data/ckpt_word16.pt`,
`fabric/genesys2/PORT-NOTES.md`'s "Word-level vocabulary (current)"
section, best val 2.022 @ iter 23000 on real Genesys2 silicon)?

```bash
python -m model.train --max-iters 23000 --lr 3e-4 --warmup 2000 --lr-decay-iters 10000 --adam-eps 1e-3 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16_epsrecipe_23k.pt --states data/states_word16_epsrecipe_23k.jsonl
```
(First run stopped at `--max-iters 15000`, val 2.719 @ iter 15000, still
improving with no divergence; re-run to `--max-iters 23000` — this
architecture trains fast, 2.61M params, ~9 min total on this GPU — for a
same-iter-count comparison against the deployed checkpoint.)

| | this recipe (eps=1e-3, decay@10000) | original deployment run (`ckpt_word16.pt`) |
|---|---|---|
| peak LR | 3e-4 | ~5e-4 |
| warmup | 2000 | ~100 (old default) |
| LR schedule | fast decay to floor by iter 10000, held flat for the remaining 13,000 | standard single cosine sweep across the full 23000-24000 iters |
| `adam-eps` | 1e-3 | 1e-8 (default; predates the `--adam-eps` flag) |
| val @ iter 23000 | **2.672** | **2.022** |

**Clear negative transfer: the recipe is worse here, not neutral.**
Never diverges (still slowly improving through iter 23000, matching the
port notes' own finding that this narrower shape "trained clean and
stable... no instability, unlike D=256"), but it converges to a
meaningfully worse val loss than the deployment checkpoint's own
existing recipe at the identical iteration count. Two compounding
reasons, not one: (1) `eps=1e-3` was tuned specifically to counteract a
failure mode this shape doesn't have — at a width that never needed
denominator-floor damping, raising `eps` just gives up Adam efficiency
for a stability benefit that was never at risk, the same tradeoff
Attempts 8-9 already documented at the wide shape; (2) the fast-decay-
then-hold schedule (`lr-decay-iters=10000` of 23000) spends 13,000 of
the run's 23,000 iterations at a frozen `3.0e-05` floor, where the
original recipe spent that whole span still slowly decaying from a much
higher peak (5e-4 vs 3e-4) — strictly less effective high-LR training
time, independent of `eps` at all.

**Takeaway: this fix is scale-specific, not a universal drop-in
improvement.** It was diagnosed on, and should stay scoped to, model
shapes wide enough (D>=256-384 at this depth) to actually exhibit the
Adam-driven divergence documented in Attempts 1-7. The currently
deployed D=128/NLAYER=12/VOCAB=1900 checkpoint's existing training
recipe (higher peak LR, standard full-length cosine decay, default
`adam-eps`) remains the better choice for that shape and should not be
replaced by this investigation's findings. Not yet tried: whether the
deployment shape's own recipe (5e-4 peak, full cosine, default eps)
could be improved further on its own terms — a separate question from
this attempt, which only tested whether the *wide-model* fix transfers
(it doesn't).

**Correction, from Attempt 13 below: this attempt's framing — "this
shape never diverges" — turned out to be wrong.** It was based on a
single existing run (`ckpt_word16.pt`'s own clean history) plus this
attempt's own recipe-ported runs, which also happened not to diverge.
Attempt 13 re-ran the *original* recipe itself and got a real
divergence on the identical hyperparameters/architecture/data. The
negative-transfer verdict above (the wide-model fix is the wrong tool
for this shape, and costs quality here) still stands — but "this shape
never diverges" does not; see Attempt 13 for the corrected picture.

### Attempt 13: reproducing the original recipe — it's not a fluke, it's variance

Attempt 12 assumed the deployment shape (D=128/NLAYER=12/VOCAB=1900)
simply doesn't have the instability problem, based on `ckpt_word16.pt`'s
own clean training history (`data/states_word16.jsonl`: monotonic val
improvement the entire 24000-iter run, no divergence). This attempt
re-runs that *exact* original recipe (not the wide-model fix) to check
whether that clean result reproduces, or was itself a lucky run.

**Recipe reverse-engineered from `data/states_word16.jsonl`'s own LR
trace** (this checkpoint predates `--lr-decay-iters`/`--adam-eps`, so
there's no command string to copy — the schedule was inferred by
matching `cosine_lr()`'s output against the logged `lr` values): peak
`lr=5e-4`, `warmup=100` (the old default), `max-iters=24000`, no
`--lr-decay-iters` (single cosine sweep across the whole run, old
behavior), default `adam-eps=1e-8`.

```bash
python -m model.train --max-iters 24000 --lr 5e-4 --warmup 100 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16_reproduce.pt --states data/states_word16_reproduce.jsonl
```
`model.train` calls `torch.manual_seed(1337)` unconditionally
(`model/train.py:236`) — every run uses the *same* seed, so init and
batch order are identical across runs, not a source of difference
between this reproduction and the original.

| iter | original (`ckpt_word16.pt`) val | reproduction val | reproduction lr |
|---|---|---|---|
| 4000 | (not logged at this iter) | 2.266 | 4.7e-04 |
| **7000-7500** | 2.150 (iter 7000, still improving) | **2.208 (best, iter 7500)** | 4.0e-04 |
| 10000 | 2.098 | 2.227 | 3.3e-04 |
| 13000 | 2.066 | 2.265 | 2.5e-04 |
| 18500 | (not logged at this iter) | 2.340 | 1.1e-04 |
| 23000 | **2.022 (best, real hardware checkpoint)** | 2.385 | 5.2e-05 |
| 24000 | 2.022 | **2.398 (final)** | 5.0e-05 |

**Real divergence, not noise.** The original run improves monotonically
end to end (2.150 → 2.022 over iters 7000-23000, never once regressing
in the logged evals). The reproduction peaks early (2.208 @ iter 7500)
and then climbs continuously for the remaining 16,500 iterations to
2.398 — the same *shape* of unbounded Adam-driven divergence documented
throughout this whole investigation (Attempts 1-7), just milder in
magnitude (+0.19 relative rise here vs. the wide model's +0.61-1.35) and
occurring under a schedule that never holds LR flat (LR keeps decaying
the entire time, unlike every wide-model attempt's floor-and-hold
structure) — so the divergence isn't even gated on LR going flat at
this shape, only on LR being low enough in absolute terms, consistent
with Attempt 11's "starts before the floor" observation.

**This means the deployment shape does carry real instability risk —
and, more surprisingly, it isn't even deterministic despite the fixed
seed.** Every AdamW configuration tried on the wide D=384 diagnostic
shape (9+ runs, every LR schedule/init/weight-decay variant in Attempts
1-6) diverged with 100% reliability. Here, *identical* hyperparameters,
*identical* data, and the *same* `torch.manual_seed(1337)` produced one
clean run (the original) and one diverging run (this reproduction) —
with no available source of run-to-run randomness to blame it on. The
likely mechanism is the same one Attempts 1-11 already characterize
(Adam's `sqrt(v_hat)+eps` step size not actually shrinking to zero at
low nominal LR), combined with ordinary GPU floating-point
non-determinism: CUDA/cuDNN matmul and reduction kernels don't
guarantee bit-identical results run to run even with a fixed seed (parallel
reduction order isn't fixed), so two runs' loss curves drift apart by a
tiny, invisible amount from iteration one — normally irrelevant, but
this shape sits close enough to a genuine edge-of-stability point that
which side of it a run lands on is sensitive to that drift. Not
"random" in the sense of a different seed or data order — genuinely the
same recipe landing differently due to sub-floating-point-epsilon
numerical noise compounding near an instability boundary.

**Revises Attempt 12's practical recommendation.** "Keep the original
recipe because this shape doesn't need stabilizing" is no longer
supportable — the original recipe produced the currently-deployed
checkpoint by getting a favorable run, not because that recipe is safe
by construction. The honest options going forward, not yet decided
between: (a) re-run the original high-LR recipe several more times and
keep the best (cheap here, ~9-10 min/run, but doesn't fix the
underlying risk, just plays the odds better); (b) add a *mild*
stabilizing margin (a smaller `adam-eps` bump than the wide model
needed, or a shorter floor-hold than Attempt 12's, tuned specifically
at this width) that trades little peak quality for real protection
against the divergence seen here; (c) accept the risk and always verify
any new deployment-shape training run against its own `states.jsonl`
curve before trusting a checkpoint, the way Conclusion 3 already
recommends project-wide. Not yet tried: any of the three.

### Attempt 14: a mild `adam-eps` margin at the deployment shape — `1e-4` works, `1e-5` doesn't

Tests option (b) from Attempt 13's closing list: keep the deployment
shape's own recipe exactly as-is (lr=5e-4, warmup=100, full cosine
decay across 24000 iters — Attempt 12 already showed changing the
schedule costs quality here) and add only a mild `adam-eps` bump on
top, well below the wide model's `1e-3` fix strength, to see whether a
small dose buys real protection without the peak-quality cost Attempt
12 paid.

```bash
# eps=1e-5
python -m model.train --max-iters 24000 --lr 5e-4 --warmup 100 --adam-eps 1e-5 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16_epsmild1e5.pt --states data/states_word16_epsmild1e5.jsonl
# eps=1e-4 / eps=3e-4 (same flags otherwise, --out/--states renamed
# epsmild1e4 / epsmild3e4)
```

| iter | eps=1e-8 (Attempt 13 reproduction) | eps=1e-5 | eps=1e-4 | eps=3e-4 |
|---|---|---|---|---|
| 4000 | 2.266 | 2.277 | 2.372 | 2.475 |
| 13000 | 2.265 | 2.259 | 2.223 | 2.212 (still falling) |
| **best** | **2.208 @ iter 7500** | **2.208 @ iter 7500** | **2.213 @ iter 10000-10500** | **2.203 @ iter 19500-20000** |
| 18000 | 2.333 | 2.318 | 2.263 | 2.205 |
| 23000 | 2.385 | 2.381 | 2.301 | 2.208 |
| 24000 (final) | 2.398 | 2.394 | 2.310 | **2.211** |
| drift, best→final | +0.190 | +0.186 | +0.097 | **+0.008** |

**`eps=1e-5` does essentially nothing** — its whole trajectory tracks
the unstabilized reproduction almost exactly (2.208 vs 2.208 peak,
2.394 vs 2.398 final, same +0.186-0.190 drift) — too small a dose at
this width's gradient-variance scale to matter, the deployment-shape
analogue of the wide model's own finding that `eps` has a real
threshold below which it does nothing (Attempts 8-9).

**`eps=1e-4` genuinely helps.** Peak val is statistically the same as
the unstabilized runs (2.213 vs 2.208 — a negligible cost, unlike
Attempt 12's `eps=1e-3`+floor-hold recipe which gave up real peak
quality), but the post-peak drift roughly halves (+0.097 vs +0.186-
0.190) and the peak itself lands later (iter 10000-10500 vs iter 7500)
— more of the run is spent still improving before the divergence risk
sets in. Not a full fix like the wide model's `eps=1e-3` (which achieved
genuine flatness, Attempt 10) — this still drifts, just much more
slowly — but it's a real, nearly-free stability margin at this shape,
unlike Attempt 12's ported recipe which cost quality for no benefit.

**`eps=3e-4` does even better — genuine flatness, not just slower
drift.** Bottoms at 2.203 @ iter 19500-20000 (still falling at iter
13000, unlike `eps=1e-4` which had already peaked by iter 10500), then
holds essentially flat through the rest of the run (2.203→2.211 over
the remaining ~4500 iters, +0.008 — statistically noise, not a trend).
This both beats `eps=1e-4`'s peak (2.203 vs 2.213) *and* nearly
eliminates the drift (+0.008 vs +0.097). It mirrors the wide model's own
finding (Attempt 9 vs 10: a mid-strength `eps` dampens without fully
fixing, a stronger one achieves genuine flatness) — just scaled down for
this narrower D=128 shape, where `3e-4` (not the wide model's `1e-3`)
turns out to be the dose that actually crosses into stability rather
than merely slowing the climb.

**This is the answer to Attempt 13's open question, revised: option (b)
works, and `eps=3e-4` is the better dose — not `1e-4`.** On top of the
deployment shape's existing recipe (unchanged peak LR, unchanged
warmup, unchanged full-length cosine decay), `eps=3e-4` gives both a
peak at least as good as the unstabilized runs' best (2.203, versus
2.208 for the lucky unstabilized case) *and* genuine post-peak
stability, superseding the `eps=1e-4` recommendation above. Not yet
tried: whether `eps` values between `3e-4` and `1e-3` land even closer
to the original clean run's 2.022, or whether `3e-4` is itself already
near the efficiency/stability sweet spot the way the wide model's
`eps=1e-3`+`lr-decay-iters~10000` combo was found to be (Attempt 11's
plateau).

**Reproducibility check**: re-ran this exact `eps=1e-4` recipe a second
time (`data/ckpt_word16_v2.pt`, same command, no `--seed` flag needed
since `model.train` fixes `torch.manual_seed(1337)` — see the Attempt 13
correction above) and got an *exact* match with this attempt's own
numbers: best val 2.213 @ iter 10000-10500, final 2.310 @ iter 24000,
matching digit-for-digit through the whole run. One pair each isn't
conclusive, but it's a suggestive contrast with the *default*-eps recipe
(`ckpt_word16.pt` vs. Attempt 13's reproduction), which diverged from
itself between two nominally identical runs — consistent with the `eps`
margin also damping the floating-point-noise sensitivity Attempt 13
identified, not just the post-peak drift once divergence starts. Kept
as a separate checkpoint (`data/ckpt_word16_v2.pt`), not promoted over
the real, hardware-verified `data/ckpt_word16.pt` — that decision is
left to whoever picks this up next, given the downstream QAT/export/
real-hardware chain a swap would affect.

### Attempt 15: eps between `3e-4` and `1e-3` — is there a better dose than `3e-4`?

Attempt 14 recommended `eps=3e-4` for the deployment shape but flagged an
open question: does any value between `3e-4` (the deployment shape's own
minimal fix) and `1e-3` (the wide model's fix) do better, or is `3e-4`
already the sweet spot? This attempt tests three doses in that gap —
`5e-4`, `7e-4`, `1e-3` — on top of the exact same unchanged deployment
recipe as Attempt 14 (`lr=5e-4, warmup=100`, full cosine decay across
24000 iters, D=128/NLAYER=12/NHEAD=2/VOCAB=1900).

```bash
python -m model.train --max-iters 24000 --lr 5e-4 --warmup 100 --adam-eps 5e-4 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16_eps5e4.pt --states data/states_word16_eps5e4.jsonl
# repeat with --adam-eps 7e-4 (out/states renamed eps7e4)
# repeat with --adam-eps 1e-3 (out/states renamed eps1e3)
```

| iter | eps=1e-8 (Att.13) | eps=1e-4 (Att.14) | eps=3e-4 (Att.14) | eps=5e-4 | eps=7e-4 | eps=1e-3 |
|---|---|---|---|---|---|---|
| 4000 | 2.266 | 2.372 | 2.475 | 2.559 | 2.635 | 2.714 |
| 8000 | 2.209 | 2.230 | 2.275 | 2.324 | 2.385 | 2.454 |
| 12000 | 2.258 | 2.217 | 2.215 | 2.244 | 2.287 | 2.351 |
| 16000 | 2.304 | 2.247 | 2.205 | 2.210 | 2.240 | 2.301 |
| 20000 | 2.355 | 2.278 | 2.203 | 2.190 | 2.213 | 2.268 |
| 24000 (final) | 2.398 | 2.310 | 2.211 | **2.187** | 2.204 | 2.256 |
| **best** | 2.208 @ 7500 | 2.213 @ 10500 | 2.203 @ 19500 | **2.185 @ 23000** | 2.204 @ 24000 | 2.256 @ 24000 |
| drift, best→final | +0.190 | +0.097 | +0.008 | **+0.001** | +0.000 | +0.000 |

**`eps=5e-4` is the new best result of the whole investigation — it beats
`eps=3e-4` on both axes at once.** Peak val 2.185 (vs 2.203), reached
later (iter 23000 vs 19500, i.e. still improving for longer before
leveling off) and holding essentially exact-flat afterward (+0.001 drift
over the last 1000 iters — noise, not a trend). This is the closest any
stabilized run has come to the original unstabilized `ckpt_word16.pt`'s
2.022, while still being a genuine converged floor rather than a
best-checkpoint gamble.

**`eps=7e-4` and `eps=1e-3` are NOT worse doses in the same sense `1e-4`
was — they're just slower, and hadn't finished converging by iter
24000.** Both are still monotonically falling at the final logged point
(no peak-then-plateau shape at all, unlike every other config in this
sweep), so "best = final" for both is an artifact of the run ending, not
evidence of an early plateau. This matches the pattern first seen in
Attempt 8 (`eps=1e-2` was "still slowly falling" at iter 15000 on the
wide model) and Attempt 9 (larger `eps` trades peak quality for
stability along a continuous spectrum) — larger `eps` damps the
denominator more aggressively, which buys stability margin at the cost
of slower effective progress. Within this fixed 24000-iteration budget,
that tradeoff nets out worse for `7e-4`/`1e-3` than for `5e-4`; whether a
longer run lets `7e-4` or `1e-3` eventually overtake `5e-4`'s 2.185 is an
open question, not tested here.

**This supersedes Attempt 14's `eps=3e-4` recommendation.** For the
deployment shape's own recipe (unchanged peak LR, warmup, full-length
cosine decay) at a 24000-iteration budget, `eps=5e-4` is both the best
peak found across the whole eps sweep (1e-8 through 1e-3) and, unlike the
unstabilized run's own best-checkpoint snapshot, a value the model
actually settles at rather than passes through on the way to divergence.
Not yet tried: doses between `3e-4` and `5e-4`, or a longer run at
`7e-4`/`1e-3` to see if the "still falling" trajectory eventually crosses
`5e-4`'s floor.

### Attempt 16: doses between `3e-4` and `5e-4` — filling the gap, no local optimum found

Attempt 15 left one gap open: does anything strictly between `3e-4` and
`5e-4` beat `5e-4`, or is `5e-4` itself the best point in that range?
This attempt tests three doses in between — `3.5e-4`, `4e-4`, `4.5e-4`
— on the same unchanged deployment recipe as Attempts 14-15 (`lr=5e-4,
warmup=100`, full cosine decay across 24000 iters,
D=128/NLAYER=12/NHEAD=2/VOCAB=1900).

```bash
python -m model.train --max-iters 24000 --lr 5e-4 --warmup 100 --adam-eps 3.5e-4 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16_eps3_5e4.pt --states data/states_word16_eps3_5e4.jsonl
# repeat with --adam-eps 4e-4 (out/states renamed eps4e4)
# repeat with --adam-eps 4.5e-4 (out/states renamed eps4_5e4)
```

| iter | eps=3e-4 (Att.14) | eps=3.5e-4 | eps=4e-4 | eps=4.5e-4 | eps=5e-4 (Att.15) |
|---|---|---|---|---|---|
| 4000 | 2.475 | 2.495 | 2.517 | 2.538 | 2.559 |
| 8000 | 2.275 | 2.284 | 2.297 | 2.311 | 2.324 |
| 12000 | 2.215 | 2.220 | 2.225 | 2.235 | 2.244 |
| 16000 | 2.205 | 2.199 | 2.201 | 2.205 | 2.210 |
| 20000 | 2.203 | 2.193 | 2.189 | 2.190 | 2.190 |
| 24000 (final) | 2.211 | 2.198 | 2.192 | 2.188 | **2.187** |
| **best** | 2.203 @ 19500 | 2.193 @ 20500 | 2.188 @ 21000 | 2.187 @ 22500 | **2.185 @ 23000** |
| drift, best→final | +0.008 | +0.005 | +0.004 | +0.002 | **+0.001** |

**No local optimum between `3e-4` and `5e-4` — every metric improves
monotonically as `eps` rises through the gap.** Best val (2.203 → 2.193
→ 2.188 → 2.187 → 2.185), drift (+0.008 → +0.005 → +0.004 → +0.002 →
+0.001), and the peak's iteration (19500 → 20500 → 21000 → 22500 →
23000, i.e. the model keeps improving for longer before leveling off at
higher `eps`) all move the same direction in lockstep. `5e-4` remains
the best point of this whole sub-sweep, sitting right at the edge of the
tested range rather than in the interior — there is no dip-and-rise
shape here, unlike the wider gap tested in Attempt 15 where `7e-4`/`1e-3`
looked worse only because they hadn't finished converging within budget.

**This confirms, rather than revises, Attempt 15's recommendation.**
`eps=5e-4` stays the practical choice (Conclusion 8 unchanged). The open
question is now sharper, not resolved: since the trend is still rising
at `5e-4` with no interior optimum below it, the real remaining question
is whether a longer training budget lets `7e-4` or `1e-3` (Attempt 15's
"still falling" configs) eventually overtake `5e-4` — not whether
anything between `3e-4` and `5e-4` does, which this attempt rules out.

### Attempt 17: a longer run at `eps=7e-4`/`1e-3` — does more room let them overtake `eps=5e-4`?

Attempt 15 left `eps=7e-4` and `eps=1e-3` unresolved: both were still
monotonically falling at iter 24000 with no plateau, so their apparent
loss to `eps=5e-4` might just be an artifact of the run ending too
early. This attempt gives both 50% more room: `--max-iters 36000`
instead of `24000`, same `lr=5e-4, warmup=100` otherwise.

**Important schedule caveat**: since `model.train`'s cosine schedule
spans `lr_decay_iters or max_iters` (`model/train.py:371`), and neither
recipe sets `--lr-decay-iters`, raising `--max-iters` to 36000 doesn't
just bolt 12000 extra iterations onto the existing 24000-iter run — it
stretches the cosine decay to span the whole 36000 iters, so LR stays
higher for longer at every matching iteration (e.g. at iter 24000: LR
is `5.0e-05`, fully decayed, in the original run, vs `1.63e-04`, still
mid-decay, in the stretched run). This is a genuinely different
schedule shape, not "the same run continued," the same caveat Attempts
9-11 dealt with when pairing `eps` changes with decay-length changes on
the wide model.

```bash
python -m model.train --max-iters 36000 --lr 5e-4 --warmup 100 --adam-eps 7e-4 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16_eps7e4_long36k.pt --states data/states_word16_eps7e4_long36k.jsonl
# repeat with --adam-eps 1e-3 (out/states renamed eps1e3_long36k)
```

| | eps=7e-4 @24k (Att.15) | eps=7e-4 @36k | eps=1e-3 @24k (Att.15) | eps=1e-3 @36k |
|---|---|---|---|---|
| val @ iter 24000 | 2.204 (final) | 2.232 (mid-decay, LR still 1.63e-04) | 2.256 (final) | 2.260 (mid-decay) |
| val @ iter 32000-33000 | — | 2.222 | — | **2.239 (best)** |
| best val | 2.204 @ 24000 | 2.221 @ 30500 | 2.256 @ 24000 | 2.239 @ 33000 |
| final val | 2.204 | 2.228 @ 36000 | 2.256 | 2.241 @ 36000 |
| drift, best→final | +0.000 | +0.008 | +0.000 | +0.002 |
| vs `eps=5e-4`'s 2.185 | worse | **worse** | worse | **worse** |

**Neither config overtakes `eps=5e-4`, even with 50% more iterations and
a matched, fully-decayed schedule.** `eps=1e-3`'s best improves from
2.256 (24k, still falling) to 2.239 (36k, genuine peak with only +0.002
drift after) — real progress, but still well short of `5e-4`'s 2.185.
`eps=7e-4` is the more surprising result: its 36k best (2.221) is
**worse** than its own 24k run's final value (2.204). Stretching the
decay to give more nominal room didn't help it reach a better floor —
it kept LR elevated for longer at every comparable iteration (val 2.232
vs 2.204 at the matched iter-24000 point) without buying enough extra
benefit from the added tail to compensate, similar in spirit to how
Attempts 9-11 found gentler decay is not a free lunch at the wide
model — more decay room only helps if the config actually needs it, and
past a point it just wastes high-LR training time.

**This closes Attempt 15's open question: `eps=5e-4` is the real winner,
not an artifact of the 24000-iteration budget cutting off `7e-4`/`1e-3`
too early.** Combined with Attempt 16 (no better dose between `3e-4`
and `5e-4`), the practical recommendation (Conclusion 8) is now fully
closed on both sides of `5e-4` — nothing lower, and nothing higher with
more room, has beaten it.

### Attempt 18: QAT the `eps=5e-4` deployment checkpoint (VOCAB=1900)

With `eps=5e-4` confirmed as the best FP recipe (Attempts 15-17), this
attempt fine-tunes it through QAT, the step every checkpoint needs
before it's a real deployment candidate — same recipe as every other
QAT pass in this project (warm-started, `--max-iters 3000`, defaults
otherwise).

```bash
python -m model.train --qat --init-from data/ckpt_word16_deploy_eps5e4.pt --max-iters 3000 \
  --tokenizer word --vocab-size 1900 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_stream16 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16_deploy_eps5e4.qat.pt --states data/states_word16_deploy_eps5e4_qat.jsonl
```

**Result: best val 2.153 @ iter 3000** — improved over its own FP
source (2.185), the same pattern the `eps=3e-4` checkpoint's QAT pass
showed (2.203 → 2.155, `fabric/genesys2/PORT-NOTES.md`'s "A new
checkpoint from the scale-up-instability investigation" section), not
the regression the original unstabilized `ckpt_word16.pt` → `.qat.pt`
pass had (2.022 → 2.074). A second data point for the pattern that
`eps`-stabilized FP checkpoints tolerate QAT fine-tuning better than
the unstabilized recipe's lucky-snapshot checkpoint does.

### Attempt 19: bit-exact RTL verification of the `eps=5e-4` QAT checkpoint (VOCAB=1900)

Exported via `model.goformer_full.params_from_ckpt()`/`save_params()`:

```bash
python -c "
from model.goformer_full import params_from_ckpt, save_params
p, model, ck = params_from_ckpt('data/ckpt_word16_deploy_eps5e4.qat.pt')
save_params(p, 'fabric/export_word16_deploy_eps5e4/goformer.npz')
"
```

Then five prompts through `fabric.stage3.run_vec_kv` at Genesys2's real
deployed RTL parameters (`--p 8 --lanes 64 --tmax 128`), matching the
`eps=3e-4` checkpoint's own five-prompt precedent (one shorter run, four
near the `TMAX=128` context limit), all seeded on-chip Gumbel sampling:

```bash
python -m fabric.stage3.run_vec_kv \
  --npz fabric/export_word16_deploy_eps5e4/goformer.npz --meta data/word_stream16/meta.json \
  --p 8 --lanes 64 --tmax 128 --prompt "once upon a time" --plen 4 --ngen 60 --seed 7 \
  --dir /tmp/kevbuild_deploy_eps5e4_p1
# + 4 more prompts at --ngen 120, seeds 1-4 (see Files touched)
```

| prompt | plen | ngen | seed | verdict |
|---|---|---|---|---|
| "once upon a time" | 4 | 60 | 7 | `match=True` |
| "the little dog" | 3 | 120 | 1 | `match=True` |
| "one day a girl named lily" | 6 | 120 | 2 | `match=True` |
| "there was a boy who" | 5 | 120 | 3 | `match=True` |
| "the old man walked into" | 5 | 120 | 4 | `match=True` |

**5/5 `VEC_KV_VERDICT match=True`** — token-for-token identical to the
Python golden reference in every run. `fabric/export_word16_deploy_eps5e4/goformer.npz`
is bit-exact verified at the real deployed RTL parameters. Not yet done:
real Vivado synthesis/implementation or programming onto actual
silicon — this checkpoint is verified in simulation only.

### Attempt 20: porting `eps=5e-4` to VOCAB=16384 — does the fix transfer to the real deployed shape?

Everything above (Attempts 1-19) targeted **VOCAB=1900**, this log's
own original scope. But the checkpoint actually flashed and running on
real Genesys2 hardware is **VOCAB=16384** (`data/ckpt_word16384.pt` →
`.qat.pt`, see `fabric/genesys2/PORT-NOTES.md`) — trained with
`--adam-eps 3e-4`, ported from this log's own Attempt 14 finding *before*
Attempts 15-17 found `5e-4` beats `3e-4` at VOCAB=1900. This attempt
asks the obvious follow-up: does `eps=5e-4` also beat `eps=3e-4` at
VOCAB=16384, the shape that actually matters now?

```bash
python -m model.train --max-iters 24000 --lr 5e-4 --warmup 100 --adam-eps 5e-4 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16384_deploy_eps5e4.pt --states data/states_word16384_deploy_eps5e4.jsonl
```

| | eps=3e-4 (deployed) | eps=5e-4 (new) |
|---|---|---|
| best val | **2.4085** | 2.4212 |
| best iter | 15500 | 21500 |
| final val | 2.4266 | **2.4241** |
| drift, best→final | +0.0181 | **+0.0029** |

**No — `eps=5e-4` is a regression at VOCAB=16384, not an improvement.**
Its peak (2.4212) is clearly worse than the deployed `eps=3e-4`
checkpoint's own peak (2.4085), the opposite of what Attempts 15-17
found at VOCAB=1900. It does buy real stability — much less post-peak
drift (+0.0029 vs +0.0181), enough that its *final*-iteration value
edges out `eps=3e-4`'s final value — but since `model.train`'s
auto-rollback saves the best-val checkpoint, not the final one, that
stability doesn't translate into a better deployment candidate here.
**The `eps=5e-4` finding does not transfer across vocab sizes** — VOCAB
size changes the loss landscape enough that each shape needs its own
`eps` tuning, not just a straight port of the last shape's answer.

### Attempt 21: VOCAB=16384's own eps sweep — finding this shape's actual sweet spot

Since `5e-4` overshot, this attempt sweeps back toward `3e-4` —
`3.5e-4`, `4e-4`, `4.5e-4` — the same increments Attempt 16 used, on
the same unchanged VOCAB=16384 recipe.

```bash
python -m model.train --max-iters 24000 --lr 5e-4 --warmup 100 --adam-eps 3.5e-4 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16384_eps3_5e4.pt --states data/states_word16384_eps3_5e4.jsonl
# repeat with --adam-eps 4e-4 (out/states renamed eps4e4)
# repeat with --adam-eps 4.5e-4 (out/states renamed eps4_5e4)
```

| eps | best val | best iter | final val | drift |
|---|---|---|---|---|
| 3e-4 (deployed) | 2.4085 | 15500 | 2.4266 | +0.0181 |
| **3.5e-4 (best)** | **2.4070** | 20000 | 2.4183 | +0.0113 |
| 4e-4 | 2.4164 | 20000 | 2.4267 | +0.0103 |
| 4.5e-4 | 2.4146 | 21500 | 2.4196 | +0.0050 |
| 5e-4 | 2.4212 | 21500 | 2.4241 | +0.0029 |

**`eps=3.5e-4` is the best dose found for VOCAB=16384** — best val
2.4070, a small but real improvement over the deployed `3e-4`'s 2.4085,
and meaningfully more stable post-peak (drift +0.0113 vs +0.0181).

**Unlike VOCAB=1900 (Attempt 16), this is NOT a clean monotonic ramp.**
`eps=4e-4`'s best val (2.4164) is genuinely worse than both its
neighbors, `3.5e-4` (2.4070) and `4.5e-4` (2.4146) — a real dip, not
noise (both flanking runs are consistently better across the whole
tail of their trajectories, not just at the best-val point). Drift
*does* still shrink monotonically with rising eps (0.0181 → 0.0113 →
0.0103 → 0.0050 → 0.0029), so the stability-vs-eps relationship holds
even where the quality-vs-eps relationship doesn't. **Takeaway: the
per-shape sweet spot has to be found per shape — VOCAB=16384's
non-monotonic landscape here is a second data point (after Attempt 20)
that a finding from one vocab size doesn't reliably predict another's,
even the direction of the curve.**

### Attempt 22: QAT the VOCAB=16384 `eps=3.5e-4` winner — the FP-level edge vanishes

```bash
python -m model.train --qat --init-from data/ckpt_word16384_eps3_5e4.pt --max-iters 3000 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_word16384_eps3_5e4.qat.pt --states data/states_word16384_eps3_5e4_qat.jsonl
```

| | FP best val | QAT best val (iter 3000) |
|---|---|---|
| eps=3e-4 (deployed) | 2.4085 | **2.3385** |
| eps=3.5e-4 (new) | **2.4070** | 2.3392 |

**QAT improves over its FP source** (2.4070 → 2.3392), continuing the
pattern from Attempts 18 and the `eps=3e-4` checkpoint's own QAT pass.
But the FP-level edge `eps=3.5e-4` had over the deployed `eps=3e-4`
(2.4070 vs 2.4085, a real if small win) **inverts after QAT** — the
deployed checkpoint's QAT result (2.3385) is now marginally better than
the new candidate's (2.3392), a 0.0007 gap that's noise-level, not a
real difference either way. **Net result: at VOCAB=16384, this whole
`eps` excursion (Attempts 20-22) ends in a wash at the QAT level** —
neither a clear win nor a clear loss over what's already deployed. Not
yet bit-exact verified via `fabric.stage3.run_vec_kv`, and not promoted
over the real deployed `data/ckpt_word16384.qat.pt`.

## Conclusions

1. **Capacity helps.** Every run's best-val checkpoint from this whole
   investigation lands in the same narrow val=2.20-2.31 range (the single
   lowest across all 9 runs is `data/ckpt_word_big4096_gentledecay.pt`,
   val=2.203 @ iter 4500 — but they're all close enough to be roughly
   interchangeable, since the instability caps every run at nearly the
   same point regardless of what else changed). The samples shown above
   (from the iter-3500 checkpoint, val=2.271) are clearly more coherent
   and less repetitive prose than the deployed D=128/VOCAB=1900
   real-hardware model, despite a "worse" raw val-loss number that isn't
   actually comparable across different vocab sizes.
2. **Training stability at width is the real bottleneck, not raw
   capacity.** This codebase's recipe (AdamW, cosine LR, the defaults
   used throughout this project) cannot currently sustain long, stable
   training at D>=256-384. Every model this size trained so far has had
   to rely on the automatic best-val rollback catching an early,
   under-trained checkpoint rather than genuine convergence.
3. **`model.train.py`'s auto-rollback is doing real, silent work.** None
   of these experiments lost anything to the instability — the saved
   `--out` checkpoint is always the best-val snapshot, not the final
   (diverged) one. Anyone repeating this doesn't need to babysit runs for
   divergence; just don't trust a checkpoint's `iter` field to mean "fully
   trained" without checking it against `states.jsonl`'s own val curve.
4. **Root cause identified: Adam's optimizer dynamics, not the model, the
   data, the width, or the LR schedule.** Ruled out as causal: LR
   magnitude, warmup length, missing GPT-2 residual-projection init
   scaling, weight decay applied to LayerNorm gains, unconstrained logit
   growth (at two very different strengths), and — decisively, in Attempt
   6 — the LR schedule itself: pinning LR completely flat at its floor for
   10,000 iterations still produced the same divergence (val +61% while LR
   never moved). Per-layer gradient-norm logging (Attempt 5) also rules
   out a single misbehaving layer: growth is synchronized across all 12
   transformer blocks and both attention/MLP sub-components, with the
   embedding/head layer comparatively the MOST stable part of the network.
   **Attempt 7 then pinned it down**: the identical model/data/ramp-decay-
   hold structure, run with plain SGD+momentum instead of AdamW, shows NO
   divergence at all — it settles at iter ~6000 and stays flat for the
   remaining 9,000 iterations, where AdamW diverged badly over the same
   window. All "fixes" tried along the way are kept in the codebase as
   permanent, unconditional improvements (correct practice regardless),
   even though none of them was *the* fix — the fix has to target Adam's
   mechanism directly. **Attempt 8 then found it**: raising AdamW's `eps`
   (default `1e-8`) to `1e-3` or `1e-2` eliminates the divergence
   entirely — the model holds flat for the full 10,000-iteration window
   instead of climbing — while lowering `beta2` (default `0.95`) to `0.9`
   does not fix it at all, tracking the default-AdamW divergence almost
   exactly. This isolates the mechanism specifically to the
   `sqrt(v_hat)+eps` denominator floor being too small at low
   gradient-variance estimates, not to the variance EMA's window length.
   `eps=1e-3` is the practical choice (converges to a genuine flat
   val=2.743, not a lucky snapshot) since `eps=1e-2` gives up most of
   Adam's efficiency advantage (floor ~3.9-4.0, close to SGD's own
   val=4.4). **Attempt 9 tested the obvious follow-up** — `eps=3e-4`
   (between the default and the fix strength) with a longer/gentler
   decay (`lr-decay-iters=8000` instead of 5000) — and found a
   continuous tradeoff, not a free lunch: best peak of any AdamW config
   (2.365 @ iter 7500, beating `eps=1e-3`'s 2.743), but still diverges
   afterward, just 3-4x slower than default AdamW. `eps` size trades
   peak quality against hold-stability along a spectrum; `1e-3` remains
   the practical choice if genuine flatness (not just a better
   rollback target) is the goal. **Attempt 10 then tried the same
   gentler-decay idea at `eps=1e-3` instead of `3e-4`**
   (`lr-decay-iters=10000`) and found the best practical config of the
   whole sweep: best val 2.606 @ iter 9500 — clearly better than
   `eps=1e-3`'s original flat floor (2.743) — with only a small residual
   drift afterward (+0.052 over 5000 iters, about 4x gentler than
   `eps=3e-4`'s drift under its own gentler decay). Confirms the
   tradeoff runs along two axes together (`eps` size and decay
   gentleness): stretching the decay at the already-stabilizing
   `eps=1e-3` buys back peak quality at a small, controlled cost to
   flatness, rather than reproducing `eps=3e-4`'s uncontrolled
   divergence. **Attempt 11 then found the ceiling**: pushing
   `lr-decay-iters` to 13000 gave no further improvement (best val 2.607,
   statistically identical to decay-iters=10000's 2.606) and the same
   residual drift magnitude (+0.058 vs +0.052 over a similar span) — a
   genuine plateau, not a curve still improving. One new detail: at
   decay-iters=13000 the val climb starts noticeably before the LR floor
   is reached, unlike every earlier attempt where it began right at the
   floor — consistent with the divergence tracking an absolute LR
   threshold rather than strictly "whenever LR goes flat." **Recommended
   practical recipe from this whole investigation: `eps=1e-3`,
   `lr-decay-iters≈10000` (of a 15000-iter run)** — best peak this axis
   reaches (~2.6) without unbounded divergence, and stretching the decay
   further buys nothing more.
5. **The recipe is scale-specific — it does not transfer to the actual
   deployed model, and makes it worse if ported blindly.** Attempt 12
   ported `eps=1e-3` + `lr-decay-iters≈10000` to the real D=128/NLAYER=12/
   VOCAB=1900 hardware shape (`data/ckpt_word16.pt`) and got a clearly
   worse result at the same iteration count (val 2.672 vs the deployed
   checkpoint's own 2.022 @ iter 23000), with the recipe's lower peak LR
   and its fast-decay-then-hold schedule spending most of the run at a
   frozen floor instead of the deployed recipe's full-length decay from a
   higher peak. This part of the finding stands: **the fix is scoped to
   model shapes wide enough to actually exhibit the Adam-driven
   divergence (D>=256-384 at this depth)** — porting it to a narrower
   shape just costs quality.
6. **But the deployment shape is NOT immune to the instability — Attempt
   12's "this shape never diverges" premise was wrong, corrected by
   Attempt 13.** Re-running the *original* deployment recipe itself
   (lr=5e-4, warmup=100, full cosine, default eps — not the wide-model
   fix) a second time, on identical hyperparameters/architecture/data,
   produced a real divergence: best val 2.208 @ iter 7500, then a
   continuous climb to 2.398 @ iter 24000 — the same *shape* of
   unbounded Adam-driven divergence as the wide model, just milder
   (+0.19 vs +0.61-1.35) and not gated on LR going flat (this run's LR
   never stopped decaying). The originally-deployed `ckpt_word16.pt`
   (val 2.022, monotonic improvement the whole run) is a *favorable*
   outcome of this recipe, not evidence the recipe is safe by
   construction — every AdamW config on the wide shape diverged with
   100% reliability across 9+ runs, but here identical settings (even
   the same fixed `torch.manual_seed(1337)` — there's no seed/batch-order
   difference available to blame) produced one clean run and one
   diverging run, i.e. the instability exists at this width too, just
   closer to a knife-edge — sensitive to ordinary GPU floating-point
   non-determinism (parallel reduction order isn't bit-fixed even under a
   fixed seed) rather than deterministic given the hyperparameters alone.
   **Practical implication: any
   future training run at this deployment shape should be checked
   against its own `states.jsonl` curve before trusting the result (see
   Conclusion 3) — a clean run isn't guaranteed just because the last one
   was clean.** Whether a mild stabilizing margin (smaller than the wide
   model needed) is worth the peak-quality cost at this shape is an open
   question, not resolved here.
7. **Resolved: a mild `adam-eps` margin is worth it — `3e-4` is the
   right dose, not `1e-4`, and `1e-5` does nothing.** Attempt 14 tested
   Conclusion 6's open question directly: kept the deployment shape's
   exact original recipe (peak LR, warmup, full cosine decay all
   unchanged) and added only an `adam-eps` bump, sweeping three values.
   `eps=1e-5` did essentially nothing (tracked the unstabilized
   trajectory almost exactly). `eps=1e-4` genuinely helped but only
   partially: same peak quality (2.213 vs 2.208) but still drifted
   afterward, just at roughly half the rate. `eps=3e-4` did best of all:
   peak 2.203 @ iter 19500-20000 — better than the unstabilized runs'
   own best (2.208) — and then genuinely flat (+0.008 drift over the
   last ~4500 iters, not a trend). Mirrors the wide model's own
   `eps`-dose-dependent transition from partial dampening to true
   flatness (Attempts 9 vs 10), just scaled down for this narrower
   width. Both `eps=1e-4` and `eps=3e-4` runs reproduced exactly on a
   second run each (`torch.manual_seed(1337)` is fixed — see Conclusion
   6's correction), consistent with the eps margin also damping the
   floating-point-noise sensitivity behind Attempt 13's run-to-run split,
   not just the drift once divergence starts. Recommended at the time:
   keep the existing recipe (lr=5e-4, warmup=100, full cosine decay) and
   add `--adam-eps 3e-4` — matches or beats the unstabilized recipe's
   own best-case peak, with genuine stability instead of a lucky-run
   gamble. **Superseded by Conclusion 8 below.**
8. **`eps=5e-4`, not `3e-4`, is the best dose found for the deployment
   shape.** Attempt 15 swept the gap between Attempt 14's `3e-4` and the
   wide model's `1e-3`. `eps=5e-4` beat `3e-4` on both peak quality
   (2.185 vs 2.203) and flatness (+0.001 vs +0.008 drift) — the best
   result of the entire investigation, and the closest any stabilized
   run has gotten to the original unstabilized checkpoint's 2.022.
   Larger doses (`7e-4`, `1e-3`) were not better in the same sense `1e-4`
   was worse than `3e-4` in Attempt 14 — they simply hadn't finished
   converging within the 24000-iteration budget (still monotonically
   falling at the final logged point, no plateau reached), consistent
   with the general "larger eps trades peak quality for stability
   margin, and takes longer to get there" pattern first seen at the wide
   shape in Attempts 8-9. **Current recommendation for future
   deployment-shape training runs: keep the existing recipe (lr=5e-4,
   warmup=100, full cosine decay) and add `--adam-eps 5e-4`.** Attempt
   16 confirmed this: every dose between `3e-4` and `5e-4` (`3.5e-4`,
   `4e-4`, `4.5e-4`) improved monotonically toward `5e-4` with no
   interior optimum, so `5e-4` is the best point of that whole range,
   not just the best of the values originally tried. **Attempt 17 closed
   the remaining question**: giving `7e-4` and `1e-3` 50% more iterations
   (`36000` vs `24000`, with a matched, fully-decayed schedule rather
   than the same schedule mid-decay) still doesn't let either overtake
   `5e-4` — best results 2.221 and 2.239 respectively, both clearly worse
   than `5e-4`'s 2.185. `eps=5e-4` is a genuine winner, not an artifact
   of the 24000-iteration budget cutting the higher-`eps` runs off too
   early — nothing tested below it or above it, with or without more
   training room, has beaten it.
9. **The `eps` finding is VOCAB-specific — it does not transfer from
   VOCAB=1900 to VOCAB=16384, the shape actually deployed on real
   hardware.** Everything in Conclusions 1-8 was diagnosed at VOCAB=1900
   (this log's original scope) or the wider D=384 diagnostic shape;
   VOCAB=16384 is a third, separate shape with its own loss landscape.
   Attempt 20 found `eps=5e-4` is a clear *regression* at VOCAB=16384
   (best val 2.4212 vs the deployed `eps=3e-4`'s 2.4085) despite being
   the clear winner at VOCAB=1900 — the opposite direction. Attempt 21's
   own VOCAB=16384 sweep found a different, smaller winner (`eps=3.5e-4`,
   best val 2.4070, a modest +0.0015 improvement) via a genuinely
   non-monotonic curve (`4e-4` dips below both its neighbors), unlike
   VOCAB=1900's clean monotonic ramp in Attempt 16. Attempt 22 then found
   even that modest FP-level edge vanishes after QAT (2.3392 vs the
   deployed checkpoint's 2.3385 — a 0.0007 gap, noise-level). **Net
   practical conclusion: do not port an `eps` value found at one VOCAB
   size to another without re-sweeping at the target shape** — not just
   the specific number, but even the *direction* of the eps-vs-quality
   relationship can flip. At VOCAB=16384 specifically, the currently
   deployed `eps=3e-4` QAT checkpoint remains the best candidate found
   so far; the `eps=3.5e-4` alternative (Attempt 21-22) is a wash, not
   an improvement, and has not been promoted.

## Phase 2: why does `SauravP97/tiny-stories-hf` train stably at similar
   width/depth when kev-gpt's own recipe doesn't?

### Motivation

Everything above answers "how do we survive Adam's divergence at this
width" with a stabilizing `eps` margin. It doesn't answer a harder
question the reference repo
[`SauravP97/tiny-stories-hf`](https://github.com/SauravP97/tiny-stories-hf)
raises directly: that repo trains an 8-layer, hidden_size=256 GPT-Neo
(~19M params, comparable scale to kev-gpt's own D=384 diagnostic shape)
to genuinely coherent, low-repetition TinyStories completions, with a
plain HF `Trainer` run and no special stabilization. If their recipe is
simply stable at this width/depth and kev-gpt's isn't, the `eps` fix
above is a patch over a solvable difference, not the actual ceiling.

**First check: is their claim even real, not a cherry-picked README?**
Ran a seeded, multi-prompt sweep against their published checkpoint using
this repo's own objective repetition detector
(`model.filter_synth_corpus.is_degenerate`) — the same detector already
used to score kev-gpt's own generations, so results are comparable
without eyeballing either side:
```bash
python model/tinystories_hf_repro/sweep_tiny_stories_hf.py
```
(loads `SauravP97/tiny-stories-19M` from the Hub, 5 seeds x 5 prompts =
25 samples, `max_new_tokens=60` to match kev-gpt's own sampling
convention — an earlier run at `max_new_tokens=150` gave a misleadingly
high flag rate, a length-mismatch artifact of the detector, not a real
quality difference). **Result: 7/25 flagged** — clearly not perfect, but
qualitatively much cleaner than kev-gpt's own real-hardware output
("to help him to to help him get to help him get"). Their claim is real
enough to be worth root-causing, not a false lead.

**Second check: does porting their exact hyperparameters onto kev-gpt's
own D=384/VOCAB=16384 word-level model fix the divergence?** Four direct
attempts, all diverged — each is a genuine falsification, not a dead
end, and each is what motivated the methodology below instead of a fifth
guess:

| Attempt | Change from kev-gpt's own recipe | Result |
|---|---|---|
| long-run, `eps=1e-3` | bare cosine, 140000-iter target | diverged, killed @ iter 36000 (val 3.400) — `data/ckpt_teacher_d384_v16384_long.pt` |
| long-run 2, `eps=3e-3` | `lr-decay-iters=15000` (fast-decay-then-hold, per Attempt 6-11's own finding) | diverged, killed @ iter 84000 (val 3.796) — `data/ckpt_teacher_d384_v16384_long2.pt` |
| `beta2=0.999`, `eps=1e-8` | matches HF `Trainer`'s own Adam defaults exactly | diverged (best val 2.316 @ iter 3500, final 4.598 @ iter 15000) — `data/ckpt_teacher_d384_v16384_beta2test.pt` |
| `batch_size=8` | matches the reference repo's own batch size, everything else kev-gpt's own default | diverged worst of all four (best val 3.212 @ iter 3500-4000, final 5.696 @ iter 15000) — `data/ckpt_teacher_d384_v16384_batch8test.pt` |

None promoted or committed (all in gitignored `data/`). Porting single
hyperparameters from their recipe onto kev-gpt's own model/loop, one at a
time, in this direction, kept failing — which is the point where the
methodology flipped:

**Reproduce their recipe faithfully first (proving it's genuinely
stable, not a lucky README run), then walk it toward kev-gpt's own setup
one variable at a time, checking for divergence at each step** — instead
of guessing which of kev-gpt's own hyperparameters to try importing into
their architecture. Every step below holds a `MAX_STEPS`/`OUT_STATES`
CLI signature (`sys.argv[1]`/`sys.argv[2]`, default 15000 steps) and logs
one JSON line per eval (every 500 steps) to `--out-states`, so any step
can be re-run standalone.

**Quick-reference table — what changed at each step, and the LR
schedule used** (kept updated as new steps are added; every step used
peak `lr=5e-4` unless noted):

| Step | Axis changed | Schedule | Result |
|---|---|---|---|
| 0 (repro) | — (baseline, HF Trainer) | linear, no warmup | stable, val 1.712 |
| 1 (loop mechanics) | HF Trainer → kev-gpt's own loop | linear, no warmup | stable, val 1.708 |
| 2a (weight_decay) | 0.01 → 0.1 | linear, no warmup | stable, val 1.674 |
| 2b (beta2) | 0.999 → 0.95 | linear, no warmup | stable, val 1.812 |
| 2c (batch_size) | 8 → 64 | linear, no warmup | stable, val 1.376 |
| 3 (tokenizer) | BPE → kev-gpt word vocab | linear, no warmup | stable, val 1.454 |
| 4 (attention scope) | local/windowed → global | linear, no warmup | stable, val 1.459 |
| 5 (attention class) | GPT-Neo module → kev-gpt's real `CausalSelfAttention`, at D=256/n_layer=8 | linear, no warmup | stable, val 1.449 |
| 6 (width/depth) | D=256/n_layer=8 → D=384/n_layer=12 (kev-gpt's actual diverging shape) | linear, no warmup | stable, val 1.284 |
| 7 (LR schedule) | linear-no-warmup → kev-gpt's real `cosine_lr`, `warmup=100` | **cosine + warmup=100** | stable, val 1.311 |
| 8 (data framing) | per-story padded rows → kev-gpt's real flat-stream `get_batch`/`load_split`, `block=128` | cosine + warmup=100 | stable, val 1.728 |
| sanity check | literal `python -m model.train`, bare CLI defaults (`lr=1e-3`) — the real production entry point, not this harness | kev-gpt's real cosine + warmup=100 | **diverged** — val 2.371 @ iter 2500 → 4.731 @ iter 15000, reproducing the original signature exactly |
| 9 (precision mechanism) | fp16+`GradScaler` → kev-gpt's real `amp_dtype()`: bf16, `GradScaler` disabled | cosine + warmup=100 | stable, val 1.716 |
| 10 (peak LR) | `lr=5e-4` → kev-gpt's real `--lr` default, `1e-3` | cosine + warmup=100 | stable, val 1.719 |

Every axis above tested at kev-gpt's own default `eps=1e-8`/`beta2=0.95`
except where the row itself is testing `beta2` (2b). See each step's own
section below for the full per-500-step trajectory and analysis.

### Step 0: reproduce `SauravP97/tiny-stories-hf`'s exact recipe

`model/tinystories_hf_repro/repro_hf_recipe.py` — plain HF `Trainer`,
their exact config: `GPTNeoConfig(hidden_size=256, num_layers=8,
num_heads=16, attention_types=[[["local"], 8]])` (GPT-Neo **local/
windowed** attention, not kev-gpt's own global attention), GPT-Neo-125M
BPE tokenizer (~50257 vocab, no `<unk>` ever), `roneneldan/TinyStories`
raw dataset (same underlying corpus kev-gpt already uses via
`keviniser.fetch_tinystories`, but NOT Kevinised — their repo trains on
the raw text), `per_device_train_batch_size=8`, `learning_rate=5e-4`,
`weight_decay=0.01`, `fp16=True`, and HF `TrainingArguments`' own
defaults for everything not set explicitly — reverse-engineered to be
`adam_beta2=0.999`, `adam_epsilon=1e-8`, `max_grad_norm=1.0`, and
`lr_scheduler_type="linear"` with **zero warmup** (pure linear decay from
`lr` to 0 over `max_steps` — confirmed by matching this run's own logged
LR values against the formula `lr*(1-step/max_steps)`; nothing in
`TrainingArguments` sets `warmup_steps`/`warmup_ratio`, so HF's default
`linear` scheduler starts decaying from step 0).

```bash
cd ~/tiny-stories-hf && source venv/bin/activate   # user's own existing HF venv, not a new isolated one
pip install "accelerate>=1.1.0"                     # HF Trainer's own hard requirement, was missing
python /home/tparng/kev-gpt/model/tinystories_hf_repro/repro_hf_recipe.py 15000 /tmp/repro_hf_states.jsonl
```
Result: val 10.87 (step 0) -> **1.712 @ step 15000**, monotonic the whole
way, zero divergence. This is the reference/ground-truth trajectory
every ablation step below is compared against — their claim of stable
training at this width/depth is real on this hardware, not
recipe-specific luck.

### Step 1: swap the framework — HF `Trainer` -> kev-gpt's own manual loop

`model/tinystories_hf_repro/ablation_step1_framework.py` — same
architecture/tokenizer/data/hyperparameters as Step 0 (batch=8, lr=5e-4,
weight_decay=0.01, beta2=0.999, eps=1e-8, their exact linear-no-warmup
schedule), but the training loop mechanics are now kev-gpt's own:
`autocast`+`GradScaler`, manual `AdamW` with kev-gpt's decay/no_decay
param-grouping (`p.dim() >= 2` gets weight decay, matching
`model/train.py`), `clip_grad_norm_(..., 1.0)`, manual `opt.step()` —
isolates "does kev-gpt's own loop code introduce the instability" from
every hyperparameter *value* difference, which come later.

```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step1_framework.py 15000 /tmp/ablation1_states.jsonl
```

**First run found a real efficiency gap** (val 2.831 vs Step 0's 1.712 at
the same step count) — user directly asked whether the dataset actually
matched; confirmed yes (HF `datasets.map()`'s cache fingerprint matched),
but chasing that question surfaced the real bug: a **double-shift labels
bug**. `get_batch()` was pre-shifting `x = rows[:, :-1]`, `y = rows[:,
1:]` (nanoGPT convention) before handing both to a HF `*ForCausalLM`
model that ALSO shifts internally (`shift_logits = logits[:, :-1]`,
`shift_labels = labels[:, 1:]`) — so the model was being trained against
labels two positions ahead of its predictions on every single step.
Fixed by passing the full, unshifted sequence (`x = rows; y =
rows.clone()`), matching what `DataCollatorForLanguageModeling` itself
does. A second, smaller confound was fixed at the same time: kev-gpt's
own bf16-by-default convention (`amp_dtype()`) was silently overriding
their explicit `fp16=True` — forced `dtype = torch.float16` to match.

After both fixes, a 500-step smoke test tracked the reference closely
(val 3.708 vs their 3.576), and the full 15000-step run **converged to
val 1.7084**, matching Step 0's 1.712 almost exactly. **Conclusion:
kev-gpt's own loop mechanics are NOT the cause of the divergence** — once
ported correctly, they're functionally equivalent to HF `Trainer`'s.

### Step 2a: `weight_decay` 0.01 -> 0.1 (kev-gpt's own value)

`model/tinystories_hf_repro/ablation_step2a_weightdecay.py` — only
change from Step 1: `WEIGHT_DECAY = 0.1`.
```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step2a_weightdecay.py 15000 /tmp/ablation2a_states.jsonl
```
Result: no divergence, final val **1.6742 @ step 15000** (best **1.6712
@ step 13500**) — slightly *better* than Step 1. **`weight_decay` ruled
out** as a cause; kept at kev-gpt's own value for every subsequent step.

### Step 2b: `beta2` 0.999 -> 0.95 (kev-gpt's own value)

`model/tinystories_hf_repro/ablation_step2b_beta2.py` — Step 2a plus
`BETA2 = 0.95` (weight_decay=0.1 carried forward).
```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step2b_beta2.py 15000 /tmp/ablation2b_states.jsonl
```
Result: no divergence, but consistently worse than Step 2a throughout
the run (~0.14-0.16 higher val loss at matched steps), final val
**1.8119 @ step 15000**. **`beta2` ruled out as a standalone cause of
catastrophic divergence** — it's a real, measurable efficiency cost (a
slower-reacting variance EMA makes each step less well-calibrated), not
an instability trigger. This mirrors Attempt 8 above, where lowering
`beta2` on kev-gpt's own D=384/VOCAB=4096 shape *also* didn't move the
divergence at all — consistent evidence that `beta2` isn't on the causal
path in either direction.

### Step 2c: `batch_size` 8 -> 64 (kev-gpt's own default)

`model/tinystories_hf_repro/ablation_step2c_batchsize.py` — Step 2b plus
`BATCH = 64` (weight_decay=0.1 + beta2=0.95 carried forward), the last of
the three individual hyperparameter differences. This laptop's GPU has
only 7.62GB total VRAM; a batch=64/seq_len=512/vocab~50257 causal-LM
loss computation OOMs (`logits.float()` alone needs ~6.1GB) — fixed with
**gradient accumulation**: `MICRO_BATCH = 8`, `ACCUM_STEPS = 8`, looping
8 micro-batches of 8 with each `.backward()` scaled by `1/ACCUM_STEPS`
before a single `scaler.step(opt)` per full effective-batch step. This
reproduces batch=64's true gradient-noise statistics without
materializing the full batch in one forward pass — a resource
workaround, not a change to what's being tested.
```bash
# smoke test first (500 steps) to confirm no OOM/crash before committing to the full run
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step2c_batchsize.py 500 /tmp/ablation2c_smoke_states.jsonl
# full run
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step2c_batchsize.py 15000 /tmp/ablation2c_states.jsonl
```
Smoke test: clean exit, no OOM (val 3.18 @ step 500 of a 500-step-total
schedule — LR fully decayed within the smoke test's own compressed
horizon, not comparable to the other steps' step-500 values).

**The full run hung twice before completing** — both times with the
same signature (main Python thread pegged at ~100% CPU, GPU utilization
~0-1%, power draw at idle 20W/80W, no traceback/OOM/dmesg error, no new
output for hours), just at different step counts (~8500ish the first
time, exactly step 11065 mid-micro-batch the second time, pinned down
precisely by the heartbeat prints added after the first hang). The
user's own explanation for at least the second hang: pausing/resuming an
unrelated Jupyter notebook sharing the same GPU can stall another
process's CUDA calls. Both hangs were unrecoverable — **the script never
saved a checkpoint**, only eval stats to `OUT_STATES`, so killing the
hung process lost all training progress each time (the underlying model/
optimizer state only ever lived in the killed process's GPU memory).
Fixed by adding real checkpoint save/resume: every eval step now also
writes `model.state_dict()`/`opt.state_dict()`/`scaler.state_dict()` to
`CKPT` (`sys.argv[3]`, defaults to `OUT_STATES` with `_ckpt.pt` in place
of `.jsonl`); if `CKPT` already exists at startup, training resumes from
that step instead of restarting at 0. Verified with a 20-step round-trip
(run to completion, rerun same paths, confirms "resumed at step 21" and
exits immediately without redoing work) before relaunching the real run.

**Third attempt, clean start, completed all 15000 steps without
incident**:

| step | val | | step | val |
|---|---|---|---|---|
| 0 | 10.911 | | 8000 | 1.4578 |
| 500 | 2.814 | | 8500 | 1.4893 |
| 1000 | 2.293 | | 9000 | 1.4964 |
| 1500 | 1.989 | | 9500 | 1.5090 |
| 2000 | 1.936 | | 10000 | 1.4622 |
| 2500 | 1.867 | | 10500 | 1.4134 |
| 3000 | 1.736 | | 11000 | 1.4152 |
| 3500 | 1.747 | | **11500** | **1.3605 (best)** |
| 4000 | 1.597 | | 12000 | 1.3766 |
| 4500 | 1.600 | | 12500 | 1.4465 |
| 5000 | 1.586 | | 13000 | 1.4033 |
| 5500 | 1.631 | | 13500 | 1.3852 |
| 6000 | 1.550 | | 14000 | 1.3880 |
| 6500 | 1.510 | | 14500 | 1.3655 |
| 7000 | 1.536 | | **15000** | **1.3762 (final)** |
| 7500 | 1.504 | | | |

349.0 minutes total (~5h49m). Val loss oscillates in a narrow 1.36-1.63
band from step ~6000 onward, trending gently downward overall (best
1.3605 @ step 11500, final 1.3762 @ step 15000) — never once shows the
sustained, runaway climb that characterizes every diverged AdamW run
earlier in this log (compare Experiment 1's train+val climbing together
for 21,500 of 24,000 iterations, or Attempt 6's val +61% at a frozen
floor LR). **`batch_size` ruled out as a cause of divergence** — this is
the last of the three individual kev-gpt hyperparameters
(`weight_decay=0.1`, `beta2=0.95`, `batch_size=64`), and with all three
now combined together on the reference architecture/tokenizer/data, the
run is still fully stable. **All three of kev-gpt's own hyperparameter
VALUES are cleared as the cause of the divergence kev-gpt's own D=384/
VOCAB=16384 word-level model shows.** The two remaining untested axes are
the tokenizer (BPE, ~50257 vocab, no `<unk>` vs kev-gpt's closed
word-level vocab with an `<unk>` fallback) and the architecture (GPT-Neo
local/windowed attention vs kev-gpt's own global attention) — one of
those two is now the leading candidate for what actually differs between
a recipe that trains stably at this width/depth and one that doesn't.

### Reproducing this investigation from scratch

```bash
cd ~/tiny-stories-hf && source venv/bin/activate   # or any venv with torch+cuda, transformers, datasets, accelerate>=1.1.0
python /home/tparng/kev-gpt/model/tinystories_hf_repro/sweep_tiny_stories_hf.py                              # sanity-check their published checkpoint's quality first
python /home/tparng/kev-gpt/model/tinystories_hf_repro/repro_hf_recipe.py 15000 /tmp/repro_hf_states.jsonl   # Step 0
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step1_framework.py 15000 /tmp/ablation1_states.jsonl
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step2a_weightdecay.py 15000 /tmp/ablation2a_states.jsonl
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step2b_beta2.py 15000 /tmp/ablation2b_states.jsonl
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step2c_batchsize.py 500 /tmp/ablation2c_smoke_states.jsonl   # smoke first, real GPU VRAM permitting
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step2c_batchsize.py 15000 /tmp/ablation2c_states.jsonl
```
First run of `repro_hf_recipe.py` downloads `roneneldan/TinyStories`
(~2GB, unauthenticated HF Hub, can be slow — launch with `nohup ... &`
and monitor the cache dir size rather than trusting a foreground
timeout) and tokenizes the full 2.1M-row train split (~5 min, cached by
`datasets.map()`'s fingerprint after the first run — copying a script to
a new filename invalidates that cache and forces re-tokenization, so
expect the ~5 min hit again per new script name in this directory).
Each script prints one line every 500 steps and writes matching JSON
lines to the path in `sys.argv[2]`; watch for `torch.OutOfMemoryError`
(a real 7.62GB VRAM ceiling on this hardware, not a bug — resolved for
batch=64 via the gradient-accumulation constants at the top of
`ablation_step2c_batchsize.py`, adjust `MICRO_BATCH` down further on a
smaller GPU) and for `ImportError: ... requires accelerate>=1.1.0` (`pip
install "accelerate>=1.1.0"` in whichever venv is running these).
`ablation_step2c_batchsize.py` specifically also checkpoints model/
optimizer/scaler state to `sys.argv[3]` (default: `sys.argv[2]` with
`_ckpt.pt` in place of `.jsonl`) every 500 steps and auto-resumes from it
if present at startup — if a run of that script hangs (watch for the
per-micro-batch heartbeat lines going stale in the log, GPU utilization
near 0% via `nvidia-smi`, and a killed process's TIME still climbing in
`ps`, all simultaneously — the exact signature seen twice in this
investigation), kill it and just re-run the identical command; it picks
up from the last saved step instead of restarting at 0.

### The architecture axis: GPT-Neo local/windowed attention vs kev-gpt's global attention

With all three of kev-gpt's own hyperparameters cleared (Steps 2a-2c),
the two remaining candidates are the tokenizer and the attention
mechanism. Comparing `CausalSelfAttention` (`model/gpt.py`) against the
installed `transformers` library's `GPTNeoSelfAttention` (confirmed by
direct instantiation that `SauravP97/tiny-stories-hf`'s exact config
selects `_attn_implementation="eager"`, i.e. the manual path below, not
a fused SDPA/flash kernel) surfaces several real differences:

1. **Windowed vs. full attention.** kev-gpt: full causal attention via
   `F.scaled_dot_product_attention(..., is_causal=True)` — every token
   attends to all previous tokens. GPT-Neo "local": a token only attends
   to the previous `window_size` tokens (default 256), via
   `bias = torch.bitwise_xor(bias, torch.tril(bias, -window_size))`. In
   this specific repro, `attention_types=[[["local"], 8]]` makes *all*
   8 layers local (not GPT-Neo's own default alternating
   global/local), and `window_size=256` is exactly half of
   `max_position_embeddings=512` — so the effective (indirect,
   stacked-layer) receptive field already covers the full sequence by
   layer 2 of 8. The restriction is real but much weaker than "windowed
   attention" usually implies at these specific settings.
2. **No `1/sqrt(head_dim)` scaling.** GPT-Neo's `_attn` computes
   `torch.matmul(query, key.transpose(-1, -2))` with no division by
   `sqrt(head_dim)` anywhere — confirmed deliberate, not an oversight
   (even the FlashAttention2 variant explicitly passes
   `softmax_scale=1.0` to override the default). kev-gpt's SDPA call
   always applies the standard scaling internally. This cuts *against*
   GPT-Neo's choice being a stabilizing factor — omitting the scaling
   is the numerically *less* safe option, and their `head_dim=16`
   (`hidden_size=256/num_heads=16`) is smaller than kev-gpt's own
   `head_dim=64` convention, compounding it further — yet their recipe
   still trains cleanly (Step 0).
3. **Explicit fp32 QK^T + softmax.** GPT-Neo upcasts query/key to fp32
   before the QK^T matmul regardless of the active autocast dtype, and
   casts the softmax output back down afterward — an explicit,
   auditable precision path. kev-gpt's SDPA call has no equivalent: the
   actual compute dtype of the fused kernel is opaque, decided by
   whichever backend SDPA dispatches to under `autocast`. Given this
   whole investigation is about logit-scale/gradient-variance blowup at
   width, and Step 1 earlier in this Phase already found one real
   silent precision confound (bf16-vs-fp16), this looked like the most
   promising untested lead — cheap to isolate (one code path, one CLI
   flag) versus a full tokenizer swap.
4. **QKV/bias structure and dropout plumbing** differ too (kev-gpt: one
   fused `qkv` Linear, `bias=False` everywhere by convention; GPT-Neo:
   three separate `q_proj`/`k_proj`/`v_proj`, `out_proj` always
   `bias=True`) but aren't live differences at these dropout=0 settings
   and aren't a plausible instability mechanism on their own.

**Tested #3 directly — falsified.** Added `GPTConfig.attn_fp32` /
`--attn-fp32` (`model/gpt.py`, `model/train.py`): when set,
`CausalSelfAttention.forward` bypasses SDPA and does the manual
GPT-Neo-style computation instead — `q`/`k` upcast to `.float()` before
the matmul, scaled by `1/sqrt(head_dim)` (kept, since scaling itself is
a separate variable from precision — see #2 above), causal-masked,
softmaxed, then cast back to `v`'s dtype before the second matmul.
Single-variable test against the exact same vanilla, *unstabilized*
recipe `beta2test`/`batch8test` used (default `--adam-eps 1e-8`,
default `--adam-beta2 0.95`, `--lr 3e-4 --warmup 2000`, no
`--lr-decay-iters` override) — the setup already known to diverge
without the Attempt-11 `eps=1e-3` fix:
```bash
python -m model.train --max-iters 15000 --lr 3e-4 --warmup 2000 --attn-fp32 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_teacher_d384_v16384_fp32attn.pt --states data/states_teacher_d384_v16384_fp32attn.jsonl
```
Throughput (~7.5 it/s) was comparable to the SDPA baseline (~6.8 it/s) —
the manual path isn't meaningfully slower at this scale.

| iter | val | logit_max | lnf_norm |
|---|---|---|---|
| 0 | 9.782 | 2.5 | 19.60 |
| 2000 | 2.611 | 19.4 | 24.59 |
| 3500 | 2.348 | 22.1 | 30.09 |
| **4000** | **2.346 (best)** | 25.0 | 31.70 |
| 5000 | 2.400 | 27.5 | 34.76 |
| 8500 | 2.973 | 28.9 | 43.95 |
| 11000 | 3.552 | 34.8 | 48.45 |
| 15000 | 4.462 | 43.5 | 51.81 |

Bottoms at iter 4000 — the same onset point every unstabilized AdamW
run at this width has shown throughout this entire log — then climbs
continuously for the remaining 11,000 iterations to val=4.462, with
`logit_max`/`lnf_norm` growing unbounded the whole way, the identical
signature Attempts 1-9 already characterized in exhaustive detail.
**The explicit fp32 QK^T upcast alone does not prevent the divergence.**
Whatever lets GPT-Neo's recipe train stably without any Adam-side
stabilization, it isn't this — the precision path around the attention
matmul was a plausible, cheap-to-test lead, and it's now cleanly ruled
out, the same way Attempts 1-4 ruled out LR schedule, init scaling,
weight-decay grouping, and z-loss earlier in this log. `GPTConfig.
attn_fp32` / `--attn-fp32` are kept in the codebase (default off,
zero behavior change) as a reusable diagnostic, not removed just
because this test came back negative.

**Where this leaves the architecture axis**: #1 (windowing) and #2 (no
`sqrt(head_dim)` scaling) remain untested directly, though #2 argues
against itself (the safer choice is already what kev-gpt does) and #1
is weak at these specific settings (window≈seq_len/2, all-local). The
tokenizer axis (BPE/no-`<unk>` vs kev-gpt's closed word-level vocab with
an `<unk>` fallback) is now the strongest remaining untested candidate.

### Step 3: swap the tokenizer — GPT-Neo BPE -> kev-gpt's own closed word-level vocab

`model/tinystories_hf_repro/ablation_step3_tokenizer.py` — Step 2c's
script (already at ALL of kev-gpt's own hyperparameter values:
`weight_decay=0.1`, `beta2=0.95`, `batch_size=64` via gradient
accumulation) with only the tokenizer swapped: replaced the GPT-Neo-125M
BPE tokenizer with kev-gpt's own `model/word_data.py` closed word-level
vocab, reusing `data/word_v16384/meta.json`'s `stoi` directly (bit-
identical to the real deployed VOCAB=16384 tokenizer, not regenerated).
Kept the same per-story framing Steps 0-2c used (one row = one story,
regex-tokenized, `<eos>` appended, truncated/padded to `SEQ_LEN=512`
with `<eos>` as filler, every `<eos>`-valued label position masked —
mirrors `pad_token==eos_token` from the BPE runs) rather than kev-gpt's
own flat-stream/random-window sampling convention
(`model/data.py`'s `load_split`), specifically to keep this a clean
single-variable swap instead of confounding tokenizer with data framing.
`GPTNeoConfig(vocab_size=16384, ...)` — everything else (architecture,
hidden_size=256, num_layers=8, num_heads=16, `attention_types=[[["local"],
8]]`, LR=5e-4 linear-no-warmup schedule) unchanged from Step 2c.
Sanity-checked the tokenizer directly before the full run (round-trips a
sample sentence exactly, 0 `<unk>` on ordinary text) — see the "Reproducing
this investigation from scratch" commands below.

```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step3_tokenizer.py 15000 /tmp/ablation3_states.jsonl
```
Model is 10.64M params (smaller than Step 2c's 19.31M — the tied
embedding/head table shrinks from ~50257×256 to 16384×256 with the
closed vocab). Throughput comparable to the BPE runs.

| step | val | | step | val |
|---|---|---|---|---|
| 0 | 9.747 | | 8000 | 1.631 |
| 500 | 2.881 | | 8500 | 1.563 |
| 1000 | 2.425 | | 9000 | 1.529 |
| 1500 | 2.188 | | 9500 | 1.580 |
| 2000 | 2.027 | | 10000 | 1.552 |
| 2500 | 1.884 | | 10500 | 1.507 |
| 3000 | 1.850 | | 11000 | 1.488 |
| 3500 | 1.783 | | 11500 | 1.507 |
| 4000 | 1.754 | | 12000 | 1.482 |
| 4500 | 1.744 | | 12500 | 1.466 |
| 5000 | 1.728 | | 13000 | 1.483 |
| 5500 | 1.660 | | **13500** | **1.446 (best)** |
| 6000 | 1.654 | | 14000 | 1.484 |
| 6500 | 1.663 | | 14500 | 1.468 |
| 7000 | 1.618 | | **15000** | **1.454 (final)** |
| 7500 | 1.634 | | | |

276.5 minutes total. Val loss trends downward the entire run, oscillating
in a narrow 1.45-1.9 band from step ~3000 onward — no bottom-then-climb
shape anywhere, including well past iter 4000 (where the fp32-attention
test above turned upward) and past iter 11065 (where Step 2c's own
transient GPU hangs happened, unrelated but the same iteration range).
**Tokenizer ruled out as a cause of divergence** — kev-gpt's own closed
word-level vocab, substituted directly into the reference architecture/
hyperparameters, trains exactly as stably as GPT-Neo's own BPE tokenizer
did in Steps 0-2c.

**This clears every axis tested so far**: framework/loop mechanics
(Step 1), all three individual hyperparameters (Steps 2a-2c), attention
precision (the `--attn-fp32` test), and now the tokenizer (Step 3) are
all ruled out, one at a time, as standalone causes of the divergence
kev-gpt's own D=384/VOCAB=16384 model shows under its own default
recipe. What remains: (a) windowed/local vs. global attention *scope*
itself (distinct from the precision test — never directly varied; Steps
0-3 all kept GPT-Neo's local/windowed attention throughout), and (b) the
possibility that no single axis is sufficient alone — the instability
might only appear from a *specific combination* neither this walk nor
the original SCALE-UP-LOG's own Attempts 1-22 has tried (e.g. kev-gpt's
own global attention combined with kev-gpt's own default `eps=1e-8`,
which is what actually diverges, vs. every stable run in this Phase
being GPT-Neo's local attention regardless of which other axis changed).
Testing (a) directly — porting kev-gpt's own `GPT`/`CausalSelfAttention`
class (global, SDPA-based) into this same walk, holding data/hyper-
parameters at their now-fully-kev-gpt values — is the natural next step
and hasn't been attempted yet.

### Step 4: swap attention SCOPE only — GPT-Neo local/windowed -> GPT-Neo "global"

`model/tinystories_hf_repro/ablation_step4_globalattn.py` — Step 3's
script (kev-gpt's own closed word-level vocab, `weight_decay=0.1`,
`beta2=0.95`, `batch_size=64` via gradient accumulation) with exactly
one line changed: `attention_types=[[["global"], 8]]` in place of
`[[["local"], 8]]`. Deliberately stayed inside `GPTNeoSelfAttention`
(not a full port of kev-gpt's own SDPA-based `CausalSelfAttention`) to
isolate attention *scope* alone — a full class swap would also change
scaling (kev-gpt scales by `1/sqrt(head_dim)`, GPT-Neo doesn't),
precision path (GPT-Neo's explicit fp32 QK^T upcast vs kev-gpt's opaque
fused SDPA kernel), and Q/K/V/bias structure all at once, confounding
several already-distinct variables from the earlier architecture
comparison.

**Verified directly, three ways, that this run really trained the
revised config** (prompted by a direct question mid-run about whether
the process might silently still be running Step 3's local-attention
code): (1) `/proc/<pid>/cmdline` showed the process executing
`ablation_step4_globalattn.py`, not a stale copy; (2) the on-disk file
at that path contains `attention_types=[[["global"], 8]]` (unedited
since the process started, so this is what it read at import time); (3)
instantiating the identical `GPTNeoConfig` fresh confirmed
`attention_layers == ['global']*8`, and directly inspecting the
resulting causal-mask buffer showed row 300 attends all the way back to
column 0 (301 allowed positions) — if still windowed at `window_size=256`
it would cut off 44 columns short of that.

```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step4_globalattn.py 15000 /tmp/ablation4_states.jsonl
```

| step | val | | step | val |
|---|---|---|---|---|
| 0 | 9.753 | | 8000 | 1.593 |
| 500 | 2.865 | | 8500 | 1.604 |
| 1000 | 2.342 | | 9000 | 1.539 |
| 1500 | 2.161 | | 9500 | 1.559 |
| 2000 | 2.046 | | 10000 | 1.580 |
| 2500 | 1.921 | | 10500 | 1.513 |
| 3000 | 1.824 | | 11000 | 1.470 |
| 3500 | 1.772 | | 11500 | 1.493 |
| 4000 | 1.797 | | 12000 | 1.480 |
| 4500 | 1.696 | | 12500 | 1.512 |
| 5000 | 1.715 | | 13000 | 1.487 |
| 5500 | 1.702 | | **13500** | **1.417 (best)** |
| 6000 | 1.633 | | 14000 | 1.422 |
| 6500 | 1.678 | | 14500 | 1.475 |
| 7000 | 1.594 | | **15000** | **1.459 (final)** |
| 7500 | 1.587 | | | |

278.1 minutes total. Val loss trends downward the whole run, oscillating
in a narrow 1.4-1.9 band from step ~5000 onward — no bottom-then-climb
shape anywhere, essentially indistinguishable in character (and closely
comparable numerically) from Step 3's own trajectory (best 1.446 @ step
13500, final 1.454 @ step 15000, vs. this run's 1.417 @ step 13500,
1.459 @ step 15000). **Attention scope (local/windowed vs. global) ruled
out as a cause of divergence** — switching it, alone, with every other
axis held at kev-gpt's own values, changes essentially nothing about the
training dynamics.

**This clears every axis identified in the Phase 2 architecture
comparison, individually**: loop mechanics (Step 1), all three
hyperparameters (Steps 2a-2c), attention precision (`--attn-fp32`),
tokenizer (Step 3), and now attention scope (Step 4) — none of them,
tested alone, reproduces kev-gpt's own D=384/VOCAB=16384 divergence.
Two possibilities remain, both un-eliminated by this walk: (a) the
divergence needs kev-gpt's own *specific* attention implementation
(SDPA's opaque precision path + `1/sqrt(head_dim)` scaling + fused
qkv + `head_dim=64`, all together, not any one swapped into GPT-Neo's
module) — the only test that would settle this is literally substituting
kev-gpt's own `GPT`/`CausalSelfAttention` class into this harness, not
an approximation of one axis inside GPT-Neo's own code; or (b) no single
architectural or tokenizer axis is sufficient alone, and the instability
is a property of the *combination* already tested exhaustively in this
log's own Attempts 1-22 (Adam's variance normalization interacting with
width, fixed by `eps`) — i.e. the reference recipe isn't stable because
of anything architectural at all, it simply never needed the `eps`
fix because nothing about GPT-Neo's setup was ever compared against
kev-gpt's own *unstabilized* `eps=1e-8` default until this whole Phase 2
walk, and every single Phase 2 run (Steps 0-4) used kev-gpt's default
`eps=1e-8`/`beta2=0.95` and never diverged regardless of architecture,
tokenizer, or hyperparameters — while kev-gpt's own D=384/VOCAB=16384
model reliably diverges at those same default settings. That asymmetry
(GPT-Neo-based configs never diverge at `eps=1e-8`; kev-gpt's own GPT
class reliably does) now can only be explained by something in kev-gpt's
own `CausalSelfAttention`/`GPT` implementation itself, making the direct
class-swap test the clear next step.

### Step 5: port kev-gpt's own real `GPT`/`CausalSelfAttention` class in — still no divergence, and why that's not yet the full answer

`model/tinystories_hf_repro/ablation_step5_kevgptattn.py` — replaces
GPT-Neo entirely: `sys.path.insert(0, "/home/tparng/kev-gpt"); from
model.gpt import GPT, GPTConfig` — kev-gpt's real, unmodified production
class, not an approximation inside GPT-Neo's module. Held at kev-gpt's
own convention: `head_dim=64` (`n_head=4`, `n_embd=256` — matching
kev-gpt's real deployed/diagnostic models, not GPT-Neo's `head_dim=16`),
`bias=False` everywhere, `attn_fp32=False` (kev-gpt's real default SDPA
path, not the already-falsified diagnostic branch). `n_layer=8` to match
Step 4's depth as closely as possible. Same data as Steps 3-4 (kev-gpt's
own word-level VOCAB=16384, per-story padded-to-512 framing, `eos`-
masked) and same hyperparameters (`weight_decay=0.1`, `beta2=0.95`,
`eps=1e-8` default, `batch_size=64` via grad accumulation, `lr=5e-4`
linear-no-warmup). kev-gpt's own `GPT.forward` uses a pre-shift
convention (`x=data[i:i+block]`, `y=data[i+1:i+1+block]`, matching
`model/train.py`'s own `get_batch`) and has no built-in loss-masking
mechanism (its real flat-stream training never needs one) — the harness
calls `model(x)` with `targets=None` to get logits only, then computes
`F.cross_entropy(..., ignore_index=-100)` itself, keeping kev-gpt's own
model code completely untouched while preserving the same eos-masking
convention Steps 0-4 used.

```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step5_kevgptattn.py 15000 /tmp/ablation5_states.jsonl
```
10.49M params (comparable to Steps 3-4's ~10.6M). Notably faster than
every GPT-Neo-based step (56.5 min total vs. Steps 3-4's 276-278 min) —
kev-gpt's fused SDPA call is meaningfully cheaper than GPT-Neo's manual
eager-mode attention (full matmul/mask/softmax/matmul, no fused kernel).

| step | val | | step | val |
|---|---|---|---|---|
| 0 | 9.746 | | 8000 | 1.599 |
| 500 | 3.001 | | 8500 | 1.574 |
| 1000 | 2.441 | | 9000 | 1.553 |
| 1500 | 2.164 | | 9500 | 1.577 |
| 2000 | 2.012 | | 10000 | 1.565 |
| 2500 | 1.866 | | 10500 | 1.511 |
| 3000 | 1.832 | | 11000 | 1.517 |
| 3500 | 1.802 | | 11500 | 1.533 |
| 4000 | 1.775 | | 12000 | 1.529 |
| 4500 | 1.712 | | 12500 | 1.541 |
| 5000 | 1.723 | | 13000 | 1.495 |
| 5500 | 1.665 | | 13500 | 1.501 |
| 6000 | 1.688 | | 14000 | 1.499 |
| 6500 | 1.661 | | 14500 | 1.466 |
| 7000 | 1.612 | | **15000** | **1.449 (best = final)** |
| 7500 | 1.638 | | | |

56.5 minutes total. Val loss trends downward the entire run, oscillating
in the same narrow ~1.4-2.0 band Steps 3-4 showed, with the true minimum
landing right at the last logged step rather than climbing away from an
earlier peak — the cleanest of the five Phase 2 runs so far, if
anything. **kev-gpt's own real attention class, ported in unmodified,
does not diverge either.**

**This is a real result, but it does NOT close the investigation — it
narrows what's actually been tested.** Every Phase 2 run (Steps 0-5) has
held three things fixed at the *reference recipe's* values, never at
kev-gpt's own actual production values, because they were inherited
from Step 0's faithful reproduction and never flagged as a variable in
their own right:
1. **Width/depth**: `n_embd=256, n_layer=8` throughout — the reference
   repo's own shape, not kev-gpt's actual diverging shape
   (`n_embd=384, n_layer=12`, `n_head=6`, from the original SCALE-UP-LOG's
   Experiment 1 onward). Step 5 tests kev-gpt's real attention CLASS,
   but not at the width where the divergence was ever actually observed.
2. **LR schedule**: `lr=5e-4`, pure linear decay from step 0, no warmup
   — the reference's own schedule (confirmed in Step 0 by reverse-
   engineering HF `TrainingArguments`' defaults), never kev-gpt's own
   actual recipe (cosine decay, `warmup=100`, different peak/floor
   shape entirely).
3. **Data framing**: per-story, padded/truncated to a fixed 512-token
   row, `eos`-masked — never kev-gpt's own actual flat continuous-
   stream/random-contiguous-window sampling (`model/data.py`'s
   `load_split` + `get_batch`), which has no padding, no story
   boundaries treated specially, and a much shorter `block_size=128` in
   the real deployed/diagnostic recipes.

None of these three was ever the variable under test in Steps 0-5 —
they were the fixed scaffolding the whole walk was built on top of,
carried over unexamined from Step 0's reproduction. **The real, still-
open question**: does kev-gpt's own attention class, AT kev-gpt's own
actual diverging width/depth (`n_embd=384, n_layer=12, n_head=6`), still
train cleanly inside this harness (kev-gpt tokenizer, `eps=1e-8`
default, this harness's LR schedule/data framing)? If it diverges once
scaled up to that shape — with every other axis this Phase 2 walk has
already cleared held fixed — that would finally isolate **width/depth
itself**, in combination with kev-gpt's own attention implementation, as
the trigger neither the original SCALE-UP-LOG (which never isolated
architecture from data/LR-schedule) nor this Phase 2 walk (which never
varied width/depth) has directly tested. If it *still* doesn't diverge
even at that width, the remaining candidates would be the LR
schedule shape or the data-framing convention themselves — both still
completely unexamined by this entire investigation.

### Step 6: scale Step 5 UP to kev-gpt's actual diverging shape — still no divergence

`model/tinystories_hf_repro/ablation_step6_fullscale.py` — Step 5
verbatim, with exactly the `GPTConfig` width/depth changed:
`n_embd=384, n_layer=12, n_head=6` (`head_dim=384/6=64`, same convention
as Step 5, just wider/deeper) in place of `n_embd=256, n_layer=8,
n_head=4` — the original SCALE-UP-LOG's own Experiment-1-onward
diverging shape, reproduced exactly (`n_layer=12`, `n_head=6`, `n_embd=384`).
Everything else identical to Step 5: kev-gpt's own real, unmodified
`GPT`/`CausalSelfAttention` class, kev-gpt's own word-level VOCAB=16384
tokenizer, `weight_decay=0.1`, `beta2=0.95`, `eps=1e-8` default,
`batch_size=64` via grad accumulation, `lr=5e-4` linear-no-warmup,
per-story padded-to-512 data framing.

```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step6_fullscale.py 15000 /tmp/ablation6_states.jsonl
```
27.53M params — matches the original SCALE-UP-LOG's Experiment 1 exactly
(`22.82M` there was VOCAB=4096; this is VOCAB=16384, hence larger,
consistent with a bigger tied embedding/head table at the same
width/depth).

| step | val | | step | val |
|---|---|---|---|---|
| 0 | 9.764 | | 8000 | 1.420 |
| 500 | 2.854 | | 8500 | 1.466 |
| 1000 | 2.256 | | 9000 | 1.453 |
| 1500 | 2.072 | | 9500 | 1.416 |
| 2000 | 1.880 | | 10000 | 1.392 |
| 2500 | 1.788 | | 10500 | 1.354 |
| 3000 | 1.728 | | 11000 | 1.336 |
| 3500 | 1.680 | | 11500 | 1.403 |
| 4000 | 1.653 | | 12000 | 1.367 |
| 4500 | 1.653 | | 12500 | 1.326 |
| 5000 | 1.557 | | 13000 | 1.326 |
| 5500 | 1.541 | | **13500** | **1.283 (best)** |
| 6000 | 1.526 | | 14000 | 1.299 |
| 6500 | 1.552 | | 14500 | 1.376 |
| 7000 | 1.445 | | **15000** | **1.284 (final)** |
| 7500 | 1.443 | | | |

114.9 minutes total. Val loss trends downward the entire run, including
through and past iter 4000-4500 — exactly the onset window where the
original SCALE-UP-LOG's Experiment 1 (this identical `n_embd=384,
n_layer=12, n_head=6, VOCAB` shape, bare defaults, cosine+warmup=100
schedule) peaked at val=2.309 and then climbed continuously to
val=5.634 by iter 24000. Here, at iter 4000-4500 val is 1.653 (flat, not
peaking) and continues falling for the entire remaining 10,500 iterations
to a final 1.284 — a completely different trajectory shape at the
identical architecture. **kev-gpt's own real attention class, at
kev-gpt's own actual diverging width/depth, still does not diverge in
this harness.**

**This closes the loop on architecture and width entirely.** Every
axis this whole Phase 2 walk identified as a candidate — loop mechanics,
all three hyperparameters, attention precision, tokenizer, attention
scope, the full real attention class, and now width/depth itself, tested
at the EXACT shape that reliably diverges elsewhere in this log — has
been individually ruled out. What's left is exactly the two structural
axes flagged at the end of Step 5, now the only ones left standing:
1. **LR schedule shape**: this whole Phase 2 walk (Steps 0-6) has used
   `lr=5e-4`, pure linear decay from step 0, no warmup — the reference
   recipe's own schedule. The original SCALE-UP-LOG's own divergence was
   always observed under kev-gpt's actual recipe: cosine decay,
   `warmup=100`, a completely different peak/floor shape and a much
   sharper LR ramp at the very start.
2. **Data framing**: per-story, padded/truncated to a fixed 512-token
   row, `eos`-masked, sampled by full-story index — vs kev-gpt's own
   actual flat continuous-stream/random-contiguous-window sampling
   (`model/data.py`'s `load_split`/`get_batch`), no padding, no
   story-boundary handling, `block_size=128` (vs this harness's 511).

Both remain completely untested by this entire investigation (the
original SCALE-UP-LOG's Attempts 1-22, and this Phase 2 walk's Steps
0-6). Given every other axis is now cleared, one of these two — most
plausibly the LR schedule, since Attempt 6 in the original SCALE-UP-LOG
already showed the divergence is NOT gated on LR continuing to decay
(freezing it at a low floor for 10,000 iterations didn't stop the
climb) but never tested starting from a genuinely different schedule
*shape* — is now the leading candidate for the actual root cause.

### Step 7: swap the LR schedule SHAPE — linear-no-warmup -> kev-gpt's real cosine+warmup=100 — still no divergence

`model/tinystories_hf_repro/ablation_step7_cosinewarmup.py` — Step 6
verbatim, with only the LR schedule function changed: a direct copy of
`model/train.py`'s own `cosine_lr(it, lr, warmup, total, min_lr_frac=0.1)`
in place of the reference recipe's `linear_lr_no_warmup`. Peak
`lr=5e-4` unchanged (only the shape — warmup, then cosine decay to a
0.1x floor — is new); `warmup=100` matches kev-gpt's own `--warmup`
CLI default exactly. Smoke-tested first (200 steps) to confirm the
schedule values match the formula exactly: step 0 logged `lr=5.00e-06`
(`5e-4 * 1/100`, warmup start) and step 200 (smoke test's own `total`)
logged `lr=5.00e-05` (the `0.1x` floor, reached exactly at the end) —
both exact.

```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step7_cosinewarmup.py 15000 /tmp/ablation7_states.jsonl
```

| step | val | lr | | step | val | lr |
|---|---|---|---|---|---|---|
| 0 | 9.811 | 5.0e-06 | | 8000 | 1.422 | 2.54e-04 |
| 500 | 2.676 | 4.99e-04 | | 8500 | 1.408 | 2.30e-04 |
| 1000 | 2.229 | 4.96e-04 | | 9000 | 1.373 | 2.07e-04 |
| 1500 | 1.922 | 4.90e-04 | | 9500 | 1.396 | 1.85e-04 |
| 2000 | 1.792 | 4.82e-04 | | 10000 | 1.385 | 1.64e-04 |
| 2500 | 1.713 | 4.72e-04 | | 10500 | 1.320 | 1.44e-04 |
| 3000 | 1.682 | 4.59e-04 | | 11000 | 1.342 | 1.25e-04 |
| 3500 | 1.679 | 4.45e-04 | | 11500 | 1.325 | 1.09e-04 |
| 4000 | 1.635 | 4.28e-04 | | 12000 | 1.299 | 9.35e-05 |
| 4500 | 1.563 | 4.10e-04 | | 12500 | 1.282 | 8.05e-05 |
| 5000 | 1.513 | 3.90e-04 | | 13000 | 1.273 | 6.97e-05 |
| 5500 | 1.561 | 3.69e-04 | | **13500** | **1.248 (best)** | 6.12e-05 |
| 6000 | 1.488 | 3.47e-04 | | 14000 | 1.351 | 5.50e-05 |
| 6500 | 1.470 | 3.24e-04 | | 14500 | 1.262 | 5.12e-05 |
| 7000 | 1.438 | 3.01e-04 | | **15000** | **1.311 (final)** | 5.0e-05 |
| 7500 | 1.471 | 2.77e-04 | | | | |

115.9 minutes total. Val loss trends downward the entire run, including
through iter 3500-4000 (val 1.679->1.635, LR still at 4.28-4.45e-04 —
*higher* than Step 6's linear-schedule LR at the same iters, 3.83-3.67e-04,
since cosine decay stays closer to peak for longer than linear decay
early on) — no turn, no climb, nothing resembling the original
SCALE-UP-LOG's divergence signature at the identical architecture. One
noisy point at iter 14000 (val jumped to 1.351 from iter 13500's 1.248)
recovered immediately at iter 14500 (1.262), confirming it was noise,
not a trend. **kev-gpt's own real LR schedule (cosine decay,
`warmup=100`), at kev-gpt's own actual diverging shape, with kev-gpt's
own real attention class, still does not diverge.**

**This closes the second of the two remaining leads from Step 6.**
Every axis identified across the entire investigation — loop mechanics,
all three hyperparameters, attention precision, tokenizer, attention
scope, the full real attention class, width/depth at the actual
diverging shape, and now the LR schedule shape itself — has been
individually ruled out as a standalone cause. **Exactly one axis remains
completely untested, in either this Phase 2 walk or the original
SCALE-UP-LOG's Attempts 1-22: the data-framing convention** — per-story,
padded/truncated to a fixed 512-token row with `eos`-masking (every run
in this entire Phase 2 walk, Steps 0-7) vs. kev-gpt's own actual flat
continuous-stream/random-contiguous-window sampling
(`model/data.py`'s `load_split`/`get_batch`, `block_size=128`, no
padding, no per-story boundaries, no masking of any kind). Every other
piece of kev-gpt's real recipe — architecture, width, depth, tokenizer,
optimizer hyperparameters, LR schedule — has now been reproduced
faithfully in this harness without reproducing the divergence. If a
final step running kev-gpt's own real `GPT`/`CausalSelfAttention` class
against kev-gpt's own real flat-stream data pipeline (not this harness's
per-story dataset) still doesn't diverge, the honest conclusion would be
that the original divergence is not reproducible from first principles
in a from-scratch harness at all, and might instead depend on something
about `model/train.py`'s own exact execution environment/state (e.g.
`torch.manual_seed(1337)` combined with this specific GPU's kernel
selection, floating-point non-determinism noted already in the original
log's Conclusion 6) rather than any single identifiable design choice.

### Step 8: swap the DATA FRAMING — per-story padded rows -> kev-gpt's real flat-stream pipeline — still no divergence

`model/tinystories_hf_repro/ablation_step8_flatstream.py` — the last
untested axis. Drops the entire HF-`datasets`-derived pipeline (no
`load_dataset`, no per-story tokenize/pad/truncate, no `<eos>`-masking)
and replaces it with kev-gpt's own real data pipeline, verbatim:
`np.memmap` over `data/word_v16384/train.bin`/`val.bin` (the exact
pre-tokenized corpus the real deployed/diagnostic checkpoints train
on — not regenerated) plus a direct copy of `model/train.py`'s own
`get_batch` (random contiguous `block`-length windows, x/y pre-shifted
by one, no padding, no per-story boundaries, every position a real
training target). `block_size=128`, matching the original SCALE-UP-LOG's
own convention exactly (Steps 0-7 all used `block=511`, inherited from
the reference recipe's `seq_len=512`). Uses kev-gpt's own
`GPT.forward(idx, targets)` directly, unmodified, with its own built-in
loss (no manual `ignore_index` wrapper needed — nothing to mask,
matching kev-gpt's real training exactly). Peak `lr=5e-4` and the
cosine+`warmup=100` schedule carried over unchanged from Step 7 (already
cleared); `weight_decay=0.1`, `beta2=0.95`, `eps=1e-8` default,
`n_embd=384/n_layer=12/n_head=6` carried over from Step 6.

Also drops the gradient-accumulation workaround entirely — `block=128`
needs far less memory than `block=511` did (`64*128*16384*4 bytes ≈
537MB` for the float32 logits cast, vs. the `~6.1GB` that forced
`MICRO_BATCH`/`ACCUM_STEPS` before), a genuine consequence of matching
kev-gpt's own real block size, not a separate choice.

```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step8_flatstream.py 15000 /tmp/ablation8_states.jsonl
```
27.53M params (same as Steps 6-7). Notably faster than every prior
step (31.0 min total) — no HF dataset download/tokenize overhead, no
padding, no gradient accumulation, and a much shorter `block=128` per
forward pass.

| step | val | | step | val |
|---|---|---|---|---|
| 0 | 9.822 | | 8000 | 1.880 |
| 500 | 2.997 | | 8500 | 1.842 |
| 1000 | 2.609 | | 9000 | 1.814 |
| 1500 | 2.392 | | 9500 | 1.805 |
| 2000 | 2.312 | | 10000 | 1.813 |
| 2500 | 2.204 | | 10500 | 1.787 |
| 3000 | 2.146 | | 11000 | 1.767 |
| 3500 | 2.079 | | 11500 | 1.780 |
| 4000 | 2.057 | | 12000 | 1.757 |
| 4500 | 2.026 | | 12500 | 1.745 |
| 5000 | 1.984 | | 13000 | 1.757 |
| 5500 | 1.955 | | **13500** | **1.713 (best)** |
| 6000 | 1.909 | | 14000 | 1.726 |
| 6500 | 1.926 | | 14500 | 1.724 |
| 7000 | 1.868 | | **15000** | **1.728 (final)** |
| 7500 | 1.905 | | | |

(Val-loss magnitudes here aren't directly comparable to Steps 0-7's —
this is a genuinely different task framing: shorter context (`block=128`
vs 511), no end-of-story padding making the tail of short sequences
trivially predictable. Only the *shape* of the trajectory is the
relevant signal.) Val loss trends downward the entire run, including
cleanly through iter 3500-4500 — the historically critical zone, and the
single most faithful reproduction of the original diverging setup
anywhere in this investigation (real attention class, real diverging
shape, real LR schedule, real data pipeline, all simultaneously).
**Still no divergence.**

**This is the decisive result of the whole investigation.** Every axis
identified across BOTH the original SCALE-UP-LOG (Attempts 1-22: LR
magnitude/warmup/decay-length, GPT-2 residual-projection init scaling,
weight-decay/LayerNorm grouping, z-loss at two strengths, SGD vs AdamW,
`eps`/`beta2`) and this entire Phase 2 walk (loop mechanics, all three
individual hyperparameters, attention precision, tokenizer, attention
scope, the full real attention class, width/depth at the exact diverging
shape, LR schedule shape, and now data framing) has now been tested,
individually, and none of them alone reproduces the divergence in a
from-scratch harness built to match kev-gpt's real recipe as closely as
possible at every step. **The honest conclusion: this specific
instability is not reproducible from first principles by varying one
design choice at a time** — which leaves two live possibilities, neither
resolved by this investigation:
1. It requires a *combination* of choices this systematic one-axis-at-a-
   time walk structurally cannot find (every step here changed exactly
   one thing relative to an already-stable baseline; a combination that
   only diverges when TWO OR MORE specific values coincide would never
   surface this way).
2. It's genuinely execution-state-dependent rather than deterministic
   given the recipe alone — the original log's own Conclusion 6 already
   found this directly: the identical deployment-shape recipe, same
   fixed `torch.manual_seed(1337)`, diverged on one run and didn't on
   another, attributed there to ordinary GPU floating-point
   non-determinism (parallel reduction order isn't bit-fixed even under
   a fixed seed). If that's the real mechanism, no amount of harness-
   level ablation can find "the" cause, because there isn't a single
   deterministic one.

**The cheapest remaining check, not yet done**: run `model.train` itself
— the actual production entry point, not this harness — with the exact
flags from the original SCALE-UP-LOG's Experiment 1 or a later Attempt,
on this same machine, right now. If it still diverges today, the
instability is confirmed real and live in the current codebase (meaning
something differs between this Phase 2 harness and the real training
path that all eight steps here failed to isolate). If it doesn't
diverge this time, that's independent evidence for possibility 2 above.

### Sanity check: the real `model.train`, bare defaults — it still diverges today

```bash
python -m model.train --max-iters 15000 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 6 --n-embd 384 --block-size 128 \
  --out data/ckpt_sanity_d384_v16384_bare.pt --states data/states_sanity_d384_v16384_bare.jsonl
```
Bare CLI defaults, nothing overridden: `--lr` defaults to `1e-3`,
`--warmup` to `100`, `--adam-eps` to `1e-8`, `--adam-beta2` to `0.95` —
the literal, never-stabilized recipe, same shape (`n_embd=384,
n_layer=12, n_head=6, VOCAB=16384, block_size=128`) as every Phase 2
Step 6-8 above. `throughput: 9.7 it/s` (`--smoke 5`), ~26 min budget.

| iter | val | logit_max | lnf_norm |
|---|---|---|---|
| 0 | 9.782 | 2.5 | 19.60 |
| 2000 | 2.379 | 24.2 | 31.39 |
| **2500** | **2.371 (best)** | 29.6 | 33.31 |
| 4000 | 2.475 | 30.5 | 38.73 |
| 6000 | 2.722 | 27.5 | 45.85 |
| 9000 | 3.262 | 31.6 | 56.23 |
| 12000 | 4.028 | 42.0 | 64.44 |
| 15000 | 4.731 | 48.0 | 69.38 |

Bottoms at iter 2500, then climbs continuously for the remaining 12,500
iterations to val=4.731, `logit_max`/`lnf_norm` growing unbounded the
whole way — the exact signature the original SCALE-UP-LOG documented
throughout Attempts 1-9. **The real production training path still
diverges today, on this exact hardware.** `model.train`'s own
auto-rollback correctly saved the iter-2500 checkpoint
(`data/ckpt_sanity_d384_v16384_bare.pt`, val=2.371).

Sampled both the rollback checkpoint (iter 2500, val 2.371) and the
diverged end (`data/ckpts/ckpt_15000.pt`, val 4.731) at 3 prompts each
(`python -m model.sample <ckpt> --prompt "..." -n 80 --seed 1`) for a
direct quality comparison. Degradation is real but more subtle than the
val-loss gap alone suggests — mostly odd, slightly incoherent phrasing
("provide her with a special touch," "noticed a find and an apple," an
abrupt unprompted mid-story restart) rather than outright gibberish or
repetition:

> **iter 2500** ("once upon a time"): *"...bobo loved to hop around in
> the grass with his friends. one day, while playing with the rain, he
> saw a big cat running towards him. the cat was scared and ran away..."*
>
> **iter 15000** ("once upon a time"): *"...she had a special light that
> she put in in her room ready for a special makeup. every day after
> that, her parents would provide her with a special touch. surprise was
> to organize their hair in the kitchen..."*

**This confirms the instability is real, live, and reproducible via the
actual codebase — not an artifact of anything in this Phase 2 harness.
Since Step 8 (kev-gpt's real attention class + real data pipeline + real
LR schedule + real hyperparameters, everything this walk could match)
did NOT diverge, something concrete still differs between Step 8 and
this sanity run.** Diffing the two scripts line-by-line against
`model/train.py` directly (not from memory) surfaced two real,
previously-untested candidates:

1. **Precision mechanism.** `model/train.py`'s `amp_dtype()`:
   ```python
   def amp_dtype(device):
       if device == "cuda":
           return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
   ```
   This GPU supports bf16 (confirmed: the smoke test printed
   `amp=torch.bfloat16`). Critically, `GradScaler` construction is
   `enabled=(device == "cuda" and dtype == torch.float16)` — since
   `dtype` is `torch.bfloat16` here, **this evaluates to `enabled=False`.
   The real recipe never uses loss scaling at all**: no dynamic scale
   factor, no gradient-overflow detection, no silently-skipped optimizer
   steps. Every single Phase 2 script (Steps 0-8) used fp16 autocast
   with `GradScaler` **enabled** — inherited unbroken from Step 0's
   faithful reproduction of the reference recipe's `fp16=True` — which
   has a real, qualitatively different safety-valve behavior: it can
   silently skip an update whenever a gradient overflows, then shrink
   the scale and retry. This axis was never varied anywhere in either
   investigation.
2. **Peak LR magnitude.** `--lr` defaults to `1e-3` in `model/train.py`;
   every Phase 2 script used `5e-4` (Step 0's inherited value; Step 7
   only tested schedule *shape*, always at that same 5e-4 peak). This
   sanity run is the first anywhere in this document to use the real
   `1e-3` default.
3. **Seed.** `model/train.py` calls `torch.manual_seed(1337)` once
   before data/model construction; no Phase 2 script sets any seed.
   Weaker candidate — the original log's own Conclusion 6 already showed
   the *same* seed doesn't guarantee reproducibility on this GPU.

Checked and ruled out as real differences: `weight_decay`/`beta2`/`eps`
defaults match exactly; `clip_grad_norm_(..., 1.0)` matches;
`eval_iters` (50 vs this harness's 20) only affects measurement noise,
not training dynamics (eval never feeds back into the loop).

### Step 9: swap the PRECISION MECHANISM — fp16+GradScaler -> kev-gpt's real bf16/no-scaler — still no divergence

`model/tinystories_hf_repro/ablation_step9_bf16noscaler.py` — Step 8
verbatim, with only `dtype`/`GradScaler` changed to a direct copy of
`model/train.py`'s own construction:
```python
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
scaler = torch.amp.GradScaler(enabled=(device == "cuda" and dtype == torch.float16))
```
Peak `lr=5e-4` unchanged (Step 8's value — only the precision mechanism
is the new variable, isolating it from the LR-magnitude candidate).
Smoke-tested first to confirm the mechanism actually engages as
expected: printed `amp dtype: torch.bfloat16` and `GradScaler enabled:
False`, matching the sanity run exactly.

```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step9_bf16noscaler.py 15000 /tmp/ablation9_states.jsonl
```

| step | val | | step | val |
|---|---|---|---|---|
| 0 | 9.778 | | 8000 | 1.875 |
| 500 | 3.031 | | 8500 | 1.867 |
| 1000 | 2.610 | | 9000 | 1.834 |
| 1500 | 2.424 | | 9500 | 1.833 |
| 2000 | 2.275 | | 10000 | 1.803 |
| **2500** | **2.214** | | 10500 | 1.789 |
| 3000 | 2.142 | | 11000 | 1.777 |
| 3500 | 2.095 | | 11500 | 1.754 |
| 4000 | 2.062 | | 12000 | 1.772 |
| 4500 | 2.027 | | 12500 | 1.759 |
| 5000 | 1.967 | | 13000 | 1.729 |
| 5500 | 1.974 | | 13500 | 1.729 |
| 6000 | 1.935 | | 14000 | 1.761 |
| 6500 | 1.933 | | 14500 | 1.740 |
| 7000 | 1.903 | | **15000** | **1.716 (best = final)** |
| 7500 | 1.898 | | | |

30.4 minutes total. At iter 2500 — the exact iteration where the bare-
defaults sanity run bottomed (val=2.371) and turned upward — this run's
val is 2.214 and still falling. The whole trajectory strictly trends
downward with normal noise, ending at its own lowest point at iter
15000. **The precision mechanism alone — bf16 with GradScaler disabled,
matching the real recipe exactly — does not reproduce the divergence.**

**This narrows the two remaining candidates to one.** Precision
mechanism is now cleared the same way every other axis was: peak LR
magnitude (`1e-3` vs the `5e-4` used everywhere in Phase 2, including
this step) is the only concrete, unexamined difference left between
Step 9 and the sanity run that diverged. Seed remains a weaker,
un-eliminated candidate given Conclusion 6's own non-determinism
finding. The natural next step is Step 9 with `lr=1e-3` substituted for
`5e-4` — the last identified variable.

### Step 10: swap PEAK LR — 5e-4 -> kev-gpt's real `--lr` default, 1e-3 — still no divergence

`model/tinystories_hf_repro/ablation_step10_lr1e3.py` — Step 9 verbatim,
`LR = 1e-3` in place of `5e-4`. Schedule shape (cosine, `warmup=100`)
and everything else (bf16/no-scaler, kev-gpt's real attention class,
real data pipeline, real diverging shape, real hyperparameters) held
exactly as Step 9. This is the last concretely identified difference
between this Phase 2 harness and the sanity run that reproduced the
original divergence — every other axis in both the original SCALE-UP-LOG
(Attempts 1-22) and this entire Phase 2 walk (Steps 0-9) has now been
tested and individually cleared. Smoke-tested first: step 0 logged
`lr=1.00e-05` (`1e-3 * 1/100`) and step 100 logged `lr=1.00e-03` (peak,
warmup complete) — exact match to the formula.

```bash
python /home/tparng/kev-gpt/model/tinystories_hf_repro/ablation_step10_lr1e3.py 15000 /tmp/ablation10_states.jsonl
```

| step | val | | step | val |
|---|---|---|---|---|
| 0 | 9.829 | | 8000 | 1.883 |
| 500 | 2.869 | | 8500 | 1.851 |
| 1000 | 2.527 | | 9000 | 1.823 |
| 1500 | 2.387 | | 9500 | 1.826 |
| 2000 | 2.260 | | 10000 | 1.804 |
| **2500** | **2.196** | | 10500 | 1.776 |
| 3000 | 2.114 | | 11000 | 1.769 |
| 3500 | 2.079 | | 11500 | 1.756 |
| 4000 | 2.080 | | 12000 | 1.743 |
| 4500 | 2.017 | | 12500 | 1.728 |
| 5000 | 1.980 | | 13000 | 1.743 |
| 5500 | 1.981 | | 13500 | 1.729 |
| 6000 | 1.970 | | 14000 | 1.728 |
| 6500 | 1.931 | | 14500 | 1.731 |
| 7000 | 1.901 | | **15000** | **1.719 (best = final)** |
| 7500 | 1.891 | | | |

30.5 minutes total. At iter 2500 — now with the LR trajectory itself
matching the sanity run exactly (`lr=9.44e-04` here vs the sanity run's
`9.4e-04` at the same iter, since both use `lr=1e-3`/`warmup=100`/cosine)
— this run's val is 2.196 and still falling, vs. the sanity run's
val=2.371 that turned upward at exactly this point. The whole
trajectory strictly trends downward with normal noise, ending at its
lowest point at iter 15000. **Peak LR magnitude does not reproduce the
divergence either.**

**Every concretely identified difference between this harness and the
real `model.train` recipe has now been tested and exhausted.** The
sanity check confirmed the divergence is real and live in the current
codebase. A line-by-line diff against `model/train.py` surfaced exactly
three candidates: precision mechanism (Step 9, cleared), peak LR
magnitude (Step 10, cleared), and seed. With the first two both cleared
— at kev-gpt's real diverging architecture, real data pipeline, real
LR schedule (shape AND magnitude), real precision mechanism, real
hyperparameters, all simultaneously — **the only unmatched variable left
anywhere in this investigation is `torch.manual_seed(1337)`**, which
`model/train.py` sets once before data/model construction and no Phase
2 script sets at all. This is a weak candidate on its own: the original
SCALE-UP-LOG's own Conclusion 6 already found that the *identical* seed,
on the *identical* recipe, produced one clean run and one diverging run
— meaning seed value alone does not deterministically control the
outcome on this GPU (parallel reduction order in CUDA kernels isn't
bit-fixed even under a fixed seed). Given that, matching the seed in a
future run might still flip an individual outcome, but wouldn't
constitute proof of *the* cause even if it did — it would just be
consistent with what Conclusion 6 already established: this instability
appears to be genuinely stochastic given the recipe, not deterministically
caused by any single design choice this investigation was able to vary
and check.

### Closing quality check: does any of this actually matter for the generated stories?

Ties back to Phase 2's original motivation (does
`SauravP97/tiny-stories-hf` produce genuinely better prose, and does
avoiding kev-gpt's own divergence close that gap) with a direct,
same-methodology comparison across four checkpoints: **A** (kev-gpt,
auto-rollback @ iter 2500, val 2.371 — what the diverging sanity run
actually produced), **B** (kev-gpt, diverged end @ iter 15000, val
4.731), **C** (kev-gpt, Step 10's fully-trained-without-divergence
checkpoint @ iter 15000, val 1.719 — the best of the three, and the
practical payoff of this whole Phase 2 walk), and the **reference**
(`SauravP97/tiny-stories-19M`, the published checkpoint Phase 2's own
sanity sweep already scored at 7/25 flagged, `max_new_tokens=60`, back
near the start of this investigation).

**Sweep 1 (matches the original sanity-sweep methodology): 25 samples
each (5 seeds x 5 prompts), `max_new_tokens=60`.** A: 17/25 flagged.
B: 6/25. C: 9/25. At this length the detector is discriminating, not
saturated, so these numbers are informative on their own — but not in
the direction raw val loss would predict. **A (lowest/"best" val loss)
is the MOST repetitive of the three; B (worst val loss, diverged) is
the LEAST.** The reason is training exposure, not loss quality: A was
caught by auto-rollback at iter 2500 — only 2500 gradient steps — while
B and C both trained the full 15,000 steps. A's flagged samples show
genuine stuck-loop repetition within a single story (e.g. the same
character repeatedly reintroduced as subject: `"the cat said... the
cat took... the cat said... the cat smiled..."`, or literal adjacent-
phrase doubling: `"it was so pretty and pretty... see it again and see
it again"`). B's divergence shows up differently — fewer bigram-
recurrence flags, but outright glitches instead: literal word-stutters
(`"she put it in in her room"`, `"the best apple she had ever had had
tasted"`) and sentences that don't parse (`"she eventually had one
very special meeting the way"`) — consistent with divergence being a
miscalibration/overconfidence problem, not a repetition problem. **C —
what this whole Phase 2 walk was chasing — has the best val loss of the
three (1.719) and sits between A and B on the repetition metric, with
no comparable glitches or stuck-loops in its own flagged samples.**

**Sweep 2 (longer horizon, closer to real usage): 12 samples each (3
seeds x 4 prompts), `max_new_tokens=250`, run against A/B/C AND the
reference model with the identical methodology.** All four saturate the
detector (A: 12/12, B: 11/12, C: 11/12, reference: 12/12) — at this
length every model naturally strings together 2-3 stories per
continuation, and template recurrence across separate stories (`"the
little girl... the little girl"`, `"the end. the end."`) trips the
bigram rule regardless of underlying quality. **The detector stops
being the useful signal here; the qualitative read is what actually
differs.** The reference model has noticeably cleaner grammar/
orthography (proper capitalization, correctly formatted quotation
marks — kev-gpt's checkpoints are lowercase/space-unnormalized by the
keviniser pipeline's own convention, not a training artifact) but its
own distinct repetition tic: nearly every sample pads out with stacked
`"The end. The end. The end."`, a more mechanical, more conspicuous
repetition than anything A/B/C produced. C (kev-gpt, no divergence) is
the closest of the three kev-gpt checkpoints to the reference's
fluency, without that stacked-ending habit, modulo the lowercase
orthography difference.

**Practical conclusion**: avoiding the divergence (Step 10's
achievement, however it was accomplished) produces a real, measurable
quality improvement over the checkpoint kev-gpt's own production
training actually yields today (C's 1.719 val vs. the sanity run's
2.371 rollback / 4.731 diverged-end, and fewer stuck-loop repetitions
than the rollback specifically) — genuinely closing a meaningful chunk
of the gap toward the reference model's own quality, though not
eliminating every difference (orthography is a separate, deliberate
project convention; the reference's own "The end." stacking shows even
a well-regarded published checkpoint has its own repetition
signature at this scale). Sweep scripts (not checked in, regeneratable):
`/tmp/claude-*/scratchpad/sweep_three_models.py` (Sweep 1),
`sweep_three_models_long.py` / `sweep_tinystories_hf_long.py` (Sweep 2).

## Phase 3: deploying Model C's recipe to real Genesys2 hardware

Model C (Step 10, D=384/n_layer=12/n_head=6, the fully-trained-without-
divergence checkpoint Phase 2 produced) is the natural next thing to
put on real silicon — Phase 2's whole point was finding a recipe that
avoids the divergence kev-gpt's own production training hits.

### Model C's actual shape (D=384) is blocked by a hard BRAM overflow

This project's own sizing formula (`GW_BLK = ceil(3D/LANES)·D +
ceil(D/LANES)·D + ceil(4D/LANES)·D + ceil(D/LANES)·4D`, LANES=64;
verified exact against the deployed D=128 shape's own documented
`GW_BLK=3072`) gives:

| | D=128 (deployed) | D=384 (Model C) | ratio |
|---|---|---|---|
| `GW_BLK` (per layer) | 3,072 words | 27,648 words | 9x |
| `GW_HEAD` (vocab GEMV, dominant term) | 32,768 words | 98,304 words | 3x |
| On-chip weight-streaming buffer → RAMB36 | 402/445 (already ~90%) | ≈3x that ≈ 1,206/445 | — |

This isn't a close call needing synthesis to confirm — `fabric/genesys2/
PORT-NOTES.md` already documents a real `place_design` failure at the
*current* D=128 shape (`ERROR: [DRC UTLZ-1] ... requires 522 ... only
445 compatible sites are available`, fixed only by moving the tokenizer
table off on-chip BRAM into DDR3, with the weight-related BRAM itself
already at 402/445). Scaling the dominant `GW_HEAD` term 3x on a design
already at ~90% utilization is a hard, structural no — no RTL
parameter or repacking trick changes this on this specific part
(xc7k325t). **D=384 cannot deploy on this hardware.**

### Retargeting Model C's recipe to D=128 — the deployable shape

Model C's actual D=384 weights can't transfer to D=128 at all (no
weight-level relationship between different widths) — "retargeting"
means training fresh at D=128 with Model C's exact recipe (bare
defaults: `lr=1e-3`, `warmup=100`, `eps=1e-8` default, `beta2=0.95`
default, real bf16/no-scaler, real flat-stream data, real cosine
schedule — every one of these already IS `model/train.py`'s own
default behavior; Model C's whole contribution was proving these
defaults, applied faithfully, don't have to diverge).

```bash
python -m model.train --max-iters 15000 \
  --tokenizer word --vocab-size 16384 --corpus data/TinyStories-train.filtered.txt \
  --data-dir data/word_v16384 --n-layer 12 --n-head 2 --n-embd 128 --block-size 128 \
  --out data/ckpt_stepC_d128_v16384.pt --states data/states_stepC_d128_v16384.jsonl
```
Auto-rollback caught the best point at **iter 8000, val=2.380** — mild
upward drift afterward (final val=2.456 @ iter 15000), matching the
"knife-edge, not catastrophic" pattern this shape's own bare recipe has
shown before (original SCALE-UP-LOG Attempt 13), not D=384's runaway
divergence.

QAT fine-tune (2000 iters, `--qat --init-from`): val **improved** to
**2.318** — QAT helped here, unlike Model C's own D=384 QAT (which
regressed slightly, best val landing at iter 0).

```bash
python -m model.export_fabric data/ckpt_stepC_d128_v16384.qat.pt --out fabric/export_stepC_d128_v16384
# 2176.0 KB packed weights, budget 3072 KB -> FITS
python -m model.validate_goformer --ckpt data/ckpt_stepC_d128_v16384.qat.pt --export fabric/export_stepC_d128_v16384
# worst cosine 1.0000000 -> PASS (gate 0.9999)
```
RTL bit-exact gate, matching this project's own 5-prompt precedent
exactly (`--p 8 --lanes 64 --tmax 128`, `fabric.stage3.run_vec_kv`):

| prompt | plen | ngen | seed | verdict |
|---|---|---|---|---|
| "once upon a time" | 4 | 60 | 7 | `match=True` |
| "the little dog" | 3 | 120 | 1 | `match=True` |
| "one day a girl named lily" | 6 | 120 | 2 | `match=True` |
| "there was a boy who" | 5 | 120 | 3 | `match=True` |
| "the old man walked into" | 5 | 120 | 4 | `match=True` |

**5/5, bit-exact.** Same architecture as the already-deployed shape
(confirmed via the actual synthesized source, not inferred: `.NLAYER
(12)` in `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`) — no new RTL,
no re-synthesis needed for this part. Loaded via the existing UART
boot path (`fabric.genesys2.send_weights`, `fabric.genesys2.
chat_over_uart`) onto the real board: `SEND_WEIGHTS_PASS`,
`KEVGPT_INTERACTIVE_READY`, live chat at **~50 tok/s**.

One real process-hygiene gotcha hit and fixed along the way (already
documented in this project's own history, hit again here): a stale
`openocd` process from a prior session silently held the JTAG/UART
interface, causing `/dev/ttyUSB0` to not enumerate. Fix: kill the stale
process, **reprogram the bitstream fresh** (guaranteed full fabric
reset — a JTAG-only CPU reset alone doesn't clear stale peripheral
state), restart `openocd`, confirm exactly one UART reader, only then
load. A `hw_target` race (`ERROR: [Labtools ...] No devices detected`)
on the first programming attempt self-resolved on retry once
`hw_server` was warm.

### A live-hardware vs. software quality gap, found and investigated

Same checkpoint, same prompts — but the FIRST 25-sample live-hardware
sweep (5 seeds × 5 prompts via `chat_over_uart`) looked meaningfully
rougher than the software samples used throughout Phase 2:

| | flagged/25 | "care" occurrences |
|---|---|---|
| Live hardware (before recalibration) | 8/25 (32%) | 66 (2.64/sample) |
| Software, `temp=0.8/top_k=40` (Phase 2's own convention) | 5/25 (20%) | 5 (0.20/sample) |

Root-caused in two steps. **Step 1**: `chat_over_uart.py` has no
`--seed`/`--temperature`/`--top-k` flags at all — the board's on-chip
Gumbel-max sampler (`fabric/stage3/gumbel.py`) runs on its own free-
running cycle-counter-seeded noise, over the FULL vocab, no top-k
truncation. Re-running the software sweep matched to that config
(`temperature=0.58, top_k=None`) reproduced the flagged-rate gap
exactly (8/25 both) but left the "care" gap almost unchanged (5 vs 66)
— **the overall degenerate-rate gap was a sampling-methodology mismatch
in my own earlier comparison, not a hardware bug; the "care" bias was
real and unexplained by config.**

**Step 2**: measuring this checkpoint's real head-logit std (via the
*correct*, D-agnostic golden reference — `model.goformer_kvq.
IntKVQSequencer(kbits=8, vbits=8, rotate=False, divfree=True)`, the
exact class the passing RTL gates use; NOT `fabric.stage3.seq_ref.
IntSequencer` directly, which still hardcodes `C=256`/`NHEAD`/
`HEAD_DIM` from the old KV260 D=256 shape and throws `IndexError` at
D=128) gave **mean std 3.734** (15 steps / 5 prompts, min 2.15, max
6.65) — ~22% *below* `gumbel.py`'s own calibration reference (std
~4.78 for `TEMP=0.58`). Git history pinpointed why: `TEMP=0.58` was
set in commit `6b49ed2` (2026-08-26 00:25:50), calibrated for the
**NLAYER=8** shape then deployed — but `NLAYER` grew to 12 only
**4h45m later** (`ddce9f7`, 05:10:15, same day), and `gumbel.py` has had
**zero commits since**. The Gumbel noise magnitude has been running on
a stale, shallower-model calibration since the very first NLAYER=12
deployment — not something this session's new checkpoint introduced,
a pre-existing latent condition this investigation happened to be the
first to measure precisely enough to notice.

### Recalibration: measure, edit, re-synthesize, re-flash, re-test

```python
new_TEMP = 0.58 * (3.734 / 4.78)   # ~= 0.4531, rounded to 0.45
```
`fabric/stage3/gumbel.py`: `TEMP = 0.58 -> 0.45`, comment rewritten
with the full measurement/git-history chain above (matching this
file's own prior recalibration comment's rigor, not a bare number
change).

Since the Gumbel LUT is baked into the bitstream at synthesis time
(not reloadable over UART like weights), this required a real rebuild
— but only a ROM-content change, not a structural RTL edit, so the
lighter `reset_run` recipe sufficed (no full project regen):
```bash
# 1) reset (must happen BEFORE staging, not after -- reset_run empties the run dir)
open_project openhwgroup.org_systems_core-v-mini-mcu_1.0.5.xpr
reset_run synth_1; reset_run impl_1

# 2) stage ROMs (plain shell, after the reset)
python -m fabric.genesys2.stage_vivado_roms \
  --npz fabric/export_stepC_d128_v16384/goformer.npz --lanes 64 --p 8 \
  --dest <project root> --dest <project root>/*.runs/synth_1

# 3) build -- NO -jobs flag (this project's own documented gotcha:
#    launch_runs synth_1 -jobs 4 hung for 1h44m+ at near-zero CPU once
#    before, a Vivado `vrs` parallel-dispatcher bug, not a slow build)
open_project openhwgroup.org_systems_core-v-mini-mcu_1.0.5.xpr
launch_runs synth_1;              wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream;  wait_on_run impl_1
```
Result: **0 Errors, 356 Warnings, 0 Critical Warnings, "Bitgen
Completed Successfully"**, ~27 minutes wall clock (well inside this
project's own 20-90 min prior-build range). Reprogrammed, reloaded
firmware, reloaded weights (`SEND_WEIGHTS_PASS` again, same
10,682,368+189,448-byte image — confirms architecture unchanged),
re-ran the identical 25-sample sweep:

| | flagged/25 | "care" | "buster's" (new) |
|---|---|---|---|
| Before (`TEMP=0.58`, stale) | 8/25 (32%) | 66 (2.64/sample) | present, not counted |
| **After (`TEMP=0.45`)** | **3/25 (12%)** | **23 (0.92/sample)** | **39 (1.56/sample)** |
| Software baseline (`temp=0.58/top_k=None`, matched config) | 8/25 | 5 (0.20/sample) | — |

**Real, measurable improvement**: flag rate 32%→12% (better than even
the matched-config software baseline), "care" down ~2.9x. **Not fully
resolved**: a new, milder fixation token ("buster's", stuck as the
repeated subject across several sentences) appeared in its place — the
same *kind* of phenomenon, reduced but not eliminated. This suggests
either a residual precision/sampling difference between real silicon
and the golden-reference simulation that a handful of seeded bit-exact
gates wouldn't catch, or some inherent property of this checkpoint's
own token-probability landscape under full-vocab (no-top-k) sampling
specifically — not yet distinguished. Open for further investigation,
not closed.

## Files touched

- `model/gpt.py`: `GPTConfig.z_loss_coef` (new field, default 0.0/off),
  `GPT.__init__`'s residual-projection init-scaling loop (new, always on),
  `GPT.forward()`'s optional z-loss term.
- `model/train.py`: `--z-loss-coef` CLI flag, `AdamW` param-grouping
  (decay/no_decay split, always on), `estimate_loss()`'s z-loss-neutral
  eval, the new `logit_std`/`logit_max`/`lnf_norm` print at every eval,
  the `layer_grad_norms()` helper plus `--layer-grad-log`/
  `--layer-grad-interval` CLI flags (opt-in, default off), `--lr-decay-iters`
  (decouples the cosine-decay horizon from `--max-iters`, default None =
  same as `--max-iters`, old behavior unchanged), `--optimizer {adamw,sgd}`
  (default `adamw`, unchanged behavior; `sgd` uses `momentum=0.9` with the
  same decay/no_decay param grouping), `--adam-eps` (default `1e-8`,
  torch's own default) and `--adam-beta2` (default `0.95`, this
  codebase's prior default), both wired into `AdamW(...)`.
- New data/checkpoints (all gitignored under `data/`, not checked in —
  regenerate via the commands above): `data/word_big4096/` (VOCAB=4096
  tokenizer + train/val bins), `data/ckpt_word_big4096*.pt` (17 checkpoints,
  one per run above), `data/states_word_big4096*.jsonl` (matching
  per-eval logs), `data/gradlog_word_big4096.jsonl` (Attempt 5's per-layer
  gradient-norm trace), `data/ckpts/ckpt_*.pt` (shared per-eval snapshot
  dir — see the overwrite caveat under Attempt 2). Attempt 12 reuses the
  existing `data/word_stream16/` tokenizer/bins (the deployment shape's
  own prepared data, not regenerated) and adds
  `data/ckpt_word16_epsrecipe.pt` / `data/ckpt_word16_epsrecipe_23k.pt`
  + matching `data/states_word16_epsrecipe*.jsonl` — deliberately
  separate filenames from the real deployed `data/ckpt_word16.pt`, which
  this attempt does not touch or replace. Attempt 13 adds
  `data/ckpt_word16_reproduce.pt` + `data/states_word16_reproduce.jsonl`
  (same deal — separate filename, doesn't touch the real deployed
  checkpoint). Attempt 14 adds `data/ckpt_word16_epsmild1e5.pt` /
  `data/ckpt_word16_epsmild1e4.pt` / `data/ckpt_word16_epsmild3e4.pt` +
  matching `data/states_word16_epsmild{1e5,1e4,3e4}.jsonl`, plus a
  reproducibility-check rerun `data/ckpt_word16_v2.pt` /
  `data/states_word16_v2.jsonl` — all standalone candidates, deliberately
  not promoted over the real deployed `data/ckpt_word16.pt`. Attempts
  15-17 add `data/ckpt_word16_eps{5e4,7e4,1e3,3_5e4,4e4,4_5e4}.pt` +
  matching `states*.jsonl`, plus the Attempt 17 long-schedule reruns
  `data/ckpt_word16_eps{7e4,1e3}_long36k.pt`. Attempt 18 adds
  `data/ckpt_word16_deploy_eps5e4.qat.pt` /
  `data/states_word16_deploy_eps5e4_qat.jsonl` (QAT of the `eps=5e-4`
  winner). Attempt 19 adds `fabric/export_word16_deploy_eps5e4/goformer.npz`
  (bit-exact verified, VOCAB=1900). Attempts 20-22 pivot to VOCAB=16384
  (see [[feedback-use-vocab16384-not-1900]] and
  [[feedback-hw-target-genesys2-ddr3]] — VOCAB=1900 is retired as of
  this point, all further training should target VOCAB=16384 only) and
  add `data/ckpt_word16384_deploy_eps5e4.pt` / `data/ckpt_word16384_eps{3_5e4,4e4,4_5e4}.pt`
  + matching `states*.jsonl`, and `data/ckpt_word16384_eps3_5e4.qat.pt` /
  `data/states_word16384_eps3_5e4_qat.jsonl` — none promoted over the
  real deployed `data/ckpt_word16384.pt` / `.qat.pt`, and none of the
  VOCAB=16384 candidates has been bit-exact verified yet (unlike the
  VOCAB=1900 `eps=5e-4` checkpoint in Attempt 19).
- **Phase 2** adds `model/tinystories_hf_repro/` (new directory,
  standalone scripts, not wired into `model/train.py`'s own CLI):
  `repro_hf_recipe.py` (Step 0), `ablation_step1_framework.py` (Step 1),
  `ablation_step2a_weightdecay.py` / `ablation_step2b_beta2.py` /
  `ablation_step2c_batchsize.py` (Steps 2a-2c), and
  `sweep_tiny_stories_hf.py` (the seeded quality check against the
  published `SauravP97/tiny-stories-19M` checkpoint). All run against
  the user's own `~/tiny-stories-hf/venv` (HF `transformers`/`datasets`/
  `accelerate`), not this repo's own `.venv` — kev-gpt's own
  `requirements.txt` is untouched. Each writes its states JSONL to
  `/tmp/` (not this repo's `data/`, since these runs don't produce a
  kev-gpt-format checkpoint) — treat those as scratch, regenerate via
  the "Reproducing this investigation from scratch" commands above
  rather than expecting them to persist.
- The architecture-axis attn-precision test adds `model/gpt.py`:
  `GPTConfig.attn_fp32` (new field, default `False`) and
  `CausalSelfAttention.forward`'s manual fp32-QK^T branch (only taken
  when `attn_fp32=True`; SDPA path fully unchanged otherwise); and
  `model/train.py`: `--attn-fp32` CLI flag threaded into `GPTConfig`
  (default off). Also adds `data/ckpt_teacher_d384_v16384_fp32attn.pt` +
  `data/states_teacher_d384_v16384_fp32attn.jsonl` — negative result,
  not promoted, kept for reference alongside the other falsified
  Phase-2 attempts.
- Step 3 adds `model/tinystories_hf_repro/ablation_step3_tokenizer.py`
  (reads `data/word_v16384/meta.json` directly, no other kev-gpt-repo
  dependency — runs standalone in `~/tiny-stories-hf/venv` like the rest
  of `tinystories_hf_repro/`). States JSONL at `/tmp/ablation3_states.jsonl`
  (scratch, not persisted) — negative result (no divergence), so nothing
  in `data/` to show for it.
- Step 4 adds `model/tinystories_hf_repro/ablation_step4_globalattn.py`
  (copy of Step 3, one line changed: `attention_types`). States JSONL at
  `/tmp/ablation4_states.jsonl` (scratch) — negative result, nothing in
  `data/`.
- Step 5 adds `model/tinystories_hf_repro/ablation_step5_kevgptattn.py`
  (imports `model.gpt.GPT`/`GPTConfig` directly via `sys.path.insert(0,
  "/home/tparng/kev-gpt")`, same trick `sweep_tiny_stories_hf.py` already
  used for `model.filter_synth_corpus` — no changes to kev-gpt's own
  `model/gpt.py`). States JSONL at `/tmp/ablation5_states.jsonl`
  (scratch) — negative result at this width/depth, nothing in `data/`.
- Step 6 adds `model/tinystories_hf_repro/ablation_step6_fullscale.py`
  (copy of Step 5, `GPTConfig` width/depth changed to `n_embd=384,
  n_layer=12, n_head=6`). States JSONL at `/tmp/ablation6_states.jsonl`
  (scratch) — negative result at kev-gpt's actual diverging shape,
  nothing in `data/`.
- Step 7 adds `model/tinystories_hf_repro/ablation_step7_cosinewarmup.py`
  (copy of Step 6, `linear_lr_no_warmup` replaced with a verbatim copy of
  `model/train.py`'s own `cosine_lr`). States JSONL at
  `/tmp/ablation7_states.jsonl` (scratch) — negative result, nothing in
  `data/`.
- Step 8 adds `model/tinystories_hf_repro/ablation_step8_flatstream.py`
  (drops HF `datasets` entirely; uses `np.memmap` over the existing
  `data/word_v16384/train.bin`/`val.bin` + a verbatim copy of
  `model/train.py`'s own `get_batch`). States JSONL at
  `/tmp/ablation8_states.jsonl` (scratch) — negative result (the
  decisive one), nothing in `data/`.
- Sanity check adds `data/ckpt_sanity_d384_v16384_bare.pt` (auto-rollback,
  iter 2500, val 2.371) + `data/states_sanity_d384_v16384_bare.jsonl`,
  via the real `python -m model.train` CLI (no new script) — **positive
  result**: reproduces the original divergence today. Per-eval snapshots
  landed in the shared `data/ckpts/` dir (see Attempt 2's overwrite
  caveat) — `data/ckpts/ckpt_02500.pt` / `ckpt_15000.pt` used for the
  before/after sample comparison.
- Step 9 adds `model/tinystories_hf_repro/ablation_step9_bf16noscaler.py`
  (copy of Step 8, `dtype`/`GradScaler` construction replaced with a
  verbatim copy of `model/train.py`'s own `amp_dtype()` logic). States
  JSONL at `/tmp/ablation9_states.jsonl` (scratch) — negative result,
  nothing in `data/`.
- Step 10 adds `model/tinystories_hf_repro/ablation_step10_lr1e3.py`
  (copy of Step 9, `LR = 1e-3` in place of `5e-4`). States JSONL at
  `/tmp/ablation10_states.jsonl` (scratch) — negative result (the last
  concretely identified variable, now also cleared), nothing in
  `data/`.
- **Phase 3** adds `data/ckpt_stepC_d128_v16384.pt` (D=128 retarget, FP,
  iter 8000, val 2.380) + `data/states_stepC_d128_v16384.jsonl`;
  `data/ckpt_stepC_d128_v16384.qat.pt` (QAT, val 2.318) + `data/
  states_stepC_d128_v16384_qat.jsonl`; `fabric/export_stepC_d128_v16384/`
  (both the `model.export_fabric` linears-only export and the
  `model.goformer_full` `goformer.npz` used by the RTL gates) — this
  checkpoint IS now the one actually flashed on the real board, real
  quality caveats and all. Also `data/ckpt_stepC_d384_v16384_fp.pt` /
  `.qat.pt` (Model C's own D=384 QAT, val 1.845 — never deployed, kept
  for reference alongside the BRAM-overflow finding). Edits `fabric/
  stage3/gumbel.py` (`TEMP` 0.58 -> 0.45, comment rewritten with the
  full measurement chain) — this is a real, permanent, committed source
  change, requiring a bitstream rebuild to take effect (already done and
  reflected in the real board's current state as of this log entry).

## Verifying the "Closing quality check" claim with a fresh seeded sweep

The Phase 2 closing quality check's "equal coherence" conclusion rested
on numbers recorded at the time, not on anything reproducible from a
checked-in script (Sweep 1/2's own scripts were noted as scratch,
`/tmp/claude-*/scratchpad/sweep_three_models*.py`, long gone). This
entry reruns that comparison from scratch — checkpoint C
(`data/ckpt_stepC_d384_v16384_fp.pt`, iter 15000, D=384/n_layer=12/
n_head=6, the ten-step ablation walk's payoff checkpoint) vs
`SauravP97/tiny-stories-19M` — with the same methodology (same
detector, same sweep shapes) and, this time, both the sweep script and
its full sample-by-sample output committed rather than left as scratch.

One incidental finding along the way: `data/ckpt_stepC_d384_v16384_fp.pt`
is NOT a `model/train.py`-format checkpoint (`{"model", "cfg", "meta",
"iter", "val"}`) — it only carries `{"model", "iter"}`, since it was
produced and later re-saved by the standalone
`ablation_step10_lr1e3.py` script, not `model/train.py`'s own save path.
Loading it for generation requires reconstructing `GPTConfig` (`block_size=128,
n_layer=12, n_head=6, n_embd=384`, vocab from `data/word_v16384/meta.json`)
by hand rather than trusting `ck["cfg"]`/`ck["meta"]` — `model/sample.py`'s
own loader would `KeyError` on this file directly.

**Sweep 1** (5 seeds x 5 prompts, `max_new_tokens=60`, each model at its
own established sampling convention — C: temp=0.8/top_k=40;
reference: temp=0.7/top_k=50): **C 6/25 flagged (24%), reference 7/25
(28%)** — within noise of the original run's numbers, same conclusion.
**Sweep 2** (3 seeds x 4 prompts, `max_new_tokens=250`): **C 12/12,
reference 12/12** — both saturate the detector identically, again
matching the original finding.

Quantifying what the original investigation only described
qualitatively: at the 250-token horizon, the reference stacks
`"The end."` **4.4x more often** than C (0.75 vs 0.17 occurrences/
sample) — a real, distinct repetition signature the reference model
carries independent of anything this investigation controls — while C's
own long-horizon failure mode is concatenating fresh, unrelated stories
(`"once upon a time"` reappearing mid-continuation, 1.50 vs 0.25
occurrences/sample) rather than looping a single one. Neither pattern
is a clear quality win for either side; the objective detector scores
both within a few points of each other at both horizons. **The "equal
coherence" claim holds** on a fresh, independently-generated sample set.

```bash
python model/tinystories_hf_repro/quality_sweep_C_vs_reference.py
python model/tinystories_hf_repro/build_repetition_audit_report.py
```
Adds (all committed, not scratch, at the user's request):
`model/tinystories_hf_repro/quality_sweep_C_vs_reference.py` (the sweep,
run under this repo's own `.venv` — it already carries `transformers`,
so this one doesn't need the separate `~/tiny-stories-hf/venv` the
earlier Phase 2 scripts required); `quality_sweep_results.json` (all 74
generated samples + detector verdicts, the committed snapshot from
today's run); `build_repetition_audit_report.py` (renders the JSON into
an HTML report); `repetition_audit_report.html` (every sample from both
sweeps, paired by matching seed/prompt-theme across the two models,
flagged span highlighted inline in the story text — the full
sample-by-sample evidence behind the two flagged/N numbers above).

## Real Genesys2 board vs. software: a fixation-word pattern the detector can't see

The checkpoint C comparison above used `data/ckpt_stepC_d384_v16384_fp.pt`
— **D=384, never deployed** (hard BRAM overflow, Phase 3's own finding).
The board runs a separately-trained D=128 retarget
(`data/ckpt_stepC_d128_v16384.pt`/`.qat.pt`), so the "equal coherence"
result above was never actually verified for the checkpoint that's
really on real hardware. Two follow-ups closed that gap.

**Real hardware vs. software, same weights.** Captured 25 real-board
samples (5 draws × 5 prompts, `fabric.genesys2.chat_over_uart`, no
controllable seed — on-chip free-running counter) against checkpoint
C's actual deployed weights, and a matched-config software rerun
(`data/ckpt_stepC_d128_v16384.pt`, the pre-QAT FP proxy — the plain
`GPT` class can't load the QAT checkpoint's Brevitas `QuantLinear`
state dict — at the board's calibrated `temp=0.45, top_k=None`).
**Real board: 3/25 flagged. Software: 11/25 flagged** — the raw
detector score makes hardware look *cleaner*. Reading the samples shows
why that's backwards: the board substitutes non-sequitur words as story
subjects — **"cardinal" (10×), "buster's"/"buster" (12×), "bustled"
(3×), "cube" (4×), "contains" (3×), "spidey's" (1×)** — 33 occurrences
across 25 hardware samples, **zero** across 25 software samples. Single-
word insertions don't trip a bigram-recurrence detector, so the flagged-
rate numbers alone tell a backwards story. This independently
reproduces and sharpens Phase 3's own "buster's" fixation-token finding
with five more specific terms. Full sample-by-sample evidence,
including a fixation-term occurrence count per model, committed as
`model/tinystories_hf_repro/hw_vs_sw_sweep_results.json` /
`hw_vs_sw_report.html` (built by an inline, not-yet-modularized script;
regenerate via the same `chat_over_uart`/`send_weights` real-hardware
sequence documented in `fabric/genesys2/PORT-NOTES.md`'s runbook).

**Root-cause investigation, in progress.** Looked up the fixation
words' actual vocab IDs in `data/word_v16384/meta.json`: `buster`=2048,
`buster's`=2049, `bustled`=2051 — tightly clustered right at 2048=2^11,
an immediately suspicious power-of-two boundary matching this exact
codebase's own repeated bug class (register widths silently floored at
11 bits across several past VOCAB-scale-up fixes, e.g. `cb39b8a`'s "two
more hardcoded 1024s", and the sibling `hf19m-genesys2` branch's
`gv_rdaddr` `reg[10:0]` bug). Checked and **ruled out**, with evidence:
- **The checkpoint/weights**: golden-reference `IntKVQSequencer` (same
  INT4/INT8-quantized `fabric/export_stepC_d128_v16384/goformer.npz`
  weights the board runs) ranks all 8 suspicious token IDs at
  logit-rank ~3,000-16,000/16,384 across 5 test prompts — nowhere near
  competitive. The trained weights are fine.
- **The Gumbel-max argmax comparison tree** in `sequencer_vec.sv`
  (`best_idx`/`wm_idx`/`pi0-3`, all `[VIDXW-1:0]` = 14 bits) — correctly
  parameterized for VOCAB=16384, no truncation found on static read.
- **The embedding-feedback addressing** (`tok_id` port, `EMBROWW`,
  `emb_row_w`) — also correctly `$clog2(VOCAB)`-parameterized throughout.
- **The argmax row-counter pipeline** (`ar`/`ard`/`amd`/`ad1`, all
  `$clog2(ARROWS+1)` = 12 bits) — correctly sized for the ARROWS=2048
  sentinel, doesn't wrap.

**Strongest remaining lead — a real verification gap, not yet a
confirmed bug.** Phase 3's "5/5 bit-exact" claim used
`fabric.stage3.run_vec_kv`, which only ever exercises the
**fully-resident** weight configuration (no `WEIGHT_STREAM_PER_LAYER`
define anywhere in that script). The real board almost certainly runs
**per-layer DDR3 streaming** (`WEIGHT_STREAM_PER_LAYER=1` — the whole
reason that feature exists is that this model doesn't fit fully
resident on-chip). A testbench for that exact path exists
(`fabric/stage3/tb/tb_seq_vec_kv_stream.sv`), but its own header says
outright: *"no run_*.py harness yet"* — it has never been wrapped into
this project's reusable gate tooling, and there's no evidence it was
ever run against the actual deployed checkpoint at VOCAB=16384. This is
the same gap-shaped pattern behind the last two real-hardware-only bugs
in this project's history (DDR3 address collision, Gumbel-precompute
hang) — both invisible to the faster fully-resident gate.

Tried and **abandoned as a dead end**: a wider fully-resident-mode seed
sweep (8 fresh seeds, 40-token generation each, via `run_vec_kv.py`) to
check whether the pattern reproduces without streaming at all. Every
single seed hit a 200s per-seed timeout with zero completions —
`iverilog`/`vvp` simulation at this VOCAB/NLAYER size is too slow for
this approach to produce a signal in a reasonable window, independent
of whether the underlying hypothesis is right or wrong.

**Not yet done**: building `run_vec_kv_stream.py` (wrapping
`tb_seq_vec_kv_stream.sv`, pulling in `weight_loader_ddr.sv`/
`mig_behav_model.sv`/`mig_read_engine.sv`/`sync_fifo.sv`) and running it
against the real deployed checkpoint across many seeds to hunt for a
streaming-vs-fully-resident divergence. Next step.

**Incidental finding along the way**: confirmed
`data/ckpt_stepC_d128_v16384.pt` is a genuinely separate, fresh
training run at D=128 (not D=384 weights misapplied) — its own saved
`cfg` says `n_embd=128`, and `tok_emb.weight` is shaped `(16384, 128)`;
a D=384 checkpoint loaded into a D=128-configured model would raise a
hard shape-mismatch error in `load_state_dict`, not silently run. Also
reran the "equal coherence" quality check specifically at D=128 (same
methodology, same reference samples reused): **D=128 6/25 flagged (60
tok) / 12/12 (250 tok) — statistically indistinguishable from D=384's
own 6/25 / 12/12**, closing the gap noted above. Software-side samples
of this exact checkpoint, across both quality sweeps (50 samples
total), never produced any of the fixation words — reinforcing that the
board-only pattern isn't inherited from a weaker checkpoint. Committed
as `model/tinystories_hf_repro/quality_sweep_d128_results.json` +
`build_retarget_audit_report.py` + `retarget_audit_report.html` (same
paired story-by-story format as `repetition_audit_report.html`, D=128
vs. reference this time).

## Fixation-word root cause, closed: a real-silicon-only numerical near-tie, not a fixable bug

Continuing the investigation above. Every simulation-reachable mechanism
checked out clean — which turned out to be the actual finding, not a
dead end:

- **The exact real seed-derivation algorithm, tested for real.** Traced
  `kevgpt_interactive/main.c`'s `chat_turn()`: `seed_state =
  (seed_state*1664525+1013904223) ^ kevgpt_cycles(dev)`, boot value
  `0x9e3779b9u`. Along the way, found `kevgpt_cycles()` is **not** the
  free-running counter its own comments describe — it's
  `xheep_kevgpt_peripheral.sv`'s `cycles_run`, which **resets to 0 on
  every `go` pulse** and just measures one step's mostly-fixed
  architecture-timing duration (confirmed via real simulation: per-pass
  cycle counts climb in a near-perfectly-regular +72/pass pattern, not
  randomly). Ran the *exact* real formula through the Python golden
  reference (`gumbel.GumbelRng`, the same algorithm the RTL implements)
  across 25 simulated turns from the real boot value: well-distributed
  seeds, **zero** fixation-word hits. The low-entropy cycle counter
  turned out not to be the story either — the LCG's own multiplicative
  step provides real mixing on its own.
- **Built real firmware instrumentation** (see "Files touched" below)
  to capture the *actual* seed a live hardware run uses, not an
  approximation. Rebuilt, reloaded, resent weights, ran a live prompt.
- **Direct hit, first draw**: real hardware, prompt "once upon a time",
  produced a reply ending in "...a cube" — one of the fixation words —
  with `KEVGPT_DEBUG_SEED,0x42da8a1f` printed alongside it.
- **Fed that exact real seed into the Python golden reference.** Output
  matches hardware token-for-token for 22 tokens, then diverges at
  generated-token 23: golden says "she **went** to the park", hardware
  said "she **saw** a big, shiny rock". (The eventual "cube" downstream
  is just wherever the now-diverged trajectory organically ends up —
  not evidence these specific words are special "attractors".)
- **Inspected the golden reference's own noise-perturbed logits at that
  exact step**: "went" wins (rank 1), "saw" is rank 2, separated by
  **415,276,205 vs 414,740,811 — a 0.13% margin**. Everything else
  trails by much more. Not a gross corruption; a razor-thin near-tie.
- **The decisive test**: fed that same real seed (`0x42da8a1f` =
  `1121618463`) through the real per-layer-DDR3-streaming RTL
  simulation (the exact path real hardware runs), same checkpoint, same
  prompt, 23 generated tokens. Result: token 23 = **15808 = "went"** —
  matching the Python golden reference exactly, **not** what real
  hardware produced.

**Conclusion.** The checkpoint is clean (established earlier). The RTL
is bit-exact to the golden reference — fully-resident and streaming,
greedy and sampled, arbitrary seeds *and* this exact real captured
seed. The tokenizer table is correct (checked byte-for-byte against the
canonical `meta.json`). The firmware's seed-derivation algorithm is
correctly implemented and was captured directly from a live run, not
guessed. Every one of those now has real-hardware-derived evidence
behind it, not just simulation. What's left, after eliminating
everything upstream, is a genuine **physical-silicon-only numerical
discrepancy** at near-tied decision boundaries — most consistent with a
marginal DDR3 timing/signal-integrity margin issue flipping a handful
of LSBs somewhere across the 12-layer computation, just enough to swing
a 0.13%-margin comparison, but not to produce anything grosser. This
is not an RTL bug, not a firmware bug, not a checkpoint issue, and not
reproducible in any simulation this project has — closing this properly
would need real electrical debugging (scope/ILA on the physical board),
not more RTL or software work. Practical mitigation is what's already
been done (the Gumbel TEMP recalibration earlier in this log, which
measurably cut the flagged rate 32%→12%) or DDR3 timing-margin
tightening in the Vivado build, not a code fix.

**Files touched**: `kevgpt-genesys2-soc`'s
`sw/applications/kevgpt_interactive/main.c` (separate repo,
`~/RVchatbot/kevgpt-genesys2-soc`) — `chat_turn()`'s seed-write now also
prints `KEVGPT_DEBUG_SEED,0x%08x,cyc=0x%08x` each turn, so any future
flagged real-hardware reply can be replayed offline through the Python
golden reference at the exact seed that produced it. No RTL or
kev-gpt-repo changes — this was a pure diagnostic addition.
