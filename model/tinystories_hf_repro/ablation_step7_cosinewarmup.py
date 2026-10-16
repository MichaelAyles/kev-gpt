"""Ablation step 7: swap the LR SCHEDULE SHAPE only -- linear decay from
step 0, no warmup (the reference recipe's own schedule, used unchanged in
every Phase 2 step 0-6) -> kev-gpt's own actual cosine_lr (model/train.py):
linear warmup for `warmup=100` iters, then cosine decay down to
`min_lr_frac=0.1` of peak by `total` iters, floor held after that. Peak LR
stays 5e-4 (Step 0-6's own value) -- ONLY the schedule shape changes, so
this stays a clean single-variable step on top of Step 6.

Step 6 (kev-gpt's real attention class, at kev-gpt's own actual diverging
width/depth n_embd=384/n_layer=12/n_head=6, everything else at kev-gpt's own
hyperparameter values) did NOT diverge either -- closing the loop on every
architecture/width/hyperparameter/tokenizer axis this whole Phase 2 walk
identified. Two structural axes were left, both never tested anywhere in
this investigation (neither the original SCALE-UP-LOG's Attempts 1-22 nor
Phase 2's Steps 0-6): the LR schedule shape, and the data-framing convention.
This step tests the LR schedule directly -- the stronger of the two leads,
since the original SCALE-UP-LOG's Attempt 6 already showed the divergence
isn't gated on LR continuing to decay (freezing it at a low floor for 10,000
iterations didn't stop the climb), but never tested a genuinely different
schedule SHAPE, only different lengths/floors of the same linear-warmup+
cosine-decay family it always used. If this run (identical to Step 6, only
the schedule swapped to kev-gpt's own real cosine_lr/warmup=100) diverges,
the LR schedule -- specifically something about starting near-zero and
ramping up over 100 steps into a cosine decay, vs. starting at full peak
and decaying linearly with no warmup at all -- is implicated. If it still
doesn't diverge, the data-framing axis (kev-gpt's own flat continuous-
stream/random-window sampling, block_size=128, vs this harness's per-story
padded-to-512 rows) becomes the last remaining untested candidate.

Held at kev-gpt's own convention: head_dim=64 (n_head=6, n_embd=384),
n_layer=12 -- Step 6's exact architecture, unchanged."""
import json
import os
import re
import sys
import time

import math
import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, "/home/tparng/kev-gpt")
from model.gpt import GPT, GPTConfig  # noqa: E402  (kev-gpt's own, real, unmodified class)

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
OUT_STATES = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ablation1_states.jsonl"
CKPT = sys.argv[3] if len(sys.argv) > 3 else OUT_STATES.replace(".jsonl", "_ckpt.pt")
BATCH = 64
MICRO_BATCH = 8  # gradient accumulation, same VRAM-ceiling workaround as Steps 2c-6
ACCUM_STEPS = BATCH // MICRO_BATCH
SEQ_LEN = 512
BLOCK = SEQ_LEN - 1  # x/y are pre-shifted, so each is BLOCK long, not SEQ_LEN
LR = 5e-4
WARMUP = 100          # kev-gpt's own --warmup default (model/train.py)
MIN_LR_FRAC = 0.1     # kev-gpt's own cosine_lr default
WEIGHT_DECAY = 0.1
BETA2 = 0.95
EPS = 1e-8

device = "cuda" if torch.cuda.is_available() else "cpu"

WORD_META_PATH = "/home/tparng/kev-gpt/data/word_v16384/meta.json"
with open(WORD_META_PATH) as f:
    _meta = json.load(f)
STOI = _meta["stoi"]
VOCAB_SIZE = _meta["vocab_size"]
UNK_ID = STOI["<unk>"]
EOS_ID = STOI["<eos>"]
TOKEN_RE = re.compile(r"[a-z']+|[.,!?;:\"-]")
print(f"kev-gpt word vocab: size={VOCAB_SIZE}  unk_id={UNK_ID}  eos_id={EOS_ID}")

print("loading dataset (cached)...")
dataset = load_dataset("roneneldan/TinyStories")

def tokenize_function(examples):
    out_ids = []
    for text in examples["text"]:
        toks = TOKEN_RE.findall(text.lower())
        ids = [STOI.get(t, UNK_ID) for t in toks]
        ids.append(EOS_ID)
        if len(ids) > SEQ_LEN:
            ids = ids[:SEQ_LEN]
        else:
            ids = ids + [EOS_ID] * (SEQ_LEN - len(ids))
        out_ids.append(ids)
    return {"input_ids": out_ids}

print("tokenizing (cached)...")
tokenized = dataset.map(tokenize_function, batched=True, num_proc=8)
tokenized["train"].set_format("torch", columns=["input_ids"])
tokenized["validation"].set_format("torch", columns=["input_ids"])
train_ds = tokenized["train"]
val_ds = tokenized["validation"].select(range(500))
print(f"train rows={len(train_ds)}  val rows={len(val_ds)}")

cfg = GPTConfig(
    block_size=BLOCK,
    vocab_size=VOCAB_SIZE,
    n_layer=12,
    n_head=6,      # head_dim=384/6=64, kev-gpt's own established convention
    n_embd=384,
    dropout=0.0,
    # bias, attn_fp32, z_loss_coef all left at GPTConfig's own defaults
    # (False, False, 0.0) -- this is kev-gpt's real, un-diagnostic-flagged path.
)
model = GPT(cfg).to(device)
print(f"model parameters: {model.num_params() / 1_000_000:.2f}M")

# Identical decay/no_decay param-grouping convention this whole investigation
# has used throughout (matches model/train.py's own optimizer construction).
decay, no_decay = [], []
for p in model.parameters():
    if not p.requires_grad:
        continue
    (decay if p.dim() >= 2 else no_decay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": WEIGHT_DECAY},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=LR, betas=(0.9, BETA2), eps=EPS)

dtype = torch.float16
ctx = torch.autocast(device_type=device, dtype=dtype)
scaler = torch.amp.GradScaler(enabled=(device == "cuda" and dtype == torch.float16))


def get_batch(ds, batch, device):
    n = len(ds)
    idx = torch.randint(0, n, (batch,)).tolist()
    rows = ds[idx]["input_ids"].to(device)  # [batch, SEQ_LEN=512]
    x = rows[:, :-1]                         # [batch, BLOCK], kev-gpt's own pre-shift
    y = rows[:, 1:].clone()                  # [batch, BLOCK], offset by one position
    y[y == EOS_ID] = -100                    # mask every eos/pad position, matches Steps 0-6
    return x, y


def compute_loss(model, x, y):
    logits, _ = model(x)  # targets=None -> kev-gpt's own forward skips its (unmasked) loss
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)


@torch.no_grad()
def estimate_val_loss(model, val_ds, batch, device, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_ds, MICRO_BATCH, device)
        with ctx:
            loss = compute_loss(model, x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def cosine_lr(it, lr, warmup, total, min_lr_frac=MIN_LR_FRAC):
    # Verbatim copy of model/train.py's own cosine_lr -- kev-gpt's real
    # schedule, not an approximation of it.
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > total:
        return lr * min_lr_frac
    ratio = (it - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return lr * (min_lr_frac + (1 - min_lr_frac) * coeff)


start_step = 0
elapsed_offset = 0.0
if os.path.exists(CKPT):
    print(f"resuming from {CKPT} ...")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    start_step = ck["step"] + 1
    elapsed_offset = ck["minutes"] * 60
    print(f"resumed at step {start_step}, {elapsed_offset/60:.1f} min already elapsed")
else:
    open(OUT_STATES, "w").close()

model.train()
t0 = time.perf_counter() - elapsed_offset
for step in range(start_step, MAX_STEPS + 1):
    lr = cosine_lr(step, LR, WARMUP, MAX_STEPS)
    for g in opt.param_groups:
        g["lr"] = lr

    if step % 500 == 0 or step == MAX_STEPS:
        val_loss = estimate_val_loss(model, val_ds, BATCH, device)
        elapsed = time.perf_counter() - t0
        print(f"step {step:>6} | lr {lr:.2e} | val {val_loss:.4f} | {elapsed/60:.1f} min")
        with open(OUT_STATES, "a") as f:
            f.write(json.dumps({"step": step, "lr": lr, "val_loss": val_loss, "minutes": elapsed / 60}) + "\n")
        torch.save({
            "step": step, "minutes": elapsed / 60,
            "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
        }, CKPT)
        if step == MAX_STEPS:
            break

    opt.zero_grad(set_to_none=True)
    for micro in range(ACCUM_STEPS):
        print(f"  step {step} micro {micro}/{ACCUM_STEPS} fetching...", flush=True)
        x, y = get_batch(train_ds, MICRO_BATCH, device)
        print(f"  step {step} micro {micro}/{ACCUM_STEPS} forward...", flush=True)
        with ctx:
            loss = compute_loss(model, x, y) / ACCUM_STEPS
        print(f"  step {step} micro {micro}/{ACCUM_STEPS} backward...", flush=True)
        scaler.scale(loss).backward()
    print(f"  step {step} optimizer step...", flush=True)
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

print("done.")
