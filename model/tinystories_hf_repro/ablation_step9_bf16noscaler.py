"""Ablation step 9: swap the PRECISION MECHANISM only -- fp16 autocast +
enabled GradScaler (dynamic loss scaling, silently skips an optimizer step
whenever a gradient overflows and shrinks the scale factor, then retries --
used unchanged in every Phase 2 script, Steps 0-8, inherited from the
reference recipe's own `fp16=True`) -> kev-gpt's own ACTUAL amp_dtype()/
GradScaler construction from model/train.py: bf16 autocast (this GPU
supports it) with GradScaler CONSTRUCTED BUT DISABLED
(`enabled=(device=="cuda" and dtype==torch.float16)` evaluates False for
bf16), so scale()/step()/update() are no-ops and every optimizer step
applies directly, unconditionally, with no overflow detection or skipping
at all. Peak LR stays 5e-4 (Step 8's own value) -- ONLY the precision
mechanism changes, keeping this a clean single-variable step.

Found by diffing Step 8 against a freshly-run real `model.train` sanity
check (bare CLI defaults, same D=384/n_layer=12/VOCAB=16384 shape) that DID
reproduce the original SCALE-UP-LOG's divergence signature today (bottomed
val=2.371 @ iter 2500, climbed to val=4.731 @ iter 15000) -- proving the
instability is real and live in the current codebase, and that SOMETHING
differs between Step 8 (never diverged) and the real recipe. This is the
strongest of the two candidates that diff surfaced (the other, peak LR
1e-3 vs 5e-4, is untested and reserved as the next step if this one comes
back negative): bf16's wider exponent range needs no loss scaling at all,
so it never has GradScaler's implicit "skip this step, it overflowed"
safety valve -- fp16+GradScaler could plausibly be silently smoothing over
exactly the kind of large-gradient event that, applied directly and
unconditionally under bf16, compounds into the observed divergence.

If THIS run diverges, the precision/scaling mechanism itself is implicated
-- a genuinely novel finding neither the original SCALE-UP-LOG's Attempts
1-22 nor this Phase 2 walk's Steps 0-8 ever considered (all of Phase 2 used
fp16+GradScaler throughout; the original log's own Attempts never varied
precision mechanism as an axis at all, since model/train.py always uses
amp_dtype()'s auto-selection). If it still doesn't diverge, peak LR
magnitude (1e-3 vs 5e-4) becomes the last concrete candidate from the diff."""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/tparng/kev-gpt")
from model.gpt import GPT, GPTConfig  # noqa: E402

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
OUT_STATES = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ablation1_states.jsonl"
CKPT = sys.argv[3] if len(sys.argv) > 3 else OUT_STATES.replace(".jsonl", "_ckpt.pt")
BATCH = 64
BLOCK = 128
LR = 5e-4              # same peak as Step 8 -- precision mechanism is the only new variable
WARMUP = 100
MIN_LR_FRAC = 0.1
WEIGHT_DECAY = 0.1
BETA2 = 0.95
EPS = 1e-8

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "/home/tparng/kev-gpt/data/word_v16384"
with open(os.path.join(DATA_DIR, "meta.json")) as f:
    _meta = json.load(f)
VOCAB_SIZE = _meta["vocab_size"]
print(f"kev-gpt word vocab: size={VOCAB_SIZE}  train_tokens={_meta['train_tokens']:,}  "
      f"val_tokens={_meta['val_tokens']:,}")

train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
print(f"train tokens={len(train_data):,}  val tokens={len(val_data):,}")

cfg = GPTConfig(
    block_size=BLOCK,
    vocab_size=VOCAB_SIZE,
    n_layer=12,
    n_head=6,
    n_embd=384,
    dropout=0.0,
)
model = GPT(cfg).to(device)
print(f"model parameters: {model.num_params() / 1_000_000:.2f}M")

decay, no_decay = [], []
for p in model.parameters():
    if not p.requires_grad:
        continue
    (decay if p.dim() >= 2 else no_decay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": WEIGHT_DECAY},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=LR, betas=(0.9, BETA2), eps=EPS)

# Verbatim: model/train.py's own amp_dtype() + GradScaler construction.
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
print(f"amp dtype: {dtype}")
ctx = torch.autocast(device_type=device, dtype=dtype)
scaler = torch.amp.GradScaler(enabled=(device == "cuda" and dtype == torch.float16))
print(f"GradScaler enabled: {scaler.is_enabled()}")


def get_batch(data, block, batch, device):
    ix = torch.randint(len(data) - block - 1, (batch,))
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_val_loss(model, data, block, batch, device, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, block, batch, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def cosine_lr(it, lr, warmup, total, min_lr_frac=MIN_LR_FRAC):
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
        val_loss = estimate_val_loss(model, val_data, BLOCK, BATCH, device)
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

    print(f"  step {step} fetching...", flush=True)
    x, y = get_batch(train_data, BLOCK, BATCH, device)
    print(f"  step {step} forward...", flush=True)
    with ctx:
        _, loss = model(x, y)
    print(f"  step {step} backward...", flush=True)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

print("done.")
