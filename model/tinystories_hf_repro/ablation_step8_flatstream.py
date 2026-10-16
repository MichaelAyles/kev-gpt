"""Ablation step 8: swap the DATA FRAMING -- HF-dataset-derived, per-story
rows padded/truncated to a fixed 512-token length with <eos>-masking (every
run in this entire Phase 2 walk, Steps 0-7) -> kev-gpt's own ACTUAL flat
continuous-stream/random-contiguous-window sampling: `model/data.py`'s
`load_split` (a raw `np.memmap` over `train.bin`/`val.bin`, the exact
pre-tokenized VOCAB=16384 word-level corpus the real deployed/diagnostic
checkpoints train on) plus `model/train.py`'s own `get_batch` (random
contiguous `block`-length windows, x/y pre-shifted by one, no padding, no
per-story boundaries, no masking of any kind -- every position is a real
training target). `block_size=128`, matching the original SCALE-UP-LOG's
own Experiment-1-onward convention exactly (this harness's Steps 0-7 all
used `block=511`, inherited from the reference recipe's `seq_len=512`,
never kev-gpt's own actual `block_size=128`).

Steps 0-7 have now individually cleared every other axis identified across
this whole investigation (loop mechanics, all three hyperparameters,
attention precision, tokenizer, attention scope, the full real attention
class, width/depth at kev-gpt's exact diverging shape, and the LR schedule
shape) -- none of them alone reproduces the divergence the original
SCALE-UP-LOG's Experiment 1 onward showed. Data framing is the one
remaining completely untested axis. If THIS run (kev-gpt's real attention
class + kev-gpt's real data pipeline + kev-gpt's real LR schedule, all
together, at kev-gpt's real diverging shape) still doesn't diverge, nothing
individually identifiable across either investigation reproduces it, and the
honest conclusion becomes that the original divergence may depend on
execution-state factors (GPU floating-point non-determinism under a fixed
seed, as the original log's own Conclusion 6 already found -- one run of
the IDENTICAL deployment-shape recipe diverged and another didn't) rather
than any single design choice.

Also drops the gradient-accumulation workaround entirely: block=128 (vs
this harness's earlier block=511) needs far less memory for the loss's
float32 logits cast (batch=64 * block=128 * vocab=16384 * 4 bytes =~
537MB, vs the ~6.1GB that forced MICRO_BATCH/ACCUM_STEPS at block=511) --
a genuine, expected consequence of matching kev-gpt's own real block size,
not a separate choice. Uses kev-gpt's own GPT.forward(idx, targets)
directly, unmodified and with its own built-in (unmasked, since there's
no padding to mask) loss -- no manual cross_entropy wrapper needed, since
this data pipeline has nothing to mask in the first place."""
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
BATCH = 64            # native, no grad accumulation needed at block=128
BLOCK = 128            # kev-gpt's own real block_size for this shape
LR = 5e-4              # same peak as Steps 0-7 (schedule SHAPE already cleared in Step 7)
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
    n_head=6,      # head_dim=384/6=64, kev-gpt's own established convention
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

dtype = torch.float16  # kept from Steps 0-7 for consistency -- not the variable under test here
ctx = torch.autocast(device_type=device, dtype=dtype)
scaler = torch.amp.GradScaler(enabled=(device == "cuda" and dtype == torch.float16))


def get_batch(data, block, batch, device):
    # Verbatim copy of model/train.py's own get_batch: random contiguous
    # windows, x/y pre-shifted by one, no padding, no masking.
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
            _, loss = model(x, y)  # kev-gpt's own unmodified, unmasked loss
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
