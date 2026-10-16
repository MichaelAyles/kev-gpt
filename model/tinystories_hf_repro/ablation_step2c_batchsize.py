"""Ablation step 1: swap ONLY the training loop/framework (HF Trainer ->
kev-gpt's own manual loop mechanics: autocast+GradScaler, clip_grad_norm_
max_norm=1.0, manual optimizer.step()), holding everything else at
SauravP97/tiny-stories-hf's exact values: GPTNeoConfig architecture,
GPT-Neo BPE tokenizer/dataset, batch_size=8, lr=5e-4, weight_decay=0.01,
beta2=0.999 (default), eps=1e-8 (default), and their exact LR schedule
shape (PURE LINEAR DECAY FROM STEP 0, NO WARMUP -- confirmed by
reverse-engineering repro_hf_recipe.py's own logged LR values against
HF TrainingArguments' defaults: no warmup_steps/warmup_ratio was set,
so lr_scheduler_type="linear" decays linearly from `lr` to 0 with zero
warmup, not kev-gpt's own cosine+warmup shape)."""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoConfig

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
OUT_STATES = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ablation1_states.jsonl"
# Checkpoint save/resume: added after this script hung twice mid-run (once
# from a stalled CUDA call after the user paused/resumed a separate
# notebook on the same GPU) with NO way to recover the lost progress --
# the original version only logged eval stats to OUT_STATES, never saved
# model/optimizer state anywhere. Every eval step now also writes CKPT;
# if CKPT already exists at startup, training resumes from it instead of
# restarting at step 0.
CKPT = sys.argv[3] if len(sys.argv) > 3 else OUT_STATES.replace(".jsonl", "_ckpt.pt")
BATCH = 64
MICRO_BATCH = 8  # gradient accumulation: this GPU (7.62GB VRAM) OOMs materializing
                 # a full batch=64 logits tensor at once (64*512*50257*4 bytes =~
                 # 6.1GB for the float32 loss cast alone) -- accumulate 8 micro-
                 # batches of 8 before stepping, reproducing batch=64's true
                 # gradient statistics (same effective batch size / noise level)
                 # without materializing it in one forward pass.
ACCUM_STEPS = BATCH // MICRO_BATCH
SEQ_LEN = 512
LR = 5e-4
WEIGHT_DECAY = 0.1
BETA2 = 0.95
EPS = 1e-8

device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading dataset (cached)...")
dataset = load_dataset("roneneldan/TinyStories")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=SEQ_LEN)

print("tokenizing (cached)...")
tokenized = dataset.map(tokenize_function, batched=True, num_proc=8)
# Keep the Arrow-backed dataset as-is (memory-mapped, lazy) and index into
# it per-batch instead of materializing all 2.1M rows into a Python list
# first -- torch.tensor(tokenized["train"]["input_ids"]) OOM'd at ~56GB RSS
# on the intermediate Python-list representation before it ever reached
# the packed tensor (~8.7GB final size).
tokenized["train"].set_format("torch", columns=["input_ids"])
tokenized["validation"].set_format("torch", columns=["input_ids"])
train_ds = tokenized["train"]
val_ds = tokenized["validation"].select(range(500))
print(f"train rows={len(train_ds)}  val rows={len(val_ds)}")

config = GPTNeoConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=SEQ_LEN,
    hidden_size=256,
    num_layers=8,
    num_heads=16,
    attention_types=[[["local"], 8]],
)
model = AutoModelForCausalLM.from_config(config).to(device)
print(f"model parameters: {model.num_parameters() / 1_000_000:.2f}M")

# kev-gpt's own decay/no_decay split (model/train.py) -- >=2D params get
# weight decay, 1D (LayerNorm/bias) don't. This is a PERMANENT feature of
# kev-gpt's own loop (found to help, kept unconditionally), not something
# being ablated here -- only the loop MECHANICS (autocast/scaler/clip) and
# theirs-vs-kev-gpt's specific hyperparameter VALUES are in play.
decay, no_decay = [], []
for p in model.parameters():
    if not p.requires_grad:
        continue
    (decay if p.dim() >= 2 else no_decay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": WEIGHT_DECAY},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=LR, betas=(0.9, BETA2), eps=EPS)

dtype = torch.float16  # match their fp16=True exactly, not kev-gpt's own bf16-by-default convention
ctx = torch.autocast(device_type=device, dtype=dtype)
scaler = torch.amp.GradScaler(enabled=(device == "cuda" and dtype == torch.float16))


def get_batch(ds, batch, device):
    # HF *ForCausalLM.forward(input_ids=, labels=) does its OWN internal
    # shift-by-one for the loss (shift_logits = logits[:, :-1], shift_labels
    # = labels[:, 1:]) -- it expects labels to be the SAME, unshifted
    # sequence as input_ids (exactly what DataCollatorForLanguageModeling
    # does: labels = input_ids.clone()). An earlier version of this function
    # did a manual nanoGPT-style pre-shift (x = rows[:-1], y = rows[1:]) --
    # a real double-shift bug: the model ends up comparing its prediction
    # for position i+1 against the label for position i+2 on every step,
    # degrading convergence the whole run (val 2.83 vs the HF Trainer
    # reproduction's 1.71 at the same step count, before this fix).
    n = len(ds)
    idx = torch.randint(0, n, (batch,)).tolist()
    rows = ds[idx]["input_ids"].to(device)  # [batch, SEQ_LEN], lazy Arrow row fetch
    x = rows
    y = rows.clone()
    y[y == tokenizer.pad_token_id] = -100  # ignore pad in loss, matches DataCollatorForLanguageModeling
    return x, y


@torch.no_grad()
def estimate_val_loss(model, val_ds, batch, device, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_ds, MICRO_BATCH, device)  # eval in micro-batches too, same OOM risk
        with ctx:
            out = model(x, labels=y)
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / len(losses)


def linear_lr_no_warmup(step, lr, total_steps):
    return lr * max(0.0, 1.0 - step / total_steps)


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
    lr = linear_lr_no_warmup(step, LR, MAX_STEPS)
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
        # Heartbeat: a prior run of this script hung silently for 5+ hours
        # (main thread pegged at ~100% CPU, GPU utilization ~0-1%, no
        # traceback/OOM/dmesg error) with no way to tell which line it was
        # stuck on, since stdout was fully block-buffered under `>` redirect.
        # flush=True here + `python -u` at launch pins down exactly which
        # step/micro-batch/stage (data fetch vs forward vs backward) any
        # future hang is stuck in, by comparing wall-clock gaps between
        # consecutive heartbeat lines in the log.
        print(f"  step {step} micro {micro}/{ACCUM_STEPS} fetching...", flush=True)
        x, y = get_batch(train_ds, MICRO_BATCH, device)
        print(f"  step {step} micro {micro}/{ACCUM_STEPS} forward...", flush=True)
        with ctx:
            out = model(x, labels=y)
            loss = out.loss / ACCUM_STEPS  # mean-reduce across the accumulated micro-batches
        print(f"  step {step} micro {micro}/{ACCUM_STEPS} backward...", flush=True)
        scaler.scale(loss).backward()
    print(f"  step {step} optimizer step...", flush=True)
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

print("done.")
