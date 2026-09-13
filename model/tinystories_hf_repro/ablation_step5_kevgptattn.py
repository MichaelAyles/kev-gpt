"""Ablation step 5: swap the WHOLE architecture -- GPT-Neo (GPTNeoSelfAttention,
local or global) -> kev-gpt's own actual `model.gpt.GPT`/`CausalSelfAttention`
class, imported unmodified from the real codebase (not an approximation of one
axis inside GPT-Neo's module, unlike Step 4's attention-scope-only swap). This
is the decisive test Step 4's own conclusion called for: every Phase 2 run so
far (Steps 0-4), regardless of architecture/tokenizer/hyperparameter
combination, used kev-gpt's own DEFAULT eps=1e-8/beta2=0.95 and never
diverged -- while kev-gpt's own D=384/VOCAB=16384 model reliably diverges at
those same default settings. If this run (kev-gpt's real class, same data/
tokenizer/hyperparameters as Step 4) diverges, kev-gpt's own attention
implementation (SDPA's opaque precision path + 1/sqrt(head_dim) scaling +
fused qkv + head_dim=64, together) is implicated. If it stays stable too, the
instability is a property of some OTHER combination this whole Phase 2 walk
hasn't isolated yet (width/depth interacting with something not yet tested).

Held at kev-gpt's own convention: head_dim=64 (n_head=4, n_embd=256, giving
the SAME head_dim kev-gpt's real deployed/diagnostic models use -- NOT
GPT-Neo's head_dim=16), bias=False everywhere (kev-gpt's RMSNorm-style
project convention), dropout=0.0, attn_fp32=False (kev-gpt's real default
SDPA path, not the already-falsified diagnostic fp32 branch). n_layer=8 to
match Step 4's num_layers=8 as closely as possible (width/depth were never
part of what's being isolated here -- only the attention/model CLASS).

Data framing: kept the SAME per-story-padded-to-512 convention as Steps 0-4
(one row = one story, truncated/padded to SEQ_LEN=512 with <eos> filler) --
NOT kev-gpt's own flat-stream/random-window sampler (model/data.py's
load_split) -- to keep architecture the only new variable relative to Step 4,
per this whole walk's one-variable-at-a-time methodology. kev-gpt's own
GPT.forward(idx, targets) does its own nanoGPT-style PRE-shift convention
(model/train.py's get_batch: x=data[i:i+block], y=data[i+1:i+1+block], y
already offset by one -- NOT HF's internal-shift convention Steps 0-4 relied
on for GPT-Neo) -- so x/y here are pre-shifted from each padded row, and
GPT.forward is called with targets=None (getting logits only) so the harness
can apply the SAME eos-masking (ignore_index=-100) Steps 0-4 used, without
touching model/gpt.py's own unmodified forward (which has no masking
mechanism at all, since its real training never needs one)."""
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, "/home/tparng/kev-gpt")
from model.gpt import GPT, GPTConfig  # noqa: E402  (kev-gpt's own, real, unmodified class)

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
OUT_STATES = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ablation1_states.jsonl"
CKPT = sys.argv[3] if len(sys.argv) > 3 else OUT_STATES.replace(".jsonl", "_ckpt.pt")
BATCH = 64
MICRO_BATCH = 8  # gradient accumulation, same VRAM-ceiling workaround as Steps 2c-4
ACCUM_STEPS = BATCH // MICRO_BATCH
SEQ_LEN = 512
BLOCK = SEQ_LEN - 1  # x/y are pre-shifted, so each is BLOCK long, not SEQ_LEN
LR = 5e-4
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
    n_layer=8,
    n_head=4,      # head_dim=256/4=64, kev-gpt's own established convention
    n_embd=256,
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
    y[y == EOS_ID] = -100                    # mask every eos/pad position, matches Steps 0-4
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
