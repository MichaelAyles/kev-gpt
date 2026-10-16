"""Ablation step 4: swap ONLY attention SCOPE -- GPT-Neo local/windowed
attention (window_size=256 over seq_len=512, all 8 layers) -> GPT-Neo
"global" attention_type (the same GPTNeoSelfAttention module, same
missing 1/sqrt(head_dim) scaling, same explicit fp32 QK^T upcast, same
separate q/k/v/out_proj structure -- only the causal-mask construction
changes: plain full lower-triangular instead of the windowed XOR mask)
-- holding everything else at Step 3's already-fully-kev-gpt values:
kev-gpt's own closed word-level vocab (VOCAB=16384), weight_decay=0.1,
beta2=0.95, batch_size=64 (via gradient accumulation). Every other axis
identified in model/SCALE-UP-LOG.md's Phase 2 architecture comparison
has been tested and cleared (loop mechanics: Step 1; all three
hyperparameters: Steps 2a-2c; attention precision: --attn-fp32;
tokenizer: Step 3) -- attention scope is the last one. Deliberately
staying inside GPTNeoSelfAttention (not porting kev-gpt's own SDPA-based
CausalSelfAttention) to isolate scope alone, since a full class swap
would also change scaling/precision/bias structure all at once."""
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GPTNeoConfig

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
OUT_STATES = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ablation1_states.jsonl"
CKPT = sys.argv[3] if len(sys.argv) > 3 else OUT_STATES.replace(".jsonl", "_ckpt.pt")
BATCH = 64
MICRO_BATCH = 8  # gradient accumulation, same VRAM-ceiling workaround as Step 2c
ACCUM_STEPS = BATCH // MICRO_BATCH
SEQ_LEN = 512
LR = 5e-4
WEIGHT_DECAY = 0.1
BETA2 = 0.95
EPS = 1e-8

device = "cuda" if torch.cuda.is_available() else "cpu"

# kev-gpt's own word-level vocab (model/word_data.py): fixed VOCAB=16384,
# top-(vocab_size-2) most frequent regex-split word/punctuation tokens,
# <unk>=0 for anything outside it, <eos>=1 as story-boundary/pad token.
# Reuses the SAME meta.json the deployed VOCAB=16384 hardware checkpoint's
# own data prep produced (data/word_v16384/meta.json) -- not regenerated,
# so this test's vocab is bit-identical to the real deployed tokenizer.
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
    # Same per-story framing as Steps 0-2c (one row = one story, truncated/
    # padded to SEQ_LEN=512) so this is a clean single-variable swap, not a
    # confound with kev-gpt's own flat-stream/random-window sampling
    # convention (model/data.py's load_split) which frames batches
    # differently. EOS doubles as pad (matches Steps 0-2c's own convention
    # of pad_token==eos_token and masking every eos-valued label position,
    # not just trailing padding -- see get_batch() below).
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

config = GPTNeoConfig(
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=SEQ_LEN,
    hidden_size=256,
    num_layers=8,
    num_heads=16,
    attention_types=[[["global"], 8]],
)
model = AutoModelForCausalLM.from_config(config).to(device)
print(f"model parameters: {model.num_parameters() / 1_000_000:.2f}M")

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
    rows = ds[idx]["input_ids"].to(device)
    x = rows
    y = rows.clone()
    y[y == EOS_ID] = -100  # mask every eos/pad position, matches Steps 0-2c's convention
    return x, y


@torch.no_grad()
def estimate_val_loss(model, val_ds, batch, device, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_ds, MICRO_BATCH, device)
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
        print(f"  step {step} micro {micro}/{ACCUM_STEPS} fetching...", flush=True)
        x, y = get_batch(train_ds, MICRO_BATCH, device)
        print(f"  step {step} micro {micro}/{ACCUM_STEPS} forward...", flush=True)
        with ctx:
            out = model(x, labels=y)
            loss = out.loss / ACCUM_STEPS
        print(f"  step {step} micro {micro}/{ACCUM_STEPS} backward...", flush=True)
        scaler.scale(loss).backward()
    print(f"  step {step} optimizer step...", flush=True)
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

print("done.")
