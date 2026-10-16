"""Train the Kevin GPT on a Kevinised corpus.

Device auto-detects (CUDA > MPS > CPU) so the same script trains on the XPS
3050 Ti and the M1. Mixed precision is used where the device supports it.

Quick throughput probe before committing to a real run:
    python -m model.train --smoke 50

Real (proof-of-life) run on the validation corpus:
    python -m model.train --max-iters 4000 --eval-interval 500
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .data import load_split, prepare
from .gpt import GPT, GPTConfig


def _prepare_data(args):
    """Dispatch to the char- or word-level tokeniser based on --tokenizer.
    Returns the meta dict; both tokenisers write the same train.bin/val.bin/
    meta.json shape under args.data_dir, so everything downstream of this
    call (load_split, meta["vocab_size"]) is tokeniser-agnostic."""
    if args.tokenizer == "word":
        from .word_data import prepare_word
        return prepare_word(args.corpus, args.data_dir, args.vocab_size)
    return prepare(args.corpus, args.data_dir)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def amp_dtype(device: str):
    """Pick an autocast dtype, or None to disable autocast on this device."""
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16  # bf16 autocast is flaky on MPS; fp16 is fine here
    return None  # cpu: plain fp32


def get_batch(data: np.ndarray, block: int, batch: int, device: str):
    ix = torch.randint(len(data) - block - 1, (batch,))
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def cosine_lr(it: int, lr: float, warmup: int, total: int, min_lr_frac: float = 0.1):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > total:
        return lr * min_lr_frac
    ratio = (it - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return lr * (min_lr_frac + (1 - min_lr_frac) * coeff)


def layer_grad_norms(model):
    """Per-layer gradient L2 norms, grouped coarsely enough to read at a
    glance (one number per block-component) but fine enough to tell WHICH
    part of the network is producing large/growing gradients -- added while
    chasing the width-scaling training instability (model/SCALE-UP-LOG.md):
    every loss-function-level fix tried there (LR, warmup, init scaling,
    weight decay, z-loss) left the divergence's ONSET TIMING unchanged,
    which rules out those as the root cause but doesn't say WHERE in the
    network it originates -- this is the next, lower-level instrument to
    find that. Call AFTER backward()+unscale_(), BEFORE clip_grad_norm_
    (clipping is a single global rescale, so it doesn't change the
    RELATIVE proportions between layers, but measuring pre-clip keeps the
    raw numbers meaningful on their own, not relative to whatever the clip
    threshold happens to be).
    """
    groups: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = p.grad.norm().item() ** 2
        if name.startswith("blocks."):
            _, idx, rest = name.split(".", 2)
            comp = {"ln1.weight": "ln1", "ln2.weight": "ln2",
                    "attn.qkv.weight": "qkv", "attn.proj.weight": "proj",
                    "mlp.0.weight": "mlp_fc", "mlp.2.weight": "mlp_proj"}.get(rest, rest)
            key = f"block{idx}.{comp}"
        elif name in ("tok_emb.weight", "head.weight"):
            key = "tok_emb"
        elif name == "pos_emb.weight":
            key = "pos_emb"
        elif name == "ln_f.weight":
            key = "ln_f"
        else:
            key = name
        groups[key] = groups.get(key, 0.0) + n
    return {k: v ** 0.5 for k, v in groups.items()}


@torch.no_grad()
def estimate_loss(model, splits, block, batch, device, iters=50):
    model.eval()
    # report pure cross-entropy even when z_loss_coef > 0, so val numbers
    # stay comparable across z-loss on/off runs -- z-loss is a TRAINING
    # regularizer, not part of the quality metric being tracked.
    z_loss_coef = model.cfg.z_loss_coef
    model.cfg.z_loss_coef = 0.0
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block, batch, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.cfg.z_loss_coef = z_loss_coef
    model.train()
    return out


def sample(model, meta, device, n_tokens=200, prompt="\n", seed=None):
    if seed is not None:
        torch.manual_seed(seed)  # fixed seed -> comparable samples across evals
    stoi, itos = meta["stoi"], meta["itos"]
    if meta.get("tokenizer") == "word":
        from .word_data import decode, tokenize, UNK
        toks = tokenize(prompt.lower()) or [UNK]
        ids = torch.tensor([[stoi.get(t, stoi[UNK]) for t in toks]],
                            dtype=torch.long, device=device)
        out = model.generate(ids, n_tokens, temperature=0.8, top_k=40)[0].tolist()
        return decode(out, itos).replace("\n", " | ")
    from .data import decode
    ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(ids, n_tokens, temperature=0.8, top_k=40)[0].tolist()
    return decode(out, itos).replace("\n", " | ")


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.train", description="Train the Kevin GPT.")
    p.add_argument("--corpus", default="data/TinyStories-valid.kevin.txt")
    p.add_argument("--data-dir", default="data/char")
    p.add_argument("--tokenizer", choices=["char", "word"], default="char",
                    help="word uses model.word_data's fixed-vocab word tokeniser "
                         "instead of char-level; needs --vocab-size sized against "
                         "GW_HEAD/GW_EMB before training (see PORT-NOTES.md).")
    p.add_argument("--vocab-size", type=int, default=1900,
                    help="word tokenizer only -- ignored for --tokenizer char "
                         "(char vocab is whatever the corpus's own char set is).")
    p.add_argument("--out", default="data/ckpt.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--attn-fp32", action="store_true",
                    help="do attention's QK^T + softmax in explicit fp32 instead "
                         "of trusting SDPA's opaque autocast-dtype path, matching "
                         "GPT-Neo's own GPTNeoSelfAttention precision convention. "
                         "Diagnostic for model.SCALE-UP-LOG.md's Phase 2 attention "
                         "comparison -- default off, unchanged SDPA behavior.")
    p.add_argument("--z-loss-coef", type=float, default=0.0,
                    help="logsumexp(logits)^2 penalty coefficient, caps logit "
                         "scale growth (PaLM/ST-MoE use ~1e-4); 0.0 = off.")
    p.add_argument("--layer-grad-log", default=None,
                    help="append per-block gradient-norm JSONL here every "
                         "--layer-grad-interval iters (model.SCALE-UP-LOG.md's "
                         "next diagnostic step); default None = off.")
    p.add_argument("--layer-grad-interval", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                    help="sgd (+momentum=0.9) has no per-parameter variance "
                         "normalization, unlike AdamW -- see model.SCALE-UP-LOG.md's "
                         "Attempt 7 for why this matters. --lr needs to be much "
                         "higher for sgd than adamw (no adaptive scaling).")
    p.add_argument("--adam-eps", type=float, default=1e-8,
                    help="AdamW's sqrt(v_hat)+eps denominator floor (torch default "
                         "1e-8). Attempt 7 pinned the scale-up instability on Adam's "
                         "variance normalization specifically; a much larger eps "
                         "(try 1e-3 to 1e-2) makes that denominator less aggressive "
                         "at small gradient-variance estimates, closer to raw-gradient "
                         "SGD-like behavior right where it matters. See "
                         "model/SCALE-UP-LOG.md's Attempt 8.")
    p.add_argument("--adam-beta2", type=float, default=0.95,
                    help="AdamW's second-moment (variance) EMA decay (this "
                         "codebase's prior default 0.95, torch default 0.999). A "
                         "lower value (try 0.9 or below) makes the variance "
                         "estimate track recent gradients more responsively "
                         "instead of averaging over a longer window that may be "
                         "slow to shrink the effective step. See "
                         "model/SCALE-UP-LOG.md's Attempt 8.")
    p.add_argument("--max-iters", type=int, default=4000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lr-decay-iters", type=int, default=None,
                    help="cosine-decay horizon, decoupled from --max-iters -- "
                         "LR reaches its floor (min_lr_frac*lr) at this iter and "
                         "HOLDS there for any remaining iters, instead of the "
                         "decay stretching across the whole run. Default: same "
                         "as --max-iters (old behavior, decay ends exactly at "
                         "the last iter). See model/SCALE-UP-LOG.md's Attempt 6 "
                         "(longer/gentler decay: get to a low LR fast, then see "
                         "if the model stays stable there for a long time).")
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--states", default="data/states.jsonl",
                   help="append (iter, losses, sample) to this JSONL each eval; "
                        "render with model.evolution. Empty to disable.")
    p.add_argument("--snapshot-dir", default="data/ckpts",
                   help="save a checkpoint at every eval here (ckpt_<iter>.pt) so "
                        "you can roll back to any point. Empty to disable.")
    p.add_argument("--compile", action="store_true", help="torch.compile (CUDA).")
    p.add_argument("--smoke", type=int, default=0,
                   help="run N iters, report tokens/sec and a time estimate, exit.")
    p.add_argument("--qat", action="store_true",
                   help="train the Brevitas INT4-weight / INT8-activation QAT model "
                        "(model.qgpt.QGPT) instead of the FP GPT. Warm-start from an FP "
                        "checkpoint with --init-from for the standard recipe.")
    p.add_argument("--init-from", default=None, metavar="CKPT",
                   help="load weights from this checkpoint before training. With --qat, "
                        "FP weights are remapped into the QAT module and Brevitas's scale "
                        "params initialise fresh (load is strict=False).")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    dtype = amp_dtype(device)
    torch.manual_seed(1337)
    print(f"device={device}  amp={dtype}")

    meta = _prepare_data(args)
    n_train = meta.get("train_chars", meta.get("train_tokens"))
    unit = "chars" if "train_chars" in meta else "tokens"
    print(f"vocab={meta['vocab_size']}  train_{unit}={n_train:,}")
    splits = {"train": load_split(args.data_dir, "train"),
              "val": load_split(args.data_dir, "val")}

    cfg = GPTConfig(
        block_size=args.block_size, vocab_size=meta["vocab_size"],
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, z_loss_coef=args.z_loss_coef,
        attn_fp32=args.attn_fp32,
    )
    if args.qat:
        from .qgpt import QGPT, load_fp_into_qat
        # Build on CPU, warm-start, run one dry forward to materialise
        # Brevitas's lazy scale buffers, THEN move to device. .to() picks up
        # everything (params + buffers, lazy or not) only after they exist.
        model = QGPT(cfg)
        print(f"params={model.num_params() / 1e6:.2f}M  (QAT: INT4 weights, INT8 acts)")
        if args.init_from:
            ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
            missing, unexpected = load_fp_into_qat(model, ck["model"])
            n_scale_missing = sum(1 for k in missing if "input_quant" in k)
            n_real_missing = len(missing) - n_scale_missing
            print(f"warm-start from {args.init_from} (iter {ck.get('iter')}): "
                  f"transferred OK, {n_scale_missing} fresh quant-scale params, "
                  f"{n_real_missing} unmatched, {len(unexpected)} unexpected")
            if n_real_missing or unexpected:
                print(f"  unmatched: {[k for k in missing if 'input_quant' not in k]}")
                print(f"  unexpected: {unexpected}")
        with torch.no_grad():
            model(torch.zeros((1, 1), dtype=torch.long))
        model.to(device)
    else:
        model = GPT(cfg).to(device)
        print(f"params={model.num_params() / 1e6:.2f}M")
        if args.init_from:
            ck = torch.load(args.init_from, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            print(f"warm-start from {args.init_from} (iter {ck.get('iter')})")
    if args.compile and device == "cuda":
        model = torch.compile(model)

    # Weight decay only on >=2D weight matrices (Linear/Embedding), not on
    # 1D params (LayerNorm gains, any biases) -- standard GPT-2 practice this
    # codebase was missing. Applying decay uniformly (the old behavior) pulls
    # LayerNorm gains toward zero regardless of what the model actually
    # needs there, fighting whatever the optimizer is otherwise trying to do
    # with them -- a real, previously-undocumented contributor investigated
    # after wider models (D=256 char-level, D=384 word-level) repeatedly hit
    # a train+val-loss-climbs-together instability a few thousand iters after
    # warmup, with LayerNorm gain norms found to nearly double over the climb.
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    # --optimizer sgd: the direct test of the leading scale-up-instability
    # hypothesis in model/SCALE-UP-LOG.md's Attempt 6 -- Adam's per-parameter
    # update is VARIANCE-NORMALIZED (lr * m_hat / (sqrt(v_hat)+eps)), so its
    # effective step size doesn't necessarily shrink to zero even at a tiny
    # nominal lr. Plain SGD (+momentum) has no such normalization: its step
    # is directly proportional to the raw gradient, so if the same
    # bottom-then-diverge pattern still shows up here, that specifically
    # rules Adam's normalization out too, not just LR scheduling.
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(
            [{"params": decay, "weight_decay": 0.1},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, args.adam_beta2), eps=args.adam_eps)
    use_amp = dtype is not None
    ctx = (torch.autocast(device_type=device, dtype=dtype) if use_amp
           else torch.autocast(device_type="cpu", enabled=False))
    scaler = torch.amp.GradScaler(enabled=(device == "cuda" and dtype == torch.float16))

    block, batch = args.block_size, args.batch_size

    # --- smoke: measure throughput, extrapolate, and stop ---
    if args.smoke:
        model.train()
        for _ in range(3):  # warmup (kernels compile / lazy init)
            x, y = get_batch(splits["train"], block, batch, device)
            with ctx:
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.smoke):
            x, y = get_batch(splits["train"], block, batch, device)
            with ctx:
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        it_s = args.smoke / dt
        tok_s = it_s * batch * block
        print(f"\nthroughput: {it_s:.1f} it/s  |  {tok_s/1e3:.0f}k tokens/s")
        for target in (2000, 4000, 8000):
            print(f"  {target} iters ~= {target/it_s/60:.1f} min "
                  f"({target*batch*block/1e6:.0f}M tokens seen)")
        return

    # --- real training loop ---
    if args.states:
        os.makedirs(os.path.dirname(args.states) or ".", exist_ok=True)
        open(args.states, "w").close()  # truncate any previous run's states
    if args.snapshot_dir:
        os.makedirs(args.snapshot_dir, exist_ok=True)
    if args.layer_grad_log:
        os.makedirs(os.path.dirname(args.layer_grad_log) or ".", exist_ok=True)
        open(args.layer_grad_log, "w").close()

    def save_ckpt(path, it, val):
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                    "meta": meta, "iter": it, "val": val, "qat": args.qat}, path)

    best_val = float("inf")
    t0 = time.perf_counter()
    for it in range(args.max_iters + 1):
        lr = cosine_lr(it, args.lr, args.warmup, args.lr_decay_iters or args.max_iters)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, splits, block, batch, device, args.eval_iters)
            el = time.perf_counter() - t0
            # printed sample is free-running; logged sample is fixed-seed so the
            # garbage->text progression reflects the model, not the dice.
            shown = sample(model, meta, device, 180)
            logged = sample(model, meta, device, 180, seed=1234)
            # Logit-scale diagnostic (added while investigating the width-
            # scaling instability, see the AdamW param-grouping comment
            # above): one no-grad forward pass on a fresh val batch, report
            # logit std/max plus ln_f's own gain norm -- confirms or refutes
            # "loss climbing because logits are inflating" directly instead
            # of inferring it from weight-norm snapshots after the fact.
            model.eval()
            with torch.no_grad():
                xs, _ = get_batch(splits["val"], block, batch, device)
                with ctx:
                    logits_dbg, _ = model(xs, xs)
                logit_std = logits_dbg.float().std().item()
                logit_max = logits_dbg.float().abs().max().item()
            lnf_norm = model.ln_f.weight.norm().item()
            model.train()
            print(f"iter {it:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| lr {lr:.1e} | {el/60:.1f} min | logit_std {logit_std:.2f} "
                  f"logit_max {logit_max:.1f} lnf_norm {lnf_norm:.2f}")
            print("   sample:", shown)
            if args.states:
                with open(args.states, "a") as f:
                    f.write(json.dumps({
                        "iter": it,
                        "train_loss": round(losses["train"], 4),
                        "val_loss": round(losses["val"], 4),
                        "minutes": round(el / 60, 2),
                        "lr": lr,
                        "sample": logged,
                    }) + "\n")
            # per-eval snapshot for manual rollback (e.g. if val turns up ~2000)
            if args.snapshot_dir:
                save_ckpt(os.path.join(args.snapshot_dir, f"ckpt_{it:05d}.pt"),
                          it, losses["val"])
            # best-val checkpoint = the automatic anti-overfit rollback
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_ckpt(args.out, it, best_val)
                print(f"   * new best val {best_val:.3f} -> {args.out}")

        if it == args.max_iters:
            break

        x, y = get_batch(splits["train"], block, batch, device)
        with ctx:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if args.layer_grad_log and it % args.layer_grad_interval == 0:
            with open(args.layer_grad_log, "a") as f:
                f.write(json.dumps({"iter": it, **layer_grad_norms(model)}) + "\n")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    print(f"done. best val {best_val:.3f}, checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
