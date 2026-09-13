"""Batched story generation from a checkpoint, for building a synthetic
fine-tuning/distillation corpus (model/SCALE-UP-LOG.md-adjacent
distillation experiment: a bigger, more coherent "teacher" checkpoint
generates a corpus a small deployed-shape model is later fine-tuned on).

model.gpt.GPT.generate() already supports a batch dimension -- each row
samples independently via torch.multinomial -- so generating BATCH copies
of one prompt in a single forward-pass sequence is much faster per-sample
than looping one prompt at a time (measured ~7x: 3.7 samples/s unbatched
vs ~26 samples/s at batch=128, plateauing there -- compute-bound at this
model size, not batch-size-bound, so bigger batches don't help further).

Prompts are drawn from a fixed short list plus random short snippets
pulled from the training corpus itself (first 3-6 words of a random
story line), for variety without needing a separate prompt dataset.

    python -m model.gen_synth_corpus data/ckpt_teacher_d384_v16384.pt \\
        --corpus data/TinyStories-train.filtered.txt \\
        --out data/teacher_synth_corpus.txt --n-samples 50000
"""
from __future__ import annotations

import argparse
import random
import re
import time

import torch

from .sample import load
from .word_data import decode, tokenize, UNK

FIXED_PROMPTS = [
    "once upon a time",
    "the sun was",
    "the dog ran",
    "a little girl",
    "she found a",
    "one day",
    "there was a",
    "the little boy",
]


def load_corpus_snippets(corpus_path: str, max_lines: int = 200_000) -> list[str]:
    lines = []
    with open(corpus_path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > max_lines:
                break
            line = line.strip()
            if 20 < len(line) < 1000:
                lines.append(line)
    return lines


def make_prompt(corpus_lines: list[str], fixed_frac: float = 0.4) -> str:
    if not corpus_lines or random.random() < fixed_frac:
        return random.choice(FIXED_PROMPTS)
    line = random.choice(corpus_lines)
    words = re.findall(r"[a-zA-Z']+", line.lower())
    n = random.randint(3, 6)
    return " ".join(words[:n]) if len(words) >= n else random.choice(FIXED_PROMPTS)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="model.gen_synth_corpus")
    ap.add_argument("ckpt", help="teacher checkpoint to generate from")
    ap.add_argument("--corpus", required=True, help="corpus to pull prompt snippets from")
    ap.add_argument("--out", required=True, help="output text file, one story per line")
    ap.add_argument("--n-samples", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=128,
                     help="completions per forward-pass sequence (same prompt, "
                          "independent sampling per row)")
    ap.add_argument("--n-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(argv)

    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta, it, val = load(a.ckpt, device)
    stoi, itos = meta["stoi"], meta["itos"]
    print(f"loaded {a.ckpt} (iter {it}, val {val:.3f}), device={device}")

    corpus_lines = load_corpus_snippets(a.corpus)
    print(f"pulled {len(corpus_lines)} candidate corpus snippets")

    n_batches = (a.n_samples + a.batch - 1) // a.batch
    t0 = time.time()
    written = 0
    with open(a.out, "w", encoding="utf-8") as out_f:
        for b in range(n_batches):
            prompt = make_prompt(corpus_lines)
            prompt_ids = [stoi.get(t, stoi[UNK]) for t in (tokenize(prompt) or [UNK])]
            ids = torch.tensor([prompt_ids] * a.batch, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(ids, a.n_tokens, temperature=a.temperature, top_k=a.top_k)
            for row in out.tolist():
                text = decode(row, itos).replace("\n", " ").strip()
                out_f.write(text + "\n")
                written += 1
            if (b + 1) % 5 == 0 or b == 0:
                elapsed = time.time() - t0
                rate = written / elapsed
                eta = (a.n_samples - written) / rate if rate > 0 else float("inf")
                print(f"batch {b+1}/{n_batches}, {written}/{a.n_samples} written, "
                      f"{elapsed:.1f}s elapsed, {rate:.1f} samples/s, ETA {eta:.0f}s")

    print(f"done. wrote {written} samples to {a.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
