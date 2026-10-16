"""Captures ~128-word stories from the reference model
(SauravP97/tiny-stories-19M, published checkpoint) for the kev-gpt-vs-
reference length/quality benchmark (see build_benchmark_128w_report.py).

Same prompts/repeat count/sampling convention as this directory's other
reference sweeps (sweep_tiny_stories_hf.py, quality_sweep_C_vs_reference.py):
5 prompts x 5 seeds = 25 samples, temp=0.7/top_k=50, objective detector
model.filter_synth_corpus.is_degenerate. max_new_tokens is set generously
above a 128-word target since GPT-NeoX-style BPE tokenization runs
~1.3 tokens/English word for this domain -- actual word counts vary
sample to sample and are reported as-is, not truncated to force exactly
128.

    python model/tinystories_hf_repro/benchmark_reference_stories_128w.py \\
        --out model/tinystories_hf_repro/benchmark_128w_reference.json
"""
import argparse
import json
import sys

sys.path.insert(0, "/home/tparng/kev-gpt")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.filter_synth_corpus import is_degenerate

MODEL_ID = "SauravP97/tiny-stories-19M"
PROMPTS = ["Once upon a time", "The sun was", "The dog ran", "A little girl", "She found a"]
SEEDS = [1, 2, 3, 4, 5]
MAX_NEW_TOKENS = 180  # generous headroom above a ~128-word target
MIN_BIGRAM_RECURRENCE = 6  # see benchmark_hw_stories_128w.py's docstring --
                            # the detector saturates at this length under the
                            # usual threshold of 3; 6 was picked from this
                            # benchmark's own actual bigram-count distribution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--min-bigram-recurrence", type=int, default=MIN_BIGRAM_RECURRENCE)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print(f"loaded {MODEL_ID}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, device={device}")

    results = []
    for seed in SEEDS:
        for prompt in PROMPTS:
            torch.manual_seed(seed)
            inputs = tok(prompt, return_tensors="pt").to(device)
            out = model.generate(
                inputs.input_ids, max_new_tokens=a.max_new_tokens, do_sample=True,
                temperature=0.7, top_k=50, pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(out[0], skip_special_tokens=True).replace("\n", " ").strip()
            n_words = len(text.split())
            reason = is_degenerate(text, min_bigram_recurrence=a.min_bigram_recurrence)

            sample = {"seed": seed, "prompt": prompt, "text": text,
                      "reason": reason, "word_count": n_words}
            results.append(sample)
            print(f"=== seed={seed} prompt={prompt!r} words={n_words} reason={reason!r} ===")
            print(text[:200])

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)

    flagged = sum(1 for s in results if s["reason"])
    avg_words = sum(s["word_count"] for s in results) / len(results)
    print(f"\nDONE,total={len(results)},flagged={flagged},avg_words={avg_words:.1f}")
    print(f"WROTE,{a.out}")


if __name__ == "__main__":
    main()
