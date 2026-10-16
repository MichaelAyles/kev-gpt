"""Filters a synthetic story corpus (model.gen_synth_corpus's output),
dropping samples with genuinely degenerate repetition before using it as
fine-tuning/distillation data.

Two checks, both needed (found the hard way -- see model/SCALE-UP-LOG.md-
adjacent distillation experiment / fabric/genesys2/PORT-NOTES.md's
"Distillation from a bigger teacher" section):
  1. Immediate exact word-doubling ("yummy yummy") -- always rejected,
     unambiguous.
  2. A bigram recurring 3+ times within one sample -- a real loop
     ("the ball ... the ball ... the ball"), not just normal noun-phrase
     reuse. A first version rejected on ANY single recurrence (2+) and
     dropped 599/600 samples, almost all false positives on ordinary
     sentence-starter reuse (". she", "a little") -- the exact same
     over-firing bug the real-hardware repetition guard hit and fixed
     (main.c's bigram guard needed a stop-id exemption). Bigrams where
     either word is punctuation-only are exempt from the recurrence count
     for the same reason.

    python -m model.filter_synth_corpus data/teacher_synth_corpus.txt \\
        --out data/teacher_synth_corpus_filtered.txt
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

from .word_data import tokenize

PUNCT_RE = re.compile(r"^[^a-zA-Z0-9']+$")


def is_punct(tok: str) -> bool:
    return bool(PUNCT_RE.match(tok))


def is_degenerate(line: str, min_bigram_recurrence: int = 3) -> str | None:
    """Returns a reason string if `line` should be dropped, else None."""
    words = tokenize(line)
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            return f"exact doubling: {words[i-1]!r} {words[i]!r}"
    bigram_counts = Counter()
    for i in range(1, len(words)):
        if is_punct(words[i - 1]) or is_punct(words[i]):
            continue
        bigram_counts[(words[i - 1], words[i])] += 1
    worst = bigram_counts.most_common(1)
    if worst and worst[0][1] >= min_bigram_recurrence:
        return f"bigram recurred {worst[0][1]}x: {worst[0][0]}"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="model.filter_synth_corpus")
    ap.add_argument("corpus", help="input text file, one story per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-bigram-recurrence", type=int, default=3)
    ap.add_argument("--show-dropped", type=int, default=5,
                     help="print this many dropped examples with their reason")
    a = ap.parse_args(argv)

    kept, dropped = [], []
    with open(a.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reason = is_degenerate(line, a.min_bigram_recurrence)
            if reason:
                dropped.append((line, reason))
            else:
                kept.append(line)

    with open(a.out, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")

    print(f"kept {len(kept)} / {len(kept) + len(dropped)}, dropped {len(dropped)}")
    if a.show_dropped:
        print(f"\n--- sample of dropped, with reason ---")
        for line, reason in dropped[: a.show_dropped]:
            print(f"[{reason}]")
            print(line)
            print()


if __name__ == "__main__":
    main()
