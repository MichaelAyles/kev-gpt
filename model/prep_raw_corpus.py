"""Normalizes a raw TinyStories-format corpus (one paragraph per line,
`<|endoftext|>` story delimiters) into a clean, small-vocab plain-text
corpus for model.train's --corpus, WITHOUT running it through keviniser's
POS-stripping transform -- keeps real grammar and sentence-ending
punctuation ('.'/'!'/'?', missing/rare in the Kevin-speak vocab) for a
"mini story teller" checkpoint, at the cost of losing the Kevin-speak
compression that's the point of the rest of this project everywhere else.

Reuses model.data.build_text's own line/marker format, so the output is a
drop-in --corpus value for model.train.

Lowercase-folds (matches char_optionA's vocab convention and
kevgpt_interactive's firmware, which already force-lowercases typed
prompts before lookup -- keeping this checkpoint's own vocab lowercase-
only avoids needing a firmware change) and normalizes curly quotes/dashes/
ellipsis to their plain-ASCII equivalents, then drops anything outside a
fixed common set. Checked empirically against data/TinyStories-valid.txt:
loses 41 of 19,092,837 characters (rare encoding artifacts -- soft
hyphens, stray accented letters, zero-width spaces, one emoji -- not real
content).

    python -m model.prep_raw_corpus data/TinyStories-valid.txt \
        -o data/TinyStories-valid.raw.txt
"""
from __future__ import annotations

import argparse

MARKER = "<|endoftext|>"
ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789 \n.,'\"!?-:;")


def normalize_line(line: str) -> str:
    line = line.lower()
    line = line.replace("‘", "'").replace("’", "'")
    line = line.replace("“", '"').replace("”", '"')
    line = line.replace("–", "-").replace("—", "-")
    line = line.replace("…", "...")
    return "".join(ch for ch in line if ch in ALLOWED)


def prep(in_path: str, out_path: str) -> None:
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip() == MARKER:
                fout.write(MARKER + "\n")
                continue
            fout.write(normalize_line(line))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="model.prep_raw_corpus")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    prep(a.input, a.output)


if __name__ == "__main__":
    main()
