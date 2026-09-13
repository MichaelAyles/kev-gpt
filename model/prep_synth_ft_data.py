"""Tokenizes a synthetic story corpus (model.filter_synth_corpus's output)
against an EXISTING checkpoint's vocab, for fine-tuning/distillation --
NOT model.word_data.prepare_word(), which always builds a fresh vocab from
whatever corpus it's given.

A synthetic corpus (thousands of short generated samples) is far too
small to build a real 16384-word vocabulary from scratch, and the
fine-tune must stay vocabulary-compatible with the checkpoint it's
--init-from'ing -- so this reuses the deployed data dir's own meta.json
stoi/itos verbatim (only source/token-count fields differ in the output
meta.json) rather than calling build_vocab() again.

    python -m model.prep_synth_ft_data data/teacher_synth_corpus_filtered.txt \\
        --src-meta data/word_v16384/meta.json --out-dir data/word_v16384_synth_ft
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .word_data import EOS, encode, tokenize


def main(argv=None):
    ap = argparse.ArgumentParser(prog="model.prep_synth_ft_data")
    ap.add_argument("corpus", help="filtered synthetic corpus, one story per line")
    ap.add_argument("--src-meta", required=True,
                     help="meta.json of the EXISTING data dir whose vocab to reuse")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    a = ap.parse_args(argv)

    with open(a.src_meta) as f:
        meta = json.load(f)
    stoi = meta["stoi"]

    all_tokens: list[str] = []
    with open(a.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_tokens.extend(tokenize(line))
            all_tokens.append(EOS)

    ids = encode(all_tokens, stoi)
    n_val = max(1, int(len(ids) * a.val_frac))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    os.makedirs(a.out_dir, exist_ok=True)
    train_ids.tofile(os.path.join(a.out_dir, "train.bin"))
    val_ids.tofile(os.path.join(a.out_dir, "val.bin"))

    unk_count = int((ids == stoi["<unk>"]).sum())
    out_meta = dict(meta)
    out_meta["source"] = os.path.abspath(a.corpus)
    out_meta["total_tokens"] = len(ids)
    out_meta["train_tokens"] = len(train_ids)
    out_meta["val_tokens"] = len(val_ids)
    out_meta["unk_frac"] = unk_count / len(ids)
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump(out_meta, f, indent=2)

    print(f"wrote {a.out_dir}: {len(ids)} tokens ({len(train_ids)} train / "
          f"{len(val_ids)} val), unk_frac={unk_count/len(ids):.4f}")


if __name__ == "__main__":
    main()
