"""Word-level tokeniser and data prep, parallel to data.py's char-level one.

Same file-format contract as data.py (meta.json + train.bin/val.bin of
uint16 ids) so model.train's downstream consumption (load_split, decode,
meta["vocab_size"]) works unchanged regardless of which tokeniser produced
the data-dir -- only `prepare_word()` differs from `data.prepare()`.

Fixed vocabulary of the N most frequent word/punctuation tokens (by simple
regex split -- no BPE, no subword merges), reserving two special ids at the
top: <unk> for anything outside the fixed vocab, <eos> as the story-boundary
token (data.py's char-level scheme used a literal "\\n" char for this; word-
level has no single "boundary character" to reuse, so it needs its own
token). Vocab members below the reserved ids are sorted alphabetically for
determinism, matching data.py's own `sorted(set(text))` convention.

Sizing note (fabric/genesys2/PORT-NOTES.md's "word-level vocabulary"
section): GW_HEAD/GW_EMB (and therefore WWORDS, the streaming weight-window
size) grow with vocab_size at a synthesis-time-relevant rate -- pick a
vocab_size, check the resulting WWORDS against the deployed BRAM bucket
BEFORE training, not after.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np

MARKER = "<|endoftext|>"
TOKEN_RE = re.compile(r"[a-z']+|[.,!?;:\"-]")
UNK = "<unk>"
EOS = "<eos>"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def build_vocab(tokens: list[str], vocab_size: int) -> tuple[dict, dict]:
    """Reserve ids 0=<unk>, 1=<eos>; the rest are the (vocab_size-2) most
    frequent tokens, alphabetically ordered among themselves for determinism.

    `tokens` includes the EOS sentinel itself (prepare_word() appends one per
    story before calling this) -- EOS is by far the single most frequent
    "token" in that stream, so it MUST be excluded from the most_common()
    candidate pool, or it wins a regular top-word slot and silently
    overwrites its own reserved id=1 (found the hard way: a real checkpoint
    trained before this exclusion ended up with EOS at id=10 and a dead,
    never-trained id=1 -- harmless to training/inference since encode() only
    ever sees the final, self-consistent stoi, but wastes a vocab slot and
    breaks anything that assumes EOS==1 by convention instead of by lookup)."""
    freq = Counter(t for t in tokens if t != EOS)
    n_words = vocab_size - 2
    top = [w for w, _ in freq.most_common(n_words)]
    top.sort()
    stoi = {UNK: 0, EOS: 1}
    for i, w in enumerate(top):
        stoi[w] = i + 2
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos


def encode(tokens: list[str], stoi: dict) -> np.ndarray:
    unk_id = stoi[UNK]
    get = stoi.get
    ids = [get(t, unk_id) for t in tokens]
    return np.array(ids, dtype=np.uint16)


def prepare_word(corpus_path: str, out_dir: str, vocab_size: int,
                  val_frac: float = 0.01) -> dict:
    """Encode a corpus (same <|endoftext|>-delimited story format data.py's
    build_text() reads) to train.bin/val.bin + meta.json under out_dir,
    word-level ids instead of char-level. Skips re-encoding if the bins
    already exist and match this corpus path + vocab_size."""
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")

    if all(os.path.exists(p) for p in (meta_path, train_path, val_path)):
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get("source") == os.path.abspath(corpus_path)
                and meta.get("vocab_size") == vocab_size):
            return meta

    stories = []
    current = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == MARKER:
                if current:
                    stories.append(" ".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        stories.append(" ".join(current))

    # tokenize each story, inserting EOS between stories (mirrors data.py's
    # SEP.join(stories)+SEP, one boundary token per story instead of one
    # boundary char)
    all_tokens: list[str] = []
    for s in stories:
        all_tokens.extend(tokenize(s.lower()))
        all_tokens.append(EOS)

    stoi, itos = build_vocab(all_tokens, vocab_size)
    ids = encode(all_tokens, stoi)

    n_val = int(len(ids) * val_frac)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    unk_count = int((ids == stoi[UNK]).sum())
    meta = {
        "source": os.path.abspath(corpus_path),
        "vocab_size": vocab_size,
        "tokenizer": "word",
        "stoi": stoi,
        "itos": {str(i): w for i, w in itos.items()},
        "total_tokens": len(ids),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "unk_frac": unk_count / len(ids),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_meta(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "meta.json")) as f:
        return json.load(f)


def decode(ids, itos: dict) -> str:
    """Join word/punctuation tokens back into readable text: no space before
    punctuation, EOS becomes a newline, UNK becomes a literal marker."""
    itos = {int(k): v for k, v in itos.items()} if isinstance(next(iter(itos)), str) else itos
    out = []
    for i in ids:
        tok = itos[int(i)]
        if tok == EOS:
            out.append("\n")
            continue
        if tok == UNK:
            tok = "<unk>"
        if out and out[-1] not in ("\n", "") and not tok.startswith(("'",)) \
                and tok not in ".,!?;:\"-":
            out.append(" ")
        out.append(tok)
    return "".join(out)


if __name__ == "__main__":
    import sys

    corpus = sys.argv[1] if len(sys.argv) > 1 else "data/TinyStories-train.filtered.txt"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/word_stream16"
    vocab_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1900
    meta = prepare_word(corpus, out, vocab_size)
    print(f"vocab_size={meta['vocab_size']}  train_tokens={meta['train_tokens']:,}  "
          f"val_tokens={meta['val_tokens']:,}  unk_frac={meta['unk_frac']:.4f}")
