"""Reports total parameter count and a per-component-group breakdown for
a training checkpoint (FP `ckpt.pt` or QAT `ckpt.qat.pt`).

Groups weights by which part of the architecture they belong to
(embeddings, transformer blocks, final layernorm, output head) rather
than listing every individual tensor -- this is what actually answers
"how big is the deployed model," since `tok_emb`/`head` dominate at
large VOCAB (see fabric/genesys2/PORT-NOTES.md's "Final architecture
reference" section, VOCAB=16384: tok_emb and head are each ~2.1M of
the ~6.57M total, comparable to all 12 transformer blocks combined).

    python -m model.count_params data/ckpt_word16384.pt
    python -m model.count_params data/ckpt_word16384.pt data/ckpt_word16384.qat.pt
"""
from __future__ import annotations

import argparse


def group_for_key(key: str) -> str:
    if "tok_emb" in key:
        return "tok_emb"
    if "pos_emb" in key:
        return "pos_emb"
    if key.startswith("lm_head") or key.split(".")[0] == "head":
        return "head"
    if (key.startswith("blocks") or key.startswith("transformer.h")
            or ".blocks." in key or key.startswith("h.")):
        return "blocks"
    if "ln_f" in key:
        return "ln_f"
    return "other:" + key


def count_params(state_dict) -> dict:
    groups: dict = {}
    for key, val in state_dict.items():
        if not hasattr(val, "numel"):
            continue
        g = group_for_key(key)
        groups[g] = groups.get(g, 0) + val.numel()
    return groups


def report(path: str):
    import torch  # deferred: only needed for this checkpoint-analysis tool

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    groups = count_params(sd)
    total = sum(groups.values())

    print(f"=== {path} ===")
    for g, n in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"{g:20s} {n:>10,} ({n / 1e6:.3f}M)")
    print(f"{'total':20s} {total:>10,} ({total / 1e6:.3f}M)")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="model.count_params")
    ap.add_argument("checkpoint", nargs="+", help="path(s) to a ckpt.pt / ckpt.qat.pt file")
    a = ap.parse_args(argv)

    for path in a.checkpoint:
        report(path)


if __name__ == "__main__":
    main()
