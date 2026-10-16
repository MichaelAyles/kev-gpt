"""Generate Kevin-speak from any saved checkpoint.

Audition snapshots to pick a rollback point by taste, not just by val loss:

    python -m model.sample data/ckpts/ckpt_02000.pt -n 300
    python -m model.sample data/ckpt.pt --prompt "once upon time" -k 3
"""

from __future__ import annotations

import argparse

import torch

from .gpt import GPT, GPTConfig
from .train import pick_device


def load(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["meta"], ck.get("iter"), ck.get("val")


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.sample", description="Sample from a checkpoint.")
    p.add_argument("ckpt", help="path to a *.pt checkpoint")
    p.add_argument("--prompt", default="\n", help="seed text")
    p.add_argument("-n", "--n-tokens", type=int, default=300)
    p.add_argument("-t", "--temperature", type=float, default=0.8)
    p.add_argument("-k", "--top-k", type=int, default=40)
    p.add_argument("--count", type=int, default=1, help="how many samples to draw")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    device = pick_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, meta, it, val = load(args.ckpt, device)
    stoi, itos = meta["stoi"], meta["itos"]
    print(f"# {args.ckpt}  (iter {it}, val {val:.3f})\n")

    # Dispatch to the word-level tokeniser (whole-word tokens, space-joined
    # decode) or char-level (from .data, one id per character) based on how
    # this checkpoint was trained -- mirrors model.train.sample()'s dispatch.
    if meta.get("tokenizer") == "word":
        from .word_data import decode, tokenize, UNK
        prompt_ids = [stoi.get(t, stoi[UNK]) for t in (tokenize(args.prompt.lower()) or [UNK])]
    else:
        from .data import decode
        prompt_ids = [stoi.get(c, 0) for c in args.prompt]

    for _ in range(args.count):
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(ids, args.n_tokens, temperature=args.temperature,
                             top_k=args.top_k)[0].tolist()
        print(decode(out, itos))
        print("-" * 60)


if __name__ == "__main__":
    main()
