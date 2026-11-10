"""R1b: rank activation-quantisation schemes for the 1-bit GEMV against the float
reference, teacher-forced on real text.

Metrics per scheme (pooled over all eval tokens):
  dNLL     mean next-token NLL change vs float (nats/token; + is worse)
  top1     argmax agreement with float
  cos      mean logit cosine (float64)

    python -m fabric.big.ref.r1b_sweep --gguf data/big/Bonsai-8B-Q1_0.gguf --T 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .actquant import QuantGemv
from .qwen3_ref import FloatGemv, Qwen3Ref, from_gguf

ROOT = Path(__file__).resolve().parents[3]


def eval_texts(tok, T):
    srcs = []
    ts = (ROOT / "data/TinyStories-valid.txt").read_text(errors="ignore")[20000:40000]
    srcs.append(("tinystories", ts))
    srcs.append(("readme", (ROOT / "README.md").read_text(errors="ignore")[2000:30000]))
    srcs.append(("master_doc", (ROOT / "docs/0-master.md").read_text(errors="ignore")[1000:30000]))
    chat = ("<|im_start|>user\nCan you explain how a transformer language model generates text, "
            "and why decoding one token at a time is limited by memory bandwidth?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return [(n, tok(s).input_ids[:T]) for n, s in srcs], chat


SCHEMES = [
    ("int", 8, 128), ("bfp", 8, 128), ("bfp", 8, 32), ("bfp", 6, 128), ("bfp", 10, 128),
    ("bfp", 12, 128), ("int", 6, 128), ("vec", 8, 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="data/big/Bonsai-8B-Q1_0.gguf")
    ap.add_argument("--tok", default="data/big/Bonsai-8B-meta")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--out", default="fabric/big/results/r1b_sweep.jsonl")
    ap.add_argument("--no-chat", action="store_true", help="skip the slow self-generated chat text")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    cfg, W, _ = from_gguf(a.gguf, device="cuda")
    ref = Qwen3Ref(cfg, W, device="cuda")
    texts, chat = eval_texts(tok, a.T)

    # chat continuation written by the float model itself (on-distribution text)
    ids = tok(chat).input_ids if not a.no_chat else None
    cache = Path(a.out).with_name(Path(a.gguf).stem + f"_chatgen_T{a.T}.json")
    if a.no_chat:
        gen = None
    elif cache.exists():
        gen = json.loads(cache.read_text())
    else:
        gen = ref.greedy(ids, a.T - len(ids))
        cache.write_text(json.dumps(gen))
    if gen is not None:
        texts.append(("chat_selfgen", ids + gen))
        print("chat sample:", repr(tok.decode(gen[:60])))

    base = {}
    ref.gemv = FloatGemv()
    for n, t in texts:
        base[n] = ref.forward_seq(t).double().cpu()

    def score(lg_all):
        nll_d, agree, cos, cnt = 0.0, 0.0, 0.0, 0
        for n, t in texts:
            f, q = base[n][:-1], lg_all[n][:-1]
            y = torch.tensor(t[1:])
            nf = torch.nn.functional.cross_entropy(f, y, reduction="sum")
            nq = torch.nn.functional.cross_entropy(q, y, reduction="sum")
            nll_d += (nq - nf).item()
            agree += (f.argmax(-1) == q.argmax(-1)).sum().item()
            cos += torch.nn.functional.cosine_similarity(f, q, dim=-1).sum().item()
            cnt += len(y)
        return nll_d / cnt, agree / cnt, cos / cnt, cnt

    fnll = sum(torch.nn.functional.cross_entropy(base[n][:-1], torch.tensor(t[1:]),
                                                 reduction="sum").item() for n, t in texts)
    ntok = sum(len(t) - 1 for _, t in texts)
    print(f"float NLL {fnll / ntok:.4f} nats/token over {ntok} tokens ({len(texts)} texts)")
    with open(a.out, "a") as fo:
        for kind, bits, grp in SCHEMES:
            ref.gemv = QuantGemv(kind, bits, grp or 128)
            t0 = time.time()
            lg = {n: ref.forward_seq(t).double().cpu() for n, t in texts}
            d, ag, cs, cnt = score(lg)
            row = dict(model=Path(a.gguf).name, scheme=repr(ref.gemv), dnll=d, top1=ag, cos=cs,
                       ntok=cnt, float_nll=fnll / ntok)
            fo.write(json.dumps(row) + "\n")
            print(f"{repr(ref.gemv):14s} dNLL {d:+.4f}  top1 {ag:.4f}  cos {cs:.6f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
