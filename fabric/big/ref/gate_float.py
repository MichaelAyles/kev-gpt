"""R1a gate: our from-scratch Qwen3 forward == HF transformers, same weights.

Both run fp32 on CPU over the identical unpacked (bf16) weights. Passes when every
position's logits agree to cosine > 0.99999 and argmax matches, and a greedy
continuation is token-identical.

    python -m fabric.big.ref.gate_float --model data/big/Bonsai-1.7B-unpacked
"""
from __future__ import annotations

import argparse

import torch

from .qwen3_ref import Qwen3Ref, from_unpacked

PROMPTS = [
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "Once upon a time, a little robot lived in a garden. One day",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/big/Bonsai-1.7B-unpacked")
    ap.add_argument("--gen", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(8)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg, W, sd = from_unpacked(a.model, device="cpu", dtype=torch.float32)
    del sd
    ref = Qwen3Ref(cfg, W, device="cpu")
    print(f"cfg {cfg}")
    ok = True
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            hl = hf(ids).logits[0]
        ref.reset()
        mine = torch.stack([ref.step(int(t)) for t in ids[0]])
        cos = torch.nn.functional.cosine_similarity(hl, mine, dim=-1)
        top = (hl.argmax(-1) == mine.argmax(-1)).float().mean().item()
        mad = (hl - mine).abs().max().item()
        with torch.no_grad():
            hg = hf.generate(ids, max_new_tokens=a.gen, do_sample=False)[0, ids.shape[1]:].tolist()
        mg = ref.greedy(ids[0].tolist(), a.gen)
        good = cos.min().item() > 0.99999 and top == 1.0 and hg == mg
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} T={ids.shape[1]} cos_min={cos.min().item():.7f} "
              f"top1={top:.3f} max|dlogit|={mad:.2e} greedy_equal={hg == mg}")
        print("   HF :", repr(tok.decode(hg)))
        print("   ref:", repr(tok.decode(mg)))
    print(f"R1A_FLOAT_VERDICT {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
