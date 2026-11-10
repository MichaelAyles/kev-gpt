"""R1c gate: the bit-true FABRIC spec (fabric_ref.py) vs the FLOAT reference on Bonsai-8B.

  1. step path == seq path bit-identical (the spec is causal and deterministic, so the RTL
     can be gated per token)
  2. teacher-forced on 4 texts (TinyStories, README, doc 0, float-model chat answer):
     top-1 agreement >= 0.99, dNLL logged, logit cosine logged
  3. GEMV accumulator stats over the whole run (shift range, floor-truncated terms,
     peak accumulator bits) — sizes the RTL accumulator

    python -u -m fabric.big.ref.r1c_gate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .fabric_ref import FabricRef
from .qwen3_ref import FloatGemv, Qwen3Ref, from_gguf
from .r1b_sweep import eval_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="data/big/Bonsai-8B-Q1_0.gguf")
    ap.add_argument("--tok", default="data/big/Bonsai-8B-meta")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--out", default="fabric/big/results/r1c_gate.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    cfg, W, _ = from_gguf(a.gguf, device="cuda")
    flt = Qwen3Ref(cfg, W, device="cuda", gemv=FloatGemv())
    fab = FabricRef(cfg, W, device="cuda")
    res = {"model": Path(a.gguf).name}

    # 1. step == seq
    probe = tok("<|im_start|>user\nHello! Who are you?<|im_end|>\n<|im_start|>assistant\n").input_ids
    t0 = time.time()
    s_seq = fab.forward_seq(probe)
    fab.reset()
    s_step = torch.stack([fab.step(t) for t in probe])
    res["step_eq_seq"] = bool(torch.equal(s_seq, s_step))
    print(f"[1] step==seq bit-identical: {res['step_eq_seq']} ({len(probe)} tok, {time.time() - t0:.0f}s)")

    # 2. teacher-forced fidelity
    texts, chat = eval_texts(tok, a.T)
    ids = tok(chat).input_ids
    cache = Path(a.out).with_name(Path(a.gguf).stem + f"_chatgen_T{a.T}.json")
    if cache.exists():
        gen = json.loads(cache.read_text())
    else:
        t0 = time.time()
        gen = flt.greedy(ids, a.T - len(ids))
        cache.write_text(json.dumps(gen))
        print(f"    chat text generated in {time.time() - t0:.0f}s")
    texts.append(("chat_selfgen", ids + gen))
    print("    chat:", repr(tok.decode(gen[:48])))
    per = {}
    tot = dict(dn=0.0, ag=0, cos=0.0, n=0, fnll=0.0)
    for name, t in texts:
        t0 = time.time()
        f = flt.forward_seq(t).double().cpu()
        q = fab.forward_seq(t).double().cpu()
        y = torch.tensor(t[1:])
        nf = torch.nn.functional.cross_entropy(f[:-1], y, reduction="sum").item()
        nq = torch.nn.functional.cross_entropy(q[:-1], y, reduction="sum").item()
        ag = (f[:-1].argmax(-1) == q[:-1].argmax(-1)).sum().item()
        cs = torch.nn.functional.cosine_similarity(f[:-1], q[:-1], dim=-1)
        n = len(y)
        per[name] = dict(n=n, top1=ag / n, dnll=(nq - nf) / n, cos_mean=cs.mean().item(),
                         cos_min=cs.min().item(), float_nll=nf / n)
        for k, v in (("dn", nq - nf), ("ag", ag), ("cos", cs.sum().item()), ("n", n), ("fnll", nf)):
            tot[k] += v
        print(f"[2] {name:13s} n={n} top1 {ag / n:.4f} dNLL {(nq - nf) / n:+.4f} "
              f"cos mean {cs.mean():.6f} min {cs.min():.5f} ({time.time() - t0:.0f}s)")
    st = fab.gemv.stats
    res.update(per_text=per, top1=tot["ag"] / tot["n"], dnll=tot["dn"] / tot["n"],
               cos=tot["cos"] / tot["n"], ntok=tot["n"], float_nll=tot["fnll"] / tot["n"],
               gemv=dict(min_shift=st.min_shift, max_shift=st.max_shift,
                         floor_frac=st.floor_terms / max(1, st.terms), max_acc_bits=st.max_acc_bits))
    ok = res["step_eq_seq"] and res["top1"] >= 0.99
    res["verdict"] = "PASS" if ok else "FAIL"
    print(f"[2] pooled n={tot['n']} top1 {res['top1']:.4f} dNLL {res['dnll']:+.4f} cos {res['cos']:.6f}")
    print(f"[3] GEMV shifts [{st.min_shift}..{st.max_shift}] floor-terms {res['gemv']['floor_frac']:.2e} "
          f"peak acc bits {st.max_acc_bits}/64")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"R1C_VERDICT {res['verdict']}")


if __name__ == "__main__":
    main()
