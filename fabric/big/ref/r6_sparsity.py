"""R6 entry experiment: how many FFN hidden units can a token skip? (doc 10, R6 "contextual sparsity")

tok/s = DDR bandwidth / bytes read per token, and the FFN is ~72% of the bytes (gate, up and
down at ~24% each). A hidden unit that contributes nothing to this token need not be read:
its row of `up` and its column of `down` can be skipped once `gate` (read in full) has said
so. This measures, on the float reference, what skipping costs in quality. No predictor and no
training: the model is the published one, only the per-token choice of units changes.

Rules (per position, per layer, keep the top `keep` fraction of the 12,288 units by score):
  gate   score = |silu(gate)|          skips rows of `up` AND columns of `down`  (R6b)
  prod   score = |silu(gate) * up|     skips columns of `down` only (needs `up` in full)
Skipped units are set to exactly zero in the input of `down`, which is what not reading them
does. Bytes saved per token (DERIVED, FFN matrices ~24% of the stream each):
  gate: (1 - keep) * 48%      prod: (1 - keep) * 24%

Reported against the dense float model on the R1c texts: top-1 agreement with the dense
argmax, and the change in NLL of the real next token. This is the float model; the bit-true
spec would need its own run once a rule is chosen.

    flock /tmp/kevbuild/gpu.lock env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python -u -m fabric.big.ref.r6_sparsity
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .qwen3_ref import FloatGemv, Qwen3Ref, from_gguf
from .r1b_sweep import eval_texts

FFN_SHARE = 0.24                                    # of the bytes per token, each of gate / up / down


class SparseGemv(FloatGemv):
    """FloatGemv that zeroes all but the top-`keep` hidden units in the input of every `down`."""

    def __init__(self):
        self.rule, self.keep, self.gate = None, 1.0, None
        self.kept_mass = []                         # share of sum |x| that survives, per layer call

    def mm(self, W, X, name=""):
        if self.rule and name.endswith("gate"):
            Y = super().mm(W, X, name)
            self.gate = Y
            return Y
        if self.rule and name.endswith("down"):
            score = (self.gate * torch.sigmoid(self.gate)).abs() if self.rule == "gate" else X.abs()
            n = score.shape[1]
            k = max(1, int(round(self.keep * n)))
            thr = score.kthvalue(n - k + 1, dim=1, keepdim=True).values
            m = score >= thr
            self.kept_mass.append(float((X.abs() * m).sum() / X.abs().sum().clamp_min(1e-30)))
            X = X * m
            self.gate = None
        return super().mm(W, X, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="data/big/Bonsai-8B-Q1_0.gguf")
    ap.add_argument("--tok", default="data/big/Bonsai-8B-meta")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--chat-cache", default="fabric/big/results/Bonsai-8B-Q1_0_chatgen_T256.json")
    ap.add_argument("--gate-keep", type=float, nargs="*", default=[0.75, 0.5, 0.35, 0.25, 0.15])
    ap.add_argument("--prod-keep", type=float, nargs="*", default=[0.5, 0.25, 0.1])
    ap.add_argument("--out", default="fabric/big/results/r6_sparsity.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    cfg, W, _ = from_gguf(a.gguf, device="cuda")
    g = SparseGemv()
    m = Qwen3Ref(cfg, W, device="cuda", gemv=g)
    texts, chat = eval_texts(tok, a.T)
    if Path(a.chat_cache).exists():
        texts.append(("chat_selfgen", (tok(chat).input_ids + json.loads(Path(a.chat_cache).read_text()))[:a.T]))

    def run(t):
        with torch.no_grad():
            return m.forward_seq(t).double().cpu()

    base = {}
    for name, t in texts:
        t0 = time.time()
        f = run(t)
        y = torch.tensor(t[1:])
        base[name] = dict(logits=f, y=y, nll=torch.nn.functional.cross_entropy(f[:-1], y, reduction="sum").item())
        print(f"dense {name}: {len(t)} tokens, NLL {base[name]['nll'] / len(y):.4f}, {time.time() - t0:.0f}s", flush=True)

    rows = []
    for rule, keeps in (("gate", a.gate_keep), ("prod", a.prod_keep)):
        for keep in keeps:
            g.rule, g.keep, g.kept_mass = rule, keep, []
            tot, per = dict(n=0, ag=0, dn=0.0), {}
            t0 = time.time()
            for name, t in texts:
                q = run(t)
                b = base[name]
                n = len(b["y"])
                nq = torch.nn.functional.cross_entropy(q[:-1], b["y"], reduction="sum").item()
                ag = int((q[:-1].argmax(-1) == b["logits"][:-1].argmax(-1)).sum())
                per[name] = dict(n=n, top1=ag / n, dnll=(nq - b["nll"]) / n)
                tot["n"] += n
                tot["ag"] += ag
                tot["dn"] += nq - b["nll"]
            L = cfg.n_layer
            mass = [sum(g.kept_mass[i::L]) / len(g.kept_mass[i::L]) for i in range(L)]
            saved = (1 - keep) * FFN_SHARE * (2 if rule == "gate" else 1)
            r = dict(rule=rule, keep=keep, bytes_saved=saved, speedup=1 / (1 - saved), top1=tot["ag"] / tot["n"],
                     dnll=tot["dn"] / tot["n"], n=tot["n"], per_text=per, kept_mass_by_layer=mass)
            rows.append(r)
            print(f"R6 {rule} keep {keep:.2f}: bytes -{100 * saved:.1f}% ({r['speedup']:.2f}x, DERIVED), top-1 vs dense {100 * r['top1']:.2f}%, "
                  f"dNLL {r['dnll']:+.4f} (per text " + ", ".join(f"{k} {v['dnll']:+.3f}" for k, v in per.items())
                  + f"), kept |x| mass layers 0/17/35: {mass[0]:.2f}/{mass[L // 2]:.2f}/{mass[-1]:.2f}, {time.time() - t0:.0f}s", flush=True)
            Path(a.out).write_text(json.dumps(dict(model=Path(a.gguf).name, T=a.T, ffn_share=FFN_SHARE,
                                                   dense={k: v["nll"] / len(v["y"]) for k, v in base.items()}, rows=rows)))
    g.rule = None


if __name__ == "__main__":
    main()
