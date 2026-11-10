"""R1 free-running sanity (HANDOFF task 3): teacher forcing can hide drift, so let the
bit-true FABRIC spec write its own text and ask the FLOAT reference what it thinks of it.

Per chat prompt (factual / explanatory / creative):
  1. fabric greedy: reset, step() the prompt, then argmax + step() for up to --n tokens,
     stopping on <|im_end|> / <|endoftext|> (the stop token is kept as the last token)
  2. float forward_seq ONCE over prompt + fabric text. Wherever float's argmax is not the
     token the fabric chose is a divergence: reported with float's top-1, the float margin
     (float logit of its top-1 minus float logit of the fabric's token) and the fabric
     margin (the same two tokens under the fabric logits, other way round). Up to the first
     divergence the fabric text IS float greedy. Small margins = a near-tie, not damage.
     Also logged: logit cosine along the text and the float NLL of the fabric's text.
  3. degeneracy flags on the generated tokens:
       loop             some period-p pattern >= 3 times in a row covering >= 6 tokens
       few_unique       unique-token fraction < 0.30        (texts of >= 16 tokens)
       repeated_4grams  distinct 4-gram fraction < 0.50     (texts of >= 16 tokens)
       empty            stopped before writing anything

Float greedy from the prompt is NOT run by default: Qwen3Ref.greedy goes through step(),
which re-expands every matrix per token (~8 s/token on the laptop GPU, ~11 min per prompt
at n = 80), and step 2 already gives its first divergence. --float-greedy runs it.

Verdict: R1_FREERUN_VERDICT PASS if no prompt raises a degeneracy flag and every
divergence is a near-tie (float margin <= --margin, default 0.5 logits = float rates its
own choice under e^0.5 = 1.65x the fabric's). Otherwise REVIEW: a human reads the text.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m fabric.big.ref.r1_freerun
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .fabric_ref import FabricRef
from .qwen3_ref import FloatGemv, Qwen3Ref, from_gguf

FMT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPTS = [
    ("factual", "List the planets of the solar system in order from the Sun."),
    ("explain", "Why is the sky blue? Answer in three sentences."),
    ("creative", "Write a short poem about a robot learning to paint."),
]
LOOP_TOK, UNIQ_MIN, D4_MIN, MIN_LEN = 6, 0.30, 0.50, 16


def degeneracy(g):
    """crude loop detectors on a token list (see module doc)."""
    n = len(g)
    uniq = len(set(g)) / max(1, n)
    g4 = [tuple(g[i:i + 4]) for i in range(n - 3)]
    d4 = len(set(g4)) / len(g4) if g4 else 1.0
    loop = dict(period=0, reps=0.0, tokens=0)
    for p in range(1, n // 2 + 1):
        run = mx = 0
        for i in range(p, n):
            run = run + 1 if g[i] == g[i - p] else 0
            mx = max(mx, run)
        if mx >= 2 * p and mx + p > loop["tokens"]:         # period p, at least 3 times in a row
            loop = dict(period=p, reps=round((mx + p) / p, 2), tokens=mx + p)
    flags = (["empty"] if n == 0 else []) + (["loop"] if loop["tokens"] >= LOOP_TOK else [])
    if n >= MIN_LEN:
        flags += (["few_unique"] if uniq < UNIQ_MIN else []) + (["repeated_4grams"] if d4 < D4_MIN else [])
    return dict(n=n, unique_frac=uniq, distinct4_frac=d4, loop=loop, flags=flags)


def free(dev):
    gc.collect()
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()


def peak(dev):
    if not str(dev).startswith("cuda"):
        return ""
    s = f", peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f}/{torch.cuda.max_memory_reserved() / 2**30:.2f} GB"
    torch.cuda.reset_peak_memory_stats()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="data/big/Bonsai-8B-Q1_0.gguf")
    ap.add_argument("--tok", default="data/big/Bonsai-8B-meta")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=80, help="new tokens per prompt")
    ap.add_argument("--out", default="fabric/big/results/r1_freerun.json")
    ap.add_argument("--margin", type=float, default=0.5, help="near-tie float logit margin")
    ap.add_argument("--prompts", nargs="+", help="subset of: " + " ".join(n for n, _ in PROMPTS))
    ap.add_argument("--float-greedy", action="store_true", help="also run float greedy from the prompt (slow)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    dev = a.device
    t0 = time.time()
    cfg, W, _ = from_gguf(a.gguf, device=dev)
    flt = Qwen3Ref(cfg, W, device=dev, gemv=FloatGemv())          # shares W with fab
    fab = FabricRef(cfg, W, device=dev)
    print(f"loaded {Path(a.gguf).name} on {dev} ({time.time() - t0:.0f}s{peak(dev)})")
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")} - {None}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = dict(model=Path(a.gguf).name, device=dev, n=a.n, margin=a.margin, prompts={},
               float_greedy="run" if a.float_greedy else
               "skipped: Qwen3Ref.greedy uses step(), which re-expands every matrix per token (~8 s/token "
               "on the laptop GPU); first_div from the single float forward_seq is where it would diverge")
    ok = True
    for name, text in PROMPTS:
        if a.prompts and name not in a.prompts:
            continue
        ids = tok(FMT.format(text)).input_ids
        P = len(ids)
        # 1. fabric greedy, step path
        t0 = time.time()
        fab.reset()
        for t in ids:
            lg = fab.step(int(t))
        tp = time.time() - t0
        print(f"[{name}] prompt {P} tok, fabric prefill {tp:.0f}s ({tp / P:.2f} s/tok{peak(dev)})")
        t0, gen, qs = time.time(), [], []
        while True:
            qs.append(lg.cpu())
            gen.append(int(torch.argmax(lg)))
            if gen[-1] in stop or len(gen) >= a.n:
                break
            lg = fab.step(gen[-1])
            if len(gen) % 10 == 0:
                print(f"    [{name}] gen {len(gen)}/{a.n} ({time.time() - t0:.0f}s)")
        tg = time.time() - t0
        ended = "stop token" if gen[-1] in stop else "n reached"
        body = gen[:-1] if gen[-1] in stop else gen
        txt = tok.decode(body)
        print(f"[{name}] {len(gen)} tokens ({ended}), {tg:.0f}s{peak(dev)}")
        print(f"[{name}] Q: {text}")
        print(f"[{name}] A: {txt!r}")
        fab.reset()
        getattr(fab.gemv, "_dec", {}).clear()       # the scale cache (~1 GB on the 8B) is rebuilt in <1 s
        free(dev)
        # 2. float, once, over the fabric's own text
        t0 = time.time()
        f = flt.forward_seq(ids + gen).cpu().double()
        fq = f[P - 1:P - 1 + len(gen)]              # float logits at the positions that predict gen[j]
        Q = torch.stack(qs).double()                # fabric logits that chose gen[j]
        g = torch.tensor(gen)
        ft = fq.argmax(-1)
        cos = torch.nn.functional.cosine_similarity(fq, Q, dim=-1)
        divs = []
        for j in torch.nonzero(ft != g)[:, 0].tolist():
            fm, qm = (fq[j, ft[j]] - fq[j, gen[j]]).item(), (Q[j, gen[j]] - Q[j, ft[j]]).item()
            divs.append(dict(i=j, fabric_tok=gen[j], fabric_txt=tok.decode([gen[j]]), float_tok=int(ft[j]),
                             float_txt=tok.decode([int(ft[j])]), float_margin=fm, fabric_margin=qm,
                             near_tie=bool(fm <= a.margin)))
        r = dict(prompt=text, n_prompt=P, n_gen=len(gen), ended=ended, gen=gen, text=txt,
                 top1=1 - len(divs) / len(gen), n_div=len(divs), first_div=divs[0]["i"] if divs else None,
                 divergences=divs, cos_mean=cos.mean().item(), cos_min=cos.min().item(),
                 float_nll_of_text=torch.nn.functional.cross_entropy(fq, g).item(),
                 fabric_nll_of_text=torch.nn.functional.cross_entropy(Q, g).item(),
                 prefill_s=round(tp, 1), gen_s=round(tg, 1))
        print(f"[{name}] float agrees on {len(gen) - len(divs)}/{len(gen)} tokens, first divergence "
              f"{r['first_div']}, cos mean {r['cos_mean']:.6f} min {r['cos_min']:.5f}, float NLL of the text "
              f"{r['float_nll_of_text']:.3f} (fabric {r['fabric_nll_of_text']:.3f}) ({time.time() - t0:.0f}s{peak(dev)})")
        for d in divs:
            print(f"    @{d['i']:3d} fabric {d['fabric_txt']!r} vs float {d['float_txt']!r}: float margin "
                  f"{d['float_margin']:.3f}, fabric margin {d['fabric_margin']:.3f}"
                  f"{'  near-tie' if d['near_tie'] else '  NOT a near-tie'}")
        del f, fq, Q
        free(dev)
        if a.float_greedy:
            t0 = time.time()
            fg = flt.greedy(ids, len(gen))
            fd = next((i for i, (x, y) in enumerate(zip(fg, gen)) if x != y), None)
            r["float_greedy"] = dict(gen=fg, text=tok.decode(fg), first_diff=fd)
            print(f"[{name}] float greedy first differs at {fd} ({time.time() - t0:.0f}s): {tok.decode(fg)!r}")
        # 3. degeneracy
        r["degeneracy"] = dg = degeneracy(body)
        lp = dg["loop"]
        print(f"[{name}] degeneracy: unique {dg['unique_frac']:.2f}, distinct-4 {dg['distinct4_frac']:.2f}, longest loop "
              f"{lp['tokens']} tok (period {lp['period']} x{lp['reps']}) -> {dg['flags'] or 'ok'}")
        r["ok"] = bool(not dg["flags"] and all(d["near_tie"] for d in divs))
        ok = ok and r["ok"]
        res["prompts"][name] = r
        out.write_text(json.dumps(res, indent=1))               # partial, in case a later prompt dies
    res["verdict"] = "PASS" if ok and res["prompts"] else "REVIEW"
    out.write_text(json.dumps(res, indent=1))
    print(f"R1_FREERUN_VERDICT {res['verdict']}")


if __name__ == "__main__":
    main()
