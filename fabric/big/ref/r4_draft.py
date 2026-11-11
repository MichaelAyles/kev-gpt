"""R4 entry study: what would a prompt-lookup drafter get accepted? (doc 10, R4 stage 1)

Greedy speculative decoding emits exactly the greedy text, so the acceptance of ANY drafter
can be replayed from a recorded greedy transcript without running the model: a weight pass
takes the last emitted token plus k drafted tokens (K = k + 1 positions), and emits
1 + (number of leading drafts that equal the true continuation) tokens.

Drafter here = prompt lookup, no training and no weights: find the longest suffix of the
context (n-gram, n from --nmax down to --nmin) that also occurs earlier in the context, take
the most recent earlier occurrence, and propose the k tokens that followed it. No match = no
draft (the pass runs one position).

    python -m fabric.big.ref.r4_draft fabric/big/results/r4_transcripts.json [more.json ...]

Input files: {name: {"ids": [prompt ids], "gen": [greedy ids]}} (also read under a top-level
"prompts" key, which is what board/pl_token_c.py --out writes).

Output per (nmin, k): tokens per weight pass, the share of passes that carried a draft, and a
wall-clock model  ms/token = (W + H * positions) / tokens  where W is the weight stream per
pass and H the per-position vector cost. Both are arguments because the hybrid (H = the A53)
and a fabric vector unit (H small) sit at very different places. Everything printed is
DERIVED from the transcripts; only the transcripts are MEASURED.
"""
from __future__ import annotations

import argparse
import json
import sys


def draft(ctx, k, nmin, nmax):
    """k tokens following the most recent earlier occurrence of the longest matching suffix."""
    L = len(ctx)
    for n in range(min(nmax, L - 1), nmin - 1, -1):
        suf = ctx[L - n:]
        for s in range(L - n - 1, -1, -1):                  # most recent first; must end before L
            if ctx[s:s + n] == suf:
                out, p = [], s + n
                while len(out) < k:
                    out.append(ctx[p] if p < L else out[p - L])   # runs off the end: keep copying
                    p += 1
                return out, n
    return [], 0


def replay(ids, gen, k, nmin, nmax, fixed_k=False):
    """-> dict(tokens, passes, positions, drafted_passes, hist[a] = passes with a accepted)."""
    ctx = list(ids) + [gen[0]]                              # gen[0] comes from the prefill pass
    i, passes, positions, drafted = 1, 0, 0, 0
    hist = [0] * (k + 1)
    while i < len(gen):
        d, _ = draft(ctx, k, nmin, nmax) if k else ([], 0)
        a = 0
        while a < len(d) and i + a < len(gen) and d[a] == gen[i + a]:
            a += 1
        emit = min(a + 1, len(gen) - i)                     # a accepted drafts + the model's own next token
        a = min(a, emit)
        passes += 1
        positions += (k + 1) if fixed_k else (len(d) + 1)
        drafted += bool(d)
        hist[a] += 1
        ctx += gen[i:i + emit]
        i += emit
    return dict(tokens=len(gen) - 1, passes=passes, positions=positions, drafted=drafted, hist=hist)


def load(paths):
    jobs = {}
    for p in paths:
        d = json.load(open(p))
        d = d.get("prompts", d)
        for name, j in d.items():
            if "ids" in j and "gen" in j and len(j["gen"]) > 1:
                jobs[f"{name}@{p.split('/')[-1]}" if name in jobs else name] = (list(j["ids"]), list(j["gen"]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 5, 7])
    ap.add_argument("--nmin", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--w-ms", type=float, default=72.0, help="weight stream + KV per pass")
    ap.add_argument("--h-ms", type=float, nargs="+", default=[100.0, 30.0, 10.0, 2.0],
                    help="vector-path cost per position per pass (hybrid ~100 untuned; fabric ~1-2)")
    ap.add_argument("--fixed-k", action="store_true", help="every pass costs k + 1 positions, draft or not")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    jobs = load(a.files)
    if not jobs:
        sys.exit("no transcripts with ids + gen")
    ntok = sum(len(g) - 1 for _, g in jobs.values())
    print(f"R4_DRAFT {len(jobs)} transcripts, {ntok} generated tokens (after the prefill token); nmax {a.nmax}")
    rows = []
    for nmin in a.nmin:
        for k in a.k:
            tot = dict(tokens=0, passes=0, positions=0, drafted=0, hist=[0] * (k + 1))
            per = {}
            for name, (ids, gen) in jobs.items():
                r = replay(ids, gen, k, nmin, a.nmax, a.fixed_k)
                per[name] = r["tokens"] / r["passes"]
                for key in ("tokens", "passes", "positions", "drafted"):
                    tot[key] += r[key]
                tot["hist"] = [x + y for x, y in zip(tot["hist"], r["hist"])]
            tpp = tot["tokens"] / tot["passes"]
            ms = {h: (a.w_ms * tot["passes"] + h * tot["positions"]) / tot["tokens"] for h in a.h_ms}
            base = {h: a.w_ms + h for h in a.h_ms}
            rows.append(dict(nmin=nmin, k=k, tokens_per_pass=tpp, drafted_frac=tot["drafted"] / tot["passes"],
                             positions_per_pass=tot["positions"] / tot["passes"], hist=tot["hist"], per=per,
                             ms_per_token=ms, speedup={h: base[h] / ms[h] for h in a.h_ms}))
            print(f"  nmin {nmin} k {k}: {tpp:.3f} tokens/pass (per transcript {min(per.values()):.2f}..{max(per.values()):.2f}), "
                  f"draft on {100 * tot['drafted'] / tot['passes']:.0f}% of passes, {tot['positions'] / tot['passes']:.2f} positions/pass; "
                  + "; ".join(f"H={h:g}: {ms[h]:.0f} ms/tok = {1e3 / ms[h]:.1f} tok/s ({base[h] / ms[h]:.2f}x)" for h in a.h_ms))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(n_transcripts=len(jobs), tokens=ntok, nmax=a.nmax, w_ms=a.w_ms, fixed_k=a.fixed_k, rows=rows), f)


if __name__ == "__main__":
    main()
