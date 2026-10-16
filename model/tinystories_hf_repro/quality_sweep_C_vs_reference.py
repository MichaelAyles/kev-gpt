"""Reproduce SCALE-UP-LOG.md's "Closing quality check" (Sweep 1 + Sweep 2
methodology) fresh: kev-gpt checkpoint C (Step 10's fully-trained-without-
divergence D=384/n_layer=12/n_head=6 checkpoint, the practical payoff of
the ten-step ablation walk) vs the published SauravP97/tiny-stories-19M
reference, same objective detector (model.filter_synth_corpus.is_degenerate),
each model sampled with its own established convention.

Sweep 1: 25 samples each (5 seeds x 5 prompts), max_new_tokens=60 -- the
detector is discriminating (not saturated) at this length.
Sweep 2: 12 samples each (3 seeds x 4 prompts), max_new_tokens=250 -- longer
horizon, closer to real usage; detector saturates here per the original
investigation, so this is printed in full for a qualitative read too.
"""
import sys
sys.path.insert(0, "/home/tparng/kev-gpt")
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.gpt import GPT, GPTConfig
from model.word_data import load_meta, decode as kev_decode, tokenize as kev_tokenize, UNK
from model.filter_synth_corpus import is_degenerate

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- kev-gpt checkpoint C ----
CKPT_C = "data/ckpt_stepC_d384_v16384_fp.pt"
meta = load_meta("data/word_v16384")
stoi, itos = meta["stoi"], meta["itos"]
cfg = GPTConfig(block_size=128, vocab_size=meta["vocab_size"], n_layer=12, n_head=6, n_embd=384, dropout=0.0)
kev_model = GPT(cfg).to(device)
ck = torch.load(CKPT_C, map_location=device, weights_only=False)
kev_model.load_state_dict(ck["model"])
kev_model.eval()
print(f"loaded {CKPT_C} (iter {ck.get('iter')}), {kev_model.num_params()/1e6:.2f}M params, device={device}")

# ---- reference ----
REF_ID = "SauravP97/tiny-stories-19M"
ref_tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
ref_tok.pad_token = ref_tok.eos_token
ref_model = AutoModelForCausalLM.from_pretrained(REF_ID).to(device)
ref_model.eval()
print(f"loaded {REF_ID}, {sum(p.numel() for p in ref_model.parameters())/1e6:.1f}M params, device={device}")


def gen_kev(prompt, seed, n_tokens, temperature=0.8, top_k=40):
    torch.manual_seed(seed)
    prompt_ids = [stoi.get(t, stoi[UNK]) for t in (kev_tokenize(prompt.lower()) or [UNK])]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = kev_model.generate(ids, n_tokens, temperature=temperature, top_k=top_k)[0].tolist()
    return kev_decode(out, itos).replace("\n", " ").strip()


def gen_ref(prompt, seed, n_tokens, temperature=0.7, top_k=50):
    torch.manual_seed(seed)
    inputs = ref_tok(prompt, return_tensors="pt").to(device)
    out = ref_model.generate(inputs.input_ids, max_new_tokens=n_tokens, do_sample=True,
                              temperature=temperature, top_k=top_k, pad_token_id=ref_tok.eos_token_id)
    return ref_tok.decode(out[0], skip_special_tokens=True)


def run_sweep(name, gen_fn, prompts, seeds, n_tokens, verbose=True):
    flagged = 0
    total = 0
    samples = []
    for seed in seeds:
        for prompt in prompts:
            text = gen_fn(prompt, seed, n_tokens)
            reason = is_degenerate(text)
            total += 1
            if reason:
                flagged += 1
            samples.append({"seed": seed, "prompt": prompt, "text": text, "reason": reason})
            if verbose:
                tag = f"[FLAG: {reason}]" if reason else "[clean]"
                print(f"=== {name} seed={seed} prompt={prompt!r} {tag} ===")
                print(text)
                print()
    print(f"--- {name} SUMMARY: {flagged}/{total} flagged ---\n")
    return flagged, total, samples


PROMPTS_KEV = ["once upon a time", "the sun was", "the dog ran", "a little girl", "she found a"]
PROMPTS_REF = ["Once upon a time", "The sun was", "The dog ran", "A little girl", "She found a"]
SEEDS5 = [1, 2, 3, 4, 5]

print("\n########## SWEEP 1: 25 samples each, max_new_tokens=60 ##########\n")
c_f1, c_t1, c_s1 = run_sweep("C(kev-gpt)", gen_kev, PROMPTS_KEV, SEEDS5, 60)
r_f1, r_t1, r_s1 = run_sweep("reference", gen_ref, PROMPTS_REF, SEEDS5, 60)

PROMPTS_KEV_L = ["once upon a time", "the sun was", "the dog ran", "a little girl"]
PROMPTS_REF_L = ["Once upon a time", "The sun was", "The dog ran", "A little girl"]
SEEDS3 = [1, 2, 3]

print("\n########## SWEEP 2: 12 samples each, max_new_tokens=250 ##########\n")
c_f2, c_t2, c_s2 = run_sweep("C(kev-gpt)", gen_kev, PROMPTS_KEV_L, SEEDS3, 250)
r_f2, r_t2, r_s2 = run_sweep("reference", gen_ref, PROMPTS_REF_L, SEEDS3, 250)

print("=" * 70)
print("FINAL SUMMARY")
print(f"Sweep 1 (n=60 tok):  C={c_f1}/{c_t1} flagged   reference={r_f1}/{r_t1} flagged")
print(f"Sweep 2 (n=250 tok): C={c_f2}/{c_t2} flagged   reference={r_f2}/{r_t2} flagged")
print("=" * 70)

RESULTS_PATH = "/home/tparng/kev-gpt/model/tinystories_hf_repro/quality_sweep_results.json"
with open(RESULTS_PATH, "w") as f:
    json.dump({
        "sweep1": {"C": c_s1, "reference": r_s1, "C_flagged": c_f1, "C_total": c_t1,
                   "ref_flagged": r_f1, "ref_total": r_t1},
        "sweep2": {"C": c_s2, "reference": r_s2, "C_flagged": c_f2, "C_total": c_t2,
                   "ref_flagged": r_f2, "ref_total": r_t2},
    }, f, indent=2)
print(f"full results -> {RESULTS_PATH}")
