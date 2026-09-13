import sys
sys.path.insert(0, "/home/tparng/kev-gpt")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.filter_synth_corpus import is_degenerate

MODEL_ID = "SauravP97/tiny-stories-19M"
prompts = ["Once upon a time", "The sun was", "The dog ran", "A little girl", "She found a"]
seeds = [1, 2, 3, 4, 5]

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"loaded {MODEL_ID}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, device={device}")

flagged = 0
total = 0
for seed in seeds:
    for prompt in prompts:
        torch.manual_seed(seed)
        inputs = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0], skip_special_tokens=True)
        reason = is_degenerate(text)
        total += 1
        if reason:
            flagged += 1
        print(f"=== seed={seed} prompt={prompt!r} {'[FLAG: '+reason+']' if reason else '[clean]'} ===")
        print(text)
        print()

print(f"--- SUMMARY: {flagged}/{total} flagged (SauravP97/tiny-stories-19M, same objective detector used on kev-gpt checkpoints) ---")
