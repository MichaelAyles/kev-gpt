"""Reproduce SauravP97/tiny-stories-hf's exact training recipe (M config:
8 layers, hidden_size=256, GPT-Neo local attention, GPT-Neo-125M BPE
tokenizer, HF Trainer defaults) at a short diagnostic scale (max_steps,
not a full epoch) to check whether it shows the same Adam divergence
kev-gpt's own D=384 tests hit repeatedly, before doing an ablation
walking the recipe toward kev-gpt's own setup one change at a time."""
import json
import sys

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTNeoConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
OUT_STATES = sys.argv[2] if len(sys.argv) > 2 else "/tmp/repro_hf_states.jsonl"

print("loading dataset...")
dataset = load_dataset("roneneldan/TinyStories")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

print("tokenizing...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=8)

config = GPTNeoConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=512,
    hidden_size=256,
    num_layers=8,
    num_heads=16,
    attention_types=[[["local"], 8]],
)
model = AutoModelForCausalLM.from_config(config)
print(f"model parameters: {model.num_parameters() / 1_000_000:.2f}M")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

class JsonlLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        with open(OUT_STATES, "a") as f:
            f.write(json.dumps({"step": state.global_step, **logs}) + "\n")

training_args = TrainingArguments(
    output_dir="/tmp/repro-hf-model",
    max_steps=MAX_STEPS,
    per_device_train_batch_size=8,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="no",
    logging_steps=500,
    learning_rate=5e-4,
    weight_decay=0.01,
    fp16=True,
    report_to=[],
)

eval_subset = tokenized_datasets["validation"].select(range(min(500, len(tokenized_datasets["validation"]))))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=eval_subset,
    data_collator=data_collator,
    callbacks=[JsonlLogger()],
)

open(OUT_STATES, "w").close()
print(f"training for {MAX_STEPS} steps, logging to {OUT_STATES} ...")
trainer.train()
print("done.")
