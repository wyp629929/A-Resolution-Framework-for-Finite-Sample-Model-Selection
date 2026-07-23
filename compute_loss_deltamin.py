#!/usr/bin/env python3
"""Compute per-sample cross-entropy loss on answer tokens only."""

import json, torch, numpy as np, os, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from scipy.stats import norm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_MODEL = "/root/autodl-tmp/.hf/modelscope/Qwen/Qwen2.5-7B-Instruct"
CKPT_PATH = "/root/paas/results/checkpoints/checkpoint-100"
DATA_PATH = "/root/paas/data/gsm8k_test.json"
MAX_SAMPLES = 200

print(f"Loading {BASE_MODEL}...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, CKPT_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded.")

with open(DATA_PATH) as f:
    data = json.load(f)[:MAX_SAMPLES]
print(f"Loaded {len(data)} examples.")

per_sample_losses = []
model.eval()

for i, item in enumerate(data):
    question = item["question"]
    answer = item["answer"]
    final_answer = answer.split("####")[-1].strip() if "####" in answer else answer.strip()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    full_text = prompt + " " + final_answer

    enc = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = prompt_enc["input_ids"].shape[1]

    # Create labels with -100 for prompt tokens (ignored in loss)
    labels = enc["input_ids"].clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(enc["input_ids"], labels=labels)

    # outputs.loss is now the average per-token loss on answer tokens only
    per_sample_losses.append(outputs.loss.item())

    if (i+1) % 50 == 0:
        print(f"  [{i+1}/{len(data)}] mean loss: {np.mean(per_sample_losses):.4f}")

losses = np.array(per_sample_losses)
var_loss = np.var(losses, ddof=1)
z = norm.ppf(0.975)
delta_min_loss = z * math.sqrt(var_loss / len(losses))

print(f"\n=== Results (N={len(losses)}) ===")
print(f"Mean per-sample loss: {np.mean(losses):.4f}")
print(f"Variance: {var_loss:.6f}")
print(f"Std: {np.std(losses, ddof=1):.4f}")
print(f"Δ_min (loss): {delta_min_loss:.4f}")

delta_min_acc = z * math.sqrt(2 * 0.5 * 0.5 / 200) * 100
print(f"\nFor reference: Δ_min (accuracy at p̄=0.5, N=200) = {delta_min_acc:.2f} pp")

print(f"\nLoss variance: {var_loss:.4f}")
print(f"Accuracy Bernoulli variance (max): 0.2500")
print(f"Ratio: {var_loss:.2f}x of Bernoulli variance")

json.dump({"N": len(losses), "mean_loss": float(np.mean(losses)),
           "variance": var_loss, "std": float(np.std(losses, ddof=1)),
           "delta_min_loss": delta_min_loss},
          open("/root/loss_deltamin_results2.json", "w"), indent=2)
print(f"\nSaved.")
