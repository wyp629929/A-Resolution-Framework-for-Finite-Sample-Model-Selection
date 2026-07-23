#!/usr/bin/env python3
"""Quick loss sanity check on 50 samples, full sequence (no masking)."""
import json, torch, numpy as np, math, sys, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from scipy.stats import norm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading...")
base = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/.hf/modelscope/Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "/root/paas/results/checkpoints/checkpoint-100")
tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/.hf/modelscope/Qwen/Qwen2.5-7B-Instruct")
tok.pad_token = tok.eos_token

with open("/root/paas/data/gsm8k_test.json") as f:
    data = json.load(f)[:50]

losses = []
for i, item in enumerate(data):
    ans = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else item["answer"].strip()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": item["question"]}],
        tokenize=False, add_generation_prompt=True)
    enc = tok(prompt + " " + ans, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(enc["input_ids"], labels=enc["input_ids"])
    losses.append(out.loss.item())

arr = np.array(losses)
var = float(np.var(arr, ddof=1))
dmin = float(norm.ppf(0.975) * math.sqrt(var / len(arr)))
print(f"Mean: {arr.mean():.4f}, Var: {var:.4f}, Dmin: {dmin:.4f}")
print(f"Percentiles: p25={np.percentile(arr,25):.4f} p50={np.percentile(arr,50):.4f} p75={np.percentile(arr,75):.4f}")
