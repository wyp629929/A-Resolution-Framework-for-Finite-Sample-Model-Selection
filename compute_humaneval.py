#!/usr/bin/env python3
"""Evaluate LoRA fine-tuned model on HumanEval (pass@1)."""

import json, torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from human_eval.data import read_problems
from human_eval.execution import check_correctness

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_MODEL = "/root/autodl-tmp/.hf/modelscope/Qwen/Qwen2.5-7B-Instruct"
CKPT_PATH = "/root/paas/results/checkpoints/checkpoint-500"
MAX_SAMPLES = 20  # subset; full = 164

print(f"Loading {BASE_MODEL}...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, CKPT_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded.")

# Load HumanEval problems
problems = read_problems()
task_ids = list(problems.keys())[:MAX_SAMPLES]
print(f"Loaded {len(task_ids)} HumanEval problems.")

results = []
model.eval()

for task_id in task_ids:
    problem = problems[task_id]
    prompt = problem["prompt"]

    messages = [{"role": "user", "content": f"Write a Python function:\n{prompt}"}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    # Extract function body
    completion = generated.strip()
    # Try to stop at function end
    if "\n\n" in completion:
        completion = completion.split("\n\n")[0]

    # Check correctness
    result = check_correctness(problem, completion, timeout=5.0)
    results.append({
        "task_id": task_id,
        "passed": result["passed"],
        "completion": completion,
    })

    passed_str = "PASS" if result["passed"] else "FAIL"
    print(f"  [{len(results)}/{len(task_ids)}] {task_id}: {passed_str}")

pass_rate = sum(r["passed"] for r in results) / len(results) * 100
print(f"\n=== HumanEval Results ===")
print(f"Samples: {len(results)}/{len(problems)}")
print(f"pass@1: {pass_rate:.1f}%")

output = {
    "model": BASE_MODEL,
    "checkpoint": CKPT_PATH,
    "num_samples": len(results),
    "pass_at_1": pass_rate,
    "results": results,
}
with open("/root/humaneval_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved to /root/humaneval_results.json")
