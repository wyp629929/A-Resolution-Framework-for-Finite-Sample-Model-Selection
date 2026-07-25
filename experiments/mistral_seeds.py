#!/usr/bin/env python3
"""Mistral-7B LoRA fine-tuning for seed replication (B2).

Trains Mistral-7B-Instruct-v0.3 with LoRA rank 16 on GSM8K
for 500 steps, saving 10 checkpoints at 50-step intervals.

Usage:
    python3 mistral_seeds.py --seed 43
    python3 mistral_seeds.py --seed 44

Each seed takes ~1-2 hours on a single RTX 4090.
Requires 14GB disk for model + ~400MB for checkpoints.
"""

import argparse, os, json, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=[43, 44])
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--output_dir", default="/root/paas/results/mistral_seeds")
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f"seed{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Mistral-7B-Instruct-v0.3...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
        local_files_only=True,
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    dataset = dataset.map(lambda x: {
        "text": f"Question: {x['question']}\nAnswer: {x['answer']}"
    })

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.total_steps,
        save_steps=args.save_every,
        save_total_limit=None,
        logging_steps=10,
        lr_scheduler_type="cosine",
        seed=args.seed,
        bf16=True,
        report_to="none",
        run_name=f"mistral_lora_seed{args.seed}",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
    )

    print(f"Training seed {args.seed} (500 steps)...")
    trainer.train()
    print(f"Training complete. Checkpoints saved to {output_dir}")

    # Verify checkpoints
    ckpts = sorted(os.listdir(output_dir))
    print(f"Checkpoints: {ckpts}")


if __name__ == "__main__":
    main()
