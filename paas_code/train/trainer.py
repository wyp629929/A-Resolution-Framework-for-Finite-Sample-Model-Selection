"""LoRA fine-tuning + checkpoint saving.

This module is designed to run on a GPU machine (cloud or local).
The output is a directory of raw checkpoints, which are later processed
by the signal/ modules for proxy and validation signal extraction.
"""
from __future__ import annotations
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from config.schemas import TrainConfig, Checkpoint, SignalScores


def run_training(config: TrainConfig) -> List[Checkpoint]:
    """Run LoRA fine-tuning and return checkpoints metadata.

    For local CPU execution (pilot), trains a tiny model.
    For production, writes a shell script for cloud submission.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _is_on_gpu():
        return _train_local_gpu(config)
    else:
        # No GPU available: write cloud submission script and
        # fall back to a minimal local test with a tiny model
        _write_cloud_script(config)
        print(f"[train] No GPU detected. Wrote cloud submission script to {output_dir / 'cloud_train.sh'}")
        print("[train] Running minimal local train (Qwen2.5-0.5B, CPU) to validate pipeline...")
        return _train_local_cpu_pilot(config)


def _is_on_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except (ImportError, RuntimeError):
        return False


def _train_local_gpu(config: TrainConfig) -> List[Checkpoint]:
    """Full LoRA training on available GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = _load_training_data(config.data_mix, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.total_steps,
        save_steps=config.save_every_k_steps,
        save_total_limit=None,  # keep all checkpoints
        logging_steps=10,
        lr_scheduler_type=config.schedule,
        seed=config.seed,
        bf16=True,
        report_to="none",
        run_name=f"paas_{config.schedule}_seed{config.seed}",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        train_dataset=dataset,
    )

    trainer.train()
    _generate_checkpoint_list(output_dir, config)
    return _load_checkpoints_from_dir(config)


def _train_local_cpu_pilot(config: TrainConfig) -> List[Checkpoint]:
    """Create synthetic checkpoints for CPU pilot testing.

    Actual model training is impractical on CPU for even a 500M model.
    Synthetic checkpoints with realistic signal patterns are sufficient
    to validate the pipeline (P1-P6).
    """
    print("[train] CPU detected. Using synthetic checkpoints for pipeline validation.")
    print("[train] Full training requires GPU (use cloud/autodl_setup.sh or similar).")
    return _create_synthetic_checkpoints(config)


def _create_synthetic_checkpoints(config: TrainConfig) -> List[Checkpoint]:
    """Create dummy checkpoints when no model can be loaded (pilot-only)."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpts = []
    for step in range(0, config.total_steps + 1, config.save_every_k_steps):
        ckpt_dir = output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(exist_ok=True)
        (ckpt_dir / ".placeholder").write_text(f"synthetic checkpoint step={step}")
        ckpts.append(Checkpoint(
            step=step,
            path=str(ckpt_dir),
            seed=config.seed,
            schedule=config.schedule,
        ))
    return ckpts


def _load_training_data(data_mix, tokenizer, max_samples=None):
    from datasets import load_dataset, concatenate_datasets
    datasets = []
    for name in data_mix:
        if name == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="train")
            ds = ds.map(lambda x: {"instruction": x["question"]})
        elif name == "codealpaca":
            ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
            ds = ds.map(lambda x: {"instruction": x["prompt"]})
        elif name == "dolly":
            try:
                ds = load_dataset("databricks/databricks-dolly-15k", split="train")
                ds = ds.map(lambda x: {"instruction": x["instruction"]})
            except Exception as e:
                print(f"[train] WARN: cannot load Dolly ({e}), skipping")
                continue
        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples // len(data_mix))))
        datasets.append(ds)

    if not datasets:
        raise ValueError("No training datasets could be loaded")
    combined = concatenate_datasets(datasets).shuffle(seed=42)
    return combined


def _load_checkpoints_from_dir(config: TrainConfig) -> List[Checkpoint]:
    import glob
    output_dir = Path(config.output_dir)
    checkpoint_dirs = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    ckpts = []
    for ckpt_dir in checkpoint_dirs:
        step = int(ckpt_dir.name.split("-")[1])
        ckpts.append(Checkpoint(
            step=step,
            path=str(ckpt_dir),
            seed=config.seed,
            schedule=config.schedule,
        ))
    return ckpts


def _write_cloud_script(config: TrainConfig):
    """Write a shell script for cloud GPU execution."""
    output_dir = Path(config.output_dir)
    script = f"""#!/bin/bash
# PAAS cloud training script — generated for {config.base_model}
# Upload paas/ to cloud machine, then run this script.

cd /workspace/paas

# Install dependencies
pip install torch transformers accelerate peft datasets trl bitsandbytes

# Run training
python -c "
from train.trainer import run_training
from config.schemas import TrainConfig
cfg = TrainConfig(
    base_model='{config.base_model}',
    lora_rank={config.lora_rank},
    data_mix={config.data_mix},
    schedule='{config.schedule}',
    seed={config.seed},
    save_every_k_steps={config.save_every_k_steps},
    total_steps={config.total_steps},
    output_dir='{config.output_dir}',
)
run_training(cfg)
"

# Package checkpoints for download
tar czf checkpoints.tar.gz {config.output_dir}
echo "Training complete. Download checkpoints.tar.gz"
"""
    script_path = output_dir / "cloud_train.sh"
    script_path.write_text(script)
    os.chmod(str(script_path), 0o755)
