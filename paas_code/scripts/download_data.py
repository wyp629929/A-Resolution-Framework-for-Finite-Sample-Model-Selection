"""Download and prepare all datasets for PAAS experiments.

Downloads:
- GSM8K (train/test, with standard answer extraction)
- CodeAlpaca_20K (train/test split)
- Dolly sample for pairwise annotation

Outputs structured JSON files under data/ directory.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path


def download_all(output_dir: str = "data"):
    """Download and prepare all datasets."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[data] Downloading datasets...")
    _download_gsm8k(out)
    _download_codealpaca(out)
    _download_dolly_sample(out)
    _create_held_out(out)

    print(f"\n[data] All datasets ready in {out.resolve()}")


def _download_gsm8k(out: Path):
    """Download GSM8K and extract structured QA pairs."""
    from datasets import load_dataset

    for split in ["train", "test"]:
        ds = load_dataset("gsm8k", "main", split=split)
        records = []
        for item in ds:
            answer_num = _extract_answer(item["answer"])
            records.append({
                "question": item["question"],
                "answer": item["answer"],
                "answer_num": answer_num,
            })

        out_path = out / f"gsm8k_{split}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  GSM8K {split}: {len(records)} examples → {out_path}")


def _extract_answer(answer_text: str) -> str | None:
    """Extract numeric answer after '####'."""
    match = re.search(r"####\s*([\d.,]+)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None


def _download_codealpaca(out: Path):
    """Download CodeAlpaca and split into train/test."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
    records = []
    for item in ds:
        records.append({
            "instruction": item["prompt"],
            "input": "",
            "output": item["completion"],
        })

    # 90/10 split
    split_idx = int(len(records) * 0.9)

    train_path = out / "codealpaca_train.json"
    with open(train_path, "w") as f:
        json.dump(records[:split_idx], f, indent=2)
    print(f"  CodeAlpaca train: {split_idx} examples → {train_path}")

    test_path = out / "codealpaca_test.json"
    with open(test_path, "w") as f:
        json.dump(records[split_idx:], f, indent=2)
    print(f"  CodeAlpaca test: {len(records) - split_idx} examples → {test_path}")


def _download_dolly_sample(out: Path):
    """Download Databricks Dolly 15k and sample prompts for pairwise annotation.

    Saves:
    - dolly_sample.json: 20 prompts with anchor responses for pairwise comparison
    """
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    import random
    rng = random.Random(42)
    sampled = []
    indices = rng.sample(range(len(ds)), 20)
    for idx in indices:
        sampled.append({
            "prompt": ds[idx]["instruction"][:512],
            "anchor_response": ds[idx]["response"][:1024],
        })

    out_path = out / "dolly_sample.json"
    with open(out_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"  Dolly sample: {len(sampled)} examples → {out_path}")


def _create_held_out(out: Path):
    """Create a small held-out set for proxy evaluation (from GSM8K test)."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    held_out = []
    for item in ds.select(range(200)):  # 200 samples
        held_out.append({
            "question": item["question"],
            "answer": item["answer"],
        })

    out_path = out / "held_out.json"
    with open(out_path, "w") as f:
        json.dump(held_out, f, indent=2)
    print(f"  Held-out: {len(held_out)} examples → {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    download_all(args.output_dir)
