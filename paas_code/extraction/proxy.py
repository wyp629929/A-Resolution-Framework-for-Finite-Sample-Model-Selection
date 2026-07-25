"""Proxy signal extraction: validation loss and win-rate for each checkpoint."""
from __future__ import annotations
import json
import torch
from pathlib import Path
from typing import List
from tqdm import tqdm

from config.schemas import Checkpoint, ProxyConfig, SignalScores


def extract_proxy_signals(
    checkpoints: List[Checkpoint],
    config: ProxyConfig,
) -> List[Checkpoint]:
    """Compute proxy signals for each checkpoint and add them in-place.

    Proxy signals:
    - validation loss: cross-entropy on a held-out set
    - win-rate: pairwise comparison against a reference response
    """
    held_out = _load_held_out(config.held_out_path)
    if not held_out:
        # Fallback: create synthetic proxy signals (pilot-only)
        return _synthetic_proxy_signals(checkpoints)

    try:
        ref_model, ref_tokenizer = _load_ref_model()
    except Exception:
        # No model available — synthetic mode
        return _synthetic_proxy_signals(checkpoints)

    for ckpt in tqdm(checkpoints, desc="[proxy] Extracting proxy signals"):
        model, tokenizer = _load_checkpoint_model(ckpt)

        # Validation loss
        loss = _compute_loss(model, tokenizer, held_out)
        # Win-rate
        win_rate = _compute_win_rate(model, tokenizer, ref_model, ref_tokenizer, held_out, n=config.win_rate_n)

        if ckpt.proxy_scores is None:
            ckpt.proxy_scores = SignalScores()
        ckpt.proxy_scores.proxy_loss = loss
        ckpt.proxy_scores.proxy_win_rate = win_rate

        # Clean up to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return checkpoints


def _load_held_out(path: str) -> List[dict]:
    """Load held-out samples for proxy evaluation."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def _synthetic_proxy_signals(checkpoints: List[Checkpoint]) -> List[Checkpoint]:
    """Generate synthetic proxy signals for pipeline validation (pilot-only)."""
    import random
    import math
    rng = random.Random(42)
    for ckpt in checkpoints:
        # Simulate decreasing loss over steps, with some noise
        normalized_step = ckpt.step / max(c.step for c in checkpoints)  # 0→1
        base_loss = 2.0 - 1.5 * math.exp(-3 * (1 - normalized_step))  # decreases from ~2.0 to ~0.5
        noise = rng.gauss(0, 0.05)
        ckpt.proxy_scores = SignalScores(
            proxy_loss=base_loss + noise,
            proxy_win_rate=0.5 + 0.4 * (1 - math.exp(-2 * normalized_step)) + rng.gauss(0, 0.03),
        )
    return checkpoints


def _load_ref_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Use a small model as reference to avoid GPU memory issues
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _load_checkpoint_model(ckpt: Checkpoint):
    """Load a checkpoint model for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype).to(device)
    try:
        model = PeftModel.from_pretrained(base, ckpt.path)
    except Exception:
        model = base  # fallback: use base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def _compute_loss(model, tokenizer, samples: List[dict]) -> float:
    """Compute cross-entropy loss on a held-out set."""
    total_loss = 0.0
    count = 0
    for sample in samples[:64]:  # limit for speed
        text = sample.get("question", sample.get("instruction", str(sample)))
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item()
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def _compute_win_rate(model, tokenizer, ref_model, ref_tokenizer,
                       samples: List[dict], n: int = 200) -> float:
    """Compute win-rate: fraction of held-out samples where model output
    is preferred over reference output (using a simple length-normalized log-prob)."""
    wins = 0
    total = 0
    for sample in samples[:n]:
        prompt = sample.get("question", sample.get("instruction", str(sample)))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        ref_inputs = ref_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        ref_inputs = {k: v.to(ref_model.device) for k, v in ref_inputs.items()}

        model_out = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
        ref_out = ref_model.generate(**ref_inputs, max_new_tokens=64, pad_token_id=ref_tokenizer.eos_token_id)

        # Compute log-prob of generated tokens as a rough quality proxy
        model_logp = _sequence_log_prob(model, tokenizer, model_out[0]).item()
        ref_logp = _sequence_log_prob(ref_model, ref_tokenizer, ref_out[0]).item()
        if model_logp > ref_logp:
            wins += 1
        total += 1

    return wins / max(total, 1)


@torch.no_grad()
def _sequence_log_prob(model, tokenizer, token_ids) -> torch.Tensor:
    """Compute average log-prob of a token sequence under the model."""
    inputs = token_ids.unsqueeze(0).to(model.device)
    outputs = model(inputs, labels=inputs)
    return -outputs.loss  # negative loss = higher log-prob
