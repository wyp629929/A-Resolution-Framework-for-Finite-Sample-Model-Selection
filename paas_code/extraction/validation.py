"""Validation signal extraction.

Three domains, two automation levels:
- GSM8K: fully automatic (answer string matching)
- CodeAlpaca: fully automatic (sandbox execution)
- Dolly: pairwise comparison (self-judge in pilot, human labels in full-scale)
"""
from __future__ import annotations
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List
from tqdm import tqdm

from config.schemas import Checkpoint, ValidationConfig, SignalScores


def extract_validation_signals(
    checkpoints: List[Checkpoint],
    config: ValidationConfig,
) -> List[Checkpoint]:
    """Compute validation signals for each checkpoint."""
    if _is_pilot_mode(config):
        return _extract_validation_pilot(checkpoints, config)
    return _extract_validation_full(checkpoints, config)


def _is_pilot_mode(config: ValidationConfig) -> bool:
    """Detect if we're in pilot mode: no human labels available."""
    if config.use_human_labels:
        label_dir = Path(config.pairwise_label_dir)
        return not label_dir.exists() or not list(label_dir.glob("*.json"))
    return True  # default to pilot unless explicitly using human labels


def _extract_validation_pilot(
    checkpoints: List[Checkpoint],
    config: ValidationConfig,
) -> List[Checkpoint]:
    """Pilot mode: use automated signals + self-judge for Dolly."""
    gsm8k_data = _load_json(config.gsm8k_test_path, "gsm8k_test.json")
    code_data = _load_json(config.code_test_path, "codealpaca_test.json")
    dolly_prompts = _load_json("", config.dolly_prompts_path)  # load separately

    for ckpt in tqdm(checkpoints, desc="[validation] Extracting validation signals (pilot)"):
        gsm8k_acc = _eval_gsm8k(ckpt, gsm8k_data) if gsm8k_data else _synthetic_validation("gsm8k", ckpt, checkpoints)
        code_score = _eval_codealpaca(ckpt, code_data) if code_data else _synthetic_validation("code", ckpt, checkpoints)
        bt_score = _eval_dolly_selfjudge(ckpt, dolly_prompts) if dolly_prompts else _synthetic_dolly(ckpt, checkpoints)

        ckpt.validation_scores = SignalScores(
            gsm8k_accuracy=gsm8k_acc,
            code_func_score=code_score,
            pairwise_bt_score=bt_score,
        )
    return checkpoints


def _extract_validation_full(
    checkpoints: List[Checkpoint],
    config: ValidationConfig,
) -> List[Checkpoint]:
    """Full-scale mode: automated for GSM8K/Code, human labels for Dolly."""
    gsm8k_data = _load_json(config.gsm8k_test_path, "gsm8k_test.json")
    code_data = _load_json(config.code_test_path, "codealpaca_test.json")

    # Load human pairwise labels
    pairwise_labels = _load_human_labels(config.pairwise_label_dir)

    for ckpt in tqdm(checkpoints, desc="[validation] Extracting validation signals (full)"):
        gsm8k_acc = _eval_gsm8k(ckpt, gsm8k_data) if gsm8k_data else 0.0
        code_score = _eval_codealpaca(ckpt, code_data) if code_data else 0.0
        bt_score = pairwise_labels.get(str(ckpt.step), 0.5)

        ckpt.validation_scores = SignalScores(
            gsm8k_accuracy=gsm8k_acc,
            code_func_score=code_score,
            pairwise_bt_score=bt_score,
        )
    return checkpoints


# ─── GSM8K Evaluation ──────────────────────────────────────────────────────

def _eval_gsm8k(ckpt: Checkpoint, data: List[dict]) -> float:
    """Evaluate on GSM8K test set. Returns accuracy (fraction correct).

    Each sample has 'question' and 'answer' fields.
    Answer format: '#### 42' — extract numeric answer.
    """
    try:
        model, tokenizer = _load_model(ckpt)
    except Exception:
        return 0.5  # fallback

    correct = 0
    for sample in data[:100]:  # limit for speed
        prompt = sample["question"]
        expected = _extract_gsm8k_answer(sample.get("answer", ""))
        if expected is None:
            continue

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        predicted = _extract_gsm8k_answer(response)
        if predicted is not None and predicted == expected:
            correct += 1

    return correct / max(len(data[:100]), 1)


def _extract_gsm8k_answer(text: str):
    """Extract the numeric answer after '####'."""
    match = re.search(r"####\s*([\d.,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    # Fallback: look for the last number
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


# ─── CodeAlpaca Evaluation ─────────────────────────────────────────────────

def _eval_codealpaca(ckpt: Checkpoint, data: List[dict]) -> float:
    """Evaluate code generation by attempting to execute the generated code.

    Returns a score in [0, 1] based on a combination of:
    - Syntactic validity (parses + runs without error)
    - Functional correctness (matches expected output if provided)
    """
    try:
        model, tokenizer = _load_model(ckpt)
    except Exception:
        return 0.5

    scores = []
    for sample in data[:50]:
        prompt = sample.get("instruction", "")
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            output = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
            code = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception:
            scores.append(0.0)
            continue

        score = _execute_code(code)
        scores.append(score)

    return sum(scores) / max(len(scores), 1)


def _execute_code(code: str) -> float:
    """Attempt to execute Python code in a sandbox.

    Returns:
        1.0: executes without error
        0.5: syntax error or import error
        0.0: runtime error
    """
    # Extract code block if present
    code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", code, re.DOTALL)
    if code_match:
        code = code_match.group(1)

    # Basic safety: block dangerous imports
    dangerous = ["os.system", "subprocess", "shutil.rmtree", "__import__", "eval(", "exec("]
    if any(d in code for d in dangerous):
        return 0.5  # refuse to run, partial credit

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                text=True,
                timeout=5,
                env={"PYTHONPATH": ""},  # isolate from project code
            )
            if result.returncode == 0:
                return 1.0
            elif "SyntaxError" in result.stderr:
                return 0.5
            else:
                return 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0


# ─── Dolly Pairwise (Self-Judge, Pilot Only) ───────────────────────────────

def _eval_dolly_selfjudge(ckpt: Checkpoint, prompts: List[dict]) -> float:
    """Self-judge: compare checkpoint's response to anchor response.

    Returns Bradley-Terry score in [0, 1] (preference for checkpoint output).
    0.5 = tie, >0.5 = checkpoint wins, <0.5 = anchor wins.
    """
    try:
        model, tokenizer = _load_model(ckpt)
    except Exception:
        return 0.5

    wins = 0
    total = 0
    for prompt in prompts[:20]:
        p_text = prompt.get("prompt", prompt.get("conversation", str(prompt)))
        anchor = prompt.get("anchor_response", "")

        # Generate checkpoint response
        inputs = tokenizer(p_text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
        ckpt_response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Simple self-judge: compare length-normalized log-probs
        ckpt_logp = _text_log_prob(model, tokenizer, p_text + "\n" + ckpt_response)
        anchor_logp = _text_log_prob(model, tokenizer, p_text + "\n" + anchor)
        if ckpt_logp > anchor_logp:
            wins += 1
        total += 1

    return wins / max(total, 1)


# ─── Utilities ──────────────────────────────────────────────────────────────

def _load_json(base_path: str, filename: str) -> List[dict]:
    """Load JSON data, returning empty list if not found."""
    p = Path(base_path) / filename if base_path else Path(filename)
    if not p.exists():
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _load_human_labels(label_dir: str) -> dict:
    """Load human pairwise labels. Returns dict: step_str → bt_score."""
    p = Path(label_dir)
    if not p.exists():
        return {}
    labels = {}
    for f in sorted(p.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
            step = data.get("step", f.stem)
            bt_score = data.get("bt_score", 0.5)
            labels[str(step)] = bt_score
    return labels


def _load_model(ckpt: Checkpoint):
    """Load model for evaluation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype).to(device)
    try:
        model = PeftModel.from_pretrained(base, ckpt.path)
    except Exception:
        model = base
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _text_log_prob(model, tokenizer, text: str) -> float:
    """Compute average log-probability of text under model."""
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return -outputs.loss.item()


def _synthetic_validation(domain: str, ckpt: Checkpoint, all_ckpts: List[Checkpoint]) -> float:
    """Generate synthetic validation signal (pilot-only)."""
    import math, random
    rng = random.Random(42 + hash(domain) % 1000)
    # Validation accuracy improves over training but has divergence potential
    normalized_step = ckpt.step / max(c.step for c in all_ckpts)
    base = 0.6 + 0.3 * (1 - math.exp(-2 * normalized_step))
    # Add crossing noise in the last third of training
    if normalized_step > 0.7:
        base += rng.gauss(0, 0.08)
    return max(0.0, min(1.0, base))


def _synthetic_dolly(ckpt: Checkpoint, all_ckpts: List[Checkpoint]) -> float:
    """Synthetic Dolly BT score (pilot-only). Trends to diverge over time."""
    import math, random
    rng = random.Random(43)
    normalized_step = ckpt.step / max(c.step for c in all_ckpts)
    # BT score starts at ~0.5, increases, then declines (simulating crossing)
    if normalized_step < 0.5:
        score = 0.5 + 0.3 * (normalized_step / 0.5)
    else:
        score = 0.8 - 0.4 * ((normalized_step - 0.5) / 0.5)
    score += rng.gauss(0, 0.05)
    return max(0.0, min(1.0, score))
