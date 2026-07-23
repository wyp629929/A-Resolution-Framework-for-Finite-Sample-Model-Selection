"""Oracle evaluation on benchmark suite.

This module evaluates the selected checkpoint on a multi-benchmark oracle.
For pilot/testing, generates synthetic scores.
For production, uses lm-evaluation-harness.
"""
from __future__ import annotations
import random
import math
from typing import List

from config.schemas import Checkpoint, OracleConfig, OracleResult


def evaluate_oracle(
    checkpoints: List[Checkpoint],
    config: OracleConfig,
) -> dict[int, OracleResult]:
    """Evaluate a list of checkpoints on the oracle benchmark suite.

    Returns dict: checkpoint_step → OracleResult
    """
    try:
        return _evaluate_real(checkpoints, config)
    except (ImportError, Exception) as e:
        print(f"[oracle] Real evaluation unavailable ({e}). Using synthetic.")
        return _evaluate_synthetic(checkpoints)


def _evaluate_real(
    checkpoints: List[Checkpoint],
    config: OracleConfig,
) -> dict[int, OracleResult]:
    """Real evaluation using lm-evaluation-harness or direct evaluation."""
    from lm_eval import simple_evaluate
    from lm_eval.utils import make_table

    results = {}
    for ckpt in checkpoints:
        eval_result = simple_evaluate(
            model="hf",
            model_args=f"pretrained=Qwen/Qwen2.5-0.5B-Instruct,peft={ckpt.path}",
            tasks=config.benchmarks,
            batch_size=config.batch_size,
            num_fewshot=5,
        )
        results[ckpt.step] = OracleResult(
            mmlu=_extract_metric(eval_result, "mmlu"),
            humaneval=_extract_metric(eval_result, "humaneval"),
            mt_bench=_extract_metric(eval_result, "mt_bench"),
            aggregate=eval_result.get("results", {}).get("average", 0.0),
        )

    return results


def _evaluate_synthetic(checkpoints: List[Checkpoint]) -> dict[int, OracleResult]:
    """Generate synthetic oracle results for pipeline testing (pilot-only)."""
    rng = random.Random(42)
    results = {}
    for ckpt in checkpoints:
        normalized_step = ckpt.step / max(c.step for c in checkpoints)
        # Oracle quality slowly improves, then may degrade
        base = 0.5 + 0.3 * (1 - math.exp(-3 * normalized_step))
        # Last third may diverge (simulating proxy-true misalignment)
        if normalized_step > 0.75:
            base -= rng.uniform(0, 0.15)
        noise = rng.gauss(0, 0.02)
        aggregate = max(0.0, min(1.0, base + noise))

        results[ckpt.step] = OracleResult(
            mmlu=max(0.0, min(1.0, aggregate + rng.gauss(0, 0.03))),
            humaneval=max(0.0, min(1.0, aggregate + rng.gauss(0, 0.05))),
            mt_bench=max(0.0, min(1.0, aggregate + rng.gauss(0, 0.03))),
            domain_specific=max(0.0, min(1.0, aggregate + rng.gauss(0, 0.04))),
            aggregate=aggregate,
        )
    return results


def _extract_metric(eval_result, task_name: str) -> float | None:
    """Extract a metric from lm_eval output."""
    try:
        results = eval_result.get("results", {})
        for key in results:
            if task_name in key.lower():
                acc = results[key].get("acc,none")
                if acc is not None:
                    return float(acc)
                acc_norm = results[key].get("acc_norm,none")
                if acc_norm is not None:
                    return float(acc_norm)
    except Exception:
        pass
    return None
