"""End-to-end experiment runner.

Orchestrates all PAAS pipeline stages in sequence.
Supports resumption: stages that already have output are skipped.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

from config.schemas import (
    ExperimentConfig,
    Checkpoint,
    TriggerEvent,
    SelectionResult,
    OracleResult,
)


def run_experiment(
    config: ExperimentConfig,
    resume: bool = True,
    force_trigger_test: bool = False,
) -> dict:
    """Run a full PAAS experiment end-to-end.

    Args:
        config: Full experiment configuration.
        resume: If True, skip stages with existing output.
        force_trigger_test: If True, run P5a/P5b trigger validation tests.

    Returns:
        Dict with all results: checkpoints, rho_trajectory, triggers,
        selection, oracle, baselines.
    """
    exp_dir = Path(config.output_dir) / config.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== PAAS Experiment: {config.experiment_name} ===")

    # ─── Stage 1: Training ──────────────────────────────────────────────
    stage_start("training", exp_dir, resume)
    from train.trainer import run_training
    train_cfg = config.train
    checkpoints = run_training(train_cfg)
    _save_stage(exp_dir, "checkpoints", [c.__dict__ for c in checkpoints])
    print(f"[pipeline] Training complete: {len(checkpoints)} checkpoints")

    # ─── Stage 2: Proxy Signals ─────────────────────────────────────────
    stage_start("proxy_signals", exp_dir, resume)
    from extraction.proxy import extract_proxy_signals
    checkpoints = extract_proxy_signals(checkpoints, config.proxy)
    _save_stage(exp_dir, "proxy_signals", [c.__dict__ for c in checkpoints])
    print("[pipeline] Proxy signals extracted")

    # ─── Stage 3: Validation Signals ────────────────────────────────────
    stage_start("validation_signals", exp_dir, resume)
    from extraction.validation import extract_validation_signals
    checkpoints = extract_validation_signals(checkpoints, config.validation)
    _save_stage(exp_dir, "validation_signals", [c.__dict__ for c in checkpoints])
    print("[pipeline] Validation signals extracted")

    # ─── Stage 4: Monitor (ρ_t + triggers) ─────────────────────────────
    stage_start("monitor", exp_dir, resume)
    from monitor.monitor import compute_rho_trajectory, evaluate_triggers
    rho_traj = compute_rho_trajectory(checkpoints, config.monitor.rho_window)
    triggers = evaluate_triggers(rho_traj, config.monitor)
    _save_stage(exp_dir, "rho_trajectory", rho_traj)
    _save_stage(exp_dir, "triggers", [e.__dict__ for e in triggers])
    print(f"[pipeline] ρ_t computed: {len(rho_traj)} points, {len(triggers)} triggers")

    # ─── Stage 5: PAAS Selection ────────────────────────────────────────
    stage_start("selection", exp_dir, resume)
    from selection.selector import select_checkpoint
    selection_result = select_checkpoint(checkpoints, triggers, config.selection)
    _save_stage(exp_dir, "selection_result", _serialize_selection(selection_result))
    print(f"[pipeline] Selected: step {selection_result.selected.step} ({selection_result.selection_mode})")

    # ─── Stage 5b: Forced Trigger Test (P5a/P5b, pilot-only) ──────────
    if force_trigger_test:
        print("[pipeline] Running forced trigger tests (P5a/P5b)...")
        _run_forced_trigger_tests(checkpoints, exp_dir)

    # ─── Stage 6: Oracle Evaluation ─────────────────────────────────────
    stage_start("oracle", exp_dir, resume)
    # Only evaluate the selected checkpoint and a few reference points
    oracle_checkpoints = _select_oracle_ckpts(checkpoints, selection_result)
    from evaluate.evaluator import evaluate_oracle
    oracle_results = evaluate_oracle(oracle_checkpoints, config.oracle)
    _save_stage(exp_dir, "oracle_results", {
        str(k): v.__dict__ for k, v in oracle_results.items()
    })
    print(f"[pipeline] Oracle evaluated: {len(oracle_results)} checkpoints")

    # ─── Stage 7: Baselines ─────────────────────────────────────────────
    stage_start("baselines", exp_dir, resume)
    from baselines.baselines import run_baseline
    baseline_results = {}
    for method in config.baselines:
        result = run_baseline(checkpoints, method)
        baseline_results[method] = _serialize_selection(result)
        print(f"[pipeline] Baseline '{method}': step {result.selected.step}")
    _save_stage(exp_dir, "baseline_results", baseline_results)

    # ─── Stage 8: Analysis ──────────────────────────────────────────────
    from analyze.report import generate_report
    report_path = generate_report(
        experiment_name=config.experiment_name,
        checkpoints=checkpoints,
        rho_trajectory=rho_traj,
        triggers=triggers,
        selection_result=selection_result,
        oracle_results=oracle_results,
        baseline_results=baseline_results,
        output_dir=config.output_dir,
    )
    print(f"[pipeline] Report: {report_path}")

    return {
        "config": config,
        "checkpoints": checkpoints,
        "rho_trajectory": rho_traj,
        "triggers": triggers,
        "selection_result": selection_result,
        "oracle_results": oracle_results,
        "baseline_results": baseline_results,
    }


def stage_start(name: str, exp_dir: Path, resume: bool):
    """Check if a stage can be skipped."""
    if resume and (exp_dir / f"{name}_done.txt").exists():
        print(f"[pipeline] Skipping {name} (already done)")
        return True
    print(f"[pipeline] === Stage: {name} ===")
    return False


def _save_stage(exp_dir: Path, name: str, data):
    """Save stage output and mark as done."""
    out_dir = exp_dir / "stages"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    (exp_dir / f"{name}_done.txt").write_text(f"done at {time.time()}")


def _run_forced_trigger_tests(checkpoints, exp_dir):
    """P5a: Directly inject caution_entered trigger to test selection logic.
       P5b: Inject triggers with a known safe checkpoint to test fallback.

    These tests bypass ρ_t computation and directly test the selection
    logic branch coverage.
    """
    from config.schemas import TriggerEvent, SelectionConfig
    from selection.selector import select_checkpoint
    import copy

    # P5a: Caution entered from the first checkpoint, no safe region exists
    triggers_a = [
        TriggerEvent(step=checkpoints[0].step, rho_at_trigger=-0.5, trigger_type="caution_entered"),
    ]
    result_a = select_checkpoint(checkpoints, triggers_a, SelectionConfig())
    print(f"  P5a (no safe region, direct trigger): mode={result_a.selection_mode}, step={result_a.selected.step}")
    _save_stage(exp_dir, "p5a_force_trigger", {
        "n_triggers": len(triggers_a),
        "mode": result_a.selection_mode,
        "selected_step": result_a.selected.step,
    })

    # P5b: Caution entered mid-training, safe region exists before it
    # First safe checkpoint is before the trigger step
    sorted_ckpts = sorted(checkpoints, key=lambda c: c.step)
    if len(sorted_ckpts) >= 4:
        mid_step = sorted_ckpts[len(sorted_ckpts) // 2].step
        triggers_b = [
            TriggerEvent(step=mid_step, rho_at_trigger=-0.5, trigger_type="caution_entered"),
        ]
        result_b = select_checkpoint(sorted_ckpts, triggers_b, SelectionConfig())
        print(f"  P5b (safe region exists): mode={result_b.selection_mode}, step={result_b.selected.step}, "
              f"fallback_safe={result_b.last_safe.step if result_b.last_safe else None}")
        _save_stage(exp_dir, "p5b_mock_fallback", {
            "n_triggers": len(triggers_b),
            "mode": result_b.selection_mode,
            "selected_step": result_b.selected.step,
            "last_safe_step": result_b.last_safe.step if result_b.last_safe else None,
        })


def _serialize_selection(sr) -> dict:
    """Deep-serialize a SelectionResult for JSON storage."""
    ckpt = sr.selected
    return {
        "selection_mode": sr.selection_mode,
        "candidate_window": sr.candidate_window,
        "selected": {
            "step": ckpt.step,
            "path": ckpt.path,
            "seed": ckpt.seed,
            "schedule": ckpt.schedule,
            "proxy_scores": {
                "proxy_loss": ckpt.proxy_scores.proxy_loss,
                "proxy_win_rate": ckpt.proxy_scores.proxy_win_rate,
                "proxy_main": ckpt.proxy_scores.proxy_main,
            },
            "validation_scores": {
                "gsm8k_accuracy": ckpt.validation_scores.gsm8k_accuracy,
                "code_func_score": ckpt.validation_scores.code_func_score,
                "pairwise_bt_score": ckpt.validation_scores.pairwise_bt_score,
                "validation_main": ckpt.validation_scores.validation_main,
            },
        },
        "last_safe": {
            "step": sr.last_safe.step,
            "path": sr.last_safe.path,
        } if sr.last_safe else None,
    }


def _select_oracle_ckpts(
    checkpoints: list,
    selection_result: SelectionResult,
    max_count: int = 5,
) -> list:
    """Select checkpoints for oracle evaluation.

    Evaluates the selected checkpoint plus a few references.
    """
    sorted_ckpts = sorted(checkpoints, key=lambda c: c.step)
    selected_step = selection_result.selected.step

    # Build dict by step to deduplicate
    oracle_map = {}
    for c in sorted_ckpts:
        oracle_map[c.step] = c

    # Ensure selected checkpoint is included
    # Add some evenly spaced reference points
    ref_steps = {selected_step}
    if len(sorted_ckpts) > 1:
        for i in [0, len(sorted_ckpts) // 2, -1]:
            ref_steps.add(sorted_ckpts[i].step)

    return [oracle_map[s] for s in sorted(ref_steps) if s in oracle_map][:max_count]
