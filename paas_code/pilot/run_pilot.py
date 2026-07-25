"""Pilot experiment: validates pipeline end-to-end before full-scale.

Pilot acceptance criteria (all must pass):
  P1: Training runs end-to-end, checkpoints saved with correct indices
  P2: Proxy signals extractable and values are in reasonable ranges
  P3: Validation signals extractable (GSM8K auto + Dolly self-judge)
  P4: ρ_t computation runs, output in [-1, 1], trend is interpretable
  P5a: Forced trigger (τ=0.95) correctly enters cautious selection path
  P5b: Mocked divergence correctly triggers fallback to last safe checkpoint
  P6: Full pipeline runs end-to-end under 30 minutes (or reasonable for model size)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

# Add parent to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.schemas import (
    PilotConfig,
    TrainConfig,
    ProxyConfig,
    ValidationConfig,
    MonitorConfig,
    SelectionConfig,
    OracleConfig,
    ExperimentConfig,
)
from pipeline.runner import run_experiment


def run_pilot():
    """Run the pilot experiment with minimal settings."""
    pc = PilotConfig()

    config = ExperimentConfig(
        experiment_name="pilot_v0",
        train=TrainConfig(
            base_model=pc.base_model,
            total_steps=pc.total_steps,
            save_every_k_steps=pc.save_every_k_steps,
            data_mix=pc.data_mix,
            schedule="cosine",
            seed=42,
        ),
        proxy=ProxyConfig(
            win_rate_n=50,  # smaller for pilot
        ),
        validation=ValidationConfig(
            use_human_labels=False,  # self-judge for pilot
        ),
        monitor=MonitorConfig(
            rho_window=pc.rho_window,
            threshold=pc.threshold,
            ci_method="bootstrap",  # pilot-only; switch to permutation for full-scale
        ),
        selection=SelectionConfig(
            window_frac=0.2,
            aggregation="uniform",
        ),
        oracle=OracleConfig(
            benchmarks=["mmlu"],  # just one for pilot
            batch_size=4,
        ),
        baselines=["proxy_best", "uniform_ensemble"],
        seed=42,
        output_dir="results",
    )

    print("=" * 60)
    print("PAAS Pilot v0")
    print("=" * 60)
    print(f"Model: {config.train.base_model}")
    print(f"Schedules: {config.train.schedule}")
    print(f"Steps: {config.train.total_steps} (save every {config.train.save_every_k_steps})")
    print(f"Pilot criteria: P1-P6 (see run_pilot.py docstring)")
    print("=" * 60)

    start = time.time()

    results = run_experiment(
        config,
        resume=False,
        force_trigger_test=pc.override_trigger_test,
    )

    elapsed = time.time() - start
    print(f"\n=== Pilot complete in {elapsed:.1f}s ===")

    # Report pass/fail for each criterion
    _report_criteria(results, elapsed)


def _report_criteria(results: dict, elapsed: float):
    """Check and report on pilot criteria."""
    rho = results.get("rho_trajectory", [])
    triggers = results.get("triggers", [])
    selection = results.get("selection_result")
    oracle = results.get("oracle_results", {})
    baselines = results.get("baseline_results", {})

    print("\n--- Pilot Criteria Report ---")

    # P1: checkpoints exist
    n_ckpts = len(results.get("checkpoints", []))
    p1 = n_ckpts > 1
    print(f"  P1 (checkpoints): {'PASS' if p1 else 'FAIL'} — {n_ckpts} checkpoints")

    # P2: proxy signals have reasonable values
    proxy_vals = [c.proxy_scores.proxy_main for c in results.get("checkpoints", []) if c.proxy_scores]
    p2 = len(proxy_vals) > 0 and all(-0.5 <= v <= 2.0 for v in proxy_vals) if proxy_vals else False
    print(f"  P2 (proxy signals): {'PASS' if p2 else 'FAIL'} — {len(proxy_vals)} values, range [{min(proxy_vals):.3f}, {max(proxy_vals):.3f}]")

    # P3: validation signals exist
    val_vals = [c.validation_scores.validation_main for c in results.get("checkpoints", []) if c.validation_scores]
    p3 = len(val_vals) > 0 and all(0.0 <= v <= 1.0 for v in val_vals) if val_vals else False
    print(f"  P3 (validation signals): {'PASS' if p3 else 'FAIL'} — {len(val_vals)} values, range [{min(val_vals):.3f}, {max(val_vals):.3f}]")

    # P4: ρ_t in [-1, 1]
    rho_vals = [p["rho_spearman"] for p in rho if p["rho_spearman"] is not None]
    p4 = len(rho_vals) > 0 and all(-1.0 <= v <= 1.0 for v in rho_vals) if rho_vals else False
    print(f"  P4 (rho range): {'PASS' if p4 else 'FAIL'} — range [{min(rho_vals):.3f}, {max(rho_vals):.3f}]")

    # P5a: forced trigger test
    p5a = selection is not None
    print(f"  P5a (forced trigger): {'PASS' if p5a else 'FAIL'} — mode={selection.selection_mode if selection else 'N/A'}")

    # P5b: fallback test
    from pathlib import Path
    exp_dir = Path("results") / "pilot_v0" / "stages"
    p5b = (exp_dir / "p5b_mock_fallback.json").exists()
    print(f"  P5b (mock fallback): {'PASS' if p5b else 'FAIL'}")

    # P6: total time
    p6 = elapsed < 1800  # 30 min
    print(f"  P6 (runtime): {'PASS' if p6 else 'FAIL'} — {elapsed:.0f}s / 1800s")

    total = sum([p1, p2, p3, p4, p5a, p5b, p6])
    print(f"\n  Result: {total}/7 criteria passed")
    if total < 7:
        print("  Review warnings above before proceeding to full-scale.")


if __name__ == "__main__":
    run_pilot()
