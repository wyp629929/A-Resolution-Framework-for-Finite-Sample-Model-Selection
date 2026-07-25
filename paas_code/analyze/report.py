"""Analysis and reporting: generate tables, plots, and summary statistics."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List

from config.schemas import Checkpoint, SelectionResult, TriggerEvent


def generate_report(
    experiment_name: str,
    checkpoints: List[Checkpoint],
    rho_trajectory: list[dict],
    triggers: List[TriggerEvent],
    selection_result: SelectionResult,
    oracle_results: dict | None,
    baseline_results: dict[str, SelectionResult] | None,
    output_dir: str = "results",
) -> str:
    """Generate a full analysis report.

    Writes JSON + Markdown summary to output_dir / experiment_name /.
    Returns the path to the report directory.
    """
    report_dir = Path(output_dir) / experiment_name / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Summary Table ────────────────────────────────────────────────
    summary = _build_summary_table(checkpoints)

    # ─── 2. Trigger Analysis ─────────────────────────────────────────────
    def _get_trigger_attr(e, attr):
        return e[attr] if isinstance(e, dict) else getattr(e, attr)

    trigger_summary = {
        "n_triggers": len(triggers),
        "trigger_steps": [_get_trigger_attr(e, "step") for e in triggers
                         if _get_trigger_attr(e, "trigger_type") == "caution_entered"],
        "exit_steps": [_get_trigger_attr(e, "step") for e in triggers
                       if _get_trigger_attr(e, "trigger_type") == "caution_exited"],
        "threshold": _get_trigger_attr(triggers[0], "rho_at_trigger") if triggers else None,
    }

    # ─── 3. Selection Summary ────────────────────────────────────────────
    def _get_sel(sr, attr):
        val = sr[attr] if isinstance(sr, dict) else getattr(sr, attr)
        return val
    sr = selection_result
    sel_selected = _get_sel(sr, "selected")
    sel_selected_step = sel_selected["step"] if isinstance(sel_selected, dict) else sel_selected.step
    sel_proxy = (sel_selected.get("proxy_scores", {}) or {}).get("proxy_main", 0) if isinstance(sel_selected, dict) else sel_selected.proxy_scores.proxy_main
    sel_val = (sel_selected.get("validation_scores", {}) or {}).get("validation_main", 0) if isinstance(sel_selected, dict) else sel_selected.validation_scores.validation_main
    last_safe = _get_sel(sr, "last_safe")

    selection_summary = {
        "selected_step": sel_selected_step,
        "selection_mode": _get_sel(sr, "selection_mode"),
        "selected_proxy_score": sel_proxy,
        "selected_val_score": sel_val,
        "last_safe_step": last_safe["step"] if isinstance(last_safe, dict) else last_safe.step if last_safe else None,
    }

    # ─── 4. Oracle Comparison ────────────────────────────────────────────
    if oracle_results:
        oracle_table = _build_oracle_table(oracle_results, selection_result)
    else:
        oracle_table = {}

    # ─── 5. Baseline Comparison ──────────────────────────────────────────
    if baseline_results:
        baseline_table = _build_baseline_table(baseline_results, oracle_results)
    else:
        baseline_table = {}

    # ─── 6. Write Outputs ────────────────────────────────────────────────
    report = {
        "experiment": experiment_name,
        "rho_trajectory": rho_trajectory,
        "triggers": [e.__dict__ for e in triggers],
        "selection": selection_summary,
        "summary_table": summary,
        "oracle": oracle_table,
        "baselines": baseline_table,
    }

    with open(report_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ─── 7. Write Markdown Summary ───────────────────────────────────────
    md = _format_markdown(report)
    with open(report_dir / "report.md", "w") as f:
        f.write(md)

    print(f"[analyze] Report written to {report_dir}")
    return str(report_dir)


def _build_summary_table(checkpoints: List[Checkpoint]) -> list[dict]:
    """Build a row-per-checkpoint summary with proxy and validation scores."""
    rows = []
    for ckpt in sorted(checkpoints, key=lambda c: c.step):
        rows.append({
            "step": ckpt.step,
            "proxy_loss": ckpt.proxy_scores.proxy_loss,
            "proxy_win_rate": ckpt.proxy_scores.proxy_win_rate,
            "proxy_main": ckpt.proxy_scores.proxy_main,
            "gsm8k": ckpt.validation_scores.gsm8k_accuracy,
            "code": ckpt.validation_scores.code_func_score,
            "sharegpt": ckpt.validation_scores.pairwise_bt_score,
            "val_main": ckpt.validation_scores.validation_main,
        })
    return rows


def _build_oracle_table(
    oracle_results: dict,
    selection: SelectionResult,
) -> dict:
    """Format oracle results with selection highlighting."""
    selected_step = selection.selected.step
    table = {}
    for step, result in oracle_results.items():
        table[str(step)] = result.__dict__
    table["selected_step"] = selected_step
    return table


def _build_baseline_table(
    baseline_results: dict,
    oracle_results: dict | None,
) -> dict:
    """Compare selection quality across baselines.

    baseline_results values can be SelectionResult objects or dicts
    (from JSON serialization).
    """
    table = {}
    for name, result in baseline_results.items():
        if isinstance(result, dict):
            selected = result.get("selected", {})
            step = selected.get("step", result.get("selected_step", "?"))
            proxy = (selected.get("proxy_scores", {}) or {}).get("proxy_main", 0)
            val = (selected.get("validation_scores", {}) or {}).get("validation_main", 0)
            mode = result.get("selection_mode", "normal")
        else:
            step = result.selected.step
            proxy = result.selected.proxy_scores.proxy_main
            val = result.selected.validation_scores.validation_main
            mode = result.selection_mode

        row = {
            "selected_step": step,
            "mode": mode,
            "proxy_main": proxy,
            "val_main": val,
        }
        if oracle_results and isinstance(step, int):
            oracle = oracle_results.get(step)
            if oracle and hasattr(oracle, "aggregate"):
                row["oracle_aggregate"] = oracle.aggregate
        table[name] = row
    return table


def _format_markdown(report: dict) -> str:
    """Format the report as a readable Markdown summary."""
    lines = [
        f"# PAAS Experiment: {report['experiment']}",
        "",
        "## ρ_t Trajectory",
        "",
        "| Step | ρ (Spearman) | n_points |",
        "|------|-------------|----------|",
    ]
    for p in report["rho_trajectory"][:30]:  # cap for readability
        rho_str = f"{p['rho_spearman']:.3f}" if p['rho_spearman'] is not None else "N/A"
        lines.append(f"| {p['step']} | {rho_str} | {p['n_points']} |")

    lines += ["", "## Trigger Events", ""]
    if report["triggers"]:
        lines.append("| Step | Type | ρ at trigger |")
        lines.append("|------|------|--------------|")
        for e in report["triggers"]:
            step = e["step"] if isinstance(e, dict) else e.step
            ttype = e["trigger_type"] if isinstance(e, dict) else e.trigger_type
            rho = e["rho_at_trigger"] if isinstance(e, dict) else e.rho_at_trigger
            lines.append(f"| {step} | {ttype} | {rho:.3f} |")
    else:
        lines.append("_No triggers fired._")

    lines += ["", "## Selection Result", ""]
    s = report["selection"]
    lines.append(f"- **Selected checkpoint**: step {s['selected_step']}")
    lines.append(f"- **Mode**: {s['selection_mode']}")
    lines.append(f"- **Proxy score**: {s['selected_proxy_score']:.4f}")
    lines.append(f"- **Validation score**: {s['selected_val_score']:.4f}")
    if s.get("last_safe_step"):
        lines.append(f"- **Last safe checkpoint**: step {s['last_safe_step']}")

    if report.get("baselines"):
        lines += ["", "## Baseline Comparison", ""]
        lines.append("| Method | Selected Step | Proxy | Validation | Oracle |")
        lines.append("|--------|--------------|-------|------------|--------|")
        for name, row in report["baselines"].items():
            oracle_val = row.get("oracle_aggregate", "N/A")
            lines.append(f"| {name} | {row['selected_step']} | {row['proxy_main']:.3f} | "
                         f"{row['val_main']:.3f} | {oracle_val if isinstance(oracle_val, str) else f'{oracle_val:.3f}'} |")

    return "\n".join(lines)
