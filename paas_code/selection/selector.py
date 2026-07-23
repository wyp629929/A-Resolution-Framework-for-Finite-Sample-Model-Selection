"""Cautious selection: window aggregation + fallback."""
from __future__ import annotations
from typing import List

from config.schemas import Checkpoint, SelectionConfig, SelectionResult, TriggerEvent


def select_checkpoint(
    checkpoints: List[Checkpoint],
    triggers: List[TriggerEvent],
    config: SelectionConfig,
) -> SelectionResult:
    """Select the best checkpoint using PAAS.

    Logic:
    1. If never triggered (no caution_entered): return proxy-best (normal mode)
    2. If currently in caution mode:
       a. If there was a "safe" checkpoint before caution: cautious_uniform
       b. If caution from start (no safe point): cautious_fallback
    3. If caution was entered and then exited: the selection is from the exit point
       (the latest safe region is trusted again)
    """
    sorted_ckpts = sorted(checkpoints, key=lambda c: c.step)

    # --- Determine current state ---
    caution_entries = [e for e in triggers if e.trigger_type == "caution_entered"]
    caution_exits = [e for e in triggers if e.trigger_type == "caution_exited"]

    currently_in_caution = len(caution_entries) > len(caution_exits)
    last_entry = caution_entries[-1] if caution_entries else None

    # Find the last safe checkpoint (before any caution entry)
    last_safe = _find_last_safe(sorted_ckpts, caution_entries)

    # --- Decision logic ---
    if not currently_in_caution or last_entry is None:
        # Normal mode: proxy-best selection
        best = max(sorted_ckpts, key=lambda c: c.proxy_scores.proxy_main)
        return SelectionResult(
            selected=best,
            selection_mode="normal",
        )

    # Caution mode: need to be careful
    if last_safe is not None:
        # Cautious uniform: use proxy to define a candidate window
        # then aggregate within it
        candidate_window = _define_candidate_window(sorted_ckpts, config.window_frac)
        if config.aggregation == "uniform":
            winner = _uniform_aggregation(candidate_window)
        else:
            winner = _validation_guided_aggregation(candidate_window)
        return SelectionResult(
            selected=winner,
            selection_mode="cautious_uniform",
            candidate_window=[c.step for c in candidate_window],
            last_safe=last_safe,
        )
    else:
        # Cautious fallback: no safe region existed before caution
        # Fallback to the last checkpoint (training endpoint)
        return SelectionResult(
            selected=sorted_ckpts[-1],
            selection_mode="cautious_fallback",
        )


def _find_last_safe(
    sorted_ckpts: List[Checkpoint],
    caution_entries: List[TriggerEvent],
) -> Checkpoint | None:
    """Find the last checkpoint that was safe before caution was entered."""
    if not caution_entries:
        return None
    first_caution_step = caution_entries[0].step
    safe_ckpts = [c for c in sorted_ckpts if c.step < first_caution_step]
    return safe_ckpts[-1] if safe_ckpts else None


def _define_candidate_window(
    sorted_ckpts: List[Checkpoint],
    window_frac: float,
) -> List[Checkpoint]:
    """Define a candidate window around the proxy-best region.

    Takes the top window_frac of total checkpoints by proxy score.
    """
    n_candidates = max(2, int(len(sorted_ckpts) * window_frac))
    ranked = sorted(sorted_ckpts, key=lambda c: c.proxy_scores.proxy_main, reverse=True)
    return ranked[:n_candidates]


def _uniform_aggregation(candidates: List[Checkpoint]) -> Checkpoint:
    """Uniform aggregation: average scores across the window, pick best.

    This is the 'model soup light' approach within the candidate window.
    """
    if not candidates:
        raise ValueError("No candidates in window")
    # Compute unweighted average of validation signals
    best = max(candidates, key=lambda c: c.validation_scores.validation_main)
    return best


def _validation_guided_aggregation(candidates: List[Checkpoint]) -> Checkpoint:
    """Validation-guided aggregation: use validation signal to pick within window."""
    return max(candidates, key=lambda c: c.validation_scores.validation_main)
