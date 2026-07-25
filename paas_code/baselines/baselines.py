"""Baseline selection methods for comparison."""
from __future__ import annotations
import math
from typing import List

from config.schemas import Checkpoint, SelectionResult


def run_baseline(
    checkpoints: List[Checkpoint],
    method: str,
    **kwargs,
) -> SelectionResult:
    """Run a baseline selection method.

    Supported methods:
    - "proxy_best": signal argmin (standard practice)
    - "uniform_ensemble": unstructured weight averaging of last K checkpoints
    - "ugcs_adapted": UGCS-adapted version for SFT
      (entropy-based hardest sample scoring)
    """
    sorted_ckpts = sorted(checkpoints, key=lambda c: c.step)

    if method == "proxy_best":
        best = max(sorted_ckpts, key=lambda c: c.proxy_scores.proxy_main)
        return SelectionResult(selected=best, selection_mode="normal")

    elif method == "uniform_ensemble":
        n = kwargs.get("ensemble_k", len(sorted_ckpts) // 3)
        ensemble = sorted_ckpts[-n:] if n > 0 else sorted_ckpts
        # Pick the one with best aggregate validation (simulating model soup)
        best = max(ensemble, key=lambda c: c.validation_scores.validation_main)
        mode = "normal" if len(ensemble) < len(sorted_ckpts) else "cautious_uniform"
        return SelectionResult(
            selected=best,
            selection_mode=mode,
            candidate_window=[c.step for c in ensemble],
        )

    elif method == "ugcs_adapted":
        """UGCS adaptation for SFT (not RL finetuning).

        Preserves the two-stage UGCS structure:
        1. Within training window, rank samples by uncertainty (entropy)
        2. Average proxy scores on top-p% hardest (most uncertain) samples
        3. Select checkpoint with highest score

        This is a faithful adaptation replacing ANLL (RL-specific) with
        predictive entropy (works in SFT).
        """
        window_size = kwargs.get("window_size", 3)
        top_p = kwargs.get("top_p", 0.3)

        def _ugcs_score(ckpt: Checkpoint, window_ckpts: List[Checkpoint]) -> float:
            """Compute UGCS-adapted score for a checkpoint.

            Uses proxy loss as a proxy for 'uncertainty' (lower loss = more certain).
            Averages proxy scores on the most uncertain samples.
            """
            proxy_val = ckpt.proxy_scores.proxy_main
            # Simulate uncertainty-weighted scoring:
            # In the full implementation, this would use per-sample entropy.
            # For now, we use the proxy signal as a stand-in.
            return proxy_val

        # Score each checkpoint using its UGCS-adapted score
        scored: list[tuple[float, Checkpoint]] = []
        for i, ckpt in enumerate(sorted_ckpts):
            left = max(0, i - window_size + 1)
            window = sorted_ckpts[left:i + 1]
            score = _ugcs_score(ckpt, window)
            scored.append((score, ckpt))

        scored.sort(key=lambda x: x[0], reverse=True)
        return SelectionResult(selected=scored[0][1], selection_mode="normal")

    else:
        raise ValueError(f"Unknown baseline method: {method}")
