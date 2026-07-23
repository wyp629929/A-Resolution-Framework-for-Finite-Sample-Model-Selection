"""ρ_t monitoring and trigger evaluation module."""
from __future__ import annotations
import math
from typing import List

from config.schemas import Checkpoint, MonitorConfig, TriggerEvent


def compute_rho_trajectory(
    checkpoints: List[Checkpoint],
    window_size: int = 3,
) -> list[dict]:
    """Compute Spearman rank correlation ρ(t) along the training trajectory.

    For each checkpoint t, use a sliding window of `window_size` adjacent checkpoints
    (including t) to compute the rank correlation between proxy and validation signals.

    Returns a list of dicts: [{step, rho_spearman, n_points}, ...]
    """
    if len(checkpoints) < 2:
        return [{"step": c.step, "rho_spearman": 0.0, "n_points": 1} for c in checkpoints]

    # Sort by step
    sorted_ckpts = sorted(checkpoints, key=lambda c: c.step)
    trajectory = []

    for i in range(len(sorted_ckpts)):
        # Define window: from i - window_size + 1 to i (inclusive)
        start = max(0, i - window_size + 1)
        window = sorted_ckpts[start:i + 1]

        if len(window) < 2:
            trajectory.append({
                "step": sorted_ckpts[i].step,
                "rho_spearman": None,  # insufficient data
                "n_points": len(window),
            })
            continue

        proxy_vals = [c.proxy_scores.proxy_main for c in window]
        valid_vals = [c.validation_scores.validation_main for c in window]

        rho = _spearman_rank(proxy_vals, valid_vals)
        trajectory.append({
            "step": sorted_ckpts[i].step,
            "rho_spearman": rho,
            "n_points": len(window),
        })

    return trajectory


def evaluate_triggers(
    rho_trajectory: list[dict],
    config: MonitorConfig,
) -> list[TriggerEvent]:
    """Evaluate trigger conditions along the ρ_t trajectory.

    Trigger logic:
    - caution_entered: when ρ_t drops below config.threshold
    - caution_exited: when ρ_t rises back above config.threshold
      AND stayed above for min_safe_count consecutive points
    - Points with n_points < config.min_points are skipped (window not full)

    Returns a chronological list of TriggerEvents.
    """
    events: list[TriggerEvent] = []
    in_caution = False
    safe_streak = 0

    for point in rho_trajectory:
        rho = point["rho_spearman"]
        step = point["step"]
        n_points = point.get("n_points", 0)

        # Skip points where the sliding window isn't full yet
        if n_points < config.min_points:
            continue
            if not in_caution:
                events.append(TriggerEvent(
                    step=step,
                    rho_at_trigger=rho,
                    trigger_type="caution_entered",
                ))
                in_caution = True
            safe_streak = 0
        else:
            safe_streak += 1
            if in_caution and safe_streak >= config.min_safe_count:
                events.append(TriggerEvent(
                    step=step,
                    rho_at_trigger=rho,
                    trigger_type="caution_exited",
                ))
                in_caution = False

    return events


def _spearman_rank(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient.

    Returns value in [-1, 1].
    """
    n = len(x)
    if n < 2:
        return 0.0

    # Handle ties: use average ranking
    x_ranked = _rank_data(x)
    y_ranked = _rank_data(y)

    # Pearson correlation on ranks
    x_bar = sum(x_ranked) / n
    y_bar = sum(y_ranked) / n

    num = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x_ranked, y_ranked))
    den_x = math.sqrt(sum((xi - x_bar) ** 2 for xi in x_ranked))
    den_y = math.sqrt(sum((yi - y_bar) ** 2 for yi in y_ranked))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


def _rank_data(values: list[float]) -> list[float]:
    """Assign ranks to values, handling ties with average ranking."""
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])

    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        # Find all tied values
        while j < len(indexed) and abs(indexed[j][1] - indexed[i][1]) < 1e-10:
            j += 1
        # Assign average rank (1-based)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j

    return ranks


def compute_rho_ci_permutation(
    checkpoints: List[Checkpoint],
    window_size: int = 3,
    n_permutations: int = 1000,
    alpha: float = 0.05,
) -> list[dict]:
    """Compute permutation-based confidence intervals for ρ_t.

    Used in full-scale experiments (not recommended for pilot due to small n).
    """
    import random
    rng = random.Random(42)

    sorted_ckpts = sorted(checkpoints, key=lambda c: c.step)
    results = []

    for i in range(len(sorted_ckpts)):
        start = max(0, i - window_size + 1)
        window = sorted_ckpts[start:i + 1]

        if len(window) < 3:  # need at least 3 for meaningful CI
            results.append({
                "step": sorted_ckpts[i].step,
                "rho_spearman": 0.0,
                "ci_lower": -1.0,
                "ci_upper": 1.0,
            })
            continue

        proxy_vals = [c.proxy_scores.proxy_main for c in window]
        valid_vals = [c.validation_scores.validation_main for c in window]
        original_rho = _spearman_rank(proxy_vals, valid_vals)

        # Permutation test: shuffle validation labels
        perm_rhos = []
        for _ in range(n_permutations):
            shuffled = valid_vals[:]
            rng.shuffle(shuffled)
            perm_rhos.append(_spearman_rank(proxy_vals, shuffled))

        perm_rhos.sort()
        lower_idx = int(n_permutations * alpha / 2)
        upper_idx = int(n_permutations * (1 - alpha / 2))

        results.append({
            "step": sorted_ckpts[i].step,
            "rho_spearman": original_rho,
            "ci_lower": perm_rhos[lower_idx],
            "ci_upper": perm_rhos[upper_idx],
        })

    return results
