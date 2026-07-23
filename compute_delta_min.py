#!/usr/bin/env python3
"""Δ_min: Pre-experimental resolution diagnostic for model selection.

Usage:
  python3 compute_delta_min.py --N 200 --p_bar 0.5
  python3 compute_delta_min.py --N 1319 --p_bar 0.5 --alpha 0.05

For paired (McNemar-based) threshold, provide discordant proportion:
  python3 compute_delta_min.py --N 200 --p_d 0.35
"""

import argparse
from math import sqrt
from scipy.stats import norm


def delta_min_indep(N, p_bar=0.5, alpha=0.05):
    """Independent two-proportion bound (Eq. 2). Conservative pre-experimental diagnostic."""
    z = norm.ppf(1 - alpha / 2)
    return z * sqrt(2 * p_bar * (1 - p_bar) / N) * 100  # in pp


def delta_min_paired(N, p_d, alpha=0.05):
    """Paired McNemar-based threshold (Eq. 1). Tighter, requires discordant proportion."""
    z = norm.ppf(1 - alpha / 2)
    return z * sqrt(p_d / N) * 100  # in pp


def quick_reference_table():
    """Print the Δ_min quick-reference table for common N and p_bar values."""
    print(f"{'N':>6} {'p=0.50':>8} {'p=0.60':>8} {'p=0.70':>8} {'p=0.80':>8}")
    print("-" * 42)
    for N in [100, 200, 500, 1000, 5000, 10000]:
        vals = [delta_min_indep(N, p, alpha=0.05) for p in [0.5, 0.6, 0.7, 0.8]]
        print(f"{N:>6} {vals[0]:>7.1f}pp {vals[1]:>7.1f}pp {vals[2]:>7.1f}pp {vals[3]:>7.1f}pp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Δ_min resolution diagnostic")
    parser.add_argument("--N", type=int, required=True, help="Evaluation budget")
    parser.add_argument("--p_bar", type=float, default=0.5, help="Baseline accuracy")
    parser.add_argument("--p_d", type=float, default=None, help="Discordant proportion (paired)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    args = parser.parse_args()

    indep = delta_min_indep(args.N, args.p_bar, args.alpha)
    print(f"Δ_min (independent) = {indep:.2f} pp  (N={args.N}, p̄={args.p_bar}, α={args.alpha})")

    if args.p_d is not None:
        paired = delta_min_paired(args.N, args.p_d, args.alpha)
        print(f"Δ_min (paired)      = {paired:.2f} pp  (N={args.N}, p_d={args.p_d}, α={args.alpha})")
        print(f"Reduction: {(1 - paired/indep) * 100:.0f}%")

    print()
    print("Resolution check: if your observed accuracy range < Δ_min, the ranking")
    print("is not statistically resolvable at this evaluation budget.")
