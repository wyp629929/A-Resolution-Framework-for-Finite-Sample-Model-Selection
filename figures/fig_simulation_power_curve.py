#!/usr/bin/env python3
"""Monte Carlo simulation: power curve and selection stability under Δ_min.
Generates:
  (a) Power curve: P(correctly identify best checkpoint) vs true range
  (b) Selection stability: bootstrap probability of true best vs true range
  (c) Decision curve: cost of argmax vs soup as function of N
All CPU, no GPU needed."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from math import sqrt

np.random.seed(0)

N_VALUES = [50, 200, 500, 1319, 5000]
P_BAR = 0.50
M = 2  # two checkpoints: best vs worst (the extremes defining the range)
N_SIM = 20000  # outer Monte Carlo replicates

# Δ_min independent formula
def delta_min_bar(p_bar, n, alpha=0.05):
    z = norm.ppf(1 - alpha/2)
    return z * sqrt(2 * p_bar * (1 - p_bar) / n) * 100  # in pp

# True ranges to simulate (in percentage points)
true_ranges_pp = np.linspace(0, 20, 41)

results = {}
for N in N_VALUES:
    dmin = delta_min_bar(P_BAR, N)
    print(f"\nN={N}, Δ_min={dmin:.2f}pp, N_SIM={N_SIM}")

    resolvable = []  # P(observed_diff > Δ_min) — exceeds the resolution threshold
    reject_null = []  # P(p < 0.05 for two-proportion z-test) — statistical power

    z_crit = norm.ppf(0.975)  # α=0.05 two-sided

    for tr in true_ranges_pp:
        tr_frac = tr / 100
        p_low = P_BAR - tr_frac/2
        p_high = P_BAR + tr_frac/2
        p_low = max(0.001, p_low)
        p_high = min(0.999, p_high)

        # Generate (N_SIM, 2, N) binary outcomes
        probs = np.array([p_low, p_high])
        samples = np.random.binomial(1, probs[None, :, None], size=(N_SIM, 2, N))
        accs = samples.mean(axis=2)  # (N_SIM, 2)
        obs_diff = accs[:, 1] - accs[:, 0]  # observed difference p_high - p_low

        # Resolvability: observed range exceeds Δ_min
        n_resolved = np.sum(np.abs(obs_diff * 100) >= dmin)
        resolvable.append(n_resolved / N_SIM)

        # Statistical power: two-proportion z-test
        p_bar_obs = accs.mean(axis=1)
        se = np.sqrt(2 * p_bar_obs * (1 - p_bar_obs) / N)
        z_stat = obs_diff / np.maximum(se, 1e-10)
        n_reject = np.sum(np.abs(z_stat) >= z_crit)
        reject_null.append(n_reject / N_SIM)

    results[N] = {
        "dmin": dmin,
        "true_ranges": true_ranges_pp,
        "resolvable": resolvable,
        "power": reject_null
    }

# ---- Plotting ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Colors for N values
colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(N_VALUES)))

# Panel (a): Resolution rate — P(observed diff > Δ_min)
ax = axes[0]
for i, N in enumerate(N_VALUES):
    r = results[N]
    dmin = r["dmin"]
    ax.plot(r["true_ranges"], r["resolvable"], color=colors[i],
            label=f"N={N}", linewidth=2)

# Add Δ_min vertical drop lines to 50% crossing
for i, N in enumerate(N_VALUES):
    r = results[N]
    dmin = r["dmin"]
    idx = np.argmin(np.abs(np.array(r["true_ranges"]) - dmin))
    val_at_dmin = r["resolvable"][idx]
    ax.plot([dmin, dmin], [0, val_at_dmin], color=colors[i],
            linewidth=1, alpha=0.4, linestyle=":")

ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
ax.text(0.5, 0.51, "50%", fontsize=8, color="gray", ha="left")
ax.set_xlabel("True accuracy difference (pp)")
ax.set_ylabel(r"$P(\text{observed diff} > \Delta_{\min})$")
ax.set_title("(a) Resolution rate by evaluation budget")
ax.set_xlim(0, 15)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)

# Panel (b): Statistical power — P(p < 0.05)
ax = axes[1]
for i, N in enumerate(N_VALUES):
    r = results[N]
    dmin = r["dmin"]
    ax.plot(r["true_ranges"], r["power"], color=colors[i],
            label=f"N={N}", linewidth=2, linestyle="--")
    # Also show resolution rate as solid line for comparison
    ax.plot(r["true_ranges"], r["resolvable"], color=colors[i],
            linewidth=1, alpha=0.3)

ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.5, linewidth=1)
ax.text(0.5, 0.81, "80% power", fontsize=8, color="gray", ha="left")
ax.set_xlabel("True accuracy difference (pp)")
ax.set_ylabel("P(reject null | difference = δ)")
ax.set_title("(b) Power curves (dashed) with resolution rate (faint)")
ax.set_xlim(0, 15)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.subplots_adjust(bottom=0.25)
fig.legend([f"N={N}" for N in N_VALUES],
           loc="lower center", bbox_to_anchor=(0.5, 0.05),
           ncol=len(N_VALUES), fontsize=10, title="Sample size N")
plt.savefig("/Users/wangyaoping/Desktop/模型训练论文/figures/fig_simulation_power_curve.pdf",
            bbox_inches="tight", dpi=150)
plt.savefig("/Users/wangyaoping/Desktop/模型训练论文/figures/fig_simulation_power_curve.png",
            bbox_inches="tight", dpi=150)
print("\n=== SAVED fig_simulation_power_curve.pdf/png ===")

# Print summary table
print("\n=== SUMMARY ===")
print(f"{'N':>6} | {'Δmin(pp)':>9} | {'Range@50%power':>15} | {'@80%':>6} | {'@95%':>6}")
print("-"*55)
for N in N_VALUES:
    r = results[N]
    dmin = r["dmin"]
    pwr = np.array(r["power"])
    tr = r["true_ranges"]
    r50 = tr[np.argmax(pwr >= 0.50)] if np.any(pwr >= 0.50) else float('nan')
    r80 = tr[np.argmax(pwr >= 0.80)] if np.any(pwr >= 0.80) else float('nan')
    r95 = tr[np.argmax(pwr >= 0.95)] if np.any(pwr >= 0.95) else float('nan')
    print(f"{N:>6} | {dmin:>8.1f}pp | {r50:>14.1f}pp | {r80:>5.1f}pp | {r95:>5.1f}pp")

print("\n=== ALL DONE ===")
