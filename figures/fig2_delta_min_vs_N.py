"""
Figure 2: Minimum detectable difference (Δ_min) as a function of held-out set size N.

Shows how many samples are needed to detect a given quality difference between
two checkpoints, using the independent two-proportion test (α=0.05).

Usage:
    python fig2_delta_min_vs_N.py

Output:
    fig2_delta_min_vs_N.png (300 DPI)
    fig2_delta_min_vs_N.pdf
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Δ_min formula: independent two-proportion test, α=0.05
# Δ_min = z_{α/2} * sqrt(2 * p̄ * (1-p̄) / N)
# ============================================================
z = 1.96
N_vals = np.linspace(50, 2000, 500)

baseline_accuracies = [0.50, 0.60, 0.70]
colors = ['#2c7bb6', '#fdae61', '#d7191c']
labels = ['p̄ = 50%', 'p̄ = 60%', 'p̄ = 70%']

fig, ax = plt.subplots(figsize=(7, 5))

# Pre-compute curves for fill
deltas = {}
for p_bar, c, lab in zip(baseline_accuracies, colors, labels):
    delta = z * np.sqrt(2 * p_bar * (1 - p_bar) / N_vals) * 100  # pp
    deltas[lab] = delta
    ax.plot(N_vals, delta, color=c, linewidth=1.8, label=lab)

# Undetectable region: fill below the most conservative (highest) Δ_min curve
# The p̄=50% curve gives the largest Δ_min → most conservative bound
delta_max = z * np.sqrt(2 * 0.50 * (1 - 0.50) / N_vals) * 100
ax.fill_between(N_vals, 0, delta_max, alpha=0.06, color='gray')
# Edge label for the region
ax.text(1020, delta_max[len(N_vals)//3] / 2,
        'Undetectable\nregion', fontsize=9, color='#999',
        ha='center', va='center', alpha=0.8,
        style='italic')

# Key reference points
ref_Ns = [200, 500, 1000, 1500, 2000]
p_bar_mid = 0.50  # match Table 3 (p̄ = 0.5)
for n in ref_Ns:
    d = z * np.sqrt(2 * p_bar_mid * (1 - p_bar_mid) / n) * 100
    ax.annotate(f'N={n}\nΔ={d:.1f}pp',
                xy=(n, d),
                xytext=(n + 60, d + 0.3),
                fontsize=8, color='#555',
                arrowprops=dict(arrowstyle='->', color='#999', lw=0.6))

# Highlight our experimental setting
ax.plot(200, 9.8, 'o', color='#333', markersize=8, zorder=5)
ax.annotate('Our setting\nN=200, Δ_min=9.8pp',
            xy=(200, 9.8),
            xytext=(280, 10.5),
            fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc',
                      edgecolor='#ccc', alpha=0.8))

# Observed range in our experiment
observed_range = 8.0
ax.axhline(y=observed_range, color='gray', linestyle='--', linewidth=1.0,
           alpha=0.7)
ax.annotate(f'Observed range\n= {observed_range} pp',
            xy=(1850, observed_range),
            fontsize=9, color='#666',
            va='bottom', ha='right')

# Intersection: the minimum N where observed range becomes detectable
# N = (z / Δ)² · 2 · p̄ · (1-p̄)
n_intersect = (z / (observed_range / 100)) ** 2 * 2 * 0.50 * (1 - 0.50)
ax.axvline(x=n_intersect, color='gray', linestyle=':', linewidth=0.8,
           alpha=0.5)
ax.annotate(f'N ≈ {n_intersect:.0f}',
            xy=(n_intersect, 1.0),
            fontsize=8, color='#666',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#ccc', alpha=0.8))

# Axis labels and limits
ax.set_xlabel('Held-out set size $N$', fontsize=12)
ax.set_ylabel(r'Minimum detectable difference $\Delta_{\min}$ (pp)', fontsize=12)
ax.set_xlim(0, 2050)
ax.set_ylim(0, None)
ax.set_xticks(np.arange(0, 2101, 250))
ax.tick_params(axis='both', labelsize=10)

ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# ============================================================
# Save
# ============================================================
plt.tight_layout()
fig.savefig('fig2_delta_min_vs_N.png', dpi=300)
fig.savefig('fig2_delta_min_vs_N.pdf')
plt.show()
