"""
Figure 1: GSM8K accuracy trajectory across 10 checkpoints (50-500 steps).
"""
import matplotlib.pyplot as plt
import numpy as np

steps     = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
accuracy  = [47.5, 54.5, 55.5, 49.0, 54.0, 55.0, 51.5, 50.5, 52.0, 50.0]
N         = 200

se = [np.sqrt(p/100 * (1 - p/100) / N) * 100 for p in accuracy]

z = 1.96
p_bar = np.mean(accuracy) / 100
delta_min = z * np.sqrt(2 * p_bar * (1 - p_bar) / N) * 100

fig, ax = plt.subplots(figsize=(7, 5))

ax.errorbar(steps, accuracy, yerr=se,
            fmt='o-', capsize=4, capthick=1.2,
            color='#2c7bb6', markersize=8, linewidth=1.5,
            label='GSM8K accuracy (N=200)')

ax.axhspan(delta_min/2, -delta_min/2,
           xmin=0, xmax=1,
           color='gray', alpha=0.12, zorder=0)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

best_idx  = np.argmax(accuracy)
worst_idx = np.argmin(accuracy)
ax.annotate(f'Best: step {steps[best_idx]}\n({accuracy[best_idx]:.1f}%)',
            xy=(steps[best_idx], accuracy[best_idx]),
            xytext=(steps[best_idx] + 30, accuracy[best_idx] + 4),
            arrowprops=dict(arrowstyle='->', color='#333', lw=0.8),
            fontsize=9, va='bottom')
ax.annotate(f'Worst: step {steps[worst_idx]}\n({accuracy[worst_idx]:.1f}%)',
            xy=(steps[worst_idx], accuracy[worst_idx]),
            xytext=(steps[worst_idx] + 15, accuracy[worst_idx] - 6),
            arrowprops=dict(arrowstyle='->', color='#333', lw=0.8),
            fontsize=9, va='top')

ax.annotate(f'Δ_min = {delta_min:.1f} pp\n(α=0.05, N=200)',
            xy=(400, delta_min/2),
            fontsize=9, color='#666',
            va='bottom', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#ccc', alpha=0.8))

ax.set_xlabel('Training step', fontsize=12)
ax.set_ylabel('GSM8K accuracy (%)', fontsize=12)
ax.set_xlim(30, 520)
ax.set_ylim(35, 65)
ax.set_xticks(steps)
ax.tick_params(axis='both', labelsize=10)

ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

plt.tight_layout()
fig.savefig('fig1_gsm8k_trajectory.png', dpi=300)
fig.savefig('fig1_gsm8k_trajectory.pdf')
plt.show()
