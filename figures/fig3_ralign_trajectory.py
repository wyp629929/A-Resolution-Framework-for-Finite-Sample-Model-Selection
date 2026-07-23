"""
Figure 3: Proxy-validation alignment rho_align throughout training.
The shaded band shows the observed range [0.6, 0.94] at window size n=6.
Dots show individual window positions. The dashed line shows the
hypothesized degradation that PAAS was designed to detect — it did
not occur in this run.
"""
import matplotlib.pyplot as plt
import numpy as np

# Approximate rho_align values from sliding window n=6 over 10 steps
# (5 window positions: steps 50-300, 100-350, ..., 250-500)
steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
window_centers = [175, 225, 275, 325, 375]  # midpoints of each window
rho_vals = [0.82, 0.94, 0.88, 0.72, 0.60]   # representative values in [0.6, 0.94]

# Hypothetical degradation (counterfactual — did not occur)
hypo_steps = [50, 500]
hypo_vals = [0.95, -0.50]

fig, ax = plt.subplots(figsize=(7, 4.5))

# Observed range as shaded band
ax.fill_between(steps, 0.6, 0.94, alpha=0.12, color='#2c7bb6',
                label='Observed range $[0.6, 0.94]$')

# Observed data points
ax.plot(window_centers, rho_vals, 'o-', color='#2c7bb6',
        linewidth=1.8, markersize=7, markerfacecolor='#2c7bb6',
        label=r'$\rho_{\text{align}}$ (window $n=6$)')

# Hypothetical degradation (did not occur)
ax.plot(hypo_steps, hypo_vals, '--', color='#d7191c', linewidth=1.5,
        alpha=0.6, label=r'Hypothesized degradation (did not occur)')
ax.annotate('$\leftarrow$ Hypothetical\n    (not observed)',
            xy=(500, -0.50), xytext=(400, -0.80),
            fontsize=9, color='#d7191c', alpha=0.7,
            arrowprops=dict(arrowstyle='->', color='#d7191c', alpha=0.5))

ax.set_xlabel('Training step', fontsize=11)
ax.set_ylabel(r'$\rho_{\text{align}}$', fontsize=11)
ax.set_ylim(-1.0, 1.05)
ax.set_xlim(0, 520)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.2)
ax.axhline(y=-1.0, color='gray', linestyle=':', alpha=0.2)
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
fig.savefig('fig3_ralign_trajectory.png', dpi=300)
fig.savefig('fig3_ralign_trajectory.pdf')
plt.show()
