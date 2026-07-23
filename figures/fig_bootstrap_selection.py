"""
Figure: Bootstrap selection probability distribution across 10 checkpoints.
Shows that no single checkpoint dominates selection at N=200.
"""
import matplotlib.pyplot as plt
import numpy as np

steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
probs = [17.9, 44.0, 7.4, 0.1, 0.5, 1.1, 10.7, 7.0, 10.3, 1.0]
accs  = [50.5, 52.0, 50.0, 46.5, 47.0, 48.5, 50.5, 50.5, 51.0, 49.5]

fig, ax = plt.subplots(figsize=(8, 4.5))

colors = ['#2c7bb6' if p > 10 else '#abd9e9' for p in probs]
bars = ax.bar([str(s) for s in steps], probs, color=colors, edgecolor='gray', linewidth=0.5)

for bar, p, a in zip(bars, probs, accs):
    if p > 5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{a:.1f}%', ha='center', fontsize=8, color='#333')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{a:.1f}%', ha='center', fontsize=7, color='#999')

ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.text(9.5, 10.8, 'Random baseline (10%)', fontsize=8, color='gray')

ax.set_xlabel('Checkpoint step', fontsize=11)
ax.set_ylabel('Selection probability (%)', fontsize=11)
ax.set_title('Bootstrap checkpoint selection distribution (N=200, 10,000 resamples)', fontsize=11)
ax.set_ylim(0, 55)
ax.set_xticks(range(len(steps)))
ax.set_xticklabels([str(s) for s in steps])

plt.tight_layout()
fig.savefig('fig_bootstrap_selection.png', dpi=300)
fig.savefig('fig_bootstrap_selection.pdf')
plt.show()
