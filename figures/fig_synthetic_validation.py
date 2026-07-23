"""
Synthetic validation: true gap vs selection accuracy, three N levels.
Shows Δ_min predicts the transition point.
"""
import matplotlib.pyplot as plt
import numpy as np, math

gaps = np.array([0, 0.5, 1, 1.5, 2, 3, 5, 8, 12, 20])
results = {
    200: [46.5, 53.2, 55.3, 60.2, 63.5, 70.9, 82.7, 93.8, 99.1, 100.0],
    500: [47.2, 54.8, 61.7, 67.7, 73.3, 81.7, 94.3, 99.3, 100.0, 100.0],
    1000: [50.2, 58.5, 65.9, 74.0, 81.2, 90.5, 99.0, 100.0, 100.0, 100.0],
}

def delta_min(N, p=0.5, z=1.96):
    return z * math.sqrt(2*p*(1-p)/N)

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = {200: '#d7191c', 500: '#fdae61', 1000: '#2c7bb6'}

for N in [200, 500, 1000]:
    ax.plot(gaps, results[N], 'o-', color=colors[N], linewidth=1.8, markersize=6,
            label=f'N={N} (Δ_min={delta_min(N)*100:.1f}pp)')
    ax.axvline(x=delta_min(N)*100, color=colors[N], linestyle='--', alpha=0.3, linewidth=1)

ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5)
ax.text(20.5, 95, '95% selection accuracy', fontsize=8, color='gray', va='center')

ax.set_xlabel('True accuracy gap (pp)', fontsize=11)
ax.set_ylabel('Best checkpoint selection accuracy (%)', fontsize=11)
ax.set_xlim(-0.5, 22)
ax.set_ylim(40, 102)
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.15)

plt.tight_layout()
fig.savefig('fig_synthetic_validation.png', dpi=300)
fig.savefig('fig_synthetic_validation.pdf')
plt.show()
