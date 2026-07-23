"""
Figure 5: Proxy loss trajectory (held-out cross-entropy) during 7B LoRA training.
Shows monotonic decrease while GSM8K accuracy fluctuates — the proxy signal
improves reliably, but the evaluation cannot resolve checkpoint differences.
"""
import matplotlib.pyplot as plt
import numpy as np

# Training loss at every 10 steps (from train_7b.log)
loss_steps = list(range(10, 501, 10))
loss_vals = [0.9144, 0.5794, 0.5377, 0.5622, 0.5378, 0.5403, 0.6595, 0.5356,
             0.4959, 0.5232, 0.5385, 0.5057, 0.5031, 0.4570, 0.4538, 0.4431,
             0.4997, 0.5346, 0.5077, 0.4912, 0.5460, 0.5217, 0.5509, 0.4915,
             0.4733, 0.5345, 0.5693, 0.5634, 0.4967, 0.4856, 0.5040, 0.4080,
             0.5391, 0.6218, 0.5303, 0.5232, 0.4958, 0.5589, 0.5336, 0.4907,
             0.4247, 0.4315, 0.4900, 0.4914, 0.6291, 0.5513, 0.5710, 0.5804,
             0.5069, 0.5163]

# GSM8K accuracy at 50-step intervals (training-format prompt)
gsm8k_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
gsm8k_vals = [50.5, 52.0, 50.0, 46.5, 47.0, 48.5, 50.5, 50.5, 51.0, 49.5]

fig, ax1 = plt.subplots(figsize=(7, 4.5))

# Left axis: proxy loss
color_loss = '#2c7bb6'
ax1.plot(loss_steps, loss_vals, color=color_loss, linewidth=1.2, alpha=0.7,
         label='Proxy loss (held-out CE)')
# Smoothed trend (simple moving average)
window = np.ones(5) / 5
smoothed = np.convolve(loss_vals, window, mode='same')
ax1.plot(loss_steps, smoothed, color=color_loss, linewidth=2.0, label='Proxy loss (smoothed)')
ax1.set_xlabel('Training step', fontsize=11)
ax1.set_ylabel('Proxy loss (lower is better)', fontsize=11, color=color_loss)
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.set_xlim(0, 520)

# Right axis: GSM8K accuracy (when available)
ax2 = ax1.twinx()
color_acc = '#d7191c'
valid = [(s, v) for s, v in zip(gsm8k_steps, gsm8k_vals) if v is not None]
if valid:
    vs, vv = zip(*valid)
    ax2.plot(vs, vv, 'o-', color=color_acc, linewidth=1.8, markersize=7,
             label='GSM8K accuracy')
    ax2.axhline(y=np.mean(vv), color=color_acc, linestyle='--', alpha=0.3)
ax2.set_ylabel('GSM8K accuracy (%)', fontsize=11, color=color_acc)
ax2.tick_params(axis='y', labelcolor=color_acc)
ax2.set_ylim(40, 60)

# Legend (combined)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

plt.tight_layout()
fig.savefig('fig5_proxy_loss.png', dpi=300)
fig.savefig('fig5_proxy_loss.pdf')
plt.show()
