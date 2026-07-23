## 5. Discussion

### 5.1 Relation to Our Initial Hypothesis (PAAS)

This investigation began with a concrete hypothesis: that the Spearman
correlation $\rho_{\text{align}}$ between proxy loss and validation quality
undergoes a detectable degradation during training, and that an algorithm
(PAAS) monitoring this correlation could trigger a switch from argmax to
cautious uniform aggregation when degradation is detected.

The algorithm's control logic was verified in a synthetic setting: an
artificially constructed $\rho_{\text{align}}$ trajectory (deliberately
manipulated from high to low) triggered the intended transition from
argmax to cautious uniform aggregation. This confirms that the triggering
and aggregation code executes correctly — it does not constitute evidence
of empirical benefit under realistic conditions. When applied to real
LoRA fine-tuning data, the premise was not supported: $\rho_{\text{align}}$
remained positive throughout training ($0.6$--$0.94$ at $n=6$), and the
observed checkpoint differences were smaller than the evaluation's
minimum detectable difference.

This is not a refutation of the algorithmic approach — it is a finding
about the regime in which checkpoint selection operates for standard
LoRA fine-tuning at practical evaluation budgets. The signal is simply
too weak for any selection method to reliably rank adjacent checkpoints.
PAAS's design may be relevant in settings with larger quality swings
between checkpoints — for instance, longer training runs or smaller
models — though we did not test these conditions and leave this as an
open question.

### 5.2 Implications for Checkpoint Selection Practice

Our central empirical finding — that at $N=200$, the minimum detectable
difference ($9.8\%$) exceeds the observed checkpoint range ($8.0\%$) —
has implications beyond the specific experimental setup:

1. **Compute $\Delta_{\min}$ before selecting.** The minimum detectable
difference is a function of evaluation budget $N$, not of the metric
chosen. Before designing a checkpoint selection strategy, a practitioner
should compute $\Delta_{\min}$ for their $N$. If expected checkpoint
differences are below this threshold, the evaluation cannot resolve them,
and any method relying on point estimates of quality will be selecting
from noise.

2. **Noise-robust aggregation is preferred over noisy argmax.** When
evaluation noise dominates, simple noise-robust strategies (uniform
averaging over a window, model soup) are preferable to argmax over a
noisy metric. This is not because these strategies are "smarter" — it is
because they are less sensitive to the precise ranking of points that
differ by less than $\Delta_{\min}$. (Trap 8 in §4 illustrates this
directly: an apparent cross-task tradeoff was later found to be
indistinguishable from noise at the same evaluation budget.) We did not
empirically verify this claim; it is a logical consequence of the
statistical bound.

3. **Reporting requirements.** Any paper making claims about checkpoint
quality differences should report (a) the held-out set size $N$,
(b) the minimum detectable difference at that $N$, and (c) whether the
observed differences exceed this threshold. Without this information,
apparent "improvements" or "declines" between checkpoints cannot be
distinguished from measurement noise.

### 5.3 Relation to Prior Work

**UGCS (Nguyen et al., 2025)** proposes selecting checkpoints by ranking
their performance on the hardest (most uncertain) samples. Its
effectiveness depends on the per-sample signal being reliable — a
condition that our findings suggest cannot be taken for granted at
typical evaluation sizes. The approach is complementary: UGCS addresses
which samples to use; our finding addresses how many samples are needed
for those rankings to be trustworthy.

**EST (Shihab et al., 2025)** detects proxy gaming through invariance
testing. This is orthogonal to our finding — gaming is an adversarial
phenomenon, while the noise floor we document is a statistical property
of finite-sample evaluation. Both can coexist; a gaming detector and a
noise-aware selection strategy could be combined.

**WARM (Ramé et al., 2024)** averages reward model weights to improve
robustness. Its motivation (reward models are noisy and benefit from
aggregation) aligns with our empirical finding: when individual
evaluations are noisy, aggregation outperforms careful selection from
noisy point estimates.

**Methodological contributions.** Our primary contribution is not a new
selection algorithm but a documented methodology for diagnosing whether
a selection problem exists in the first place. The $\Delta_{\min}$
framework and the eight documented traps constitute a toolkit that
practitioners can apply to their own settings before investing in
sophisticated selection methods. In this sense, the paper is closer to
a "checklist for checkpoint selection" than to a proposal of a specific
selection rule.
