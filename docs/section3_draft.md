## 3. The Core Empirical Finding

> **Takeaway**: Under a frozen experimental protocol with adequate statistical
> controls, we find that GSM8K accuracy across 500 steps of LoRA fine-tuning
> is statistically indistinguishable from noise. The observed range of 8.0
> percentage points is below the minimum detectable difference of 9.8\% for our
> evaluation size (N=200). This provides a quantitative lower bound on the
> checkpoint-quality signal available for selection.

### 3.1 Experimental Protocol

After the diagnostic process documented in §4, we ran a clean experiment
(v2_cosine_seed42) under the following frozen definitions:

- **Model**: Qwen2.5-7B-Instruct, LoRA rank 16, 500 steps, cosine schedule
- **Training data**: GSM8K + CodeAlpaca + Dolly (40K mixed instruction examples)
- **Checkpoints**: 10, saved every 50 steps (steps 50--500)
- **Proxy signal**: Held-out loss on question+answer concatenation (frozen definition, see Trap 3)
- **Validation signal**: GSM8K accuracy on 200 test questions with **few-shot prompting** (3 examples demonstrating the `####<answer>` format; see Trap 6)
- **Data provenance**: All results are `data_source: "real_gpu"`, with no synthetic fallback

### 3.2 GSM8K Trajectory

Table 1 shows the full trajectory:

```
+--------+----------+
|  Step  |  GSM8K   |
+--------+----------+
|   50   |  47.5\%  |
|  100   |  54.5\%  |
|  150   |  55.5\%  |
|  200   |  49.0\%  |
|  250   |  54.0\%  |
|  300   |  55.0\%  |
|  350   |  51.5\%  |
|  400   |  50.5\%  |
|  450   |  52.0\%  |
|  500   |  50.0\%  |
+--------+----------+
```

**Table 1**: GSM8K accuracy across 10 checkpoints from a single 500-step
LoRA fine-tuning run (v2_cosine_seed42). Each value is based on N=200
test questions with few-shot prompting.

### 3.3 Statistical Analysis

We assess the trajectory using three complementary tests.

**Trend test.** The Spearman rank correlation between training step and
GSM8K accuracy is $\rho_{\text{trend}} = -0.091$ ($n = 10$). This is
indistinguishable from zero, indicating no monotonic improvement or
degradation across training.

**Adjacent comparison test.** For each adjacent pair of checkpoints,
we perform an independent two-proportion z-test (the evaluation questions
differ in practice due to generation nondeterminism, making this a
conservative choice). Results:

```
+-----------------+--------+--------+-------+
|    Pair         |  Δ     |  z     |  p    |
+-----------------+--------+--------+-------+
|  50  → 100     | +7.0\% |  1.40  | 0.161 |
| 100  → 150     | +1.0\% |  0.20  | 0.841 |
| 150  → 200     | -6.5\% | -1.30  | 0.193 |
| 200  → 250     | +5.0\% |  1.00  | 0.317 |
| 250  → 300     | +1.0\% |  0.20  | 0.841 |
| 300  → 350     | -3.5\% | -0.70  | 0.483 |
| 350  → 400     | -1.0\% | -0.20  | 0.841 |
| 400  → 450     | +1.5\% |  0.30  | 0.764 |
| 450  → 500     | -2.0\% | -0.40  | 0.689 |
+-----------------+--------+--------+-------+
```

**Table 2**: Independent two-proportion z-tests for each adjacent checkpoint
pair. No comparison reaches statistical significance at $\alpha = 0.05$.
The smallest $p$-value is 0.161 (step 50 vs 100).

Since our claim is the absence of significant differences (rather than
the presence), multiple comparison correction would only make the tests
more conservative, further supporting this conclusion. We report
uncorrected $p$-values for transparency.

**Range test.** The best checkpoint (step 150, 55.5\%) differs from the worst
(step 50, 47.5\%) by 8.0 percentage points. The two-proportion test gives
$z = 1.60$, $p = 0.109$ — not significant at $\alpha = 0.05$. Because this
compares the extremal pair selected from 10 checkpoints (a post-hoc
selection), the effective significance threshold is stricter than
$\alpha = 0.05$; the reported $p$-value therefore understates the
evidence for no difference.

### 3.4 The Minimum Detectable Difference

The lack of significance is not a statement about the model — it is a
statement about the evaluation's resolving power. For a held-out set of
size $N$ with accuracy $\bar{p}$, the minimum detectable difference
between two checkpoints (two-proportion test, $\alpha = 0.05$) is:

\[
\Delta_{\min} = z_{\alpha/2} \cdot \sqrt{ \frac{2 \bar{p} (1 - \bar{p})}{N} }
\]

For $N = 200$ and $\bar{p} \approx 0.5$, this gives $\Delta_{\min} = 9.8\%$.
The observed range of 8.0\% falls below this threshold. Table 3 shows
$\Delta_{\min}$ for common evaluation sizes:

```
+--------+-----------+
|  N     | Δ_min     |
+--------+-----------+
|  100   |  13.9\%   |
|  200   |   9.8\%   |
|  500   |   6.2\%   |
| 1000   |   4.4\%   |
+--------+-----------+
```

**Table 3**: Minimum detectable difference between two checkpoints as a
function of held-out set size N (independent two-proportion test,
$\alpha = 0.05$, $\bar{p} = 0.5$).

A paired test (McNemar) would provide a tighter bound but requires
per-sample correctness records, which our evaluation pipeline did not
retain — this is itself a methodological limitation (see §6).

### 3.5 Implications

This finding is not that "our fine-tuning failed" or that "the proxy was
bad." It is that **at N=200, the evaluation system lacks the statistical
power to rank checkpoints meaningfully**, regardless of the metric used.

This applies to held-out accuracy, F1 score, or any metric computed on
a finite sample. The noise floor is a property of the evaluation budget
(N), not the choice of metric. Any selection method — proxy argmax,
validation argmax, or otherwise — that relies on ranking checkpoints by
a metric evaluated on N samples faces this same limitation.

The minimum detectable difference is a quantity any practitioner can
compute before designing a checkpoint selection strategy. If the expected
quality differences between checkpoints are smaller than $\Delta_{\min}$
at their evaluation budget, then checkpoint ranking is selecting from
noise, not from signal. This constrains any method that relies on point
estimates of quality at this evaluation budget — even adaptive selection
methods must ultimately resolve differences at or above $\Delta_{\min}$
to make a reliable decision.
