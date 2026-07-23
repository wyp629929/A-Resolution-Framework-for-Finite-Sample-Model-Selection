# Diagnostic Paper Outline

## Working Title

**When Checkpoints Are Noise: A Methodological Autopsy of Proxy-Based Selection in LLM Fine-Tuning**

*Alternative: Seven Traps and a Falsification: What We Learned Trying to Detect Proxy-Quality Misalignment in LLM Fine-Tuning*

---

## Core Claim

In standard LLM fine-tuning (7B LoRA, 500 steps, multi-task), the magnitude of evaluation noise from held-out sets of typical size (200 samples) **systematically exceeds** the magnitude of real quality differences between adjacent checkpoints. This means that any selection method based on ranking checkpoints by a noisy metric is selecting from noise, not from signal — regardless of whether the metric is proxy loss or task-specific accuracy.

This finding is not a "failure" — it is a measurable quantity that practitioners can compute for their own setting and use to determine whether checkpoint selection is meaningful at their evaluation budget.

---

## Suggested Structure

### 1. Introduction (~1.5 pages)

**Hook**: "Given 10 checkpoints from an LLM fine-tuning run, how confidently can you pick the best one?"

**Problem**: Standard practice picks the lowest-loss checkpoint. We set out to study when and why this fails. What we found was different — the evaluation resolution itself is too coarse to distinguish checkpoints reliably at practical held-out set sizes.

**Research questions** (drive the paper, no PAAS backstory needed):
- RQ1: Given a held-out evaluation set of size N, what is the minimum quality difference between checkpoints that can be reliably detected?
- RQ2: In a standard LoRA fine-tuning run, how do actual checkpoint differences compare to this threshold?
- RQ3: What analytical traps can lead to false conclusions about proxy-quality alignment?

### 2. Background and Motivation (~1 page)

- Why checkpoint selection matters (deployment decisions)
- The intuitive case for proxy-based selection (loss monotonically decreases, so lower loss = better model)
- Known limitations (Instruct-SkillMix: "validation loss does not always correspond to generation quality")
- PAAS as our starting hypothesis: monitor ρ_align between proxy and validation, trigger cautious selection when ρ_align degrades
- Brief experimental setup: Qwen2.5-7B, LoRA, 500 steps, multi-task data

### 3. The Core Empirical Finding (~2 pages)

**Clean statistical result, the paper's centerpiece.**

3.1 The GSM8K trajectory (v2, frozen definitions, few-shot):

| Step | GSM8K |
|------|-------|
| 50 | 47.5% |
| 100 | 54.5% |
| 150 | 55.5% |
| 200 | 49.0% |
| 250 | 54.0% |
| 300 | 55.0% |
| 350 | 51.5% |
| 400 | 50.5% |
| 450 | 52.0% |
| 500 | 50.0% |

3.2 Statistical analysis:
- Spearman ρ_trend = -0.091 (n=10): no monotonic trend
- All adjacent differences: p > 0.15 (independent two-proportion test)
- Best (55.5%) vs worst (47.5%): p = 0.109 (independent two-proportion test)
- Minimum detectable difference (α=0.05, n=200): **9.8%** (independent two-proportion test)
- Observed max range: **8.0%** (below the minimum detectable threshold)

3.3 The "Minimum Detectable Difference" as a general tool:
- Formula: Δ_min = z_{α/2} · √(2·p̄·(1-p̄)/n) (independent two-proportion test)
- Table for common held-out sizes: n=100 → 13.9%, n=200 → 9.8%, n=500 → 6.2%, n=1000 → 4.4%
- A paired test (McNemar) would provide a tighter bound but requires per-sample correctness data, which our evaluation pipeline did not save — a lesson for future experiments (Limitations §6)
- Usage: before running a checkpoint selection experiment, compute Δ_min for your N. If checkpoint differences are below this threshold, the evaluation cannot resolve them regardless of the metric chosen.
- This is a contribution that does not depend on our specific experimental results — it's a tool any practitioner can use.

3.4 Why this matters (not a failure, a measurement):
- The finding is not "our proxy was bad" but "at N=200, the evaluation system lacks statistical power to rank checkpoints."
- This applies regardless of whether the evaluation metric is proxy loss, accuracy, or F1 — it's a property of sample size, not metric choice.

### 4. Eight Analytical Traps Encountered (~4 pages)

**Not a laundry list — each trap is a case study of a specific artifact that can (and did) produce false conclusions.**

4.1 Organization: three categories of traps

| Category | Traps | Theme |
|----------|-------|-------|
| A. Statistical artifacts | T1 (Spearman n=3), T8 (Hypothesis 2) | Small-n statistics create false patterns |
| B. Engineering failures | T2 (synthetic fallback), T5 (GPU fragmentation) | Silent code paths corrupt data |
| C. Definitional ambiguity | T3 (proxy definition), T4 (proxy_main priority), T6/T7 (evaluation format) | Undefined terms produce contradictory results |

4.2 Each trap written in consistent format (~0.5 page each):
- Phenomenon (what we saw)
- Initial interpretation (what we thought)
- How discovered (was it an audit? A deeper check?)
- Root cause (one clear paragraph)
- Lesson for the field (generalizable, not specific to our setup)

### 5. Discussion (~1.5 pages)

5.1 **The PAAS story (briefly)**:
We designed PAAS to detect ρ_align degradation. The algorithm was sound, but the premise — that ρ_align undergoes detectable degradation in typical LoRA fine-tuning — was not supported by the data. This is not a refutation of the approach; it is a finding about the regime in which checkpoint selection operates (the differences are smaller than the measurement noise).

5.2 **Implications for checkpoint selection practice**:
- If your held-out set is N < 500, the statistical power to rank checkpoints may be insufficient regardless of your metric
- Compute Δ_min before designing your selection strategy
- "Pick the lowest loss" is not just naive — in low-N regimes, it's selecting from noise
- Noise-robust aggregation (model soup, uniform averaging) should theoretically outperform noisy argmax in low-N regimes, though this specific claim requires dedicated empirical verification beyond this work

5.3 **Relation to prior work**:
- UGCS (Nguyen et al.): uncertainty-based selection, assumes per-sample signals are reliable — our finding suggests sample-level reliability itself depends on N
- EST (Shihab et al.): proxy gaming detection — orthogonal, but their evaluation pipeline should also report Δ_min
- WARM (Ramé et al.): weight averaging — consistent with our finding that uniform aggregation is noise-robust

### 6. Limitations (~0.5 page)

- Single model scale (7B), single training regime (LoRA, 500 steps)
- Single task type (math reasoning; code evaluation was attempted (CodeAlpaca) but the evaluation pipeline itself had unresolved issues — multi-language outputs, inconsistent formatting, sandbox failures — preventing reliable measurement; this is documented as Trap 7)
- The Δ_min framework uses the independent two-proportion test, which assumes unpaired samples. A paired test (McNemar) would be more appropriate for our setting (same questions evaluated across all checkpoints), but applying it requires per-sample correctness records. Our evaluation pipeline stored only aggregate accuracy per checkpoint — this is itself a methodological limitation and a lesson for future work: evaluation scripts should always retain per-sample results alongside aggregate metrics
- Our conclusions about N=200 are specific to GSM8K; other tasks may have different noise properties

### 7. Conclusion (~0.3 page)

Checkpoint selection is only meaningful when the evaluation signal exceeds the noise floor. We quantified this noise floor for a typical fine-tuning setup and found it dominates. We provide (a) a reusable tool (Δ_min), (b) a catalog of eight analytical traps that produce false conclusions in this domain, and (c) the specific finding that at N=200, adjacent checkpoints in LoRA fine-tuning are statistically indistinguishable.

---

## Figures and Tables

| # | Content | Type | Note |
|---|---------|------|------|
| Fig 1 | GSM8K trajectory | Scatter + error bars (±SE) | Visual, not redundant with Table 1 |
| Fig 2 | Δ_min(N) curve | Line plot | N=50 to 1000, both tests shown |
| Table 1 | Pairwise significance tests | Table | Adjacent comparisons, p-values, Δ_min |
| Table 2 | Δ_min for common N sizes | Table | Both independent and paired bounds |
| Table 3 | Summary of 8 traps | Table | Categorized by type |

---

## Open Question

> The central question for the outline: should Section 3 (core empirical finding) or Section 4 (8 traps) come first?

My recommendation: **Section 3 first**. Lead with the clean statistical finding (Δ_min > observed range). The traps then serve as supporting evidence — "and here is why getting to this finding was hard." If the traps come first, the paper reads as a laundry list of warnings without a central positive result.
