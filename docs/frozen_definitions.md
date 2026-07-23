# PAAS Project: Frozen Definitions

> Last updated: 2026-07-04
> Status: PRE-EXPERIMENT — these definitions must be finalized before any new experiment is run.
> All subsequent experiments MUST reference this document. Any change to these definitions
> constitutes a new experiment version and must be noted explicitly.

---

## 1. Proxy Signal

**Definition**: The held-out loss computed on QUESTION + ANSWER concatenation, with loss averaged over ALL tokens in the sequence.

**How it's calculated**:
```python
text = sample["question"] + "\n" + sample["answer"]
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs, labels=inputs["input_ids"])
proxy_loss = outputs.loss.item()  # averaged over all tokens
```

**What is NOT the proxy**:
- Question-only perplexity (strawman, rejected)
- Win-rate against a reference model (rejected — synthetic fallback)
- Task-specific accuracy metrics (those are validation)
- Perplexity on answer tokens only (future work if needed — but not the default)

**Rationale**: This matches what practitioners actually use for checkpoint selection: they hold out a small set of (prompt, response) pairs and compute loss. It's computationally cheap and standard.

---

## 2. Validation Signal

**Definition**: Two independent task-specific accuracy metrics, reported separately AND as a simple average.

**Components**:
1. **GSM8K accuracy**: Binary answer correctness via exact numeric match (after `####` extraction). 200 test samples.
2. **CodeAlpaca executability**: Binary score (1 = code executes without error inside sandbox, 0 = error). 100 test samples.

**Storage**: Each checkpoint's validation data MUST retain individual task scores. The combined score is for reporting convenience; the sub-task trajectories are the primary analytical signal.

```python
# Stored alongside combined score:
checkpoint.validation_scores.gsm8k_accuracy  # independent trajectory
checkpoint.validation_scores.code_func_score # independent trajectory
# Combined for summary:
validation_main = (gsm8k_accuracy + code_func_score) / 2
```

**Critical note (from v0 audit)**: Sub-task scores can diverge (GSM8K peaked at step 350 and declined, while CodeAlpaca continued improving through step 500). This cross-task tradeoff may be a more interesting phenomenon than proxy-validation misalignment. **The combined average must not erase this signal.**

---

## 3. Spearman Rank Correlation (ρ_t)

**Window size**: `n >= 6`. n=3 is explicitly rejected (discretization artifact, only 5 possible values).

**Scope**: Within-run (per-experiment), NOT pooled across experiments. Each experiment's training trajectory produces its own ρ_t.

**Calculation**:
```python
# For each checkpoint at step t, window = checkpoints[t-n+1 : t+1]
# ρ_t = Spearman(proxy_loss values in window, validation_main values in window)
rho_t = spearman(window_proxy_losses, window_validation_scores)
```

**Reporting**: Report ρ_t as a trajectory (not a single aggregate number). When summarizing, report mean and range, not just a point estimate.

**Minimum interpretable value**: n < 6 → mark as "insufficient data / unreliable".

---

## 4. "Optimal" and "Wins" Definitions

**Oracle**: The checkpoint with the highest validation_main among ALL checkpoints in a single experiment.

**PAAS-wins**: PAAS-selected checkpoint's validation_main ≥ proxy_argmax-selected checkpoint's validation_main within the SAME experiment.

**Tolerance**: No tolerance. Comparisons are exact (floating point within 1e-6). No "approximate optimal" claims.

**"Reaches optimal"**: The selected checkpoint's validation_main is within 0.1% of the global oracle value for that experiment.

---

## 5. Data Source Marking

Every JSON result file MUST include a `data_source` field with exactly one of these values:
- `"real_gpu"` — produced by a GPU training run with frozen definitions
- `"synthetic_verification"` — deliberately synthetic, used for algorithm verification only
- `"synthetic_fallback"` — code fell back to synthetic due to configuration error (WARNING: NOT for paper)

**Format**: Each result file includes:
```json
{
  "data_source": "real_gpu",
  "data_timestamp": "2026-07-04T12:00:00Z",
  "git_commit": "abc123def"  // or version tag
}
```

**Implementation**: The `_save_stage` function in `pipeline/runner.py` MUST inject these fields into every saved result file.
Reading code MUST check these fields and print a bold WARNING if reading from any synthetic source.

---

## 6. Experiment Versioning

Each experiment configuration gets a version number. Current: v0 (exploratory, definitions not frozen).
Next experiment: v1 (first experiment under frozen definitions).

Version numbering:
- v0: All experiments before 2026-07-04 (definitions unfrozen, for reference only)
- v1+: Experiments after definitions are frozen

---

## 7. Synthetic Verification Scenario

Before running any new GPU experiments, PAAS must first pass a synthetic verification:
where the data is deliberately constructed to have ρ_t go from high (≈1.0) to low (< 0.3).
If PAAS cannot demonstrate a clear advantage in this controlled setting, the algorithm
needs redesign before real experiments are attempted.

Synthetic parameters:
- 10 checkpoints, proxy and validation scores deliberately constructed
- Phase 1 (steps 1-5): both increase together (ρ ≈ 1.0)
- Phase 2 (steps 6-8): proxy diverges, validation continues up (ρ → negative)
- Phase 3 (steps 9-10): both flat (no information)
- Verify: PAAS enters caution mode in Phase 2 and switches to uniform aggregation

---

## 8. Qwen → LLaMA Decision

Pending. The choice between Qwen2.5-7B and LLaMA-3-8B depends on:
- Availability (gated model access)
- Reproducibility at scale
- Consistency with related work (most use LLaMA)
- If Qwen is used, Limitations must explicitly note the substitution

---

## 9. Mandatory Audit Process

Before any experimental result can be:
- Written into a paper draft
- Claimed as a "finding" or "discovery"
- Used to motivate a design decision

It MUST pass the following audit checklist:

### Audit Checklist (apply to every new experiment)

1. **[CONFIRM_DEFINITIONS]** Are proxy and validation computed using the frozen definitions from Sections 1-3? (Check code, not memory.)
2. **[CONFIRM_DATASOURCE]** Is the result file's `data_source` field `"real_gpu"`? If `"synthetic_*"`, STOP — this data is not for paper claims.
3. **[VERIFY_SINGLE_NUMBER]** For any single-number claim (e.g., "ρ=0.45", "PAAS wins 4/4"):
   - Can the number be independently reproduced from the raw data?
   - What's the n (sample size)? Is n sufficient for the claimed precision?
   - Could a small change in definition change the number? (If yes, the definition may not be frozen enough.)
4. **[CHECK_COUNTEREXAMPLE]** Is there an experiment where the claimed phenomenon does NOT occur? Report it alongside the successes.
5. **[SUBSCORE_CHECK]** When reporting combined metrics, verify that sub-task scores (GSM8K, Code) do not tell a contradictory story.

### Rule

No result is "paper-ready" until it has passed this audit. The audit is run by a separate process (not by the person who produced the result). If this document's Revision Log shows that definitions changed between Experiment A and Experiment B, results from A and B cannot be compared within a single paper claim without explicit versioning.

---

## 10. Hypothesis 2: Cross-Task Skill Tradeoff — Replication Criteria

This section defines the judgement standard for the next experiment cycle (v1). Written BEFORE experiments are run.

### Hypothesis

During multi-task instruction fine-tuning (GSM8K + CodeAlpaca), the quality trajectories of individual tasks can diverge in the later stages of training — one task's accuracy declines while another continues improving.

### Replication Criteria (ALL must be met)

**C1 — Directional divergence**: In ≥ 2/3 of new experiments, GSM8K and CodeAlpaca trajectories in the last 40% of training steps (steps 300-500) show Spearman correlations with **opposite signs** (one positive, one negative).

**C2 — Minimum signal strength**: Each trajectory's Spearman correlation has |ρ| > 0.3 (to avoid counting noise as divergence).

**C3 — Schedule diversity**: Experiments span at least 2 different LR schedules (cosine is required since the original observation was on cosine).

### Partial Replication Classification

If results across 3 experiments are mixed (e.g., 1 clear divergence, 1 weak, 1 none), the hypothesis is classified as **NOT REPLICATED**. This prevents "partial replication" from being used as a narrative loophole.

### What This Means for the Paper

| Outcome | Action |
|---|---|
| Replicated (C1+C2+C3 met) | Core phenomenon becomes cross-task divergence. Algorithm framework needs redesign — not PAAS as-is |
| Not replicated | Cross-task divergence may have been a v0 coincidence. Third option: write a diagnostic paper documenting analytical traps encountered |

### Experiment Design (v1)

- Model: Qwen2.5-7B-Instruct, LoRA rank=16
- Training: 500 steps, save every 50 steps
- Proxy: question+answer held-out loss (Section 1)
- Validation: GSM8K + CodeAlpaca independent trajectories (Section 2)
- Schedules: cosine (required) + at least one other
- Count: 2-3 experiments (we are in replication mode, not full-scale yet)

## Revision Log

| Date | Change | Author |
|---|---|---|
| 2026-07-04 | Initial frozen definitions (v1) | Audit process |
