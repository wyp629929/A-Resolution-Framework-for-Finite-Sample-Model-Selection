# PAAS Pilot (v0)

**Purpose**: Validate engineering pipeline end-to-end before full-scale experiments.

## Important Warnings

### ⚠️ Self-Judge Bias (ShareGPT)
The pilot uses **LLaMA-3-8B self-judge** for ShareGPT pairwise comparisons.
Since the checkpoints being evaluated are fine-tuned from the same base model
family, this introduces **self-preference bias** documented in LLM-as-judge
literature: models tend to prefer outputs matching their own style.

**This signal is NOT valid for full-scale experiments.**
Full-scale must use human pairwise annotation (≥2 raters, Krippendorff's α).

### ⚠️ Bootstrap CI
Pilot uses n=6 bootstrap CI for ρ_t, which is unreliable for small samples.
This is acceptable here (testing code paths, not statistical claims).
Full-scale must use permutation tests or report raw ρ + sample size.

### ⚠️ Synthetic Data Fallback
If no GPU is available, the pilot falls back to synthetic proxy/validation signals.
These simulate realistic patterns (improvement → divergence) but do not
replace real model evaluations.

## Acceptance Criteria

| ID | Criterion | How to Verify |
|----|-----------|---------------|
| P1 | Training produces ≥2 checkpoints with correct step indices | Check `checkpoints-*.json` has valid step values |
| P2 | Proxy signals in reasonable range | `proxy_main` values between -0.5 and 2.0 |
| P3 | Validation signals in [0, 1] | `val_main` values between 0.0 and 1.0 |
| P4 | ρ_t in [-1, 1] | All `rho_spearman` values within bounds |
| P5a | Forced trigger enters cautious selection | Selection mode is not "normal" |
| P5b | Mock divergence triggers fallback | `p5b_mock_fallback.json` exists with last_safe_step set |
| P6 | Pipeline completes in < 30 min | Check elapsed time |

## Running

```bash
cd paas/
python pilot/run_pilot.py
```

Output: `results/pilot_v0/report/report.md`

## Full-Scale Transition Checklist

- [ ] Pilot passes ≥ 6/7 criteria
- [ ] ρ_t trends are interpretable (not constant NaN, not always 1.0)
- [ ] Human pairwise annotation protocol is ready
- [ ] Cloud GPU (AutoDL) resources confirmed
- [ ] CI method switched from bootstrap to permutation
