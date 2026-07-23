# Editorial Decision Package

**Manuscript:** "Evaluating Evaluation Budgets: A Pre-Experimental Diagnostic for LLM Benchmark Comparisons"
**Journal:** *Machine Learning* (Springer, MLJ)
**Review Round:** First round
**Editorial Synthesizer:** Acting Editor-in-Chief

---

## Part A: Consolidated Review Summary

### A1. Consensus Points (raised by ≥3 reviewers)

The following issues appear across three or more independent reviews and represent the strongest signal for required revision:

| # | Issue | Raised by | Description |
|---|-------|-----------|-------------|
| C1 | **Literature survey methodology is inadequate** | EIC (M2), Methodology, Domain (R5), Devil's Advocate (CRITICAL #2) | 19-paper convenience sample, single annotator, p̄=0.5 assumption. The 79% claim — likely the most cited result — is built on methodologically insufficient foundations. Wilson CI spans 34 percentage points [57%, 91%]. |
| C2 | **Paper structure: lead with N=1319** | EIC (M1), Domain (R4), Devil's Advocate (MAJOR #7) | Presenting N=200 as the primary experiment and N=1319 as a robustness check is an artifact of the authors' own design choice. The full test set was available throughout. Consensus: restructure to lead with the most informative evaluation budget. |
| C3 | **Independent vs. paired Δ_min must be addressed** | EIC, Methodology, Perspective (R6) | All experiments use the independent (two-sample) Δ_min, but the data are paired (same models, same test items). McNemar-based Δ_min could be substantially smaller. The paper must report both and quantify the gap. |
| C4 | **Multiple comparison correction is inconsistent and incomplete** | EIC (M3), Methodology, Devil's Advocate (MAJOR #8) | Bonferroni is mentioned in Implication 2 but never applied to experimental data. Uncorrected p-values appear in Table 2. Only Bonferroni is discussed; Holm, Benjamini-Hochberg, Tukey HSD, and max-t adjustments should be considered. Checkpoints are correlated — Bonferroni is conservative. |
| C5 | **Missing experimental details** | Domain (R2, R3, R6, R7), Methodology | Hyperparameters (lr, batch size, optimizer, scheduling, decoding params), evaluation protocol (8-shot CoT is GSM8K community standard), and answer extraction method are not reported. Experiment is under-specified for reproducibility. |

### A2. Divergent Opinions

Where reviewers substantively disagree in emphasis or interpretation:

| Issue | View A | View B | Assessment |
|-------|--------|--------|------------|
| **Novelty of Δ_min** | Methodology: "core statistical reasoning is sound" (favorable) | Perspective (R1) + Devil's Advocate (CRITICAL #1): Δ_min is standard power analysis, textbook material since Cohen (1988) — relabeling, not a new framework | The disagreement is real, but both agree the paper should front-load this acknowledgment. The contribution lies in *application context*, not in statistical derivation. |
| **Severity of independent vs. paired gap** | Methodology: "Moderate" | Perspective (R6): "more consequential than acknowledged — if McNemar Δ_min is tighter, the central claim might weaken" | Perspective's concern deserves attention in revision. The paper should quantify the paired Δ_min and discuss implications for its claims. |
| **Interpretation of 44% bootstrap result** | Paper's framing: "noise dominates evaluation rankings" | Devil's Advocate (CRITICAL #3): 4.4× uniform baseline suggests weak signal exists, not that noise dominates. Normalized entropy 0.717 supports "partially concentrated" not "completely diffuse." | A substantive interpretive disagreement. The paper should address this alternative reading directly. |
| **Is the core finding trivial?** | Devil's Advocate (MAJOR #9): "anyone with basic stats knows N=200 can't detect small effects" | EIC: "core idea is genuinely useful" | The Devil's Advocate challenge must be addressed explicitly, but the editorial view is that the paper's value is in demonstrating the *prevalence* and *practical consequences* of this known statistical fact in a specific domain. |
| **N=1319 interpretation** | Paper's framing: "winner's curse" — N=200 was overestimating | Devil's Advocate (MAJOR #7): equally consistent with "N=200 was just noisy" — range shrinks from 5.5pp to 3.0pp | The paper should acknowledge this alternative explanation and address it with evidence. |
| **Orthogonal to AIC/BIC** | Paper asserts Δ_min is orthogonal to model selection criteria | Perspective (R2): "straw man — they operate in such different domains that the comparison is almost trivial" | The framing should be softened or dropped. |

### A3. Devil's Advocate CRITICAL Issues — Special Flag

Three CRITICAL-level challenges from the Devil's Advocate review (R5) mandate explicit response in the revision:

**DA-C1: Δ_min as "diagnostic" vs. standard power analysis.** The relabeling concern is amplified by the Perspective reviewer's independent observation (R4-R1) that Eq. 2 is the minimum detectable effect for a two-proportion z-test, textbook material since Cohen (1988). **Required action:** The paper must clearly distinguish what is novel (the application framework, the bootstrap calibration procedure, the empirical survey) from what is standard (the statistical derivation of Δ_min). Failure to do so risks the perception that the paper's central contribution is terminological.

**DA-C2: 79% claim from 19-paper survey.** This is the paper's most striking empirical result and the most methodologically vulnerable. 95% Wilson CI spans 34pp. No formal inter-rater reliability. No correction for multiple claims per paper. Small-N benchmarks (100% violation) are circular with the framework's premise. **Required action:** The survey must be expanded to a proper systematic review (PRISMA-compliant, dual extraction, cluster-robust standard errors), or all claims must be proportionally hedged. The EIC (M2) provides three options; (a) is preferred, (b) is acceptable, (c) is not acceptable.

**DA-C3: 44% bootstrap selection: noise or signal?** The paper interprets 44% as evidence that "noise dominates." The alternative — that 4.4× uniform baseline represents weak but real signal — is not discussed. Normalized entropy 0.717 is closer to "partially concentrated" than "completely diffuse" (which would be 1.0). **Required action:** Add explicit discussion of this alternative interpretation and justify why the "noise-dominant" reading is preferred.

---

## Part B: Editorial Decision Letter

[Date]

**Decision: Major Revision**

Dear Authors,

Thank you for submitting your manuscript "Evaluating Evaluation Budgets: A Pre-Experimental Diagnostic for LLM Benchmark Comparisons" to *Machine Learning*. The paper has been reviewed by four regular reviewers and one Devil's Advocate reviewer, with the Acting Editor-in-Chief providing a separate assessment.

All reviewers recognize the practical importance of your core question — how researchers can prospectively assess whether their evaluation budget is adequate for reliable LLM comparisons — and acknowledge the value of the bootstrap calibration procedure and empirical survey. The paper addresses a genuine need in the rapidly growing field of LLM evaluation.

**However, the reviews converge on several non-negotiable issues that must be resolved before the paper can be considered for publication.**

### Conditions for Acceptance

Acceptance is contingent on satisfactory resolution of the following must-address issues in a revised manuscript:

1. **Restructure the paper to lead with N=1319 (Consensus C2).** Presenting N=200 as the primary experiment and N=1319 as a robustness check is an artifact of your own design choice. The full test set (GSM8K) was available from the start. Lead with the most informative evaluation budget and treat N=200 as a controlled demonstration of the diagnostic.

2. **Address the literature survey methodology (Consensus C1; EIC M2; DA-C2).** The 79% violation rate is the most impactful empirical claim in your paper and the most methodologically vulnerable. The 95% Wilson CI spans 34 percentage points [57%, 91%]. The review panel identifies three options:
   - **(a) Preferred:** Expand to a proper systematic review following PRISMA guidelines, with dual extraction, inter-rater reliability assessment, and cluster-robust standard errors accounting for multiple claims per paper.
   - **(b) Acceptable:** Retain the current sample but proportionally hedge all claims, explicitly reporting the wide confidence interval and methodological limitations (single annotator, convenience sample, p̄=0.5 assumption).
   - **(c) Not acceptable:** Leave the literature survey as-is.
   
   Option (a) is strongly preferred; option (c) will result in rejection.

3. **Report independent and paired Δ_min throughout (Consensus C3; EIC; Methodology; Perspective R6).** Your experiments use paired data (same models, same test items) but report the independent (two-sample) Δ_min. McNemar-based Δ_min could be substantially smaller. Report both versions throughout and discuss the gap.

4. **Consistent and expanded multiple comparison correction (Consensus C4; EIC M3; Methodology; DA-MAJOR #8).** Bonferroni is mentioned in Implication 2 but never applied to the experimental data. Uncorrected p-values appear in Table 2. Additionally, consider Holm-Bonferroni, Benjamini-Hochberg FDR, Tukey HSD, or max-t adjustments. Because checkpoint evaluations are correlated, Bonferroni is known to be conservative in this setting.

5. **Explicitly address the three Devil's Advocate CRITICAL challenges (DA-C1, DA-C2, DA-C3) as described in Section A3 of this decision letter.** These are:
   - Distinguish what is novel (application framework, bootstrap calibration, empirical survey) from what is standard statistics (Δ_min = minimum detectable effect for two-proportion z-test).
   - Address the competing interpretation of the 44% bootstrap result: 4.4× uniform baseline could indicate weak signal rather than dominant noise. Discuss normalized entropy 0.717 in this context.
   - Address the relabeling concern by clearly situating Δ_min within the established power analysis literature (Cohen, 1988; Leeb & Pötscher, 2005).

### Discretionary but Strongly Recommended Issues

The following issues would significantly strengthen the paper and should be addressed in the revision:

6. **Restructure the "orthogonal to AIC/BIC" framing (Perspective R2).** One reviewer argues this is a straw man. Whether or not the authors agree, the framing should be recalibrated.

7. **Explicitly verify the small-δ condition (Proposition 1) for all experiments (Methodology; EIC).** This is currently assumed but not verified.

8. **Add sensitivity analysis for p̄ (Perspective R5; DA-MAJOR #4).** Pre-experimental use of Δ_min requires knowing p̄. A misspecified guess (e.g., 0.5→0.9) changes Δ_min by 2×. Report Δ_min across a range of plausible p̄ values.

9. **Increase the number of seeds / replications (Methodology).** Three seeds are insufficient to characterize between-seed variability. Report results with more seeds or add explicit discussion of this limitation.

10. **Replicate on at least one additional model family (Domain R1).** The current experiments use a single model family (Llama 2-7B). Replication on Llama 3.1-8B or Mistral-7B would substantially strengthen generalizability.

11. **Report full experimental details for reproducibility (Consensus C5; Domain R2, R3, R6).** Include learning rate, batch size, optimizer, scheduler, decoding parameters (temperature, top-p), and answer extraction method.

12. **Add the following supplemental analyses:**
    - Bootstrap confidence interval for the Δ_min threshold itself (Methodology).
    - Simulated paired-data validation (Methodology).
    - Connection to Cohen's h and the arcsin transformation (Perspective R3).
    - Discussion of HELM calibration framework and Open LLM Leaderboard bootstrapping (Domain R8).
    - Post-selection inference references (Leeb & Pötscher) (Perspective R5).

13. **Test the model soup recommendation systematically** beyond a single trajectory (Domain R10; DA-MAJOR #5).

14. **Add discussion of pitfalls listed by multiple reviewers:** decoding sensitivity, metric versioning, instruction sensitivity, and the limitation to binary metrics (Domain R6, R7; DA-MAJOR #6).

### Summary

In summary, your paper addresses a timely and important problem. The core diagnostic concept is sound, and the empirical demonstration is interesting. However, the current version has significant methodological weaknesses in the literature survey, inconsistent application of statistical corrections, and a structural framing that understates the most informative evidence.

A Major Revision that addresses the five must-consensus issues above and engages substantively with the three Devil's Advocate CRITICAL challenges has a strong path to acceptance. A point-by-point response to all reviews — including discretionary items — is expected.

We look forward to receiving your revised manuscript.

Sincerely,

Acting Editor-in-Chief
*Machine Learning* (Springer)

---

## Part C: Prioritized Revision Roadmap

### P0: Preconditions for Re-Review (Non-Negotiable)

These items must be addressed for the paper to proceed to a second round of review. Failure to adequately address any P0 item will result in rejection.

| ID | Task | Reviewer(s) | Action Item | Expected Impact |
|----|------|-------------|-------------|-----------------|
| P0.1 | **Restructure paper to lead with N=1319** | EIC (M1), Domain (R4), DA (#7) | Move N=1319 analysis to primary position. Treat N=200 as a controlled demonstration of what the diagnostic predicts. Reframe Section 4 (N=1319 analysis) and Section 3 (N=200 budget analysis). | Corrects the most consequential framing error. Prevents reader confusion about why the authors chose a suboptimal budget. |
| P0.2 | **Strengthen literature survey methodology** | EIC (M2), Methodology, Domain (R5), DA (CRITICAL #2) | Expand to PRISMA-compliant systematic review (preferred) or hedge all claims and report CI (acceptable). Must address: sample size (n=19 → larger), single annotator (→ dual extraction), p̄=0.5 assumption (→ use benchmark-typical values), no IAA reporting. | Protects the paper's most cited claim (79% violation rate). Current methodology would not survive scrutiny from a statistically sophisticated reader. |
| P0.3 | **Address Δ_min novelty framing** | DA (CRITICAL #1), Perspective (R1) | Clearly state in §2.1 that Eq. 2 is the minimum detectable effect for a two-proportion z-test (Cohen, 1988). Explicitly separate: (a) what is standard statistics, (b) what is new application/calibration, (c) what is empirical contribution. | Prevents "relabeling" criticism. Positions the paper's contribution where it belongs — in the application framework, calibration procedure, and empirical findings. |
| P0.4 | **Report paired and independent Δ_min** | EIC, Methodology, Perspective (R6) | Derive and report McNemar-based Δ_min alongside independent Δ_min for all experiments. Quantify the gap in §3, §4, and §5. If paired Δ_min is substantially smaller, discuss implications for the paper's central claims. | Directly addresses the most consequential methodological gap between theory and practice in the paper. |
| P0.5 | **Consistent multiple comparison correction** | EIC (M3), Methodology, DA (#8) | Apply correction to Table 2. Report both uncorrected and corrected values. Add at least one alternative to Bonferroni (Holm or BH-FDR). Discuss correlation structure among checkpoints. | Brings the paper's practice in line with its own principles. Table 2 currently violates Implication 2. |

### P1: Strongly Recommended (Would Significantly Strengthen the Paper)

These items are not absolute preconditions, but addressing them will substantially strengthen the revision.

| ID | Task | Reviewer(s) | Action Item | Expected Impact |
|----|------|-------------|-------------|-----------------|
| P1.1 | **Explicitly address DA-C3 (44% interpretation)** | DA (CRITICAL #3) | Acknowledge the alternative reading: 4.4× uniform baseline = weak signal, not pure noise. Discuss normalized entropy 0.717 as "partially concentrated" vs. "completely diffuse" (1.0). Justify why the noise-dominant interpretation is preferred. | Demonstrates intellectual honesty and preempts a natural criticism. |
| P1.2 | **Add p̄ sensitivity analysis** | Perspective (R5), DA (MAJOR #4) | Report Δ_min across a range of plausible p̄ values (e.g., 0.3–0.95). Show how the diagnostic threshold changes with misspecified p̄. | Establishes robustness of the diagnostic to the pre-experimental guess problem. |
| P1.3 | **Replicate on additional model family** | Domain (R1) | Run experiments on Llama 3.1-8B or Mistral-7B. Show that the Δ_min diagnostic generalizes beyond a single architecture/family. | Addresses the most obvious generalizability concern. |
| P1.4 | **Report complete experimental details** | Domain (R2, R3, R6, R7) | Add: learning rate, batch size, optimizer, scheduler, temperature, top-p, max tokens, answer extraction method (matching, exact, numerical). | Enables reproducibility. Current experiment is under-specified. |
| P1.5 | **Increase replications (seeds)** | Methodology | Report results with ≥10 seeds or add explicit limitation discussion. | Characterizes between-seed variability more reliably. |
| P1.6 | **Address N=1319 alternative explanation (DA #7)** | DA (MAJOR #7) | Acknowledge that the shrinking range (5.5pp → 3.0pp) is equally consistent with "N=200 was just noisy" as with "winner's curse overestimation." Provide evidence distinguishing these. | Prevents a straightforward alternative explanation from undermining the narrative. |
| P1.7 | **Address triviality concern (DA #9)** | DA (MAJOR #9), Perspective (R7) | Explicitly acknowledge that the statistical principles are basic. Re-frame the contribution as demonstrating prevalence and practical consequences, not deriving novel statistics. | Concedes the point while re-anchoring the contribution. |
| P1.8 | **Verify small-δ condition explicitly** | Methodology, EIC | Check that all experiments satisfy the small-δ condition for Proposition 1. Report results or add discussion of violations. | Ensures the formal result applies to the experimental setting. |
| P1.9 | **Add bootstrap CI for Δ_min threshold** | Methodology | Compute and report bootstrap confidence intervals for the Δ_min point estimate. | Quantifies sampling uncertainty in the diagnostic itself. |

### P2: Discretionary Improvements (Strengthen but Optional)

These items would improve the paper but are not required for acceptance. Prioritize after P0 and P1 are complete.

| ID | Task | Reviewer(s) | Action Item | Expected Impact |
|----|------|-------------|-------------|-----------------|
| P2.1 | **Soften or drop "orthogonal to AIC/BIC" framing** | Perspective (R2) | Recalibrate or remove the comparison. | Eliminates a straw-man perception. |
| P2.2 | **Connect to Cohen's h and arcsin transformation** | Perspective (R3) | Show that in variance-stabilized space, Δ_min = z·√(1/N), independent of p̄. | Elegant theoretical connection to established literature. |
| P2.3 | **Add Leeb & Pötscher (2005) reference** | Perspective (R5) | Discuss post-selection inference in the pitfalls section. | Strengthens the theoretical grounding of the "even resolvable selection doesn't guarantee valid inference" argument. |
| P2.4 | **Add Bayesian perspective section** | EIC | Brief discussion of how Bayesian approaches (Bayes factors, posterior intervals) relate to the Δ_min framework. | Broadens the methodological framing. |
| P2.5 | **Expand scope discussion to continuous metrics** | DA (MAJOR #6) | Acknowledge limitation to binary metrics. Discuss which ML evaluation scenarios fall outside scope. | Honest scope boundary. |
| P2.6 | **Add HELM and Open LLM Leaderboard discussion** | Domain (R8) | Connect Δ_min to existing calibration frameworks. | Situates the work in the broader LLM evaluation ecosystem. |
| P2.7 | **Systematic model soup test** | Domain (R10), DA (MAJOR #5) | Test model soup recommendation across multiple trajectories/initializations. | Strengthens the actionable recommendation. |
| P2.8 | **Add more pitfalls** | Domain (R6, R7) | Decoding sensitivity, metric versioning, instruction sensitivity, within-trajectory vs. across-papers separation. | Comprehensive pitfalls section. |
| P2.9 | **Add paired simulation to synthetic validation** | Methodology | Simulate paired-data scenario to validate McNemar-based Δ_min. | Strengthens the methodological foundation. |
| P2.10 | **PAAS backstory in introduction** | DA (MINOR #10) | Consider streamlining or relocating the backstory. | Improves narrative flow. |
| P2.11 | **Report both pre/post power thresholds (50% vs. 80%)** | Methodology | Add 80% alongside 50%. | Aligns with field standard. |
| P2.12 | **Use benchmark-typical p̄ in lit survey** | Domain (R5), Methodology | Replace p̄=0.5 with values empirically observed in the benchmark literature. | More realistic Δ_min estimates. |

### Implementation Guidance

**Recommended order of revision:**

1. **Structural:** P0.1 (restructure paper) — this is a "big tent" change that affects every section and should be done first.

2. **Methodological corrections:** P0.4 (paired Δ_min) → P0.5 (multiple comparisons) → P1.8 (small-δ) — these are technical fixes that integrate with the restructured paper.

3. **Literature survey:** P0.2 (lit survey expansion) — this is a substantial piece of work. Start early. It can proceed in parallel with step 2 if different co-authors handle different tasks.

4. **Framing and interpretation:** P0.3 (novelty) → P1.1 (44% interpretation) → P1.6 (N=1319 alternative) → P1.7 (triviality) → P2.1 (AIC/BIC) — these are textual revisions that calibrate the paper's claims.

5. **Empirical strengthening:** P1.3 (additional model) → P1.5 (more seeds) → P1.2 (p̄ sensitivity) → P1.9 (bootstrap CI) — these are additional experiments.

6. **Supplementary material:** All P2 items.

The total revision scope is substantial but manageable. The editorial view is that the paper's core contribution — a practical, pre-experimental diagnostic for evaluation budget adequacy in LLM comparisons — is genuinely useful and worth the effort required to bring it to publication standard.
