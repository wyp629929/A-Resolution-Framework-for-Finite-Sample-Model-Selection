# Perspective Review Report

**Reviewer:** Peer Reviewer 3 (Perspective Reviewer -- cross-disciplinary, traditional statistics / econometrics / classical model selection)
**Paper:** "When Model Selection Cannot Be Resolved: A Finite-Sample Evaluation Framework for Machine Learning"
**Target journal:** Machine Learning (Springer, MLJ)

---

## Summary Assessment

The paper proposes a Delta_min framework for quantifying whether a given evaluation budget provides sufficient statistical resolution to distinguish between candidate models. The empirical demonstrations are instructive, the pitfall catalog is valuable, and the core message -- that model selection is frequently conducted at sample sizes that cannot support the intended comparisons -- is important for the ML community. However, from a classical statistical perspective, the paper substantially overclaims the novelty of its methodological contribution. The Delta_min statistic is a textbook power analysis for the two-proportion z-test, a tool that has existed for nearly a century. The paper's genuine contributions lie in its application of this known tool to a specific domain and its documentation of diagnostic failures, not in the statistical framework itself. I elaborate on this and related concerns below.

---

## Major Comments

### 1. Delta_min is Standard Power Analysis, Not a New Framework

The paper's central formula (Eq. 2) is the minimum detectable effect (MDE) for an independent two-proportion z-test:

\[
\Delta_{\min} = z_{\alpha/2} \cdot \sqrt{2\bar{p}(1-\bar{p})/N}
\]

This is taught in every introductory statistics course. It appears in Cohen (1988, 1992), in every power analysis software package (G*Power, PASS, R's `pwr` package), and in any textbook on sample size determination. The paper acknowledges the connection to power analysis (Section 8.5, lines 366-368), stating that they "reinterpret classical statistical power as a model-selection resolution limit," but this framing undersells the debt. It is more accurate to say: the paper applies standard power analysis to the model selection problem and provides empirical evidence that many deployed comparisons fall below standard power thresholds.

**Recommendation.** The paper should front-load this acknowledgment. A reader with statistical training will immediately recognize Eq. 2 and may dismiss the paper as reinventing the wheel if the authors do not explicitly clarify what is genuinely novel (the application domain, the paired structure, the empirical demonstrations) versus what is textbook material. The current framing in Section 8.5 that Delta_min addresses "orthogonal questions" to AIC/BIC is the right direction, but the paper needs an equally straightforward statement about its relationship to classical power analysis.

### 2. The "Orthogonal to AIC/BIC" Claim is Correct but Misleadingly Narrowed

Section 8.5 correctly notes that AIC/BIC address model complexity relative to data, while Delta_min addresses whether the evaluation budget can detect performance differences at all. However, this comparison is a straw man. AIC/BIC and Delta_min operate in such different domains that the "orthogonality" is almost trivial:

- AIC/BIC are used when the researcher controls the model specification (which parameters to include) and needs to penalize complexity. They compare models on the training data, not on a held-out validation set.
- Delta_min is used when the researcher has already-trained models (checkpoints) and needs to determine whether a held-out test set can rank them.

These are different stages of the research pipeline with different data and different goals. The comparison to AIC/BIC is therefore not the right benchmark. The more relevant comparison is to:

- **Cross-validation standard errors.** When a practitioner runs k-fold CV, the standard deviation across folds provides a natural error bar. Two models whose CV means differ by less than one standard error are not considered meaningfully different (the "one-SE rule" in Breiman et al., 1984; Hastie et al., 2009). This is functionally equivalent to Delta_min but requires no additional computation.
- **McNemar's test.** For the paper's paired design (same N questions evaluated at every checkpoint), McNemar's test is the standard tool and provides tighter bounds than the independent two-proportion test the paper uses. The authors acknowledge this (Section 6, lines 373-374, and limitations), but the acknowledgment is buried.

**Recommendation.** Replace the "orthogonal to AIC/BIC" framing with a more precise positioning relative to CV standard errors (which are the most common practical tool for this problem) and McNemar's test (which is the most appropriate statistical tool for this problem). The Delta_min contribution is that it provides a pre-experimental diagnostic, unlike post-hoc CV or significance testing -- but this framing requires the caveats I raise in Comment 3.

### 3. The "Pre-Experimental Diagnostic" Claim is Overstated

The paper positions Delta_min as something a practitioner computes "before training begins" (line 367). However, computing Delta_min via Eq. 2 requires \bar{p}, the expected accuracy, which is unknown before training. The paper suggests using a "guess" for \bar{p}, but the sensitivity analysis is not fully explored.

Consider: at N=200, if the true accuracy is 0.90, Delta_min is 4.2 pp; if it is 0.50, Delta_min is 9.8 pp -- more than double. A researcher who "guesses" 0.5 but observes 0.9 will compute an overly conservative threshold and may incorrectly conclude their experiment is underpowered. A researcher who guesses 0.9 but observes 0.5 will compute an overly optimistic threshold and may be misled into believing their comparison is resolvable when it is not.

The paper implicitly acknowledges this by providing the reference table (Table 8) with multiple \bar{p} columns, which partially mitigates the concern. But the rhetorical framing -- "before investing in model selection, determine whether the evaluation budget is sufficient" (line 380) -- implies a precision that the tool does not deliver without knowing \bar{p}.

There is a deeper issue: in many research contexts, \bar{p} and the expected effect size \delta are learned from the same pilot experiments that the paper warns are underpowered. This creates a circular dependency: you need Delta_min to determine if your pilot is adequate, but you need pilot data to estimate \bar{p} for Delta_min.

**Recommendation.** The paper should (a) present a sensitivity analysis showing how Delta_min varies with misspecified \bar{p}, (b) propose a sequential approach (start with a conservative \bar{p}=0.5, update as data accumulates), and (c) soften the "before training" language to "before finalizing the evaluation budget" or "as a design-stage diagnostic."

### 4. Missing Connection to Standardized Effect Sizes (Cohen's h)

For a cross-disciplinary audience, the lack of connection to Cohen's h -- the standard effect size measure for proportions -- is a notable omission. Cohen's h is defined as:

\[
h = 2 \cdot \arcsin(\sqrt{p_1}) - 2 \cdot \arcsin(\sqrt{p_2})
\]

Under the arcsin transformation, the variance-stabilized Delta_min becomes simply:

\[
\Delta_{\min}^{\text{(arcsin)}} = z_{\alpha/2} \cdot \sqrt{1/N}
\]

This is independent of \bar{p}, which is both a simplification and a corrective to the paper's framing. It reveals that in the variance-stabilized space, the resolution threshold is solely a function of N -- not of \bar{p}, which appears in Eq. 2 only because the paper uses the untransformed scale. This has two implications:

1. **The \bar{p} dependence in Eq. 2 is an artifact of the untransformed scale**, not a fundamental property of the resolution limit. In the transformed space, Delta_min is the same for all accuracy levels. The paper should acknowledge this or justify why the untransformed scale is preferable.

2. **Cohen's h provides standardized benchmarks** (small h=0.2, medium h=0.5, large h=0.8) that could help practitioners calibrate what Delta_min values are meaningful in their domain. A Delta_min of 9.8 pp at \bar{p}=0.5 corresponds to h = 2\arcsin(\sqrt{0.598}) - 2\arcsin(\sqrt{0.5}) \approx 0.20, which is a "small" effect size by Cohen's conventions. This frames the paper's findings in a language familiar to the social sciences and econometrics.

**Recommendation.** Add a discussion of Cohen's h and the arcsin transformation to demonstrate how the framework relates to established effect size conventions. This would strengthen the paper's claim to cross-disciplinary relevance.

### 5. Missing References: Leeb & Potscher and Post-Selection Inference

The paper lacks any citation to the extensive literature on post-model-selection inference, most notably the work of Leeb and Potscher (2005, 2008). Their fundamental result is that, without strong assumptions (e.g., the "beta-min" condition that the true parameter is bounded away from zero), any inference conducted after model selection is invalid -- standard confidence intervals and p-values do not have their nominal properties. This is directly relevant to the paper's argument:

- If the evaluation budget cannot resolve model differences (Delta_min exceeds the observed gap), then the selected model is effectively random with respect to the true ranking.
- Leeb and Potscher's work implies that even if you *could* resolve the differences, any subsequent inference conditional on the selection is invalid unless the true accuracy gaps are substantially larger than Delta_min.

This is not a minor oversight. The paper's argument that underpowered selection undermines reproducibility is consonant with the post-selection inference literature, and citing it would (a) ground the paper in established statistical theory, (b) reveal that the problem is even more severe than the paper suggests, and (c) demonstrate cross-disciplinary awareness.

Other missing references:
- **Freedman (1983), "A note on screening regression equations."** An early demonstration that model selection on noise produces unstable and irreproducible results. This is the same insight that motivates Delta_min.
- **Ioannidis (2005), "Why most published research findings are false"; Button et al. (2013), "Power failure: why small sample size undermines the reliability of neuroscience."** The replication crisis literature documents exactly the same phenomenon (underpowered studies produce inflated effects and non-replicable findings) that the paper describes for model selection. Citing this would connect the paper to a broader scientific crisis and strengthen its claim to relevance.
- **Berk et al. (2013), "Valid post-selection inference."** More recent developments in the selective inference literature that directly address when model selection can and cannot be trusted.

**Recommendation.** Add a paragraph connecting Delta_min to the post-selection inference literature. This would significantly strengthen the paper's theoretical grounding and reveal the problem as even more fundamental than a power analysis alone suggests.

### 6. The Independent vs. Paired Test Issue is More Consequential Than Acknowledged

The paper uses the independent two-proportion test as its default Delta_min, acknowledging that the paired McNemar test would be more appropriate but the per-sample data were not saved. This is presented as a minor limitation (lines 373-374), but it has substantive consequences for the paper's central claim.

The variance of the paired difference D_i is:

\[
\text{Var}(D_i) = p_{01} + p_{10} - (p_{01} - p_{10})^2
\]

where p_{01} is the probability that model A is correct and model B is wrong, and p_{10} is the reverse. For positively correlated evaluations (the same easy questions are answered correctly by both models), p_{01} + p_{10} can be much smaller than 2\bar{p}(1-\bar{p}) (the independent variance). This means the paired Delta_min can be substantially smaller than the paper's reported value.

If the paired Delta_min at N=200 is, say, 5-6 pp instead of 9.8 pp, then the paper's claim that "the observed range (8.0 pp) falls below Delta_min" might no longer hold -- the range of 8.0 pp could exceed the paired threshold. While the bootstrap analysis independently supports the conclusion, the paper's headline statistic (Delta_min = 9.8 pp exceeds range = 8.0 pp) depends on using the more conservative (and less appropriate) test.

**Recommendation.** Either (a) estimate the paired variance using the bootstrap resamples already conducted (bootstrapping preserves the paired structure and can estimate Var(D_i) from the data), or (b) rerun the evaluation pipeline to save per-sample correctness data, or (c) explicitly bound how much tighter the paired threshold would be under varying correlation assumptions. As it stands, the central quantitative claim rests on a test that the authors acknowledge is not the most appropriate for their design.

### 7. Practical Adoption: Would an Econometrician or Traditional ML Researcher Find This Useful?

From an econometrics perspective, the paper's core message is: "before comparing models, check whether your sample size gives you enough power to detect the differences you care about." This is a universal principle that any applied statistician would recognize. The novelty is not the principle but the specific application to LLM fine-tuning checkpoints, where the paper convincingly shows that typical evaluation budgets (N ~ 200) are inadequate.

However, the paper's framing as a "new framework" may alienate the very practitioners it seeks to reach. An econometrician reading this paper might ask: "Why is this in a machine learning journal? This is basic statistics that I teach in my first-year graduate course." The paper would be more effective if it presented Delta_min less as a novel method and more as a field guide and wake-up call for the ML community, with explicit connections to well-established statistical concepts.

**Recommendation.** Consider restructuring the paper to lead more prominently with the empirical findings and the diagnostic pitfalls (which are genuinely novel) while clearly positioning the statistical methodology as "an application of standard power analysis to the model selection problem." This would better serve the cross-disciplinary audience the paper claims to address.

---

## Minor Comments

1. **The bootstrap analysis is the paper's strongest empirical contribution.** The Delta_min framework is useful as a communication tool, but the bootstrap selection distribution (Figure 4, showing the best model selected in only 44% of resamples) is a more direct and intuitive demonstration of instability. Consider giving the bootstrap analysis more prominent placement.

2. **The synthetic validation experiment (Figure 1) could be more informative.** The simulation confirms that selection accuracy aligns with Delta_min at N=200, 500, 1000. Adding a condition where the data are generated with a true gap below Delta_min would demonstrate that the framework correctly identifies irresolvable comparisons (rather than just confirming that large gaps are detectable).

3. **The literature survey (Section 6) is a convenience sample, and the 79% figure should be caveated more heavily.** The paper acknowledges this, but the abstract and conclusion present the 79% figure without the sample limitations. Given the small sample (N=19), the Wilson CI of [57%, 91%] is wide enough that the headline number should be treated cautiously.

4. **The practitioner's quick-reference table (Table 8) is genuinely useful** and should be highlighted as a primary contribution. This is the format that practitioners will actually use.

---

## Verdict

**Revise with moderate changes.** The paper addresses an important problem and provides valuable empirical demonstrations and a useful diagnostic taxonomy. However, the methodological framing needs to be significantly recalibrated: the statistical contribution is not novel, but the application domain and empirical findings are. The relationship to classical power analysis, Cohen's h, cross-validation error bars, and the post-selection inference literature must be properly acknowledged. The pre-experimental diagnostic claim needs caveats about the dependence on unknown \bar{p}. The paired-test issue should be resolved or more carefully bounded. With these revisions, the paper could make a meaningful contribution to ML practice.

---

## References for Author Consideration

- Berk, R., Brown, L., Buja, A., Zhang, K., & Zhao, L. (2013). Valid post-selection inference. *Annals of Statistics*, 41(2), 802-837.
- Button, K. S., Ioannidis, J. P. A., Mokrysz, C., Nosek, B. A., Flint, J., Robinson, E. S. J., & Munafo, M. R. (2013). Power failure: why small sample size undermines the reliability of neuroscience. *Nature Reviews Neuroscience*, 14, 365-376.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Cohen, J. (1992). A power primer. *Psychological Bulletin*, 112(1), 155-159.
- Freedman, D. A. (1983). A note on screening regression equations. *The American Statistician*, 37(2), 152-155.
- Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Medicine*, 2(8), e124.
- Leeb, H., & Potscher, B. M. (2005). Model selection and inference: Facts and fiction. *Econometric Theory*, 21(1), 21-59.
- Leeb, H., & Potscher, B. M. (2008). Sparse estimators and the oracle property, or the return of Hodges' estimator. *Journal of Econometrics*, 142(1), 201-211.
