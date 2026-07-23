## 6. Limitations

**Generalization scope.** All experiments use a single model scale (7B),
a single training regime (LoRA, 500 steps, cosine schedule), and a single
task type (mathematical reasoning via GSM8K). Our conclusions about the
relationship between held-out set size $N$ and minimum detectable
difference are statistical and should generalize, but the specific
numerical threshold ($\Delta_{\min} = 9.8\%$ at $N=200$) will vary with
task difficulty, metric choice, and evaluation design.

**Code evaluation.** We attempted to extend our analysis to code
generation (CodeAlpaca) but found that the evaluation pipeline itself
had unresolved issues: multi-language outputs, inconsistent formatting,
and sandbox dependency failures combined to make the scores unreliable
(Trap 7, §4). This limits our empirical claim to a single task. Future
work should extend the analysis to additional domains with robust
evaluation pipelines.

**Statistical framework.** Our $\Delta_{\min}$ uses the independent
two-proportion test, which is conservative for our setting (the same
questions are evaluated across all checkpoints, making paired tests more
appropriate). A McNemar test would provide a tighter bound but requires
per-sample correctness records, which our evaluation pipeline did not
retain. We flag this as a methodological lesson: evaluation scripts
should always store per-sample results alongside aggregate metrics.

**Training configuration.** All experiments use 500 steps of LoRA
fine-tuning on a pre-trained 7B model. Different configurations (longer
training, full fine-tuning, smaller base models, different data mixes)
may produce different checkpoint dynamics. Our finding applies most
directly to the regime we tested; broader claims require broader
experimentation.

## 7. Conclusion

We set out to understand when and why proxy-based checkpoint selection
fails in LLM fine-tuning. The answer was not what we expected: the
evaluation noise floor itself — the minimum quality difference that can
be reliably detected with a given held-out set — can exceed the actual
differences between checkpoints, making selection from point estimates
of quality an ill-posed problem at practical evaluation budgets.

Our specific empirical finding is that at $N=200$, the minimum detectable
difference ($9.8\%$) exceeds the observed checkpoint range ($8.0\%$)
in a standard 500-step LoRA fine-tuning run. No checkpoint-to-checkpoint
difference reaches statistical significance. This is not a failure of any
particular selection method — it is a property of the evaluation budget.

Along the way, we encountered and documented eight analytical traps
that can produce false conclusions in this domain, ranging from
statistical artifacts to silent engineering failures. These traps, our
$\Delta_{\min}$ framework, and the frozen-definitions methodology we
developed constitute the paper's primary contribution: a diagnostic
checklist for practitioners to apply before investing in checkpoint
selection strategies.

This paper is closer to an autopsy than a proposal. We believe the
diagnostic tools are more useful than the algorithm we originally set
out to build.
