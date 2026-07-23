## 1. Introduction

Given ten checkpoints from an LLM fine-tuning run, how confidently can
you pick the best one? The standard answer — pick the checkpoint with
the lowest validation loss — is known to be imperfect. But the nature of
its imperfection is poorly understood: is it a rare failure mode triggered
by distribution shift or overfitting, or is it a systematic limitation of
the practice itself?

This paper reports on a diagnostic investigation. We set up a standard
LoRA fine-tuning of Qwen2.5-7B (500 steps, multi-task data), generated
10 checkpoints, and attempted to answer three questions:

1. Given a held-out evaluation set of size $N$, what is the minimum
   quality difference between checkpoints that can be reliably detected?
2. In this fine-tuning run, how do actual checkpoint differences compare
   to this threshold?
3. What analytical traps can lead to false conclusions about
   proxy-quality misalignment?

The answers, in brief:

1. At $N=200$ (a typical evaluation budget), the minimum detectable
   difference is $9.8$ percentage points (independent two-proportion test,
   $\alpha=0.05$). This is a quantity any practitioner can compute for
   their own setting before designing a selection strategy.
2. The observed range across all 10 checkpoints was $8.0$ percentage
   points — below the detectability threshold. No checkpoint-to-checkpoint
   difference reached statistical significance ($\min p = 0.161$), and
   neither did the best-vs-worst comparison ($p = 0.109$). In this
   setting, the evaluation noise floor exceeded the checkpoint-quality
   signal. Any method that relies on point estimates of quality at this
   evaluation budget must contend with this same noise floor.
3. We encountered eight distinct analytical traps on the way to this
   finding, ranging from statistical artifacts (Spearman discretization
   at $n=3$) to silent engineering failures (undetected synthetic
   fallback) to definitional ambiguities (three mutually incompatible
   versions of "the proxy"). Each is documented as a case study in §4.

This paper is closer to a diagnostic checklist for proxy-based checkpoint
selection than to a proposal of a specific selection rule. Our initial
hypothesis — that a monitoring algorithm (PAAS) could detect and
compensate for proxy-quality misalignment — was not supported by the
data in this setup (§5.1). The tools and warnings we developed in the
process of testing that hypothesis ($\Delta_{\min}$, the eight traps)
are, we believe, more broadly useful than the algorithm itself.
