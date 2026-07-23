## 2. Background

### 2.1 The Checkpoint Selection Problem

LLM fine-tuning produces a sequence of model checkpoints $c_1, \dots, c_T$.
At the end of training, one must be chosen for deployment. The standard
selection rule is to evaluate each checkpoint on a held-out validation set
and pick the one with the lowest loss (or highest task-specific metric).
This is cheap, simple, and the default in most training frameworks.

The implicit assumption is that the validation metric is a reliable
indicator of deployment quality — that lower loss means a better model.
This assumption has been questioned — Instruct-SkillMix
(ICLR 2025) observes that choosing the checkpoint with the lowest
validation cross-entropy does not lead to the best downstream
performance — but the underlying failure mode is rarely characterized
quantitatively.

### 2.2 Our Initial Hypothesis

We began with a specific hypothesis: that the Spearman correlation
$\rho_{\text{align}}$ between a cheap proxy signal (held-out loss) and
a more expensive validation signal (task-specific accuracy) undergoes
detectable degradation during training — that the proxy and validation
initially agree, then diverge. An algorithm monitoring this correlation
(PAAS) could detect the divergence and switch selection strategies.

This hypothesis led us to design a monitoring framework, run fine-tuning
experiments, and collect the trajectories of $\rho_{\text{align}}$ along
training. The data told a different story, which we report below.

### 2.3 Experimental Setup

All experiments use Qwen2.5-7B-Instruct with LoRA (rank 16), trained
for 500 steps on a mixed dataset of GSM8K, CodeAlpaca, and Dolly
(approximately 40K instruction examples). Ten checkpoints are saved at
50-step intervals. Training uses a single RTX 4090 (24 GB), batch size 2,
gradient accumulation 4.

The proxy signal is held-out cross-entropy loss on question+answer text
(200 held-out pairs). The validation signal is GSM8K accuracy (200 test
questions, evaluated with three few-shot examples to establish output
format — see Trap 6). All experiments and code are documented with
frozen definitions and data provenance flags (detailed in the
experimental protocol, §3.1) to distinguish real GPU runs from
synthetic fallback.
