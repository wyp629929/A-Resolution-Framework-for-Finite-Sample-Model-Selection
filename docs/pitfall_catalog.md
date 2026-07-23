# Pitfall Catalog: Analytical Traps in Proxy-Quality Misalignment Research

> This document catalogs every significant analytical trap encountered during our investigation
> of proxy-quality alignment in LLM fine-tuning. Each entry follows the structure:
> **Phenomenon → Initial Misinterpretation → Discovery → Root Cause → Lesson**
>
> These traps are the primary methodological contribution of the diagnostic paper.
> They are not "mistakes" — they are the discovered signal, systematically documented.

---

## Trap 1: Spearman Discretization Artifact (n=3)

### Phenomenon
In 5/5 experiments, the Spearman rank correlation ρ_align (proxy-vs-validation
alignment within a sliding window) dropped to exactly **-1.0** at step 400,
suggesting a perfect ranking inversion.

### Initial Misinterpretation
"We have discovered a robust, reproducible crossing phenomenon: proxy and validation
systematically disagree at step 400, independent of seed or learning rate schedule."

### Discovery
When the sliding window was increased from n=3 to n=6, the ρ=-1 disappeared entirely.
With n=6, ρ remained positive (0.6-0.94) across all experiments. With n=10, ρ was
0.88-0.94 — strongly positive.

### Root Cause
With n=3, Spearman's rank correlation has only 5 possible values: {-1, -0.5, 0, 0.5, 1}.
Any ranking configuration where proxy and validation are in opposite order over 3 points
automatically yields ρ=-1. This is not a meaningful measurement — it's a quantization
artifact of the sample size.

### Lesson
**Window size for Spearman correlation must be large enough that the set of possible
values is dense enough for the comparison being made.** n=3 should never be used
for checkpoint selection monitoring. n ≥ 6 is the minimum for meaningful interpretation.

**Notation note**: Throughout this investigation, ρ (or ρ_align) refers to the
proxy-vs-validation alignment within a sliding window. This is distinct from
**ρ_trend**, which measures step-number vs. accuracy monotonicity (used in the
final GSM8K statistical analysis). These are different quantities and can differ
arbitrarily — a system can show high ρ_align (proxy and validation move together)
while ρ_trend is near zero (neither metric changes systematically with training).
This document uses ρ for proxy-vs-validation alignment unless otherwise noted.

---

## Trap 2: Silent Synthetic Fallback in Proxy Extraction

### Phenomenon
The proxy signal appeared to show strong correlation with validation
(ρ ≈ 0.90-0.94 in the initial reports), providing clean evidence for the paper's narrative.

### Initial Misinterpretation
"Proxy and validation are highly correlated, which validates our experimental setup
and provides a baseline for detecting when this correlation degrades."

### Discovery
When the code was re-examined, it was found that the held-out evaluation file path
was configured as `"data/held_out"` but the actual file was `"data/held_out.json"`.
The file-not-found error was silently caught, and `_synthetic_proxy_signals()` was called
instead of the real model evaluation. The synthetic signals were deterministic functions
of step number, producing identical values across experiments.

### Root Cause
The `_load_held_out()` function returned `[]` when the file wasn't found, and
`extract_proxy_signals()` checked `if not held_out:` → fell back to synthetic.
No warning was printed, no data_source flag was set, no error was raised.
The synthetic data was structurally indistinguishable from real data in the output format.

> **Verification status**: Fully confirmed. The config default was
> `held_out_path = "data/held_out"` (no extension), while the actual file was
> `data/held_out.json`. The Path.exists() check failed silently.
> Verified by: (a) code inspection of config default, (b) `ls` on GPU instance
> confirming the filename mismatch, (c) proxy extraction producing identical
> values across all 4 experiments as a consequence.

### Lesson
**Every pipeline stage that can fall back to synthetic/mock data MUST:**
1. Print a conspicuous WARNING when fallback is triggered
2. Set a `data_source` flag in the output (real/synthetic/unknown)
3. Require explicit opt-in for synthetic mode in production runs
4. Include the data_source flag in ALL downstream aggregations so that
   a reader (human or script) can instantly verify data provenance

---

## Trap 3: Proxy Definition Ambiguity

### Phenomenon
Across the investigation, three different versions of "proxy loss" were used,
each producing different correlations with validation:
- **Question-only perplexity** (frozen definition target): flat, ρ≈0.35
- **Question+answer perplexity**: also flat, ρ≈0.35
- **Proxy win-rate** (synthetic): ρ≈0.93

### Initial Misinterpretation
"We find that proxy loss is systematically uncorrelated with validation quality."

### Discovery
Each proxy definition was explored at different stages of the investigation.
The "proxy" concept was not frozen until after 3 revisions. At one point, abstract
claims were being written about a signal whose definition had not been stabilized.

### Root Cause
No single frozen definition of "proxy" existed at the start. The codebase had
`proxy_loss`, `proxy_win_rate`, and `proxy_main` (which preferred win_rate over loss),
all with different semantics. The paper's claims about "proxy argmax failure" depended
on which proxy was used:
- `proxy_loss` argmax picks step 500 → agrees with validation
- `proxy_win_rate` argmax picks step 450 → disagrees with validation

### Lesson
**Before any experiment, freeze all variable definitions in a written document.**
This includes: the exact computation formula, the data source, and which specific
variable is "the proxy" for paper purposes. Any change to the definition constitutes
a new experiment version and must be noted. After freezing, the code should raise
an assertion error if an undefined or ambiguous proxy configuration is used.

---

## Trap 4: proxy_main Default Priority Hiding the True Signal

### Phenomenon
The selection code and baselines used `proxy_scores.proxy_main` as the selection
criterion. This computed differently than raw proxy_loss, silently changing the
selection behavior.

### Initial Misinterpretation
"Proxy argmax consistently picks step 450, which has lower validation than step 500.
This demonstrates proxy argmax failure."

### Discovery
When `proxy_main` was traced through the code, it was found to prefer
`proxy_win_rate` over `proxy_loss`:
```python
@property
def proxy_main(self):
    if self.proxy_win_rate is not None:
        return self.proxy_win_rate  # prefers win_rate!
    if self.proxy_loss is not None:
        return -self.proxy_loss
    return 0.0
```

Since `proxy_win_rate` was always computed alongside `proxy_loss`, `proxy_main`
was always using win_rate. When checked directly, the raw proxy_loss values showed
a FLAT profile across all 500 steps (no argmax signal at all), while proxy_win_rate
(synthetic) showed a clear but artificial peak at step 450.

### Root Cause
A silent default priority in a property accessor. No caller code checked whether
`proxy_main` was actually using loss or win_rate. The default was reasonable in
isolation but had unintended consequences for selection logic.

### Lesson
**Any aggregated or computed metric with fallback logic must be explicitly documented
and checked.** The selection code should specify exactly which metric to use, not
rely on default priority chains. When multiple metrics are available, run selection
on all of them and report disagreements.

---

## Trap 5: GPU Memory Fragmentation → Device Offload → Evaluation Crashes

### Phenomenon
GSM8K evaluation returned ~7% accuracy for all checkpoints, when 50-55% was expected.
This looked like catastrophic model failure.

### Initial Misinterpretation
"Checkpoints after step 300 have severely degraded quality. The model may be
catastrophically forgetting."
(This misinterpretation interacted with Trap 6 — see timeline below.)

### Timeline (Trap 5 and Trap 6)
Two independent issues produced ~7% accuracy through different mechanisms,
affecting different evaluation scripts:

1. **Standalone scripts** (error_overlap.py, model_soup.py) → 7% from **Trap 6 only**
   (format mismatch). No GPU issue — model generated coherent Python code.
2. **Pipeline's fix_and_rerun.py** → worked correctly (65-99%) — aggressive memory cleanup.
3. **v1 experiment script** → 7% from **Trap 6 primarily**, PLUS memory fragmentation
   on later checkpoints. Two issues compounded.

### Discovery
When the raw generated text was examined, the model was outputting Python code and
explanatory text rather than direct answers. However, when the model was loaded
freshly (not in a loop), it generated reasonable answers. The issue was that
repeated model loading/unloading in a 10-checkpoint loop caused GPU memory
fragmentation, leading `device_map="auto"` to offload some layers to CPU.
Generation then failed because input_ids were on CUDA while model weights were mixed
between CUDA and CPU.

The earlier evaluation scripts (the pipeline's `fix_and_rerun.py`) worked correctly
because they had more aggressive memory cleanup between iterations.

### Root Cause
**Two independent issues, same symptom:**
1. **Format mismatch** (Trap 6): The model didn't output `#### answer` format.
   This was the dominant cause of the 7% reading.
2. **Memory fragmentation**: PyTorch's CUDA allocator fragments over multiple
   load/unload cycles. After 4-5 iterations, `device_map="auto"` offloads layers
   to CPU. The error manifests as garbled text mixed with correct output.

### Lesson
**When multiple bugs produce the same symptom, fix one at a time and verify.**
For multi-checkpoint evaluation loops in particular:
1. Use `torch.cuda.empty_cache()` AND `torch.cuda.synchronize()` between iterations
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
3. Monitor GPU memory usage across iterations, not just at start
4. When accuracy drops by >50%, check raw generated text before interpreting

---

## Trap 6: GSM8K Answer Format Mismatch (Few-Shot Required)

### Phenomenon
GSM8K evaluation returned ~7% accuracy with raw question prompts, but 47-55% with
few-shot prompting. The same model, same checkpoint, same evaluation set — different
prompt format.

### Initial Misinterpretation
"The LoRA fine-tuning didn't teach the model to solve GSM8K problems."
(followed by Trap 5's garbled-text hypothesis, and then this one)

### Discovery
The model was generating intermediate reasoning steps and code solutions instead of
the `#### <number>` answer format that `extract_answer()` expected. When 3 few-shot
examples demonstrating `Q: ... A: ... #### answer` format were prepended to the prompt,
accuracy jumped from 7% to 47-55%.

### Root Cause
The Qwen2.5-7B-Instruct base model was pre-trained with a specific chat template.
When prompted with raw question text (without few-shot examples or the chat template),
it didn't know what output format to use. The SFT training (with
`dataset_text_field="instruction"`) trained on plain question text, but the model's
pre-training bias toward conversational formats dominated during inference.

### Lesson
**For evaluation of instruction-tuned models:**
1. Always use few-shot prompting to establish the expected output format
2. Validate the evaluation on at least 10 manually inspected samples before
   trusting aggregate numbers
3. Document the exact prompt template used so results are reproducible
4. If accuracy jumps dramatically with prompt changes, suspect format mismatch,
   not model capability change

---

## Trap 7: CodeAlpaca Evaluation Format Fragility

### Phenomenon
A code evaluation script using few-shot prompting and sandbox execution returned
score 0.0 for 10/10 samples from a 7B model that clearly can generate valid code.

### Initial Misinterpretation
"The fine-tuned model cannot generate executable code."

### Discovery
Inspection of raw outputs revealed three independent failure modes:
1. **Multi-language outputs**: Prompts asking for Java, SQL, or JavaScript
   produced valid code in those languages, but the sandbox only ran Python
2. **Format inconsistency**: Generated code sometimes used ```python```
   markers, sometimes raw code, sometimes included `Output:` annotations.
   The extraction regex failed on most formats.
3. **Dependency failures**: Imports like `sklearn`, `pandas` worked but
   triggered `ModuleNotFoundError` in restricted sandbox environments.

### Root Cause
The evaluation assumed code generation would follow a single, predictable format
and that all prompts would produce Python code. CodeAlpaca's instruction set
spans multiple programming languages. A robust code evaluation system requires
language detection, language-appropriate execution, and format-agnostic extraction.

### Lesson
**Code generation evaluation requires:**
1. Language detection before execution (not all CodeAlpaca prompts target Python)
2. Format-agnostic code extraction (handle both fenced and unfenced code blocks)
3. Graceful handling of dependencies and imports in sandbox environments
4. Raw output inspection on a subset before trusting aggregate scores
5. When sandbox execution is unreliable, a simpler heuristic (syntax check +
   function presence) may be more robust than full execution

---

## Trap 8: Hypothesis Confirmation Bias — The Cross-Task Tradeoff That Wasn't

### Phenomenon
A single experiment (cosine_seed42 in v0) showed GSM8K accuracy peaking at step 350
(0.826) then declining to step 500 (0.751), while CodeAlpaca continued improving.
This suggested a "cross-task skill tradeoff" — the model traded math accuracy for
code ability.

### Initial Misinterpretation
"We have discovered a real cross-task quality divergence during multi-task
fine-tuning. This is a more interesting phenomenon than proxy-validation
misalignment and should become the paper's core narrative."

This was compelling enough that it was codified as **Hypothesis 2** in the
frozen definitions document, with explicit replication criteria.

### Discovery
When a clean v2 experiment was run under frozen definitions (correct proxy,
few-shot GSM8K evaluation), the full n=10 trajectory showed:
- **Spearman ρ_trend = -0.091** (no monotonic trend between step and accuracy)
- **All adjacent checkpoint differences**: p > 0.15 (not significant)
- **Best vs worst**: 55.5% vs 47.5%, p = 0.109 (not significant)
- **Minimum detectable difference** (α=0.05, n=200): 9.8%
- **Observed max range**: 8.0%

The "peak then decline" pattern was within measurement noise. The hypothesis
was formally classified as **NOT REPLICATED** per pre-defined criteria.

### Root Cause
The original observation used v0 data with synthetic proxy signals, inadequate
sample size (200 GSM8K questions), and retrospective pattern-finding rather
than pre-registered hypothesis testing.

### Lesson
**A compelling single observation is not a finding until it survives independent
replication with frozen definitions and adequate statistical power.** The antidote:
1. Pre-register replication criteria before running the replication
2. If the replication fails, document the failure — it IS the finding

---

## Summary: Cross-Cutting Lessons

| # | Trap | Type | Impact |
|---|------|------|--------|
| 1 | Spearman n=3 discretization | Statistical | False positive crossing detection |
| 2 | Silent synthetic fallback | Engineering | 3 weeks of false narrative |
| 3 | Proxy definition ambiguity | Definitional | 3 contradictory paper narratives |
| 4 | Proxy_main default priority | Code logic | Wrong selection criterion used |
| 5 | GPU memory fragmentation | Engineering | Garbled evaluation data (7% vs 55%) |
| 6 | GSM8K format mismatch | Evaluation | 7% vs 55% accuracy on same model |
| 7 | CodeAlpaca format fragility | Evaluation | 100% failure rate, all false negatives |
| 8 | Hypothesis confirmation bias | Meta | Re-allocation of 2 weeks of effort |

### Meta-Lesson: The Audit Loop

The most important lesson spans all seven traps: **no result should be trusted until
it has been independently audited.** Every trap in this catalog was discovered not
by the person who produced the result, but by a subsequent audit step. The audit
loop is:

1. **Initial result** → looks clean, supports the narrative
2. **Audit** → repeat the measurement with different settings, check raw data,
   verify code paths, check for silent fallbacks
3. **Discovery** → the clean result was an artifact
4. **This is the actual discovery** → not noise, not failure, but a documented
   methodological insight

When this loop alternates between "exciting result" and "turns out it was an artifact,"
the pattern SHOULD evoke caution — but the artifacts themselves ARE the contribution.
Each trap represents something that can mislead any researcher working on similar problems.
Documented honestly, they become the paper's primary value.
