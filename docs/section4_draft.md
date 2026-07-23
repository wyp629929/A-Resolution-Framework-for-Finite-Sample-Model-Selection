## 4. Eight Analytical Traps Encountered

The finding in §3 was not reached directly. It emerged from a sequence of
analytical dead ends, each of which initially pointed toward a different
conclusion — some more interesting than the correct one. We document them here
because the traps themselves are the primary methodological contribution:
each represents a failure mode that can mislead any researcher investigating
proxy-quality alignment.

The traps fall into three categories; all share a meta-theme —
a compelling single observation is not a finding until it survives
independent replication with adequate statistical power
(Trap 8 makes this explicit).

| Category | Traps | Theme |
|----------|-------|-------|
| A. Statistical | T1, T8 | Small sample sizes create false patterns |
| B. Engineering | T2, T5 | Silent code paths corrupt data silently |
| C. Definitional | T3, T4, T6, T7 | Undefined terms produce contradictory results |

### Trap 1: Spearman Discretization at Small Window Sizes

**Phenomenon.** Computing Spearman's $\rho$ between proxy loss and validation
accuracy over a sliding window of $n=3$ checkpoints produced $\rho = -1.0$
at step 400 in 5/5 experiments — a perfect ranking inversion suggesting
systematic proxy-quality disagreement.

**Initial interpretation.** "We have discovered a robust, reproducible
crossing phenomenon, independent of seed and learning rate schedule."

**How it was discovered.** Increasing the window to $n=6$ eliminated the
effect entirely ($\rho$ remained in $[0.6, 0.94]$). At $n=10$, $\rho$ was
$0.88$--$0.94$ — strongly positive throughout.

**Root cause.** With $n=3$, Spearman's $\rho$ can take only five discrete
values $\{-1, -0.5, 0, 0.5, 1\}$. Any configuration where proxy and
validation are inversely ordered over three points yields $\rho = -1$.
This is not a meaningful measurement of alignment — it is a quantization
artifact.

**Lesson.** Small-window Spearman correlations should never be reported
without listing the set of possible values. $n \ge 6$ is the minimum for
meaningful interpretation.

---

### Trap 2: Silent Synthetic Fallback

**Phenomenon.** The proxy signal showed $\rho_{\text{align}} \approx 0.93$
with validation — strong evidence that the experimental setup was working.

**Initial interpretation.** "Proxy and validation are highly correlated.
This validates our measurement infrastructure."

**How it was discovered.** The proxy values were identical across all
four experiments at every step — impossible if they reflected real
checkpoint differences. The config file used `held_out_path = "data/held_out"`
while the actual file was `data/held_out.json`. The file-not-found error
silently triggered `_synthetic_proxy_signals()`, a deterministic function
of step number. No warning was printed; no data provenance flag was set.

**Root cause.** The code responsible for loading held-out data returned
an empty list on file-not-found. The next check (`if not held_out:`) fell
back to synthetic generation. Error handling was silent by design —
intended for development convenience, but activated in production.

**Lesson.** Any pipeline stage with a synthetic fallback must:
(1) print a conspicuous WARNING when triggered,
(2) set a `data_source` flag on all output,
(3) require explicit opt-in for synthetic mode in production.
We had none of these.

---

### Trap 3: Proxy Definition Ambiguity

**Phenomenon.** The paper's central claim — whether proxy argmax picks
a different checkpoint than validation argmax — depended entirely on which
of three incompatible "proxy" definitions was used:
question-only perplexity, question+answer perplexity, or win-rate against
a reference model. Each gave a different answer.

**Initial interpretation.** "We find that proxy loss is systematically
uncorrelated with validation quality." (Version 1 of the paper narrative.)

**How it was discovered.** Three months into the investigation, we had
not frozen a single definition of "the proxy." The codebase simultaneously
contained `proxy_loss`, `proxy_win_rate`, and `proxy_main` (which silently
preferred win-rate over loss). The paper's story changed each time we
changed which one we looked at.

**Root cause.** No written, agreed-upon definition existed at the start.
The `SignalScores` dataclass aggregated multiple metrics without a designated
"canonical" one.

**Lesson.** Freeze all variable definitions before the first experiment.
Write them down. Any change is a new experiment version. The code should
enforce this. (The canonical definition adopted throughout §3 is
question+answer perplexity on held-out samples, as specified in the
experimental protocol.)

---

### Trap 4: Hidden Default Priority in Aggregated Metrics

**Phenomenon.** The selection code and baselines used
`proxy_scores.proxy_main` as the selection criterion. This consistently
picked step 450, while validation favored step 500 — a clear argmax failure.

**Initial interpretation.** "Proxy argmax systematically selects a worse
checkpoint than validation argmax. This is the core misalignment."

**How it was discovered.** `proxy_main` was defined to prefer
`proxy_win_rate` when available, and fall back to `-proxy_loss` otherwise.
Since win-rate was always computed alongside loss, `proxy_main` was always
reflecting the win-rate — a synthetic signal — rather than the held-out loss.
The raw proxy_loss values showed no argmax signal at all (flat across all
500 steps), while the win-rate (synthetic) showed a clean but artificial peak
at step 450.

**Root cause.** A silent default priority in a property accessor. No caller
code checked which underlying metric `proxy_main` was actually using.

**Lesson.** Any aggregated metric with fallback logic must be explicitly
documented and checked. When multiple metrics are available, run selection
on all of them and report disagreements. (This is a direct downstream
consequence of Trap 2: the silent synthetic fallback produced a
plausible but artificial proxy signal, which `proxy_main`'s priority
chain then made the default selection criterion.)

---

### Trap 5: GPU Memory Fragmentation (and its interaction with Trap 6)

**Phenomenon.** Multi-checkpoint evaluation loops crashed after 4-5
iterations with errors like `"input_ids is on cuda, whereas the model is
on cpu"` or `"CUDA out of memory"`. Earlier iterations were fine; later
ones produced garbled text that looked like ~7% accuracy.

**Timeline.** Two independent issues produced the same symptom:
1. **Standalone scripts** (error_overlap.py, model_soup.py) showed 7% due to
   a prompt format issue (Trap 6), not memory. No GPU issue was present.
2. **Pipeline scripts** (fix_and_rerun.py) worked correctly (65--99%)
   because they had aggressive memory cleanup between iterations.
3. **The v1 experiment** (the first experiment with frozen definitions;
   see Trap 6 for the GSM8K evaluation details) showed 7% from Trap 6,
   but ALSO hit memory fragmentation on later checkpoints.

**Root cause.** PyTorch's CUDA memory allocator fragments over repeated
model load/unload cycles. After 4-5 iterations, `device_map="auto"` silently
offloads layers to CPU. The error primarily affects later checkpoints,
which can be mistaken for model degradation.

**Lesson.** (a) When multiple bugs produce the same symptom, fix one at a
time and verify. (b) For multi-checkpoint loops: use `torch.cuda.empty_cache()`,
`torch.cuda.synchronize()`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
(c) When accuracy drops by >50% between adjacent checkpoints, inspect raw
generated text before interpreting the change as model degradation.

---

### Trap 6: Evaluation Format Mismatch (Few-Shot Required)

**Phenomenon.** The v1 GSM8K evaluation (first experiment under frozen
definitions, preceding the v2 results in §3) returned ~7% accuracy with
raw question prompts, but 47--55% with three few-shot examples (the v2
results in §3.1). Same model, same checkpoints, same 200 questions —
only the prompt changed.

**Initial interpretation.** "The fine-tuned model cannot solve GSM8K
problems." (The same symptom led to different misinterpretations in
different scripts — see Trap 5.)

**How it was discovered.** The model was generating Python code, explanatory
text, and intermediate reasoning steps — coherent, not garbled. But it
almost never produced the `#### <number>` format that `extract_answer()`
expected. Prepending three few-shot examples in `Q: ... A: ... #### answer`
format immediately brought accuracy to 47--55%.

**Root cause.** The Qwen2.5-7B-Instruct base model was pre-trained with
a specific chat template. Raw question text alone did not signal which
output format to use. The SFT training (using `dataset_text_field`) trained
on plain instruction text, but the model's pre-training bias dominated
during zero-shot inference.

**Lesson.** Evaluate instruction-tuned models with few-shot prompting.
Validate on at least 10 manually inspected samples before trusting aggregate
numbers. Document the exact prompt template for reproducibility.

---

### Trap 7: Code Generation Evaluation Format Fragility

**Phenomenon.** A code evaluation script with few-shot prompting and
sandbox execution returned score 0.0 for 10/10 samples from a 7B model
that clearly can generate valid Python code.

**Initial interpretation.** "The fine-tuned model cannot generate executable
code."

**How it was discovered.** Raw output inspection revealed three independent
failure modes: (1) prompts asking for Java or SQL produced valid code in
those languages, but the sandbox only ran Python; (2) generated code
inconsistently used ```python``` markers, raw code, or included `Output:`
annotations — the extraction regex failed on most formats; (3) imports
like `sklearn` and `pandas` triggered `ModuleNotFoundError` in restricted
sandbox environments. None of these reflected model capability.

**Root cause.** The evaluation assumed code generation would follow a
single format and target only Python. CodeAlpaca's instruction set spans
multiple languages and expects no single output convention.

**Lesson.** Code evaluation requires language detection, format-agnostic
extraction, and dependency handling before scores can be trusted. When
sandbox execution is unreliable, a simpler heuristic (syntax check +
function presence) may be more robust than full execution.

---

### Trap 8: Hypothesis Confirmation Bias — The Cross-Task Tradeoff That Wasn't

**Phenomenon.** A single experiment showed GSM8K accuracy peaking at step
350 (0.826) then declining to step 500 (0.751), while CodeAlpaca continued
improving — a "cross-task skill tradeoff."

**Initial interpretation.** "We have discovered a real cross-task quality
divergence. This is more interesting than proxy-validation misalignment
and should become the paper's core narrative." This was codified as
Hypothesis 2 in the frozen definitions, with pre-registered replication
criteria.

**How it was discovered.** When a clean experiment was run under frozen
definitions (correct proxy, few-shot prompting), the full $n=10$ trajectory
showed $\rho_{\text{trend}} = -0.091$, all adjacent $p > 0.15$, and the
best-vs-worst gap of 8.0\% at $p = 0.109$. The "peak then decline" was
within measurement noise. The hypothesis was classified as **not replicated**
per the pre-defined criteria.

**Root cause.** The original observation was drawn from data produced by
the same evaluation pipeline later found to have a prompt format issue
(Trap 6: no few-shot prompting), combined with inadequate sample size
($N=200$) and retrospective pattern-finding. (Data provenance: these
pre-date the `data_source` tagging fix in Trap 2, and the GSM8K values
themselves were not affected by Trap 2's synthetic proxy issue.)
A compelling single observation was elevated to a narrative before it
survived replication.

**Lesson.** A compelling single observation is not a finding until it
survives independent replication with frozen definitions and adequate
statistical power. Pre-register replication criteria before running the
replication. If the replication fails, document the failure — it IS the
finding. This trap is a meta-example of the entire catalog: the most
dangerous error is not any individual bug, but the human tendency to find
patterns in noise and build narratives around them.
