"""PAAS configuration schemas and data structures.

All pipeline stages communicate through these dataclasses.
No external dependencies beyond Python 3.10+ standard library + dataclasses.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# ─── Core data structures ────────────────────────────────────────────────────

@dataclass
class SignalScores:
    """Proxy and validation scores for one checkpoint."""
    proxy_loss: float | None = None
    proxy_win_rate: float | None = None
    gsm8k_accuracy: float | None = None
    code_func_score: float | None = None
    pairwise_bt_score: float | None = None

    @property
    def proxy_main(self) -> float:
        """Primary proxy signal: prefer win_rate if available, else loss inverted."""
        if self.proxy_win_rate is not None:
            return self.proxy_win_rate
        if self.proxy_loss is not None:
            return -self.proxy_loss  # invert so higher = better
        return 0.0

    @property
    def validation_main(self) -> float:
        """Primary validation signal: aggregate of available validation scores."""
        scores = []
        if self.gsm8k_accuracy is not None:
            scores.append(self.gsm8k_accuracy)
        if self.code_func_score is not None:
            scores.append(self.code_func_score)
        if self.pairwise_bt_score is not None:
            scores.append(self.pairwise_bt_score)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


@dataclass
class Checkpoint:
    """A single checkpoint produced during training."""
    step: int
    path: str
    seed: int
    schedule: str
    proxy_scores: SignalScores = field(default_factory=SignalScores)
    validation_scores: SignalScores = field(default_factory=SignalScores)
    metadata: dict = field(default_factory=dict)


@dataclass
class TriggerEvent:
    step: int
    rho_at_trigger: float
    trigger_type: Literal["caution_entered", "caution_exited", "no_window"]


@dataclass
class SelectionResult:
    selected: Checkpoint
    selection_mode: Literal["normal", "cautious_uniform", "cautious_fallback"]
    candidate_window: list[int] | None = None
    last_safe: Checkpoint | None = None


@dataclass
class OracleResult:
    mmlu: float | None = None
    humaneval: float | None = None
    mt_bench: float | None = None
    domain_specific: float | None = None
    aggregate: float | None = None


# ─── Configuration dataclasses ───────────────────────────────────────────────

@dataclass
class TrainConfig:
    base_model: str = "meta-llama/Llama-3-8B-hf"
    lora_rank: int = 16
    data_mix: list[str] = field(default_factory=lambda: ["gsm8k", "codealpaca", "dolly"])
    schedule: str = "cosine"
    seed: int = 42
    save_every_k_steps: int = 100
    total_steps: int = 1000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    output_dir: str = "results/checkpoints"
    cloud_machine: str = "AutoDL"  # identifier for cloud submission


@dataclass
class ProxyConfig:
    held_out_path: str = "data/held_out"
    win_rate_n: int = 200
    judge_model: str | None = None  # None = use training model self-judge (pilot-only)


@dataclass
class ValidationConfig:
    gsm8k_test_path: str = "data/gsm8k_test"
    code_test_path: str = "data/codealpaca_test"
    dolly_prompts_path: str = "data/dolly_sample.json"
    dolly_anchor_path: str = "data/dolly_anchor_responses.json"
    pairwise_label_dir: str = "data/pairwise_labels"
    use_human_labels: bool = False  # False = use self-judge for pilot


@dataclass
class MonitorConfig:
    rho_window: int = 3
    threshold: float = 0.5
    min_safe_count: int = 2
    min_points: int = 3  # don't evaluate triggers until window is full
    ci_method: Literal["bootstrap", "permutation"] = "permutation"


@dataclass
class SelectionConfig:
    window_frac: float = 0.2
    aggregation: Literal["uniform", "validation_guided"] = "uniform"


@dataclass
class OracleConfig:
    benchmarks: list[str] = field(default_factory=lambda: ["mmlu", "humaneval", "mt_bench"])
    batch_size: int = 8


@dataclass
class PilotConfig:
    """Pilot-specific overrides. Uses tiny model + reduced scope to validate pipeline."""
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    total_steps: int = 200
    save_every_k_steps: int = 20
    data_mix: list[str] = field(default_factory=lambda: ["gsm8k"])  # Dolly + CodeAlpaca signal mocked
    rho_window: int = 3
    threshold: float = 0.5
    override_trigger_test: bool = True  # if True, run P5a/P5b forced-trigger tests


@dataclass
class ExperimentConfig:
    experiment_name: str = "paas_default"
    train: TrainConfig = field(default_factory=TrainConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    baselines: list[str] = field(default_factory=lambda: ["proxy_best", "uniform_ensemble"])
    seed: int = 42
    output_dir: str = "results"
