"""Tuning configuration schemas."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    strategy: Literal["grid", "random", "optuna", "halving_grid", "halving_random"] = "random"
    metric: str = "accuracy"  # or 'mse', 'f1', etc.
    n_trials: int = 10
    timeout: int | None = None  # Seconds
    search_space: dict[str, list[Any]] = field(default_factory=dict)  # e.g. {"C": [0.1, 1.0, 10.0]}
    strategy_params: dict[str, Any] = field(
        default_factory=dict
    )  # e.g. {"factor": 3, "sampler": "tpe"}
    cv_enabled: bool = True
    cv_folds: int = 5
    cv_type: Literal[
        "k_fold", "stratified_k_fold", "time_series_split", "shuffle_split", "nested_cv"
    ] = "k_fold"
    cv_shuffle: bool = True
    cv_random_state: int = 42
    random_state: int = 42
    # Parallelism — set by the backend from settings, not by the user directly.
    n_jobs: int = 1
    parallel_backend: str = ""


@dataclass
class TuningResult:
    """Result of a tuning session."""

    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    trials: list[dict[str, Any]]  # List of {params, score}
    scoring_metric: str | None = None  # Actual sklearn metric used (e.g. "f1_weighted")
