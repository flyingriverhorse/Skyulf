"""Tuning configuration schemas."""

from dataclasses import dataclass, field
from typing import Any, Literal

from ...types import DEFAULT_RANDOM_STATE


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
    cv_random_state: int = DEFAULT_RANDOM_STATE
    random_state: int = DEFAULT_RANDOM_STATE
    # Console progress for core-only use (a self-updating trial line on TTYs
    # plus an end-of-run best/trials summary). Off by default; the backend
    # never sets it and supplies its own progress_callback instead.
    progress: bool = False
    # Column to sort by chronologically when cv_type == "time_series_split"
    # (also dropped from features so it doesn't leak into training). Mirrors
    # the `time_column` parameter accepted by cross_validation.perform_cross_validation().
    cv_time_column: str | None = None
    # Parallelism — set by the backend from settings, not by the user directly.
    n_jobs: int = 1
    parallel_backend: str = ""
    # F-13: after tuning a binary classifier, also search the decision
    # threshold that maximises the configured metric on the validation split
    # and apply it in predict(). Off by default so nothing changes silently;
    # requires a validation split, a classifier with predict_proba, and a
    # binary target — otherwise it logs a skip and leaves predict on the
    # default decision rule.
    tune_threshold: bool = False


@dataclass
class TuningResult:
    """Result of a tuning session."""

    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    trials: list[dict[str, Any]]  # List of {params, score}
    scoring_metric: str | None = None  # Actual sklearn metric used (e.g. "f1_weighted")
    # F-13: per-class decision thresholds selected on the validation split
    # (populated only when tune_threshold=True and the gates pass). ``None``
    # means predict() keeps the model's default decision rule.
    decision_thresholds: dict[Any, float] | None = None
    # Name of the hard-label metric the thresholds were tuned against (may
    # differ from the tuning metric when that metric needs probabilities,
    # e.g. roc_auc falls back to balanced_accuracy).
    decision_threshold_metric: str | None = None
