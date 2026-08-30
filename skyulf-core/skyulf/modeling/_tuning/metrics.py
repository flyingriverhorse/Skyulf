"""Tuning metric validation, aliasing, and scorer resolution.

Leaf module (F-18 split of ``engine.py``). Problem type is taken as an
explicit argument; the binary ``pos_label`` scorer pinning that fixes
string-label targets lives here so the fold loop and the searcher
strategies share one resolution path.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer, make_scorer

from .schemas import TuningConfig

INVALID_REGRESSION_METRICS = frozenset(
    {
        "accuracy",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "f1_weighted",
        "balanced_accuracy",
        "log_loss",
        "matthews_corrcoef",
        "roc_auc_weighted",
        "roc_auc_ovr",
        "roc_auc_ovo",
        "roc_auc_ovr_weighted",
        "roc_auc_ovo_weighted",
        "pr_auc",
        "pr_auc_weighted",
        "g_score",
    }
)

METRIC_ALIAS_MAP: dict[str, str] = {
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "explained_variance": "explained_variance",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "f1_weighted": "f1_weighted",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
    "roc_auc_ovr": "roc_auc_ovr",
    "roc_auc_ovo": "roc_auc_ovo",
    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    "roc_auc_ovo_weighted": "roc_auc_ovo_weighted",
    "log_loss": "neg_log_loss",
    "matthews_corrcoef": "matthews_corrcoef",
}

# Binary-default sklearn scorers whose score function takes ``pos_label``.
# roc_auc looks like one but isn't: roc_auc_score has no pos_label
# parameter (it derives the positive class from the label space), and
# multiclass variants (f1_weighted, ...) plus accuracy/balanced_accuracy/
# matthews_corrcoef don't take it either.
BINARY_POS_LABEL_METRICS: frozenset[str] = frozenset({"f1", "precision", "recall"})


def validate_metric_for_problem_type(problem_type: str, metric: str) -> None:
    """Raises a clear ``ValueError`` if a classification-only metric is used for regression."""
    if problem_type == "regression" and metric in INVALID_REGRESSION_METRICS:
        raise ValueError(
            f"Configuration Error: You selected '{metric}' as the tuning metric, "
            "but this is a Regression model. "
            "Accuracy/F1/AUC are for Classification only. "
            "Please open 'Advanced Settings' on this node and select a regression metric "
            "(e.g., R2, RMSE, MAE)."
        )


def is_multiclass_target(y: Any) -> bool:
    """Returns whether ``y`` (a Series or ndarray) has more than 2 unique classes."""
    if isinstance(y, pd.Series):
        return y.nunique() > 2
    if isinstance(y, np.ndarray):
        return len(np.unique(y)) > 2
    return False


def weight_metric_for_multiclass(metric: str, original_metric: str) -> str:
    """Switches a binary-default metric to its weighted variant for multiclass targets."""
    weighted = f"{metric}_weighted"
    # roc_auc needs special handling (ovr/ovo) usually, but weighted often works for simple cases
    if original_metric == "roc_auc":  # Check original config metric name just in case
        return "roc_auc_ovr_weighted"
    return weighted


def resolve_metric(config: TuningConfig, y: Any, problem_type: str) -> str:
    """Validates the metric against the problem type, maps friendly aliases to sklearn
    scoring strings, and switches binary-default metrics to weighted for multiclass targets.
    """
    metric = config.metric

    # --- VALIDATION: Metric Consistency Check ---
    # The schema defaults metric to "accuracy". If the user is doing Regression but "accuracy"
    # (or another classification metric) is selected, we raise a clear error instead of crashing deeply in sklearn.
    validate_metric_for_problem_type(problem_type, metric)
    # -----------------------------------------------

    # Map common user-friendly metrics to sklearn scoring strings
    if metric in METRIC_ALIAS_MAP:
        metric = METRIC_ALIAS_MAP[metric]

    if problem_type == "classification":
        # Check if target is multiclass
        is_multiclass = is_multiclass_target(y)

        # If multiclass and metric is binary-default, switch to weighted
        # Note: We check against the mapped names now (e.g. "f1", "precision")
        if is_multiclass and metric in ["f1", "precision", "recall", "roc_auc"]:
            metric = weight_metric_for_multiclass(metric, config.metric)

    return metric


def resolve_scorer(metric: str, y: Any, problem_type: str | None) -> Any:
    """The sklearn scorer for *metric*, with the binary ``pos_label`` default fixed.

    f1/precision/recall scorers assume ``pos_label=1``; targets whose
    label space does not contain 1 (e.g. raw string labels the fold-aware
    wrapper scores against before the chain encodes them) make every fold
    raise ``pos_label=1 is not a valid label`` and surface as all-NaN trials.
    Pin ``pos_label`` to the sorted-last class — the same convention
    ``apply_thresholds`` uses for the positive class — whenever the default
    cannot match. Numeric targets containing 1 keep the stock scorer.
    """
    scorer = get_scorer(metric)
    if problem_type != "classification":
        return scorer
    if metric not in BINARY_POS_LABEL_METRICS:
        return scorer
    classes = np.unique(np.asarray(y))
    if classes.size != 2 or 1 in classes.tolist():
        return scorer
    pos_label = classes[1].item() if hasattr(classes[1], "item") else classes[1]
    return make_scorer(
        scorer._score_func,
        response_method=scorer._response_method,
        pos_label=pos_label,
    )
