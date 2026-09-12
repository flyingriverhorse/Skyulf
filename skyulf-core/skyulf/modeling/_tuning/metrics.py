"""Tuning metric validation, aliasing, and scorer resolution.

Leaf module (F-18 split of ``engine.py``). Problem type is taken as an
explicit argument; the binary ``pos_label`` scorer pinning that fixes
string-label targets lives here so the fold loop and the searcher
strategies share one resolution path.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, get_scorer, make_scorer
from sklearn.preprocessing import label_binarize

from .._evaluation.metrics import geometric_mean_score
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
    "pr_auc": "average_precision",
    "log_loss": "neg_log_loss",
    "matthews_corrcoef": "matthews_corrcoef",
}

# Binary-default sklearn scorers whose score function takes ``pos_label``.
# roc_auc looks like one but isn't: roc_auc_score has no pos_label
# parameter (it derives the positive class from the label space), and
# multiclass variants (f1_weighted, ...) plus accuracy/balanced_accuracy/
# matthews_corrcoef don't take it either. ``average_precision`` (the scorer
# ``pr_auc`` aliases to) does default to pos_label=1, so it needs the same
# pinning as f1.
BINARY_POS_LABEL_METRICS: frozenset[str] = frozenset(
    {"average_precision", "f1", "precision", "recall"}
)


def _weighted_pr_auc(estimator: Any, X: Any, y_true: Any) -> float:
    """Score weighted PR-AUC against the estimator's probability class axis.

    Holdouts may omit trained classes, so their observed labels cannot define
    the probability columns. Binary models score only the trained positive
    class, including when that class is absent from the holdout.
    """
    classes = estimator.classes_
    proba = np.asarray(estimator.predict_proba(X))
    if len(classes) == 2:
        pos_probs = proba if proba.ndim == 1 else proba[:, 1]
        return average_precision_score(y_true, pos_probs, pos_label=classes[1])
    return average_precision_score(
        label_binarize(y_true, classes=classes), proba, average="weighted"
    )


def _pr_auc_weighted_scorer() -> Any:
    """Return an estimator-aware scorer so absent holdout classes stay aligned."""
    return _weighted_pr_auc


_G_SCORE_NEEDS_IMBLEARN = (
    "Configuration Error: 'g_score' requires the imbalanced-learn package. "
    "Install it, or select a metric that does not need it (e.g. 'f1_weighted')."
)


def _g_score(y_true: Any, y_pred: Any) -> float:
    """Weighted geometric-mean recall over hard predictions.

    ``geometric_mean_score`` cannot be handed to ``make_scorer`` directly: its
    signature declares ``pos_label``, so sklearn injects one, and resolving it
    raises ``pos_label=1 is not a valid label`` on a string label space. This
    signature declares no such parameter, so nothing is injected and
    ``average="weighted"`` ignores imblearn's own default.
    """
    metric = geometric_mean_score
    if metric is None:
        raise ValueError(_G_SCORE_NEEDS_IMBLEARN)
    return metric(y_true, y_pred, average="weighted")


def _g_score_scorer() -> Any:
    """Builds the weighted geometric-mean-recall scorer, read off ``predict``.

    Raises:
        ValueError: If ``imbalanced-learn`` is absent. Refusing here is what makes
            the search fail as a configuration error instead of scoring nothing and
            reporting "All trials failed".
    """
    if geometric_mean_score is None:
        raise ValueError(_G_SCORE_NEEDS_IMBLEARN)
    return make_scorer(_g_score, response_method="predict")


# Tuning metrics sklearn has no scorer name for. Held as builders, not scorers, so
# a missing optional dependency surfaces as one clear configuration error when the
# metric is asked for rather than at import time or as the stock
# "'g_score' is not a valid scoring value".
CUSTOM_SCORER_BUILDERS: dict[str, Callable[[], Any]] = {
    "pr_auc_weighted": _pr_auc_weighted_scorer,
    "g_score": _g_score_scorer,
}


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
    # ``pr_auc`` is aliased to sklearn's ``average_precision``, whose suffixed form
    # is not a scorer; the weighted PR-AUC is built locally instead.
    if original_metric == "pr_auc":
        return "pr_auc_weighted"
    return weighted


def resolve_metric(config: TuningConfig, y: Any, problem_type: str) -> str:
    """Validates the metric against the problem type and maps it to a scorer.

    Friendly aliases become sklearn scoring strings, and binary-default
    metrics switch to weighted for multiclass targets.
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
        if is_multiclass and metric in [
            "average_precision",
            "f1",
            "precision",
            "recall",
            "roc_auc",
        ]:
            metric = weight_metric_for_multiclass(metric, config.metric)

    return metric


def resolve_scorer(metric: str, y: Any, problem_type: str | None) -> Any:
    """The scorer for *metric*, with the binary ``pos_label`` default fixed.

    Names sklearn has no scorer for (``pr_auc_weighted``, ``g_score``) are built
    locally; everything else goes through ``get_scorer``.

    f1/precision/recall scorers assume ``pos_label=1``; targets whose
    label space does not contain 1 (e.g. raw string labels the fold-aware
    wrapper scores against before the chain encodes them) make every fold
    raise ``pos_label=1 is not a valid label`` and surface as all-NaN trials.
    Pin ``pos_label`` to the sorted-last class — the same convention
    ``apply_thresholds`` uses for the positive class — whenever the default
    cannot match. Numeric targets containing 1 keep the stock scorer.
    """
    builder = CUSTOM_SCORER_BUILDERS.get(metric)
    if builder is not None:
        return builder()
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
