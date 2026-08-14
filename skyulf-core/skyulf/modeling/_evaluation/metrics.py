"""Evaluation metrics calculation."""

import contextlib
import importlib
import math
import numbers
import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from ...engines import SkyulfDataFrame
from ...modeling.sklearn_wrapper import SklearnBridge

_imblearn_metrics = None
with contextlib.suppress(ModuleNotFoundError):
    _imblearn_metrics = importlib.import_module("imblearn.metrics")

geometric_mean_score = None
if _imblearn_metrics is not None:
    geometric_mean_score = getattr(_imblearn_metrics, "geometric_mean_score", None)

DEFAULT_SILHOUETTE_SAMPLE_SIZE = 10_000
DEFAULT_SILHOUETTE_RANDOM_STATE = 42
_NAN_CLUSTER_LABEL = object()
SilhouetteSampleSize = int | np.integer[Any]


def _validate_silhouette_sample_size(sample_size: Any) -> int:
    """Validate silhouette sample caps and return a plain Python integer."""
    if isinstance(sample_size, (bool, np.bool_)) or not isinstance(
        sample_size, (numbers.Integral, np.integer)
    ):
        raise ValueError("silhouette_sample_size must be an integer")
    validated_sample_size = int(sample_size)
    if validated_sample_size < 2:
        raise ValueError("silhouette_sample_size must be at least 2")
    return validated_sample_size


def _cluster_label_key(label: Any) -> Any:
    """Return a stable dictionary key for a scalar predicted cluster label."""
    if isinstance(label, (float, np.floating)) and np.isnan(label):
        return _NAN_CLUSTER_LABEL
    return label


def _collect_silhouette_representatives(
    labels: np.ndarray,
    *,
    sample_size: SilhouetteSampleSize,
) -> dict[Any, int]:
    """Retain first cluster occurrences without exceeding the scoring cap."""
    sample_size = _validate_silhouette_sample_size(sample_size)
    representatives: dict[Any, int] = {}
    for index, label in enumerate(labels):
        key = _cluster_label_key(label)
        if key in representatives:
            continue
        if len(representatives) == sample_size:
            raise ValueError(
                f"silhouette_sample_size={sample_size} is too small for more than "
                f"{sample_size} clusters; increase it above the number of clusters "
                "when scoring datasets larger than the cap"
            )
        representatives[key] = int(index)
    return representatives


def _select_silhouette_sample_indices(
    labels: np.ndarray,
    *,
    sample_size: SilhouetteSampleSize,
    random_state: int,
    representative_by_label: dict[Any, int] | None = None,
) -> np.ndarray:
    """Select a deterministic bounded silhouette sample that keeps every cluster represented."""
    sample_size = _validate_silhouette_sample_size(sample_size)
    n_samples = len(labels)
    if n_samples <= sample_size:
        return np.arange(n_samples, dtype=int)

    representatives = (
        representative_by_label
        if representative_by_label is not None
        else _collect_silhouette_representatives(labels, sample_size=sample_size)
    )
    n_clusters = len(representatives)
    if sample_size <= n_clusters:
        raise ValueError(
            f"silhouette_sample_size={sample_size} is too small for {n_clusters} clusters; "
            "increase it above the number of clusters when scoring datasets larger than the cap"
        )

    required_indices = list(representatives.values())
    required_index_set = set(required_indices)
    remaining_slots = sample_size - len(required_indices)
    optional_indices: list[int] = []
    optional_seen = 0
    rng = np.random.RandomState(random_state)

    for index in range(n_samples):
        if index in required_index_set:
            continue
        optional_seen += 1
        if len(optional_indices) < remaining_slots:
            optional_indices.append(index)
            continue
        replacement_index = rng.randint(optional_seen)
        if replacement_index < remaining_slots:
            optional_indices[replacement_index] = index

    selected = required_indices + optional_indices
    return np.asarray(selected, dtype=int)


def calculate_classification_metrics(
    model: Any,
    X: pd.DataFrame | SkyulfDataFrame,
    y: pd.Series | Any,
    *,
    X_np: Any = None,
    y_np: Any = None,
    predictions: Any = None,
    proba: Any = None,
) -> dict[str, float]:
    """Compute classification metrics for predictions.

    ``X_np``/``y_np``/``predictions``/``proba`` let a caller that already
    converted ``X``/``y`` to numpy and/or already called
    ``model.predict()``/``model.predict_proba()`` (e.g.
    ``evaluate_classification_model``) pass those results straight through,
    avoiding a redundant conversion/inference pass on the same data. When
    omitted, each is (re)computed here exactly as before.
    """

    # Convert to Numpy for compatibility (skip if the caller already has it)
    if X_np is None or y_np is None:
        X_np, y_np = SklearnBridge.to_sklearn((X, y))

    # Use DataFrame directly if possible to preserve feature names
    # Only convert to numpy if model doesn't support pandas or if X is not pandas

    if predictions is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            predictions = model.predict(X_np)

    # For metrics calculation, we might need numpy arrays for y
    y_arr = y_np

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_arr, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_arr, predictions)),
        "precision_weighted": float(
            precision_score(y_arr, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_arr, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_arr, predictions, average="weighted", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_arr, predictions)),
    }

    _add_binary_unweighted_metrics(metrics, model, y_arr, predictions)

    if geometric_mean_score is not None:
        with contextlib.suppress(Exception):
            metrics["g_score"] = float(geometric_mean_score(y_arr, predictions, average="weighted"))

    _add_probability_based_metrics(metrics, model, X_np, y_arr, proba=proba)

    return metrics


def _add_binary_unweighted_metrics(
    metrics: dict[str, float], model: Any, y_arr: Any, predictions: Any
) -> None:
    """Adds unweighted precision/recall/f1 to ``metrics`` in-place for binary classification.

    Determines the actual positive-class label rather than relying on sklearn's default
    pos_label=1, which raises (silently swallowed below) for non-{0,1} binary labels
    (e.g. "yes"/"no", {1,2}, {-1,1}) — mirrors the pos_label resolution already used in
    _evaluation/classification.py. No-op (and swallows errors) outside binary classification.
    """
    try:
        unique_classes = np.unique(y_arr)
        if len(unique_classes) == 2:
            classes_ = getattr(model, "classes_", None)
            pos_label = (
                classes_[1] if classes_ is not None and len(classes_) == 2 else unique_classes[1]
            )
            metrics["precision"] = float(
                precision_score(
                    y_arr, predictions, average="binary", pos_label=pos_label, zero_division=0
                )
            )
            metrics["recall"] = float(
                recall_score(
                    y_arr, predictions, average="binary", pos_label=pos_label, zero_division=0
                )
            )
            metrics["f1"] = float(
                f1_score(y_arr, predictions, average="binary", pos_label=pos_label, zero_division=0)
            )
    except Exception:
        pass


def _add_multiclass_roc_pr_auc_metrics(
    metrics: dict[str, float], y_arr: Any, proba: Any, classes: Any, class_count: int
) -> None:
    """Adds OVR/OVO ROC-AUC variants and weighted PR-AUC to ``metrics`` in-place for multiclass proba."""
    # OVR variants
    ovr_weighted = float(
        roc_auc_score(y_arr, proba, multi_class="ovr", average="weighted", labels=classes)
    )
    metrics["roc_auc_weighted"] = ovr_weighted  # kept for backward compat
    metrics["roc_auc_ovr_weighted"] = ovr_weighted
    metrics["roc_auc_ovr"] = float(
        roc_auc_score(y_arr, proba, multi_class="ovr", average="macro", labels=classes)
    )
    # OVO variants
    metrics["roc_auc_ovo"] = float(
        roc_auc_score(y_arr, proba, multi_class="ovo", average="macro", labels=classes)
    )
    metrics["roc_auc_ovo_weighted"] = float(
        roc_auc_score(y_arr, proba, multi_class="ovo", average="weighted", labels=classes)
    )
    y_indicator = label_binarize(y_arr, classes=classes)
    metrics["pr_auc_weighted"] = float(
        average_precision_score(y_indicator, proba, average="weighted")
    )


def _add_roc_pr_auc_metrics(
    metrics: dict[str, float], model: Any, y_arr: Any, proba: Any, class_count: int
) -> None:
    """Adds ROC-AUC/PR-AUC metrics (binary or multiclass OVR/OVO) to ``metrics`` in-place.

    Swallows errors so a single failing metric doesn't drop the others already computed.
    """
    try:
        if class_count == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_arr, proba[:, 1]))
            metrics["pr_auc"] = float(average_precision_score(y_arr, proba[:, 1]))
        else:
            # Explicitly pass the full label set the model was trained on
            # (`classes`, resolved below) so a CV fold whose validation
            # split happens not to contain every trained class doesn't
            # raise "Number of classes in y_true not equal to columns
            # in y_score" — previously swallowed silently by the
            # surrounding except, dropping these metrics entirely.
            classes = getattr(model, "classes_", None)
            if classes is None or len(classes) != class_count:
                classes = np.arange(class_count)

            _add_multiclass_roc_pr_auc_metrics(metrics, y_arr, proba, classes, class_count)
    except Exception:
        pass  # nosec B110 - weighted PR-AUC is an optional extra metric


def _add_probability_based_metrics(
    metrics: dict[str, float], model: Any, X_np: Any, y_arr: Any, *, proba: Any = None
) -> None:
    """Adds log-loss, ROC-AUC and PR-AUC metrics to ``metrics`` in-place, using ``predict_proba``.

    No-op if the model doesn't expose ``predict_proba``, or any of these optional metrics
    fail to compute (errors are swallowed so other metrics are still returned).
    ``proba`` lets a caller that already called ``model.predict_proba()`` pass the result
    through instead of triggering another redundant inference pass.
    """
    try:
        if proba is None and hasattr(model, "predict_proba"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                proba = model.predict_proba(X_np)
        if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
            class_count = proba.shape[1]
            with contextlib.suppress(Exception):
                metrics["log_loss"] = float(log_loss(y_arr, proba))
            _add_roc_pr_auc_metrics(metrics, model, y_arr, proba, class_count)
    except Exception:
        pass  # nosec B110 - multiclass PR-AUC block is optional; other metrics still returned


def calculate_regression_metrics(
    model: Any,
    X: pd.DataFrame | SkyulfDataFrame,
    y: pd.Series | Any,
    *,
    X_np: Any = None,
    y_np: Any = None,
    predictions: Any = None,
) -> dict[str, float]:
    """Compute regression metrics for predictions.

    ``X_np``/``y_np``/``predictions`` let a caller that already converted
    ``X``/``y`` to numpy and/or already called ``model.predict()`` (e.g.
    ``evaluate_regression_model``) pass those results straight through,
    avoiding a redundant conversion/inference pass on the same data.
    """

    # Convert to Numpy for compatibility (skip if the caller already has it)
    if X_np is None or y_np is None:
        X_np, y_np = SklearnBridge.to_sklearn((X, y))

    # Use DataFrame directly if possible to preserve feature names
    if predictions is None:
        predictions = model.predict(X_np)

    y_arr = y_np

    mse_value = mean_squared_error(y_arr, predictions)
    metrics: dict[str, float] = {
        "mae": float(mean_absolute_error(y_arr, predictions)),
        "mse": float(mse_value),
        "rmse": float(math.sqrt(mse_value)),
        "r2": float(r2_score(y_arr, predictions)),
        "mape": float(mean_absolute_percentage_error(y_arr, predictions)),
        "explained_variance": float(explained_variance_score(y_arr, predictions)),
    }

    return metrics


def calculate_clustering_metrics(
    X: pd.DataFrame | pl.DataFrame | SkyulfDataFrame,
    labels: Any,
    *,
    silhouette_sample_size: SilhouetteSampleSize = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    random_state: int = DEFAULT_SILHOUETTE_RANDOM_STATE,
) -> dict[str, float]:
    """Compute unsupervised clustering-quality metrics for a fitted model's labels.

    All three metrics only need the feature matrix and the cluster labels
    (no ground-truth target), so they can be computed on any split a KMeans
    model has genuinely predicted on (train/test/validation alike).
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    X_np, _ = SklearnBridge.to_sklearn((X, None))
    labels_np = np.asarray(labels).ravel()
    silhouette_sample_size = _validate_silhouette_sample_size(silhouette_sample_size)
    # Guard before the row-count check: polars collapses a 0-column frame to
    # shape (0, 0), so otherwise a 0-feature input raises the misleading
    # row-count error instead of the sklearn-style "0 feature" message.
    if X_np.ndim != 2 or X_np.shape[1] == 0:
        raise ValueError(
            f"Found array with 0 feature(s) (shape={X_np.shape}) while a minimum of 1 is required."
        )
    if X_np.shape[0] != len(labels_np):
        raise ValueError("X and labels must have the same number of rows")

    n_samples = len(labels_np)
    representative_by_label = _collect_silhouette_representatives(
        labels_np,
        sample_size=silhouette_sample_size,
    )
    n_unique = len(representative_by_label)
    metrics: dict[str, float] = {"n_clusters": float(n_unique)}

    # These metrics are undefined for fewer than 2 clusters, or when the
    # cluster count reaches the sample count — guard rather than let sklearn raise.
    if 1 < n_unique < n_samples:
        sampled_indices = _select_silhouette_sample_indices(
            labels_np,
            sample_size=silhouette_sample_size,
            random_state=random_state,
            representative_by_label=representative_by_label,
        )
        sampled_X = X_np[sampled_indices]
        sampled_labels = labels_np[sampled_indices]
        metrics["silhouette_score"] = float(silhouette_score(sampled_X, sampled_labels))
        metrics["silhouette_sample_size"] = float(len(sampled_indices))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_np, labels_np))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_np, labels_np))

    return metrics
