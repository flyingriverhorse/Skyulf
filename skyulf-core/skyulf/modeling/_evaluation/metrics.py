"""Evaluation metrics calculation."""

import contextlib
import importlib
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
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


def calculate_classification_metrics(
    model: Any, X: pd.DataFrame | SkyulfDataFrame, y: pd.Series | Any
) -> dict[str, float]:
    """Compute classification metrics for predictions."""

    # Convert to Numpy for compatibility
    X_np, y_np = SklearnBridge.to_sklearn((X, y))

    # Use DataFrame directly if possible to preserve feature names
    # Only convert to numpy if model doesn't support pandas or if X is not pandas

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

    # Add unweighted metrics for binary classification
    try:
        unique_classes = np.unique(y_arr)
        if len(unique_classes) == 2:
            # Determine the actual positive-class label rather than relying on
            # sklearn's default pos_label=1, which raises (silently swallowed
            # below) for non-{0,1} binary labels (e.g. "yes"/"no", {1,2},
            # {-1,1}) — mirrors the pos_label resolution already used in
            # _evaluation/classification.py.
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

    if geometric_mean_score is not None:
        with contextlib.suppress(Exception):
            metrics["g_score"] = float(geometric_mean_score(y_arr, predictions, average="weighted"))

    try:
        if hasattr(model, "predict_proba"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                proba = model.predict_proba(X_np)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                class_count = proba.shape[1]
                with contextlib.suppress(Exception):
                    metrics["log_loss"] = float(log_loss(y_arr, proba))
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

                        # OVR variants
                        ovr_weighted = float(
                            roc_auc_score(
                                y_arr, proba, multi_class="ovr", average="weighted", labels=classes
                            )
                        )
                        metrics["roc_auc_weighted"] = ovr_weighted  # kept for backward compat
                        metrics["roc_auc_ovr_weighted"] = ovr_weighted
                        metrics["roc_auc_ovr"] = float(
                            roc_auc_score(
                                y_arr, proba, multi_class="ovr", average="macro", labels=classes
                            )
                        )
                        # OVO variants
                        metrics["roc_auc_ovo"] = float(
                            roc_auc_score(
                                y_arr, proba, multi_class="ovo", average="macro", labels=classes
                            )
                        )
                        metrics["roc_auc_ovo_weighted"] = float(
                            roc_auc_score(
                                y_arr, proba, multi_class="ovo", average="weighted", labels=classes
                            )
                        )
                        y_indicator = label_binarize(y_arr, classes=classes)
                        metrics["pr_auc_weighted"] = float(
                            average_precision_score(y_indicator, proba, average="weighted")
                        )
                except Exception:
                    pass  # nosec B110 - weighted PR-AUC is an optional extra metric
    except Exception:
        pass  # nosec B110 - multiclass PR-AUC block is optional; other metrics still returned

    return metrics


def calculate_regression_metrics(
    model: Any, X: pd.DataFrame | SkyulfDataFrame, y: pd.Series | Any
) -> dict[str, float]:
    """Compute regression metrics for predictions."""

    # Convert to Numpy for compatibility
    X_np, y_np = SklearnBridge.to_sklearn((X, y))

    # Use DataFrame directly if possible to preserve feature names
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
