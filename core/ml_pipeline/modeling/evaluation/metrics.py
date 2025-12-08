"""Evaluation metrics calculation."""

from __future__ import annotations

import importlib
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.preprocessing import label_binarize

_imblearn_metrics = None
try:
    _imblearn_metrics = importlib.import_module("imblearn.metrics")
except ModuleNotFoundError:
    pass

geometric_mean_score = None
if _imblearn_metrics is not None:
    geometric_mean_score = getattr(_imblearn_metrics, "geometric_mean_score", None)


def calculate_classification_metrics(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Compute classification metrics for predictions."""
    
    # Use DataFrame directly if possible to preserve feature names
    # Only convert to numpy if model doesn't support pandas or if X is not pandas
    
    predictions = model.predict(X)
    
    # For metrics calculation, we might need numpy arrays for y
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else y

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_arr, predictions)),
        "precision_weighted": float(precision_score(y_arr, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_arr, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_arr, predictions, average="weighted", zero_division=0)),
    }

    # Add unweighted metrics for binary classification
    try:
        unique_classes = np.unique(y_arr)
        if len(unique_classes) == 2:
            metrics["precision"] = float(precision_score(y_arr, predictions, average="binary", zero_division=0))
            metrics["recall"] = float(recall_score(y_arr, predictions, average="binary", zero_division=0))
            metrics["f1"] = float(f1_score(y_arr, predictions, average="binary", zero_division=0))
    except Exception:
        pass

    if geometric_mean_score is not None:
        try:
            metrics["g_score"] = float(geometric_mean_score(y_arr, predictions, average="weighted"))
        except Exception:
            pass

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                class_count = proba.shape[1]
                try:
                    if class_count == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_arr, proba[:, 1]))
                        metrics["pr_auc"] = float(average_precision_score(y_arr, proba[:, 1]))
                    else:
                        metrics["roc_auc_weighted"] = float(
                            roc_auc_score(y_arr, proba, multi_class="ovr", average="weighted")
                        )
                        classes = getattr(model, "classes_", None)
                        if classes is None or len(classes) != class_count:
                            classes = np.arange(class_count)
                        y_indicator = label_binarize(y_arr, classes=classes)
                        metrics["pr_auc_weighted"] = float(
                            average_precision_score(y_indicator, proba, average="weighted")
                        )
                except Exception:
                    pass
    except Exception:
        pass

    return metrics


def calculate_regression_metrics(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Compute regression metrics for predictions."""
    
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else X
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else y

    predictions = model.predict(X_arr)
    mse_value = mean_squared_error(y_arr, predictions)
    metrics: Dict[str, float] = {
        "mae": float(mean_absolute_error(y_arr, predictions)),
        "mse": float(mse_value),
        "rmse": float(math.sqrt(mse_value)),
        "r2": float(r2_score(y_arr, predictions)),
        "mape": float(mean_absolute_percentage_error(y_arr, predictions)),
    }
    
    return metrics
