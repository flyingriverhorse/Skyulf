"""Training task exports and backward-compatible helpers."""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ...shared import _prepare_training_data, _resolve_training_inputs
from ...shared import evaluation as _shared_evaluation
from .execution import _train_and_save_model
from .workflow import (
    _run_training_workflow,
    dispatch_training_job,
    train_model,
    cancel_training_task,
)

def _classification_metrics(model, X, y):
    """Proxy to shared implementation for backward compatibility."""

    return _shared_evaluation._classification_metrics(model, X, y)


def _regression_metrics(model, X, y):
    """Proxy that honors monkeypatching of metric functions on this module."""

    original_r2 = _shared_evaluation.r2_score
    original_mape = _shared_evaluation.mean_absolute_percentage_error
    _shared_evaluation.r2_score = r2_score
    _shared_evaluation.mean_absolute_percentage_error = mean_absolute_percentage_error
    try:
        return _shared_evaluation._regression_metrics(model, X, y)
    finally:
        _shared_evaluation.r2_score = original_r2
        _shared_evaluation.mean_absolute_percentage_error = original_mape


__all__ = [
    "_classification_metrics",
    "_prepare_training_data",
    "_resolve_training_inputs",
    "_regression_metrics",
    "_train_and_save_model",
    "_run_training_workflow",
    "accuracy_score",
    "average_precision_score",
    "dispatch_training_job",
    "cancel_training_task",
    "f1_score",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
    "roc_auc_score",
    "train_model",
]
