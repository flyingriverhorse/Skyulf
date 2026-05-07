"""Evaluation module for Skyulf models."""

from .classification import evaluate_classification_model
from .common import downsample_curve, sanitize_metrics
from .metrics import calculate_classification_metrics, calculate_regression_metrics
from .regression import evaluate_regression_model
from .schemas import (
    ClassificationEvaluation,
    ConfusionMatrixData,
    CurveData,
    CurvePoint,
    ModelEvaluationReport,
    RegressionEvaluation,
    ResidualsData,
)

__all__ = [
    "evaluate_classification_model",
    "evaluate_regression_model",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "downsample_curve",
    "sanitize_metrics",
    "ModelEvaluationReport",
    "ClassificationEvaluation",
    "RegressionEvaluation",
    "ConfusionMatrixData",
    "CurveData",
    "CurvePoint",
    "ResidualsData",
]
