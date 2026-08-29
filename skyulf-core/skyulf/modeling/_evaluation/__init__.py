"""Evaluation module for Skyulf models."""

from .classification import evaluate_classification_model
from .clustering import evaluate_clustering_model
from .common import downsample_curve, sanitize_metrics
from .metrics import (
    calculate_classification_metrics,
    calculate_clustering_metrics,
    calculate_regression_metrics,
)
from .regression import evaluate_regression_model
from .schemas import (
    ClassificationEvaluation,
    ClusterCentroid,
    ClusteringEvaluation,
    ConfusionMatrixData,
    CurveData,
    CurvePoint,
    ModelEvaluationReport,
    RegressionEvaluation,
    ResidualsData,
)
from .thresholds import apply_thresholds, optimize_thresholds

__all__ = [
    "ClassificationEvaluation",
    "ClusterCentroid",
    "ClusteringEvaluation",
    "ConfusionMatrixData",
    "CurveData",
    "CurvePoint",
    "ModelEvaluationReport",
    "RegressionEvaluation",
    "ResidualsData",
    "apply_thresholds",
    "calculate_classification_metrics",
    "calculate_clustering_metrics",
    "calculate_regression_metrics",
    "downsample_curve",
    "evaluate_classification_model",
    "evaluate_clustering_model",
    "evaluate_regression_model",
    "optimize_thresholds",
    "sanitize_metrics",
]
