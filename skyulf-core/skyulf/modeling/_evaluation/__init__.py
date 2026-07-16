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

__all__ = [
    "evaluate_classification_model",
    "evaluate_regression_model",
    "evaluate_clustering_model",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "calculate_clustering_metrics",
    "downsample_curve",
    "sanitize_metrics",
    "ModelEvaluationReport",
    "ClassificationEvaluation",
    "RegressionEvaluation",
    "ClusteringEvaluation",
    "ClusterCentroid",
    "ConfusionMatrixData",
    "CurveData",
    "CurvePoint",
    "ResidualsData",
]
