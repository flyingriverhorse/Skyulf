"""Schemas for model evaluation artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CurvePoint(BaseModel):
    """A single point in a 2D curve."""

    x: float
    y: float


class CurveData(BaseModel):
    """Data for a curve (ROC, PR, etc)."""

    name: str
    points: List[CurvePoint]
    auc: Optional[float] = None


class ConfusionMatrixData(BaseModel):
    """Confusion matrix data."""

    labels: List[str]
    matrix: List[List[int]]


class ClassificationEvaluation(BaseModel):
    """Classification specific evaluation data."""

    confusion_matrix: Optional[ConfusionMatrixData] = None
    roc_curves: List[CurveData] = Field(default_factory=list)
    pr_curves: List[CurveData] = Field(default_factory=list)


class ResidualsData(BaseModel):
    """Residuals data for regression."""

    predicted: List[float]
    residuals: List[float]
    actual: List[float]


class RegressionEvaluation(BaseModel):
    """Regression specific evaluation data."""

    residuals: Optional[ResidualsData] = None
    prediction_error: Optional[Any] = None


class ModelEvaluationReport(BaseModel):
    """Evaluation report for a single dataset."""

    dataset_name: str
    metrics: Dict[str, float]
    classification: Optional[ClassificationEvaluation] = None
    regression: Optional[RegressionEvaluation] = None
