"""Schemas for model evaluation artifacts."""

from typing import Any

from pydantic import BaseModel, Field


class CurvePoint(BaseModel):
    """A single point in a 2D curve."""

    x: float
    y: float


class CurveData(BaseModel):
    """Data for a curve (ROC, PR, etc)."""

    name: str
    points: list[CurvePoint]
    auc: float | None = None


class ConfusionMatrixData(BaseModel):
    """Confusion matrix data."""

    labels: list[str]
    matrix: list[list[int]]


class ClassificationEvaluation(BaseModel):
    """Classification specific evaluation data."""

    confusion_matrix: ConfusionMatrixData | None = None
    roc_curves: list[CurveData] = Field(default_factory=list)
    pr_curves: list[CurveData] = Field(default_factory=list)


class ResidualsData(BaseModel):
    """Residuals data for regression."""

    predicted: list[float]
    residuals: list[float]
    actual: list[float]


class RegressionEvaluation(BaseModel):
    """Regression specific evaluation data."""

    residuals: ResidualsData | None = None
    prediction_error: Any | None = None


class ModelEvaluationReport(BaseModel):
    """Evaluation report for a single dataset."""

    dataset_name: str
    metrics: dict[str, float]
    classification: ClassificationEvaluation | None = None
    regression: RegressionEvaluation | None = None
