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


class ClusterCentroid(BaseModel):
    """A single cluster's centroid (mean feature values) and size."""

    cluster_id: int
    size: int
    percentage: float
    center: dict[str, float]
    profile: str = ""
    """Auto-generated human-readable label (e.g. "High petal_length, Low petal_width"),
    derived from the cluster's most distinguishing features vs. the dataset average.
    Not a real-world name (e.g. a species) — just a description of what makes this
    cluster stand out numerically."""


class ClusteringEvaluation(BaseModel):
    """Clustering (unsupervised) specific evaluation data."""

    n_clusters: int
    cluster_sizes: dict[str, int] = Field(default_factory=dict)
    centroids: list[ClusterCentroid] = Field(default_factory=list)
    reference_crosstab: dict[str, dict[str, int]] | None = None
    """Optional: if the user designated a "reference column" (e.g. a known label
    like species name, not used as a training feature), this is a
    cluster_id -> {reference_value: row_count} breakdown, letting the user see
    e.g. "Cluster 0 is 92% setosa" without the model ever seeing that column."""
    reference_column: str | None = None


class ModelEvaluationReport(BaseModel):
    """Evaluation report for a single dataset."""

    dataset_name: str
    metrics: dict[str, float]
    classification: ClassificationEvaluation | None = None
    regression: RegressionEvaluation | None = None
    clustering: ClusteringEvaluation | None = None
