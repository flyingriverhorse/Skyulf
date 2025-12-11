"""Schemas for model evaluation artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ModelEvaluationConfusionMatrix(BaseModel):
    """Structured confusion matrix payload for classification diagnostics."""

    labels: List[str] = Field(default_factory=list)
    matrix: List[List[int]] = Field(default_factory=list)
    normalized: Optional[List[List[float]]] = None
    totals: List[int] = Field(default_factory=list)
    accuracy: Optional[float] = None


class ModelEvaluationRocCurve(BaseModel):
    """Receiver operating characteristic curve payload."""

    label: str
    fpr: List[float] = Field(default_factory=list)
    tpr: List[float] = Field(default_factory=list)
    thresholds: List[float] = Field(default_factory=list)
    auc: Optional[float] = None


class ModelEvaluationPrecisionRecallCurve(BaseModel):
    """Precision/recall curve payload."""

    label: str
    recall: List[float] = Field(default_factory=list)
    precision: List[float] = Field(default_factory=list)
    thresholds: List[float] = Field(default_factory=list)
    average_precision: Optional[float] = None


class ModelEvaluationResidualHistogram(BaseModel):
    """Histogram summary for residual diagnostics."""

    bin_edges: List[float] = Field(default_factory=list)
    counts: List[int] = Field(default_factory=list)


class ModelEvaluationResidualPoint(BaseModel):
    """Sampled residual point used for scatter plots."""

    actual: float
    predicted: float


class ModelEvaluationResiduals(BaseModel):
    """Residual diagnostics payload for regression analyses."""

    histogram: ModelEvaluationResidualHistogram
    scatter: List[ModelEvaluationResidualPoint] = Field(default_factory=list)
    summary: Dict[str, float] = Field(default_factory=dict)


class ModelEvaluationSplitPayload(BaseModel):
    """Evaluation artefacts for a single dataset split."""

    split: str
    row_count: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    confusion_matrix: Optional[ModelEvaluationConfusionMatrix] = None
    roc_curves: List[ModelEvaluationRocCurve] = Field(default_factory=list)
    pr_curves: List[ModelEvaluationPrecisionRecallCurve] = Field(default_factory=list)
    residuals: Optional[ModelEvaluationResiduals] = None
    notes: List[str] = Field(default_factory=list)


class ModelEvaluationReport(BaseModel):
    """Aggregated evaluation report."""

    job_id: str
    pipeline_id: Optional[str] = None
    node_id: Optional[str] = None
    generated_at: datetime
    problem_type: Literal["classification", "regression"]
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    splits: Dict[str, ModelEvaluationSplitPayload] = Field(default_factory=dict)
