"""Regression split evaluation helpers."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .schemas import (
    ModelEvaluationResidualHistogram,
    ModelEvaluationResidualPoint,
    ModelEvaluationResiduals,
    ModelEvaluationSplitPayload,
)

from .metrics import calculate_regression_metrics
from .common import _downsample_indices, _sanitize_structure

logger = logging.getLogger(__name__)

_MAX_SCATTER_POINTS = 750


def build_regression_split_report(
    model: Any,
    *,
    split_name: str,
    features: Optional[pd.DataFrame],
    target: Optional[pd.Series],
    include_residuals: bool = True,
    max_scatter_points: int = _MAX_SCATTER_POINTS,
) -> ModelEvaluationSplitPayload:
    """Compute evaluation artefacts for a regression split."""

    notes: List[str] = []

    if features is None or target is None or target.empty:
        return ModelEvaluationSplitPayload(
            split=split_name,
            row_count=0,
            metrics={},
            notes=["Split has no rows available for evaluation."],
        )

    feature_frame = features
    y_array = target.to_numpy(dtype=float)

    try:
        predictions = model.predict(feature_frame)
    except Exception as exc:
        logger.exception("Model predictions failed during regression evaluation: %s", exc)
        return ModelEvaluationSplitPayload(
            split=split_name,
            row_count=int(y_array.shape[0]),
            metrics={},
            notes=["Model predictions failed."],
        )

    metrics = calculate_regression_metrics(model, feature_frame, target)
    metric_warnings: List[str] = []
    metrics = _sanitize_structure(metrics, warnings=metric_warnings, context=f"{split_name} metrics")
    if metric_warnings:
        notes.extend(metric_warnings)
        
    residual_payload: Optional[ModelEvaluationResiduals] = None

    if include_residuals:
        residuals = y_array - predictions
        bin_count = min(30, max(10, residuals.shape[0] // 3))
        hist_counts, bin_edges = np.histogram(residuals, bins=bin_count)
        histogram = ModelEvaluationResidualHistogram(
            bin_edges=[float(value) for value in bin_edges.tolist()],
            counts=[int(value) for value in hist_counts.tolist()],
        )

        scatter_indices = _downsample_indices(residuals.shape[0], max_scatter_points)
        scatter_points = [
            ModelEvaluationResidualPoint(
                actual=float(y_array[idx]),
                predicted=float(predictions[idx]),
            )
            for idx in scatter_indices
        ]

        summary_raw = {
            "residual_min": float(np.min(residuals)),
            "residual_max": float(np.max(residuals)),
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
        }
        summary_warnings: List[str] = []
        summary = _sanitize_structure(summary_raw, warnings=summary_warnings, context="residual summary")
        notes.extend(summary_warnings)

        residual_payload = ModelEvaluationResiduals(
            histogram=histogram,
            scatter=scatter_points,
            summary=summary or {},
        )

    return ModelEvaluationSplitPayload(
        split=split_name,
        row_count=int(y_array.shape[0]),
        metrics=metrics,
        residuals=residual_payload,
        notes=notes,
    )
