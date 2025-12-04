"""Classification split evaluation helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .schemas import (
    ModelEvaluationConfusionMatrix,
    ModelEvaluationPrecisionRecallCurve,
    ModelEvaluationRocCurve,
    ModelEvaluationSplitPayload,
)

from .metrics import calculate_classification_metrics
from .common import _align_thresholds, _downsample_indices, _sanitize_structure

logger = logging.getLogger(__name__)

_MAX_CURVE_POINTS = 500


def _clamp_non_finite_thresholds(
    values: np.ndarray,
    *,
    split_name: str,
    curve_name: str,
) -> Tuple[np.ndarray, Optional[str]]:
    if not values.size:
        return values, None

    sanitized = values.astype(float, copy=True)
    finite_mask = np.isfinite(sanitized)
    if np.all(finite_mask):
        return sanitized, None

    finite_values = sanitized[finite_mask]
    fallback_high = float(np.max(finite_values)) if finite_values.size else 0.0
    fallback_low = float(np.min(finite_values)) if finite_values.size else 0.0

    pos_inf_mask = np.isposinf(sanitized)
    neg_inf_mask = np.isneginf(sanitized)
    nan_mask = np.isnan(sanitized)

    if pos_inf_mask.any():
        sanitized[pos_inf_mask] = fallback_high
    if neg_inf_mask.any():
        sanitized[neg_inf_mask] = fallback_low
    if nan_mask.any():
        sanitized[nan_mask] = fallback_low

    note: Optional[str] = None
    nan_replacements = int(np.count_nonzero(nan_mask))
    if nan_replacements:
        note = (
            f"Clamped {nan_replacements} undefined {curve_name} threshold(s) while evaluating {split_name}."
        )

    return sanitized, note


def _build_confusion_matrix_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> ModelEvaluationConfusionMatrix:
    label_indices = np.arange(len(label_names))
    counts = confusion_matrix(y_true, y_pred, labels=label_indices)
    totals = counts.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = counts.astype(float) / totals[:, None]
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    accuracy = float(np.trace(counts) / counts.sum()) if counts.sum() else None

    return ModelEvaluationConfusionMatrix(
        labels=[str(name) for name in label_names],
        matrix=counts.astype(int).tolist(),
        normalized=normalized.tolist(),
        totals=totals.astype(int).tolist(),
        accuracy=accuracy,
    )


def build_classification_split_report(
    model: Any,
    *,
    split_name: str,
    features: Optional[pd.DataFrame],
    target: Optional[pd.Series],
    label_names: Optional[Sequence[str]] = None,
    include_confusion: bool = True,
    include_curves: bool = True,
    max_curve_points: int = _MAX_CURVE_POINTS,
) -> ModelEvaluationSplitPayload:
    """Compute evaluation artefacts for a classification split."""

    if features is None or target is None or target.empty:
        return ModelEvaluationSplitPayload(
            split=split_name,
            row_count=0,
            metrics={},
            notes=["Split has no rows available for evaluation."],
        )

    target_array = target.to_numpy()
    
    # 1. Predictions
    try:
        predictions = model.predict(features)
    except Exception as exc:
        logger.exception("Model predictions failed during evaluation: %s", exc)
        return ModelEvaluationSplitPayload(
            split=split_name,
            row_count=len(target_array),
            metrics={},
            notes=["Model predictions failed."],
        )

    # 2. Metrics
    metrics = calculate_classification_metrics(model, features, target)
    metric_warnings: List[str] = []
    metrics = _sanitize_structure(metrics, warnings=metric_warnings, context=f"{split_name} metrics")
    
    # 3. Resolve Labels
    model_classes = getattr(model, "classes_", None)
    if label_names:
        effective_labels = [str(name) for name in label_names]
    elif model_classes is not None:
        effective_labels = [str(c) for c in model_classes]
    else:
        effective_labels = [str(c) for c in np.unique(np.concatenate([target_array, predictions]))]

    # 4. Confusion Matrix
    confusion_payload = None
    if include_confusion:
        confusion_payload = _build_confusion_matrix_payload(target_array, predictions, effective_labels)

    # 5. Curves (ROC/PR)
    roc_curves = []
    pr_curves = []
    notes = list(metric_warnings)

    if include_curves:
        # Get probabilities
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(features)
            except Exception:
                pass
        
        if proba is not None:
            # Handle binary vs multiclass
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            
            n_classes = proba.shape[1]
            classes_to_plot = range(n_classes)
            if n_classes == 2:
                classes_to_plot = [1] # Only plot positive class for binary

            # Encode targets
            # Assuming target_array matches model.classes_ order
            # If model.classes_ exists, map target values to indices
            encoded_targets = target_array
            if model_classes is not None:
                class_map = {c: i for i, c in enumerate(model_classes)}
                # Fallback for unmapped
                encoded_targets = np.array([class_map.get(x, -1) for x in target_array])

            for i in classes_to_plot:
                label = effective_labels[i] if i < len(effective_labels) else f"Class {i}"
                
                # Binary mask for this class
                mask = (encoded_targets == i).astype(int)
                scores = proba[:, i]
                
                if mask.sum() == 0 or mask.sum() == len(mask):
                    continue # Skip if only one class present

                # ROC
                try:
                    fpr, tpr, thresholds = roc_curve(mask, scores)
                    auc = roc_auc_score(mask, scores)
                    
                    indices = _downsample_indices(len(fpr), max_curve_points)
                    roc_curves.append(ModelEvaluationRocCurve(
                        label=label,
                        fpr=fpr[indices].tolist(),
                        tpr=tpr[indices].tolist(),
                        thresholds=thresholds[indices].tolist(),
                        auc=float(auc)
                    ))
                except Exception:
                    pass

                # PR
                try:
                    precision, recall, thresholds = precision_recall_curve(mask, scores)
                    # thresholds is 1 shorter than p/r
                    thresholds = _align_thresholds(thresholds, len(precision))
                    
                    indices = _downsample_indices(len(precision), max_curve_points)
                    pr_curves.append(ModelEvaluationPrecisionRecallCurve(
                        label=label,
                        precision=precision[indices].tolist(),
                        recall=recall[indices].tolist(),
                        thresholds=thresholds[indices].tolist(),
                        average_precision=None # Calculate if needed
                    ))
                except Exception:
                    pass

    return ModelEvaluationSplitPayload(
        split=split_name,
        row_count=len(target_array),
        metrics=metrics,
        confusion_matrix=confusion_payload,
        roc_curves=roc_curves,
        pr_curves=pr_curves,
        notes=notes,
    )
