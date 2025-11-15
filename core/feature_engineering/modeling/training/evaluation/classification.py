"""Classification split evaluation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from core.feature_engineering.schemas import (
    ModelEvaluationConfusionMatrix,
    ModelEvaluationPrecisionRecallCurve,
    ModelEvaluationRocCurve,
    ModelEvaluationSplitPayload,
)

from ...shared import _classification_metrics
from .common import _align_thresholds, _downsample_indices, _sanitize_structure

logger = logging.getLogger(__name__)

_MAX_CURVE_POINTS = 500


@dataclass
class _ClassificationContext:
    features: pd.DataFrame
    target_array: np.ndarray
    notes: List[str]


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


def _classification_empty_payload(split_name: str, notes: List[str]) -> ModelEvaluationSplitPayload:
    return ModelEvaluationSplitPayload(
        split=split_name,
        row_count=0,
        metrics={},
        confusion_matrix=None,
        roc_curves=[],
        pr_curves=[],
        residuals=None,
        notes=notes,
    )


def _prepare_classification_context(
    features: Optional[pd.DataFrame],
    target: Optional[pd.Series],
    split_name: str,
) -> Tuple[Optional[_ClassificationContext], Optional[ModelEvaluationSplitPayload]]:
    if features is None or target is None or target.empty:
        notes = ["Split has no rows available for evaluation."]
        return None, _classification_empty_payload(split_name, notes)
    return _ClassificationContext(features=features, target_array=target.to_numpy(), notes=[]), None


def _safe_classification_predictions(
    model,
    context: _ClassificationContext,
    split_name: str,
) -> Tuple[Optional[np.ndarray], Optional[ModelEvaluationSplitPayload]]:
    try:
        predictions = model.predict(context.features)
        return predictions, None
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Model predictions failed during evaluation: %s", exc)
        notes = [*context.notes, "Model predictions failed; see server logs for details."]
        return None, ModelEvaluationSplitPayload(
            split=split_name,
            row_count=int(context.target_array.shape[0]),
            metrics={},
            confusion_matrix=None,
            roc_curves=[],
            pr_curves=[],
            residuals=None,
            notes=notes,
        )


def _compute_classification_metrics(
    model,
    context: _ClassificationContext,
    split_name: str,
) -> Tuple[Dict[str, Any], List[str]]:
    metrics = _classification_metrics(model, context.features, context.target_array)
    metric_warnings: List[str] = []
    sanitized = _sanitize_structure(metrics, warnings=metric_warnings, context=f"{split_name} metrics")
    return sanitized, metric_warnings


def _resolve_effective_labels(
    label_names: Optional[Sequence[str]],
    model_classes: Optional[Sequence[Any]],
    target_array: np.ndarray,
    predictions: np.ndarray,
) -> List[str]:
    if label_names and len(list(label_names)):
        return [str(name) for name in label_names]
    if model_classes is not None and len(model_classes):
        return [str(value) for value in model_classes]
    discovered = sorted({int(value) for value in np.unique(np.concatenate((target_array, predictions)))})
    return [str(value) for value in discovered]


def _build_confusion_output(
    include_confusion: bool,
    target_array: np.ndarray,
    predictions: np.ndarray,
    labels: Sequence[str],
) -> Optional[ModelEvaluationConfusionMatrix]:
    if not include_confusion:
        return None
    return _build_confusion_matrix_payload(target_array, predictions, labels)


def _resolve_probability_outputs(model, feature_frame: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
    proba: Optional[np.ndarray] = None
    notes: List[str] = []
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(feature_frame)
        except Exception as exc:  # pragma: no cover - optional
            logger.debug("predict_proba failed for evaluation: %s", exc)
    if proba is None and hasattr(model, "decision_function"):
        try:
            decision = model.decision_function(feature_frame)
            if decision.ndim == 1:
                probabilities = 1 / (1 + np.exp(-decision))
                proba = np.vstack([1 - probabilities, probabilities]).T
        except Exception:
            logger.debug("decision_function unavailable for evaluation.")
    return proba, notes


def _prepare_positive_class_contexts(
    proba: np.ndarray,
    encoded_targets: np.ndarray,
    effective_labels: Sequence[str],
    split_name: str,
) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray]], List[str]]:
    notes: List[str] = []
    probability_matrix = proba
    if probability_matrix.ndim == 1:
        probability_matrix = np.vstack([1 - probability_matrix, probability_matrix]).T

    class_count = probability_matrix.shape[1]
    if class_count <= 1:
        notes.append("Model returned a single class; ROC/PR curves skipped.")
        return [], notes

    contexts: List[Tuple[int, np.ndarray, np.ndarray]] = []
    class_indices = range(class_count)
    if class_count == 2:
        class_indices = [class_count - 1]

    for class_idx in class_indices:
        scores = probability_matrix[:, class_idx]
        mask = (encoded_targets == class_idx).astype(int)
        positive_total = int(mask.sum())
        negative_total = int(mask.size - positive_total)
        if positive_total == 0:
            label = effective_labels[class_idx] if class_idx < len(effective_labels) else f"Class {class_idx}"
            notes.append(f"No samples for {label}; skipped ROC/PR for that class.")
            continue
        if negative_total == 0:
            label = effective_labels[class_idx] if class_idx < len(effective_labels) else f"Class {class_idx}"
            notes.append(f"All samples belonged to {label}; skipped ROC/PR for that class.")
            continue

        contexts.append((class_idx, scores, mask))

    if not contexts:
        notes.append("ROC/PR curves skipped because no class had both positive and negative samples.")

    return contexts, notes


def _encode_targets_for_probability_curves(
    target_array: np.ndarray,
    model_classes: Optional[Sequence[Any]],
) -> Tuple[Optional[np.ndarray], List[str]]:
    notes: List[str] = []

    if model_classes is None or not len(model_classes):
        try:
            encoded = target_array.astype(int)
            return encoded, notes
        except Exception:
            notes.append(
                "Model classes unavailable and target labels are non-numeric; ROC/PR curves skipped."
            )
            return None, notes

    class_to_index = {value: idx for idx, value in enumerate(model_classes)}
    encoded = np.full(target_array.shape[0], -1, dtype=int)

    missing = 0
    for idx, value in enumerate(target_array):
        mapped = class_to_index.get(value)
        if mapped is None:
            missing += 1
            continue
        encoded[idx] = mapped

    if missing:
        notes.append(
            f"{missing} sample(s) had labels absent from the model class list; treated as negatives for curves."
        )

    return encoded, notes


def _build_roc_curve_payload(
    positive_mask: np.ndarray,
    positive_scores: np.ndarray,
    effective_labels: Sequence[str],
    positive_index: int,
    *,
    max_curve_points: int,
    split_name: str,
) -> Tuple[Optional[ModelEvaluationRocCurve], List[str]]:
    notes: List[str] = []

    try:
        fpr, tpr, roc_thresholds = roc_curve(positive_mask, positive_scores)
        roc_auc = roc_auc_score(positive_mask, positive_scores)
    except ValueError as exc:
        logger.debug("ROC computation failed: %s", exc)
        return None, notes

    if not (fpr.size and tpr.size):
        return None, notes

    aligned_thresholds = _align_thresholds(roc_thresholds, fpr.size)
    fpr_array = np.asarray(fpr, dtype=float)
    tpr_array = np.asarray(tpr, dtype=float)
    thresholds_array = np.asarray(aligned_thresholds, dtype=float)

    valid_mask = np.isfinite(fpr_array) & np.isfinite(tpr_array)
    if not np.all(valid_mask):
        removed = int(np.count_nonzero(~valid_mask))
        if np.any(valid_mask):
            fpr_array = fpr_array[valid_mask]
            tpr_array = tpr_array[valid_mask]
            notes.append(f"Removed {removed} non-finite ROC points while evaluating {split_name}.")
        else:
            notes.append(f"ROC curve skipped for {split_name} because all points were non-finite.")
            return None, notes

        thresholds_array = thresholds_array[valid_mask]

    if not (fpr_array.size and tpr_array.size):
        return None, notes

    thresholds_array, clamp_note = _clamp_non_finite_thresholds(
        thresholds_array,
        split_name=split_name,
        curve_name="ROC",
    )
    if clamp_note:
        notes.append(clamp_note)

    idx = _downsample_indices(fpr_array.size, max_curve_points)
    thresholds_list = thresholds_array[idx].tolist()
    fpr_list = fpr_array[idx].tolist()
    tpr_list = tpr_array[idx].tolist()
    curve_warning: List[str] = []
    sanitized_thresholds = _sanitize_structure(
        thresholds_list,
        warnings=curve_warning,
        context=f"{split_name} ROC thresholds",
    )
    sanitized_auc = _sanitize_structure(
        roc_auc,
        warnings=curve_warning,
        context=f"{split_name} ROC auc",
    )
    if curve_warning:
        notes.extend(curve_warning)

    return (
        ModelEvaluationRocCurve(
            label=(
                effective_labels[positive_index]
                if positive_index < len(effective_labels)
                else f"Class {positive_index}"
            ),
            fpr=fpr_list,
            tpr=tpr_list,
            thresholds=sanitized_thresholds,
            auc=sanitized_auc if isinstance(sanitized_auc, float) else None,
        ),
        notes,
    )


def _build_pr_curve_payload(
    positive_mask: np.ndarray,
    positive_scores: np.ndarray,
    effective_labels: Sequence[str],
    positive_index: int,
    *,
    max_curve_points: int,
    split_name: str,
) -> Tuple[Optional[ModelEvaluationPrecisionRecallCurve], List[str]]:
    notes: List[str] = []

    try:
        precision, recall, pr_thresholds = precision_recall_curve(positive_mask, positive_scores)
        ap_score = average_precision_score(positive_mask, positive_scores)
    except ValueError as exc:
        logger.debug("PR computation failed: %s", exc)
        return None, notes

    if not (precision.size and recall.size):
        return None, notes

    aligned_pr_thresholds = _align_thresholds(pr_thresholds, precision.size)
    precision_array = np.asarray(precision, dtype=float)
    recall_array = np.asarray(recall, dtype=float)
    thresholds_array = np.asarray(aligned_pr_thresholds, dtype=float)

    valid_mask = np.isfinite(precision_array) & np.isfinite(recall_array)
    if not np.all(valid_mask):
        removed = int(np.count_nonzero(~valid_mask))
        if np.any(valid_mask):
            precision_array = precision_array[valid_mask]
            recall_array = recall_array[valid_mask]
            notes.append(f"Removed {removed} non-finite PR points while evaluating {split_name}.")
        else:
            notes.append(f"PR curve skipped for {split_name} because all points were non-finite.")
            return None, notes

        thresholds_array = thresholds_array[valid_mask]

    if not (precision_array.size and recall_array.size):
        return None, notes

    thresholds_array, clamp_note = _clamp_non_finite_thresholds(
        thresholds_array,
        split_name=split_name,
        curve_name="PR",
    )
    if clamp_note:
        notes.append(clamp_note)

    idx = _downsample_indices(precision_array.size, max_curve_points)
    thresholds_list = thresholds_array[idx].tolist()
    precision_list = precision_array[idx].tolist()
    recall_list = recall_array[idx].tolist()
    curve_warning: List[str] = []
    sanitized_thresholds = _sanitize_structure(
        thresholds_list,
        warnings=curve_warning,
        context=f"{split_name} PR thresholds",
    )
    sanitized_average_precision = _sanitize_structure(
        ap_score,
        warnings=curve_warning,
        context=f"{split_name} average precision",
    )
    if curve_warning:
        notes.extend(curve_warning)

    return (
        ModelEvaluationPrecisionRecallCurve(
            label=effective_labels[positive_index]
            if positive_index < len(effective_labels)
            else f"Class {positive_index}",
            recall=recall_list,
            precision=precision_list,
            thresholds=sanitized_thresholds,
            average_precision=(
                sanitized_average_precision if isinstance(sanitized_average_precision, float) else None
            ),
        ),
        notes,
    )


def _build_classification_curves(
    model,
    feature_frame: pd.DataFrame,
    target_array: np.ndarray,
    effective_labels: Sequence[str],
    *,
    include_curves: bool,
    max_curve_points: int,
    split_name: str,
) -> Tuple[List[ModelEvaluationRocCurve], List[ModelEvaluationPrecisionRecallCurve], List[str]]:
    if not include_curves:
        return [], [], []

    roc_payloads: List[ModelEvaluationRocCurve] = []
    pr_payloads: List[ModelEvaluationPrecisionRecallCurve] = []
    curve_notes: List[str] = []

    proba, probability_notes = _resolve_probability_outputs(model, feature_frame)
    if probability_notes:
        curve_notes.extend(probability_notes)
    if proba is None:
        curve_notes.append("Probability outputs unavailable; ROC/PR curves skipped.")
        return roc_payloads, pr_payloads, curve_notes

    classes_attr = getattr(model, "classes_", None)
    encoded_targets, encoding_notes = _encode_targets_for_probability_curves(target_array, classes_attr)
    if encoding_notes:
        curve_notes.extend(encoding_notes)
    if encoded_targets is None:
        return roc_payloads, pr_payloads, curve_notes

    contexts, context_notes = _prepare_positive_class_contexts(
        proba,
        encoded_targets,
        effective_labels,
        split_name,
    )
    if context_notes:
        curve_notes.extend(context_notes)
    if not contexts:
        return roc_payloads, pr_payloads, curve_notes

    for positive_index, positive_scores, positive_mask in contexts:
        roc_payload, roc_notes = _build_roc_curve_payload(
            positive_mask,
            positive_scores,
            effective_labels,
            positive_index,
            max_curve_points=max_curve_points,
            split_name=split_name,
        )
        if roc_payload is not None:
            roc_payloads.append(roc_payload)
        if roc_notes:
            curve_notes.extend(roc_notes)

        pr_payload, pr_notes = _build_pr_curve_payload(
            positive_mask,
            positive_scores,
            effective_labels,
            positive_index,
            max_curve_points=max_curve_points,
            split_name=split_name,
        )
        if pr_payload is not None:
            pr_payloads.append(pr_payload)
        if pr_notes:
            curve_notes.extend(pr_notes)

    return roc_payloads, pr_payloads, curve_notes


def build_classification_split_report(
    model,
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

    context, early_payload = _prepare_classification_context(features, target, split_name)
    if early_payload is not None:
        return early_payload
    assert context is not None  # for type checkers

    predictions, failure_payload = _safe_classification_predictions(model, context, split_name)
    if failure_payload is not None:
        return failure_payload
    if predictions is None:
        raise RuntimeError("Classification predictions missing despite successful preprocessing")

    metrics, metric_warnings = _compute_classification_metrics(model, context, split_name)
    if metric_warnings:
        context.notes.extend(metric_warnings)

    classes_attr = getattr(model, "classes_", None)
    effective_labels = _resolve_effective_labels(label_names, classes_attr, context.target_array, predictions)

    confusion_payload = _build_confusion_output(
        include_confusion,
        context.target_array,
        predictions,
        effective_labels,
    )

    roc_payloads, pr_payloads, curve_notes = _build_classification_curves(
        model,
        context.features,
        context.target_array,
        effective_labels,
        include_curves=include_curves,
        max_curve_points=max_curve_points,
        split_name=split_name,
    )
    if curve_notes:
        context.notes.extend(curve_notes)

    return ModelEvaluationSplitPayload(
        split=split_name,
        row_count=int(context.target_array.shape[0]),
        metrics=metrics,
        confusion_matrix=confusion_payload,
        roc_curves=roc_payloads,
        pr_curves=pr_payloads,
        residuals=None,
        notes=context.notes,
    )


__all__ = ["build_classification_split_report"]
