"""Model evaluation diagnostics node and helpers."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from core.feature_engineering.schemas import (
    ModelEvaluationConfusionMatrix,
    ModelEvaluationNodeSignal,
    ModelEvaluationPrecisionRecallCurve,
    ModelEvaluationResidualHistogram,
    ModelEvaluationResidualPoint,
    ModelEvaluationResiduals,
    ModelEvaluationRocCurve,
    ModelEvaluationSplitPayload,
)

from .model_training_tasks import _classification_metrics, _regression_metrics

logger = logging.getLogger(__name__)

_DEFAULT_SPLITS: Tuple[str, ...] = ("train", "validation", "test")
_MAX_CURVE_POINTS = 500
_MAX_SCATTER_POINTS = 750

__all__ = [
    "apply_model_evaluation",
    "build_classification_split_report",
    "build_regression_split_report",
]


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            logger.debug("Failed to parse ISO datetime '%s'", value)
    return None


def _normalize_splits(raw_value: Any) -> List[str]:
    if not raw_value:
        return []
    if isinstance(raw_value, str):
        parts = [entry.strip().lower() for entry in raw_value.split(",")]
    elif isinstance(raw_value, Iterable):
        parts = [str(entry).strip().lower() for entry in raw_value]
    else:
        return []
    normalized: List[str] = []
    for entry in parts:
        if not entry:
            continue
        if entry in {"train", "training"}:
            normalized.append("train")
        elif entry in {"validation", "valid", "val"}:
            normalized.append("validation")
        elif entry in {"test", "testing"}:
            normalized.append("test")
    return normalized


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, (int, np.integer)):
        return True
    return False


def _coerce_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _sanitize_structure(value: Any, *, warnings: List[str], context: str) -> Any:
    if isinstance(value, dict):
        result: Dict[Any, Any] = {}
        for key, inner in value.items():
            result[key] = _sanitize_structure(inner, warnings=warnings, context=context)
        return result
    if isinstance(value, (list, tuple)):
        sanitized_items = [
            _sanitize_structure(item, warnings=warnings, context=context) for item in value
        ]
        return type(value)(sanitized_items)
    if isinstance(value, (float, np.floating, int, np.integer)):
        if _is_finite_number(value):
            return float(value) if isinstance(value, (float, np.floating)) else int(value)
        warnings.append(f"Removed non-finite numeric value from {context}.")
        return None
    return value


def _downsample_indices(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=int)
    indices = np.linspace(0, length - 1, num=limit, dtype=int)
    return np.unique(indices)


def _align_thresholds(thresholds: np.ndarray, target_size: int) -> np.ndarray:
    if thresholds.size == target_size:
        return thresholds
    if thresholds.size == 0:
        return np.zeros(target_size, dtype=float)
    if thresholds.size == target_size - 1:
        return np.append(thresholds, thresholds[-1])
    if thresholds.size > target_size:
        return thresholds[:target_size]
    pad_size = target_size - thresholds.size
    return np.append(thresholds, np.full(pad_size, thresholds[-1]))


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

    notes: List[str] = []
    if features is None or target is None or target.empty:
        notes.append("Split has no rows available for evaluation.")
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

    X_array = features.to_numpy(dtype=np.float64)
    y_array = target.to_numpy()

    try:
        predictions = model.predict(X_array)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Model predictions failed during evaluation: %s", exc)
        notes.append("Model predictions failed; see server logs for details.")
        return ModelEvaluationSplitPayload(
            split=split_name,
            row_count=int(y_array.shape[0]),
            metrics={},
            confusion_matrix=None,
            roc_curves=[],
            pr_curves=[],
            residuals=None,
            notes=notes,
        )

    metrics = _classification_metrics(model, X_array, y_array)
    metric_warnings: List[str] = []
    metrics = _sanitize_structure(metrics, warnings=metric_warnings, context=f"{split_name} metrics")
    confusion_payload: Optional[ModelEvaluationConfusionMatrix] = None
    roc_payloads: List[ModelEvaluationRocCurve] = []
    pr_payloads: List[ModelEvaluationPrecisionRecallCurve] = []

    classes_attr = getattr(model, "classes_", None)
    if label_names and len(list(label_names)):
        effective_labels = [str(name) for name in label_names]
    elif classes_attr is not None and len(classes_attr):
        effective_labels = [str(value) for value in classes_attr]
    else:
        discovered = sorted({int(value) for value in np.unique(np.concatenate((y_array, predictions)))})
        effective_labels = [str(value) for value in discovered]

    if include_confusion:
        confusion_payload = _build_confusion_matrix_payload(y_array, predictions, effective_labels)

    if include_curves:
        proba: Optional[np.ndarray] = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_array)
            except Exception as exc:  # pragma: no cover - optional
                logger.debug("predict_proba failed for evaluation: %s", exc)
        if proba is None and hasattr(model, "decision_function"):
            try:
                decision = model.decision_function(X_array)
                if decision.ndim == 1:
                    proba = 1 / (1 + np.exp(-decision))
                    proba = np.vstack([1 - proba, proba]).T
            except Exception:
                logger.debug("decision_function unavailable for evaluation.")

        if proba is None:
            notes.append("Probability outputs unavailable; ROC/PR curves skipped.")
        else:
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            class_count = proba.shape[1]
            if class_count <= 1:
                notes.append("Model returned a single class; ROC/PR curves skipped.")
            elif class_count > 2:
                notes.append("Multi-class curves are not yet supported; showing top-one-vs-rest metrics only.")
            if class_count >= 2:
                classes = np.arange(class_count)
                positive_index = class_count - 1
                if class_count > 2:
                    # Evaluate the class with highest average probability as representative
                    positive_index = int(np.argmax(proba.mean(axis=0)))
                positive_scores = proba[:, positive_index]
                positive_mask = (y_array == positive_index).astype(int)

                try:
                    fpr, tpr, roc_thresholds = roc_curve(positive_mask, positive_scores)
                    roc_auc = roc_auc_score(positive_mask, positive_scores)
                except ValueError as exc:
                    logger.debug("ROC computation failed: %s", exc)
                    fpr, tpr, roc_thresholds, roc_auc = np.array([]), np.array([]), np.array([]), None

                if fpr.size and tpr.size:
                    aligned_thresholds = _align_thresholds(roc_thresholds, fpr.size)
                    fpr_array = np.asarray(fpr, dtype=float)
                    tpr_array = np.asarray(tpr, dtype=float)
                    thresholds_array = np.asarray(aligned_thresholds, dtype=float)

                    valid_mask = np.isfinite(fpr_array) & np.isfinite(tpr_array) & np.isfinite(thresholds_array)
                    if not np.all(valid_mask):
                        removed = int(np.count_nonzero(~valid_mask))
                        if np.any(valid_mask):
                            fpr_array = fpr_array[valid_mask]
                            tpr_array = tpr_array[valid_mask]
                            thresholds_array = thresholds_array[valid_mask]
                            notes.append(
                                f"Removed {removed} non-finite ROC points while evaluating {split_name}."
                            )
                        else:
                            notes.append(
                                f"ROC curve skipped for {split_name} because all points were non-finite."
                            )
                            fpr_array = np.array([])

                    if fpr_array.size and tpr_array.size:
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
                        roc_payloads.append(
                            ModelEvaluationRocCurve(
                                label=effective_labels[positive_index]
                                if positive_index < len(effective_labels)
                                else f"Class {positive_index}",
                                fpr=fpr_list,
                                tpr=tpr_list,
                                thresholds=sanitized_thresholds,
                                auc=sanitized_auc if isinstance(sanitized_auc, float) else None,
                            )
                        )
                        if curve_warning:
                            notes.extend(curve_warning)

                try:
                    precision, recall, pr_thresholds = precision_recall_curve(positive_mask, positive_scores)
                    ap_score = average_precision_score(positive_mask, positive_scores)
                except ValueError as exc:
                    logger.debug("PR computation failed: %s", exc)
                    precision, recall, pr_thresholds, ap_score = np.array([]), np.array([]), np.array([]), None

                if precision.size and recall.size:
                    aligned_pr_thresholds = _align_thresholds(pr_thresholds, precision.size)
                    precision_array = np.asarray(precision, dtype=float)
                    recall_array = np.asarray(recall, dtype=float)
                    thresholds_array = np.asarray(aligned_pr_thresholds, dtype=float)

                    valid_mask = (
                        np.isfinite(precision_array)
                        & np.isfinite(recall_array)
                        & np.isfinite(thresholds_array)
                    )
                    if not np.all(valid_mask):
                        removed = int(np.count_nonzero(~valid_mask))
                        if np.any(valid_mask):
                            precision_array = precision_array[valid_mask]
                            recall_array = recall_array[valid_mask]
                            thresholds_array = thresholds_array[valid_mask]
                            notes.append(
                                f"Removed {removed} non-finite PR points while evaluating {split_name}."
                            )
                        else:
                            notes.append(
                                f"PR curve skipped for {split_name} because all points were non-finite."
                            )
                            precision_array = np.array([])

                    if precision_array.size and recall_array.size:
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
                        pr_payloads.append(
                            ModelEvaluationPrecisionRecallCurve(
                                label=effective_labels[positive_index]
                                if positive_index < len(effective_labels)
                                else f"Class {positive_index}",
                                recall=recall_list,
                                precision=precision_list,
                                thresholds=sanitized_thresholds,
                                average_precision=sanitized_average_precision
                                if isinstance(sanitized_average_precision, float)
                                else None,
                            )
                        )
                        if curve_warning:
                            notes.extend(curve_warning)

    if metric_warnings:
        notes.extend(metric_warnings)

    return ModelEvaluationSplitPayload(
        split=split_name,
        row_count=int(y_array.shape[0]),
        metrics=metrics,
        confusion_matrix=confusion_payload,
        roc_curves=roc_payloads,
        pr_curves=pr_payloads,
        residuals=None,
        notes=notes,
    )


def build_regression_split_report(
    model,
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
            confusion_matrix=None,
            roc_curves=[],
            pr_curves=[],
            residuals=None,
            notes=["Split has no rows available for evaluation."],
        )

    X_array = features.to_numpy(dtype=np.float64)
    y_array = target.to_numpy(dtype=float)

    try:
        predictions = model.predict(X_array)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Model predictions failed during regression evaluation: %s", exc)
        return ModelEvaluationSplitPayload(
            split=split_name,
            row_count=int(y_array.shape[0]),
            metrics={},
            confusion_matrix=None,
            roc_curves=[],
            pr_curves=[],
            residuals=None,
            notes=["Model predictions failed; see server logs for details."],
        )

    metrics = _regression_metrics(model, X_array, y_array)
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
        summary = _sanitize_structure(summary_raw, warnings=summary_warnings, context=f"{split_name} residual summary")
        if summary_warnings:
            notes.extend(summary_warnings)

        residual_payload = ModelEvaluationResiduals(
            histogram=histogram,
            scatter=scatter_points,
            summary=summary,
        )

    return ModelEvaluationSplitPayload(
        split=split_name,
        row_count=int(y_array.shape[0]),
        metrics=metrics,
        confusion_matrix=None,
        roc_curves=[],
        pr_curves=[],
        residuals=residual_payload,
        notes=notes,
    )


def apply_model_evaluation(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    *,
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, ModelEvaluationNodeSignal]:
    """Return unchanged frame alongside light-weight node diagnostics."""

    node_id = node.get("id")
    node_id_str = str(node_id) if node_id is not None else None

    config = (node.get("data") or {}).get("config") or {}
    raw_job_id = config.get("training_job_id")
    training_job_id = str(raw_job_id).strip() if isinstance(raw_job_id, str) else (
        str(raw_job_id) if raw_job_id is not None else ""
    )
    training_job_id = training_job_id or None

    raw_splits = config.get("splits")
    splits = _normalize_splits(raw_splits)
    if not splits and training_job_id:
        splits = list(_DEFAULT_SPLITS)

    last_evaluated = _parse_iso_datetime(config.get("last_evaluated_at"))

    signal = ModelEvaluationNodeSignal(
        node_id=node_id_str,
        training_job_id=training_job_id,
        splits=splits,
        has_evaluation=bool(last_evaluated),
        last_evaluated_at=last_evaluated,
        notes=[],
    )

    if not training_job_id:
        signal.notes.append("Select a training job from the sidebar to unlock evaluation diagnostics.")
        summary = "Model evaluation: waiting for training job selection"
        return frame, summary, signal

    if not splits:
        signal.notes.append("No dataset splits selected; configure at least one to run diagnostics.")
        summary = f"Model evaluation for job {training_job_id}: no splits selected"
        return frame, summary, signal

    split_label = ", ".join(splits)
    summary = f"Model evaluation configured for job {training_job_id} on {split_label}"
    return frame, summary, signal
