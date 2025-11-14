"""Draft training node validations for pipeline previews."""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import pandas as pd

from core.feature_engineering.schemas import (
    TrainModelDraftReadinessSnapshot,
    TrainModelDraftTargetInsight,
)

ProblemType = Literal["classification", "regression"]
DEFAULT_PROBLEM_TYPE: ProblemType = "classification"


def _normalize_problem_type(raw_value: Any) -> ProblemType:
    """Sanitize the configured problem type, defaulting to classification."""

    if isinstance(raw_value, str):
        lowered = raw_value.strip().lower()
        if lowered in {"classification", "regression"}:
            return "classification" if lowered == "classification" else "regression"
    return DEFAULT_PROBLEM_TYPE


def _infer_problem_type(series: pd.Series, preferred: Any) -> ProblemType:
    if preferred == "regression":
        return "regression"
    return DEFAULT_PROBLEM_TYPE


def apply_train_model_draft(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, TrainModelDraftReadinessSnapshot]:
    """
    Inspect the preview frame and surface basic modelling readiness signals.

    Note: During preview, this passes through the input DataFrame unchanged.
    The actual model training happens asynchronously via Celery jobs.
    After training, the Train Model node should be treated as outputting
    model metadata (job_id, artifact_uri, metrics) rather than data.

    For downstream nodes (Evaluation, Feature Importance, etc.), they should:
    1. Query training jobs via API to get available models
    2. Allow user selection of which model(s) to use
    3. Load model artifacts via job_id/artifact_uri
    """
    metadata = TrainModelDraftReadinessSnapshot(row_count=int(frame.shape[0]))
    metadata.feature_columns = [str(column) for column in frame.columns]
    metadata.feature_count = len(metadata.feature_columns)

    if frame.empty:
        metadata.blockers.append("No data available in preview sample")
        metadata.ready_for_training = False
        return frame, "Train model draft: no data available", metadata

    data = node.get("data") or {}
    config = data.get("config") or {}

    raw_target = config.get("target_column")
    target_column = str(raw_target).strip() if raw_target is not None else ""
    configured_problem_type: ProblemType = _normalize_problem_type(config.get("problem_type"))
    target_summary = TrainModelDraftTargetInsight(
        name=target_column,
        configured_problem_type=configured_problem_type,
    )
    metadata.target = target_summary if target_column else None

    if not target_column:
        metadata.blockers.append("Target column not configured")
        metadata.ready_for_training = False
        return frame, "Train model draft: target column not configured", metadata

    if target_column not in frame.columns:
        metadata.blockers.append(f"Target column '{target_column}' not found in preview data")
        metadata.ready_for_training = False
        return frame, f"Train model draft: target column '{target_column}' not found", metadata

    target_series = frame[target_column]
    target_summary.pandas_dtype = str(target_series.dtype)
    target_summary.missing_count = int(target_series.isna().sum())
    target_summary.distinct_count = int(target_series.nunique(dropna=True))

    problem_type: ProblemType = _infer_problem_type(target_series, configured_problem_type)
    target_summary.inferred_problem_type = problem_type

    feature_columns = [column for column in frame.columns if column != target_column]
    metadata.feature_columns = [str(column) for column in feature_columns]
    metadata.feature_count = len(metadata.feature_columns)

    if not feature_columns:
        metadata.blockers.append("Requires at least one feature column besides the target")
        metadata.ready_for_training = False
        return frame, "Train model draft: requires at least one feature column besides the target", metadata

    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
    non_numeric_features = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(frame[column])]
    feature_columns_with_missing = [column for column in feature_columns if frame[column].isna().any()]

    metadata.numeric_features = [str(column) for column in numeric_features]
    metadata.non_numeric_features = [str(column) for column in non_numeric_features]
    metadata.features_with_missing = [str(column) for column in feature_columns_with_missing]

    parts = [
        f"Train model draft: target '{target_column}' ({problem_type})",
        f"rows {frame.shape[0]}, features {len(feature_columns)}",
        f"numeric={len(numeric_features)} non-numeric={len(non_numeric_features)}",
        f"target-cardinality={target_summary.distinct_count}",
    ]

    if target_summary.missing_count:
        parts.append(f"target-missing={target_summary.missing_count}")
        metadata.warnings.append(f"Target column contains {target_summary.missing_count} missing values")

    if feature_columns_with_missing:
        preview = ", ".join(str(column) for column in feature_columns_with_missing[:3])
        if len(feature_columns_with_missing) > 3:
            preview = f"{preview}, ..."
        parts.append(f"feature-missing: {preview}")
        missing_preview = ", ".join(metadata.features_with_missing[:5])
        if len(metadata.features_with_missing) > 5:
            missing_preview = f"{missing_preview} ..."
        metadata.warnings.append(f"Feature columns with missing values: {missing_preview}")

    if problem_type == "classification" and target_summary.distinct_count < 2:
        parts.append("insufficient target classes")
        metadata.blockers.append("Classification requires at least two distinct target classes")

    if problem_type == "regression" and not numeric_features:
        parts.append("no numeric features for regression")
        metadata.blockers.append("Regression requires at least one numeric feature column")

    metadata.ready_for_training = len(metadata.blockers) == 0

    return frame, "; ".join(parts), metadata
