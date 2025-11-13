"""Data snapshot helpers for feature engineering previews."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, List, Literal, Optional

import pandas as pd

from core.data_ingestion.serialization import JSONSafeSerializer
from core.feature_engineering.schemas import (
    PipelinePreviewColumnSchema,
    PipelinePreviewColumnStat,
    PipelinePreviewMetrics,
    PipelinePreviewResponse,
    PipelinePreviewRowMissingStat,
    PipelinePreviewSchema,
    PipelinePreviewSignals,
    TrainModelDraftReadinessSnapshot,
)

PipelineLogicalFamily = Literal[
    "numeric",
    "integer",
    "categorical",
    "string",
    "datetime",
    "boolean",
    "unknown",
]


def _infer_logical_family(series: pd.Series) -> PipelineLogicalFamily:
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "numeric"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_string_dtype(series):
        return "string"
    return "unknown"


def _normalize_snapshot_value(value: Any) -> Any:
    """Normalize preview statistics for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(round(value, 6))
    if isinstance(value, (int, bool)):
        return value
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive
        return None


def build_data_snapshot_response(
    frame: pd.DataFrame,
    *,
    target_node_id: Optional[str],
    preview_rows: int,
    preview_total_rows: int,
    initial_sample_rows: Optional[int],
    applied_steps: List[str],
    metrics_requested_sample_size: int,
    modeling_signals: Optional[TrainModelDraftReadinessSnapshot] = None,
    signals: Optional[PipelinePreviewSignals] = None,
    include_signals: bool = True,
) -> PipelinePreviewResponse:
    """Construct the pipeline preview response for a data snapshot."""
    preview_frame = frame.reset_index(drop=True)
    preview_row_count = int(preview_frame.shape[0])
    column_count = int(preview_frame.shape[1])

    schema_columns: List[PipelinePreviewColumnSchema] = []
    if column_count:
        for column_name in preview_frame.columns:
            series = preview_frame[column_name]
            pandas_dtype = str(series.dtype)
            logical_family = _infer_logical_family(series)

            schema_columns.append(
                PipelinePreviewColumnSchema(
                    name=str(column_name),
                    pandas_dtype=pandas_dtype,
                    logical_family=logical_family,
                    nullable=bool(series.isna().any()),
                )
            )

    schema_signature: Optional[str] = None
    if schema_columns:
        payload = [
            {
                "name": column.name,
                "pandas_dtype": column.pandas_dtype,
                "logical_family": column.logical_family,
                "nullable": column.nullable,
            }
            for column in schema_columns
        ]
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        schema_signature = hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    schema = PipelinePreviewSchema(signature=schema_signature, columns=schema_columns)

    baseline_sample_rows = initial_sample_rows if isinstance(initial_sample_rows, int) else preview_row_count
    baseline_sample_rows = max(baseline_sample_rows, 0)

    try:
        baseline_total_rows = int(preview_total_rows)
    except (TypeError, ValueError):
        baseline_total_rows = 0

    # Calculate estimated full dataset row count after operations
    if baseline_total_rows > 0 and baseline_sample_rows > 0:
        # Estimate full dataset impact: (current_sample / original_sample) * original_total
        ratio = preview_row_count / baseline_sample_rows
        estimated_full_dataset_rows = int(baseline_total_rows * ratio)
        estimated_full_dataset_rows = max(estimated_full_dataset_rows, 0)
    else:
        estimated_full_dataset_rows = preview_row_count

    # Calculate statistics from sample
    sample_duplicate_rows = int(preview_frame.duplicated().sum()) if preview_row_count else 0
    sample_missing_by_column = preview_frame.isna().sum()
    sample_missing_cells = int(sample_missing_by_column.sum())

    # Estimate full dataset statistics using ratio
    if baseline_sample_rows > 0 and preview_row_count > 0:
        # Scale up sample stats to full dataset
        scale_factor = baseline_total_rows / baseline_sample_rows if baseline_total_rows > 0 else 1.0
        estimated_duplicate_rows = int(sample_duplicate_rows * scale_factor)
        estimated_missing_cells = int(sample_missing_cells * scale_factor)
    else:
        estimated_duplicate_rows = sample_duplicate_rows
        estimated_missing_cells = sample_missing_cells

    metrics = PipelinePreviewMetrics(
        row_count=estimated_full_dataset_rows,
        column_count=column_count,
        duplicate_rows=estimated_duplicate_rows,
        missing_cells=estimated_missing_cells,
        preview_rows=preview_row_count,
        total_rows=baseline_total_rows if baseline_total_rows > 0 else preview_row_count,
        requested_sample_size=metrics_requested_sample_size,
    )

    if include_signals:
        active_signals = signals or PipelinePreviewSignals()
        if modeling_signals and active_signals.modeling is None:
            active_signals.modeling = modeling_signals
    else:
        active_signals = None

    modeling_payload = modeling_signals
    if modeling_payload is None and active_signals is not None:
        modeling_payload = active_signals.modeling

    # Calculate full dataset impact for applied steps
    full_dataset_delta = estimated_full_dataset_rows - (
        baseline_total_rows if baseline_total_rows > 0 else preview_row_count
    )
    sample_delta = preview_row_count - baseline_sample_rows
    summary_message = (
        f"Full dataset after operations: {estimated_full_dataset_rows:,} rows "
        f"(original {baseline_total_rows if baseline_total_rows > 0 else preview_row_count:,}, "
        f"change {full_dataset_delta:+d}); "
        f"Preview sample: {preview_row_count:,} rows (delta {sample_delta:+d})"
    )
    enriched_steps = list(applied_steps or [])
    if summary_message not in enriched_steps:
        enriched_steps.append(summary_message)

    if preview_row_count == 0 and not column_count:
        return PipelinePreviewResponse(
            node_id=target_node_id,
            columns=[],
            sample_rows=[],
            metrics=metrics,
            column_stats=[],
            applied_steps=enriched_steps or ["No nodes executed"],
            schema_summary=schema,
            modeling_signals=modeling_payload,
            signals=active_signals,
        )

    row_missing_stats: List[PipelinePreviewRowMissingStat] = []

    column_stats: List[PipelinePreviewColumnStat] = []

    sample_limit = min(preview_row_count, 1000)  # Sample size for recommendations
    sample_preview_rows = preview_frame.head(sample_limit).to_dict("records") if sample_limit else []

    return PipelinePreviewResponse(
        node_id=target_node_id,
        columns=[str(col) for col in preview_frame.columns],
        sample_rows=JSONSafeSerializer.clean_for_json(sample_preview_rows),
        metrics=metrics,
        column_stats=column_stats,
        applied_steps=enriched_steps or ["No nodes executed"],
        row_missing_stats=row_missing_stats,
        schema_summary=schema,
        modeling_signals=modeling_payload,
        signals=active_signals,
    )


__all__ = [
    "build_data_snapshot_response",
]
