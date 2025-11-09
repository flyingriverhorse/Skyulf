"""Dataset profiling helpers for feature engineering nodes."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from pandas.api import types as pd_types

from core.data_ingestion.serialization import JSONSafeSerializer
from core.feature_engineering.schemas import (
    QuickProfileColumnSummary,
    QuickProfileCorrelation,
    QuickProfileDatasetMetrics,
    QuickProfileNumericSummary,
    QuickProfileValueCount,
)


def _determine_semantic_type(series: pd.Series) -> str:
    dtype = series.dtype
    if pd_types.is_bool_dtype(dtype):
        return "boolean"
    if pd_types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd_types.is_numeric_dtype(dtype):
        return "numeric"
    if pd_types.is_string_dtype(dtype):
        return "text"
    return str(dtype)


def build_quick_profile_payload(frame: pd.DataFrame) -> Dict[str, Any]:
    """Compute a lightweight dataset profile summarizing core statistics."""
    working_frame = frame.copy()

    row_count = int(working_frame.shape[0])
    column_count = int(working_frame.shape[1])

    duplicate_rows = int(working_frame.duplicated().sum()) if row_count else 0
    unique_rows = max(row_count - duplicate_rows, 0)
    missing_cells = int(working_frame.isna().sum().sum())
    total_cells = row_count * column_count if column_count else 0
    missing_percentage = float(round((missing_cells / total_cells) * 100.0, 2)) if total_cells else 0.0

    metrics = QuickProfileDatasetMetrics(
        row_count=row_count,
        column_count=column_count,
        missing_cells=missing_cells,
        missing_percentage=missing_percentage,
        duplicate_rows=duplicate_rows,
        unique_rows=unique_rows,
    )

    column_summaries: List[QuickProfileColumnSummary] = []
    warning_messages: Set[str] = set()

    for column_name in working_frame.columns:
        series = working_frame[column_name]
        dtype_repr = str(series.dtype)
        semantic_type = _determine_semantic_type(series)

        missing_count = int(series.isna().sum()) if row_count else 0
        missing_pct = float(round((missing_count / row_count) * 100.0, 2)) if row_count else 0.0
        distinct_count = int(series.nunique(dropna=True)) if row_count else 0

        sample_values_raw = series.dropna().head(5).tolist()
        sample_values = JSONSafeSerializer.clean_for_json(sample_values_raw)

        numeric_summary = None
        if pd_types.is_numeric_dtype(series) and not pd_types.is_bool_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce").astype(float).dropna()
            if not numeric_series.empty:
                percentiles = numeric_series.quantile([0.25, 0.5, 0.75])
                numeric_summary = QuickProfileNumericSummary(
                    mean=float(numeric_series.mean()),
                    std=float(numeric_series.std(ddof=0)) if numeric_series.size > 1 else 0.0,
                    minimum=float(numeric_series.min()),
                    maximum=float(numeric_series.max()),
                    percentile_25=float(percentiles.loc[0.25]) if 0.25 in percentiles.index else None,
                    percentile_50=float(percentiles.loc[0.5]) if 0.5 in percentiles.index else None,
                    percentile_75=float(percentiles.loc[0.75]) if 0.75 in percentiles.index else None,
                )

        top_values: List[QuickProfileValueCount] = []
        if row_count:
            value_counts = series.value_counts(dropna=False).head(5)
            for value, count in value_counts.items():
                count_value = int(count)
                if count_value < 0:
                    continue
                percentage = float(round((count_value / row_count) * 100.0, 2)) if row_count else 0.0
                if pd.isna(value):
                    display_value: Any = "Missing"
                else:
                    display_value = JSONSafeSerializer.clean_for_json(value)
                top_values.append(
                    QuickProfileValueCount(
                        value=display_value,
                        count=count_value,
                        percentage=percentage,
                    )
                )

        column_summary = QuickProfileColumnSummary(
            name=str(column_name),
            dtype=dtype_repr,
            semantic_type=semantic_type,
            missing_count=missing_count,
            missing_percentage=missing_pct,
            distinct_count=distinct_count,
            sample_values=sample_values if isinstance(sample_values, list) else [],
            numeric_summary=numeric_summary,
            top_values=top_values,
        )

        column_summaries.append(column_summary)

        if missing_pct >= 30.0:
            warning_messages.add(
                f"Column '{column_summary.name}' has {missing_pct:.1f}% missing values."
            )
        if distinct_count <= 1 and row_count > 0:
            warning_messages.add(
                f"Column '{column_summary.name}' has low variability in the sampled data."
            )

    numeric_columns = [
        column
        for column in working_frame.columns
        if (
            pd_types.is_numeric_dtype(working_frame[column])
            and not pd_types.is_bool_dtype(working_frame[column])
            and working_frame[column].notna().any()
        )
    ]

    correlations: List[QuickProfileCorrelation] = []
    if len(numeric_columns) >= 2:
        correlation_matrix = working_frame[numeric_columns].corr(method="pearson")
        correlation_entries: List[Tuple[str, str, float]] = []
        for index, column_a in enumerate(numeric_columns):
            for column_b in numeric_columns[index + 1 :]:
                coefficient = correlation_matrix.at[column_a, column_b]
                if pd.isna(coefficient):
                    continue
                coefficient_value = float(coefficient)
                correlation_entries.append((column_a, column_b, coefficient_value))
        correlation_entries.sort(key=lambda item: abs(item[2]), reverse=True)
        for column_a, column_b, coefficient in correlation_entries[:25]:
            correlations.append(
                QuickProfileCorrelation(
                    column_a=column_a,
                    column_b=column_b,
                    coefficient=coefficient,
                )
            )
            if abs(coefficient) >= 0.85:
                warning_messages.add(
                    f"Columns '{column_a}' and '{column_b}' show strong correlation ({coefficient:.2f})."
                )

    return {
        "metrics": metrics,
        "columns": column_summaries,
        "correlations": correlations,
        "warnings": sorted(warning_messages),
    }


__all__ = ["build_quick_profile_payload"]
