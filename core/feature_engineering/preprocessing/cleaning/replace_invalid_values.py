"""Replace invalid numeric values based on configurable rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from ...shared.utils import _auto_detect_numeric_columns, _coerce_string_list
from core.feature_engineering.schemas import (
    ReplaceInvalidValuesAppliedColumnSignal,
    ReplaceInvalidValuesNodeSignal,
)

MODE_LABEL_MAP = {
    "negative_to_nan": "negative to missing",
    "zero_to_nan": "zero to missing",
    "percentage_bounds": "percentage bounds",
    "age_bounds": "age bounds",
    "custom_range": "custom bounds",
}


@dataclass
class _ReplacementPlan:
    target_columns: List[str]
    auto_columns: List[str]
    missing_columns: List[str]
    mode: str
    min_value: Optional[float]
    max_value: Optional[float]


@dataclass
class _ReplacementResult:
    frame: pd.DataFrame
    processed_columns: int
    total_replacements: int
    skipped_columns: List[str]
    processed_signals: List[ReplaceInvalidValuesAppliedColumnSignal]


def _resolve_columns(frame: pd.DataFrame, configured_columns: List[str]) -> Tuple[List[str], List[str], List[str]]:
    target_columns = [column for column in configured_columns if column in frame.columns]
    missing_columns = [column for column in configured_columns if column not in frame.columns]

    auto_columns: List[str] = []
    if not target_columns:
        auto_columns = _auto_detect_numeric_columns(frame)
        target_columns = auto_columns

    return list(target_columns), list(auto_columns), list(missing_columns)


def _normalize_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or "negative_to_nan").strip().lower()
    if mode in MODE_LABEL_MAP:
        return mode
    return "negative_to_nan"


def _coerce_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _compute_bounds(
    mode: str,
    min_value: Optional[float],
    max_value: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    if mode == "percentage_bounds":
        lower = min_value if min_value is not None else 0.0
        upper = max_value if max_value is not None else 100.0
        return lower, upper
    if mode == "age_bounds":
        lower = min_value if min_value is not None else 0.0
        upper = max_value if max_value is not None else 120.0
        return lower, upper
    return min_value, max_value


def _build_replacement_mask(
    numeric_series: pd.Series,
    numeric_valid: pd.Series,
    mode: str,
    min_value: Optional[float],
    max_value: Optional[float],
) -> Optional[pd.Series]:
    mask = pd.Series(False, index=numeric_series.index)

    if mode == "negative_to_nan":
        threshold = min_value if min_value is not None else 0.0
        lower_mask = numeric_valid & (numeric_series < threshold)
        upper_mask = (
            pd.Series(False, index=numeric_series.index)
            if max_value is None
            else numeric_valid & (numeric_series > max_value)
        )
        return lower_mask | upper_mask

    if mode == "zero_to_nan":
        return numeric_valid & numeric_series.abs().le(1e-12)

    if mode == "age_bounds":
        lower, upper = _compute_bounds(mode, min_value, max_value)
        return numeric_valid & ((numeric_series < lower) | (numeric_series > upper))

    if mode == "custom_range":
        if min_value is None and max_value is None:
            return None
        if min_value is not None:
            mask = mask | (numeric_valid & (numeric_series < min_value))
        if max_value is not None:
            mask = mask | (numeric_valid & (numeric_series > max_value))
        return mask

    return mask


def _replacement_token(series: pd.Series) -> Any:
    if pd_types.is_integer_dtype(series) and not pd_types.is_bool_dtype(series):
        return pd.NA
    return np.nan


def _process_percentage_column(
    working_frame: pd.DataFrame,
    column: str,
    series: pd.Series,
    numeric_series: pd.Series,
    min_value: Optional[float],
    max_value: Optional[float],
    auto_columns: List[str],
) -> Tuple[Optional[ReplaceInvalidValuesAppliedColumnSignal], Optional[str]]:
    numeric_valid = numeric_series.notna()
    finite_values = numeric_series[numeric_valid]

    if finite_values.empty:
        return None, column

    scale_min = float(finite_values.min())
    scale_max = float(finite_values.max())

    if np.isclose(scale_max, scale_min):
        return None, column

    scaled = (numeric_series - scale_min) / (scale_max - scale_min)
    scaled *= 100.0

    clip_lower = 0.0 if min_value is None else float(min_value)
    clip_upper = 100.0 if max_value is None else float(max_value)
    scaled = scaled.clip(lower=clip_lower, upper=clip_upper)

    updated_series = series.copy()
    updated_series.loc[numeric_valid] = scaled.loc[numeric_valid]
    working_frame[column] = updated_series

    original_values = numeric_series.loc[numeric_valid]
    scaled_values = scaled.loc[numeric_valid]
    changes = int(
        (~np.isclose(original_values.to_numpy(), scaled_values.to_numpy(), equal_nan=True)).sum()
    )

    signal = ReplaceInvalidValuesAppliedColumnSignal(
        column=str(column),
        mode="percentage_bounds",
        mode_label=MODE_LABEL_MAP.get("percentage_bounds", "percentage_bounds"),
        replacements=changes,
        auto_detected=column in auto_columns,
        dtype=str(series.dtype),
        min_value=scale_min,
        max_value=scale_max,
    )

    return signal, None


def _process_column(
    working_frame: pd.DataFrame,
    column: str,
    mode: str,
    min_value: Optional[float],
    max_value: Optional[float],
    auto_columns: List[str],
) -> Tuple[Optional[ReplaceInvalidValuesAppliedColumnSignal], Optional[str]]:
    series = working_frame[column]

    if pd_types.is_bool_dtype(series):
        return None, column

    numeric_series = pd.to_numeric(series, errors="coerce")

    if mode == "percentage_bounds":
        return _process_percentage_column(
            working_frame,
            column,
            series,
            numeric_series,
            min_value,
            max_value,
            auto_columns,
        )

    numeric_valid = numeric_series.notna()
    mask = _build_replacement_mask(numeric_series, numeric_valid, mode, min_value, max_value)

    if mask is None:
        return None, column

    invalid_count = int(mask.sum())
    if invalid_count:
        replacement_token = _replacement_token(series)
        updated_series = series.copy()
        updated_series.loc[mask] = replacement_token
        working_frame[column] = updated_series

    signal = ReplaceInvalidValuesAppliedColumnSignal(
        column=str(column),
        mode=mode,
        mode_label=MODE_LABEL_MAP.get(mode, mode),
        replacements=invalid_count,
        auto_detected=column in auto_columns,
        dtype=str(series.dtype),
        min_value=min_value,
        max_value=max_value,
    )

    return signal, None


def _initialize_plan(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[Optional[_ReplacementPlan], List[str], ReplaceInvalidValuesNodeSignal]:
    node_dict = node if isinstance(node, dict) else {}
    node_id = node_dict.get("id") if node_dict else None
    data = node_dict.get("data") if node_dict else None
    config = (data or {}).get("config") or {}
    configured_columns = _coerce_string_list(config.get("columns"))
    signal = ReplaceInvalidValuesNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )

    if frame.empty:
        return None, configured_columns, signal

    target_columns, auto_columns, missing_columns = _resolve_columns(frame, configured_columns)

    mode = _normalize_mode(config.get("mode"))
    min_value = _coerce_float(config.get("min_value"))
    max_value = _coerce_float(config.get("max_value"))

    plan = _ReplacementPlan(
        target_columns=target_columns,
        auto_columns=auto_columns,
        missing_columns=missing_columns,
        mode=mode,
        min_value=min_value,
        max_value=max_value,
    )
    return plan, configured_columns, signal


def _execute_plan(frame: pd.DataFrame, plan: _ReplacementPlan) -> _ReplacementResult:
    working_frame = frame.copy()
    processed_columns = 0
    total_replacements = 0
    skipped_columns: List[str] = []
    processed_signals: List[ReplaceInvalidValuesAppliedColumnSignal] = []

    for column in plan.target_columns:
        signal_record, skipped = _process_column(
            working_frame,
            column,
            plan.mode,
            plan.min_value,
            plan.max_value,
            plan.auto_columns,
        )

        if skipped is not None:
            skipped_columns.append(skipped)
            continue

        if signal_record is None:
            continue

        processed_columns += 1
        total_replacements += signal_record.replacements
        processed_signals.append(signal_record)

    return _ReplacementResult(
        frame=working_frame,
        processed_columns=processed_columns,
        total_replacements=total_replacements,
        skipped_columns=skipped_columns,
        processed_signals=processed_signals,
    )


def _update_signal_with_plan(
    signal: ReplaceInvalidValuesNodeSignal,
    plan: _ReplacementPlan,
    result: _ReplacementResult,
) -> None:
    signal.auto_detected_columns = list(plan.auto_columns)
    signal.missing_columns = list(plan.missing_columns)
    signal.mode = plan.mode
    signal.min_value = plan.min_value
    signal.max_value = plan.max_value
    signal.skipped_columns = sorted(set(result.skipped_columns))
    signal.total_replacements = result.total_replacements
    signal.processed_columns = result.processed_signals


def _build_summary(
    plan: _ReplacementPlan,
    configured_columns: List[str],
    result: _ReplacementResult,
) -> str:
    summary_parts = [
        f"Replace invalid values: processed {result.processed_columns} column(s)"
    ]

    if result.total_replacements:
        summary_parts.append(f"flagged {result.total_replacements} value(s)")
    else:
        summary_parts.append("no changes detected")

    summary_parts.append(f"mode: {MODE_LABEL_MAP.get(plan.mode, plan.mode)}")

    if plan.mode == "percentage_bounds":
        clip_lower = 0.0 if plan.min_value is None else float(plan.min_value)
        clip_upper = 100.0 if plan.max_value is None else float(plan.max_value)
        summary_parts.append(
            f"scaled to percentage range [{clip_lower:.2f}, {clip_upper:.2f}]"
        )

    if plan.mode == "custom_range" and (plan.min_value is not None or plan.max_value is not None):
        lower_label = str(plan.min_value) if plan.min_value is not None else "-inf"
        upper_label = str(plan.max_value) if plan.max_value is not None else "inf"
        summary_parts.append(f"bounds [{lower_label}, {upper_label}]")

    if not configured_columns and plan.auto_columns:
        summary_parts.append("auto-detected numeric columns")

    if result.skipped_columns:
        summary_parts.append(
            f"skipped {len(set(result.skipped_columns))} unsupported column(s)"
        )

    if plan.missing_columns:
        summary_parts.append(f"{len(plan.missing_columns)} column(s) not found")

    return "; ".join(summary_parts)


def apply_replace_invalid_values(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, ReplaceInvalidValuesNodeSignal]:
    """Replace out-of-range numeric entries with missing values."""
    plan, configured_columns, signal = _initialize_plan(frame, node)

    if plan is None:
        return frame, "Replace invalid values: no data available", signal

    if not plan.target_columns:
        return frame, "Replace invalid values: no numeric columns detected", signal

    result = _execute_plan(frame, plan)
    _update_signal_with_plan(signal, plan, result)

    if result.processed_columns == 0:
        return frame, "Replace invalid values: no eligible numeric columns", signal

    summary = _build_summary(plan, configured_columns, result)
    return result.frame, summary, signal
