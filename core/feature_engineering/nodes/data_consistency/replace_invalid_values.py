"""Replace invalid numeric values based on configurable rules."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from ..feature_eng.utils import _auto_detect_numeric_columns, _coerce_string_list
from core.feature_engineering.schemas import (
    ReplaceInvalidValuesAppliedColumnSignal,
    ReplaceInvalidValuesNodeSignal,
)


def apply_replace_invalid_values(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, ReplaceInvalidValuesNodeSignal]:
    """Replace out-of-range numeric entries with missing values."""
    data = node.get("data") or {}
    config = data.get("config") or {}

    node_id = node.get("id") if isinstance(node, dict) else None

    configured_columns = _coerce_string_list(config.get("columns"))
    signal = ReplaceInvalidValuesNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )

    if frame.empty:
        return frame, "Replace invalid values: no data available", signal

    target_columns = [column for column in configured_columns if column in frame.columns]
    missing_columns = [column for column in configured_columns if column not in frame.columns]

    auto_columns: List[str] = []
    if not target_columns:
        auto_columns = _auto_detect_numeric_columns(frame)
        target_columns = auto_columns

    signal.auto_detected_columns = list(auto_columns)
    signal.missing_columns = list(missing_columns)

    if not target_columns:
        return frame, "Replace invalid values: no numeric columns detected", signal

    mode = str(config.get("mode") or "negative_to_nan").strip().lower()
    if mode not in {
        "negative_to_nan",
        "zero_to_nan",
        "percentage_bounds",
        "age_bounds",
        "custom_range",
    }:
        mode = "negative_to_nan"
    signal.mode = mode

    def _coerce_float(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    min_value = _coerce_float(config.get("min_value"))
    max_value = _coerce_float(config.get("max_value"))
    signal.min_value = min_value
    signal.max_value = max_value

    working_frame = frame.copy()
    processed = 0
    total_replacements = 0
    skipped_columns: List[str] = []
    processed_signals: List[ReplaceInvalidValuesAppliedColumnSignal] = []

    mode_label_map = {
        "negative_to_nan": "negative to missing",
        "zero_to_nan": "zero to missing",
        "percentage_bounds": "percentage bounds",
        "age_bounds": "age bounds",
        "custom_range": "custom bounds",
    }

    for column in target_columns:
        series = working_frame[column]
        if pd_types.is_bool_dtype(series):
            skipped_columns.append(column)
            continue

        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_valid = numeric_series.notna()
        mask = pd.Series(False, index=series.index)

        if mode == "negative_to_nan":
            threshold = min_value if min_value is not None else 0.0
            lower_mask = numeric_valid & (numeric_series < threshold)
            upper_mask = (
                pd.Series(False, index=series.index)
                if max_value is None
                else numeric_valid & (numeric_series > max_value)
            )
            mask = lower_mask | upper_mask
        elif mode == "zero_to_nan":
            mask = numeric_valid & numeric_series.abs().le(1e-12)
        elif mode == "percentage_bounds":
            lower = min_value if min_value is not None else 0.0
            upper = max_value if max_value is not None else 100.0
            mask = numeric_valid & ((numeric_series < lower) | (numeric_series > upper))
        elif mode == "age_bounds":
            lower = min_value if min_value is not None else 0.0
            upper = max_value if max_value is not None else 120.0
            mask = numeric_valid & ((numeric_series < lower) | (numeric_series > upper))
        else:
            if min_value is None and max_value is None:
                skipped_columns.append(column)
                continue
            if min_value is not None:
                mask = mask | (numeric_valid & (numeric_series < min_value))
            if max_value is not None:
                mask = mask | (numeric_valid & (numeric_series > max_value))

        invalid_count = int(mask.sum())

        replacement_token: Any
        if pd_types.is_integer_dtype(series) and not pd_types.is_bool_dtype(series):
            replacement_token = pd.NA
        else:
            replacement_token = np.nan

        if invalid_count:
            updated_series = series.copy()
            updated_series.loc[mask] = replacement_token
            working_frame[column] = updated_series
            total_replacements += invalid_count

        processed += 1
        processed_signals.append(
            ReplaceInvalidValuesAppliedColumnSignal(
                column=str(column),
                mode=mode,
                mode_label=mode_label_map.get(mode, mode),
                replacements=invalid_count,
                auto_detected=column in auto_columns,
                dtype=str(series.dtype),
                min_value=min_value,
                max_value=max_value,
            )
        )

    signal.skipped_columns = sorted(set(skipped_columns))
    signal.total_replacements = total_replacements
    signal.processed_columns = processed_signals

    if processed == 0:
        return frame, "Replace invalid values: no eligible numeric columns", signal

    summary_parts = [f"Replace invalid values: processed {processed} column(s)"]
    if total_replacements:
        summary_parts.append(f"flagged {total_replacements} value(s)")
    else:
        summary_parts.append("no changes detected")

    summary_parts.append(f"mode: {mode_label_map.get(mode, mode)}")

    if mode == "custom_range" and (min_value is not None or max_value is not None):
        lower_label = str(min_value) if min_value is not None else "-inf"
        upper_label = str(max_value) if max_value is not None else "inf"
        summary_parts.append(f"bounds [{lower_label}, {upper_label}]")

    if not configured_columns and auto_columns:
        summary_parts.append("auto-detected numeric columns")

    if skipped_columns:
        summary_parts.append(f"skipped {len(skipped_columns)} unsupported column(s)")

    if missing_columns:
        summary_parts.append(f"{len(missing_columns)} column(s) not found")

    return working_frame, "; ".join(summary_parts), signal
