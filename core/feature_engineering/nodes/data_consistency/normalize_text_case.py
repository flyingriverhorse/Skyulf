"""Normalize text case for configured text columns."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from ..feature_eng.utils import _auto_detect_text_columns, _coerce_string_list
from core.feature_engineering.schemas import (
    NormalizeTextCaseAppliedColumnSignal,
    NormalizeTextCaseNodeSignal,
)


def apply_normalize_text_case(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, NormalizeTextCaseNodeSignal]:
    """Apply consistent casing across selected text columns."""
    node_id = node.get("id") if isinstance(node, dict) else None

    data = node.get("data") or {}
    config = data.get("config") or {}

    configured_columns = _coerce_string_list(config.get("columns"))
    signal = NormalizeTextCaseNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )

    if frame.empty:
        return frame, "Normalize text case: no data available", signal

    target_columns = [column for column in configured_columns if column in frame.columns]
    missing_columns = [column for column in configured_columns if column not in frame.columns]

    auto_columns: List[str] = []
    if not target_columns:
        auto_columns = _auto_detect_text_columns(frame)
        target_columns = auto_columns

    signal.auto_detected_columns = list(auto_columns)
    signal.missing_columns = list(missing_columns)

    if not target_columns:
        return frame, "Normalize text case: no eligible text columns", signal

    mode = str(config.get("mode") or "lower").strip().lower()
    if mode not in {"lower", "upper", "title", "sentence"}:
        mode = "lower"
    signal.mode = mode

    def _sentence_case(value: Any) -> Any:
        if value is pd.NA or value is None:
            return value
        text = str(value)
        if not text:
            return text
        leading_len = len(text) - len(text.lstrip())
        leading = text[:leading_len]
        remainder = text[leading_len:]
        if not remainder:
            return text
        return leading + remainder[0].upper() + remainder[1:].lower()

    mode_label_map = {
        "lower": "lowercase",
        "upper": "uppercase",
        "title": "title case",
        "sentence": "sentence case",
    }

    working_frame = frame.copy()
    processed_columns: List[str] = []
    updated_cells = 0
    skipped_columns: List[str] = []
    total_rows = int(frame.shape[0])

    for column in target_columns:
        if column not in working_frame.columns:
            continue

        series = working_frame[column]
        if not (
            pd_types.is_string_dtype(series)
            or pd_types.is_object_dtype(series)
            or pd_types.is_categorical_dtype(series)
        ):
            skipped_columns.append(column)
            continue

        string_series = series.astype("string")
        if mode == "upper":
            normalized = string_series.str.upper()
        elif mode == "title":
            normalized = string_series.str.title()
        elif mode == "sentence":
            normalized = string_series.map(_sentence_case).astype("string")
        else:
            normalized = string_series.str.lower()

        before = string_series.fillna("").to_numpy(dtype=object)
        after = normalized.fillna("").to_numpy(dtype=object)
        changes = int(np.not_equal(before, after).sum())

        if changes:
            if pd_types.is_object_dtype(series):
                working_frame[column] = normalized.astype(object)
            elif pd_types.is_string_dtype(series):
                working_frame[column] = normalized.astype(series.dtype)
            else:
                working_frame[column] = normalized.astype("string")
            updated_cells += changes

        processed_columns.append(column)
        signal.applied_columns.append(
            NormalizeTextCaseAppliedColumnSignal(
                column=str(column),
                updated_cells=changes,
                total_rows=total_rows,
                dtype=str(series.dtype),
            )
        )

    if not processed_columns:
        signal.skipped_columns = sorted(set(skipped_columns))
        return frame, "Normalize text case: no eligible text columns", signal

    signal.processed_columns = processed_columns
    signal.skipped_columns = sorted(set(skipped_columns))
    signal.updated_cells = updated_cells

    summary_parts = [f"Normalize text case: processed {len(processed_columns)} column(s)"]
    if updated_cells:
        summary_parts.append(f"updated {updated_cells} cell(s)")
    else:
        summary_parts.append("no changes detected")

    summary_parts.append(f"mode: {mode_label_map.get(mode, mode)}")

    if not configured_columns and auto_columns:
        summary_parts.append("auto-detected text columns")

    if skipped_columns:
        summary_parts.append(f"skipped {len(skipped_columns)} unsupported column(s)")

    if missing_columns:
        summary_parts.append(f"{len(missing_columns)} column(s) not found")

    return working_frame, "; ".join(summary_parts), signal
