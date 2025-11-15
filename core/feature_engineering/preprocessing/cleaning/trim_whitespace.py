"""Trim leading and trailing whitespace for configured text columns."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from ...shared.utils import _auto_detect_text_columns, _coerce_string_list
from core.feature_engineering.schemas import (
    TrimWhitespaceAppliedColumnSignal,
    TrimWhitespaceNodeSignal,
)


def apply_trim_whitespace(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, TrimWhitespaceNodeSignal]:
    """Apply whitespace trimming to selected text columns."""
    node_id = node.get("id") if isinstance(node, dict) else None

    data = node.get("data") or {}
    config = data.get("config") or {}

    configured_columns = _coerce_string_list(config.get("columns"))
    signal = TrimWhitespaceNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )

    if frame.empty:
        return frame, "Trim whitespace: no data available", signal

    target_columns = [column for column in configured_columns if column in frame.columns]
    missing_columns = [column for column in configured_columns if column not in frame.columns]

    auto_columns: List[str] = []
    if not target_columns:
        auto_columns = _auto_detect_text_columns(frame)
        target_columns = auto_columns

    signal.auto_detected_columns = list(auto_columns)
    signal.missing_columns = list(missing_columns)

    if not target_columns:
        return frame, "Trim whitespace: no eligible text columns", signal

    mode = str(config.get("mode") or "both").strip().lower()
    if mode not in {"leading", "trailing", "both"}:
        mode = "both"
    signal.mode = mode

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
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            skipped_columns.append(column)
            continue

        string_series = series.astype("string")
        if mode == "leading":
            transformed = string_series.str.lstrip()
        elif mode == "trailing":
            transformed = string_series.str.rstrip()
        else:
            transformed = string_series.str.strip()

        before = string_series.fillna("").to_numpy(dtype=object)
        after = transformed.fillna("").to_numpy(dtype=object)
        changes = int(np.not_equal(before, after).sum())

        if changes:
            if pd_types.is_object_dtype(series):
                working_frame[column] = transformed.astype(object)
            elif pd_types.is_string_dtype(series):
                working_frame[column] = transformed.astype(series.dtype)
            else:
                working_frame[column] = transformed.astype("string")
            updated_cells += changes

        processed_columns.append(column)
        signal.applied_columns.append(
            TrimWhitespaceAppliedColumnSignal(
                column=str(column),
                updated_cells=changes,
                total_rows=total_rows,
                dtype=str(series.dtype),
            )
        )

    if not processed_columns:
        signal.skipped_columns = sorted(set(skipped_columns))
        return frame, "Trim whitespace: no eligible text columns", signal

    signal.processed_columns = processed_columns
    signal.skipped_columns = sorted(set(skipped_columns))
    signal.updated_cells = updated_cells

    summary_parts = [f"Trim whitespace: processed {len(processed_columns)} column(s)"]
    if updated_cells:
        summary_parts.append(f"updated {updated_cells} cell(s)")
    else:
        summary_parts.append("no changes detected")

    if not configured_columns and auto_columns:
        summary_parts.append("auto-detected text columns")

    if skipped_columns:
        summary_parts.append(f"skipped {len(skipped_columns)} unsupported column(s)")

    if missing_columns:
        summary_parts.append(f"{len(missing_columns)} column(s) not found")

    return working_frame, "; ".join(summary_parts), signal
