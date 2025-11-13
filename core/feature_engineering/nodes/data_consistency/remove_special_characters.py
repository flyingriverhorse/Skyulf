"""Remove special characters from configured text columns."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from ...shared.utils import _auto_detect_text_columns, _coerce_string_list
from core.feature_engineering.schemas import (
    RemoveSpecialCharactersAppliedColumnSignal,
    RemoveSpecialCharactersNodeSignal,
)


def apply_remove_special_characters(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, RemoveSpecialCharactersNodeSignal]:
    """Clean special characters out of selected text columns."""
    node_id = node.get("id") if isinstance(node, dict) else None

    data = node.get("data") or {}
    config = data.get("config") or {}

    configured_columns = _coerce_string_list(config.get("columns"))
    signal = RemoveSpecialCharactersNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )

    if frame.empty:
        return frame, "Remove special characters: no data available", signal

    configured_columns = _coerce_string_list(config.get("columns"))
    target_columns = [column for column in configured_columns if column in frame.columns]
    missing_columns = [column for column in configured_columns if column not in frame.columns]

    auto_columns: List[str] = []
    if not target_columns:
        auto_columns = _auto_detect_text_columns(frame)
        target_columns = auto_columns

    signal.auto_detected_columns = list(auto_columns)
    signal.missing_columns = list(missing_columns)

    if not target_columns:
        return frame, "Remove special characters: no eligible text columns", signal

    mode = str(config.get("mode") or "keep_alphanumeric").strip().lower()
    replacement_value = str(config.get("replacement") or "")

    pattern_map = {
        "keep_alphanumeric": re.compile(r"[^0-9A-Za-z]+"),
        "keep_alphanumeric_space": re.compile(r"[^0-9A-Za-z\s]+"),
        "letters_only": re.compile(r"[^A-Za-z]+"),
        "digits_only": re.compile(r"[^0-9]+"),
    }

    if mode not in pattern_map:
        mode = "keep_alphanumeric"

    collapse_whitespace = mode in {"keep_alphanumeric_space"} or replacement_value == " "
    pattern = pattern_map[mode]

    mode_label_map = {
        "keep_alphanumeric": "alphanumeric only",
        "keep_alphanumeric_space": "alphanumeric + space",
        "letters_only": "letters only",
        "digits_only": "digits only",
    }

    signal.mode = mode
    signal.replacement = replacement_value if replacement_value else None

    working_frame = frame.copy()
    processed = 0
    updated_cells = 0
    skipped_columns: List[str] = []
    processed_signals: List[RemoveSpecialCharactersAppliedColumnSignal] = []
    total_rows = int(frame.shape[0])

    for column in target_columns:
        series = working_frame[column]
        if not (
            pd_types.is_string_dtype(series)
            or pd_types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            skipped_columns.append(column)
            continue

        string_series = series.astype("string")

        def _clean(entry: Any) -> Any:
            if entry is pd.NA or entry is None:
                return entry
            text = str(entry)
            cleaned = pattern.sub(replacement_value, text)
            if collapse_whitespace:
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned

        cleaned_series = string_series.map(_clean).astype("string")

        before = string_series.fillna("").to_numpy(dtype=object)
        after = cleaned_series.fillna("").to_numpy(dtype=object)
        changes = int(np.not_equal(before, after).sum())

        if changes:
            if pd_types.is_object_dtype(series):
                working_frame[column] = cleaned_series.astype(object)
            elif pd_types.is_string_dtype(series):
                working_frame[column] = cleaned_series.astype(series.dtype)
            else:
                working_frame[column] = cleaned_series.astype("string")
            updated_cells += changes

        processed += 1
        processed_signals.append(
            RemoveSpecialCharactersAppliedColumnSignal(
                column=str(column),
                mode=mode,
                mode_label=mode_label_map.get(mode, mode),
                updated_cells=changes,
                replacement=replacement_value if replacement_value else None,
                auto_detected=column in auto_columns,
                dtype=str(series.dtype),
                total_rows=total_rows,
            )
        )

    if processed == 0:
        signal.skipped_columns = sorted(set(skipped_columns))
        signal.processed_columns = processed_signals
        return frame, "Remove special characters: no eligible text columns", signal

    summary_parts = [f"Remove special characters: processed {processed} column(s)"]
    if updated_cells:
        summary_parts.append(f"updated {updated_cells} value(s)")
    else:
        summary_parts.append("no changes detected")

    summary_parts.append(f"mode: {mode_label_map.get(mode, mode)}")

    if replacement_value:
        summary_parts.append(f"replacement='{replacement_value}'")

    if not configured_columns and auto_columns:
        summary_parts.append("auto-detected text columns")

    if skipped_columns:
        summary_parts.append(f"skipped {len(skipped_columns)} unsupported column(s)")

    if missing_columns:
        summary_parts.append(f"{len(missing_columns)} column(s) not found")

    signal.skipped_columns = sorted(set(skipped_columns))
    signal.total_updated_cells = updated_cells
    signal.processed_columns = processed_signals

    return working_frame, "; ".join(summary_parts), signal
