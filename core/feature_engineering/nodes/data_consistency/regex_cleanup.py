"""Apply regex-based cleanup transformations to text columns."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from ..feature_eng.utils import (
    TWO_DIGIT_YEAR_PIVOT,
    _auto_detect_text_columns,
    _coerce_string_list,
)
from core.feature_engineering.schemas import (
    RegexCleanupAppliedColumnSignal,
    RegexCleanupNodeSignal,
)


def apply_regex_cleanup(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, RegexCleanupNodeSignal]:
    """Normalize text columns using regex-driven cleanup strategies."""
    data = node.get("data") or {}
    config = data.get("config") or {}

    node_id = node.get("id") if isinstance(node, dict) else None

    configured_columns = _coerce_string_list(config.get("columns"))
    signal = RegexCleanupNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )

    if frame.empty:
        return frame, "Regex replace: no data available", signal

    target_columns = [column for column in configured_columns if column in frame.columns]
    missing_columns = [column for column in configured_columns if column not in frame.columns]

    auto_columns: List[str] = []
    if not target_columns:
        auto_columns = _auto_detect_text_columns(frame)
        target_columns = auto_columns

    signal.auto_detected_columns = list(auto_columns)
    signal.missing_columns = list(missing_columns)

    if not target_columns:
        return frame, "Regex replace: no eligible text columns", signal

    mode = str(config.get("mode") or "normalize_slash_dates").strip().lower()
    pattern_text = config.get("pattern")
    replacement_text = config.get("replacement")
    replacement_value = str(replacement_text) if isinstance(replacement_text, str) else ""

    mode_label_map = {
        "normalize_slash_dates": "normalize dates",
        "collapse_whitespace": "collapse whitespace",
        "extract_digits": "extract digits",
        "custom": "custom pattern",
    }

    compiled_pattern: Optional[re.Pattern[str]] = None

    if mode == "custom":
        if not isinstance(pattern_text, str) or not pattern_text.strip():
            signal.mode = "custom"
            signal.pattern = pattern_text.strip() if isinstance(pattern_text, str) else None
            return frame, "Regex replace: custom pattern required", signal
        try:
            compiled_pattern = re.compile(pattern_text)
        except re.error as exc:
            signal.mode = "custom"
            signal.pattern = pattern_text
            return frame, f"Regex replace: invalid pattern ({exc})", signal
    elif mode not in {"normalize_slash_dates", "collapse_whitespace", "extract_digits"}:
        mode = "normalize_slash_dates"

    date_pattern = re.compile(r"\b(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})\b")

    def _normalize_date_text(text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            month = int(match.group(1))
            day = int(match.group(2))
            year_token = match.group(3)
            if len(year_token) == 2:
                year_value = int(year_token)
                year_value += 2000 if year_value < TWO_DIGIT_YEAR_PIVOT else 1900
            else:
                year_value = int(year_token)
            return f"{year_value:04d}-{month:02d}-{day:02d}"

        return date_pattern.sub(_replace, text)

    signal.mode = mode
    if mode == "custom" and isinstance(pattern_text, str):
        signal.pattern = pattern_text
    elif mode != "custom":
        signal.pattern = None

    working_frame = frame.copy()
    processed = 0
    updated_cells = 0
    skipped_columns: List[str] = []
    processed_records: List[RegexCleanupAppliedColumnSignal] = []

    for column in target_columns:
        series = working_frame[column]
        if not (
            pd_types.is_string_dtype(series)
            or pd_types.is_object_dtype(series)
            or pd_types.is_categorical_dtype(series)
        ):
            skipped_columns.append(column)
            continue

        string_series = series.astype("string")

        def _transform(entry: Any) -> Any:
            if entry is pd.NA or entry is None:
                return entry
            text = str(entry)
            if not text:
                return text
            if mode == "normalize_slash_dates":
                return _normalize_date_text(text)
            if mode == "collapse_whitespace":
                return re.sub(r"\s+", " ", text).strip()
            if mode == "extract_digits":
                return re.sub(r"\D+", "", text)
            if compiled_pattern is not None:
                return compiled_pattern.sub(replacement_value, text)
            return text

        transformed = string_series.map(_transform).astype("string")

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

        processed += 1
        processed_records.append(
            RegexCleanupAppliedColumnSignal(
                column=str(column),
                mode=mode,
                mode_label=mode_label_map.get(mode, mode),
                updated_cells=changes,
                auto_detected=column in auto_columns,
                dtype=str(series.dtype),
                pattern=str(pattern_text) if mode == "custom" and isinstance(pattern_text, str) else None,
            )
        )

    signal.processed_columns = processed_records
    signal.skipped_columns = sorted(set(skipped_columns))
    signal.total_updated_cells = updated_cells

    if processed == 0:
        return frame, "Regex replace: no eligible text columns", signal

    summary_parts = [f"Regex replace: processed {processed} column(s)"]
    if updated_cells:
        summary_parts.append(f"updated {updated_cells} value(s)")
    else:
        summary_parts.append("no changes detected")

    summary_parts.append(f"mode: {mode_label_map.get(mode, mode)}")

    if mode == "custom" and isinstance(pattern_text, str):
        summary_parts.append(f"pattern='{pattern_text}'")

    if not configured_columns and auto_columns:
        summary_parts.append("auto-detected text columns")

    if skipped_columns:
        summary_parts.append(f"skipped {len(skipped_columns)} unsupported column(s)")

    if missing_columns:
        summary_parts.append(f"{len(missing_columns)} column(s) not found")

    return working_frame, "; ".join(summary_parts), signal
