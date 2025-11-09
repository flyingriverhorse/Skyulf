"""Apply regex-based cleanup transformations to text columns."""

from __future__ import annotations

import re
from dataclasses import dataclass
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

MODE_LABEL_MAP = {
    "normalize_slash_dates": "normalize dates",
    "collapse_whitespace": "collapse whitespace",
    "extract_digits": "extract digits",
    "custom": "custom pattern",
}

DATE_PATTERN = re.compile(r"\b(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})\b")


@dataclass
class _RegexProcessingResult:
    frame: pd.DataFrame
    processed_columns: List[RegexCleanupAppliedColumnSignal]
    skipped_columns: List[str]
    updated_cells: int


def _resolve_columns(frame: pd.DataFrame, configured_columns: List[str]) -> Tuple[List[str], List[str], List[str]]:
    target_columns = [column for column in configured_columns if column in frame.columns]
    missing_columns = [column for column in configured_columns if column not in frame.columns]

    auto_columns: List[str] = []
    if not target_columns:
        auto_columns = _auto_detect_text_columns(frame)
        target_columns = auto_columns

    return list(target_columns), list(auto_columns), list(missing_columns)


def _normalize_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or "normalize_slash_dates").strip().lower()
    if mode in MODE_LABEL_MAP:
        return mode
    return "normalize_slash_dates"


def _compile_custom_pattern(pattern_text: Any) -> Tuple[Optional[re.Pattern[str]], Optional[str], Optional[str]]:
    if not isinstance(pattern_text, str) or not pattern_text.strip():
        cleaned = pattern_text.strip() if isinstance(pattern_text, str) else None
        return None, "Regex replace: custom pattern required", cleaned
    try:
        return re.compile(pattern_text), None, pattern_text
    except re.error as exc:
        return None, f"Regex replace: invalid pattern ({exc})", pattern_text


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

    return DATE_PATTERN.sub(_replace, text)


def _transform_entry(
    entry: Any,
    mode: str,
    compiled_pattern: Optional[re.Pattern[str]],
    replacement_value: str,
) -> Any:
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


def _transform_series(
    series: pd.Series,
    mode: str,
    compiled_pattern: Optional[re.Pattern[str]],
    replacement_value: str,
) -> Tuple[pd.Series, pd.Series]:
    string_series = series.astype("string")
    transformed = string_series.map(
        lambda entry: _transform_entry(entry, mode, compiled_pattern, replacement_value)
    ).astype("string")
    return string_series, transformed


def _is_textual_series(series: pd.Series) -> bool:
    return bool(
        pd_types.is_string_dtype(series)
        or pd_types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _count_changes(original: pd.Series, transformed: pd.Series) -> int:
    before = original.fillna("").to_numpy(dtype=object)
    after = transformed.fillna("").to_numpy(dtype=object)
    return int(np.not_equal(before, after).sum())


def _cast_transformed(transformed: pd.Series, original: pd.Series) -> pd.Series:
    if pd_types.is_object_dtype(original):
        return transformed.astype(object)
    if pd_types.is_string_dtype(original):
        return transformed.astype(original.dtype)
    return transformed.astype("string")


def _process_column(
    working_frame: pd.DataFrame,
    column: str,
    mode: str,
    compiled_pattern: Optional[re.Pattern[str]],
    replacement_value: str,
    auto_columns: List[str],
    pattern_text: Optional[str],
) -> Tuple[Optional[RegexCleanupAppliedColumnSignal], Optional[str]]:
    series = working_frame[column]
    if not _is_textual_series(series):
        return None, column

    original, transformed = _transform_series(series, mode, compiled_pattern, replacement_value)
    changes = _count_changes(original, transformed)

    if changes:
        working_frame[column] = _cast_transformed(transformed, series)

    signal = RegexCleanupAppliedColumnSignal(
        column=str(column),
        mode=mode,
        mode_label=MODE_LABEL_MAP.get(mode, mode),
        updated_cells=changes,
        auto_detected=column in auto_columns,
        dtype=str(series.dtype),
        pattern=pattern_text if mode == "custom" else None,
    )

    return signal, None


def _initialize_signal(node: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], RegexCleanupNodeSignal]:
    node_dict = node if isinstance(node, dict) else {}
    data = node_dict.get("data") if node_dict else None
    config = (data or {}).get("config") or {}
    configured_columns = _coerce_string_list(config.get("columns"))
    node_id = node_dict.get("id") if node_dict else None
    signal = RegexCleanupNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )
    return config, list(configured_columns), signal


def _resolve_mode_and_pattern(
    config: Dict[str, Any],
    signal: RegexCleanupNodeSignal,
) -> Tuple[str, Optional[re.Pattern[str]], Optional[str], str, Optional[str]]:
    mode = _normalize_mode(config.get("mode"))
    pattern_text = config.get("pattern")
    replacement_text = config.get("replacement")
    replacement_value = str(replacement_text) if isinstance(replacement_text, str) else ""

    compiled_pattern: Optional[re.Pattern[str]] = None
    cleaned_pattern_text: Optional[str] = None
    error_message: Optional[str] = None

    if mode == "custom":
        compiled_pattern, error_message, cleaned_pattern_text = _compile_custom_pattern(pattern_text)
        signal.pattern = cleaned_pattern_text
        signal.mode = "custom"
        if error_message:
            return mode, compiled_pattern, cleaned_pattern_text, replacement_value, error_message
    else:
        signal.pattern = None

    signal.mode = mode
    return mode, compiled_pattern, cleaned_pattern_text, replacement_value, None


def _process_target_columns(
    frame: pd.DataFrame,
    target_columns: List[str],
    mode: str,
    compiled_pattern: Optional[re.Pattern[str]],
    replacement_value: str,
    auto_columns: List[str],
    cleaned_pattern_text: Optional[str],
    signal: RegexCleanupNodeSignal,
) -> _RegexProcessingResult:
    working_frame = frame.copy()
    skipped_columns: List[str] = []
    processed_records: List[RegexCleanupAppliedColumnSignal] = []
    updated_cells = 0

    for column in target_columns:
        column_signal, skipped = _process_column(
            working_frame,
            column,
            mode,
            compiled_pattern,
            replacement_value,
            auto_columns,
            cleaned_pattern_text,
        )

        if skipped is not None:
            skipped_columns.append(skipped)
            continue

        if column_signal is None:
            continue

        processed_records.append(column_signal)
        updated_cells += column_signal.updated_cells

    signal.processed_columns = processed_records
    return _RegexProcessingResult(
        frame=working_frame,
        processed_columns=processed_records,
        skipped_columns=skipped_columns,
        updated_cells=updated_cells,
    )


def _build_summary(
    result: _RegexProcessingResult,
    mode: str,
    configured_columns: List[str],
    auto_columns: List[str],
    missing_columns: List[str],
    cleaned_pattern_text: Optional[str],
) -> str:
    processed_count = len(result.processed_columns)
    summary_parts = [f"Regex replace: processed {processed_count} column(s)"]

    if result.updated_cells:
        summary_parts.append(f"updated {result.updated_cells} value(s)")
    else:
        summary_parts.append("no changes detected")

    summary_parts.append(f"mode: {MODE_LABEL_MAP.get(mode, mode)}")

    if mode == "custom" and cleaned_pattern_text:
        summary_parts.append(f"pattern='{cleaned_pattern_text}'")

    if not configured_columns and auto_columns:
        summary_parts.append("auto-detected text columns")

    if result.skipped_columns:
        summary_parts.append(
            f"skipped {len(set(result.skipped_columns))} unsupported column(s)"
        )

    if missing_columns:
        summary_parts.append(f"{len(missing_columns)} column(s) not found")

    return "; ".join(summary_parts)


def apply_regex_cleanup(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, RegexCleanupNodeSignal]:
    """Normalize text columns using regex-driven cleanup strategies."""
    config, configured_columns, signal = _initialize_signal(node)

    if frame.empty:
        return frame, "Regex replace: no data available", signal

    target_columns, auto_columns, missing_columns = _resolve_columns(frame, configured_columns)
    signal.auto_detected_columns = list(auto_columns)
    signal.missing_columns = list(missing_columns)

    if not target_columns:
        return frame, "Regex replace: no eligible text columns", signal

    mode, compiled_pattern, cleaned_pattern_text, replacement_value, error_message = _resolve_mode_and_pattern(
        config, signal
    )

    if error_message:
        return frame, error_message, signal

    result = _process_target_columns(
        frame,
        target_columns,
        mode,
        compiled_pattern,
        replacement_value,
        auto_columns,
        cleaned_pattern_text if mode == "custom" else None,
        signal,
    )

    signal.skipped_columns = sorted(set(result.skipped_columns))
    signal.total_updated_cells = result.updated_cells

    if not result.processed_columns:
        return frame, "Regex replace: no eligible text columns", signal

    summary = _build_summary(
        result,
        mode,
        configured_columns,
        auto_columns,
        missing_columns,
        cleaned_pattern_text if mode == "custom" else None,
    )

    return result.frame, summary, signal
