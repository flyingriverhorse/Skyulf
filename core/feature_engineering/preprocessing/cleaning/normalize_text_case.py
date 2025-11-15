"""Normalize text case for configured text columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from ...shared.utils import _auto_detect_text_columns, _coerce_string_list
from core.feature_engineering.schemas import (
    NormalizeTextCaseAppliedColumnSignal,
    NormalizeTextCaseNodeSignal,
)

MODE_LABEL_MAP = {
    "lower": "lowercase",
    "upper": "uppercase",
    "title": "title case",
    "sentence": "sentence case",
}


@dataclass
class _ProcessingResult:
    frame: pd.DataFrame
    processed_columns: List[str]
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


def _resolve_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or "lower").strip().lower()
    return mode if mode in MODE_LABEL_MAP else "lower"


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


def _is_textual_series(series: pd.Series) -> bool:
    return bool(
        pd_types.is_string_dtype(series)
        or pd_types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _normalize_series(series: pd.Series, mode: str) -> pd.Series:
    string_series = series.astype("string")
    if mode == "upper":
        return string_series.str.upper()
    if mode == "title":
        return string_series.str.title()
    if mode == "sentence":
        return string_series.map(_sentence_case).astype("string")
    return string_series.str.lower()


def _count_changes(original: pd.Series, normalized: pd.Series) -> int:
    before = original.astype("string").fillna("").to_numpy(dtype=object)
    after = normalized.fillna("").to_numpy(dtype=object)
    return int(np.not_equal(before, after).sum())


def _cast_normalized(normalized: pd.Series, original: pd.Series) -> pd.Series:
    if pd_types.is_object_dtype(original):
        return normalized.astype(object)
    if pd_types.is_string_dtype(original):
        return normalized.astype(original.dtype)
    return normalized.astype("string")


def _process_column(
    working_frame: pd.DataFrame,
    column: str,
    mode: str,
    total_rows: int,
) -> Tuple[bool, int, NormalizeTextCaseAppliedColumnSignal, bool]:
    if column not in working_frame.columns:
        empty_signal = NormalizeTextCaseAppliedColumnSignal(
            column=str(column),
            updated_cells=0,
            total_rows=total_rows,
            dtype="unknown",
        )
        return False, 0, empty_signal, True

    series = working_frame[column]
    if not _is_textual_series(series):
        return False, 0, NormalizeTextCaseAppliedColumnSignal(
            column=str(column),
            updated_cells=0,
            total_rows=total_rows,
            dtype=str(series.dtype),
        ), True

    normalized = _normalize_series(series, mode)
    changes = _count_changes(series, normalized)

    if changes:
        working_frame[column] = _cast_normalized(normalized, series)

    column_signal = NormalizeTextCaseAppliedColumnSignal(
        column=str(column),
        updated_cells=changes,
        total_rows=total_rows,
        dtype=str(series.dtype),
    )

    return True, changes, column_signal, False


def _initialize_signal(node: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], NormalizeTextCaseNodeSignal]:
    node_dict = node if isinstance(node, dict) else {}
    node_id = node_dict.get("id") if node_dict else None
    data = node_dict.get("data") if node_dict else None
    config = (data or {}).get("config") or {}
    configured_columns = _coerce_string_list(config.get("columns"))
    signal = NormalizeTextCaseNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )
    return config, list(configured_columns), signal


def _apply_to_columns(
    frame: pd.DataFrame,
    target_columns: List[str],
    mode: str,
    signal: NormalizeTextCaseNodeSignal,
) -> _ProcessingResult:
    working_frame = frame.copy()
    processed_columns: List[str] = []
    skipped_columns: List[str] = []
    updated_cells = 0
    total_rows = int(frame.shape[0])

    for column in target_columns:
        _, changes, column_signal, skipped = _process_column(
            working_frame, column, mode, total_rows
        )

        if skipped:
            skipped_columns.append(column)
            continue

        processed_columns.append(column)
        updated_cells += changes
        signal.applied_columns.append(column_signal)

    return _ProcessingResult(
        frame=working_frame,
        processed_columns=processed_columns,
        skipped_columns=skipped_columns,
        updated_cells=updated_cells,
    )


def _build_summary(
    result: _ProcessingResult,
    mode: str,
    configured_columns: List[str],
    auto_columns: List[str],
    missing_columns: List[str],
) -> str:
    summary_parts = [
        f"Normalize text case: processed {len(result.processed_columns)} column(s)",
    ]

    if result.updated_cells:
        summary_parts.append(f"updated {result.updated_cells} cell(s)")
    else:
        summary_parts.append("no changes detected")

    summary_parts.append(f"mode: {MODE_LABEL_MAP.get(mode, mode)}")

    if not configured_columns and auto_columns:
        summary_parts.append("auto-detected text columns")

    if result.skipped_columns:
        summary_parts.append(
            f"skipped {len(set(result.skipped_columns))} unsupported column(s)"
        )

    if missing_columns:
        summary_parts.append(f"{len(missing_columns)} column(s) not found")

    return "; ".join(summary_parts)


def apply_normalize_text_case(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, NormalizeTextCaseNodeSignal]:
    """Apply consistent casing across selected text columns."""
    config, configured_columns, signal = _initialize_signal(node)

    if frame.empty:
        return frame, "Normalize text case: no data available", signal

    target_columns, auto_columns, missing_columns = _resolve_columns(frame, configured_columns)
    signal.auto_detected_columns = list(auto_columns)
    signal.missing_columns = list(missing_columns)

    if not target_columns:
        return frame, "Normalize text case: no eligible text columns", signal

    mode = _resolve_mode(config.get("mode"))
    signal.mode = mode

    result = _apply_to_columns(frame, target_columns, mode, signal)
    skipped_unique = sorted(set(result.skipped_columns))
    signal.skipped_columns = skipped_unique

    if not result.processed_columns:
        return frame, "Normalize text case: no eligible text columns", signal

    signal.processed_columns = result.processed_columns
    signal.updated_cells = result.updated_cells

    summary = _build_summary(
        result,
        mode,
        configured_columns,
        auto_columns,
        missing_columns,
    )

    return result.frame, summary, signal
