"""Standardize date formatting for data consistency nodes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from ...shared.utils import (
    _auto_detect_datetime_columns,
    _auto_detect_text_columns,
    _coerce_string_list,
)
from core.feature_engineering.schemas import (
    StandardizeDatesAppliedColumnSignal,
    StandardizeDatesNodeSignal,
)


MODE_FORMAT_MAP = {
    "iso_date": "%Y-%m-%d",
    "iso_datetime": "%Y-%m-%d %H:%M:%S",
    "month_day_year": "%m/%d/%Y",
    "day_month_year": "%d/%m/%Y",
}

MODE_LABEL_MAP = {
    "iso_date": "ISO date",
    "iso_datetime": "ISO datetime",
    "month_day_year": "MM/DD/YYYY",
    "day_month_year": "DD/MM/YYYY",
}

DAYFIRST_MODES = {"day_month_year", "day_first"}
AUTO_DETECT_KEYS = ("auto_detect", "autoDetect", "auto", "auto_detect_columns")


class DateStandardizationStats:
    def __init__(self) -> None:
        self.processed_columns = 0
        self.converted_values = 0
        self.parse_failures = 0
        self.mode_counts: Counter[str] = Counter()
        self.auto_detected_columns: Set[str] = set()
        self.missing_columns: Set[str] = set()
        self.skipped_columns: Set[str] = set()
        self.processed_signals: List[StandardizeDatesAppliedColumnSignal] = []

    def register_missing(self, column: str) -> None:
        self.missing_columns.add(column)

    def register_skipped(self, column: str) -> None:
        self.skipped_columns.add(column)

    def register_failures(self, failures: int) -> None:
        if failures:
            self.parse_failures += failures

    def record_processed(
        self,
        column: str,
        mode_value: str,
        changes: int,
        failures: int,
        auto_detected: bool,
        dtype: str,
    ) -> None:
        self.processed_columns += 1
        self.converted_values += changes
        self.parse_failures += failures
        if mode_value:
            self.mode_counts[mode_value] += 1
        if auto_detected:
            self.auto_detected_columns.add(column)
        self.processed_signals.append(
            StandardizeDatesAppliedColumnSignal(
                column=str(column),
                mode=mode_value,
                mode_label=MODE_LABEL_MAP.get(mode_value, mode_value),
                converted_values=changes,
                parse_failures=failures,
                auto_detected=auto_detected,
                dtype=dtype,
            )
        )

    def apply_to_signal(self, signal: StandardizeDatesNodeSignal) -> None:
        signal.auto_detected_columns = sorted(self.auto_detected_columns)
        signal.missing_columns = sorted(self.missing_columns)
        signal.skipped_columns = sorted(self.skipped_columns)
        signal.total_converted_values = self.converted_values
        signal.total_parse_failures = self.parse_failures
        signal.mode_counts = dict(self.mode_counts)
        signal.processed_columns = self.processed_signals


@dataclass
class _DatePlan:
    strategies: List[Dict[str, Any]]
    target_columns: List[str]
    auto_columns: List[str]
    missing_columns: List[str]
    default_mode: str


def _resolve_mode_value(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in MODE_FORMAT_MAP:
        return normalized
    return "iso_date"


def _normalize_strategies(config: Dict[str, Any], default_mode: str) -> List[Dict[str, Any]]:
    raw_strategies = config.get("format_strategies")
    if not isinstance(raw_strategies, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for entry in raw_strategies:
        if not isinstance(entry, dict):
            continue
        mode_value = _resolve_mode_value(entry.get("mode") or config.get("mode") or default_mode)
        columns = _coerce_string_list(entry.get("columns"))
        auto_detect = any(entry.get(key) for key in AUTO_DETECT_KEYS)
        if not columns and not auto_detect:
            continue
        normalized.append(
            {
                "mode": mode_value,
                "columns": columns,
                "auto_detect": bool(auto_detect),
            }
        )
    return normalized


def _detect_auto_columns(frame: pd.DataFrame) -> List[str]:
    detected = _auto_detect_datetime_columns(frame)
    if not detected:
        detected = _auto_detect_text_columns(frame)
    return [column for column in detected if column in frame.columns]


def _format_column(
    working_frame: pd.DataFrame,
    column: str,
    target_format: str,
    dayfirst: bool,
) -> Tuple[bool, int, int, bool]:
    series = working_frame[column]
    candidate_datetime: Optional[pd.Series] = None

    if pd_types.is_datetime64_any_dtype(series):
        candidate_datetime = pd.to_datetime(series, errors="coerce")
    elif (
        pd_types.is_string_dtype(series)
        or pd_types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        candidate_datetime = pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=dayfirst,
        )
    else:
        return False, 0, 0, True

    if candidate_datetime is None:
        return False, 0, 0, True

    try:
        candidate_datetime = candidate_datetime.dt.tz_localize(None)  # type: ignore[call-arg]
    except (AttributeError, TypeError, ValueError):
        pass

    mask = candidate_datetime.notna()
    original_notna = series.notna()
    failures = int((original_notna & ~mask).sum())

    if not mask.any():
        return False, 0, failures, True

    formatted_values = candidate_datetime.dt.strftime(target_format)
    string_series = series.astype("string")
    formatted_series = string_series.copy()
    formatted_series.loc[mask] = formatted_values.loc[mask].astype("string")

    before = string_series.fillna("").to_numpy(dtype=object)
    after = formatted_series.fillna("").to_numpy(dtype=object)
    changes = int(np.not_equal(before, after).sum())

    if changes:
        working_frame[column] = formatted_series

    return True, changes, failures, False


def _resolve_target_columns(
    frame: pd.DataFrame,
    configured_columns: Iterable[str],
) -> Tuple[List[str], List[str], List[str]]:
    configured = list(configured_columns)
    target_columns = [column for column in configured if column in frame.columns]
    missing_columns = [column for column in configured if column not in frame.columns]

    auto_columns: List[str] = []
    if not target_columns:
        auto_columns = _detect_auto_columns(frame)
        target_columns = auto_columns

    return target_columns, auto_columns, missing_columns


def _build_candidate_columns(
    strategy: Dict[str, Any],
    frame: pd.DataFrame,
    processed_columns: Set[str],
    remaining_auto_columns: List[str],
    stats: DateStandardizationStats,
) -> Tuple[List[str], Set[str], List[str]]:
    manual_columns = list(strategy.get("columns", []))
    auto_detect_flag = bool(strategy.get("auto_detect"))

    for column in manual_columns:
        if column not in frame.columns:
            stats.register_missing(column)

    existing_manual = [
        column
        for column in manual_columns
        if column in frame.columns and column not in processed_columns
    ]

    filtered_auto = [
        column
        for column in remaining_auto_columns
        if column not in processed_columns and column not in existing_manual
    ]

    candidate_columns = existing_manual[:]
    added_auto: List[str] = []

    if auto_detect_flag:
        for column in filtered_auto:
            if column in candidate_columns:
                continue
            candidate_columns.append(column)
            added_auto.append(column)

        filtered_auto = [column for column in filtered_auto if column not in added_auto]

    return candidate_columns, set(added_auto), filtered_auto


def _apply_strategy_column(
    column: str,
    mode_value: str,
    target_format: str,
    dayfirst: bool,
    frame: pd.DataFrame,
    working_frame: pd.DataFrame,
    stats: DateStandardizationStats,
    auto_added: Set[str],
) -> None:
    processed, changes, failures, skipped = _format_column(
        working_frame,
        column,
        target_format,
        dayfirst,
    )

    if not processed:
        stats.register_failures(failures)
        if skipped:
            stats.register_skipped(column)
        return

    stats.record_processed(
        column=column,
        mode_value=mode_value,
        changes=changes,
        failures=failures,
        auto_detected=column in auto_added,
        dtype=str(frame[column].dtype),
    )


def _apply_strategies(
    frame: pd.DataFrame,
    strategies: List[Dict[str, Any]],
    signal: StandardizeDatesNodeSignal,
) -> Tuple[pd.DataFrame, str]:
    working_frame = frame.copy()
    stats = DateStandardizationStats()

    available_auto_columns = _detect_auto_columns(frame)
    remaining_auto_columns = [column for column in available_auto_columns]
    processed_columns: Set[str] = set()

    for strategy in strategies:
        mode_value = _resolve_mode_value(strategy.get("mode"))
        target_format = MODE_FORMAT_MAP.get(mode_value, "%Y-%m-%d")
        dayfirst = mode_value in DAYFIRST_MODES

        candidate_columns, added_auto, remaining_auto_columns = _build_candidate_columns(
            strategy,
            frame,
            processed_columns,
            remaining_auto_columns,
            stats,
        )

        for column in candidate_columns:
            if column in processed_columns:
                continue

            _apply_strategy_column(
                column=column,
                mode_value=mode_value,
                target_format=target_format,
                dayfirst=dayfirst,
                frame=frame,
                working_frame=working_frame,
                stats=stats,
                auto_added=added_auto,
            )
            processed_columns.add(column)

    stats.apply_to_signal(signal)

    if stats.processed_columns == 0:
        summary_bits = ["Standardize dates: no convertible values"]
        if stats.missing_columns:
            summary_bits.append(f"{len(stats.missing_columns)} column(s) not found")
        if stats.skipped_columns:
            summary_bits.append(f"skipped {len(stats.skipped_columns)} unsupported column(s)")
        if stats.parse_failures:
            summary_bits.append(f"{stats.parse_failures} value(s) could not be parsed")
        if stats.auto_detected_columns:
            summary_bits.append(
                f"auto-detected {len(stats.auto_detected_columns)} column(s)"
            )
        return frame, "; ".join(summary_bits)

    summary_parts = [
        f"Standardize dates: processed {stats.processed_columns} column(s)"
    ]
    if stats.converted_values:
        summary_parts.append(f"formatted {stats.converted_values} value(s)")
    else:
        summary_parts.append("no changes detected")

    if stats.mode_counts:
        mode_details = ", ".join(
            f"{MODE_LABEL_MAP.get(mode, mode)} ({count})"
            for mode, count in sorted(stats.mode_counts.items())
        )
        summary_parts.append(f"formats applied: {mode_details}")

    if stats.auto_detected_columns:
        summary_parts.append(
            f"auto-detected {len(stats.auto_detected_columns)} column(s)"
        )

    if stats.parse_failures:
        summary_parts.append(f"{stats.parse_failures} value(s) could not be parsed")

    if stats.skipped_columns:
        summary_parts.append(
            f"skipped {len(stats.skipped_columns)} unsupported column(s)"
        )

    if stats.missing_columns:
        summary_parts.append(f"{len(stats.missing_columns)} column(s) not found")

    return working_frame, "; ".join(summary_parts)


def _apply_default_mode(
    frame: pd.DataFrame,
    target_columns: List[str],
    auto_columns: List[str],
    missing_columns: List[str],
    mode: str,
    configured_columns: List[str],
    signal: StandardizeDatesNodeSignal,
) -> Tuple[pd.DataFrame, str]:
    working_frame = frame.copy()
    stats = DateStandardizationStats()
    stats.missing_columns.update(missing_columns)
    stats.auto_detected_columns.update(auto_columns)

    target_format = MODE_FORMAT_MAP.get(mode, "%Y-%m-%d")
    dayfirst = mode in DAYFIRST_MODES
    auto_columns_set = set(auto_columns)

    for column in target_columns:
        processed, changes, failures, skipped = _format_column(
            working_frame,
            column,
            target_format,
            dayfirst,
        )

        if not processed:
            stats.register_failures(failures)
            if skipped:
                stats.register_skipped(column)
            continue

        stats.record_processed(
            column=column,
            mode_value=mode,
            changes=changes,
            failures=failures,
            auto_detected=column in auto_columns_set,
            dtype=str(frame[column].dtype),
        )

    stats.apply_to_signal(signal)

    if stats.processed_columns == 0:
        return frame, "Standardize dates: no convertible values"

    summary_parts = [
        f"Standardize dates: processed {stats.processed_columns} column(s)"
    ]
    if stats.converted_values:
        summary_parts.append(f"formatted {stats.converted_values} value(s)")
    else:
        summary_parts.append("no changes detected")

    summary_parts.append(f"format: {MODE_LABEL_MAP.get(mode, target_format)}")

    if stats.parse_failures:
        summary_parts.append(f"{stats.parse_failures} value(s) could not be parsed")

    if not configured_columns and auto_columns:
        summary_parts.append("auto-detected date columns")

    if stats.skipped_columns:
        summary_parts.append(
            f"skipped {len(stats.skipped_columns)} unsupported column(s)"
        )

    if stats.missing_columns:
        summary_parts.append(f"{len(stats.missing_columns)} column(s) not found")

    return working_frame, "; ".join(summary_parts)


def _initialize_signal(
    node: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], StandardizeDatesNodeSignal]:
    node_dict = node if isinstance(node, dict) else {}
    node_id = node_dict.get("id") if node_dict else None
    data = node_dict.get("data") if node_dict else None
    config = (data or {}).get("config") or {}
    configured_columns = _coerce_string_list(config.get("columns"))
    signal = StandardizeDatesNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )
    return config, configured_columns, signal


def _build_plan(
    frame: pd.DataFrame,
    config: Dict[str, Any],
    configured_columns: List[str],
) -> _DatePlan:
    default_mode = _resolve_mode_value(config.get("mode"))
    strategies = _normalize_strategies(config, default_mode)

    if strategies:
        target_columns: List[str] = []
        auto_columns: List[str] = []
        missing_columns: List[str] = []
    else:
        target_columns, auto_columns, missing_columns = _resolve_target_columns(
            frame, configured_columns
        )

    return _DatePlan(
        strategies=strategies,
        target_columns=target_columns,
        auto_columns=auto_columns,
        missing_columns=missing_columns,
        default_mode=default_mode,
    )


def apply_standardize_date_formats(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, StandardizeDatesNodeSignal]:
    """Normalize date-like columns based on the node configuration."""
    config, configured_columns, signal = _initialize_signal(node)

    if frame.empty:
        return frame, "Standardize dates: no data available", signal

    plan = _build_plan(frame, config, configured_columns)

    if plan.strategies:
        working_frame, summary = _apply_strategies(frame, plan.strategies, signal)
        return working_frame, summary, signal

    if not plan.target_columns:
        signal.auto_detected_columns = sorted(set(plan.auto_columns))
        signal.missing_columns = sorted(set(plan.missing_columns))
        return frame, "Standardize dates: no eligible columns", signal

    working_frame, summary = _apply_default_mode(
        frame,
        plan.target_columns,
        plan.auto_columns,
        plan.missing_columns,
        plan.default_mode,
        list(configured_columns),
        signal,
    )

    return working_frame, summary, signal


__all__ = ["apply_standardize_date_formats"]
