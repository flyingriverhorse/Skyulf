"""Standardize date formatting for data consistency nodes."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from ..feature_eng.utils import (
    _auto_detect_datetime_columns,
    _auto_detect_text_columns,
    _coerce_string_list,
)
from core.feature_engineering.schemas import (
    StandardizeDatesAppliedColumnSignal,
    StandardizeDatesNodeSignal,
)


def apply_standardize_date_formats(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, StandardizeDatesNodeSignal]:
    """Normalize date-like columns based on the node configuration."""
    data = node.get("data") or {}
    config = data.get("config") or {}

    node_id = node.get("id") if isinstance(node, dict) else None

    format_map = {
        "iso_date": "%Y-%m-%d",
        "iso_datetime": "%Y-%m-%d %H:%M:%S",
        "month_day_year": "%m/%d/%Y",
        "day_month_year": "%d/%m/%Y",
    }
    mode_label_map = {
        "iso_date": "ISO date",
        "iso_datetime": "ISO datetime",
        "month_day_year": "MM/DD/YYYY",
        "day_month_year": "DD/MM/YYYY",
    }

    def resolve_mode_value(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in format_map:
            return normalized
        return "iso_date"

    configured_columns = _coerce_string_list(config.get("columns"))
    signal = StandardizeDatesNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )

    if frame.empty:
        return frame, "Standardize dates: no data available", signal

    raw_strategies = config.get("format_strategies")
    normalized_strategies: List[Dict[str, Any]] = []
    if isinstance(raw_strategies, list):
        for entry in raw_strategies:
            if not isinstance(entry, dict):
                continue
            mode_value = resolve_mode_value(entry.get("mode") or config.get("mode"))
            columns = _coerce_string_list(entry.get("columns"))
            auto_detect = bool(
                entry.get("auto_detect")
                or entry.get("autoDetect")
                or entry.get("auto")
                or entry.get("auto_detect_columns")
            )
            if not columns and not auto_detect:
                continue
            normalized_strategies.append(
                {
                    "mode": mode_value,
                    "columns": columns,
                    "auto_detect": auto_detect,
                }
            )

    total_rows = int(frame.shape[0])

    if normalized_strategies:
        working_frame = frame.copy()
        available_auto_columns = _auto_detect_datetime_columns(frame)
        if not available_auto_columns:
            available_auto_columns = _auto_detect_text_columns(frame)
        remaining_auto_columns = [column for column in available_auto_columns if column in frame.columns]

        processed_columns: Set[str] = set()
        missing_columns_set: Set[str] = set()
        skipped_columns_set: Set[str] = set()
        auto_detected_columns: Set[str] = set()
        total_processed = 0
        converted_values = 0
        parse_failures = 0
        mode_counts: Counter[str] = Counter()
        processed_signals: List[StandardizeDatesAppliedColumnSignal] = []

        def format_column(column: str, target_format: str, dayfirst: bool) -> Tuple[bool, int, int, bool]:
            series = working_frame[column]
            candidate_datetime: Optional[pd.Series] = None

            if pd_types.is_datetime64_any_dtype(series):
                candidate_datetime = pd.to_datetime(series, errors="coerce")
            elif (
                pd_types.is_string_dtype(series)
                or pd_types.is_object_dtype(series)
                or pd_types.is_categorical_dtype(series)
            ):
                candidate_datetime = pd.to_datetime(
                    series,
                    errors="coerce",
                    infer_datetime_format=True,
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

        for strategy in normalized_strategies:
            mode_value = resolve_mode_value(strategy.get("mode"))
            target_format = format_map.get(mode_value, "%Y-%m-%d")
            dayfirst = mode_value in {"day_month_year", "day_first"}

            manual_columns = strategy.get("columns", [])
            auto_detect_flag = bool(strategy.get("auto_detect"))

            for column in manual_columns:
                if column not in frame.columns:
                    missing_columns_set.add(column)

            existing_manual = [
                column
                for column in manual_columns
                if column in frame.columns and column not in processed_columns
            ]
            for column in existing_manual:
                if column in remaining_auto_columns:
                    remaining_auto_columns.remove(column)

            candidate_columns = existing_manual[:]
            added_auto: List[str] = []

            if auto_detect_flag:
                for column in list(remaining_auto_columns):
                    if column in processed_columns or column in candidate_columns:
                        continue
                    candidate_columns.append(column)
                    added_auto.append(column)
                    remaining_auto_columns.remove(column)

            for column in candidate_columns:
                if column in processed_columns:
                    continue
                processed, changes, failures, skipped = format_column(column, target_format, dayfirst)
                if failures:
                    parse_failures += failures
                processed_columns.add(column)
                if not processed:
                    if skipped:
                        skipped_columns_set.add(column)
                    continue
                total_processed += 1
                converted_values += changes
                mode_counts[mode_value] += 1
                auto_flag = column in added_auto
                if auto_flag:
                    auto_detected_columns.add(column)
                processed_signals.append(
                    StandardizeDatesAppliedColumnSignal(
                        column=str(column),
                        mode=mode_value,
                        mode_label=mode_label_map.get(mode_value, mode_value),
                        converted_values=changes,
                        parse_failures=failures,
                        auto_detected=auto_flag,
                        dtype=str(frame[column].dtype),
                    )
                )

        signal.auto_detected_columns = sorted(auto_detected_columns)
        signal.missing_columns = sorted(missing_columns_set)
        signal.skipped_columns = sorted(skipped_columns_set)
        signal.total_converted_values = converted_values
        signal.total_parse_failures = parse_failures
        signal.mode_counts = dict(mode_counts)
        signal.processed_columns = processed_signals

        if total_processed == 0:
            summary_bits = ["Standardize dates: no convertible values"]
            if missing_columns_set:
                summary_bits.append(f"{len(missing_columns_set)} column(s) not found")
            if skipped_columns_set:
                summary_bits.append(f"skipped {len(skipped_columns_set)} unsupported column(s)")
            if parse_failures:
                summary_bits.append(f"{parse_failures} value(s) could not be parsed")
            if auto_detected_columns:
                summary_bits.append(f"auto-detected {len(auto_detected_columns)} column(s)")
            return frame, "; ".join(summary_bits), signal

        summary_parts = [f"Standardize dates: processed {total_processed} column(s)"]
        if converted_values:
            summary_parts.append(f"formatted {converted_values} value(s)")
        else:
            summary_parts.append("no changes detected")

        if mode_counts:
            mode_details = ", ".join(
                f"{mode_label_map.get(mode, mode)} ({count})"
                for mode, count in sorted(mode_counts.items())
            )
            summary_parts.append(f"formats applied: {mode_details}")

        if auto_detected_columns:
            summary_parts.append(f"auto-detected {len(auto_detected_columns)} column(s)")

        if parse_failures:
            summary_parts.append(f"{parse_failures} value(s) could not be parsed")

        if skipped_columns_set:
            summary_parts.append(f"skipped {len(skipped_columns_set)} unsupported column(s)")

        if missing_columns_set:
            summary_parts.append(f"{len(missing_columns_set)} column(s) not found")

        return working_frame, "; ".join(summary_parts), signal

    target_columns = [column for column in configured_columns if column in frame.columns]
    missing_columns = [column for column in configured_columns if column not in frame.columns]

    auto_columns: List[str] = []
    if not target_columns:
        auto_columns = _auto_detect_datetime_columns(frame)
        if not auto_columns:
            auto_columns = _auto_detect_text_columns(frame)
        target_columns = auto_columns

    signal.auto_detected_columns = sorted(set(auto_columns))
    signal.missing_columns = sorted(set(missing_columns))

    if not target_columns:
        return frame, "Standardize dates: no eligible columns", signal

    mode = resolve_mode_value(config.get("mode"))
    target_format = format_map.get(mode, "%Y-%m-%d")
    dayfirst = mode in {"day_month_year", "day_first"}

    working_frame = frame.copy()
    processed = 0
    converted_values = 0
    parse_failures = 0
    skipped_columns: List[str] = []
    processed_signals: List[StandardizeDatesAppliedColumnSignal] = []

    for column in target_columns:
        series = working_frame[column]
        candidate_datetime: Optional[pd.Series] = None

        if pd_types.is_datetime64_any_dtype(series):
            candidate_datetime = pd.to_datetime(series, errors="coerce")
        elif (
            pd_types.is_string_dtype(series)
            or pd_types.is_object_dtype(series)
            or pd_types.is_categorical_dtype(series)
        ):
            candidate_datetime = pd.to_datetime(
                series,
                errors="coerce",
                infer_datetime_format=True,
                dayfirst=dayfirst,
            )
        else:
            skipped_columns.append(column)
            continue

        if candidate_datetime is None:
            skipped_columns.append(column)
            continue

        try:
            candidate_datetime = candidate_datetime.dt.tz_localize(None)  # type: ignore[call-arg]
        except (AttributeError, TypeError, ValueError):
            pass

        mask = candidate_datetime.notna()
        original_notna = series.notna()
        if not mask.any():
            if original_notna.any():
                parse_failures += int(original_notna.sum())
            skipped_columns.append(column)
            continue

        formatted_values = candidate_datetime.dt.strftime(target_format)
        string_series = series.astype("string")
        formatted_series = string_series.copy()
        formatted_series.loc[mask] = formatted_values.loc[mask].astype("string")

        before = string_series.fillna("").to_numpy(dtype=object)
        after = formatted_series.fillna("").to_numpy(dtype=object)
        changes = int(np.not_equal(before, after).sum())

        failures = int((original_notna & ~mask).sum())
        parse_failures += failures

        if changes:
            working_frame[column] = formatted_series
            converted_values += changes

        processed += 1
        processed_signals.append(
            StandardizeDatesAppliedColumnSignal(
                column=str(column),
                mode=mode,
                mode_label=mode_label_map.get(mode, mode),
                converted_values=changes,
                parse_failures=failures,
                auto_detected=column in auto_columns,
                dtype=str(series.dtype),
            )
        )

    signal.skipped_columns = sorted(set(skipped_columns))
    signal.total_converted_values = converted_values
    signal.total_parse_failures = parse_failures
    signal.mode_counts = {mode: processed}
    signal.processed_columns = processed_signals

    if processed == 0:
        return frame, "Standardize dates: no convertible values", signal

    summary_parts = [f"Standardize dates: processed {processed} column(s)"]
    if converted_values:
        summary_parts.append(f"formatted {converted_values} value(s)")
    else:
        summary_parts.append("no changes detected")

    summary_parts.append(f"format: {mode_label_map.get(mode, target_format)}")

    if parse_failures:
        summary_parts.append(f"{parse_failures} value(s) could not be parsed")

    if not configured_columns and auto_columns:
        summary_parts.append("auto-detected date columns")

    if skipped_columns:
        summary_parts.append(f"skipped {len(skipped_columns)} unsupported column(s)")

    if missing_columns:
        summary_parts.append(f"{len(missing_columns)} column(s) not found")

    return working_frame, "; ".join(summary_parts), signal


__all__ = ["apply_standardize_date_formats"]
