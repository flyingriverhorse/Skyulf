"""Standardize alias and typo replacements for data consistency nodes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from ..feature_eng.utils import (
    ALIAS_PUNCTUATION_TABLE,
    COMMON_BOOLEAN_ALIASES,
    COUNTRY_ALIAS_MAP,
    _auto_detect_text_columns,
    _coerce_config_boolean,
    _coerce_string_list,
    _normalize_alias_key,
    _parse_custom_alias_pairs,
)
from core.feature_engineering.schemas import (
    ReplaceAliasesAppliedColumnSignal,
    ReplaceAliasesNodeSignal,
)

MODE_LABEL_MAP = {
    "canonicalize_country_codes": "country aliases",
    "normalize_boolean": "boolean tokens",
    "punctuation": "punctuation cleanup",
    "custom": "custom map",
}


@dataclass
class _AliasTaskPlan:
    tasks: List[Dict[str, Any]]
    custom_pairs: Dict[str, str]
    fallback_mode: str
    auto_detected_columns: List[str]
    missing_columns: List[str]
    any_auto_detect_enabled: bool
    custom_task_count: int


@dataclass
class _AliasExecutionResult:
    frame: pd.DataFrame
    processed_columns: int
    replacements: int
    skipped_columns: List[str]
    strategies_executed: int
    mode_labels_used: Set[str]
    skipped_custom_strategies: int
    processed_signals: List[ReplaceAliasesAppliedColumnSignal]


def _normalize_mode(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return fallback
        if normalized in MODE_LABEL_MAP:
            return normalized
        if normalized in {"country", "country_codes", "country_aliases"}:
            return "canonicalize_country_codes"
        if normalized in {"boolean", "normalize_booleans", "normalize_boolean_tokens", "boolean_tokens"}:
            return "normalize_boolean"
        if normalized in {"punctuation_cleanup", "strip_punctuation"}:
            return "punctuation"
    return fallback


def _coerce_auto_detect_flag(config: Dict[str, Any], configured_columns: List[str]) -> bool:
    auto_detect_raw: Any = None
    auto_detect_configured = False
    for key in ("auto_detect", "autoDetect", "auto_detect_columns", "auto", "auto_detect_text"):
        if isinstance(config, dict) and key in config:
            auto_detect_raw = config[key]
            auto_detect_configured = True
            break

    fallback_auto_detect = (
        _coerce_config_boolean(auto_detect_raw, False) if auto_detect_configured else False
    )

    if not auto_detect_configured and not configured_columns:
        fallback_auto_detect = True

    return fallback_auto_detect


def _sanitize_columns(columns: List[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for column in columns:
        normalized = str(column or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _normalize_strategy_entry(
    raw_entry: Any,
    fallback_mode: str,
    fallback_auto_detect: bool,
) -> Dict[str, Any] | None:
    if not isinstance(raw_entry, dict):
        return None

    columns = _sanitize_columns(_coerce_string_list(raw_entry.get("columns")))

    auto_value: Any = None
    for key in ("auto_detect", "autoDetect", "auto_detect_columns", "auto", "auto_detect_text"):
        if key in raw_entry:
            auto_value = raw_entry[key]
            break

    auto_flag = _coerce_config_boolean(auto_value, fallback_auto_detect)
    mode_value = _normalize_mode(raw_entry.get("mode"), fallback_mode)

    if not columns and not auto_flag:
        return None

    return {"mode": mode_value, "columns": columns, "auto_detect": auto_flag}


def _prepare_strategies(
    config: Dict[str, Any],
    configured_columns: List[str],
    fallback_mode: str,
    fallback_auto_detect: bool,
) -> List[Dict[str, Any]]:
    raw_strategies = config.get("alias_strategies") or config.get("strategies")
    strategies: List[Dict[str, Any]] = []

    if isinstance(raw_strategies, list):
        for entry in raw_strategies:
            normalized = _normalize_strategy_entry(entry, fallback_mode, fallback_auto_detect)
            if normalized:
                strategies.append(normalized)
    elif isinstance(raw_strategies, dict):
        normalized = _normalize_strategy_entry(raw_strategies, fallback_mode, fallback_auto_detect)
        if normalized:
            strategies.append(normalized)

    if strategies:
        return strategies

    fallback_columns = _sanitize_columns(configured_columns)
    auto_flag = fallback_auto_detect
    if not fallback_columns and not auto_flag:
        auto_flag = True

    strategies.append(
        {
            "mode": fallback_mode,
            "columns": fallback_columns,
            "auto_detect": auto_flag,
        }
    )
    return strategies


def _assign_columns_to_tasks(
    strategies: List[Dict[str, Any]],
    frame: pd.DataFrame,
    text_candidates: List[str],
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    assigned_columns: Set[str] = set()
    auto_detected_columns: List[str] = []
    missing_columns: List[str] = []
    tasks: List[Dict[str, Any]] = []

    for strategy in strategies:
        manual_columns: List[str] = []
        for column in strategy["columns"]:
            if column in assigned_columns:
                continue
            if column in frame.columns:
                manual_columns.append(column)
                assigned_columns.add(column)
            else:
                missing_columns.append(column)

        auto_columns: List[str] = []
        if strategy["auto_detect"]:
            for column in text_candidates:
                if column in assigned_columns:
                    continue
                if column not in frame.columns:
                    continue
                auto_columns.append(column)
                assigned_columns.add(column)
            if auto_columns:
                auto_detected_columns.extend(auto_columns)

        combined_columns = manual_columns + auto_columns
        tasks.append(
            {
                "mode": strategy["mode"],
                "manual_columns": manual_columns,
                "auto_columns": auto_columns,
                "columns": combined_columns,
                "auto_detect": strategy["auto_detect"],
            }
        )

    return tasks, missing_columns, auto_detected_columns


def _build_alias_map(normalized_mode: str, custom_pairs: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    combined_aliases: Dict[str, str] = {}

    if normalized_mode == "normalize_boolean":
        combined_aliases.update(COMMON_BOOLEAN_ALIASES)
    elif normalized_mode == "punctuation":
        pass
    elif normalized_mode == "custom":
        combined_aliases.update(custom_pairs)
    else:
        normalized_mode = "canonicalize_country_codes"
        combined_aliases.update(COUNTRY_ALIAS_MAP)

    if custom_pairs and normalized_mode != "custom":
        combined_aliases.update(custom_pairs)

    return normalized_mode, combined_aliases


def _canonical_country(value: str, aliases: Dict[str, str]) -> str:
    normalized_key = _normalize_alias_key(value)
    if normalized_key in aliases:
        return aliases[normalized_key]
    sanitized = value.translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
    if sanitized and sanitized != value:
        if len(sanitized) in {2, 3}:
            return sanitized.upper()
        return sanitized.title()
    return value


def _normalize_boolean_value(value: str, aliases: Dict[str, str]) -> str:
    normalized_key = _normalize_alias_key(value)
    return aliases.get(normalized_key, value)


def _strip_punctuation(value: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or value


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


def _transform_alias_entry(
    entry: Any,
    normalized_mode: str,
    aliases: Dict[str, str],
) -> Any:
    if entry is pd.NA or entry is None:
        return entry
    text = str(entry)
    if not text:
        return text
    if normalized_mode == "punctuation":
        return _strip_punctuation(text)
    if normalized_mode in {"normalize_boolean", "custom"}:
        normalized_key = _normalize_alias_key(text)
        return aliases.get(normalized_key, text)
    return _canonical_country(text, aliases)


def _transform_alias_series(
    series: pd.Series,
    normalized_mode: str,
    aliases: Dict[str, str],
) -> Tuple[pd.Series, pd.Series]:
    string_series = series.astype("string")
    transformed = string_series.map(
        lambda entry: _transform_alias_entry(entry, normalized_mode, aliases)
    ).astype("string")
    return string_series, transformed


def _process_alias_column(
    working_frame: pd.DataFrame,
    column: str,
    normalized_mode: str,
    aliases: Dict[str, str],
    auto_detected_columns: Set[str],
    total_rows: int,
) -> Tuple[Optional[ReplaceAliasesAppliedColumnSignal], Optional[str]]:
    series = working_frame[column]
    if not _is_textual_series(series):
        return None, column

    original, transformed = _transform_alias_series(series, normalized_mode, aliases)
    replacements = _count_changes(original, transformed)

    if replacements:
        working_frame[column] = _cast_transformed(transformed, series)

    signal = ReplaceAliasesAppliedColumnSignal(
        column=str(column),
        mode=normalized_mode,
        mode_label=MODE_LABEL_MAP.get(normalized_mode, normalized_mode),
        replacements=replacements,
        total_rows=total_rows,
        auto_detected=column in auto_detected_columns,
        dtype=str(series.dtype),
    )

    return signal, None


def _evaluate_task_mode(
    task: Dict[str, Any],
    custom_pairs: Dict[str, str],
    fallback_mode: str,
) -> Tuple[Optional[str], bool]:
    normalized_mode = _normalize_mode(task.get("mode"), fallback_mode)

    if not task.get("columns"):
        return None, normalized_mode == "custom" and not custom_pairs

    if normalized_mode == "custom" and not custom_pairs:
        return None, True

    return normalized_mode, False


def _initialize_signal(
    node: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], ReplaceAliasesNodeSignal]:
    node_dict = node if isinstance(node, dict) else {}
    node_id = node_dict.get("id") if node_dict else None
    data = node_dict.get("data") if node_dict else None
    config = (data or {}).get("config") or {}
    configured_columns = _coerce_string_list(config.get("columns"))
    signal = ReplaceAliasesNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )
    return config, list(configured_columns), signal


def _create_task_plan(
    frame: pd.DataFrame,
    config: Dict[str, Any],
    configured_columns: List[str],
) -> _AliasTaskPlan:
    custom_pairs = _parse_custom_alias_pairs(config.get("custom_pairs"))
    fallback_mode = _normalize_mode(config.get("mode"), "canonicalize_country_codes")
    fallback_auto_detect = _coerce_auto_detect_flag(config, configured_columns)
    strategies = _prepare_strategies(config, configured_columns, fallback_mode, fallback_auto_detect)

    text_candidates = _auto_detect_text_columns(frame)
    tasks, missing_columns, auto_detected_columns = _assign_columns_to_tasks(
        strategies, frame, text_candidates
    )

    any_auto_detect_enabled = any(strategy.get("auto_detect") for strategy in strategies)
    custom_task_count = sum(
        1 for task in tasks if _normalize_mode(task.get("mode"), fallback_mode) == "custom"
    )

    return _AliasTaskPlan(
        tasks=tasks,
        custom_pairs=custom_pairs,
        fallback_mode=fallback_mode,
        auto_detected_columns=auto_detected_columns,
        missing_columns=missing_columns,
        any_auto_detect_enabled=bool(any_auto_detect_enabled),
        custom_task_count=custom_task_count,
    )


def _execute_alias_tasks(
    frame: pd.DataFrame,
    plan: _AliasTaskPlan,
) -> _AliasExecutionResult:
    working_frame = frame.copy()
    total_rows = int(frame.shape[0])
    processed_columns = 0
    replacements = 0
    skipped_columns: List[str] = []
    strategies_executed = 0
    mode_labels_used: Set[str] = set()
    skipped_custom_strategies = 0
    processed_signals: List[ReplaceAliasesAppliedColumnSignal] = []
    auto_detected_set = set(plan.auto_detected_columns)

    for task in plan.tasks:
        normalized_mode, custom_skipped = _evaluate_task_mode(
            task, plan.custom_pairs, plan.fallback_mode
        )
        if normalized_mode is None:
            if custom_skipped:
                skipped_custom_strategies += 1
            continue

        effective_mode, alias_map = _build_alias_map(normalized_mode, plan.custom_pairs)
        strategy_processed = False

        for column in task.get("columns", []):
            if column not in working_frame.columns:
                continue

            signal_record, skipped = _process_alias_column(
                working_frame,
                column,
                effective_mode,
                alias_map,
                auto_detected_set,
                total_rows,
            )

            if skipped:
                skipped_columns.append(skipped)
                continue

            if signal_record is None:
                continue

            processed_columns += 1
            replacements += signal_record.replacements
            processed_signals.append(signal_record)
            strategy_processed = True
            if signal_record.mode_label is not None:
                mode_labels_used.add(signal_record.mode_label)

        if strategy_processed:
            strategies_executed += 1

    return _AliasExecutionResult(
        frame=working_frame,
        processed_columns=processed_columns,
        replacements=replacements,
        skipped_columns=skipped_columns,
        strategies_executed=strategies_executed,
        mode_labels_used=mode_labels_used,
        skipped_custom_strategies=skipped_custom_strategies,
        processed_signals=processed_signals,
    )


def _build_alias_summary(
    result: _AliasExecutionResult,
    plan: _AliasTaskPlan,
) -> str:
    summary_parts = [
        f"Replace aliases: processed {result.processed_columns} column(s)"
    ]

    if result.strategies_executed > 1:
        summary_parts[0] += f" across {result.strategies_executed} strategies"

    if result.replacements:
        summary_parts.append(f"standardized {result.replacements} value(s)")
    else:
        summary_parts.append("no changes detected")

    if result.mode_labels_used:
        summary_parts.append(f"modes: {', '.join(sorted(result.mode_labels_used))}")

    if plan.custom_pairs:
        summary_parts.append("custom replacements included")

    auto_detected_set = set(plan.auto_detected_columns)
    if auto_detected_set:
        summary_parts.append(f"auto-detected {len(auto_detected_set)} column(s)")
    elif plan.any_auto_detect_enabled:
        summary_parts.append("auto-detect enabled")

    skipped_columns_set = sorted(set(result.skipped_columns))
    if skipped_columns_set:
        summary_parts.append(f"skipped {len(skipped_columns_set)} unsupported column(s)")

    missing_columns_set = sorted(set(plan.missing_columns))
    if missing_columns_set:
        summary_parts.append(f"{len(missing_columns_set)} column(s) not found")

    if result.skipped_custom_strategies and not plan.custom_pairs:
        summary_parts.append("custom strategy skipped (no mappings)")

    return "; ".join(summary_parts)


def apply_replace_aliases_typos(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, ReplaceAliasesNodeSignal]:
    """Apply alias and typo normalization based on the node configuration."""
    config, configured_columns, signal = _initialize_signal(node)

    if frame.empty:
        return frame, "Replace aliases: no data available", signal

    plan = _create_task_plan(frame, config, configured_columns)
    signal.custom_pairs_used = bool(plan.custom_pairs)

    signal.auto_detected_columns = sorted(set(plan.auto_detected_columns))
    signal.missing_columns = sorted(set(plan.missing_columns))

    result = _execute_alias_tasks(frame, plan)

    signal.skipped_columns = sorted(set(result.skipped_columns))
    signal.skipped_custom_strategies = result.skipped_custom_strategies
    signal.strategies_executed = result.strategies_executed
    signal.replacements = result.replacements
    signal.modes_used = sorted(result.mode_labels_used)
    signal.processed_columns = result.processed_signals

    if result.processed_columns == 0:
        if (
            result.skipped_custom_strategies
            and result.skipped_custom_strategies == plan.custom_task_count
        ):
            return frame, "Replace aliases: no custom mappings configured", signal
        return frame, "Replace aliases: no eligible text columns", signal

    summary = _build_alias_summary(result, plan)
    return result.frame, summary, signal


__all__ = ["apply_replace_aliases_typos"]
