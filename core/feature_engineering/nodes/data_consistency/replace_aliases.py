"""Standardize alias and typo replacements for data consistency nodes."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

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


def apply_replace_aliases_typos(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, ReplaceAliasesNodeSignal]:
    """Apply alias and typo normalization based on the node configuration."""
    node_id = node.get("id") if isinstance(node, dict) else None

    data = node.get("data") or {}
    config = data.get("config") or {}

    configured_columns = _coerce_string_list(config.get("columns"))
    signal = ReplaceAliasesNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
    )

    if frame.empty:
        return frame, "Replace aliases: no data available", signal

    custom_pairs = _parse_custom_alias_pairs(config.get("custom_pairs"))
    signal.custom_pairs_used = bool(custom_pairs)

    def _normalize_mode(value: Any, fallback: str) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return fallback
            if normalized in {"canonicalize_country_codes", "normalize_boolean", "punctuation", "custom"}:
                return normalized
            if normalized in {"country", "country_codes", "country_aliases"}:
                return "canonicalize_country_codes"
            if normalized in {"boolean", "normalize_booleans", "normalize_boolean_tokens", "boolean_tokens"}:
                return "normalize_boolean"
            if normalized in {"punctuation_cleanup", "strip_punctuation"}:
                return "punctuation"
        return fallback

    fallback_mode = _normalize_mode(config.get("mode"), "canonicalize_country_codes")

    missing_columns: List[str] = []

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

    raw_strategies = config.get("alias_strategies") or config.get("strategies")

    def _sanitize_columns(columns: List[str]) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for column in columns:
            normalized = str(column or "").strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    def _normalize_strategy_entry(raw_entry: Any) -> Dict[str, Any] | None:
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

    strategies: List[Dict[str, Any]] = []
    if isinstance(raw_strategies, list):
        for entry in raw_strategies:
            normalized_strategy = _normalize_strategy_entry(entry)
            if normalized_strategy:
                strategies.append(normalized_strategy)
    elif isinstance(raw_strategies, dict):
        normalized_strategy = _normalize_strategy_entry(raw_strategies)
        if normalized_strategy:
            strategies.append(normalized_strategy)

    if not strategies:
        fallback_strategy_columns = _sanitize_columns(configured_columns)
        fallback_strategy_auto = fallback_auto_detect
        if not fallback_strategy_columns and not fallback_strategy_auto:
            fallback_strategy_auto = True
        strategies.append(
            {
                "mode": fallback_mode,
                "columns": fallback_strategy_columns,
                "auto_detect": fallback_strategy_auto,
            }
        )

    text_candidates = _auto_detect_text_columns(frame)
    assigned_columns: Set[str] = set()
    auto_detected_columns: List[str] = []
    normalized_tasks: List[Dict[str, Any]] = []
    any_auto_detect_enabled = any(strategy["auto_detect"] for strategy in strategies)
    signal.auto_detected_columns = []
    total_rows = int(frame.shape[0])

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
        normalized_tasks.append(
            {
                "mode": strategy["mode"],
                "manual_columns": manual_columns,
                "auto_columns": auto_columns,
                "columns": combined_columns,
                "auto_detect": strategy["auto_detect"],
            }
        )

    working_frame = frame.copy()
    processed_columns = 0
    replacements = 0
    skipped_columns: List[str] = []
    strategies_executed = 0
    mode_labels_used: Set[str] = set()
    skipped_custom_strategies = 0
    processed_signals: List[ReplaceAliasesAppliedColumnSignal] = []

    mode_label_map = {
        "canonicalize_country_codes": "country aliases",
        "normalize_boolean": "boolean tokens",
        "punctuation": "punctuation cleanup",
        "custom": "custom map",
    }

    for task in normalized_tasks:
        columns = task["columns"]
        if not columns:
            if _normalize_mode(task["mode"], fallback_mode) == "custom" and not custom_pairs:
                skipped_custom_strategies += 1
            continue

        normalized_mode = _normalize_mode(task["mode"], fallback_mode)

        if normalized_mode == "custom" and not custom_pairs:
            skipped_custom_strategies += 1
            continue

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

        strategy_had_processed = False

        def _canonical_country(value: str) -> str:
            normalized_key = _normalize_alias_key(value)
            if normalized_key in combined_aliases:
                return combined_aliases[normalized_key]
            sanitized = value.translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
            if sanitized and sanitized != value:
                if len(sanitized) in {2, 3}:
                    return sanitized.upper()
                return sanitized.title()
            return value

        def _normalize_boolean_value(value: str) -> str:
            normalized_key = _normalize_alias_key(value)
            return combined_aliases.get(normalized_key, value)

        def _strip_punctuation(value: str) -> str:
            cleaned = re.sub(r"[^\w\s]", "", value)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned or value

        for column in columns:
            if column not in frame.columns:
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

            def _transform(entry: Any) -> Any:
                if entry is pd.NA or entry is None:
                    return entry
                text = str(entry)
                if not text:
                    return text
                if normalized_mode == "punctuation":
                    return _strip_punctuation(text)
                if normalized_mode == "normalize_boolean":
                    return _normalize_boolean_value(text)
                if normalized_mode == "custom":
                    normalized_key = _normalize_alias_key(text)
                    return combined_aliases.get(normalized_key, text)
                return _canonical_country(text)

            transformed = string_series.map(_transform).astype("string")

            before = string_series.fillna("").to_numpy(dtype=object)
            after = transformed.fillna("").to_numpy(dtype=object)
            change_count = int(np.not_equal(before, after).sum())

            if change_count:
                if pd_types.is_object_dtype(series):
                    working_frame[column] = transformed.astype(object)
                elif pd_types.is_string_dtype(series):
                    working_frame[column] = transformed.astype(series.dtype)
                else:
                    working_frame[column] = transformed.astype("string")
                replacements += change_count

            processed_columns += 1
            strategy_had_processed = True
            processed_signals.append(
                ReplaceAliasesAppliedColumnSignal(
                    column=str(column),
                    mode=normalized_mode,
                    mode_label=mode_label_map.get(normalized_mode, normalized_mode),
                    replacements=change_count,
                    total_rows=total_rows,
                    auto_detected=column in auto_detected_columns,
                    dtype=str(series.dtype),
                )
            )

        if strategy_had_processed:
            strategies_executed += 1
            mode_labels_used.add(mode_label_map.get(normalized_mode, normalized_mode))

    if processed_columns == 0:
        signal.auto_detected_columns = sorted(set(auto_detected_columns))
        signal.missing_columns = sorted(set(missing_columns))
        signal.skipped_columns = sorted(set(skipped_columns))
        signal.skipped_custom_strategies = skipped_custom_strategies
        signal.strategies_executed = strategies_executed
        signal.replacements = replacements
        signal.modes_used = sorted(mode_labels_used)
        signal.processed_columns = processed_signals
        if skipped_custom_strategies and skipped_custom_strategies == len(
            [task for task in normalized_tasks if _normalize_mode(task["mode"], fallback_mode) == "custom"]
        ):
            return frame, "Replace aliases: no custom mappings configured", signal
        return frame, "Replace aliases: no eligible text columns", signal

    summary_parts = [f"Replace aliases: processed {processed_columns} column(s)"]
    if strategies_executed > 1:
        summary_parts[0] += f" across {strategies_executed} strategies"

    if replacements:
        summary_parts.append(f"standardized {replacements} value(s)")
    else:
        summary_parts.append("no changes detected")

    if mode_labels_used:
        summary_parts.append(f"modes: {', '.join(sorted(mode_labels_used))}")

    if custom_pairs:
        summary_parts.append("custom replacements included")

    auto_detected_set = set(auto_detected_columns)
    if auto_detected_set:
        summary_parts.append(f"auto-detected {len(auto_detected_set)} column(s)")
    elif any_auto_detect_enabled:
        summary_parts.append("auto-detect enabled")

    skipped_columns_set = sorted(set(skipped_columns))
    if skipped_columns_set:
        summary_parts.append(f"skipped {len(skipped_columns_set)} unsupported column(s)")

    missing_columns_set = sorted(set(missing_columns))
    if missing_columns_set:
        summary_parts.append(f"{len(missing_columns_set)} column(s) not found")

    if skipped_custom_strategies and not custom_pairs:
        summary_parts.append("custom strategy skipped (no mappings)")

    signal.auto_detected_columns = sorted(set(auto_detected_columns))
    signal.missing_columns = sorted(set(missing_columns))
    signal.skipped_columns = sorted(set(skipped_columns))
    signal.skipped_custom_strategies = skipped_custom_strategies
    signal.strategies_executed = strategies_executed
    signal.replacements = replacements
    signal.modes_used = sorted(mode_labels_used)
    signal.processed_columns = processed_signals

    return working_frame, "; ".join(summary_parts), signal


__all__ = ["apply_replace_aliases_typos"]
