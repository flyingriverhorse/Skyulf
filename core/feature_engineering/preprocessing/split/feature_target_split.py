"""Feature/target separation helpers for preprocessing nodes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

from core.feature_engineering.schemas import FeatureTargetSplitNodeSignal


def _normalize_target_column(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, str):
        return raw_value.strip()
    return str(raw_value).strip()


def _normalize_feature_columns(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []

    values: List[Any]
    if isinstance(raw_value, list):
        values = raw_value
    elif isinstance(raw_value, str):
        values = raw_value.split(",")
    else:
        return []

    normalized: List[str] = []
    for entry in values:
        if entry is None:
            continue
        if isinstance(entry, str):
            text = entry.strip()
        else:
            text = str(entry).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _update_target_metadata(
    frame: pd.DataFrame,
    target_column: str,
    total_rows: int,
    signal: FeatureTargetSplitNodeSignal,
) -> Tuple[bool, List[str]]:
    available_columns = list(frame.columns)
    target_present = bool(target_column) and target_column in available_columns

    if not target_column:
        signal.warnings.append("Target column not configured.")
    elif not target_present:
        signal.warnings.append(f"Target column '{target_column}' not found in preview data.")

    if target_present:
        target_series = frame[target_column]
        signal.target_dtype = str(target_series.dtype)
        target_missing = int(target_series.isna().sum())
        signal.target_missing_count = target_missing
        if total_rows > 0:
            signal.target_missing_percentage = (target_missing / total_rows) * 100.0
        if target_missing:
            signal.notes.append(f"Target column includes {target_missing} missing row(s).")
    else:
        signal.target_missing_count = 0
        signal.target_missing_percentage = None

    return target_present, available_columns


def _resolve_feature_sets(
    available_columns: Sequence[str],
    target_column: str,
    configured_features: Sequence[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    if configured_features:
        resolved = [
            column
            for column in configured_features
            if column in available_columns and column != target_column
        ]
        missing = [column for column in configured_features if column not in available_columns]
        auto_included: List[str] = []
        excluded = [
            column
            for column in available_columns
            if column not in resolved and column != target_column
        ]
    else:
        resolved = [column for column in available_columns if column != target_column]
        auto_included = list(resolved)
        missing = []
        excluded = []

    return resolved, auto_included, missing, excluded


def _apply_feature_metadata(
    signal: FeatureTargetSplitNodeSignal,
    resolved: Sequence[str],
    auto_included: Sequence[str],
    missing: Sequence[str],
    excluded: Sequence[str],
) -> None:
    signal.feature_columns = [str(column) for column in resolved]
    signal.auto_included_columns = [str(column) for column in auto_included]
    signal.excluded_columns = [str(column) for column in excluded]
    signal.missing_feature_columns = [str(column) for column in missing]

    if not resolved:
        signal.warnings.append(
            "No feature columns resolved. Select feature columns explicitly or ensure the dataset "
            "contains columns other than the target."
        )

    if missing:
        preview = ", ".join(missing[:3])
        if len(missing) > 3:
            preview = f"{preview}, ..."
        signal.warnings.append(f"Configured feature column(s) not found: {preview}.")

    if auto_included:
        signal.notes.append(f"Auto-selected {len(auto_included)} feature column(s).")

    if excluded and signal.configured_feature_columns:
        preview = ", ".join(excluded[:3])
        if len(excluded) > 3:
            preview = f"{preview}, ..."
        signal.notes.append(
            f"Excluded {len(excluded)} column(s) from the feature matrix: {preview}."
        )


def _calculate_feature_missing_counts(
    frame: pd.DataFrame,
    resolved_features: Iterable[str],
) -> Dict[str, int]:
    return {
        str(column): int(frame[column].isna().sum())
        for column in resolved_features
        if column in frame.columns
    }


def _compose_summary(
    target_column: str,
    target_present: bool,
    resolved_features: Sequence[str],
    auto_included: Sequence[str],
    missing_features: Sequence[str],
    excluded_columns: Sequence[str],
    configured_features: Sequence[str],
    target_missing_count: int,
    total_rows: int,
) -> str:
    summary_parts: List[str] = ["Feature/target split"]
    if target_column:
        summary_parts.append(
            f"target '{target_column}'" if target_present else f"target '{target_column}' missing"
        )
    else:
        summary_parts.append("target not set")
    summary_parts.append(f"features={len(resolved_features)}")
    if auto_included:
        summary_parts.append(f"auto={len(auto_included)}")
    if missing_features:
        summary_parts.append(f"missing={len(missing_features)}")
    if excluded_columns and configured_features:
        summary_parts.append(f"excluded={len(excluded_columns)}")
    if target_missing_count:
        summary_parts.append(f"target-missing={target_missing_count}")
    summary_parts.append(f"rows={total_rows}")
    return "; ".join(summary_parts)


def _ordered_columns(
    frame: pd.DataFrame,
    resolved_features: Sequence[str],
    target_column: str,
    target_present: bool,
) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []

    for column in resolved_features:
        if column in frame.columns and column not in seen:
            ordered.append(column)
            seen.add(column)

    if target_present and target_column not in seen:
        ordered.append(target_column)
        seen.add(target_column)

    for column in frame.columns:
        if column not in seen:
            ordered.append(column)
            seen.add(column)

    return ordered


def apply_feature_target_split(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, FeatureTargetSplitNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None
    data = node.get("data") or {}
    config = data.get("config") or {}

    target_column = _normalize_target_column(config.get("target_column") or config.get("target"))
    configured_features = _normalize_feature_columns(config.get("feature_columns"))

    signal = FeatureTargetSplitNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        target_column=target_column or None,
        configured_feature_columns=[str(column) for column in configured_features],
    )

    total_rows = int(frame.shape[0])
    signal.total_rows = total_rows
    signal.preview_row_count = total_rows

    if frame.empty:
        summary = "Separate features & target: no data available"
        return frame.copy(), summary, signal

    target_present, available_columns = _update_target_metadata(
        frame,
        target_column,
        total_rows,
        signal,
    )

    resolved_features, auto_included, missing_features, excluded_columns = _resolve_feature_sets(
        available_columns,
        target_column,
        configured_features,
    )

    _apply_feature_metadata(
        signal,
        resolved_features,
        auto_included,
        missing_features,
        excluded_columns,
    )

    signal.feature_missing_counts = _calculate_feature_missing_counts(frame, resolved_features)

    summary = _compose_summary(
        target_column,
        target_present,
        resolved_features,
        auto_included,
        missing_features,
        excluded_columns,
        configured_features,
        signal.target_missing_count,
        total_rows,
    )

    ordered_columns = _ordered_columns(frame, resolved_features, target_column, target_present)
    result_frame = frame.loc[:, ordered_columns].copy()

    return result_frame, summary, signal


__all__ = ["apply_feature_target_split"]
