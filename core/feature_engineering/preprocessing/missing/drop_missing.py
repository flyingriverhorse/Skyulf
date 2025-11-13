"""Feature engineering helpers for dropping missing data."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from core.feature_engineering.schemas import (
    DropMissingColumnsNodeSignal,
    DropMissingRowsNodeSignal,
)


def apply_drop_missing_columns(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, DropMissingColumnsNodeSignal]:
    """Remove columns whose missingness meets configured criteria."""
    data = node.get("data") or {}
    config = data.get("config") or {}
    threshold = config.get("missing_threshold")
    configured_columns = config.get("columns") or []

    try:
        threshold_value = float(threshold) if threshold is not None else None
    except (TypeError, ValueError):
        threshold_value = None

    total_columns_before = int(frame.shape[1])
    requested_columns = [str(column) for column in configured_columns if str(column)]
    drop_candidates = {
        str(column)
        for column in configured_columns
        if column in frame.columns
    }
    auto_dropped: List[str] = []

    if threshold_value is not None and not frame.empty:
        missing_pct = frame.isna().mean().mul(100)
        auto_dropped = [col for col, value in missing_pct.items() if value >= threshold_value]
        drop_candidates.update(auto_dropped)

    if not drop_candidates:
        signal = DropMissingColumnsNodeSignal(
            columns=[],
            removed_columns=[],
            requested_columns=requested_columns,
            auto_detected_columns=[],
            removed_count=0,
            total_columns_before=total_columns_before,
            total_columns_after=total_columns_before,
            threshold=threshold_value,
        )
        return frame, "Drop columns: no columns met the criteria", signal

    removable_columns = [col for col in drop_candidates if col in frame.columns]
    remaining_frame = frame.drop(columns=removable_columns, errors="ignore")
    total_columns_after = int(remaining_frame.shape[1])
    summary_parts = [f"Drop columns: removed {len(removable_columns)} column(s)"]
    if auto_dropped:
        summary_parts.append(f"threshold auto-drop ({len(auto_dropped)})")

    signal = DropMissingColumnsNodeSignal(
        columns=sorted(removable_columns),
        removed_columns=sorted(removable_columns),
        requested_columns=requested_columns,
        auto_detected_columns=sorted(auto_dropped),
        removed_count=len(removable_columns),
        total_columns_before=total_columns_before,
        total_columns_after=total_columns_after,
        threshold=threshold_value,
    )
    return remaining_frame, "; ".join(summary_parts), signal


def apply_drop_missing_rows(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, DropMissingRowsNodeSignal]:
    """Remove rows whose missingness meets configured criteria."""
    if frame.empty:
        signal = DropMissingRowsNodeSignal(
            removed_rows=0,
            removed_count=0,
            total_rows_before=0,
            total_rows_after=0,
            drop_if_any_missing=False,
            threshold=None,
        )
        return frame, "Drop rows: no rows available", signal

    data = node.get("data") or {}
    config = data.get("config") or {}

    drop_if_any_missing = bool(config.get("drop_if_any_missing"))
    threshold = config.get("missing_threshold")

    try:
        threshold_value = float(threshold) if threshold is not None else None
    except (TypeError, ValueError):
        threshold_value = None

    total_rows_before = int(frame.shape[0])

    if drop_if_any_missing:
        filtered_frame = frame.dropna(axis=0, how="any")
        dropped_count = int(frame.shape[0] - filtered_frame.shape[0])
        summary = (
            f"Drop rows: removed {dropped_count} row(s) with any missing values"
            if dropped_count
            else "Drop rows: no rows contained missing values"
        )
        signal = DropMissingRowsNodeSignal(
            removed_rows=dropped_count,
            removed_count=dropped_count,
            total_rows_before=total_rows_before,
            total_rows_after=int(filtered_frame.shape[0]),
            drop_if_any_missing=True,
            threshold=None,
        )
        return filtered_frame, summary, signal

    if threshold_value is None:
        signal = DropMissingRowsNodeSignal(
            removed_rows=0,
            removed_count=0,
            total_rows_before=total_rows_before,
            total_rows_after=total_rows_before,
            drop_if_any_missing=False,
            threshold=None,
        )
        return frame, "Drop rows: threshold not configured", signal

    missing_pct = frame.isna().mean(axis=1).mul(100)
    mask = missing_pct >= threshold_value
    filtered_frame = frame.loc[~mask].copy()
    dropped_count = int(mask.sum())
    summary = (
        f"Drop rows: removed {dropped_count} row(s) at ≥{threshold_value:.0f}% missing"
        if dropped_count
        else "Drop rows: no rows met the threshold"
    )
    signal = DropMissingRowsNodeSignal(
        removed_rows=dropped_count,
        removed_count=dropped_count,
        total_rows_before=total_rows_before,
        total_rows_after=int(filtered_frame.shape[0]),
        drop_if_any_missing=False,
        threshold=threshold_value,
    )
    return filtered_frame, summary, signal
