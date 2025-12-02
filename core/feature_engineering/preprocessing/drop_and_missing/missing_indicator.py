"""Missing value indicator preprocessing helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from core.feature_engineering.schemas import (
    MissingIndicatorAppliedColumnSignal,
    MissingIndicatorNodeSignal,
)


def apply_missing_value_flags(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, MissingIndicatorNodeSignal]:
    """Generate binary indicator columns for configured missing values."""
    node_id = node.get("id") if isinstance(node, dict) else None

    data = node.get("data") or {}
    config = data.get("config") or {}

    raw_columns = config.get("columns")
    if isinstance(raw_columns, list):
        configured_columns = [str(column).strip() for column in raw_columns if str(column).strip()]
    elif isinstance(raw_columns, str):
        configured_columns = [segment.strip() for segment in raw_columns.split(",") if segment.strip()]
    else:
        configured_columns = []

    suffix_value = config.get("flag_suffix")
    flag_suffix = str(suffix_value).strip() if isinstance(suffix_value, str) else ""
    if not flag_suffix:
        flag_suffix = "_was_missing"

    signal = MissingIndicatorNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
        indicator_suffix=flag_suffix,
    )

    if frame.empty:
        return frame, "Missing value indicator: no data available", signal

    target_columns = [column for column in configured_columns if column in frame.columns]
    missing_columns = [column for column in configured_columns if column not in frame.columns]

    auto_columns: List[str] = []
    if not target_columns:
        auto_columns = [column for column in frame.columns if frame[column].isna().any()]
        target_columns = auto_columns

    signal.auto_detected_columns = list(auto_columns)
    signal.evaluated_columns = list(target_columns)
    signal.missing_columns = list(missing_columns)

    if not target_columns:
        return frame, "Missing value indicator: no eligible columns", signal

    working_frame = frame.copy()
    created_columns: List[str] = []
    skipped_columns: List[str] = []
    total_flagged = 0
    total_rows = int(frame.shape[0])

    for column in target_columns:
        if column not in working_frame.columns:
            continue

        indicator_name = f"{column}{flag_suffix}"
        indicator_series = frame[column].isna().astype(int)
        flagged_count = int(indicator_series.sum())
        total_flagged += flagged_count

        preexisting = indicator_name in frame.columns

        if indicator_name in working_frame.columns:
            working_frame[indicator_name] = indicator_series
        else:
            insert_at = (
                working_frame.columns.get_loc(column) + 1
                if column in working_frame.columns
                else len(working_frame.columns)
            )
            working_frame.insert(insert_at, indicator_name, indicator_series)
            created_columns.append(indicator_name)

        signal.indicators.append(
            MissingIndicatorAppliedColumnSignal(
                source_column=str(column),
                indicator_column=str(indicator_name),
                flagged_rows=flagged_count,
                total_rows=total_rows,
                created=not preexisting,
                overwritten_existing=preexisting,
            )
        )

    signal.total_flagged_rows = total_flagged
    signal.skipped_columns = sorted(set(skipped_columns))

    summary_parts = [f"Missing value indicator: processed {len(target_columns)} column(s)"]
    if created_columns:
        summary_parts.append(f"added {len(created_columns)} indicator column(s)")
    if total_flagged:
        summary_parts.append(f"flagged {total_flagged} row(s)")
    summary = "; ".join(summary_parts)

    return working_frame, summary, signal
