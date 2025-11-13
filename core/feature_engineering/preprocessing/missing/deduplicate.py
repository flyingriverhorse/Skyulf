"""Feature engineering helper for removing duplicate rows."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from core.feature_engineering.schemas import RemoveDuplicatesNodeSignal
from core.feature_engineering.shared.utils import _normalize_remove_duplicates_config


def apply_remove_duplicates(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, RemoveDuplicatesNodeSignal]:
    """Remove duplicate rows based on configured subset and keep strategy."""
    if frame.empty:
        signal = RemoveDuplicatesNodeSignal(
            removed_rows=0,
            total_rows_before=0,
            total_rows_after=0,
            keep="first",
            subset_columns=[],
            missing_columns=[],
        )
        return frame, "Remove duplicates: no data available", signal

    data = node.get("data") or {}
    config = data.get("config") or {}
    normalized = _normalize_remove_duplicates_config(config)

    configured_columns = normalized.columns
    available_columns = set(frame.columns)
    subset_columns = [column for column in configured_columns if column in available_columns]
    missing_columns = [column for column in configured_columns if column not in available_columns]

    before_rows = int(frame.shape[0])
    keep_mode = normalized.keep
    keep_param: Union[str, bool]
    if keep_mode == "none":
        keep_param = False
    else:
        keep_param = keep_mode

    deduped_frame = frame.drop_duplicates(subset=subset_columns or None, keep=keep_param).copy()
    after_rows = int(deduped_frame.shape[0])
    removed_rows = before_rows - after_rows

    summary_parts: List[str] = []
    if removed_rows > 0:
        summary_parts.append(f"Remove duplicates: removed {removed_rows} row(s)")
    else:
        summary_parts.append("Remove duplicates: no duplicate rows removed")

    summary_parts.append(f"keep={keep_mode}")

    if subset_columns:
        preview = ", ".join(subset_columns[:3])
        if len(subset_columns) > 3:
            preview = f"{preview}, …"
        summary_parts.append(f"subset: {preview}")
    else:
        summary_parts.append("subset: all columns")

    if missing_columns:
        preview = ", ".join(missing_columns[:3])
        if len(missing_columns) > 3:
            preview = f"{preview}, …"
        summary_parts.append(f"{len(missing_columns)} column(s) not found ({preview})")

    summary = "; ".join(summary_parts)
    signal = RemoveDuplicatesNodeSignal(
        removed_rows=removed_rows,
        total_rows_before=before_rows,
        total_rows_after=after_rows,
        keep=str(keep_mode),
        subset_columns=subset_columns,
        missing_columns=missing_columns,
    )
    return deduped_frame, summary, signal
