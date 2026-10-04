"""Causal temporal context application that preserves requested rows and fitted state."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ._history_state import bounded_tail, decode_rows, history_columns, ordered_groups, records
from .history import current_history, propose_history


def _validate_forward(previous: list[dict], incoming: list[dict], params: dict) -> None:
    """Reject late/replayed observations instead of silently changing prior predictions."""
    old = ordered_groups(previous, params)
    new = ordered_groups(incoming, params)
    time = params["sort_by"]
    for key, rows in new.items():
        if key in old and rows[0][time] <= old[key][-1][time]:
            raise ValueError(
                "Temporal history requires new observations after the saved entity time; "
                "late, overlapping or replayed rows need an explicit history reset."
            )


def _combined_frame(frame: Any, previous: list[dict], params: dict) -> Any:
    """Promote context and incoming values together without narrowing saved history."""
    columns = history_columns(params)
    if isinstance(frame, pl.DataFrame):
        if not previous:
            return frame.select(columns)
        context = pl.from_pandas(pd.DataFrame(previous, columns=columns))
        context = context.with_columns(
            pl.col(name).cast(frame.schema[name])
            for name in columns
            if context[name].null_count() == len(context)
        )
        return pl.concat([context, frame.select(columns)], how="vertical_relaxed")
    if not previous:
        return frame[columns].copy()
    context = pd.DataFrame(previous, columns=columns)
    for name in columns:
        if context[name].isna().all():
            dtype = frame[name].convert_dtypes().dtype
            context[name] = context[name].astype(dtype)
    return pd.concat([context, frame[columns]], ignore_index=True)


def _restore_output(
    frame: Any, transformed: Any, positions: Any, offset: int, columns: list[str]
) -> Any:
    """Attach derived features in caller order without exposing context rows."""
    order = np.argsort(positions)
    added = [name for name in transformed.columns if name not in columns]
    if isinstance(frame, pl.DataFrame):
        result = transformed[order.tolist()].slice(offset)
        return frame.with_columns([result[name] for name in added])
    result = transformed.iloc[order].iloc[offset:]
    output = frame.copy()
    for name in added:
        output[name] = result[name].array
    return output


def apply_history(frame: Any, target: Any, params: dict, apply: Any) -> tuple[Any, Any]:
    """Calculate on context plus input, proposing a bounded next state after success."""
    training = params.get("_history_training", False)
    saved = [] if training else current_history(params)
    previous = decode_rows(saved, params)
    incoming = records(frame, params)
    _validate_forward(previous, incoming, params)
    combined = _combined_frame(frame, previous, params)
    positions = np.arange(len(combined))
    transformed, positions = apply(combined, positions, params | {"history_mode": "batch"})
    result = _restore_output(frame, transformed, positions, len(previous), history_columns(params))
    if not training:
        propose_history(params, bounded_tail(previous + incoming, params))
    return result, target
