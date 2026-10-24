"""Package-internal memory and scalar contracts for pandas/Polars frames."""

import math
import pickle
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


def frame_bytes(frame: pd.DataFrame | pl.DataFrame) -> int:
    """Count the actual frame allocation used for the memory guard."""
    if isinstance(frame, pd.DataFrame):
        return int(frame.memory_usage(index=True, deep=True).sum())
    return int(frame.estimated_size())


def output_scalar(value: Any) -> Any:
    """Convert bounded pandas/NumPy scalars without changing their logical type."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError("Prediction output contains a nonfinite number.")
    return value


def bounded_frame(
    selected: Any,
    columns: tuple[str, ...],
    record_key_columns: tuple[str, ...],
    max_rows: int,
    max_bytes: int,
    *,
    require_unique_keys: bool = True,
    subject: str = "Source increment",
) -> pd.DataFrame:
    """Read bounded rows, optionally deferring key uniqueness until label eligibility.

    Features and predictions require unique keys by default. Label readers can
    defer that check until future revisions are excluded; null keys and transfer
    limits remain invalid in either mode. ``subject`` names the source in limit errors.
    """
    from skyulf.engines.spark_engine import (  # noqa: PLC0415
        normalize_spark_decimals,
        restore_spark_decimal_dtypes,
    )

    selected = selected.select(*columns).orderBy(*record_key_columns).limit(max_rows + 1)
    selected, conversions = normalize_spark_decimals(
        selected, columns=set(columns).difference(record_key_columns)
    )
    records: list[dict[str, Any]] = []
    decoded_bytes = 0
    for row in selected.toLocalIterator():
        if len(records) == max_rows:
            raise ValueError(f"{subject} exceeds max_rows.")
        record = row.asDict(recursive=True) if hasattr(row, "asDict") else dict(row)
        decoded_bytes += len(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
        if decoded_bytes > max_bytes:
            raise ValueError(f"{subject} exceeds max_bytes.")
        records.append(record)
    frame = restore_spark_decimal_dtypes(
        pd.DataFrame.from_records(records, columns=columns), conversions
    )
    if frame_bytes(frame) > max_bytes:
        raise ValueError("Source local frame exceeds max_bytes.")
    if frame.loc[:, list(record_key_columns)].isna().any().any():
        raise ValueError("Source row keys must not be null.")
    if require_unique_keys and frame.duplicated(subset=list(record_key_columns)).any():
        raise ValueError("Source row keys must be globally unique within this increment.")
    return frame
