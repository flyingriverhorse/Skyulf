"""Bounded, detached samples of pipeline nodes' resolved preview artifacts."""

import json
import math
import reprlib
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from skyulf.data.dataset import SplitDataset

MAX_ROWS = 50
MAX_COLUMNS = 100
MAX_CELL_CHARACTERS = 500
MAX_SAMPLE_BYTES = 256 * 1024
MAX_RUN_SAMPLE_BYTES = 8 * 1024 * 1024
SPLITS = ("train", "test", "validation")


def unavailable_side(reason: str, *, error: bool = False) -> dict[str, Any]:
    """Describe missing measured data without fabricating an empty table."""
    return {"status": "error" if error else "unavailable", "reason": reason, "tables": []}


@dataclass
class NodeInspectionCapture:
    """Keep one node's samples independent of mutable artifacts within its assigned budget."""

    node_id: str
    sample_budget: int = MAX_SAMPLE_BYTES
    input: dict[str, Any] = field(
        default_factory=lambda: unavailable_side("The node did not execute in this preview.")
    )
    output: dict[str, Any] = field(
        default_factory=lambda: unavailable_side("The node did not execute in this preview.")
    )
    input_resolved: bool = False


def _table_parts(value: Any, port: str) -> list[tuple[Any, str, str | None]]:
    """Flatten only the supported split and feature/target containers, at most six tables."""
    if isinstance(value, SplitDataset):
        parts = [(split, getattr(value, split)) for split in SPLITS]
    elif isinstance(value, dict) and any(split in value for split in SPLITS):
        parts = [(split, value.get(split)) for split in SPLITS]
    else:
        parts = [(None, value)]

    tables = []
    for split, part in parts:
        if part is None:
            continue
        if isinstance(part, tuple) and len(part) == 2:
            tables.extend([(part[0], "X", split), (part[1], "y", split)])
        else:
            tables.append((part, port, split))
    return tables


def _as_frame(value: Any, port: str) -> pd.DataFrame | pl.DataFrame | None:
    """Recognize native frames, wrappers and one/two-dimensional feature or target arrays."""
    if isinstance(value, (pd.DataFrame, pl.DataFrame)):
        return value
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        value = to_native()
        if isinstance(value, (pd.DataFrame, pl.DataFrame)):
            return value
    if isinstance(value, pd.Series):
        return value.to_frame(name=value.name if value.name is not None else port)
    if isinstance(value, pl.Series):
        return value.to_frame()
    if isinstance(value, np.ndarray) and value.ndim in (1, 2):
        if value.ndim == 1:
            return pd.DataFrame({port: value})
        return pd.DataFrame(value)
    return None


def _cell(value: Any, limit: int) -> tuple[Any, bool]:
    """Return a JSON-safe scalar or bounded display string, detached from source cells."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None, False
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool | int):
        return value, False
    if isinstance(value, float):
        return (value if math.isfinite(value) else None), False
    if isinstance(value, (datetime, date, time)):
        value = value.isoformat()
    elif isinstance(value, (timedelta, Decimal)):
        value = str(value)
    if isinstance(value, str):
        return (value[: limit - 1] + "…", True) if len(value) > limit else (value, False)

    # reprlib bounds common nested containers before a display string is built.
    # Arbitrary object repr methods are not needed for data inspection.
    if isinstance(value, (list, tuple, dict, set, frozenset, bytes)):
        display = reprlib.repr(value)
    else:
        display = f"<{type(value).__name__}>"
    return display[:limit], True


def _snapshot_table(
    frame: pd.DataFrame | pl.DataFrame, port: str, split: str | None, budget: int
) -> dict[str, Any]:
    """Sample a frame by position before serialization, retaining its actual preview shape."""
    row_count, column_count = frame.shape
    sample = (
        frame.iloc[:MAX_ROWS, :MAX_COLUMNS]
        if isinstance(frame, pd.DataFrame)
        else frame.head(MAX_ROWS)[:, :MAX_COLUMNS]
    )
    names = [str(column) for column in sample.columns]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate display column names cannot be represented as row records.")
    columns = [
        {"name": name, "dtype": str(dtype)}
        for name, dtype in zip(names, sample.dtypes, strict=True)
    ]
    values = (
        sample.itertuples(index=False, name=None)
        if isinstance(sample, pd.DataFrame)
        else sample.iter_rows()
    )
    rows = []
    truncated = row_count > MAX_ROWS or column_count > MAX_COLUMNS
    # Leave room for JSON escaping and row keys even for six wide split tables.
    cell_limit = min(MAX_CELL_CHARACTERS, max(16, budget // (max(1, len(names)) * 8)))
    for cells in values:
        row = {}
        for name, value in zip(names, cells, strict=True):
            row[name], shortened = _cell(value, cell_limit)
            truncated = truncated or shortened
        size = len(json.dumps(row, separators=(",", ":"), allow_nan=False)) + 1
        if size > budget:
            truncated = True
            break
        rows.append(row)
        budget -= size
    return {
        "port": port,
        "split": split,
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "rows": rows,
        "truncated": truncated,
    }


def snapshot_side(
    value: Any, port: str, *, sample_budget: int = MAX_SAMPLE_BYTES
) -> dict[str, Any]:
    """Capture bounded tables without making inspection failures fail pipeline execution.

    Counts describe the data processed by preview, whose loaders sample up to
    1,000 rows. Each side shares at most 256 KiB for JSON sample rows, separate
    from its bounded column schemas. Bulk inspection may assign a smaller
    allowance so every captured node shares the overall run budget fairly.
    """
    try:
        parts = _table_parts(value, port)
        if not parts:
            return unavailable_side("No data artifact was produced for this side.")
        table_budget = max(0, min(MAX_SAMPLE_BYTES, sample_budget) - 2 * len(parts)) // len(parts)
        tables = []
        for part, table_port, split in parts:
            frame = _as_frame(part, table_port)
            if frame is None:
                return unavailable_side("This artifact shape does not support tabular inspection.")
            tables.append(_snapshot_table(frame, table_port, split, table_budget))
        return {"status": "available", "reason": None, "tables": tables}
    except Exception:  # noqa: BLE001 - inspection must not change execution behavior
        return unavailable_side("The preview sample could not be captured.", error=True)
