"""Tiny shared helpers for preprocessing nodes.

These exist to remove a handful of patterns that recur across most
Appliers/Calculators after the dual-engine dispatch refactor (see
``temp/preprocessing_dual_engine_split_plan.md``). Keep this module small
and dependency-free — anything heavier belongs in ``utils`` or ``engines``.

Boundary with ``dispatcher.py``:
    * ``dispatcher.py`` owns the *control flow* — ``apply_dual_engine`` picks the
      Polars vs Pandas branch and packs/unpacks the pipeline I/O for a whole node.
    * ``_helpers.py`` owns *leaf utilities* called from inside those branches
      (column resolution, engine predicates like ``is_polars``, ``to_pandas``
      coercion, safe scaling). Helpers never dispatch a full node; the
      dispatcher never implements column-level logic.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from ..engines import EngineName, SkyulfDataFrame, get_engine


def resolve_valid_columns(X: Any, requested: Iterable[str]) -> list[str]:
    """Filter ``requested`` to columns that actually exist on ``X``.

    Works for any frame exposing ``.columns`` (Pandas, Polars, our wrapper).
    """
    cols = list(X.columns)
    cols_set = set(cols)
    return [c for c in requested if c in cols_set]


def safe_scale(scale_arr: np.ndarray) -> np.ndarray:
    """Replace zeros in a scale vector with 1.0 to avoid division by zero.

    Mutates and returns the same array — callers always pass a slice/copy.
    """
    scale_arr[scale_arr == 0] = 1.0
    return scale_arr


def to_pandas(X: Any) -> pd.DataFrame:
    """Coerce a frame to Pandas if it isn't already.

    The dispatcher already does this for ``apply``; expose it for ``fit``
    paths that bypass the dispatcher (e.g. shared subset-selection helpers).
    """
    return X.to_pandas() if hasattr(X, "to_pandas") else X


def is_polars(X: Any) -> bool:
    """Return ``True`` when ``X`` is backed by the Polars engine.

    Centralises the ``engine.name == EngineName.POLARS`` check so node modules
    never branch on the engine inline. Node files should call this (or the
    dual-engine dispatcher) instead of importing ``EngineName`` themselves.
    """
    return get_engine(X).name == EngineName.POLARS


def auto_detect_text_columns(df: pd.DataFrame | SkyulfDataFrame) -> list[str]:
    """Return string-like columns from either a Pandas or Polars frame."""
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        import polars as pl

        return [
            c for c, t in zip(df.columns, df.dtypes) if t in [pl.Utf8, pl.Categorical, pl.Object]
        ]
    return list(df.select_dtypes(include=["object", "string", "category"]).columns)


def auto_detect_numeric_columns(df: pd.DataFrame | SkyulfDataFrame) -> list[str]:
    """Return numeric columns from either a Pandas or Polars frame."""
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        import polars as pl

        numeric_dtypes = [
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ]
        return [c for c, t in zip(df.columns, df.dtypes) if t in numeric_dtypes]
    return list(df.select_dtypes(include=["number"]).columns)


def auto_detect_datetime_columns(df: pd.DataFrame | SkyulfDataFrame) -> list[str]:
    """Return datetime/date columns from either a Pandas or Polars frame."""
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        import polars as pl

        return [
            c
            for c, t in zip(df.columns, df.dtypes)
            if t in [pl.Date, pl.Datetime] or isinstance(t, pl.Datetime)
        ]
    return list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
