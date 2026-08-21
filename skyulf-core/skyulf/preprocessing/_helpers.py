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

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

from ..engines import EngineName, SkyulfDataFrame, get_engine
from ..utils import resolve_columns


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


def resolve_columns_then_to_pandas(
    X: Any,
    config: dict[str, Any],
    default_selection_func: Callable[[Any], list[str]] | None = None,
    target_column_key: str = "target_column",
) -> tuple[pd.DataFrame, list[str]]:
    """Resolve the columns to process natively, then convert only that subset to pandas.

    ``resolve_columns``/``detect_numeric_columns`` already work directly on raw
    Polars frames, so column resolution doesn't require a conversion. Many fit
    routines are sklearn/pandas-bound for the actual math, but converting only
    the selected columns instead of the full input frame avoids paying for
    unrelated columns on wide frames (large win when few columns of many are
    selected, neutral when most/all columns are selected).
    """
    columns = resolve_columns(X, config, default_selection_func, target_column_key)
    if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
        select_cols = [c for c in columns if c in X.columns]
        X = (X.select(select_cols) if select_cols else X).to_pandas()
    else:
        X = to_pandas(X)
    return X, columns


def resolve_columns_then_to_numpy(
    X: Any,
    config: dict[str, Any],
    default_selection_func: Callable[[Any], list[str]] | None = None,
    target_column_key: str = "target_column",
) -> tuple[np.ndarray, list[str]]:
    """Resolve columns natively, then convert only that subset straight to numpy.

    Prefer this over ``resolve_columns_then_to_pandas`` whenever the caller's
    only use for the converted frame is immediately handing it to sklearn (or
    another numpy-based consumer) with no Pandas-only step (no ``errors="coerce"``
    NaN handling, no ``.quantile()``/interpolation semantics, no indexed
    ``pd.Series`` masking) in between. sklearn estimators accept numpy arrays
    directly, so Polars frames can skip the Pandas hop entirely: Polars
    ``.select(cols).to_numpy()`` is native, no Pandas involved. Pandas inputs
    still go through ``DataFrame.to_numpy()`` (also native, no extra copy vs.
    the old ``pandas -> pandas -> numpy`` path).
    """
    columns = resolve_columns(X, config, default_selection_func, target_column_key)
    if not columns:
        return np.empty((0, 0)), columns
    if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
        select_cols = [c for c in columns if c in X.columns]
        X_np = X.select(select_cols).to_numpy() if select_cols else np.empty((0, 0))
    else:
        subset = X[columns]
        # Nullable extension dtypes (Int64, Float64...) to_numpy() as object
        # arrays full of pd.NA, which crash sklearn (F-10). Force the
        # float64/NaN representation the Polars path produces natively.
        if any(isinstance(dt, pd.api.extensions.ExtensionDtype) for dt in subset.dtypes):
            X_np = subset.to_numpy(dtype="float64", na_value=np.nan)
        else:
            X_np = subset.to_numpy()
    return X_np, columns


def select_then_to_pandas(X: Any, requested: Iterable[str]) -> pd.DataFrame:
    """Narrow to ``requested`` columns natively (if Polars), then convert to pandas.

    For fit routines that validate/consume a small, explicitly-named set of
    columns (e.g. lat/lon pairs, an explicit interaction/polynomial column
    list) rather than an auto-detected set. Column existence isn't
    required here — validation of missing columns happens after conversion,
    so error messages stay identical to full-frame-conversion behavior.
    """
    if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
        select_cols = resolve_valid_columns(X, requested)
        return (X.select(select_cols) if select_cols else X).to_pandas()
    return to_pandas(X)


def select_then_to_numpy(X: Any, requested: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    """Narrow to ``requested`` columns (filtering out missing ones), then go straight to numpy.

    Same "small explicitly-named column list" case as ``select_then_to_pandas``,
    but for callers whose only downstream use is a numpy-based consumer (e.g.
    sklearn ``PolynomialFeatures``/``PolynomialFeatures.fit``) with no
    Pandas-only step in between — skips the Pandas hop entirely for Polars
    inputs. Returns the actually-present column list alongside the array so
    callers can keep it in sync with the array's column order.
    """
    valid_cols = resolve_valid_columns(X, requested)
    if not valid_cols:
        return np.empty((0, 0)), []
    if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
        return X.select(valid_cols).to_numpy(), valid_cols
    return X[valid_cols].to_numpy(), valid_cols


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
            c
            for c, t in zip(df.columns, df.dtypes, strict=True)
            if t in [pl.Utf8, pl.Categorical, pl.Object]
        ]
    return list(df.select_dtypes(include=["object", "string", "category"]).columns)


def auto_detect_numeric_columns(df: pd.DataFrame | SkyulfDataFrame) -> list[str]:
    """Return numeric columns from either a Pandas or Polars frame."""
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        from ..engines import POLARS_NUMERIC_DTYPES

        return [c for c, t in zip(df.columns, df.dtypes, strict=True) if t in POLARS_NUMERIC_DTYPES]
    return list(df.select_dtypes(include=["number"]).columns)


def auto_detect_datetime_columns(df: pd.DataFrame | SkyulfDataFrame) -> list[str]:
    """Return datetime/date columns from either a Pandas or Polars frame."""
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        import polars as pl

        return [
            c
            for c, t in zip(df.columns, df.dtypes, strict=True)
            if t in [pl.Date, pl.Datetime] or isinstance(t, pl.Datetime)
        ]
    return list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
