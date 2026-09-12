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
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl

from ..engines import (
    POLARS_NUMERIC_DTYPES,
    EngineName,
    PandasBackedFrame,
    PolarsBackedFrame,
    SkyulfDataFrame,
    get_engine,
)
from ..utils import is_decimal_series, resolve_columns
from ._schema import SkyulfSchema


def resolve_valid_columns(X: Any, requested: Iterable[str]) -> list[str]:
    """Filter ``requested`` to columns that actually exist on ``X``.

    Works for any frame exposing ``.columns`` (Pandas, Polars, our wrapper).
    """
    cols = list(X.columns)
    cols_set = set(cols)
    # Order-preserving dedupe: polars `.select` raises DuplicateError on
    # repeated output names where pandas silently duplicated them.
    return [c for c in dict.fromkeys(requested) if c in cols_set]


def promote_configured_columns_to_float64(
    input_schema: SkyulfSchema, config: dict[str, Any]
) -> SkyulfSchema:
    """Promote configured, existing columns to ``float64`` in a schema."""
    if "columns" in config:
        selected = config["columns"]
    else:
        selected = [
            col
            for col in input_schema.column_list()
            if _is_numeric_schema_dtype(input_schema.dtypes.get(col))
        ]
    if not selected:
        return input_schema

    out = input_schema
    for col in selected:
        if col in input_schema.columns and input_schema.dtypes.get(col) != "float64":
            out = out.with_dtype(col, "float64")
    return out


def _is_numeric_schema_dtype(dtype: str | None) -> bool:
    """Return whether an engine-neutral schema dtype is numeric."""
    return bool(dtype) and dtype.lower().startswith(("int", "uint", "float", "decimal"))


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


def decimal_columns_to_float(X: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert selected Decimal object columns for numeric math without mutating input."""
    converted = {
        col: X[col].to_numpy(dtype=np.float64, na_value=np.nan)
        for col in columns
        if is_decimal_series(X[col])
    }
    return X.assign(**converted) if converted else X


def select_rows_by_position(y: Any, positions: Any) -> Any:
    """Return ``y`` restricted and reordered to exactly ``positions``.

    ``positions`` is an integer array-like of row indices into the *original*
    ``y``, so one value serves both row-changing operations: the kept indices
    of a filter and the argsort of a sort. ``X`` and ``y`` are given the same
    ``positions``, which is what keeps them aligned by construction rather than
    by two independently-correct-looking selections.

    Every ``y`` shape the dispatcher accepts is handled, and the input's type is
    preserved. Returning ``y`` untouched for an unrecognised shape is what
    silently desynchronised ``X`` and ``y`` — and
    ``_check_xy_engine_parity`` documents lists and numpy arrays as
    engine-neutral, so both really do reach here on either engine.

    Args:
        y: The paired target. ``None`` passes straight through.
        positions: Integer row positions, or ``None`` when no row changed.

    Returns:
        ``y`` holding exactly the requested rows, in the requested order.

    Raises:
        TypeError: If ``y`` is a shape whose rows cannot be selected here.
    """
    if y is None or positions is None:
        return y
    if isinstance(y, (pl.Series, pl.DataFrame)):
        # ``gather`` accepts a polars Series, a numpy array or a list of ints.
        return y.gather(positions)
    # A polars Series of positions cannot index a pandas frame or a numpy array,
    # and numpy arrays have no ``.to_numpy()``, so normalise for both origins.
    idx = positions.to_numpy() if hasattr(positions, "to_numpy") else np.asarray(positions)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        return y.iloc[idx]
    if isinstance(y, np.ndarray):
        return y[idx]
    if isinstance(y, list):
        return [y[int(i)] for i in idx]
    # Names the function rather than the operation on purpose: a message reading
    # "cannot select rows from y" matches the SELECT ... FROM shape that Bandit
    # and SonarCloud's injection rule look for, and this line gets filed as a
    # string-built query.
    raise TypeError(
        f"Unsupported y type for select_rows_by_position: {type(y).__name__}. "
        "Expected a polars, pandas, numpy or list target (or None)."
    )


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

    Selected pandas Decimal object columns become float64 for numeric math;
    the caller's frame and unselected columns retain their original values.
    """
    columns = resolve_columns(X, config, default_selection_func, target_column_key)
    if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
        select_cols = [c for c in columns if c in X.columns]
        X = (X.select(select_cols) if select_cols else X).to_pandas()
    else:
        X = to_pandas(X)
    return decimal_columns_to_float(X, columns), columns


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

    Decimal columns are converted to float64 before NumPy consumers run, with
    missing values represented as NaN on both engines.
    """
    columns = resolve_columns(X, config, default_selection_func, target_column_key)
    if not columns:
        return np.empty((0, 0)), columns
    if is_polars(X):
        select_cols = [c for c in columns if c in X.columns]
        subset = X.select(select_cols)
        decimal_cols = [
            c
            for c, dt in zip(subset.columns, subset.dtypes, strict=True)
            if isinstance(dt, pl.Decimal)
        ]
        if decimal_cols:
            subset = subset.with_columns(pl.col(decimal_cols).cast(pl.Float64))
        X_np = subset.to_numpy()
    else:
        subset = decimal_columns_to_float(to_pandas(X)[columns], columns)
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
        polars_df = cast(PolarsBackedFrame, df)
        return [
            c
            for c, t in zip(polars_df.columns, polars_df.dtypes, strict=True)
            if t in [pl.Utf8, pl.Categorical, pl.Object]
        ]
    return list(
        cast(PandasBackedFrame, df).select_dtypes(include=["object", "string", "category"]).columns
    )


def auto_detect_numeric_columns(df: pd.DataFrame | SkyulfDataFrame) -> list[str]:
    """Return numeric columns, including Decimals, without cardinality exclusions."""
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        polars_df = cast(PolarsBackedFrame, df)
        return [
            c
            for c, t in zip(polars_df.columns, polars_df.dtypes, strict=True)
            if t in POLARS_NUMERIC_DTYPES or isinstance(t, pl.Decimal)
        ]
    frame = to_pandas(df)
    numeric = set(frame.select_dtypes(include=["number"]).columns)
    return [c for c in frame.columns if c in numeric or is_decimal_series(frame[c])]


def auto_detect_datetime_columns(df: pd.DataFrame | SkyulfDataFrame) -> list[str]:
    """Return datetime/date columns from either a Pandas or Polars frame."""
    engine = get_engine(df)
    if engine.name == EngineName.POLARS:
        polars_df = cast(PolarsBackedFrame, df)
        return [
            c
            for c, t in zip(polars_df.columns, polars_df.dtypes, strict=True)
            if t in [pl.Date, pl.Datetime] or isinstance(t, pl.Datetime)
        ]
    return list(
        cast(PandasBackedFrame, df).select_dtypes(include=["datetime", "datetimetz"]).columns
    )
