import contextlib
import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd

from skyulf.engines import EngineName

from .data.dataset import SplitDataset
from .engines import SkyulfDataFrame, get_engine

logger = logging.getLogger(__name__)


def _data_stats_from_tuple(data: tuple[Any, Any]) -> tuple[int, set[str]]:
    """Extract row count and column set from an (X, y) tuple's X element."""
    rows = 0
    cols: set[str] = set()
    first = cast(Any, data[0])
    if hasattr(first, "shape"):
        rows = int(first.shape[0])
        if hasattr(first, "columns"):
            cols = set(first.columns)
    return rows, cols


def _warn_split_columns_mismatch(
    label: str, split_cols: set[str], train_cols: set[str], reflect_text: str
) -> None:
    """Log a warning when a SplitDataset split's columns differ from train's."""
    if split_cols and split_cols != train_cols:
        logger.warning(
            "get_data_stats: SplitDataset.%s columns %s differ from "
            "SplitDataset.train columns %s; reporting train's column set, "
            "which may not reflect %s actual shape.",
            label,
            sorted(split_cols),
            sorted(train_cols),
            reflect_text,
        )


def _data_stats_from_split_dataset(data: SplitDataset) -> tuple[int, set[str]]:
    """Sum row counts across a SplitDataset's train/test/validation splits.

    Warns if test/validation columns differ from train's (train's columns are
    always the ones reported).
    """
    rows, cols = get_data_stats(data.train)

    r, test_cols = get_data_stats(data.test)
    rows += r
    _warn_split_columns_mismatch("test", test_cols, cols, "test/validation data")

    if data.validation is not None:
        r, val_cols = get_data_stats(data.validation)
        rows += r
        _warn_split_columns_mismatch("validation", val_cols, cols, "validation data")

    return rows, cols


def get_data_stats(
    data: pd.DataFrame | SkyulfDataFrame | tuple[Any, Any] | SplitDataset,
) -> tuple[int, set[str]]:
    """
    Calculates row count and column set for various data structures.
    Supports DataFrame, (X, y) tuple, and SplitDataset.
    """
    # Check for DataFrame-like object (Pandas, Polars, Wrapper)
    if hasattr(data, "shape") and hasattr(data, "columns") and not isinstance(data, tuple):
        payload = cast(Any, data)
        return int(payload.shape[0]), set(payload.columns)
    if isinstance(data, tuple) and len(data) == 2:
        return _data_stats_from_tuple(data)
    if isinstance(data, SplitDataset):
        return _data_stats_from_split_dataset(data)

    return 0, set()


def unpack_pipeline_input(
    data: pd.DataFrame | SkyulfDataFrame | tuple[Any, Any],
) -> tuple[Any, Any | None, bool]:
    """
    Unpacks input which might be a DataFrame or a (X, y) tuple.
    Returns: (X, y, is_tuple)

    `X` is typed `Any` (instead of `pd.DataFrame | SkyulfDataFrame`) because
    callers branch on the runtime engine (pandas / polars / wrapper) and
    ty's narrowing on `pd.DataFrame[0]` would otherwise widen X to
    `Series[Any] | ...`, poisoning every downstream `with_columns` /
    `to_pandas` call. The runtime contract is unchanged.
    """
    if isinstance(data, (pd.DataFrame, SkyulfDataFrame)):
        return data, None, False
    if isinstance(data, tuple):
        return data[0], data[1], True
    return data, None, False


def _pack_polars_output(X: Any, y: Any) -> Any:
    """Re-attach y to a Polars-backed X (wrapped or raw), returning the same wrapper shape as X."""
    import polars as pl

    engine = get_engine(X)

    y_series = y
    if not isinstance(y, (pl.Series, pl.DataFrame)):
        # Try to convert y to Series; let it fail or handle otherwise
        with contextlib.suppress(Exception):
            y_series = pl.Series(y)

    # Helper to get raw df
    raw_df = X
    is_wrapped = False

    if isinstance(X, pl.DataFrame):
        raw_df = X
        is_wrapped = False
    elif hasattr(X, "_df"):
        raw_df = X._df
        is_wrapped = True

    # Merge
    if isinstance(y_series, pl.Series):
        if y_series.name == "":
            y_series = y_series.alias("target")
        result = raw_df.with_columns(y_series)
    elif isinstance(y_series, pl.DataFrame):
        result = raw_df.hstack(y_series.get_columns())
    else:
        result = raw_df.hstack(y_series)

    if is_wrapped:
        return engine.wrap(result)
    return result


def _pack_pandas_output(X: Any, y: Any) -> pd.DataFrame:
    """Re-attach y to a Pandas-backed X, aligning indices and validating matching row counts."""
    X_pd = X.to_pandas() if hasattr(X, "to_pandas") else X
    y_pd = y.to_pandas() if hasattr(y, "to_pandas") else y

    # Concatenating on mismatched indices (e.g. a row-dropping step that
    # updated X but not y) would otherwise silently NaN-pad/duplicate rows
    # instead of raising. A row-count mismatch always indicates a real bug
    # upstream, so fail loudly with a clear message. When counts match but
    # indices differ (the common, benign case), reset both to a shared
    # positional index before concatenating so rows still line up.
    if len(X_pd) != len(y_pd):
        raise ValueError(
            "pack_pipeline_output: X and y have different row counts "
            f"({len(X_pd)} vs {len(y_pd)}); cannot safely reattach y to X. "
            "This usually means a preprocessing step dropped/added rows for "
            "one but not the other."
        )
    if hasattr(X_pd, "index") and hasattr(y_pd, "index") and not X_pd.index.equals(y_pd.index):
        X_pd = X_pd.reset_index(drop=True)
        y_pd = y_pd.reset_index(drop=True)

    return pd.concat([X_pd, y_pd], axis=1)


def pack_pipeline_output(
    X: Any, y: Any | None, was_tuple: bool
) -> pd.DataFrame | SkyulfDataFrame | tuple[Any, Any]:
    """
    Packs output back into a tuple if the input was a tuple and y is present.
    Otherwise, if y is present, concatenates it back to X.

    `X` is typed `Any` because callers pass pandas / polars / wrapper /
    numpy-backed frames interchangeably, and ty can't narrow the exact type
    through the engine dispatch below.
    """
    if was_tuple and y is None:
        # Caller said the input was a tuple but lost y along the way.
        # Surface this so wiring/upstream bugs don't silently degrade the
        # pipeline shape (Bug 9d in merge_system_audit.md).
        logger.warning(
            "pack_pipeline_output: was_tuple=True but y is None; tuple shape lost. "
            "Caller likely fed a (X, None) placeholder upstream."
        )

    if was_tuple and y is not None:
        return (X, y)

    if y is not None:
        # Re-attach y to X
        engine = get_engine(X)

        if getattr(engine, "name", "") == "polars":
            return _pack_polars_output(X, y)

        # Default to Pandas behavior (convert if needed or assume Pandas)
        return _pack_pandas_output(X, y)

    return X


def _is_binary_numeric(series: pd.Series | Any) -> bool:
    """Check if a numeric series contains only 0s and 1s (or close to them)."""
    # Handle Polars Series
    if hasattr(series, "n_unique"):
        unique_vals = series.drop_nulls().unique()
        if len(unique_vals) > 2:
            return False
        return all(np.isclose(val, 0) or np.isclose(val, 1) for val in unique_vals)

    # Pandas Series
    unique_vals = series.dropna().unique()
    if len(unique_vals) > 2:
        return False

    # Check if values are close to 0 or 1
    return all(np.isclose(val, 0) or np.isclose(val, 1) for val in unique_vals)


def _polars_numeric_dtype_cols(frame: Any) -> list[str]:
    """List the columns of a Polars frame whose dtype is numeric (float/int/uint)."""
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
    return [c for c, t in zip(frame.columns, frame.dtypes, strict=True) if t in numeric_dtypes]


def _polars_column_excluded(series: Any, exclude_binary: bool, exclude_constant: bool) -> bool:
    """Check whether a Polars numeric series should be excluded (boolean/empty/binary/constant)."""
    import polars as pl

    if series.dtype == pl.Boolean:
        return True

    valid = series.drop_nulls()
    if valid.is_empty():
        return True
    if exclude_binary and _is_binary_numeric(valid):
        return True
    return bool(exclude_constant and valid.n_unique() < 2)


def _detect_numeric_columns_polars(
    frame: Any, exclude_binary: bool, exclude_constant: bool
) -> list[str]:
    """Find numeric-like columns in a Polars frame, applying binary/constant exclusion rules."""
    detected: list[str] = []

    for col in _polars_numeric_dtype_cols(frame):
        series = frame[col]
        if _polars_column_excluded(series, exclude_binary, exclude_constant):
            continue
        detected.append(col)

    return detected


def _pandas_column_excluded(series: Any, exclude_binary: bool, exclude_constant: bool) -> bool:
    """Check whether a Pandas numeric series should be excluded (boolean/non-numeric/empty/binary/constant)."""
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return True
    # Strict Numeric Check (Align with Polars behavior)
    if not pd.api.types.is_numeric_dtype(dtype):
        return True

    valid = series.dropna()
    if valid.empty:
        return True
    if exclude_binary and _is_binary_numeric(valid):
        return True
    return bool(exclude_constant and valid.nunique() < 2)


def _detect_numeric_columns_pandas(
    frame: pd.DataFrame, exclude_binary: bool, exclude_constant: bool
) -> list[str]:
    """Find numeric-like columns in a Pandas frame, applying binary/constant exclusion rules."""
    detected: list[str] = []
    seen: set[str] = set()

    for column in frame.columns:
        if column in seen:
            continue

        series = frame[column]
        if _pandas_column_excluded(series, exclude_binary, exclude_constant):
            continue

        detected.append(column)
        seen.add(column)

    return detected


def detect_numeric_columns(
    frame: pd.DataFrame | SkyulfDataFrame,
    exclude_binary: bool = True,
    exclude_constant: bool = True,
) -> list[str]:
    """
    Find numeric-like columns.

    Args:
        frame: DataFrame to analyze
        exclude_binary: If True, excludes columns with only 0/1 values.
        exclude_constant: If True, excludes columns with 0 or 1 unique value.
    """
    engine = get_engine(frame)

    if engine.name == EngineName.POLARS:
        return _detect_numeric_columns_polars(frame, exclude_binary, exclude_constant)

    # Convert to Pandas for analysis if not already
    if not isinstance(frame, pd.DataFrame) and hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()

    return _detect_numeric_columns_pandas(frame, exclude_binary, exclude_constant)


def _resolve_explicit_columns(df: pd.DataFrame | SkyulfDataFrame, cols: list[str]) -> list[str]:
    """Filter an explicit column list to those existing in df, deduping while preserving order.

    A duplicated column name would otherwise be processed twice by stateful
    calculators like encoders/scalers.
    """
    seen: set[str] = set()
    deduped = []
    for c in cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _resolve_auto_detected_columns(
    df: pd.DataFrame | SkyulfDataFrame,
    config: dict[str, Any],
    default_selection_func: Callable[[pd.DataFrame | SkyulfDataFrame], list[str]],
    target_column_key: str,
) -> list[str]:
    """Auto-detect columns via default_selection_func, excluding the target column."""
    cols = default_selection_func(df)

    # Exclude target column during auto-detection
    target_col = config.get(target_column_key)
    if target_col and target_col in cols:
        cols = [c for c in cols if c != target_col]

    # Filter for existence (though auto-detect usually returns existing cols)
    return [c for c in cols if c in df.columns]


def resolve_columns(
    df: pd.DataFrame | SkyulfDataFrame,
    config: dict[str, Any],
    default_selection_func: Callable[[pd.DataFrame | SkyulfDataFrame], list[str]] | None = None,
    target_column_key: str = "target_column",
) -> list[str]:
    """
    Resolves the list of columns to process based on configuration and auto-detection.

    Logic:
    1. If 'columns' is explicitly provided in config, use it (filtering for existence in df).
       - Does NOT exclude target column if explicitly requested.
    2. If 'columns' is missing/empty and default_selection_func is provided:
       - Auto-detect columns using the function.
       - Exclude the target column (if specified in config) from this auto-detected list.
    3. Filter to ensure all columns exist in the dataframe.
    """
    cols = config.get("columns")

    # Case 1: Explicit columns provided
    if cols:
        return _resolve_explicit_columns(df, cols)

    # Case 2: Auto-detection
    if default_selection_func:
        return _resolve_auto_detected_columns(df, config, default_selection_func, target_column_key)

    return []


def user_picked_no_columns(config: dict[str, Any]) -> bool:
    """Return True when the caller explicitly passed `columns: []`.

    Many preprocessing UIs let the user multi-select the columns to operate
    on. When every box is unchecked, the user's intent is unambiguously
    "do nothing for this node". `resolve_columns` would otherwise treat an
    empty list as "auto-detect everything", which silently scales /
    encodes / imputes the whole frame — a confusing footgun.

    Calculators that present an explicit column picker should call this at
    the top of `fit()` and short-circuit with `{}` when it returns True.
    Calculators that don't show a picker (e.g. feature-selection nodes
    where empty really does mean "use all eligible") should NOT call this.
    """
    if "columns" not in config:
        return False
    cols = config.get("columns")
    return isinstance(cols, list) and len(cols) == 0
