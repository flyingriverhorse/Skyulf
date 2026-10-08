"""Bound and compare local diagnostic frames without retaining sampled values."""

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
from pandas.testing import assert_frame_equal as assert_pandas_equal
from polars.testing import assert_frame_equal as assert_polars_equal

Frame = pd.DataFrame | pl.DataFrame
_SCALARS = (str, bytes, bool, int, float, complex, date, datetime, time, timedelta, Decimal)


class ProbeFailure(ValueError):
    """Carry a stable diagnostic reason without copying private sample values."""


def _immutable_scalar(value: Any) -> bool:
    """Reject shared mutable Python objects that pandas deep copies do not detach."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if type(value) in _SCALARS:
        return True
    if isinstance(value, np.generic):
        return value.dtype.kind in "biufcmMUS"
    return type(value) in (pd.Timestamp, pd.Timedelta)


def _check_pandas_objects(frame: pd.DataFrame) -> None:
    """Check object cells, categorical labels and index labels before copying."""
    for name in frame.columns:
        series = frame[name]
        if isinstance(series.dtype, pd.CategoricalDtype):
            values = series.cat.categories
        elif pd.api.types.is_object_dtype(series.dtype):
            values = series
        else:
            continue
        if not all(_immutable_scalar(value) for value in values):
            raise ValueError("Probe frames require immutable scalar cells.")
    labels = frame.index.to_frame(index=False) if isinstance(frame.index, pd.MultiIndex) else None
    values = labels.to_numpy().flat if labels is not None else frame.index
    if not all(_immutable_scalar(value) for value in values):
        raise ValueError("Probe frames require immutable scalar index labels.")


def validate_frame(frame: Frame, max_rows: int, max_bytes: int) -> None:
    """Enforce input and intermediate frame budgets before retaining a copy."""
    if type(frame) not in (pd.DataFrame, pl.DataFrame):
        raise TypeError("Probe requires a pandas or Polars DataFrame.")
    if len(frame) > max_rows:
        raise ValueError("Probe frame exceeds max_rows; select an explicit sample first.")
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Probe requires unique columns.")
    if isinstance(frame, pd.DataFrame):
        _check_pandas_objects(frame)
        size = int(frame.memory_usage(index=True, deep=True).sum())
    else:
        if any(dtype == pl.Object or dtype.is_nested() for dtype in frame.dtypes):
            raise ValueError("Probe frames require immutable scalar cells.")
        size = frame.estimated_size()
    if size > max_bytes:
        raise ValueError("Probe frame exceeds max_bytes; select an explicit sample first.")


def probe_sizes(chunk_sizes: tuple[int, ...], max_rows: int, max_bytes: int) -> tuple[int, ...]:
    """Always test singleton partitions and reject malformed diagnostic budgets."""
    for value in (max_rows, max_bytes):
        if type(value) is not int or value < 1:
            raise ValueError("Probe budgets must be positive integers.")
    if not isinstance(chunk_sizes, (tuple, list)) or len(chunk_sizes) > 16:
        raise ValueError("Provide at most 16 chunk sizes.")
    if any(type(value) is not int or value < 1 for value in chunk_sizes):
        raise ValueError("Chunk sizes must be positive integers.")
    return tuple(dict.fromkeys((1, *chunk_sizes)))


def copy_frame(frame: Frame) -> Frame:
    """Detach validated local frame storage before invoking trusted custom code."""
    if isinstance(frame, pl.DataFrame):
        return frame.clone()
    result = frame.copy(deep=True)
    # pandas deliberately shares immutable Index storage even for a deep copy.
    # Trusted callbacks can nevertheless mutate its exposed numpy backing arrays.
    result.index = _copy_index(frame.index)
    result.columns = _copy_index(frame.columns)
    for name in result.columns:
        series = result[name]
        if isinstance(series.dtype, pd.CategoricalDtype):
            result[name] = pd.Categorical.from_codes(
                series.cat.codes.to_numpy(copy=True),
                categories=_copy_index(series.cat.categories),
                ordered=series.cat.ordered,
            )
    return result


def _copy_index(index: pd.Index) -> pd.Index:
    """Detach categorical and nested index metadata as well as positional arrays."""
    if isinstance(index, pd.CategoricalIndex):
        categories = pd.Categorical.from_codes(
            index.codes.copy(),
            categories=_copy_index(index.categories),
            ordered=cast(pd.CategoricalDtype, index.dtype).ordered,
        )
        return pd.CategoricalIndex(categories, name=index.name)
    if isinstance(index, pd.MultiIndex):
        # pandas accepts Index levels; its stubs omit this category-preserving form.
        levels: list[Any] = [_copy_index(level) for level in index.levels]
        return pd.MultiIndex(
            levels=levels,
            codes=[code.tolist() for code in index.codes],
            names=index.names,
            sortorder=getattr(index, "sortorder", None),
        )
    return index.copy(deep=True)


def slice_frame(frame: Frame, start: int, size: int) -> Frame:
    """Select positional chunks while retaining the caller's pandas index."""
    return (
        frame.iloc[start : start + size]
        if isinstance(frame, pd.DataFrame)
        else frame.slice(start, size)
    )


def reverse_frame(frame: Frame) -> Frame:
    """Reverse positions without adding a diagnostic feature or target."""
    return frame.iloc[::-1] if isinstance(frame, pd.DataFrame) else frame.reverse()


def assert_same(expected: Frame, actual: Frame, reason: str = "output_mismatch") -> None:
    """Compare engine, ordered schema, nulls, index and values exactly."""
    if type(expected) is not type(actual):
        raise ProbeFailure(reason)
    try:
        if isinstance(expected, pd.DataFrame):
            assert isinstance(actual, pd.DataFrame)
            assert_pandas_equal(expected, actual, check_exact=True, check_index_type=False)
        else:
            assert isinstance(actual, pl.DataFrame)
            assert_polars_equal(expected, actual, check_exact=True)
    except (AssertionError, TypeError, ValueError):
        raise ProbeFailure(reason) from None
