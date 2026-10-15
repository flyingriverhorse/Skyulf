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


def renumber_frame(frame: Frame, labels: pd.Index | None = None) -> Frame:
    """Relabel pandas rows with a fresh RangeIndex, as each Spark Arrow batch arrives.

    With ``labels`` the original labels are restored instead. Polars frames
    carry no row labels and are returned unchanged.
    """
    if not isinstance(frame, pd.DataFrame):
        return frame
    if labels is None:
        fresh = pd.RangeIndex(len(frame))
        labels = fresh if not frame.index.equals(fresh) else pd.RangeIndex(1000, 1000 + len(frame))
    return frame.set_axis(labels)


_SAME = object()


def _changed(
    value: Any,
    existing: list[Any],
    categories: list[Any] | None,
    bounds: tuple[int, int] | None,
    text: bool,
) -> Any:
    """Return a noticeably different value of the same type, or _SAME to leave it."""
    if value is None or value is pd.NA or value is pd.NaT:
        return _SAME
    if categories is not None:
        return next((item for item in categories if item != value), _SAME)
    if isinstance(value, str):
        return _changed_text(existing) if text else _SAME
    return _changed_number(value, bounds)


def _changed_text(existing: list[Any]) -> str:
    """Return a text value absent from the sample."""
    text = "__skyulf_perturbed__"
    while text in existing:
        text += "_"
    return text


def _changed_number(value: Any, bounds: tuple[int, int] | None) -> Any:
    """Flip booleans and move numbers by half their size, keeping integer dtypes."""
    if isinstance(value, bool | np.bool_):
        return not value
    if isinstance(value, int | np.integer):
        step = max(1, int(abs(int(value)) * 0.5))
        low, high = bounds or (-(2**63), 2**63 - 1)
        # Move down when moving up would overflow the column's integer type.
        changed = int(value) + step if int(value) + step <= high else int(value) - step
        return min(max(changed, low), high)
    if isinstance(value, float | np.floating):
        return _SAME if np.isnan(value) else value + max(1.0, abs(value)) * 0.5
    return _SAME


def _integer_bounds(dtype: Any) -> tuple[int, int] | None:
    """Return the representable range of an integer dtype, or None for other dtypes."""
    try:
        info = np.iinfo(dtype)
    except (TypeError, ValueError):
        return None
    return int(info.min), int(info.max)


def perturb_row(frame: Frame, position: int, *, text: bool = True) -> Frame:
    """Change one row's inputs noticeably while keeping every column's dtype.

    Numbers move by half their size (at least 1 for integers, staying inside the
    integer type's range), booleans flip, text becomes a value absent from the
    sample (unless ``text`` is false), categoricals switch to another existing
    category, and nulls and other types stay as they are.
    """
    if isinstance(frame, pd.DataFrame):
        return _perturb_pandas(frame, position, text)
    result = frame.clone()
    for name, dtype in frame.schema.items():
        values = frame.get_column(name).to_list()
        categorical = dtype == pl.Categorical or isinstance(dtype, pl.Enum)
        bounds = _integer_bounds(str(dtype).lower()) if dtype.is_integer() else None
        value = _SAME if categorical else _changed(values[position], values, None, bounds, text)
        if value is not _SAME:
            values[position] = value
            result = result.with_columns(pl.Series(name, values, dtype=dtype))
    return result


def _perturb_pandas(frame: pd.DataFrame, position: int, text: bool) -> pd.DataFrame:
    """Perturb one pandas row column by column on a detached copy."""
    result = cast(pd.DataFrame, copy_frame(frame))
    for column in range(frame.shape[1]):
        series = frame.iloc[:, column]
        categories = (
            list(series.cat.categories) if isinstance(series.dtype, pd.CategoricalDtype) else None
        )
        dtype = getattr(series.dtype, "numpy_dtype", series.dtype)
        value = _changed(
            series.iloc[position], series.tolist(), categories, _integer_bounds(dtype), text
        )
        if value is not _SAME:
            result.iloc[position, column] = value
    return result


def without_row(frame: Frame, position: int) -> Frame:
    """Drop one row by position, keeping the other rows' labels and dtypes."""
    if isinstance(frame, pd.DataFrame):
        return frame.iloc[[item for item in range(len(frame)) if item != position]]
    return pl.concat([frame.slice(0, position), frame.slice(position + 1)])


def reverse_frame(frame: Frame) -> Frame:
    """Reverse positions without adding a diagnostic feature or target."""
    if isinstance(frame, pd.DataFrame):
        return frame.iloc[::-1]
    if frame.width == 0:
        # With no values or index to reorder, only height matters; native reverse loses it.
        return frame.clone()
    return frame.reverse()


def assert_same(expected: Frame, actual: Frame, reason: str = "output_mismatch") -> None:
    """Compare engine, ordered schema, nulls, index and values exactly."""
    if type(expected) is not type(actual):
        raise ProbeFailure(reason)
    try:
        if isinstance(expected, pd.DataFrame):
            assert_pandas_equal(
                expected, cast(pd.DataFrame, actual), check_exact=True, check_index_type=False
            )
        else:
            assert_polars_equal(expected, cast(pl.DataFrame, actual), check_exact=True)
    except (AssertionError, TypeError, ValueError):
        raise ProbeFailure(reason) from None
