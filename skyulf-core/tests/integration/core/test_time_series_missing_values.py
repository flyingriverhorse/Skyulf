"""Time-series features share missing-value semantics without altering source data."""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.preprocessing.time_series.lag import LagFeaturesApplier, LagFeaturesCalculator
from skyulf.preprocessing.time_series.rolling import (
    RollingAggregateApplier,
    RollingAggregateCalculator,
)

_ENGINES = ["pandas", "pandas_wrapper", "polars", "polars_wrapper"]
_AGGREGATIONS = ["mean", "sum", "min", "max", "median", "std"]


def _frame(engine: str, data: dict[str, list], dtype: str = "float64") -> Any:
    """Build real NaN-bearing frames without converting NaN to Polars null."""
    if engine.startswith("polars"):
        frame = pl.DataFrame(data).with_columns(
            pl.col("x").cast(pl.Float32 if dtype == "float32" else pl.Float64)
        )
        return SkyulfPolarsWrapper(frame) if engine.endswith("wrapper") else frame
    frame_pd = pd.DataFrame(data).astype({"x": dtype})
    frame_pd.index = pd.Index([7] * len(frame_pd))
    return SkyulfPandasWrapper(frame_pd) if engine.endswith("wrapper") else frame_pd


def _native(frame: Any) -> Any:
    """Unwrap public frame adapters for source-value and dtype assertions."""
    return frame.to_native() if hasattr(frame, "to_native") else frame


def _target(engine: str, values: list[int], target_kind: str) -> Any:
    """Supply supported target containers with deliberately duplicated pandas indices."""
    if target_kind == "frame":
        return None
    if target_kind == "list":
        return values
    if target_kind == "array":
        return np.array(values)
    if engine.startswith("polars"):
        return pl.Series("target", values)
    return pd.Series(values, index=[2] * len(values), name="target")


def _assert_source_columns(actual: Any, expected: Any) -> None:
    """Missing-value normalization must leave original values and dtypes intact."""
    actual, expected = _native(actual), _native(expected)
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual[list(expected.columns)], expected)
    else:
        assert_frame_equal(actual.select(expected.columns), expected)


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("aggregation", _AGGREGATIONS)
@pytest.mark.parametrize("min_periods", [1, 2])
def test_rolling_ignores_float_nan_without_changing_source(engine, aggregation, min_periods):
    """NaN and null must count as missing observations in every rolling aggregation."""
    data = {"x": [1.0, np.nan, 3.0, None, 5.0, 7.0, np.nan]}
    X = _frame(engine, data)
    expected = {
        "mean": [1, 1, 3, 3, 5, 6, 7],
        "sum": [1, 1, 3, 3, 5, 12, 7],
        "min": [1, 1, 3, 3, 5, 5, 7],
        "max": [1, 1, 3, 3, 5, 7, 7],
        "median": [1, 1, 3, 3, 5, 6, 7],
        "std": [np.nan, np.nan, np.nan, np.nan, np.nan, np.sqrt(2), np.nan],
    }[aggregation]
    if min_periods == 2:
        expected = [np.nan] * 5 + [expected[5], np.nan]
    artifact = RollingAggregateCalculator().fit(
        X,
        {"columns": ["x"], "window": 2, "min_periods": min_periods, "aggregations": [aggregation]},
    )

    result = RollingAggregateApplier().apply(X, artifact)

    np.testing.assert_allclose(
        _native(result)[f"x_roll_{aggregation}_2"].to_numpy(), expected, equal_nan=True
    )
    _assert_source_columns(result, _frame(engine, data))
    _assert_source_columns(X, _frame(engine, data))


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("target_kind", ["frame", "list", "array", "series"])
def test_grouped_rolling_skips_nan_after_sorting_and_preserves_targets(engine, target_kind):
    """Rolling windows must skip NaN within each sorted group without moving target pairs."""
    data = {
        "time": list(range(1, 9)),
        "group": ["a", "b"] * 4,
        "x": [1.0, 10.0, np.nan, 20.0, 3.0, np.nan, 5.0, 40.0],
        "date": [date(2024, 1, 1)] * 8,
    }
    order = [4, 1, 7, 0, 6, 3, 5, 2]
    shuffled = {key: [values[i] for i in order] for key, values in data.items()}
    X = _frame(engine, shuffled, "float32")
    y = _target(engine, [100 * (i + 1) for i in order], target_kind)
    payload = X if y is None else (X, y)
    artifact = RollingAggregateCalculator().fit(
        payload,
        {
            "columns": ["x"],
            "window": 2,
            "min_periods": 1,
            "aggregations": _AGGREGATIONS,
            "group_by": ["group"],
            "sort_by": "time",
        },
    )

    result = RollingAggregateApplier().apply(payload, artifact)
    output = result if y is None else result[0]

    expected = {
        "mean": [1, 10, 1, 15, 3, 20, 4, 40],
        "sum": [1, 10, 1, 30, 3, 20, 8, 40],
        "min": [1, 10, 1, 10, 3, 20, 3, 40],
        "max": [1, 10, 1, 20, 3, 20, 5, 40],
        "median": [1, 10, 1, 15, 3, 20, 4, 40],
        "std": [np.nan, np.nan, np.nan, np.sqrt(50), np.nan, np.nan, np.sqrt(2), np.nan],
    }
    for aggregation, values in expected.items():
        np.testing.assert_allclose(
            _native(output)[f"x_roll_{aggregation}_2"].to_numpy(), values, equal_nan=True
        )
    if y is not None:
        np.testing.assert_array_equal(np.asarray(result[1]), np.arange(100, 900, 100))
        np.testing.assert_array_equal(np.asarray(y), [100 * (i + 1) for i in order])
    _assert_source_columns(output, _frame(engine, data, "float32"))
    _assert_source_columns(X, _frame(engine, shuffled, "float32"))


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("with_y", [False, True])
def test_lag_drop_na_removes_rows_missing_source_or_lag(engine, with_y):
    """A missing source or shifted NaN must remove the row even when no rows survive."""
    X = _frame(engine, {"x": [1.0, np.nan, 3.0]})
    payload = (X, [10, 20, 30]) if with_y else X
    artifact = LagFeaturesCalculator().fit(
        payload, {"columns": ["x"], "lags": [1], "drop_na": True}
    )

    result = LagFeaturesApplier().apply(payload, artifact)
    output = result[0] if with_y else result

    assert list(output.columns) == ["x", "x_lag_1"]
    assert len(output) == 0
    if with_y:
        assert result[1] == []


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("target_kind", ["frame", "list", "array", "series"])
@pytest.mark.parametrize("drop_na", [False, True])
def test_grouped_lag_filters_nan_and_null_after_sorting(engine, target_kind, drop_na):
    """The keep mask must include all missing features and preserve sorted X/y pairs."""
    data = {
        "time": list(range(1, 13)),
        "group": ["a", "b"] * 5 + ["a", "a"],
        "x": [1.0, 10.0, np.nan, 20.0, 3.0, 30.0, 5.0, 40.0, 7.0, 50.0, None, 9.0],
        "other": [1.0] * 5 + [np.nan] + [1.0] * 6,
        "text": ["ok"] * 7 + [None] + ["ok"] * 4,
        "date": [date(2024, 1, 1)] * 8 + [None] + [date(2024, 1, 1)] * 3,
    }
    order = [4, 1, 7, 0, 6, 3, 11, 5, 9, 2, 10, 8]
    shuffled = {key: [values[i] for i in order] for key, values in data.items()}
    X = _frame(engine, shuffled, "float32")
    y = _target(engine, [100 * (i + 1) for i in order], target_kind)
    payload = X if y is None else (X, y)
    artifact = LagFeaturesCalculator().fit(
        payload,
        {
            "columns": ["x"],
            "lags": [1],
            "group_by": ["group"],
            "sort_by": "time",
            "drop_na": drop_na,
        },
    )

    result = LagFeaturesApplier().apply(payload, artifact)
    output = result if y is None else result[0]

    positions = [3, 6, 9] if drop_na else list(range(12))
    expected_lags = [np.nan, np.nan, 1, 10, np.nan, 20, 3, 30, 5, 40, 7, np.nan]
    expected_data = {key: [values[i] for i in positions] for key, values in data.items()}
    _assert_source_columns(output, _frame(engine, expected_data, "float32"))
    np.testing.assert_allclose(
        _native(output)["x_lag_1"].to_numpy(), [expected_lags[i] for i in positions], equal_nan=True
    )
    if y is not None:
        np.testing.assert_array_equal(np.asarray(result[1]), [100 * (i + 1) for i in positions])
        np.testing.assert_array_equal(np.asarray(y), [100 * (i + 1) for i in order])
    _assert_source_columns(X, _frame(engine, shuffled, "float32"))


@pytest.mark.parametrize("wrapped", [False, True], ids=["native", "wrapped"])
def test_lag_drop_na_preserves_zero_column_polars_frame(wrapped):
    """An empty feature frame must remain a no-op when configured lag columns are absent."""
    X: Any = pl.DataFrame()
    if wrapped:
        X = SkyulfPolarsWrapper(X)
    artifact = LagFeaturesCalculator().fit(
        X, {"columns": ["missing"], "lags": [1], "drop_na": True}
    )

    result = LagFeaturesApplier().apply(X, artifact)

    assert type(result) is type(X)
    assert result.shape == (0, 0)
    assert_frame_equal(_native(result), _native(X))
