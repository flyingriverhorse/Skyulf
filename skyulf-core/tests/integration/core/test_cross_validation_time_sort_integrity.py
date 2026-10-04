"""Chronological CV and tuning must retain every paired observation and feature."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.cross_validation import _sort_by_time, perform_cross_validation
from skyulf.modeling.regression import (
    LinearRegressionApplier,
    LinearRegressionCalculator,
)

_ENGINES = ["pandas", "pandas_wrapper", "polars", "polars_wrapper"]
_SORTED_ROWS = [2, 5, 4, 8, 0, 6, 10, 11, 1, 3, 7, 9]


def _make_data(engine: str, target_kind: str, all_missing: bool = False) -> tuple[Any, Any]:
    """Include duplicate dates/index labels and feature names that collided with y."""
    days = [3, None, 1, None, 2, 1, 3, None, 2, None, 4, 4]
    dates = pd.to_datetime(
        np.array([f"2024-01-{day:02d}" if day else None for day in days], dtype=object)
    )
    frame: Any = pd.DataFrame(
        {
            "time": pd.NaT if all_missing else dates,
            "row_id": list(range(12)),
            "__cv_y__": list(range(20, 8, -1)),
            "target": [i % 3 for i in range(12)],
        },
        index=[8, 8, 3, 9, 1, 3, 6, 2, 9, 4, 0, 0],
    )
    values = list(range(100, 112))
    target: Any = values
    if target_kind == "array":
        target = np.array(values)
    elif target_kind != "list":
        name = "target" if target_kind == "named" else None
        target = pd.Series(values, index=list(range(30, 18, -1)), name=name)
    if engine.startswith("polars"):
        frame = pl.from_pandas(frame)
        if isinstance(target, pd.Series):
            target = pl.Series(str(target.name or ""), target.to_numpy())
        if engine.endswith("wrapper"):
            frame = SkyulfPolarsWrapper(frame)
    elif engine.endswith("wrapper"):
        frame = SkyulfPandasWrapper(frame)
    return frame, target


class _ObservedPreprocessor:
    """Observe real fold inputs while allowing the actual models to fit and score."""

    def __init__(self) -> None:
        """Keep separate training and held-out observations for chronology checks."""
        self.training: list[tuple[Any, Any]] = []
        self.validation: list[tuple[Any, Any]] = []

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Record each training fold and final refit without changing its data."""
        self.training.append((X, y))
        return X, y

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Record the held-out observations paired with each training fold."""
        self.validation.append((X, y))
        return X, y


def _assert_rows(data: tuple[Any, Any], expected_rows: list[int]) -> None:
    """Compare all original feature values and paired targets in the expected order."""
    X, y = data
    assert list(X.columns) == ["row_id", "__cv_y__", "target"]
    np.testing.assert_array_equal(X.to_numpy(), [[row, 20 - row, row % 3] for row in expected_rows])
    np.testing.assert_array_equal(np.asarray(y), [100 + row for row in expected_rows])


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("target_kind", ["list", "array", "unnamed", "named"])
def test_time_series_cv_preserves_rows_features_and_targets(engine, target_kind):
    """Missing dates and target-name collisions must not alter any CV observation."""
    X, y = _make_data(engine, target_kind)
    observed = _ObservedPreprocessor()

    result = perform_cross_validation(
        LinearRegressionCalculator(),
        LinearRegressionApplier(),
        X,
        y,
        config={},
        n_folds=2,
        cv_type="time_series_split",
        time_column="time",
        preprocessing=observed,
    )

    assert len(result["folds"]) == len(observed.training) == len(observed.validation) == 2
    _assert_rows(observed.training[0], _SORTED_ROWS[:4])
    _assert_rows(observed.validation[0], _SORTED_ROWS[4:8])
    _assert_rows(observed.training[1], _SORTED_ROWS[:8])
    _assert_rows(observed.validation[1], _SORTED_ROWS[8:])
    assert list(X.columns) == ["time", "row_id", "__cv_y__", "target"]
    np.testing.assert_array_equal(np.asarray(X["row_id"]), np.arange(12))
    np.testing.assert_array_equal(np.asarray(y), np.arange(100, 112))


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("target_kind", ["list", "unnamed", "named"])
def test_time_series_tuning_refits_every_original_observation(engine, target_kind):
    """The shared time sort must preserve all rows and features through public tuning."""
    X, y = _make_data(engine, target_kind)
    observed = _ObservedPreprocessor()

    model, result = TuningCalculator(LinearRegressionCalculator()).fit(
        X,
        y,
        TuningConfig(
            strategy="grid",
            metric="r2",
            search_space={"fit_intercept": [True]},
            cv_type="time_series_split",
            cv_folds=2,
            n_jobs=1,
        ),
        preprocessing=observed,
    )

    assert len(observed.training) == 3
    _assert_rows(observed.training[-1], _SORTED_ROWS)
    assert model.n_features_in_ == 3
    assert result.best_score == pytest.approx(1.0)


@pytest.mark.parametrize("engine", _ENGINES)
def test_time_sort_all_missing_dates_keeps_original_order(engine):
    """An entirely missing time key must retain every row once in its original order."""
    X, y = _make_data(engine, "named", all_missing=True)

    sorted_data = _sort_by_time(X, y, "time", None, logging.getLogger(__name__))

    _assert_rows(sorted_data, list(range(12)))


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("target_kind", ["list", "tuple"])
@pytest.mark.parametrize("target_length", ["short", "long"])
def test_time_series_cv_rejects_mismatched_targets(engine, target_kind, target_length):
    """Sorting must reject unmatched rows instead of discarding or duplicating targets."""
    X, y = _make_data(engine, "list")
    y = y[:-1] if target_length == "short" else [*y, 112]
    if target_kind == "tuple":
        y = tuple(y)

    with pytest.raises(ValueError, match="same number of rows"):
        perform_cross_validation(
            LinearRegressionCalculator(),
            LinearRegressionApplier(),
            X,
            y,
            config={},
            n_folds=2,
            cv_type="time_series_split",
            time_column="time",
        )
