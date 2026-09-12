"""Time-series CV must recognize native dates before fitting fold preprocessing."""

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.modeling.cross_validation import perform_cross_validation
from skyulf.modeling.regression import LinearRegressionApplier, LinearRegressionCalculator

_TEMPORAL_INPUTS = [
    ("pandas", "date"),
    ("pandas_wrapper", "date"),
    ("pandas", "datetime"),
    ("pandas", "datetimetz"),
    ("polars", "date"),
    ("polars_wrapper", "date"),
    ("polars", "datetime"),
]


def _make_data(engine: str, temporal_type: str, days: list[int | None]) -> tuple[Any, Any]:
    """Pair shuffled dates with numeric row identities and independently indexed targets."""
    dates = [date(2024, 1, day) if day is not None else None for day in days]
    frame: Any = pd.DataFrame(
        {
            "time": pd.Series(dates, dtype=object),
            "row_id": range(len(days)),
            "value": [day if day is not None else 99 for day in days],
        }
    )
    target: Any = pd.Series(
        frame["value"].to_numpy() + 100,
        index=list(range(100, 100 + len(days))),
        name="target",
    )
    if temporal_type != "date":
        frame["time"] = pd.to_datetime(frame["time"], utc=temporal_type == "datetimetz")
    frame.index = [i // 2 for i in range(len(days))]
    if engine.startswith("polars"):
        frame = pl.from_pandas(frame)
        target = pl.Series("target", target.to_numpy())
        if engine.endswith("wrapper"):
            frame = SkyulfPolarsWrapper(frame)
    elif engine.endswith("wrapper"):
        frame = SkyulfPandasWrapper(frame)
    return frame, target


class _ObservedPreprocessor:
    """Capture actual fold inputs and remove nonnumeric columns before real model fitting."""

    def __init__(self) -> None:
        """Keep training and held-out rows separate to expose chronology violations."""
        self.training: list[tuple[Any, Any]] = []
        self.validation: list[tuple[Any, Any]] = []

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Record training data before selecting the model's numeric features."""
        self.training.append((X, y))
        return self._numeric_features(X), y

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Record validation data before applying the same feature selection."""
        self.validation.append((X, y))
        return self._numeric_features(X), y

    @staticmethod
    def _numeric_features(X: Any) -> Any:
        """Let CV run successfully even when an unrecognized date key reaches preprocessing."""
        columns = ["row_id", "value"]
        return X.select(columns) if hasattr(X, "select") else X[columns]


def _run_cv(X: Any, y: Any, time_column: str | None = None) -> tuple[Any, Any, list[str]]:
    """Run public CV with a real estimator and observable fold-local preprocessing."""
    observed = _ObservedPreprocessor()
    messages: list[str] = []
    result = perform_cross_validation(
        LinearRegressionCalculator(),
        LinearRegressionApplier(),
        X,
        y,
        config={},
        n_folds=3,
        cv_type="time_series_split",
        time_column=time_column,
        preprocessing=observed,
        log_callback=messages.append,
    )
    return result, observed, messages


@pytest.mark.parametrize(("engine", "temporal_type"), _TEMPORAL_INPUTS)
def test_time_series_cv_auto_detects_native_dates_before_fold_preprocessing(engine, temporal_type):
    """Date object inputs must not train on future rows before validation on earlier dates."""
    X, y = _make_data(engine, temporal_type, [8, 1, 3, 9, 2, 6, 11, 4, 7, 12, 5, 10])

    result, observed, messages = _run_cv(X, y)

    boundaries = [
        (max(train[0]["value"]), min(validation[0]["value"]))
        for train, validation in zip(observed.training, observed.validation, strict=True)
    ]
    assert boundaries == [(3, 4), (6, 7), (9, 10)]
    for X_fold, y_fold in [*observed.training, *observed.validation]:
        assert "time" not in X_fold.columns
        np.testing.assert_array_equal(np.asarray(y_fold), np.asarray(X_fold["value"]) + 100)
    assert result["aggregated_metrics"]["r2"]["mean"] == pytest.approx(1.0)
    assert any("auto-detected datetime column 'time'" in message for message in messages)


@pytest.mark.parametrize(("engine", "temporal_type"), _TEMPORAL_INPUTS)
def test_time_series_cv_date_detection_preserves_ties_missing_rows_and_targets(
    engine, temporal_type
):
    """Native-date detection must retain stable ties and all missing rows with paired targets."""
    X, y = _make_data(engine, temporal_type, [3, None, 1, None, 2, 1, 3, None, 2, None, 4, 4])
    if engine.startswith("pandas") and temporal_type == "date":
        X.iloc[[1, 3, 7, 9], 0] = [None, pd.NA, pd.NaT, np.nan]
    original_target = np.asarray(y).copy()

    result, observed, messages = _run_cv(X, y)

    assert [list(train[0]["row_id"]) for train in observed.training] == [
        [2, 5, 4],
        [2, 5, 4, 8, 0, 6],
        [2, 5, 4, 8, 0, 6, 10, 11, 1],
    ]
    assert [list(validation[0]["row_id"]) for validation in observed.validation] == [
        [8, 0, 6],
        [10, 11, 1],
        [3, 7, 9],
    ]
    for X_fold, y_fold in [*observed.training, *observed.validation]:
        np.testing.assert_array_equal(np.asarray(y_fold), original_target[list(X_fold["row_id"])])
    assert len(result["folds"]) == 3
    assert list(X["row_id"]) == list(range(12))
    np.testing.assert_array_equal(np.asarray(y), original_target)
    assert any("data sorted by 'time'" in message for message in messages)


def test_time_series_cv_uses_first_temporal_column_in_dataframe_order():
    """Earlier native dates must take precedence over later datetime dtypes and skip text."""
    X, y = _make_data("pandas", "date", [8, 1, 3, 9, 2, 6, 11, 4, 7, 12, 5, 10])
    X.insert(0, "text", ["2024-01-01"] * 12)
    X["later_datetime"] = pd.date_range("2024-02-01", periods=12)

    result, observed, messages = _run_cv(X, y)

    assert list(observed.training[0][0]["value"]) == [1, 2, 3]
    assert list(observed.validation[0][0]["value"]) == [4, 5, 6]
    assert "time" not in observed.training[0][0].columns
    assert "later_datetime" in observed.training[0][0].columns
    assert len(result["folds"]) == 3
    assert any("auto-detected datetime column 'time'" in message for message in messages)


@pytest.mark.parametrize(
    "values",
    [
        ["2024-01-03", "2024-01-01", "2024-01-02"],
        [date(2024, 1, 3), "2024-01-01", date(2024, 1, 2)],
        [date(2024, 1, 3), 1, date(2024, 1, 2)],
        [date(2024, 1, 3), datetime(2024, 1, 1), date(2024, 1, 2)],
        [None, pd.NA, pd.NaT],
    ],
    ids=["text_dates", "mixed_text", "mixed_number", "mixed_datetime", "all_missing"],
)
def test_time_series_cv_does_not_infer_unusable_object_columns(values):
    """Object columns without homogeneous dates must retain the documented row-order fallback."""
    X, y = _make_data("pandas", "date", [8, 1, 3, 9, 2, 6, 11, 4, 7, 12, 5, 10])
    X["time"] = np.array(values * 4, dtype=object)

    result, observed, messages = _run_cv(X, y)

    assert [list(train[0]["row_id"]) for train in observed.training] == [
        [0, 1, 2],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
    ]
    assert all("time" in train[0].columns for train in observed.training)
    assert len(result["folds"]) == 3
    assert any("no datetime column found" in message for message in messages)


def test_time_series_cv_explicit_all_missing_object_dates_keeps_rows():
    """An explicit all-missing object key must still be dropped without losing observations."""
    X, y = _make_data("pandas", "date", [None] * 12)
    X["value"] = range(12)
    y = pd.Series(range(100, 112))

    result, observed, messages = _run_cv(X, y, time_column="time")

    assert list(observed.training[-1][0]["row_id"]) == list(range(9))
    assert list(observed.validation[-1][0]["row_id"]) == [9, 10, 11]
    assert all("time" not in train[0].columns for train in observed.training)
    assert result["aggregated_metrics"]["r2"]["mean"] == pytest.approx(1.0)
    assert any("data sorted by 'time'" in message for message in messages)
