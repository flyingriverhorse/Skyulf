"""Threshold tuning must score labels from the same transformed validation rows."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import accuracy_score

from skyulf.pipeline import SkyulfPipeline


@pytest.fixture(params=["pandas", "polars"])
def engine(request, monkeypatch):
    """Select the engine explicitly so environment defaults cannot change a case."""
    monkeypatch.setenv("SKYULF_ENGINE", request.param)
    return request.param


def _frame(engine: str, data: dict[str, list]) -> Any:
    """Build raw input on the engine exercised by the current case."""
    return pl.DataFrame(data) if engine == "polars" else pd.DataFrame(data)


def _pipeline(engine: str, steps: list[dict[str, Any]]) -> SkyulfPipeline:
    """Fit a weak linear classifier whose validation decisions expose misalignment."""
    pipeline = SkyulfPipeline(
        {
            "preprocessing": steps,
            "modeling": {"type": "logistic_regression", "params": {"C": 0.001}},
        }
    )
    pipeline.fit(
        _frame(
            engine,
            {"time": list(range(20)), "value": list(range(20)), "target": [0] * 10 + [1] * 10},
        ),
        target_column="target",
    )
    return pipeline


def _rolling_step() -> dict[str, Any]:
    """Use sorting with a finite rolling value on every validation row."""
    return {
        "name": "rolling",
        "transformer": "RollingAggregate",
        "params": {"columns": ["value"], "window": 2, "min_periods": 1, "sort_by": "time"},
    }


@pytest.mark.parametrize("target_shape", ["series", "other_engine_series", "numpy", "list"])
def test_threshold_tuning_aligns_labels_after_sorting(engine, target_shape):
    """Reversing raw rows must not turn correct predictions into one constant class."""
    pipeline = _pipeline(engine, [_rolling_step()])
    X_val = _frame(engine, {"time": list(range(13, 5, -1)), "value": list(range(13, 5, -1))})
    labels = [1] * 4 + [0] * 4
    if target_shape == "series":
        y_val = pl.Series("target", labels) if engine == "polars" else pd.Series(labels)
    elif target_shape == "other_engine_series":
        y_val = pd.Series(labels) if engine == "polars" else pl.Series("target", labels)
    else:
        y_val = np.asarray(labels) if target_shape == "numpy" else labels
    if engine == "pandas":
        # Pairing is positional even when caller indexes differ or repeat.
        X_val.index = [4, 4, 2, 2, 9, 9, 1, 1]
        if isinstance(y_val, pd.Series):
            y_val.index = list(range(30, 38))

    thresholds = pipeline.optimize_thresholds(X_val, y_val, metric=accuracy_score)

    assert thresholds == {0: 0.5, 1: 0.5}
    np.testing.assert_array_equal(
        pipeline.predict(X_val, use_tuned_thresholds=True), [0] * 4 + [1] * 4
    )


@pytest.mark.parametrize("filter_kind", ["lag", "bounds"])
def test_threshold_tuning_filters_labels_with_validation_rows(engine, filter_kind):
    """Rows removed by preprocessing must also leave the labels scored by the metric."""
    if filter_kind == "lag":
        steps = [
            {
                "name": "lag",
                "transformer": "LagFeatures",
                "params": {"columns": ["value"], "lags": [1], "sort_by": "time", "drop_na": True},
            }
        ]
        expected = [0] * 3 + [1] * 4
    else:
        steps = [
            _rolling_step(),
            {
                "name": "bounds",
                "transformer": "ManualBounds",
                "params": {"bounds": {"time": {"lower": 7, "upper": 12}}},
            },
        ]
        expected = [0] * 3 + [1] * 3
    pipeline = _pipeline(engine, steps)
    X_val = _frame(engine, {"time": list(range(13, 5, -1)), "value": list(range(13, 5, -1))})
    seen_labels = []

    def metric(y_true, y_pred):
        """Observe the real optimizer's labels while computing its accuracy."""
        seen_labels.append(np.asarray(y_true).tolist())
        return accuracy_score(y_true, y_pred)

    pipeline.optimize_thresholds(X_val, [1] * 4 + [0] * 4, metric=metric)

    assert seen_labels and all(labels == expected for labels in seen_labels)
    np.testing.assert_array_equal(pipeline.predict(X_val, use_tuned_thresholds=True), expected)


def test_threshold_tuning_without_preprocessing_preserves_input_order(engine):
    """An unchanged validation row set must keep the caller's label order."""
    pipeline = _pipeline(engine, [])
    X_val = _frame(engine, {"time": list(range(13, 5, -1)), "value": list(range(13, 5, -1))})

    thresholds = pipeline.optimize_thresholds(X_val, [1] * 4 + [0] * 4, metric=accuracy_score)

    assert thresholds == {0: 0.5, 1: 0.5}
    np.testing.assert_array_equal(
        pipeline.predict(X_val, use_tuned_thresholds=True), [1] * 4 + [0] * 4
    )


@pytest.mark.parametrize("labels", [[1] * 4 + [0] * 3, [1] * 4 + [0] * 5])
def test_threshold_tuning_rejects_mismatched_raw_lengths(engine, labels):
    """Sorting must not hide extra labels or fail through accidental positional indexing."""
    pipeline = _pipeline(engine, [_rolling_step()])
    X_val = _frame(engine, {"time": list(range(13, 5, -1)), "value": list(range(13, 5, -1))})

    with pytest.raises(ValueError, match="inconsistent numbers of samples|same number of rows"):
        pipeline.optimize_thresholds(X_val, labels, metric=lambda y_true, y_pred: 1.0)

    assert pipeline._tuned_thresholds is None


def test_threshold_tuning_propagates_metric_failure_without_replacing_thresholds(engine):
    """An unsuccessful search must leave previously usable decision thresholds intact."""
    pipeline = _pipeline(engine, [_rolling_step()])
    X_val = _frame(engine, {"time": list(range(13, 5, -1)), "value": list(range(13, 5, -1))})
    y_val = [1] * 4 + [0] * 4
    previous = pipeline.optimize_thresholds(X_val, y_val, metric=accuracy_score)

    def fail_metric(y_true, y_pred):
        """Make the caller's scoring failure visible through the public API."""
        raise RuntimeError("scoring failed")

    with pytest.raises(RuntimeError, match="scoring failed"):
        pipeline.optimize_thresholds(X_val, y_val, metric=fail_metric)

    assert pipeline._tuned_thresholds == previous
