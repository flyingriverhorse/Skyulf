"""Supervised selectors must preserve array-like targets and reject missing labels."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.feature_selection import (
    ModelBasedSelectionCalculator,
    UnivariateSelectionCalculator,
)


@pytest.fixture(params=["pandas", "polars"])
def frame_factory(request, monkeypatch):
    """Exercise native engines without inheriting the caller's engine override."""
    monkeypatch.setenv("SKYULF_ENGINE", request.param)
    return pd.DataFrame if request.param == "pandas" else pl.DataFrame


@pytest.fixture(params=[UnivariateSelectionCalculator, ModelBasedSelectionCalculator])
def calculator(request):
    """Both supervised selectors share the same target preparation contract."""
    return request.param()


@pytest.mark.parametrize("target_kind", ["list", "ndarray", "series", "polars"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("explicit_task", [False, True])
def test_array_like_target_matches_series(
    frame_factory, calculator, target_kind, task, explicit_task
):
    """Container changes must preserve task inference, numeric values and selected features."""
    values = ["a"] * 6 + ["b"] * 6 if task == "classification" else np.arange(12) + 0.25
    expected_y = pd.Series(values, index=np.arange(12) + 20, name="label")
    original_y = expected_y.copy(deep=True)
    target = {
        "list": expected_y.tolist,
        "ndarray": expected_y.to_numpy,
        "series": lambda: expected_y,
        "polars": lambda: pl.Series("label", expected_y.tolist()),
    }[target_kind]()
    frame = frame_factory({"signal": np.arange(12), "noise": [0, 1, 0] * 4})
    config = {"k": 1, "max_features": 1}
    if explicit_task:
        config["problem_type"] = task
    expected = calculator.fit((frame, expected_y), config)

    actual = calculator.fit((frame, target), config)

    pd.testing.assert_series_equal(expected_y, original_y)
    assert actual == expected
    assert actual["selected_columns"] == ["signal"]


@pytest.mark.parametrize(
    "values,dtype",
    [
        pytest.param(["a", "a", None, "b", "b", "b"], "object", id="none"),
        pytest.param(["a", "a", np.nan, "b", "b", "b"], "object", id="object-nan"),
        pytest.param(["a", "a", pd.NA, "b", "b", "b"], "string", id="string-na"),
        pytest.param(["a", "a", None, "b", "b", "b"], "category", id="category"),
        pytest.param([0, 0, pd.NA, 1, 1, 1], "Int64", id="nullable-integer"),
        pytest.param([0, 0, np.nan, 1, 1, 1], "float64", id="numeric-nan"),
    ],
)
@pytest.mark.parametrize("embedded_target", [False, True])
def test_missing_classification_labels_are_rejected(
    frame_factory, calculator, values, dtype, embedded_target
):
    """Missing labels must fail before they become a synthetic class in feature scores."""
    labels = pd.Series(values, dtype=dtype, name="label")
    source = pd.DataFrame({"signal": [0, 1, 100, 3, 4, 5], "noise": [0, 1, 0, 1, 0, 1]})
    if embedded_target:
        source["label"] = labels
    frame = source if frame_factory is pd.DataFrame else pl.from_pandas(source)
    data = frame if embedded_target else (frame, labels)

    with pytest.raises(ValueError, match="(?i)(target|y).*(missing|NaN|null)"):
        calculator.fit(data, {"target_column": "label", "problem_type": "classification", "k": 1})
