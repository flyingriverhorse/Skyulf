"""Pin synchronous capture of an actual row boundary during preprocessing."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skyulf.modeling.base import extract_xy
from skyulf.preprocessing.pipeline import FeatureEngineer


def _step(name, transformer, **params):
    """Build ordered operations without sharing mutable configuration across tests."""
    return {"name": name, "transformer": transformer, "params": params}


def _frame():
    """Make original feature values identify the exact rows in each partition."""
    return pd.DataFrame({"x": np.arange(12, dtype=float), "target": np.arange(12) % 2})


def test_callback_captures_first_actual_split_before_later_transformations():
    """Later scaling and skipped splitters cannot replace the original raw partition."""
    captured = []

    def capture(payload):
        """Copy synchronously because later steps may mutate the supplied frame."""
        captured.append(deepcopy(payload))

    engineer = FeatureEngineer(
        [
            _step("columns", "feature_target_split", target_column="target"),
            _step("split", "TrainTestSplitter", target_column="target", random_state=None),
            _step("scale", "StandardScaler", columns=["x"]),
            _step("again", "Split", target_column="target", test_size=0.4),
        ]
    )
    transformed, metrics = engineer.fit_transform(_frame(), on_split=capture)
    assert len(captured) == 1
    raw_x, raw_y = extract_xy(captured[0].train, "target")
    processed_x, processed_y = extract_xy(transformed.train, "target")
    np.testing.assert_array_equal(raw_x["x"], _frame().loc[raw_x.index, "x"])
    np.testing.assert_array_equal(raw_y, processed_y)
    assert abs(processed_x["x"].mean()) < 1e-12
    assert list(metrics["steps"]) == ["0:columns", "1:split", "2:scale", "3:again"]


def test_callback_is_not_called_for_an_already_split_input():
    """A skipped splitter after preprocessing is not a new raw-data boundary."""
    already_split, _metrics = FeatureEngineer(
        [_step("split", "TrainTestSplitter", target_column="target", random_state=42)]
    ).fit_transform(_frame())
    captured = []
    FeatureEngineer(
        [
            _step("scale", "StandardScaler", columns=["x"]),
            _step("again", "TrainTestSplitter", target_column="target"),
        ]
    ).fit_transform(already_split, on_split=captured.append)
    assert captured == []


def test_feature_target_separation_does_not_invoke_the_row_split_callback():
    """Separating X and y does not create an independent training partition."""
    captured = []
    output, _metrics = FeatureEngineer(
        [_step("columns", "feature_target_split", target_column="target")]
    ).fit_transform(_frame(), on_split=captured.append)
    assert len(output[0]) == len(_frame())
    assert captured == []


def test_callback_failure_stops_before_later_preprocessing_fits():
    """Unavailable snapshot storage must abort before a learned transform can hide the raw rows."""
    engineer = FeatureEngineer(
        [
            _step("split", "TrainTestSplitter", target_column="target"),
            _step("scale", "StandardScaler", columns=["x"]),
        ]
    )

    def unavailable(payload):
        """Simulate a consumer unable to preserve the emitted partition."""
        raise RuntimeError("snapshot unavailable")

    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        engineer.fit_transform(_frame(), on_split=unavailable)
    assert engineer.fitted_steps == []
