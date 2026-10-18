"""Zero WOE smoothing is valid only when every learned category has both target classes."""

import pickle
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.encoding.woe import WOEEncoderApplier, WOEEncoderCalculator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("target", [[0, 1, 0, 1], [1, 0, 1, 0]])
def test_zero_smoothing_reports_undefined_category_ratios(engine, target):
    """Pure categories must give actionable errors instead of division-by-zero or math-domain failures."""
    frame = pd.DataFrame({"cat": ["a", "b", "a", "b"], "target": target})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    with pytest.raises(ValueError, match="regularization.*0.*both target classes"):
        WOEEncoderCalculator().fit(
            frame, {"columns": ["cat"], "target_column": "target", "regularization": 0}
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_zero_smoothing_valid_mapping_survives_disk_replay(engine, tmp_path):
    """Zero is not globally forbidden when each category has positive counts in both classes."""
    frame = pd.DataFrame({"cat": ["a", "a", "a", "b", "b", "b"], "target": [0, 0, 1, 0, 1, 1]})
    native: Any = pl.from_pandas(frame) if engine == "polars" else frame
    state = WOEEncoderCalculator().fit(
        native, {"columns": ["cat"], "target_column": "target", "regularization": 0}
    )
    path = tmp_path / "woe.pkl"
    path.write_bytes(pickle.dumps(state))
    restored = pickle.loads(path.read_bytes())
    assert restored["mappings"]["cat"] == {"a": np.log(2), "b": np.log(0.5)}
    for request in (frame, pl.from_pandas(frame)):
        actual = WOEEncoderApplier().apply(request, restored)
        np.testing.assert_allclose(np.asarray(actual["cat"]), [np.log(2)] * 3 + [np.log(0.5)] * 3)


def test_zero_smoothing_cross_fit_complement_requires_defined_ratios():
    """A valid full-data map cannot conceal an undefined out-of-fold training complement."""
    frame = pd.DataFrame({"cat": ["a", "a", "b", "b"], "target": [0, 1, 0, 1]})
    config = {"columns": ["cat"], "target_column": "target", "regularization": 0}
    assert WOEEncoderCalculator().fit(frame, config)["mappings"]["cat"] == {"a": 0.0, "b": 0.0}
    with pytest.raises(ValueError, match="regularization.*0.*both target classes"):
        WOEEncoderCalculator().fit_transform_train(frame, config)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_zero_smoothing_cross_fit_remains_supported_when_all_ratios_exist(engine):
    """Valid zero smoothing must keep leakage-safe training available rather than rejecting the option."""
    frame = pd.DataFrame({"cat": ["a"] * 40, "target": [0, 1] * 20})
    native: Any = pl.from_pandas(frame) if engine == "polars" else frame
    state, transformed = WOEEncoderCalculator().fit_transform_train(
        native, {"columns": ["cat"], "target_column": "target", "regularization": 0}
    )
    assert state["mappings"]["cat"] == {"a": 0.0}
    assert np.isfinite(np.asarray(transformed["cat"])).all()
    np.testing.assert_array_equal(np.asarray(transformed["target"]), frame["target"].to_numpy())
