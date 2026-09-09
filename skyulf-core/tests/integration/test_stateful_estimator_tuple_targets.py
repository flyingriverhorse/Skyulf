"""Regression coverage for target exclusion from held-out modeling tuples."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from skyulf.data.dataset import SplitDataset
from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("wrapped", [False, True], ids=["native", "wrapped"])
@pytest.mark.parametrize("tuned", [False, True], ids=["plain", "tuned"])
@pytest.mark.parametrize("explicit_y", [False, True], ids=["embedded-y", "explicit-y"])
@pytest.mark.parametrize("held_out", ["test", "validation"])
def test_fit_predict_excludes_embedded_target_from_every_tuple(
    engine: str, wrapped: bool, tuned: bool, explicit_y: bool, held_out: str
) -> None:
    """Held-out tuples must use training's features without changing rows or explicit labels."""
    values = {
        "train": [-4, -3, -2, -1, 1, 2, 3, 4],
        "test": [3, -3, 2, -2],
        "validation": [-2, 2, -3, 3],
    }
    labels = {
        "train": [0, 0, 0, 0, 1, 1, 1, 1],
        "test": [1, 0, 1, 0],
        "validation": [0, 1, 0, 1],
    }
    splits: dict[str, Any] = {}
    originals: dict[str, tuple[pd.DataFrame, pd.Series | None]] = {}
    for name, feature_values in values.items():
        frame = pd.DataFrame(
            {"x": feature_values},
            index=pd.Index([7, 7, 3, 3] * (len(feature_values) // 4), name="row"),
        )
        # The other held-out split has target-free features so both prediction
        # branches independently expose the embedded-target failure.
        embed_target = name in ("train", held_out)
        if embed_target:
            frame["target"] = [1 - label for label in labels[name]] if explicit_y else labels[name]
        target = None
        if explicit_y or not embed_target:
            target = pd.Series(labels[name], index=frame.index + 100, name="explicit_target")
        originals[name] = (
            frame.copy(deep=True),
            None if target is None else target.copy(deep=True),
        )

        X: Any = frame
        y: Any = target
        if engine == "polars":
            X = pl.from_pandas(frame)
            y = None if target is None else pl.Series("explicit_target", target.to_numpy())
            if wrapped:
                X = SkyulfPolarsWrapper(X)
        elif wrapped:
            X = SkyulfPandasWrapper(frame)
        splits[name] = (X, y)

    calculator: Any = LogisticRegressionCalculator()
    applier: Any = LogisticRegressionApplier()
    config: dict[str, Any] = {}
    if tuned:
        calculator = TuningCalculator(calculator)
        applier = TuningApplier(applier)
        config = {
            "strategy": "grid",
            "search_space": {"C": [1.0]},
            "metric": "accuracy",
            "cv_enabled": False,
            "n_jobs": 1,
        }
    estimator = StatefulEstimator(calculator, applier, "tuple-target-regression")
    predictions = estimator.fit_predict(SplitDataset(**splits), "target", config)

    assert set(predictions) == {"train", "test", "validation"}
    for name, expected_labels in labels.items():
        np.testing.assert_array_equal(predictions[name], expected_labels)
        original_X, original_y = originals[name]
        actual_X, actual_y = splits[name]
        if wrapped:
            actual_X = actual_X.to_native()
        if engine == "polars":
            assert_frame_equal(actual_X.to_pandas(), original_X.reset_index(drop=True))
            if original_y is not None:
                assert actual_y.to_list() == original_y.tolist()
        else:
            assert_frame_equal(actual_X, original_X)
            assert predictions[name].index.equals(original_X.index)
            if original_y is not None:
                assert_series_equal(actual_y, original_y)
        if original_y is None:
            assert actual_y is None
    model = estimator.model
    if isinstance(model, tuple):
        model = model[0]
    assert model is not None
    assert model.n_features_in_ == 1
