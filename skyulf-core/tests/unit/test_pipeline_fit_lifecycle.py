"""Regression coverage for replacing a pipeline's fitted state."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import accuracy_score

from skyulf.data.dataset import SplitDataset
from skyulf.pipeline import SkyulfPipeline


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("tuning", [False, True], ids=["model", "tuner"])
@pytest.mark.parametrize("failure", ["preprocessing", "model", "held_out_prediction"])
def test_failed_refit_invalidates_prediction_and_allows_recovery(engine, tuning, failure):
    """Any failed replacement must block prediction until a complete fit succeeds."""
    frame = pd.DataFrame({"x": np.arange(24, dtype=float), "target": np.arange(24) * 2.0})

    def split(train: pd.DataFrame, test: pd.DataFrame) -> SplitDataset:
        """Keep the same data and held-out failure across both supported engines."""
        if engine == "polars":
            return SplitDataset(train=pl.from_pandas(train), test=pl.from_pandas(test))
        return SplitDataset(train=train, test=test)

    modeling: dict[str, Any] = {"type": "linear_regression"}
    if tuning:
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "ridge_regression"},
            "strategy": "grid",
            "metric": "r2",
            "search_space": {"alpha": [0.0]},
            "cv_folds": 2,
            "n_jobs": 1,
        }
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}}
            ],
            "modeling": modeling,
        }
    )
    pipeline.fit(split(frame, frame.iloc[:0]), target_column="target")
    query = pd.DataFrame({"x": [25.0]})
    np.testing.assert_allclose(pipeline.predict(query), [50.0])

    replacement = frame.assign(x=frame["x"] + 100)
    held_out = frame.iloc[:0]
    if failure == "preprocessing":
        replacement = replacement.assign(x="invalid")
    elif failure == "model":
        replacement = replacement.assign(target=np.nan)
    else:
        # This fails after the replacement model has already been fitted.
        held_out = frame.iloc[:2].assign(x=np.nan)
        if tuning:
            # Tuning catches evaluation errors, so fail its held-out transform instead.
            held_out = held_out.assign(x="invalid")

    with pytest.raises((ValueError, TypeError, pl.exceptions.InvalidOperationError)):
        pipeline.fit(split(replacement, held_out), target_column="target")

    assert not pipeline.is_fitted()
    with pytest.raises(ValueError, match="not fitted"):
        pipeline.predict(query)

    pipeline.fit(split(frame, frame.iloc[:0]), target_column="target")
    assert pipeline.is_fitted()
    np.testing.assert_allclose(pipeline.predict(query), [50.0])


@pytest.mark.parametrize("relabel", [False, True], ids=["same_labels", "new_labels"])
@pytest.mark.parametrize("tuning", [False, True], ids=["model", "tuner"])
def test_refit_requires_new_decision_thresholds(relabel, tuning):
    """A replacement classifier cannot reuse decision thresholds from its predecessor."""
    frame = pd.DataFrame(
        {"x": np.arange(40, dtype=float), "target": (np.arange(40) >= 20).astype(int)}
    )
    modeling: dict[str, Any] = {"type": "logistic_regression"}
    if tuning:
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "logistic_regression"},
            "strategy": "grid",
            "metric": "accuracy",
            "search_space": {"C": [1.0]},
            "cv_folds": 2,
            "n_jobs": 1,
        }
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": modeling})
    pipeline.fit(SplitDataset(train=frame, test=frame.iloc[:0]), target_column="target")
    pipeline.optimize_thresholds(frame[["x"]], frame["target"], accuracy_score, grid_points=3)
    pipeline.predict(frame[["x"]], use_tuned_thresholds=True)

    replacement = (
        frame.assign(target=frame["target"].map({0: "no", 1: "yes"})) if relabel else frame
    )
    pipeline.fit(SplitDataset(train=replacement, test=replacement.iloc[:0]), target_column="target")

    assert pipeline.is_fitted()
    with pytest.raises(ValueError, match="optimize_thresholds"):
        pipeline.predict(frame[["x"]], use_tuned_thresholds=True)

    thresholds = pipeline.optimize_thresholds(
        replacement[["x"]], replacement["target"], accuracy_score, grid_points=3
    )
    assert set(thresholds) == set(replacement["target"])
    np.testing.assert_array_equal(
        pipeline.predict(replacement[["x"]], use_tuned_thresholds=True), replacement["target"]
    )
