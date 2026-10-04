"""Prediction must never return a shortened batch without input-row provenance."""

import pickle
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import accuracy_score

from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.pipeline import FeatureEngineer


def _frame(engine: str, values: list[float], *, target: bool = False) -> Any:
    """Build native inputs, keeping duplicate unordered pandas indices as a control."""
    frame = pd.DataFrame({"x": values}, index=[7, 2, 7] if len(values) == 3 else None)
    if target:
        frame["target"] = 2 * frame["x"] + 1
    return frame if engine == "pandas" else pl.from_pandas(frame)


def _pipeline(engine: str, transformer: str) -> SkyulfPipeline:
    """Fit a real linear relationship through the selected outlier step."""
    params: dict[str, Any] = {"columns": ["x"]}
    if transformer == "ManualBounds":
        params["bounds"] = {"x": {"lower": 0, "upper": 99}}
    if transformer == "EllipticEnvelope":
        params["random_state"] = 42
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "outliers", "transformer": transformer, "params": params}],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(_frame(engine, list(map(float, range(100))), target=True), target_column="target")
    return pickle.loads(pickle.dumps(pipeline))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("transformer", ["IQR", "ZScore", "ManualBounds", "EllipticEnvelope"])
def test_predict_rejects_filtered_rows_without_changing_transform_semantics(engine, transformer):
    """Saved pipelines must not serialize two predictions as if they described three inputs."""
    pipeline = _pipeline(engine, transformer)
    data = _frame(engine, [20.0, 1000.0, 40.0])
    original = data.copy() if isinstance(data, pd.DataFrame) else data.clone()
    transformed = pipeline.feature_engineer.transform(data)
    assert len(transformed) == 2
    with pytest.raises(ValueError, match="outliers.*row count.*3.*2"):
        pipeline.predict(data)
    if isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(data, original)
    else:
        assert data.equals(original)
    # A rejected batch must not invalidate the already-fitted pipeline.
    np.testing.assert_allclose(pipeline.predict(_frame(engine, [20.0, 40.0])), [41.0, 81.0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_predict_rejects_all_filtered_rows_before_the_model_empty_input_error(engine):
    """The user must see the filtering stage and counts even when no observation survives."""
    pipeline = _pipeline(engine, "IQR")
    with pytest.raises(ValueError, match="outliers.*row count.*2.*0"):
        pipeline.predict(_frame(engine, [1000.0, 2000.0]))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_winsorize_keeps_one_prediction_per_input_and_applies_fitted_clipping(engine):
    """Row-preserving outlier handling must continue clipping, not be skipped at inference."""
    pipeline = _pipeline(engine, "Winsorize")
    data = _frame(engine, [20.0, 1000.0, 40.0])
    transformed = pipeline.feature_engineer.transform(data)
    model = pipeline.model_estimator
    assert model is not None
    expected = model.applier.predict(transformed, model.model)
    actual = pipeline.predict(data)
    np.testing.assert_allclose(actual, expected)
    assert len(actual) == 3
    assert float(transformed["x"][1] if engine == "polars" else transformed["x"].iloc[1]) < 1000


class _ChangingApplier:
    """Model a plugin that changes cardinality despite its unrecognized node type."""

    def __init__(self, grow: bool):
        """Choose a contraction or expansion to verify both invalid directions."""
        self.grow = grow

    def apply(self, data, params):
        """Return the changed frame without mutating the caller's input."""
        return pd.concat([data, data.iloc[:1]]) if self.grow else data.iloc[:-1]


class _ArrayApplier:
    """Represent a row-preserving plugin that returns sklearn-ready arrays."""

    def apply(self, data, params):
        """Retain the target when converting a supported feature-target pair."""
        if isinstance(data, tuple):
            return np.asarray(data[0]), data[1]
        return np.asarray(data)


@pytest.mark.parametrize("with_target", [False, True])
def test_prediction_transform_accepts_row_preserving_array_plugins(with_target):
    """Cardinality validation must not confuse a NumPy result with an empty dataframe."""
    engineer = FeatureEngineer([])
    engineer.fitted_steps = [
        {"name": "array", "type": "CustomArray", "applier": _ArrayApplier(), "artifact": {}}
    ]
    frame = pd.DataFrame({"x": [1, 2, 3]})
    target = np.array([0, 1, 0])
    data = (frame, target) if with_target else frame
    expected = engineer.transform(data)
    actual = engineer.transform(data, preserve_rows=True)
    if with_target:
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] is target
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("grow", [False, True])
def test_prediction_transform_checks_every_step_including_plugins(grow):
    """A later step must not conceal an earlier row-count change by restoring batch size."""
    engineer = FeatureEngineer([])
    engineer.fitted_steps = [
        {
            "name": "plugin",
            "type": "CustomPlugin",
            "applier": _ChangingApplier(grow),
            "artifact": {},
        },
        {
            "name": "inverse",
            "type": "CustomPlugin",
            "applier": _ChangingApplier(not grow),
            "artifact": {},
        },
    ]
    data = pd.DataFrame({"x": [1, 2, 3]})
    assert len(engineer.transform(data)) == 3
    with pytest.raises(ValueError, match="plugin.*row count"):
        engineer.transform(data, preserve_rows=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_threshold_prediction_rejects_filtered_inputs_too(engine):
    """Probability-based decisions must enforce the same batch contract as plain prediction."""
    frame = _frame(engine, list(map(float, range(100))), target=True)
    if isinstance(frame, pd.DataFrame):
        frame["target"] = (frame["x"] >= 50).astype(int)
    else:
        frame = frame.with_columns((pl.col("x") >= 50).cast(pl.Int64).alias("target"))
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "outliers", "transformer": "IQR", "params": {"columns": ["x"]}}
            ],
            "modeling": {"type": "decision_tree_classifier", "params": {"random_state": 42}},
        }
    )
    pipeline.fit(frame, target_column="target")
    pipeline.optimize_thresholds(
        _frame(engine, [20.0, 40.0, 60.0, 80.0]), np.array([0, 0, 1, 1]), accuracy_score
    )
    with pytest.raises(ValueError, match="outliers.*row count.*3.*2"):
        pipeline.predict(_frame(engine, [20.0, 1000.0, 40.0]), use_tuned_thresholds=True)
