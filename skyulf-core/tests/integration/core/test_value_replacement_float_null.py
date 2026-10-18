"""Null replacement must leave floating features usable by fitted numeric preprocessing."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf import SkyulfPipeline
from skyulf.core.schema import SkyulfSchema
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.preprocessing.cleaning.value_replacement import (
    ValueReplacementApplier,
    ValueReplacementCalculator,
)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float64", "float64[pyarrow]"])
@pytest.mark.parametrize(
    "rules",
    [
        {"mapping": {"-200": None}},
        {"mapping": {"x": {"-200": None}}},
        {"to_replace": -200, "value": None},
        {"to_replace": [-200], "value": [None]},
        {"to_replace": {-200: pd.NA}},
        {"replacements": [{"old": -200, "new": pd.NA}]},
    ],
)
def test_saved_float_null_replacement_preserves_dtype_and_chunks(dtype, rules):
    """A sentinel must not make numeric dtype depend on whether a request contains that sentinel."""
    frame = pd.DataFrame({"x": pd.Series([-200.0, 2.5, None, 4.5], dtype=dtype)})
    frame.index = [5, 5, 7, 9]
    before = frame.copy(deep=True)
    config = {"columns": ["x"], **rules}
    state = ValueReplacementCalculator().fit(frame, config)
    saved = pickle.dumps(state)
    restored = pickle.loads(saved)
    expected = frame.copy()
    expected["x"] = expected["x"].mask(expected["x"] == -200, np.nan)
    for indices in ([0, 1, 2, 3], [0], [1], []):
        actual = ValueReplacementApplier().apply(frame.iloc[indices], restored)
        pd.testing.assert_frame_equal(actual, expected.iloc[indices])
    assert pickle.dumps(restored) == saved
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float64", "float64[pyarrow]"])
def test_float_null_preview_matches_guaranteed_runtime_dtype(dtype):
    """Schema preview must keep a float feature available to downstream numeric column pickers."""
    frame = pd.DataFrame({"x": pd.Series([-200, 2.5], dtype=dtype)})
    schema = SkyulfSchema.from_dataframe(frame)
    config = {"columns": ["x"], "mapping": {-200: None}}
    assert ValueReplacementCalculator().infer_output_schema(schema, config) == schema


@pytest.mark.parametrize("missing", [pd.NaT, np.datetime64("NaT")])
def test_float_preview_does_not_treat_temporal_missing_as_numeric_null(missing):
    """Preview must not promise float storage for missing scalars outside numeric null rules."""
    schema = SkyulfSchema.from_columns(["x"], {"x": "float64"})
    inferred = ValueReplacementCalculator().infer_output_schema(
        schema, {"columns": ["x"], "mapping": {-200: missing}}
    )
    assert inferred == SkyulfSchema.from_columns(["x"], {})


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_null_normalization_keeps_nonnumeric_replacement_semantics(dtype):
    """Changing float null rules must not infer numeric types for text or object features."""
    frame = pd.DataFrame({"x": pd.Series(["unknown", "ok", None], dtype=dtype)})
    config = {"columns": ["x"], "mapping": {"unknown": None}}
    state = ValueReplacementCalculator().fit(frame, config)
    expected = frame.copy()
    expected["x"] = expected["x"].replace({"unknown": None})
    pd.testing.assert_frame_equal(ValueReplacementApplier().apply(frame, state), expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_sentinel_pipeline_retains_numeric_imputation(engine, tmp_path, monkeypatch):
    """The documented JSON-safe sentinel recipe must train and score through a numeric imputer."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    values = np.arange(1.0, 41.0)
    frame = pd.DataFrame({"x": values.copy(), "target": values * 2 + 1})
    frame.loc[::7, "x"] = -200
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "sentinel",
                    "transformer": "ValueReplacement",
                    "params": {"columns": ["x"], "mapping": {-200: None}},
                },
                {
                    "name": "impute",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["x"], "strategy": "mean"},
                },
            ],
            "modeling": {"type": "ridge_regression", "params": {"alpha": 0.1}},
        }
    )
    pipeline.fit(pl.from_pandas(frame) if engine == "polars" else frame, target_column="target")
    save_pipeline(pipeline, tmp_path / "sentinel")
    restored = load_pipeline(tmp_path / "sentinel")
    query = pd.DataFrame({"x": [-200.0, 3.0, None]})
    native = pl.from_pandas(query) if engine == "polars" else query
    actual = predict_pipeline(native, restored)
    expected = np.asarray(pipeline.predict(native))
    np.testing.assert_allclose(actual["prediction"], expected)
    pieces = pd.concat(
        [
            predict_pipeline(
                native.slice(i, 1) if isinstance(native, pl.DataFrame) else native.iloc[[i]],
                restored,
            )
            for i in range(3)
        ],
        ignore_index=True,
    )
    np.testing.assert_allclose(actual["prediction"], pieces["prediction"])
    assert np.isfinite(actual["prediction"]).all()
