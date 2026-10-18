"""Learned group deviations must use training means even for unseen scoring groups."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.preprocessing.feature_generation.generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)


def _config():
    """Request the new mean-deviation method through the existing FeatureMath operation."""
    return {
        "operations": [
            {
                "operation_type": "group_agg",
                "method": "deviation",
                "input_columns": ["group"],
                "secondary_columns": ["value"],
                "output_column": "deviation",
            }
        ]
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_deviation_uses_saved_means_global_fallback_and_null_group(engine):
    """Unknown and valueless groups use the training global mean; scoring companions cannot change it."""
    train = pd.DataFrame(
        {"group": ["a", "a", "b", None, "empty"], "value": [2.0, 4.0, 12.0, 22.0, None]}
    )
    frame = pl.from_pandas(train) if engine == "polars" else train
    state = pickle.loads(pickle.dumps(FeatureGenerationCalculator().fit(frame, _config())))
    original = deepcopy(state)
    sample = pd.DataFrame(
        {
            "group": ["a", "b", "unseen", None, "empty", "a"],
            "value": [8.0, 14.0, 20.0, 25.0, 30.0, None],
        },
        index=[9, 1, 9, 3, 4, -1],
    )
    sample = pl.from_pandas(sample) if engine == "polars" else sample
    applier = FeatureGenerationApplier()
    result = applier.apply(sample, state)
    assert "deviation" in result.columns
    np.testing.assert_allclose(
        result["deviation"].to_numpy(), [5.0, 2.0, 10.0, 3.0, 20.0, np.nan], equal_nan=True
    )
    for positions in ([0], [2, 3], [5, 1, 0], []):
        subset = sample[positions] if isinstance(sample, pl.DataFrame) else sample.iloc[positions]
        got = applier.apply(subset, state)["deviation"].to_numpy()
        np.testing.assert_allclose(got, result["deviation"].to_numpy()[positions], equal_nan=True)
    capability = applier.inference_capability(state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert state == original


def test_deviation_canonical_keys_do_not_merge_boolean_integer_or_literal_nan():
    """Typed group identities must not collide through pandas equality or string serialization."""
    train = pd.DataFrame(
        {
            "group": pd.Series([True, 1, "1", None, "nan"], dtype=object),
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    state = FeatureGenerationCalculator().fit(train, _config())
    query = train.assign(value=100.0)
    result = FeatureGenerationApplier().apply(query, state)
    assert "deviation" in result.columns
    assert result["deviation"].to_list() == [90.0, 80.0, 70.0, 60.0, 50.0]


@pytest.mark.parametrize("bad", [True, 2, "1"])
def test_deviation_saved_mapping_rejects_unknown_key_version(bad):
    """Saved deviation maps require the exact key contract before declaring row-local inference."""
    state = FeatureGenerationCalculator().fit(
        pd.DataFrame({"group": ["a"], "value": [1.0]}), _config()
    )
    assert state["operations"][0]["group_agg_mapping"] is not None
    state["operations"][0]["group_agg_mapping"]["category_key_version"] = bad
    with pytest.raises(ValueError, match="version"):
        FeatureGenerationApplier.validate_inference_state(dict(state))


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan, True])
@pytest.mark.parametrize("field", ["global_value", "values"])
def test_deviation_rejects_nonfinite_or_boolean_saved_means(bad, field):
    """New deviation statistics must not admit fabricated Boolean or nonfinite mean values."""
    state = FeatureGenerationCalculator().fit(
        pd.DataFrame({"group": ["a"], "value": [1.0]}), _config()
    )
    mapping = state["operations"][0]["group_agg_mapping"]
    assert mapping is not None
    mapping[field] = [bad] if field == "values" else bad
    with pytest.raises(ValueError, match="finite"):
        FeatureGenerationApplier.validate_inference_state(dict(state))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_deviation_nonfinite_training_values_fail_before_publishing_state(engine):
    """A nonfinite training value must fail clearly instead of saving an unusable global fallback."""
    frame = pd.DataFrame({"group": ["a", "b"], "value": [1.0, np.inf]})
    with pytest.raises(ValueError, match="finite"):
        FeatureGenerationCalculator().fit(
            pl.from_pandas(frame) if engine == "polars" else frame, _config()
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_all_missing_training_values_produce_typed_missing_deviations(engine):
    """No observed training mean must leave deviations missing without breaking empty batches."""
    train = pd.DataFrame({"group": ["a"], "value": [np.nan]})
    native = pl.from_pandas(train) if engine == "polars" else train
    state = FeatureGenerationCalculator().fit(native, _config())
    query = (
        native.with_columns(pl.lit(8.0).alias("value"))
        if isinstance(native, pl.DataFrame)
        else train.assign(value=8.0)
    )
    result = FeatureGenerationApplier().apply(query, state)
    assert pd.isna(result["deviation"].to_numpy()).all()
    empty = query.head(0)
    got = FeatureGenerationApplier().apply(empty, state)
    assert str(got["deviation"].dtype).lower() == "float64"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_deviation_survives_real_model_save_load_and_cross_engine_replay(engine, tmp_path):
    """A saved model must replay learned deviations for new groups without fitting on scoring rows."""
    frame = pd.DataFrame(
        {
            "group": [1, 1, 2, 2, 3, 3],
            "value": [2.0, 4.0, 8.0, 10.0, 14.0, 16.0],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "deviation", "transformer": "FeatureMath", "params": _config()}
            ],
            "modeling": {"type": "decision_tree_regressor", "params": {"random_state": 42}},
        }
    )
    pipeline.fit(pl.from_pandas(frame) if engine == "polars" else frame, target_column="target")
    save_pipeline(pipeline, tmp_path / "deviation")
    restored = load_pipeline(tmp_path / "deviation")
    query = pd.DataFrame({"group": [1, 4, 2], "value": [3.0, 15.0, 9.0]})
    expected = predict_pipeline(query, restored)["prediction"].to_numpy()
    for apply_engine in ("pandas", "polars"):
        for positions in ([0, 1, 2], [2, 0], [1]):
            subset = query.iloc[positions]
            got = predict_pipeline(
                pl.from_pandas(subset) if apply_engine == "polars" else subset, restored
            )
            np.testing.assert_array_equal(got["prediction"].to_numpy(), expected[positions])
    assert (tmp_path / "deviation").exists()
