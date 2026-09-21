"""Observed feature schemas for portable model inference."""

import gc
import pickle
import weakref
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.schema import SkyulfSchema
from skyulf.data.dataset import SplitDataset
from skyulf.engines import get_engine
from skyulf.pipeline import SkyulfPipeline


def _frame() -> pd.DataFrame:
    """Include unsorted feature names and a target in the middle of the frame."""
    return pd.DataFrame(
        {
            "zeta": np.arange(24, dtype=np.int64),
            "target": np.arange(24) * 2.0 + 1,
            "category": ["b", "a"] * 12,
            "alpha": np.arange(24, dtype=np.int64) % 3,
        }
    )


def _pipeline(tuning: bool = False) -> SkyulfPipeline:
    """Train a real model after changing feature names, dtypes and order."""
    modeling: dict[str, Any] = {"type": "ridge_regression", "params": {"alpha": 0.1}}
    if tuning:
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "ridge_regression"},
            "strategy": "grid",
            "metric": "r2",
            "search_space": {"alpha": [0.1]},
            "cv_folds": 2,
            "n_jobs": 1,
        }
    return SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "encode",
                    "transformer": "OneHotEncoder",
                    "params": {"columns": ["category"]},
                },
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": ["zeta"]},
                },
            ],
            "modeling": modeling,
        }
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("payload", ["frame", "tuple", "split", "split_tuple", "wrapped"])
@pytest.mark.parametrize("tuning", [False, True], ids=["model", "tuner"])
def test_inference_schemas_describe_actual_training_features(engine, payload, tuning):
    """Serving must preserve the raw and learned feature order before NumPy loses names."""
    frame: Any = _frame()
    if engine == "polars":
        frame = pl.from_pandas(frame)
    data: Any = frame
    if payload in {"tuple", "split_tuple"}:
        features = frame.drop("target") if engine == "polars" else frame.drop(columns="target")
        data = features, frame["target"]
    if payload in {"split", "split_tuple"}:
        data = SplitDataset(train=data, test=frame[:2])
    if payload == "wrapped":
        data = get_engine(frame).wrap(frame)
    pipeline = _pipeline(tuning)

    pipeline.fit(cast(Any, data), target_column="target")

    assert pipeline._inference_schemas is not None
    raw, modeled = pipeline._inference_schemas
    int64, float64, int8 = (
        ("int64", "float64", "int8") if engine == "pandas" else ("Int64", "Float64", "Int8")
    )
    assert raw.columns == ("zeta", "category", "alpha")
    assert raw.dtypes == {
        "zeta": int64,
        "category": "object" if engine == "pandas" else "String",
        "alpha": int64,
    }
    assert modeled.columns == ("zeta", "alpha", "category_a", "category_b")
    assert modeled.dtypes == {
        "zeta": float64,
        "alpha": int64,
        "category_a": int8,
        "category_b": int8,
    }


@pytest.mark.parametrize("tuning", [False, True], ids=["model", "tuner"])
@pytest.mark.parametrize("failure", ["preprocessing", "model", "held_out_transform"])
def test_failed_refit_clears_inference_schemas(tuning, failure):
    """A failed replacement must not advertise schemas from an obsolete or partial model."""
    pipeline = _pipeline(tuning)
    frame = _frame()
    pipeline.fit(frame, target_column="target")
    assert pipeline._inference_schemas is not None
    replacement = frame
    held_out = frame[:2]
    if failure == "preprocessing":
        replacement = frame.assign(zeta="invalid")
    elif failure == "model":
        replacement = frame.assign(target=np.nan)
    else:
        held_out = frame[:2].assign(zeta="invalid")

    with pytest.raises((TypeError, ValueError)):
        pipeline.fit(SplitDataset(train=replacement, test=held_out), target_column="target")

    assert pipeline._inference_schemas is None
    assert not pipeline.is_fitted()


def test_preprocessing_only_fit_has_no_model_inference_schemas():
    """Feature engineering alone cannot advertise a fitted model interface."""
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {}})

    pipeline.fit(_frame(), target_column="target")

    assert pipeline._inference_schemas is None


@pytest.mark.parametrize("tuning", [False, True], ids=["model", "tuner"])
def test_successful_refit_replaces_both_inference_schemas(tuning):
    """A new feature dtype and category vocabulary must replace the old serving contract."""
    pipeline = _pipeline(tuning)
    frame = _frame()
    pipeline.fit(frame, target_column="target")
    replacement = frame.assign(zeta=frame["zeta"].astype(np.float32), category="c")

    pipeline.fit(replacement, target_column="target")

    assert pipeline._inference_schemas is not None
    raw, modeled = pipeline._inference_schemas
    assert raw.dtypes["zeta"] == "float32"
    assert modeled.columns == ("zeta", "alpha", "category_c")


def test_schema_capture_does_not_retain_training_samples():
    """Capturing portable metadata must not keep any training frame alive."""
    pipeline = _pipeline()
    frame = _frame()
    reference = weakref.ref(frame)

    pipeline.fit(frame, target_column="target")
    del frame
    gc.collect()

    assert reference() is None
    assert pipeline._inference_schemas is not None
    assert all(isinstance(schema, SkyulfSchema) for schema in pipeline._inference_schemas)
    for schema in pipeline._inference_schemas:
        assert set(asdict(schema)) == {"columns", "dtypes"}
        assert all(isinstance(value, str) for value in schema.columns)
        assert all(isinstance(value, str) for value in schema.dtypes.values())


def test_inference_schemas_survive_pickle_without_changing_predictions_or_fingerprint():
    """Existing persisted pipelines keep exact schemas while retaining their serving behavior."""
    pipeline = _pipeline()
    assert pipeline._inference_schemas is None
    frame = _frame()
    pipeline.fit(frame, target_column="target")
    query = frame.drop(columns="target")[:3]
    fingerprint = pipeline.fingerprint()

    restored = pickle.loads(pickle.dumps(pipeline))

    assert restored._inference_schemas == pipeline._inference_schemas
    assert restored.fingerprint() == fingerprint
    np.testing.assert_allclose(restored.predict(query), pipeline.predict(query))
