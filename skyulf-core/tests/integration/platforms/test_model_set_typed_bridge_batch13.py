"""Model-set assembly must preserve native typed features while retaining keyed row identity."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.schema import SchemaMismatchError
from skyulf.inference._manifest import ColumnSpec
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.model_set import ComponentReference, load_model_set, save_model_set
from skyulf.inference.model_set_scoring import score_model_set
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture
def typed_model_set(tmp_path):
    """Persist a real mixed-type component whose preprocessing uses every special column."""
    dates = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(12)]
    frame = pl.DataFrame({"x": list(np.arange(12.0)), "target": list(np.arange(12.0) * 2)})
    frame = frame.with_columns(
        pl.Series("date", [value.date() for value in dates], dtype=pl.Date),
        pl.Series("time", dates, dtype=pl.Datetime("us", "UTC")),
        pl.Series("category", ["a", "b"] * 6, dtype=pl.Categorical),
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "date",
                    "transformer": "DateFeatures",
                    "params": {
                        "columns": ["date", "time"],
                        "features": ["day"],
                        "drop_original": True,
                    },
                },
                {
                    "name": "category",
                    "transformer": "OneHotEncoder",
                    "params": {
                        "columns": ["category"],
                        "drop_original": True,
                        "handle_unknown": "ignore",
                    },
                },
                {
                    "name": "impute",
                    "transformer": "SimpleImputer",
                    "params": {"strategy": "most_frequent"},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(frame, target_column="target")
    component_path = tmp_path / "component"
    save_pipeline(pipeline, component_path)
    local = load_pipeline(component_path)
    reference = ComponentReference(name="typed", version="1", digest=local.manifest.pipeline_sha256)
    save_model_set(
        tmp_path / "set",
        {"amount": (reference, component_path)},
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    artifact = load_model_set(tmp_path / "set")
    query = frame.drop("target").head(3).with_columns(pl.Series("id", [2**53 + 1, 3, 2**53 + 3]))
    return artifact, local, query


@pytest.mark.parametrize("missing", [False, True])
def test_model_set_preserves_date_and_other_native_types(typed_model_set, missing):
    """Date, timezone and categories must survive assembly with nulls and exact large keys."""
    artifact, local, query = typed_model_set
    if missing:
        query = query.with_columns(
            pl.when(pl.col("id") == 3).then(None).otherwise(pl.col(name)).alias(name)
            for name in ["date", "time", "category"]
        )
    original = query.clone()
    expected = predict_pipeline(query.drop("id"), local)
    actual = score_model_set(query, artifact).frame
    assert actual["id"].tolist() == [2**53 + 1, 3, 2**53 + 3]
    np.testing.assert_allclose(actual["amount__prediction"].astype(float), expected["prediction"])
    assert query.equals(original)


def test_model_set_preserves_exact_arrow_input_and_pandas_index(typed_model_set):
    """Already lossless pandas transport must retain null-safe Arrow types and caller indices."""
    artifact, local, query = typed_model_set
    pandas_query = query.to_pandas(use_pyarrow_extension_array=True)
    pandas_query.index = pd.Index([8, 2, 8], name="row")
    expected = predict_pipeline(query.drop("id"), local)
    actual = score_model_set(pandas_query, artifact).frame
    assert actual.index.equals(pandas_query.index)
    np.testing.assert_allclose(actual["amount__prediction"].astype(float), expected["prediction"])


def test_model_set_does_not_reinterpret_wrong_raw_date_type(typed_model_set):
    """The bridge may preserve a native Date but must not cast caller datetime or string data."""
    artifact, _, query = typed_model_set
    for dtype in (pl.String, pl.Datetime("ms")):
        invalid = query.with_columns(pl.col("date").cast(dtype))
        with pytest.raises(SchemaMismatchError, match="date"):
            score_model_set(invalid, artifact)


def test_model_set_preserves_large_nullable_integer_features(tmp_path):
    """Large integer categories and nulls must produce distinct real predictions after assembly."""
    large = 2**53 + 1
    frame = pl.DataFrame(
        {
            "x": pl.Series([large, large + 2, None] * 4, dtype=pl.Int64),
            "target": [10.0, 20.0, 30.0] * 4,
        }
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "category",
                    "transformer": "OneHotEncoder",
                    "params": {"columns": ["x"], "drop_original": True, "include_missing": True},
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(frame, target_column="target")
    path = tmp_path / "component"
    save_pipeline(pipeline, path)
    local = load_pipeline(path)
    reference = ComponentReference(name="large", version="1", digest=local.manifest.pipeline_sha256)
    artifact = save_model_set(
        tmp_path / "set",
        {"amount": (reference, path)},
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    query = frame.drop("target").head(3).with_columns(pl.Series("id", [3, 1, 2]))
    actual = score_model_set(query, artifact).frame
    assert actual["id"].tolist() == [3, 1, 2]
    np.testing.assert_allclose(actual["amount__prediction"].astype(float), [10.0, 20.0, 30.0])
