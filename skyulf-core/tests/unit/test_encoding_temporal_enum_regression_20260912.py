"""Regression coverage for portable dummy keys and automatic Enum encoding."""

import json
import pickle
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.encoding import DummyEncoderApplier, DummyEncoderCalculator
from skyulf.preprocessing.encoding._common import detect_categorical_columns
from skyulf.registry import NodeRegistry


def _frame(values: list[Any], engine: str) -> Any:
    """Construct native inputs without normalizing away dtype differences."""
    data = {"v": values}
    return pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)


_VALUES = [
    pytest.param([True, False, True, False], id="bool"),
    pytest.param([date(2024, 1, 1), date(2024, 1, 2)] * 2, id="date"),
    pytest.param([datetime(2024, 1, 1), datetime(2024, 1, 2)] * 2, id="datetime"),
    pytest.param(
        [datetime(2024, 1, 1, 12, 3, 4, 123456), datetime(2024, 1, 2, 5, 6)] * 2,
        id="fractional_datetime",
    ),
]


@pytest.mark.parametrize("values", _VALUES)
@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
def test_dummy_saved_categories_replay_on_either_engine(values, fit_engine, apply_engine):
    """Changing engines must preserve known indicators, names, nulls and source rows."""
    train = _frame(values, fit_engine)
    artifact = DummyEncoderCalculator().fit(train, {"columns": ["v"]})
    saved = json.loads(json.dumps(artifact))
    expected = DummyEncoderApplier().apply(train, saved)
    replay = _frame([*values, None], apply_engine)
    actual = DummyEncoderApplier().apply(replay, saved)

    assert list(actual.columns) == list(expected.columns)
    np.testing.assert_array_equal(actual.to_numpy()[:4], expected.to_numpy())
    assert actual.to_numpy().sum(axis=1).tolist() == [1, 1, 1, 1, 0]
    assert list(replay.columns) == ["v"]


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
def test_dummy_midnight_category_is_independent_of_fractional_batch_companion(fit_engine):
    """A later timestamp must not change a known midnight value's rendered key."""
    artifact = DummyEncoderCalculator().fit(
        _frame([datetime(2024, 1, 1)], fit_engine), {"columns": ["v"]}
    )
    replay = pd.DataFrame({"v": [datetime(2024, 1, 1), datetime(2024, 1, 1, 12, 0)]})
    actual = DummyEncoderApplier().apply(replay, artifact)
    assert actual.to_numpy().sum(axis=1).tolist() == [1, 0]


@pytest.mark.parametrize("values", _VALUES)
@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
def test_dummy_saved_model_predictions_survive_engine_change(values, fit_engine, tmp_path):
    """Persisted real model predictions must retain fitted category meaning at serving."""
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "dummy", "transformer": "DummyEncoder", "params": {"columns": ["v"]}}
            ],
            "modeling": {"type": "linear_regression", "params": {}},
        }
    )
    train = _frame(values, fit_engine)
    targets = [10.0, 30.0, 10.0, 30.0]
    if fit_engine == "polars":
        train = train.with_columns(pl.Series("target", targets))
    else:
        train["target"] = targets
    pipeline.fit(train, target_column="target")
    path = str(tmp_path / "dummy_pipeline.pkl")
    pipeline.save(path)
    loaded = SkyulfPipeline.load(path)
    replay_engine = "pandas" if fit_engine == "polars" else "polars"
    prediction = loaded.predict(_frame(values, replay_engine))
    np.testing.assert_allclose(prediction, targets, atol=1e-10)


_AUTO_ENCODERS = [
    "DummyEncoder",
    "HashEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "WOEEncoder",
]


@pytest.mark.parametrize("node_type", _AUTO_ENCODERS)
def test_enum_auto_selection_matches_explicit_encoding_and_pandas_category(node_type):
    """Each consumer of the common selector must actually encode native Enum inputs."""
    values = ["a", "b", "a", "b", None, "a"]
    train = pl.DataFrame(
        {"v": pl.Series(values, dtype=pl.Enum(["b", "a", "unused"])), "keep": range(6)}
    )
    labels = pl.Series("target", [0, 1, 0, 1, 0, 1])
    calculator = NodeRegistry.get_calculator(node_type)()
    applier = NodeRegistry.get_applier(node_type)()
    config = {"target_type": "binary"}
    automatic = calculator.fit((train, labels), config)
    explicit = calculator.fit((train, labels), {**config, "columns": ["v"]})
    restored = pickle.loads(pickle.dumps(automatic))
    actual = applier.apply(train, restored)
    expected = applier.apply(train, explicit)
    assert automatic.get("columns") == ["v"]
    assert actual.equals(expected)
    assert actual["keep"].to_list() == list(range(6))
    assert detect_categorical_columns(train.to_pandas()) == ["v"]

    held_out = pl.DataFrame(
        {"v": pl.Series(["b", "new"], dtype=pl.Enum(["new", "b"])), "keep": [7, 8]}
    )
    assert applier.apply(held_out, restored).equals(applier.apply(held_out, explicit))


@pytest.mark.parametrize("values", [[], [None, None]])
def test_enum_selector_uses_declared_dtype_even_without_observed_categories(values):
    """Empty or all-null Enum columns still need their explicit categorical semantics."""
    frame = pl.DataFrame({"v": pl.Series(values, dtype=pl.Enum(["a", "b"]))})
    assert detect_categorical_columns(frame) == ["v"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_dummy_literal_string_categories_remain_case_and_format_sensitive(engine):
    """Temporal and boolean normalization must never reinterpret literal string labels."""
    values = ["True", "true", "2024-01-01", "2024-01-01 00:00:00", "1.0", "1"]
    frame = _frame(values, engine)
    artifact = DummyEncoderCalculator().fit(frame, {"columns": ["v"]})
    out = DummyEncoderApplier().apply(frame, artifact)
    assert artifact["categories"]["v"] == sorted(values)
    assert out.to_numpy().sum(axis=1).tolist() == [1] * len(values)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["bool", "datetime"])
def test_unversioned_dummy_artifacts_retain_saved_feature_names_and_indicators(engine, kind):
    """Updating the runtime must not change the feature contract of older fitted models."""
    if kind == "bool":
        values = [True, False]
        categories = ["False", "True"] if engine == "pandas" else ["false", "true"]
        expected = [[0, 1], [1, 0]]
    else:
        values = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        categories = ["2024-01-01", "2024-01-02"]
        if engine == "polars":
            categories = [f"{value} 00:00:00.000000" for value in categories]
        expected = [[1, 0], [0, 1]]
    artifact = {"columns": ["v"], "categories": {"v": categories}, "drop_first": False}
    actual = DummyEncoderApplier().apply(_frame(values, engine), artifact)
    assert list(actual.columns) == [f"v_{category}" for category in categories]
    assert actual.to_numpy().tolist() == expected


@pytest.mark.parametrize("version", [True, 0, 2, "1"])
def test_dummy_rejects_unsupported_saved_category_key_versions(version):
    """Unknown artifact formats must fail clearly instead of silently zeroing categories."""
    artifact = {"columns": ["v"], "categories": {"v": ["True"]}, "category_key_version": version}
    with pytest.raises(ValueError, match="Unsupported category key version"):
        DummyEncoderApplier().apply(pd.DataFrame({"v": [True]}), artifact)


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("time_zone", [None, "Europe/Vilnius"])
def test_dummy_datetime_precision_and_timezone_survive_engine_change(fit_engine, time_zone):
    """Distinct nanosecond categories must survive engine conversion without truncation."""
    series = pd.Series(
        pd.to_datetime(["2024-01-01 12:00:00.000000001", "2024-01-01 12:00:00.000000002"])
    )
    if time_zone:
        series = series.dt.tz_localize(time_zone)
    pandas_frame = pd.DataFrame({"v": series})
    polars_frame = pl.from_pandas(pandas_frame)
    frames = {"pandas": pandas_frame, "polars": polars_frame}
    artifact = DummyEncoderCalculator().fit(frames[fit_engine], {"columns": ["v"]})
    first = DummyEncoderApplier().apply(pandas_frame, artifact)
    second = DummyEncoderApplier().apply(polars_frame, artifact)
    assert len(artifact["categories"]["v"]) == 2
    assert first.to_numpy().tolist() == second.to_numpy().tolist() == [[1, 0], [0, 1]]


def test_dummy_date_to_pandas_datetime_conversion_keeps_fitted_category():
    """The normal Polars-to-pandas conversion must not turn date categories into unknowns."""
    train = pl.DataFrame({"v": [date(2024, 1, 1), date(2024, 1, 2)]})
    artifact = DummyEncoderCalculator().fit(train, {"columns": ["v"]})
    converted = train.to_pandas()
    assert pd.api.types.is_datetime64_any_dtype(converted["v"])
    assert DummyEncoderApplier().apply(converted, artifact).to_numpy().tolist() == [[1, 0], [0, 1]]


@pytest.mark.parametrize("node_type", _AUTO_ENCODERS)
def test_explicit_empty_encoder_selection_still_skips_enum_columns(node_type):
    """Native Enum discovery must not override the user's explicit empty selection."""
    frame = pl.DataFrame({"v": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"]))})
    artifact = NodeRegistry.get_calculator(node_type)().fit(frame, {"columns": []})
    output = NodeRegistry.get_applier(node_type)().apply(frame, artifact)
    assert artifact.get("columns", []) == []
    assert output.equals(frame)


@pytest.mark.parametrize("pandas_dtype", ["float32", "Float32"])
@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
def test_dummy_float32_keys_keep_dtype_precision_on_both_engines(
    pandas_dtype, fit_engine, apply_engine
):
    """Scalar formatting must not expand Float32 values and silently lose known categories."""
    values = [1.0, 1.2, 2.3, None]
    frames = {
        "pandas": pd.DataFrame({"v": pd.Series(values, dtype=pandas_dtype)}),
        "polars": pl.DataFrame({"v": pl.Series(values, dtype=pl.Float32)}),
    }
    artifact = DummyEncoderCalculator().fit(frames[fit_engine], {"columns": ["v"]})
    restored = json.loads(json.dumps(artifact))
    actual = DummyEncoderApplier().apply(frames[apply_engine], restored)

    assert artifact["categories"]["v"] == ["1", "1.2", "2.3"]
    assert actual.to_numpy().tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_dummy_numeric_categorical_keys_keep_original_dtype_rendering(dtype):
    """Numeric categorical storage must retain its existing labels when datetime keys change."""
    values = [1.2, 2.3] if dtype != "int64" else [1, 2]
    frame = pd.DataFrame({"v": pd.Categorical(pd.Series(values, dtype=dtype))})
    expected = sorted(frame["v"].astype(str).to_list())
    artifact = DummyEncoderCalculator().fit(frame, {"columns": ["v"]})
    actual = DummyEncoderApplier().apply(frame, artifact)

    assert artifact["categories"]["v"] == expected
    assert actual.to_numpy().tolist() == [[1, 0], [0, 1]]


@pytest.mark.parametrize("dtype", ["category", "object"])
def test_dummy_datetime_categories_keep_canonical_rendering_with_object_storage(dtype):
    """Narrow datetime normalization must still handle categorical and object timestamps."""
    values = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02 12:00")]
    frame = pd.DataFrame({"v": pd.Series(values, dtype=dtype)})
    artifact = DummyEncoderCalculator().fit(frame, {"columns": ["v"]})
    replay = pl.DataFrame({"v": values})
    actual = DummyEncoderApplier().apply(replay, artifact)

    assert artifact["categories"]["v"] == ["2024-01-01", "2024-01-02 12:00:00"]
    assert actual.to_numpy().tolist() == [[1, 0], [0, 1]]
