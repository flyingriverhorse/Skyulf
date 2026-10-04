"""Regress missing target identity, integer bounds, native NaNs and text numeric transforms."""

import pickle
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines import EngineRegistry
from skyulf.preprocessing._target_labels import original_target_labels
from skyulf.preprocessing.casting import CastingApplier, CastingCalculator
from skyulf.preprocessing.encoding.label import LabelEncoderApplier, LabelEncoderCalculator
from skyulf.preprocessing.encoding.ordinal import OrdinalEncoderApplier, OrdinalEncoderCalculator
from skyulf.preprocessing.inspection import DatasetProfileApplier, DatasetProfileCalculator
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.preprocessing.transformations.general import (
    GeneralTransformationApplier,
    GeneralTransformationCalculator,
)
from skyulf.preprocessing.transformations.simple import (
    SimpleTransformationApplier,
    SimpleTransformationCalculator,
)


@pytest.fixture(params=["pandas", "polars"])
def engine(request, monkeypatch):
    """Keep both native engines independent of ambient engine overrides."""
    monkeypatch.setenv("SKYULF_ENGINE", request.param)
    return request.param


@pytest.mark.parametrize(
    "kind,calculator,applier",
    [
        ("LabelEncoder", LabelEncoderCalculator, LabelEncoderApplier),
        ("OrdinalEncoder", OrdinalEncoderCalculator, OrdinalEncoderApplier),
    ],
)
@pytest.mark.parametrize("missing_count", [0, 1, 2])
@pytest.mark.parametrize("embedded", [False, True])
def test_target_mapping_preserves_repeated_nan_labels(
    engine, kind, calculator, applier, missing_count, embedded
):
    """Recording a mapping must not invent classes or reject repeated missing labels."""
    values = [1.0, 2.0, *([np.nan] * missing_count)]
    data = {"feature": list(range(len(values)))}
    if embedded:
        data["target"] = values
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    labels = pd.Series(values, name="target") if engine == "pandas" else pl.Series("target", values)
    payload = frame if embedded else (frame, labels)
    config = {"columns": ["target"], "target_column": "target"}
    artifact = calculator().fit(payload, config)
    expected = applier().apply(payload, artifact)
    engineer = FeatureEngineer([{"name": "encode", "transformer": kind, "params": config}])

    actual, _ = engineer.fit_transform(payload, target_column="target")

    actual_labels = actual["target"] if embedded else actual[1]
    expected_labels = expected["target"] if embedded else expected[1]
    np.testing.assert_array_equal(np.asarray(actual_labels), np.asarray(expected_labels))
    steps = pickle.loads(pickle.dumps(engineer.fitted_steps))
    decoded = original_target_labels(steps, actual_labels)
    np.testing.assert_allclose(decoded.astype(float), values, equal_nan=True)
    assert len(steps[0]["artifact"].get("target_label_map", {})) == len(set(actual_labels))
    np.testing.assert_allclose(np.asarray(labels), values, equal_nan=True)


def test_untouched_nan_target_does_not_create_mapping(engine):
    """Encoding a feature must not create an unusable NaN-to-NaN target mapping."""
    data = {"feature": ["a", "b", "a", "b"], "target": [1.0, 2.0, np.nan, np.nan]}
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    engineer = FeatureEngineer(
        [{"name": "encode", "transformer": "LabelEncoder", "params": {"columns": ["feature"]}}]
    )

    output, _ = engineer.fit_transform(frame, target_column="target")

    np.testing.assert_allclose(np.asarray(output["target"]), data["target"], equal_nan=True)
    assert "target_label_map" not in engineer.fitted_steps[0]["artifact"]


def test_genuine_target_label_collision_still_raises():
    """Missing values and literal text must not silently share an original class identity."""
    frame = pd.DataFrame({"target": pd.Series(["nan", np.nan], dtype=object)})
    engineer = FeatureEngineer(
        [{"name": "encode", "transformer": "LabelEncoder", "params": {"columns": ["target"]}}]
    )

    with pytest.raises(ValueError, match="one-to-one"):
        engineer.fit_transform(frame, target_column="target")


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
@pytest.mark.parametrize("float_dtype", ["float32", "float64"])
@pytest.mark.parametrize("coerce", [False, True])
def test_float_integer_upper_bound_cannot_wrap(engine, dtype, float_dtype, coerce):
    """The exact exclusive integer boundary must raise or become null instead of wrapping."""
    info = np.iinfo(dtype)
    scalar = np.dtype(float_dtype).type
    boundary = scalar(info.max + 1)
    below = np.nextafter(boundary, scalar(-np.inf))
    values = np.array([0, below, boundary], dtype=float_dtype)
    frame = (
        pd.DataFrame({"value": values}) if engine == "pandas" else pl.DataFrame({"value": values})
    )
    original = np.asarray(frame["value"]).copy()
    params = CastingCalculator().fit(
        frame, {"columns": ["value"], "target_type": dtype, "coerce_on_error": coerce}
    )

    if coerce:
        output = CastingApplier().apply(frame, params)
        actual = output["value"].to_list()
        assert actual[:2] == [0, int(below)]
        assert pd.isna(actual[2])
        assert str(output["value"].dtype).lower() == dtype
    else:
        with pytest.raises((OverflowError, ValueError, pl.exceptions.InvalidOperationError)):
            CastingApplier().apply(frame, params)
    np.testing.assert_array_equal(np.asarray(frame["value"]), original)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_native_integer_extrema_retain_exact_precision(engine, dtype):
    """Fixing float comparisons must not reject or round exact native integer extrema."""
    info = np.iinfo(dtype)
    values = np.array([info.min, info.max], dtype=dtype)
    frame = (
        pd.DataFrame({"value": values}) if engine == "pandas" else pl.DataFrame({"value": values})
    )
    params = CastingCalculator().fit(
        frame, {"columns": ["value"], "target_type": dtype, "coerce_on_error": False}
    )

    output = CastingApplier().apply(frame, params)

    assert output["value"].to_list() == [int(info.min), int(info.max)]


@pytest.mark.parametrize("float_dtype", ["float32", "float64"])
@pytest.mark.parametrize("case", ["mixed", "all-missing", "empty"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_dataset_profile_counts_native_nan_as_missing(engine, float_dtype, case, wrapped):
    """Native NaNs must contribute to missingness and stay outside numeric summary statistics."""
    values = {"mixed": [2.0, np.nan, None, 4.0], "all-missing": [np.nan, None], "empty": []}[case]
    if engine == "pandas":
        frame = pd.DataFrame({"value": pd.Series(values, dtype=float_dtype)})
    else:
        dtype = pl.Float32 if float_dtype == "float32" else pl.Float64
        frame = pl.DataFrame({"value": pl.Series(values, dtype=dtype)})
    original = np.asarray(frame["value"]).copy()
    data = EngineRegistry.wrap(frame) if wrapped else frame

    artifact = DatasetProfileCalculator().fit(data, {})

    profile = artifact["profile"]
    assert profile["missing"]["value"] == (0 if case == "empty" else 2)
    assert profile["numeric_stats"]["value"]["count"] == (2 if case == "mixed" else 0)
    mean = profile["numeric_stats"]["value"]["mean"]
    assert mean == pytest.approx(3.0) if case == "mixed" else pd.isna(mean)
    assert DatasetProfileApplier().apply(data, cast(dict[str, Any], artifact)) is data
    np.testing.assert_allclose(np.asarray(frame["value"]), original, equal_nan=True)


@pytest.mark.parametrize(
    "calculator,applier",
    [
        (SimpleTransformationCalculator, SimpleTransformationApplier),
        (GeneralTransformationCalculator, GeneralTransformationApplier),
    ],
)
@pytest.mark.parametrize("method", ["log", "sqrt", "square", "cube_root", "reciprocal", "exp"])
@pytest.mark.parametrize("text_input", [False, True])
def test_math_transformations_accept_numeric_text(engine, calculator, applier, method, text_input):
    """Simple operations must coerce numeric text and preserve existing native float precision."""
    values = ["-4", "0", " 9 ", "invalid", None] if text_input else [-4.0, 0.0, 9.0, np.nan, None]
    source = pd.DataFrame({"value": pd.Series(values, dtype=object if text_input else "float32")})
    if engine == "pandas":
        frame: Any = source
    else:
        frame = pl.DataFrame(
            {"value": pl.Series(values, dtype=pl.String if text_input else pl.Float32)}
        )
    original = frame.clone() if engine == "polars" else cast(Any, frame).copy(deep=True)
    config = {"transformations": [{"column": "value", "method": method}]}
    reference = applier().apply(source, calculator().fit(source, config))
    params = calculator().fit(frame, config)

    output = applier().apply(frame, params)

    np.testing.assert_allclose(
        np.asarray(output["value"], dtype=float),
        reference["value"].to_numpy(),
        rtol=1e-6,
        equal_nan=True,
    )
    if not text_input:
        assert output["value"].to_numpy().dtype == reference["value"].to_numpy().dtype
    assert frame.equals(original)
