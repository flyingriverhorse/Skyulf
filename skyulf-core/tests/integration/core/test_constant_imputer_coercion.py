"""Explicit text constants from Canvas are fitted as exact per-column scalar values."""

import pandas as pd
import polars as pl
import pytest

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.integrations.databricks.training.shared.preprocessing_checks import seal_partition_probe
from skyulf.preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator


@pytest.mark.parametrize(
    "dtype,text,expected",
    [
        ("int64[pyarrow]", "9007199254740993", 9007199254740993),
        ("double[pyarrow]", "2.5", 2.5),
        ("bool[pyarrow]", "false", False),
        ("float16", "2.5", 2.5),
    ],
)
def test_arrow_and_float16_constants_keep_native_numeric_or_boolean_values(dtype, text, expected):
    """Arrow-backed pandas columns and float16 must not receive raw text from the fitted constant map."""
    frame = pd.DataFrame({"value": pd.Series([None], dtype=dtype)})
    config = {"columns": ["value"], "strategy": "constant", "fill_value": text}
    state = SimpleImputerCalculator().fit(frame, config)
    assert type(state["fill_values"]["value"]) is type(expected)
    result = SimpleImputerApplier().apply(frame, state)
    assert result["value"].to_list() == [expected]
    assert pd.api.types.is_numeric_dtype(result["value"].dtype) or pd.api.types.is_bool_dtype(
        result["value"].dtype
    )


@pytest.mark.parametrize("dtype,text", [("uint8[pyarrow]", "300"), ("float16", "1e10")])
def test_arrow_and_float16_constant_overflow_is_rejected_during_fit(dtype, text):
    """A text constant outside its numeric storage bounds must fail before replay."""
    frame = pd.DataFrame({"value": pd.Series([None], dtype=dtype)})
    with pytest.raises(ValueError, match="value"):
        SimpleImputerCalculator().fit(
            frame, {"columns": ["value"], "strategy": "constant", "fill_value": text}
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_constant_text_is_coerced_per_numeric_column_without_rewriting_text(engine):
    """A mixed numeric/text selection must retain each column's scalar semantics and schema."""
    frame = pd.DataFrame(
        {
            "integer": pd.Series([1, None], dtype="Int64"),
            "number": [1.5, None],
            "text": pd.Series(["a", None], dtype="string"),
        }
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    config = {"columns": list(frame.columns), "strategy": "constant", "fill_value": "0"}
    state = SimpleImputerCalculator().fit(native, config)
    result = SimpleImputerApplier().apply(native, state)
    assert state["fill_values"] == {"integer": 0, "number": 0.0, "text": "0"}
    assert type(state["fill_values"]["integer"]) is int
    assert result["integer"].to_list() == [1, 0]
    assert result["number"].to_list() == [1.5, 0.0]
    assert result["text"].to_list() == ["a", "0"]
    validated = SimpleImputerApplier.validate_fitted_state(dict(state))
    assert SimpleImputerApplier.resolve_fitted_config(config, validated)["fill_value"] == "0"
    with pytest.raises(ValueError):
        SimpleImputerApplier.resolve_fitted_config({**config, "fill_value": "1"}, validated)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_large_integer_constant_does_not_round_through_float(engine):
    """Integer text beyond 2**53 must remain exact in saved fill values and replay."""
    big = 2**53 + 1
    frame = pd.DataFrame({"value": pd.Series([1, None], dtype="Int64")})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    config = {"columns": ["value"], "strategy": "constant", "fill_value": str(big)}
    state = SimpleImputerCalculator().fit(native, config)
    result = SimpleImputerApplier().apply(native, state)
    assert type(state["fill_values"]["value"]) is int
    assert result["value"].to_list() == [1, big]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "dtype,value",
    [
        ("Int8", "1.5"),
        ("Int8", "128"),
        ("UInt64", "-1"),
        ("float64", "inf"),
        ("boolean", "0"),
        ("boolean", "maybe"),
    ],
)
def test_invalid_or_lossy_constant_strings_fail_during_fit(engine, dtype, value):
    """Invalid casts must fail with the selected column name before a fitted artifact is published."""
    frame = pd.DataFrame({"value": pd.Series([None], dtype=dtype)})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    with pytest.raises(ValueError, match="value"):
        SimpleImputerCalculator().fit(
            native, {"columns": ["value"], "strategy": "constant", "fill_value": value}
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("text,expected", [("true", True), ("FALSE", False)])
def test_boolean_text_constants_use_explicit_boolean_words(engine, text, expected):
    """Boolean values must be parsed explicitly rather than using Python truthiness on text."""
    frame = pd.DataFrame({"value": pd.Series([None], dtype="boolean")})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    config = {"columns": ["value"], "strategy": "constant", "fill_value": text}
    state = SimpleImputerCalculator().fit(native, config)
    assert SimpleImputerApplier().apply(native, state)["value"].to_list() == [expected]
    assert type(state["fill_values"]["value"]) is bool
    validated = SimpleImputerApplier.validate_fitted_state(dict(state))
    assert SimpleImputerApplier.resolve_fitted_config(config, validated)["fill_value"] == text


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_numeric_text_constant_keeps_real_saved_model_predictions(engine, tmp_path):
    """A fitted pipeline must save numeric fills from UI text and replay without per-request inference."""
    frame = pd.DataFrame({"value": [2.0, None, 6.0, 8.0], "target": [1.0, 0.0, 3.0, 4.0]})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "fill",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["value"], "strategy": "constant", "fill_value": "0"},
                }
            ],
            "modeling": {"type": "decision_tree_regressor", "params": {"random_state": 42}},
        }
    )
    pipeline.fit(pl.from_pandas(frame) if engine == "polars" else frame, target_column="target")
    save_pipeline(pipeline, tmp_path / "constant")
    restored = load_pipeline(tmp_path / "constant")
    query = frame.drop(columns="target")
    for apply_engine in ("pandas", "polars"):
        result = predict_pipeline(
            pl.from_pandas(query) if apply_engine == "polars" else query, restored
        )
        assert result["prediction"].to_list() == frame["target"].to_list()
    assert (tmp_path / "constant").exists()


def test_constant_text_uses_existing_sealed_partition_admission(tmp_path):
    """Coercing a UI scalar must keep strict saved-state admission without adding a new codec path."""
    frame = pd.DataFrame({"value": [1.0, None, 3.0, 4.0] * 5, "target": [1.0, 0.0, 3.0, 4.0] * 5})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "fill",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["value"], "strategy": "constant", "fill_value": "0"},
                }
            ],
            "modeling": {"type": "ridge_regression"},
        }
    )
    pipeline.fit(frame, target_column="target")
    save_pipeline(pipeline, tmp_path / "probe")
    artifact = load_pipeline(tmp_path / "probe")
    artifact = seal_partition_probe(artifact, frame, tmp_path / "probe", enabled=True)
    certificate = require_partition_safe_pipeline(artifact)
    assert certificate.steps[0].action == "apply"
