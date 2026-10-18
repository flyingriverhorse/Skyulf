"""Schema-only previews must abstain when native output dtypes need row values or engine identity."""

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing import SkyulfSchema
from skyulf.preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator
from skyulf.preprocessing.time_series.lag import LagFeaturesApplier, LagFeaturesCalculator
from skyulf.preprocessing.time_series.rolling import (
    RollingAggregateApplier,
    RollingAggregateCalculator,
)


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("dtype", ["int64", "Int64", "float32", "Float32", "unknown"])
def test_simple_statistic_preview_abstains_for_value_or_engine_dependent_types(strategy, dtype):
    """Observed pandas integers and nullable or Polars inputs need different promotions."""
    schema = SkyulfSchema.from_columns(["value"], {"value": dtype})
    assert (
        SimpleImputerCalculator().infer_output_schema(
            schema, {"columns": ["value"], "strategy": strategy}
        )
        is None
    )


@pytest.mark.parametrize("selection", [{}, {"columns": None}])
def test_simple_statistic_preview_abstains_for_automatic_selection(selection):
    """Schema labels cannot reproduce the value-dependent automatic numeric selection."""
    schema = SkyulfSchema.from_columns(["value"], {"value": "float64"})
    assert SimpleImputerCalculator().infer_output_schema(schema, selection) is None


@pytest.mark.parametrize(
    "dtype,strategy,fill_value,output_dtype",
    [
        ("Float32", "most_frequent", None, "float64"),
        ("Float32", "constant", "2.5", "float64"),
        ("Int64", "constant", 2.5, "float64"),
        ("float64", "constant", True, "object"),
    ],
)
def test_simple_non_statistic_preview_abstains_when_native_fill_changes_type(
    dtype, strategy, fill_value, output_dtype
):
    """Mode and constant fills can widen nullable numbers or turn doubles into objects."""
    frame = pd.DataFrame({"value": pd.Series([1, None, 3], dtype=dtype)})
    config = {"columns": ["value"], "strategy": strategy, "fill_value": fill_value}
    calc = SimpleImputerCalculator()
    result = SimpleImputerApplier().apply(frame, calc.fit(frame, config))
    assert str(result["value"].dtype) == output_dtype
    assert calc.infer_output_schema(SkyulfSchema.from_dataframe(frame), config) is None


@pytest.mark.parametrize(
    "calculator", [SimpleImputerCalculator, LagFeaturesCalculator, RollingAggregateCalculator]
)
@pytest.mark.parametrize("columns", [[], ["absent"]])
def test_empty_resolved_selection_preserves_schema_and_real_fit(calculator, columns):
    """Explicit empty and missing-only selections are no-ops rather than unknown schemas."""
    frame = pd.DataFrame({"value": [1, 2]})
    schema = SkyulfSchema.from_dataframe(frame)
    config = {"columns": columns}
    fitted = calculator().fit(frame, config)
    assert not fitted or fitted["columns"] == columns
    assert calculator().infer_output_schema(schema, config) == schema


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
def test_simple_float64_preview_matches_real_null_and_observed_outputs(engine, strategy):
    """Known double inputs retain double output despite native imputation details."""
    frame = pd.DataFrame({"value": [1.0, None, 3.0]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    config = {"columns": ["value"], "strategy": strategy}
    calc = SimpleImputerCalculator()
    preview = calc.infer_output_schema(SkyulfSchema.from_dataframe(native), config)
    result = SimpleImputerApplier().apply(native, calc.fit(native, config))
    assert preview is not None
    assert preview.dtypes["value"].lower() == str(result["value"].dtype).lower() == "float64"
    fill = {"mean": 2.0, "median": 2.0, "most_frequent": 1.0, "constant": 0.0}[strategy]
    assert result["value"].to_list() == [1.0, fill, 3.0]


@pytest.mark.parametrize("dtype", ["int64", "uint8", "bool", "unknown"])
def test_lag_preview_abstains_for_native_integer_boolean_and_unknown_types(dtype):
    """Inserted missing rows change pandas native integer and Boolean dtypes but not Polars."""
    schema = SkyulfSchema.from_columns(["value"], {"value": dtype})
    assert LagFeaturesCalculator().infer_output_schema(schema, {"columns": ["value"]}) is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "dtype,values",
    [
        ("float32", [1.0, 2.0]),
        ("Int64", [1, 2]),
        ("boolean", [True, False]),
        ("string", ["a", "b"]),
    ],
)
def test_lag_preview_keeps_shift_stable_native_types(engine, empty, dtype, values):
    """Nullable integers, booleans, strings and floats retain their native dtype even for empty apply."""
    frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    config = {"columns": ["value"], "lags": [1]}
    calc = LagFeaturesCalculator()
    state = calc.fit(native, config)
    preview = calc.infer_output_schema(SkyulfSchema.from_dataframe(native), config)
    request = native.head(0) if empty else native
    result = LagFeaturesApplier().apply(request, state)
    assert preview is not None
    assert preview.dtypes["value_lag_1"] == str(result["value_lag_1"].dtype)
    assert len(result) == (0 if empty else 2)


@pytest.mark.parametrize(
    "dtype,aggregations",
    [
        ("int64", ["sum"]),
        ("Int64", ["min"]),
        ("UInt32", ["max"]),
        ("float32", ["mean"]),
        ("Float32", ["std"]),
        ("unknown", ["mean"]),
    ],
)
def test_rolling_preview_abstains_when_engines_disagree(dtype, aggregations):
    """Polars retains narrow or integral aggregate outputs where pandas returns doubles."""
    schema = SkyulfSchema.from_columns(["value"], {"value": dtype})
    assert (
        RollingAggregateCalculator().infer_output_schema(
            schema, {"columns": ["value"], "aggregations": aggregations}
        )
        is None
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "dtype,aggregations",
    [
        ("int64", ["mean", "median", "std"]),
        ("float64", ["sum", "min", "max", "mean", "median", "std"]),
    ],
)
def test_rolling_preview_retains_provably_double_outputs(engine, empty, dtype, aggregations):
    """Double sources and integer statistical aggregates have predictable double output on both engines."""
    frame = pd.DataFrame({"value": pd.Series([1, 2, 3], dtype=dtype), "group": ["a", "a", "a"]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    config = {
        "columns": ["value"],
        "aggregations": aggregations,
        "group_by": ["group"],
        "window": 2,
    }
    calc = RollingAggregateCalculator()
    state = calc.fit(native, config)
    preview = calc.infer_output_schema(SkyulfSchema.from_dataframe(native), config)
    result = RollingAggregateApplier().apply(native.head(0) if empty else native, state)
    assert preview is not None
    for aggregation in aggregations:
        column = f"value_roll_{aggregation}_2"
        assert preview.dtypes[column] == str(result[column].dtype).lower() == "float64"
    assert len(result) == (0 if empty else 3)
