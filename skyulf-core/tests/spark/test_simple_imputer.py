"""Native imputation must learn bounded state and keep distributed input in Spark."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.core.portable_state import decode_state, encode_state
from skyulf.preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator
from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest.mark.parametrize("fit_engine", ["pandas", "polars", "spark"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars", "spark"])
@pytest.mark.parametrize("strategy", ["mean", "constant"])
def test_cross_engine_state_uses_training_only(spark, fit_engine, apply_engine, strategy):
    """Held-out values must never replace learned fill values during cross-engine apply."""
    local = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    train = {
        "pandas": lambda: local,
        "polars": lambda: pl.from_pandas(local),
        "spark": lambda: spark.createDataFrame([(1.0,), (None,), (3.0,)], "x double"),
    }[fit_engine]()
    params = SimpleImputerCalculator().fit(
        train, {"columns": ["x"], "strategy": strategy, "fill_value": 7.0}
    )
    _, params = decode_state(encode_state("SimpleImputer", params, max_bytes=8192))
    held_out = pd.DataFrame({"x": [100.0, np.nan]})
    data = {
        "pandas": lambda: held_out,
        "polars": lambda: pl.from_pandas(held_out),
        "spark": lambda: spark.createDataFrame([(100.0,), (None,)], "x double"),
    }[apply_engine]()
    out = SimpleImputerApplier().apply(data, params)
    values = [row.x for row in out.collect()] if apply_engine == "spark" else out["x"].to_list()
    assert values == [100.0, 2.0 if strategy == "mean" else 7.0]
    assert params["missing_counts"] == {"x": 1}


def test_native_mean_counts_null_nan_and_preserves_all_empty(spark):
    """All-missing numeric columns must remain missing rather than receive invented means."""
    train = spark.createDataFrame(
        [(1.0, None), (None, None), (float("nan"), None), (3.0, None)], "x double, empty double"
    )
    state = SimpleImputerCalculator().fit(train, {"columns": ["x", "empty"]})
    assert state["fill_values"] == {"x": 2.0, "empty": None}
    assert state["missing_counts"] == {"x": 2, "empty": 4}
    out = SimpleImputerApplier().apply(train, state).collect()
    assert [row.x for row in out] == [1.0, 2.0, 2.0, 3.0]
    assert all(row.empty is None for row in out)


def test_apply_is_lazy_and_contains_no_python_udf(spark, monkeypatch, capsys):
    """Transform must consist of native expressions without an implicit data action."""
    frame = spark.createDataFrame([(None,), (5.0,)], "`a.b` double")
    state = {
        "type": "simple_imputer",
        "strategy": "constant",
        "columns": ["a.b"],
        "fill_values": {"a.b": 4.0},
        "missing_counts": {"a.b": 1},
        "total_missing": 1,
    }

    def forbidden(*args, **kwargs):
        """Catch driver materialization while constructing the apply plan."""
        pytest.fail("Action in native apply")

    with monkeypatch.context() as patch:
        for name in ("collect", "count", "toPandas", "toArrow", "first"):
            patch.setattr(type(frame), name, forbidden)
        output = SimpleImputerApplier().apply(frame, state)
    output.explain(extended=True)
    plan = capsys.readouterr().out
    assert (
        "PythonUDF" not in plan and "BatchEvalPython" not in plan and "ArrowEvalPython" not in plan
    )
    assert [row[0] for row in output.collect()] == [4.0, 5.0]


@pytest.mark.parametrize("strategy", ["median", "most_frequent", "mode"])
def test_unsupported_strategy_fails_before_actions(spark, monkeypatch, strategy):
    """Native mean must never stand in for unsupported fitting strategies."""
    frame = spark.range(2)

    def forbidden(*args, **kwargs):
        """Make an early action observable."""
        pytest.fail("Action before unsupported rejection")

    monkeypatch.setattr(type(frame), "collect", forbidden)
    with pytest.raises(UnsupportedExecutionError):
        SimpleImputerCalculator().fit(frame, {"columns": ["id"], "strategy": strategy})


def test_feature_engineer_defaults_keys_and_state_budget(spark):
    """The default strategy must normalize before preflight and preserve keys and labels."""
    data = spark.createDataFrame([(1, None, 0), (2, 4.0, 1)], "id long, x double, label long")
    steps = [{"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}}]
    engineer = FeatureEngineer(
        steps, frame_spec=FrameSpec(("id",), "label"), execution_options=ExecutionOptions("spark")
    )
    output, _ = engineer.fit_transform(data.repartition(2))
    assert sorted(tuple(row) for row in output.collect()) == [(1, 4.0, 0), (2, 4.0, 1)]
    limited = FeatureEngineer(
        steps,
        frame_spec=FrameSpec(("id",), "label"),
        execution_options=ExecutionOptions("spark", state_max_bytes=10),
    )
    with pytest.raises(ValueError, match="max_bytes"):
        limited.fit_transform(data)


def test_empty_selection_does_not_become_auto_selection(spark):
    """An explicit deselection must remain a no-op even through the Spark runner."""
    data = spark.createDataFrame([(1, None), (2, 4.0)], "id long, x double")
    engineer = FeatureEngineer(
        [{"name": "fill", "transformer": "SimpleImputer", "params": {"columns": []}}],
        frame_spec=FrameSpec(("id",)),
        execution_options=ExecutionOptions("spark"),
    )
    output, _ = engineer.fit_transform(data)
    assert engineer.fitted_steps[0]["artifact"] == {}
    assert output.orderBy("id").first().x is None


def test_auto_numeric_selection_matches_local_heuristics(spark):
    """Automatic mean must skip strings, constants and binary columns without collecting raw data."""
    data = spark.createDataFrame(
        [(1, 2.0, 1.0, "a"), (2, 4.0, 0.0, None), (3, None, None, "b")],
        "id long, x double, binary double, text string",
    )
    engineer = FeatureEngineer(
        [{"name": "fill", "transformer": "SimpleImputer"}],
        frame_spec=FrameSpec(("id",)),
        execution_options=ExecutionOptions("spark"),
    )
    output, _ = engineer.fit_transform(data)
    assert engineer.fitted_steps[0]["artifact"]["columns"] == ["x"]
    assert output.orderBy("id").collect()[2].x == 3.0


def test_string_constant_and_invalid_mean(spark):
    """String constants are explicit and cannot masquerade as numerical means."""
    data = spark.createDataFrame([(None,), ("a",)], "text string")
    state = SimpleImputerCalculator().fit(
        data, {"columns": ["text"], "strategy": "constant", "fill_value": "missing"}
    )
    assert [r.text for r in SimpleImputerApplier().apply(data, state).collect()] == ["missing", "a"]
    with pytest.raises(TypeError, match="numeric"):
        SimpleImputerCalculator().fit(data, {"columns": ["text"], "strategy": "mean"})
    with pytest.raises(ValueError, match="explicit"):
        SimpleImputerCalculator().fit(data, {"columns": ["text"], "strategy": "constant"})


def test_fit_collects_one_aggregate_and_empty_selection_is_lazy(spark, monkeypatch):
    """Learning may collect bounded statistics, but never the training frame itself."""
    data = spark.createDataFrame([(1.0,), (None,), (3.0,)], "x double")
    original = type(data).first
    schemas = []

    def record_first(frame):
        """Record the aggregate boundary without blocking its execution."""
        schemas.append(frame.columns)
        return original(frame)

    monkeypatch.setattr(type(data), "first", record_first)
    state = SimpleImputerCalculator().fit(data, {"columns": ["x"]})
    assert state["fill_values"] == {"x": 2.0}
    assert schemas == [["n0", "v0"]]
    assert SimpleImputerCalculator().fit(data, {"columns": []}) == {}
    assert len(schemas) == 1


def test_integer_constant_preserves_large_values_and_mean_promotes(spark):
    """Constant filling must not round signed integers through a double conversion."""
    large = 2**53 + 1
    data = spark.createDataFrame([(large,), (None,)], "x long")
    state = SimpleImputerCalculator().fit(
        data, {"columns": ["x"], "strategy": "constant", "fill_value": large}
    )
    out = SimpleImputerApplier().apply(data, state)
    assert out.dtypes == [("x", "bigint")]
    assert [row.x for row in out.collect()] == [large, large]
    small = spark.createDataFrame([(1,), (2,), (None,)], "x long")
    mean = SimpleImputerCalculator().fit(small, {"columns": ["x"]})
    assert [row.x for row in SimpleImputerApplier().apply(small, mean).collect()] == [1, 2, 1.5]


def test_apply_rejects_ambiguous_columns(spark):
    """A direct applier must not silently collapse duplicate input column names."""
    data = spark.createDataFrame([(1.0, 2.0)], ["x", "x"])
    state = {
        "type": "simple_imputer",
        "strategy": "mean",
        "columns": ["x"],
        "fill_values": {"x": 2.0},
        "missing_counts": {"x": 1},
        "total_missing": 1,
    }
    with pytest.raises(ValueError, match="Duplicate"):
        SimpleImputerApplier().apply(data, state)


def test_transform_state_budget_fails_before_validation_actions(spark, monkeypatch):
    """Oversized fitted state must fail before distributed transform validation starts."""
    data = spark.createDataFrame([(1, None), (2, 4.0)], "id long, x double")
    engineer = FeatureEngineer(
        [{"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}}],
        frame_spec=FrameSpec(("id",)),
        execution_options=ExecutionOptions("spark"),
    )
    engineer.fit_transform(data)
    engineer.execution_options = ExecutionOptions("spark", state_max_bytes=10)

    def forbidden(*args, **kwargs):
        """Make any data action preceding state validation fail the test."""
        pytest.fail("Action before state validation")

    monkeypatch.setattr(type(data), "collect", forbidden)
    with pytest.raises(ValueError, match="max_bytes"):
        engineer.transform(data)


@pytest.mark.parametrize("engine", ["pandas", "polars", "spark"])
def test_auto_selection_keeps_three_near_binary_values(spark, engine):
    """Tolerance near zero/one must not turn three distinct values into a binary feature."""
    local = pd.DataFrame({"x": [0.0, 1e-9, 1.0, np.nan]})
    data = {
        "pandas": lambda: local,
        "polars": lambda: pl.from_pandas(local),
        "spark": lambda: spark.createDataFrame([(0.0,), (1e-9,), (1.0,), (None,)], "x double"),
    }[engine]()
    state = SimpleImputerCalculator().fit(data, {})
    assert state["columns"] == ["x"]
    assert state["fill_values"]["x"] == pytest.approx((1 + 1e-9) / 3)


@pytest.mark.parametrize("fitted_columns", [["x"], ["x", "X"]])
def test_apply_rejects_case_collisions_from_imported_state(spark, fitted_columns):
    """Restoring learned columns must not create ambiguous Spark output names."""
    previous = spark.conf.get("spark.sql.caseSensitive")
    spark.conf.set("spark.sql.caseSensitive", "false")
    try:
        data = spark.createDataFrame([(None,), (100.0,)], "X double")
        state = {
            "type": "simple_imputer",
            "strategy": "mean",
            "columns": fitted_columns,
            "fill_values": dict.fromkeys(fitted_columns, 2.0),
            "missing_counts": dict.fromkeys(fitted_columns, 1),
            "total_missing": len(fitted_columns),
        }
        with pytest.raises(ValueError, match="collide"):
            SimpleImputerApplier().apply(data, state)
    finally:
        spark.conf.set("spark.sql.caseSensitive", previous)
