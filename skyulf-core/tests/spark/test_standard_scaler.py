"""Native scaling must preserve learned statistics and distributed execution."""

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.core.portable_state import decode_state, encode_state
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.preprocessing.scaling.standard import StandardScalerApplier, StandardScalerCalculator


@pytest.mark.parametrize("fit_engine", ["pandas", "polars", "spark"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars", "spark"])
@pytest.mark.parametrize(
    "with_mean,with_std", [(True, True), (True, False), (False, True), (False, False)]
)
def test_cross_engine_flags_population_variance_and_order(
    spark, fit_engine, apply_engine, with_mean, with_std
):
    """Sample variance, relearning or positional column mapping would change held-out outputs."""
    local = pd.DataFrame({"x": [1.0, 3.0, np.nan], "z": [10.0, 14.0, np.nan]})
    train = {
        "pandas": lambda: local,
        "polars": lambda: pl.from_pandas(local),
        "spark": lambda: spark.createDataFrame(
            [(1.0, 10.0), (3.0, 14.0), (None, None)], "x double, z double"
        ),
    }[fit_engine]()
    state = StandardScalerCalculator().fit(
        train, {"columns": ["z", "x"], "with_mean": with_mean, "with_std": with_std}
    )
    _, state = decode_state(encode_state("StandardScaler", state, max_bytes=8192))
    assert state["columns"] == ["z", "x"]
    assert state["mean"] == ([12.0, 2.0] if with_mean or with_std else None)
    assert state["var"] == ([4.0, 1.0] if with_std else None)
    assert state["scale"] == ([2.0, 1.0] if with_std else None)
    held = pd.DataFrame({"id": [20, 10], "x": [100.0, np.nan], "z": [20.0, np.nan]})
    batch = {
        "pandas": lambda: held,
        "polars": lambda: pl.from_pandas(held),
        "spark": lambda: spark.createDataFrame(
            [(20, 100.0, 20.0), (10, None, None)], "id long, x double, z double"
        ).repartition(2),
    }[apply_engine]()
    output = StandardScalerApplier().apply(batch, state)
    records = {
        "pandas": lambda: output.to_dict("records"),
        "polars": lambda: output.to_dicts(),
        "spark": lambda: [row.asDict() for row in output.collect()],
    }[apply_engine]()
    rows = {row["id"]: row for row in records}
    expected = {
        (True, True): [98.0, 4.0],
        (True, False): [98.0, 8.0],
        (False, True): [100.0, 10.0],
        (False, False): [100.0, 20.0],
    }[(with_mean, with_std)]
    np.testing.assert_allclose([rows[20]["x"], rows[20]["z"]], expected, rtol=1e-10, atol=1e-12)
    assert pd.isna(rows[10]["x"]) and pd.isna(rows[10]["z"])
    assert list(output.columns) == ["id", "x", "z"]


def test_null_nan_constant_and_single_observation(spark):
    """Missing values are ignored for learning; zero variance uses unit scale."""
    train = spark.createDataFrame(
        [
            (1.0, 5.0, None, 7.0),
            (3.0, 5.0, None, None),
            (None, 5.0, None, None),
            (float("nan"), None, None, None),
        ],
        "x double, constant double, empty double, single double",
    )
    state = StandardScalerCalculator().fit(train, {"columns": train.columns})
    assert state["mean"] is not None and state["var"] is not None and state["scale"] is not None
    np.testing.assert_allclose(state["mean"], [2, 5, np.nan, 7], equal_nan=True)
    np.testing.assert_allclose(state["var"], [1, 0, np.nan, 0], equal_nan=True)
    np.testing.assert_allclose(state["scale"], [1, 1, np.nan, 1], equal_nan=True)
    batch = spark.createDataFrame([(3.0, 10.0, 4.0, 8.0)], train.schema)
    row = StandardScalerApplier().apply(batch, state).first()
    assert [row.x, row.constant, row.single] == [1.0, 5.0, 1.0]
    assert math.isnan(row.empty)


@pytest.mark.parametrize("partitions", [1, 2, 7])
def test_large_offset_small_variance(spark, partitions):
    """Stable distributed variance must retain spread hidden by E[x*x] - E[x]**2."""
    values = [1e12 + offset for offset in (0.0, 0.125, 0.25, 0.375)]
    frame = spark.createDataFrame([(value,) for value in values], "x double").repartition(
        partitions
    )
    state = StandardScalerCalculator().fit(frame, {"columns": ["x"]})
    # Double rounding at this offset is about 1e-4, appreciable versus a 0.14 std.
    assert state["var"] == pytest.approx([0.01953125], rel=0.002, abs=1e-12)
    reference = StandardScalerCalculator().fit(pd.DataFrame({"x": values}), {"columns": ["x"]})
    held = spark.createDataFrame([(1e12 + 0.5,)], "x double")
    actual = StandardScalerApplier().apply(held, state).first().x
    expected = (
        StandardScalerApplier().apply(pd.DataFrame({"x": [1e12 + 0.5]}), reference).iloc[0, 0]
    )
    assert actual == pytest.approx(expected, rel=0.002, abs=1e-12)


def test_near_constant_uses_sklearn_error_bound(spark):
    """Almost constant high-offset features must not be amplified by tiny scales."""
    values = [1e12, 1e12 + 0.0001220703125]
    state = StandardScalerCalculator().fit(
        spark.createDataFrame([(v,) for v in values], "x double"), {"columns": ["x"]}
    )
    assert state["var"] is not None and state["var"][0] > 0
    assert state["scale"] == [1.0]


@pytest.mark.parametrize("with_std", [True, False])
def test_large_constant_mean_does_not_accumulate_roundoff(spark, with_std):
    """Summation error must not turn a constant training feature into a nonzero output."""
    value = 1e12 + 0.125
    frame = spark.range(10000, numPartitions=2).selectExpr("1000000000000.125D AS x")
    state = StandardScalerCalculator().fit(frame, {"columns": ["x"], "with_std": with_std})
    assert state["mean"] == [value]
    assert StandardScalerApplier().apply(frame, state).first().x == 0.0


@pytest.mark.parametrize("large,small", [(1e16, 1.0), (1e8, 1e-8)])
def test_mixed_sign_mean_does_not_lose_small_observations(spark, large, small):
    """Reference shifting must not discard near-zero values in a mixed-sign distribution."""
    frame = spark.createDataFrame([(-large,), (large,), (small,)], "x double").coalesce(1)
    state = StandardScalerCalculator().fit(frame, {"columns": ["x"], "with_std": False})
    assert state["mean"] == pytest.approx([small / 3], rel=1e-10, abs=1e-20)
    held = spark.createDataFrame([(small,)], "x double")
    assert StandardScalerApplier().apply(held, state).first().x == pytest.approx(2 * small / 3)


def test_explicit_empty_training_fails_but_empty_selection_is_noop(spark):
    """Zero rows cannot produce fitted statistics; explicit deselection remains valid."""
    frame = spark.createDataFrame([], "x double")
    with pytest.raises(ValueError, match="empty|0 sample"):
        StandardScalerCalculator().fit(frame, {"columns": ["x"]})
    assert StandardScalerCalculator().fit(frame, {"columns": []}) == {}


def test_native_apply_is_lazy_quotes_names_and_guards_zero_scale(spark, monkeypatch, capsys):
    """Literal names and zero scales must work without actions or Python UDFs."""
    frame = spark.createDataFrame([(4, 9.0)], "`a.b` long, `a``b` double")
    state = {
        "type": "standard_scaler",
        "columns": ["a`b", "a.b"],
        "mean": [1.0, 2.0],
        "var": [0.0, 0.0],
        "scale": [0.0, 0.0],
        "with_mean": True,
        "with_std": True,
    }

    def forbidden(*args, **kwargs):
        """Any action during plan construction would violate lazy native apply."""
        pytest.fail("Action in native apply")

    with monkeypatch.context() as patch:
        for name in ("collect", "count", "first", "toPandas", "toArrow"):
            patch.setattr(type(frame), name, forbidden)
        output = StandardScalerApplier().apply(frame, state)
    output.explain(extended=True)
    plan = capsys.readouterr().out
    assert not any(name in plan for name in ("PythonUDF", "BatchEvalPython", "ArrowEvalPython"))
    assert tuple(output.first()) == (2.0, 8.0)


def test_pipeline_auto_selection_and_protected_columns(spark):
    """Automatic scaling must exclude keys, target, strings, binary and constant features."""
    data = spark.createDataFrame(
        [(1, 2.0, 1.0, 5.0, "a", 10.0), (2, 4.0, 0.0, 5.0, "b", 20.0)],
        "id long, x double, binary double, fixed double, text string, label double",
    )
    engineer = FeatureEngineer(
        [{"name": "scale", "transformer": "StandardScaler"}],
        frame_spec=FrameSpec(("id",), "label"),
        execution_options=ExecutionOptions("spark"),
    )
    output, _ = engineer.fit_transform(data.repartition(2))
    assert engineer.fitted_steps[0]["artifact"]["columns"] == ["x"]
    assert [tuple(row) for row in output.orderBy("id").collect()] == [
        (1, -1.0, 1.0, 5.0, "a", 10.0),
        (2, 1.0, 0.0, 5.0, "b", 20.0),
    ]
    assert engineer.transform(data.drop("label")).columns == data.columns[:-1]


@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
@pytest.mark.parametrize("flags", [(True, True), (False, False)])
def test_infinite_training_rejected(spark, value, flags):
    """Disabled centering/scaling must not hide invalid training observations."""
    frame = spark.createDataFrame([(1.0,), (value,)], "x double")
    with pytest.raises(ValueError, match="finite|infinity"):
        StandardScalerCalculator().fit(
            frame, {"columns": ["x"], "with_mean": flags[0], "with_std": flags[1]}
        )


def test_invalid_dtype_fails_before_action(spark, monkeypatch):
    """String parsing must not silently enter a numeric Spark scaling pipeline."""
    frame = spark.createDataFrame([("1",), ("3",)], "x string")

    def forbidden(*args, **kwargs):
        """No distributed work should precede schema rejection."""
        pytest.fail("Action before dtype rejection")

    monkeypatch.setattr(type(frame), "first", forbidden)
    with pytest.raises(TypeError, match="numeric"):
        StandardScalerCalculator().fit(frame, {"columns": ["x"]})


def test_fit_collects_only_bounded_statistics(spark, monkeypatch):
    """Learning must aggregate in Spark and transfer O(columns) state instead of input rows."""
    frame = spark.range(10000).selectExpr("CAST(id AS DOUBLE) AS x", "CAST(id + 5 AS DOUBLE) AS z")
    original_first = type(frame).first
    shapes = []

    def record_first(aggregate):
        """Observe the materialization boundary without replacing Spark computation."""
        row = original_first(aggregate)
        shapes.append(len(row))
        return row

    def forbidden(*args, **kwargs):
        """Direct local conversion must never occur inside fitting."""
        pytest.fail("Local conversion during native fit")

    with monkeypatch.context() as patch:
        patch.setattr(type(frame), "first", record_first)
        patch.setattr(type(frame), "toPandas", forbidden)
        patch.setattr(type(frame), "toArrow", forbidden)
        state = StandardScalerCalculator().fit(frame, {"columns": ["x", "z"]})
    assert len(shapes) == 2 and all(width <= 10 for width in shapes)
    assert state["mean"] == [4999.5, 5004.5]
    assert state["var"] == pytest.approx([8333333.25, 8333333.25], rel=1e-10, abs=1e-12)


def test_single_row_and_explicit_boolean(spark):
    """Explicit booleans and one-observation features follow local numeric scaling."""
    frame = spark.createDataFrame([(True, 7)], "flag boolean, x long")
    state = StandardScalerCalculator().fit(frame, {"columns": ["x", "flag"]})
    assert state["mean"] == [7.0, 1.0]
    assert state["scale"] == [1.0, 1.0]
    assert tuple(StandardScalerApplier().apply(frame, state).first()) == (0.0, 0.0)


@pytest.mark.parametrize("columns", [["x", "x"], ["x", "X"]])
def test_ambiguous_input_names_fail_before_actions(spark, columns):
    """Spark identifier resolution must not silently select or drop the wrong feature."""
    previous = spark.conf.get("spark.sql.caseSensitive")
    spark.conf.set("spark.sql.caseSensitive", "false")
    try:
        frame = spark.createDataFrame([(1.0, 3.0)], columns)
        with pytest.raises(ValueError, match="Duplicate"):
            StandardScalerCalculator().fit(frame, {"columns": ["x"]})
    finally:
        spark.conf.set("spark.sql.caseSensitive", previous)
