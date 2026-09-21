"""Pin Spark detection, lazy operations and the local conversion boundary."""

import pytest

from skyulf.engines import (
    DistributedDataFrame,
    EngineRegistry,
    SkyulfDataFrame,
    SkyulfSparkWrapper,
    get_engine,
)
from skyulf.engines.sklearn_bridge import SklearnBridge


def test_detection_preserves_default(spark):
    """Recognized distributed inputs must not change unrelated local jobs."""
    before = get_engine()
    frame = spark.range(6).repartition(2)
    engine = get_engine(frame)
    assert engine.name == "spark"
    assert engine.is_compatible(frame)
    assert get_engine() is before
    assert get_engine(engine.wrap(frame)) is engine


def test_wrapper_selects_literal_columns_and_empty_rows(spark):
    """Projection must preserve special column names and zero-column row counts."""
    frame = spark.createDataFrame([(1, 2), (3, 4)], ["a.b", "c`d"])
    wrapped = EngineRegistry.wrap(frame)
    assert isinstance(wrapped, SkyulfSparkWrapper)
    assert isinstance(wrapped, DistributedDataFrame)
    assert not isinstance(wrapped, SkyulfDataFrame)
    assert wrapped.to_native() is frame
    assert wrapped.schema == frame.schema
    assert wrapped.columns == ["a.b", "c`d"]
    selected = wrapped.select(["c`d", "a.b"]).to_native()
    assert [tuple(row) for row in selected.collect()] == [(2, 1), (4, 3)]
    assert wrapped.select([]).to_native().count() == 2


@pytest.mark.parametrize("operation", ["len", "shape", "to_pandas", "to_numpy", "to_arrow"])
def test_wrapper_blocks_local_materialization(spark, monkeypatch, operation):
    """A local API must fail before triggering any distributed data action."""
    frame = spark.range(4)
    wrapped = EngineRegistry.wrap(frame)
    assert isinstance(wrapped, SkyulfSparkWrapper)

    def forbidden(*args, **kwargs):
        """Detect driver materialization even when the final exception is correct."""
        pytest.fail("Unexpected Spark action")

    for name in ("collect", "count", "toPandas", "toArrow"):
        monkeypatch.setattr(type(frame), name, forbidden)
    with pytest.raises(TypeError, match="Spark.*distributed"):
        if operation == "len":
            len(wrapped)
        elif operation == "shape":
            _ = wrapped.shape
        else:
            getattr(wrapped, operation)()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("target", [False, True])
def test_sklearn_rejects_distributed_input(spark, wrapped, target):
    """Both model features and labels must avoid implicit driver collection."""
    frame = spark.range(4)
    data = EngineRegistry.wrap(frame) if wrapped else frame
    data = ([[1], [2]], data) if target else data
    with pytest.raises(TypeError, match="Spark.*distributed"):
        SklearnBridge.to_sklearn(data, validate_features=True)


def test_engine_blocks_local_creation_and_numpy(spark):
    """Callers must provide their own Spark session and explicit native creation."""
    engine = get_engine(spark.range(1))
    with pytest.raises(TypeError, match="Spark.*distributed"):
        engine.to_numpy(spark.range(1))
    for method in (engine.from_pandas, engine.create_dataframe):
        with pytest.raises(TypeError, match="SparkSession"):
            method({"a": [1]})


def test_wrap_and_select_do_not_start_sessions_or_collect(spark, monkeypatch):
    """Adapter metadata and projection must stay lazy in a caller-owned session."""
    frame = spark.range(10).repartition(2)

    def forbidden(*args, **kwargs):
        """Fail on hidden session creation or row materialization."""
        pytest.fail("Unexpected session creation or Spark action")

    monkeypatch.setattr(type(spark).Builder, "getOrCreate", forbidden)
    for name in ("collect", "count", "toPandas", "toArrow"):
        monkeypatch.setattr(type(frame), name, forbidden)
    wrapped = EngineRegistry.wrap(frame)
    assert isinstance(wrapped, SkyulfSparkWrapper)
    selected = wrapped.select(["id"])
    assert selected.columns == ["id"]
    assert selected.schema == frame.schema
    assert wrapped.to_native() is frame


def test_spark_namespace_object_is_not_a_dataframe(spark):
    """Detection must reject namespace lookalikes even when PySpark is installed."""
    lookalike = type("Lookalike", (), {"__module__": "pyspark.sql.dataframe"})()
    with pytest.raises(TypeError, match="pyspark.sql.DataFrame"):
        get_engine(lookalike)
