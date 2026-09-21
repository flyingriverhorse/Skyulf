"""Exercise the single-frame, keyed Spark feature-engineering boundary."""

import pytest

from skyulf.core.capabilities import ExecutionCapability, UnsupportedExecutionError
from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.core.schema import SkyulfSchema
from skyulf.engines import EngineRegistry
from skyulf.preprocessing.dispatcher import apply_dual_engine, fit_dual_engine
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.registry import NodeRegistry


@pytest.fixture
def native_node(monkeypatch):
    """A test-only native node proves routing without advertising built-in support."""
    calls = []

    class Calculator:
        """Record selected features without triggering a data action."""

        def fit(self, frame, config):
            """Keep only portable column metadata in fitted state."""
            calls.append(config)
            return fit_dual_engine(frame, config, {"spark": lambda X, y, p: dict(p)})

    class Applier:
        """Exercise native expressions on a repartitioned frame."""

        def apply(self, frame, params):
            """Double selected features while retaining keys and labels in place."""

            def apply_native(X, y, p):
                """Use literal quoting to avoid dotted-name path interpretation."""
                for name in p["columns"]:
                    column = X["`" + name.replace("`", "``") + "`"]
                    X = X.withColumn(name, column * 2)
                return X.repartition(2), None

            return apply_dual_engine(frame, params, {"spark": apply_native})

    monkeypatch.setattr(NodeRegistry, "_calculators", dict(NodeRegistry._calculators))
    monkeypatch.setattr(NodeRegistry, "_appliers", dict(NodeRegistry._appliers))
    capabilities = tuple(
        ExecutionCapability("spark", op, "native", "preserve", "row") for op in ("fit", "apply")
    )
    NodeRegistry.register("TestSparkDouble", Applier, execution_capabilities=capabilities)(
        Calculator
    )
    return calls


def engineer(steps=None, target="label"):
    """Build the explicitly opted-in Spark entry point used by these contracts."""
    return FeatureEngineer(
        steps or [],
        frame_spec=FrameSpec(("id",), target),
        execution_options=ExecutionOptions("spark"),
    )


def test_key_target_alignment_and_unknown_metrics(spark, native_node):
    """Repartitioning must not detach labels or introduce keys into the features."""
    frame = spark.createDataFrame([(1, 2.0, 0), (2, 3.0, 1)], ["id", "a.b", "label"])
    pipeline = engineer([{"name": "double", "transformer": "TestSparkDouble"}])
    result, metrics = pipeline.fit_transform(frame.repartition(2))
    assert sorted(tuple(row) for row in result.collect()) == [(1, 4.0, 0), (2, 6.0, 1)]
    assert native_node[0]["columns"] == ["a.b"]
    assert metrics["summary"]["rows_in"] is None
    assert metrics["summary"]["peak_memory_bytes"] is None
    assert sorted(tuple(row) for row in pipeline.transform(frame).collect()) == [
        (1, 4.0, 0),
        (2, 6.0, 1),
    ]
    assert pipeline.transform(frame.drop("label")).columns == ["id", "a.b"]


@pytest.mark.parametrize("rows", [[(1, 2.0), (1, 3.0)], [(None, 2.0), (2, 3.0)]])
def test_invalid_keys_rejected(spark, rows):
    """Duplicate or null keys cannot identify predictions reliably."""
    frame = spark.createDataFrame(rows, "id long, value double")
    with pytest.raises(ValueError, match="row_keys"):
        engineer(target=None).fit_transform(frame)


@pytest.mark.parametrize("schema", ["id long, value array<int>", "id long, value decimal(10,2)"])
def test_unsupported_schema_rejected(spark, schema):
    """Unsupported dtypes must fail before any native node executes."""
    with pytest.raises(TypeError, match="Unsupported Spark dtype"):
        engineer(target=None).fit_transform(spark.createDataFrame([], schema))


def test_missing_and_colliding_columns_rejected(spark):
    """Schema ambiguity must not silently change identity or feature selection."""
    with pytest.raises(ValueError, match="Missing"):
        engineer().fit_transform(spark.range(2))
    with pytest.raises(ValueError, match="Duplicate"):
        engineer(target=None).fit_transform(spark.range(2).selectExpr("id", "id"))


def test_full_preflight_before_any_action(spark, native_node, monkeypatch):
    """A later unsupported node must block the entire run before key validation or fit."""
    frame = spark.range(2)

    def forbidden(*args, **kwargs):
        """Detect early work hidden by a later capability error."""
        pytest.fail("Spark action before preflight")

    for name in ("collect", "count", "toPandas"):
        monkeypatch.setattr(type(frame), name, forbidden)
    steps = [
        {"name": "first", "transformer": "TestSparkDouble"},
        {"name": "last", "transformer": "SimpleImputer", "params": {"strategy": "median"}},
    ]
    with pytest.raises(UnsupportedExecutionError, match="SimpleImputer"):
        engineer(steps, target=None).fit_transform(frame)
    assert native_node == []


def test_spark_requires_opt_in_and_single_frame(spark):
    """Separate distributed labels must never be aligned by row position."""
    frame = spark.range(2)
    with pytest.raises(ValueError, match="frame_spec.*execution_options"):
        FeatureEngineer([]).fit_transform(frame)
    with pytest.raises(TypeError, match="single Spark"):
        engineer(target=None).fit_transform((frame, frame))


def test_schema_and_dispatcher_keep_native_data(spark):
    """Schema extraction and dispatch must preserve Spark types and wrapper shape."""
    frame = spark.range(2)
    wrapped = EngineRegistry.wrap(frame)
    assert SkyulfSchema.from_dataframe(wrapped).dtypes == {"id": "int64"}
    result = apply_dual_engine(wrapped, {}, {"spark": lambda X, y, p: (X, None)})
    assert result.to_native() is frame
    with pytest.raises(TypeError, match="single Spark"):
        fit_dual_engine((frame, frame), {}, {"spark": lambda X, y, p: {}})


def test_special_key_names_are_literal(spark):
    """Internal validation names must not collide with user keys or SQL quoting."""
    frame = spark.createDataFrame([(1, 2, 3.0)], ["a.b", "c`d", "__skyulf_count"])
    pipeline = FeatureEngineer(
        [], frame_spec=FrameSpec(("a.b", "c`d")), execution_options=ExecutionOptions("spark")
    )
    output, _ = pipeline.fit_transform(frame)
    assert tuple(output.first()) == (1, 2, 3.0)


@pytest.mark.parametrize("mode", ["drop", "expand", "key", "target"])
def test_false_preservation_declaration_rejected(spark, native_node, monkeypatch, mode):
    """A custom declaration must not silently authorize identity or label corruption."""
    frame = spark.createDataFrame([(1, 2.0, 0), (2, 3.0, 1)], ["id", "value", "label"])
    applier = NodeRegistry.get_applier("TestSparkDouble")

    def corrupt(self, data, params):
        """Deliberately violate the declared row-preserving contract."""
        if mode == "drop":
            return data.limit(1)
        if mode == "expand":
            return data.unionByName(data)
        name = "id" if mode == "key" else "label"
        return data.withColumn(name, data[name] + 1)

    monkeypatch.setattr(applier, "apply", corrupt)
    pipeline = engineer([{"name": "bad", "transformer": "TestSparkDouble"}])
    with pytest.raises(ValueError, match="changed row_keys/target"):
        pipeline.fit_transform(frame)
    assert pipeline.fitted_steps == []


@pytest.mark.parametrize("column", ["id", "label"])
def test_explicit_protected_features_rejected(spark, native_node, column):
    """Explicit feature selection must not bypass key/target exclusion."""
    pipeline = engineer(
        [{"name": "bad", "transformer": "TestSparkDouble", "params": {"columns": [column]}}]
    )
    with pytest.raises(ValueError, match="cannot include row_keys or target"):
        pipeline.fit_transform(spark.range(2))
    assert native_node == []


def test_default_metrics_never_count_or_convert(spark, native_node, monkeypatch):
    """Validation actions are explicit; normal profiling must not collect whole frames."""
    frame = spark.createDataFrame([(1, 2.0, 0)], ["id", "value", "label"])

    def forbidden(*args, **kwargs):
        """Catch the local profiler or any eager whole-frame conversion."""
        pytest.fail("Implicit count or local conversion")

    for name in ("count", "toPandas", "toArrow"):
        monkeypatch.setattr(type(frame), name, forbidden)
    pipeline = engineer([{"name": "double", "transformer": "TestSparkDouble"}])
    output, metrics = pipeline.fit_transform(EngineRegistry.wrap(frame))
    assert output.to_native().columns == frame.columns
    assert metrics["rows_out"] is None


def test_transform_preflight_and_missing_features(spark, native_node, monkeypatch):
    """Apply needs its own permission and all fitted feature columns."""
    frame = spark.createDataFrame([(1, 2.0, 0)], ["id", "value", "label"])
    pipeline = engineer([{"name": "double", "transformer": "TestSparkDouble"}])
    pipeline.fit_transform(frame)
    with pytest.raises(ValueError, match="missing or duplicated"):
        pipeline.transform(frame.drop("value"))
    monkeypatch.setattr(
        NodeRegistry.get_calculator("TestSparkDouble"), "__execution_capabilities__", ()
    )
    with pytest.raises(UnsupportedExecutionError, match="apply"):
        pipeline.transform(frame)


def test_case_insensitive_collision_rejected(spark):
    """Ambiguous SQL identifiers must produce an early schema error."""
    frame = spark.createDataFrame([(1, 2)], ["id", "ID"])
    with pytest.raises(ValueError, match="Duplicate"):
        engineer(target=None).fit_transform(frame)


def test_internal_aggregate_alias_respects_case_insensitive_keys(spark):
    """Validation must not collide with user keys under Spark's SQL resolver."""
    frame = spark.createDataFrame([(1,)], ["__SKYULF_COUNT"])
    pipeline = FeatureEngineer(
        [], frame_spec=FrameSpec(("__SKYULF_COUNT",)), execution_options=ExecutionOptions("spark")
    )
    output, _ = pipeline.fit_transform(frame)
    assert output is frame


@pytest.mark.parametrize(
    "kind,effect,context",
    [
        ("python_batch", "preserve", "row"),
        ("native", "filter", "row"),
        ("native", "preserve", "window"),
    ],
)
def test_only_native_preserving_apply_is_routed(
    spark, native_node, monkeypatch, kind, effect, context
):
    """Capability metadata for other paths must not activate this native entry point."""
    calculator = NodeRegistry.get_calculator("TestSparkDouble")
    monkeypatch.setattr(
        calculator,
        "__execution_capabilities__",
        (
            ExecutionCapability("spark", "fit", "native", "preserve", "global"),
            ExecutionCapability("spark", "apply", kind, effect, context),
        ),
    )
    pipeline = engineer([{"name": "blocked", "transformer": "TestSparkDouble"}], target=None)
    with pytest.raises(UnsupportedExecutionError, match="native row-preserving"):
        pipeline.fit_transform(spark.range(2))
    assert native_node == []
