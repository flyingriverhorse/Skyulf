"""Native Spark preprocessing must preserve the frozen local prediction function."""

import importlib
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import skyulf.inference as inference
from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.core.schema import SchemaMismatchError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.bundle import build_bundle, load_bundle, predict_local, save_bundle
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest.fixture
def regression_bundle(fitted_regression_pipeline):
    """Reuse real pandas/Polars training without fitting anything on Spark input."""
    return build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))


def _predict(frame, bundle, **kwargs):
    """Exercise the public API with the smallest supported explicit contract."""
    return inference.predict_spark(
        frame,
        bundle,
        frame_spec=kwargs.pop("frame_spec", FrameSpec(row_keys=("id",))),
        options=kwargs.pop("options", ExecutionOptions("spark", python_batch_rows=2)),
        mode=kwargs.pop("mode", "native_features"),
        **kwargs,
    )


@pytest.mark.parametrize("partitions,arrow_rows", [(1, 1), (3, 5)])
def test_native_path_matches_persisted_local_bundle(
    spark, regression_bundle, tmp_path, monkeypatch, partitions, arrow_rows
):
    """Repartitioning, missing values and Arrow boundaries must not change keyed predictions."""
    save_bundle(regression_bundle, tmp_path / "bundle")
    bundle = load_bundle(tmp_path / "bundle")
    raw = pd.DataFrame({"x": [150.0, 80.0, np.nan, 110.0], "z": [9.0, 3.0, 4.0, np.nan]})
    expected = predict_local(raw, bundle)
    frame = spark.createDataFrame(
        [
            (10, 150.0, 9.0, "wide"),
            (11, 80.0, 3.0, "wide"),
            (12, None, 4.0, "wide"),
            (2**53 + 1, 110.0, float("nan"), "wide"),
        ],
        "id long, x double, z double, unused string",
    ).repartition(partitions)
    old_limit = spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", str(arrow_rows))

    def forbidden(*args, **kwargs):
        """Prediction must never train or materialize a distributed input locally."""
        pytest.fail("Inference attempted fitting or driver materialization")

    collect = type(frame).collect
    validation_plans = []

    def bounded_collect(self):
        """Only existing one-row validation results may reach the driver in inference."""
        plan = self._jdf.queryExecution().logical().toString()
        assert plan.startswith("GlobalLimit 1"), plan
        validation_plans.append(plan)
        return collect(self)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(FeatureEngineer, "fit_transform", forbidden)
            patch.setattr(type(frame), "toPandas", forbidden)
            patch.setattr(type(frame), "toLocalIterator", forbidden)
            patch.setattr(type(frame), "collect", bounded_collect)
            output = _predict(frame, bundle)
        rows = output.orderBy("id").collect()
        plan = output._jdf.queryExecution().executedPlan().toString()
        assert "MapInPandas" in plan
        assert not any(name in plan for name in ("ArrowEvalPython", "BatchEvalPython", "PythonUDF"))
        assert output.columns == ["id", "prediction"]
        assert output.schema["id"] == frame.schema["id"]
        assert spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch") == str(arrow_rows)
        assert validation_plans
        assert [row.id for row in rows] == [10, 11, 12, 2**53 + 1]
        np.testing.assert_allclose(
            [row.prediction for row in rows], expected.prediction, rtol=1e-10
        )
    finally:
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", old_limit)


@pytest.mark.parametrize("empty", [False, True])
def test_empty_partitions_and_composite_key_schema(spark, regression_bundle, empty):
    """Empty partitions and inputs must preserve explicit key types and prediction schema."""
    frame = spark.createDataFrame(
        [] if empty else [(1, 2, 3, 4, "a", True, 150.0, 9.0)],
        "tiny byte, small short, id int, large long, text string, flag boolean, x double, z double",
    ).repartition(3)
    keys = ("tiny", "small", "id", "large", "text", "flag")
    output = _predict(frame, regression_bundle, frame_spec=FrameSpec(row_keys=keys))
    rows = output.collect()
    assert output.schema.fields[:-1] == [frame.schema[key] for key in keys]
    assert output.schema["prediction"].dataType.typeName() == "double"
    assert [tuple(row)[:-1] for row in rows] == ([] if empty else [(1, 2, 3, 4, "a", True)])
    np.testing.assert_allclose([row.prediction for row in rows], [] if empty else [280.0])


@pytest.mark.parametrize(
    "change,match",
    [
        ("order", "order"),
        ("missing", "missing"),
        ("dtype", "dtype"),
        ("duplicate", "[Dd]uplicate"),
        ("key_feature", "colli|row_keys"),
        ("key_output", "colli|prediction"),
        ("output_input", "colli|prediction"),
        ("date_key", "key.*dtype|key.*type"),
        ("target", "target"),
        ("stage", "raw|stage"),
        ("engine", "engine|spark"),
        ("checksum", "checksum"),
        ("runtime", "runtime"),
        ("streaming", "[Ss]treaming"),
    ],
)
def test_preflight_rejects_invalid_contract_before_actions(
    spark, regression_bundle, fitted_regression_pipeline, monkeypatch, change, match
):
    """Schema, identity and bundle errors must fail before a distributed validation job."""
    frame = spark.createDataFrame([(1, 150.0, 9.0)], "id long, x double, z double")
    kwargs = {}
    bundle = regression_bundle
    if change == "order":
        frame = frame.select("id", "z", "x")
    elif change == "missing":
        frame = frame.drop("z")
    elif change == "dtype":
        frame = frame.selectExpr("id", "cast(x as string) as x", "z")
    elif change == "duplicate":
        frame = frame.selectExpr("id", "x", "z", "x as X")
    elif change == "key_feature":
        kwargs["frame_spec"] = FrameSpec(row_keys=("x",))
    elif change == "key_output":
        frame = frame.withColumnRenamed("id", "PREDICTION")
        kwargs["frame_spec"] = FrameSpec(row_keys=("PREDICTION",))
    elif change == "output_input":
        frame = frame.selectExpr("*", "1 as prediction")
    elif change == "date_key":
        frame = frame.selectExpr("date'2026-01-01' as id", "x", "z")
    elif change == "target":
        kwargs["frame_spec"] = FrameSpec(row_keys=("id",), target="z")
    elif change == "stage":
        bundle = build_bundle(
            fitted_regression_pipeline, input_stage="features", feature_order=("x", "z")
        )
    elif change == "mode":
        kwargs["mode"] = "python_pipeline"
    elif change == "engine":
        kwargs["options"] = ExecutionOptions("pandas")
    elif change == "checksum":
        bundle = replace(bundle, model_payload=b"broken")
    elif change == "streaming":
        frame = (
            spark.readStream.format("rate")
            .load()
            .selectExpr("value as id", "cast(value as double) as x", "cast(value as double) as z")
        )
    elif change == "runtime":
        from skyulf.inference._manifest import semantic_digest

        manifest = bundle.manifest.model_copy(
            update={
                "requirements": tuple(
                    (name, "0.0.0" if name == "scikit-learn" else version)
                    for name, version in bundle.manifest.requirements
                )
            }
        )
        manifest = manifest.model_copy(update={"semantic_digest": semantic_digest(manifest)})
        bundle = replace(bundle, manifest=manifest)

    def forbidden(*args, **kwargs):
        """Even the bounded key validation action is forbidden for invalid metadata."""
        pytest.fail("Invalid contract reached a Spark action")

    for action in ("collect", "count", "first", "take", "toPandas", "toLocalIterator"):
        monkeypatch.setattr(type(frame), action, forbidden)
    with pytest.raises(
        (ValueError, TypeError, SchemaMismatchError, UnsupportedExecutionError), match=match
    ):
        _predict(frame, bundle, **kwargs)


@pytest.mark.parametrize("ids", [[1, 1], [1, None]])
def test_duplicate_or_null_keys_are_rejected(spark, regression_bundle, ids):
    """Rows cannot be associated safely with predictions when business keys are ambiguous."""
    frame = spark.createDataFrame([(key, 150.0, 9.0) for key in ids], "id long, x double, z double")
    with pytest.raises(ValueError, match="unique and non-null"):
        _predict(frame, regression_bundle)


def test_native_feature_dtype_mismatch_is_rejected_before_actions(spark, monkeypatch):
    """Native dtype promotion cannot silently change the local model's recorded schema."""
    train = pd.DataFrame({"x": [1, 3, 5, 7], "y": [2.0, 6.0, 10.0, 14.0]})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}}
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="y")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
    frame = spark.createDataFrame([(1, 2)], "id long, x long")

    def forbidden(*args, **kwargs):
        """A dtype incompatibility is known from Spark expressions without executing them."""
        pytest.fail("Feature dtype mismatch reached a Spark action")

    monkeypatch.setattr(type(frame), "collect", forbidden)
    with pytest.raises(SchemaMismatchError, match="dtype"):
        _predict(frame, bundle)


def test_classification_is_supported_with_manifest_output(spark):
    """Distributed classification must return the declared label and probabilities."""
    train = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0], "y": ["no", "no", "yes", "yes"]})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="y")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
    frame = spark.createDataFrame([(1, 2.0)], "id long, x double")

    rows = _predict(frame, bundle).collect()
    assert len(rows) == 1
    assert rows[0].id == 1
    assert rows[0].prediction == "yes"
    assert rows[0].probability_0 < rows[0].probability_1


@pytest.mark.parametrize("kind", ["integer", "boolean"])
@pytest.mark.parametrize("mode", ["native_features", "python_pipeline"])
def test_nullable_integral_features_are_rejected_before_actions(spark, monkeypatch, kind, mode):
    """Arrow must not silently widen nullable integral or boolean model features."""
    values = [1, 3, 5, 7] if kind == "integer" else [False, True, False, True]
    train = pd.DataFrame({"x": values, "y": [2.0, 6.0, 10.0, 14.0]})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 3, "random_state": 17},
            },
        }
    )
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="y")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
    if kind == "integer":
        local = pd.DataFrame({"x": pd.Series([5, None, 7], dtype="Int64")})
        assert len(predict_local(local, bundle)) == 3
    dtype = "long" if kind == "integer" else "boolean"
    frame = spark.createDataFrame([(1, values[0]), (2, None)], f"id long, x {dtype}")

    def forbidden(*args, **kwargs):
        """Unsupported transport schemas are known before key validation needs any rows."""
        pytest.fail("Nullable model feature reached a Spark action")

    monkeypatch.setattr(type(frame), "collect", forbidden)
    with pytest.raises(UnsupportedExecutionError, match="[Nn]ullable.*Arrow"):
        _predict(frame, bundle, mode=mode)


def test_python_pipeline_rejects_nullable_integer_before_arrow_with_frozen_fe(spark, monkeypatch):
    """Python FE workers must reject nullable integer raw inputs before precision loss."""
    types = importlib.import_module("pyspark.sql.types")
    train = pd.DataFrame({"x": [1, 3, 5, 7], "y": [2.0, 6.0, 10.0, 14.0]})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "fill",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["x"]},
                },
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": ["x"]},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="y")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
    frame = spark.createDataFrame(
        [(1, 2**53 + 1), (2, None)],
        types.StructType(
            [
                types.StructField("id", types.LongType(), False),
                types.StructField("x", types.LongType(), True),
            ]
        ),
    )

    def forbidden(*args, **kwargs):
        """Transport safety must fail before distributed key validation."""
        pytest.fail("Nullable raw integer reached a Spark action")

    monkeypatch.setattr(type(frame), "collect", forbidden)
    with pytest.raises(UnsupportedExecutionError, match="[Nn]ullable.*Arrow"):
        _predict(frame, bundle, mode="python_pipeline")


def test_nonnullable_integer_features_preserve_supported_inference(spark):
    """The conservative nullable transport guard must still accept declared nonnullable integers."""
    types = importlib.import_module("pyspark.sql.types")
    train = pd.DataFrame({"x": [1, 3, 5, 7], "y": [2.0, 6.0, 10.0, 14.0]})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="y")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
    large = 2**53 + 1
    frame = spark.createDataFrame(
        [(large, large)],
        types.StructType(
            [
                types.StructField("id", types.LongType(), False),
                types.StructField("x", types.LongType(), False),
            ]
        ),
    )
    rows = _predict(frame, bundle).collect()
    expected = predict_local(pd.DataFrame({"x": [large]}), bundle)
    assert [row.id for row in rows] == [large]
    np.testing.assert_allclose([row.prediction for row in rows], expected.prediction)


def test_worker_loads_once_and_bounds_estimator_batches(regression_bundle, monkeypatch):
    """Iterator reuse must avoid repeated deserialization and bound each estimator call."""
    module = importlib.import_module("skyulf.inference.spark")
    load_model = module.load_model
    loaded = []
    sizes = []

    def recorded_load(payload, manifest):
        """Observe the real loader and estimator without replacing their behavior."""
        model = load_model(payload, manifest)
        predict = model.predict

        def recorded_predict(values):
            """The estimator must receive only the two trained feature columns."""
            sizes.append(values.shape)
            return predict(values)

        model.predict = recorded_predict
        loaded.append(model)
        return model

    monkeypatch.setattr(module, "load_model", recorded_load)
    worker = module._prediction_iterator(
        regression_bundle.model_payload, regression_bundle.manifest, ("id",), 2
    )
    batch = pd.DataFrame({"id": range(5), "x": np.zeros(5), "z": np.zeros(5)})
    result = pd.concat(worker(iter([batch, batch.head(0), batch.iloc[:3]])), ignore_index=True)
    assert len(loaded) == 1
    assert sizes == [(2, 2), (2, 2), (1, 2), (2, 2), (1, 2)]
    assert result.id.tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
    np.testing.assert_allclose(result.prediction, loaded[0].intercept_)


def test_worker_checks_runtime_before_loading_model(regression_bundle, monkeypatch):
    """A worker with incompatible packages must reject even driver-validated model bytes."""
    module = importlib.import_module("skyulf.inference.spark")
    metadata = importlib.import_module("skyulf.inference._manifest")
    requirements = regression_bundle.manifest.requirements
    monkeypatch.setattr(
        metadata,
        "runtime_requirements",
        lambda: tuple(
            (name, "0.0.0" if name == "scikit-learn" else version) for name, version in requirements
        ),
    )

    def forbidden(*args, **kwargs):
        """Pickle must not run on an incompatible worker."""
        pytest.fail("Runtime mismatch reached model loading")

    monkeypatch.setattr(module, "load_model", forbidden)
    worker = module._prediction_iterator(
        regression_bundle.model_payload, regression_bundle.manifest, ("id",), 2
    )
    with pytest.raises(ValueError, match="runtime mismatch"):
        list(worker(iter([])))


def test_only_keys_and_model_features_reach_worker(spark, regression_bundle, monkeypatch):
    """Unused nested or wide fields must be projected before Arrow serialization."""
    frame = spark.createDataFrame([(1, 150.0, 9.0)], "id long, x double, z double")
    frame = frame.selectExpr("id", "x", "array('large', 'unused') as baggage", "z")
    observed = []
    original = type(frame).mapInPandas

    def recorded_map(self, func, schema, **kwargs):
        """Observe the real distributed boundary and retain its actual execution."""
        observed.append(self.columns)
        return original(self, func, schema, **kwargs)

    monkeypatch.setattr(type(frame), "mapInPandas", recorded_map)
    rows = _predict(frame, regression_bundle).collect()
    assert observed == [["id", "x", "z"]]
    np.testing.assert_allclose([row.prediction for row in rows], [280.0])


def test_inference_import_does_not_require_spark():
    """Publishing the Spark entry point must not make optional runtimes mandatory."""
    code = """
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyspark', 'mlflow'}:
            raise ImportError('optional runtime deliberately unavailable')
sys.meta_path.insert(0, BlockExtras())
from skyulf.inference import predict_spark, predict_local
assert callable(predict_spark) and callable(predict_local)
assert not any(name.split('.')[0] in {'pyspark', 'mlflow'} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
