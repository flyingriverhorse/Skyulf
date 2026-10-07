"""Fail-closed Spark pyfunc admission and exact named transport contracts."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from skyulf.data.dataset import SplitDataset  # noqa: E402
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline  # noqa: E402
from skyulf.integrations.mlflow.spark import spark_model  # noqa: E402
from skyulf.pipeline import SkyulfPipeline  # noqa: E402


@pytest.fixture
def artifact(tmp_path):
    """A fitted numeric pipeline isolates adapter admission from preprocessing."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    pipeline = SkyulfPipeline({"modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    save_local_pipeline(pipeline, tmp_path / "model")
    return load_local_pipeline(tmp_path / "model")


@pytest.mark.parametrize(
    "uri", ["models:/model@champion", "models:/model/latest", "runs:/run/model"]
)
def test_mutable_or_unregistered_uri_rejected_before_udf(artifact, monkeypatch, uri):
    """Lazy workers cannot follow a mutable alias after the driver pins a model."""
    udf = Mock()
    monkeypatch.setattr(mlflow.pyfunc, "spark_udf", udf)
    with pytest.raises(ValueError, match="concrete"):
        spark_model.predict_spark_pyfunc(
            None,
            None,
            model_uri=uri,
            artifact=artifact,
            record_key_columns=("id",),
            env_manager="local",
        )
    udf.assert_not_called()


def test_worker_revalidates_certificate_and_runtime_source(artifact, tmp_path):
    """A certificate is detached evidence, not permission to accept another payload."""
    from skyulf.integrations.mlflow.models.local_model import SkyulfLocalPythonModel

    certificate = spark_model.partition_safety_certificate(artifact)
    model = SkyulfLocalPythonModel(None, certificate, spark_model.runtime_source_digest())
    model.load_context(SimpleNamespace(artifacts={"local_pipeline": str(tmp_path / "model")}))
    assert model.predict(None, pd.DataFrame({"x": [5.0]}))["prediction"].iloc[0] == pytest.approx(
        10
    )
    certificate["pipeline_sha256"] = "0" * 64
    changed = SkyulfLocalPythonModel(None, certificate, spark_model.runtime_source_digest())
    with pytest.raises(ValueError, match="certificate"):
        changed.load_context(SimpleNamespace(artifacts={"local_pipeline": str(tmp_path / "model")}))
    with pytest.raises(ValueError, match="runtime source"):
        spark_model.validate_worker_certificate(
            artifact, spark_model.partition_safety_certificate(artifact), "0" * 64
        )


def test_legacy_package_rejected_before_udf(artifact, monkeypatch):
    """A legacy whole-frame package cannot become distributed solely via driver inspection."""
    udf = Mock()
    monkeypatch.setattr(mlflow.pyfunc, "spark_udf", udf)
    monkeypatch.setattr(
        spark_model, "_download_package", lambda *args: ("package", SimpleNamespace(metadata={}))
    )
    with pytest.raises(ValueError, match="certificate"):
        spark_model.predict_spark_pyfunc(
            None,
            SimpleNamespace(columns=["id", "x"]),
            model_uri="models:/model/1",
            artifact=artifact,
            record_key_columns=("id",),
            env_manager="local",
        )
    udf.assert_not_called()


def test_concrete_package_download_uses_explicit_stores_without_global_mutation(monkeypatch):
    """The UDF must use the same downloaded package that admission inspected."""
    client = object()
    factory = Mock(return_value=client)
    download = Mock(return_value="downloaded/concrete-package")
    model = object()
    monkeypatch.setattr(spark_model, "make_registry_client", factory)
    monkeypatch.setattr(spark_model, "download_registered_package", download)
    monkeypatch.setattr(mlflow.models.Model, "load", Mock(return_value=model))
    before = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    path, loaded = spark_model._download_package(
        "models:/catalog.schema.model/19", "tracking-explicit", "registry-explicit"
    )
    factory.assert_called_once_with(mlflow, "tracking-explicit", "registry-explicit")
    download.assert_called_once_with(
        mlflow, client, "catalog.schema.model", "19", "tracking-explicit"
    )
    assert path == "downloaded/concrete-package" and loaded is model
    assert (mlflow.get_tracking_uri(), mlflow.get_registry_uri()) == before


def test_additive_certificate_cannot_override_original_package_identity(artifact):
    """The certificate supplements the original digest and cannot hide mismatched metadata."""
    from skyulf.integrations.mlflow.models.local_model import _signature

    metadata = {
        spark_model.SAFETY_KEY: spark_model.partition_safety_certificate(artifact),
        spark_model.SOURCE_KEY: spark_model.runtime_source_digest(),
        "skyulf_artifact_kind": "local_pipeline",
        "skyulf_execution_scope": "whole_frame_local",
        "local_pipeline_digest": "0" * 64,
    }
    inputs, outputs = spark_model._contract(artifact)
    with pytest.raises(ValueError, match="identity"):
        spark_model._validate_package(
            SimpleNamespace(metadata=metadata, signature=_signature(artifact)),
            spark_model.partition_safety_certificate(artifact),
            inputs,
            outputs,
        )


def test_local_logging_adds_certificate_without_changing_scope(artifact, tmp_path, monkeypatch):
    """New inspected packages retain legacy execution scope and bundle worker source."""
    from skyulf.integrations.mlflow.models import local_model

    calls = []
    monkeypatch.setattr(local_model, "make_tracking_client", lambda uri: Mock())
    monkeypatch.setattr(local_model, "scrub_local_artifact_uri", lambda *args: None)
    monkeypatch.setattr(mlflow.pyfunc, "save_model", lambda **kwargs: calls.append(kwargs))
    local_model.log_local_model(tmp_path / "model", run_id="run", artifact_path="model")
    saved = calls[0]
    assert saved["metadata"]["skyulf_execution_scope"] == "whole_frame_local"
    assert saved["metadata"][spark_model.SAFETY_KEY] == spark_model.partition_safety_certificate(
        artifact
    )
    assert saved["metadata"][spark_model.SOURCE_KEY] == spark_model.runtime_source_digest()
    assert saved["code_paths"][0].endswith("skyulf")


def test_independent_model_set_certificate_checks_every_component(artifact, tmp_path):
    """Independent model branches require their own inspected payload evidence."""
    from skyulf.inference._manifest import ColumnSpec
    from skyulf.inference._model_set_manifest import ComponentReference
    from skyulf.inference.model_set import save_model_set

    reference = ComponentReference(
        name="model", version="1", digest=artifact.manifest.pipeline_sha256
    )
    model_set = save_model_set(
        tmp_path / "set",
        {
            "first": (reference, tmp_path / "model"),
            "second": (reference.model_copy(update={"name": "second"}), tmp_path / "model"),
        },
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    certificate = spark_model.partition_safety_certificate(model_set)
    assert certificate["composition_contract"] == "independent_components_v1"
    assert set(certificate["components"]) == {"first", "second"}
    assert certificate["components"]["first"] == spark_model.partition_safety_certificate(artifact)
    assert [name for name, _ in certificate["output_schema"]] == [
        "id",
        "first__prediction",
        "first__scoring_status",
        "first__exclusion_reason",
        "second__prediction",
        "second__scoring_status",
        "second__exclusion_reason",
    ]


def test_custom_composition_rejected_even_when_callable_is_row_local(artifact, tmp_path):
    """An arbitrary callback is not admitted from its name or a claimed safe flag."""
    from skyulf.inference._manifest import ColumnSpec
    from skyulf.inference._model_set_manifest import ComponentReference
    from skyulf.inference.model_set import save_model_set

    reference = ComponentReference(
        name="model", version="1", digest=artifact.manifest.pipeline_sha256
    )
    model_set = save_model_set(
        tmp_path / "set",
        {"first": (reference, tmp_path / "model")},
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        composition_source="def combine(inputs, predictions, params):\n    return predictions[['first__prediction']].rename(columns={'first__prediction': 'business'})\n",
        composition_config={
            "outputs": [
                {
                    "name": "business",
                    "version": "1",
                    "function": "combine",
                    "params": {},
                    "required_components": ["first"],
                    "columns": [{"name": "business", "dtype": "float64"}],
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="independent_components_v1"):
        spark_model.partition_safety_certificate(model_set)
    assert spark_model.optional_partition_certificate(model_set) is None


@pytest.fixture
def expressions(monkeypatch):
    """Capture native Spark expressions without starting an unavailable local JVM."""
    import sys
    from types import ModuleType

    class Expression:
        """Keep column identity, casts and aliases observable at the UDF boundary."""

        def __init__(self, name, cast=None, alias=None):
            self.name, self.cast_type, self.output_name = name, cast, alias

        def cast(self, dtype):
            """Return a typed expression retaining the literal source field."""
            return Expression(self.name, dtype, self.output_name)

        def alias(self, name):
            """Retain original names inside the struct passed to MLflow."""
            return Expression(self.name, self.cast_type, name)

    sql = cast(Any, ModuleType("pyspark.sql"))
    sql.functions = SimpleNamespace(col=Expression, struct=lambda *columns: columns)
    monkeypatch.setitem(sys.modules, "pyspark", ModuleType("pyspark"))
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    return sql.functions


def test_nullable_named_struct_casts_exact_int64_before_arrow(expressions):
    """Large nullable integers must never pass through a floating Arrow representation."""
    frame = SimpleNamespace(
        schema={
            "large.id": SimpleNamespace(dataType=SimpleNamespace(simpleString=lambda: "bigint")),
            "amount": SimpleNamespace(dataType=SimpleNamespace(simpleString=lambda: "double")),
        }
    )
    columns = spark_model._named_inputs(frame, [("large.id", "Int64"), ("amount", "float64")])
    assert [(column.name, column.cast_type, column.output_name) for column in columns] == [
        ("`large.id`", "string", "large.id"),
        ("`amount`", None, "amount"),
    ]
    frame.schema["large.id"].dataType.simpleString = lambda: "double"
    with pytest.raises(ValueError, match="exact native Int64"):
        spark_model._named_inputs(frame, [("large.id", "Int64")])


def test_worker_wheel_carries_exact_source_and_dependency_metadata(tmp_path):
    """Isolated virtualenv workers can install unreleased Skyulf code without PyPI lookup."""
    import hashlib
    import zipfile
    from importlib.metadata import distribution

    from skyulf.integrations.mlflow.spark._spark_environment import snapshot_worker_environment

    package = distribution("skyulf-core")
    pins = [f"skyulf-core=={package.version}", "numpy==2.2.6", "pandas==2.3.3"]
    paths, requirements, digest = snapshot_worker_environment(tmp_path, pins)
    assert requirements[1:] == pins[1:]
    assert requirements[0].startswith("code/skyulf_core-")
    assert requirements[0].endswith(".whl")
    assert digest == spark_model.runtime_source_digest()
    with zipfile.ZipFile(paths[1]) as wheel:
        metadata = next(name for name in wheel.namelist() if name.endswith("/METADATA"))
        assert wheel.read(metadata).decode() == (
            package.read_text("METADATA") or package.read_text("PKG-INFO")
        )
        source = wheel.read("skyulf/integrations/mlflow/spark/spark_model.py")
        assert (
            hashlib.sha256(source).hexdigest()
            == hashlib.sha256(
                __import__("pathlib").Path(spark_model.__file__).read_bytes()
            ).hexdigest()
        )
        assert all("__pycache__" not in name for name in wheel.namelist())


def test_missing_inputs_rejected_before_udf(artifact, monkeypatch):
    """Missing feature columns fail before loading Spark workers or a model package."""
    udf = Mock()
    monkeypatch.setattr(mlflow.pyfunc, "spark_udf", udf)
    with pytest.raises(ValueError, match="missing"):
        spark_model.predict_spark_pyfunc(
            None,
            SimpleNamespace(columns=["id"]),
            model_uri="models:/model/1",
            artifact=artifact,
            record_key_columns=("id",),
            env_manager="local",
        )
    udf.assert_not_called()


def test_invalid_environment_rejected_before_udf(artifact, monkeypatch):
    """Worker environment selection is explicit and cannot silently use an unknown mode."""
    udf = Mock()
    monkeypatch.setattr(mlflow.pyfunc, "spark_udf", udf)
    with pytest.raises(ValueError, match="env_manager"):
        spark_model.predict_spark_pyfunc(
            None,
            None,
            model_uri="models:/model/1",
            artifact=artifact,
            record_key_columns=("id",),
            env_manager="automatic",
        )
    udf.assert_not_called()


def test_single_pipeline_keeps_record_keys_out_of_model_input(artifact, monkeypatch):
    """A single-model route cannot accidentally score a raw record identifier as a feature."""
    udf = Mock()
    monkeypatch.setattr(mlflow.pyfunc, "spark_udf", udf)
    with pytest.raises(ValueError, match="record keys.*model input"):
        spark_model.predict_spark_pyfunc(
            None,
            SimpleNamespace(columns=["x"]),
            model_uri="models:/model/1",
            artifact=artifact,
            record_key_columns=("x",),
            env_manager="local",
        )
    udf.assert_not_called()


@pytest.fixture
def mlflow_struct_converter(monkeypatch):
    """Use MLflow's real converter with only its optional Spark type classes replaced."""
    import sys
    from types import ModuleType

    types = cast(Any, ModuleType("pyspark.sql.types"))
    for name in (
        "IntegerType",
        "LongType",
        "FloatType",
        "DoubleType",
        "BooleanType",
        "StringType",
        "ArrayType",
        "MapType",
    ):
        setattr(types, name, type(name, (), {}))

    class StructType:
        """Expose the field lookup protocol consumed by MLflow without a Spark JVM."""

        def __init__(self, fields):
            self.fields = fields

        def fieldNames(self):
            """Return ordered fields as the real Spark schema does."""
            return [field.name for field in self.fields]

        def __getitem__(self, name):
            """Resolve a named field's scalar Spark type."""
            return next(field for field in self.fields if field.name == name)

    types.StructType = StructType
    sql = cast(Any, ModuleType("pyspark.sql"))
    sql.types = types
    monkeypatch.setitem(sys.modules, "pyspark", ModuleType("pyspark"))
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.types", types)

    def convert(result):
        """Run installed MLflow conversion over every declared string field."""
        schema = StructType(
            [SimpleNamespace(name=name, dataType=types.StringType()) for name in result.columns]
        )
        return mlflow.pyfunc._convert_struct_values(result, schema)

    return convert


def test_certified_set_spark_output_preserves_nulls_and_local_dtypes(
    artifact,
    tmp_path,
    mlflow_struct_converter,
):
    """MLflow must retain SQL null outcomes while ordinary local pyfunc keeps StringDtype."""
    from skyulf.inference._manifest import ColumnSpec
    from skyulf.inference._model_set_manifest import ComponentReference
    from skyulf.inference.model_set import save_model_set
    from skyulf.integrations.mlflow.models.model_set import SkyulfModelSetPythonModel, _signature

    reference = ComponentReference(
        name="model", version="1", digest=artifact.manifest.pipeline_sha256
    )
    model_set = save_model_set(
        tmp_path / "set",
        {"first": (reference, tmp_path / "model")},
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    model = SkyulfModelSetPythonModel(
        None,
        spark_model.partition_safety_certificate(model_set),
        spark_model.runtime_source_digest(),
    )
    model.load_context(SimpleNamespace(artifacts={"model_set": str(tmp_path / "set")}))
    query = pd.DataFrame({"id": [2**53 + 1, 2**53 + 2], "x": [5.0, 6.0]})
    local = model.predict(None, query)
    distributed = model.predict(None, query, params={"skyulf_spark_output": True})
    assert str(local["first__exclusion_reason"].dtype) == "string"
    assert str(distributed["first__exclusion_reason"].dtype) == "object"
    converted = mlflow_struct_converter(
        distributed[["first__scoring_status", "first__exclusion_reason"]]
    )
    assert converted["first__exclusion_reason"].isna().all()
    assert converted["first__scoring_status"].eq("predicted").all()
    assert distributed.id.tolist() == query.id.tolist()
    package = tmp_path / "pyfunc"
    mlflow.pyfunc.save_model(
        path=str(package),
        python_model=model,
        artifacts={"model_set": str(tmp_path / "set")},
        signature=_signature(model_set, spark_certified=True),
        pip_requirements=[f"mlflow=={mlflow.__version__}"],
    )
    loaded = mlflow.pyfunc.load_model(str(package))
    pd.testing.assert_frame_equal(loaded.predict(query), local)
    loaded_output = loaded.predict(
        query, params={"skyulf_spark_output": True, "skyulf_spark_batch_rows": 1}
    )
    pd.testing.assert_frame_equal(loaded_output, distributed)


def test_uncertified_pyfunc_cannot_enable_spark_output(artifact, tmp_path, monkeypatch):
    """The transport flag cannot authorize an uncertified legacy whole-frame artifact."""
    from skyulf.integrations.mlflow.models import local_model

    model = local_model.SkyulfLocalPythonModel()
    model.load_context(SimpleNamespace(artifacts={"local_pipeline": str(tmp_path / "model")}))
    score = Mock()
    monkeypatch.setattr(local_model, "score_local_pipeline", score)
    with pytest.raises(ValueError, match="certif"):
        model.predict(None, pd.DataFrame({"x": [5.0]}), params={"skyulf_spark_output": True})
    score.assert_not_called()


@pytest.mark.parametrize("certified", [False, True])
def test_spark_package_requires_exact_output_transport_param_schema(artifact, certified):
    """A worker package that cannot preserve string nulls must fail before UDF creation."""
    from skyulf.integrations.mlflow.models.local_model import _signature

    certificate = spark_model.partition_safety_certificate(artifact)
    metadata = {
        spark_model.SAFETY_KEY: certificate,
        spark_model.SOURCE_KEY: spark_model.runtime_source_digest(),
        "skyulf_artifact_kind": "local_pipeline",
        "skyulf_execution_scope": "whole_frame_local",
        "local_pipeline_digest": artifact.manifest.pipeline_sha256,
    }
    package = SimpleNamespace(
        metadata=metadata, signature=_signature(artifact, spark_certified=certified)
    )
    inputs, outputs = spark_model._contract(artifact)
    if certified:
        spark_model._validate_package(package, certificate, inputs, outputs)
        assert package.signature.params.to_dict() == [
            {"name": "skyulf_spark_output", "type": "boolean", "default": False, "shape": None},
            {"name": "skyulf_spark_batch_rows", "type": "long", "default": 10000, "shape": None},
        ]
    else:
        with pytest.raises(ValueError, match="output transport contract"):
            spark_model._validate_package(package, certificate, inputs, outputs)


def test_spark_output_rejects_unreviewed_mixed_null_string_semantics():
    """Future exclusions cannot silently stringify missing values inside mixed batches."""
    from skyulf.inference._manifest import ColumnSpec
    from skyulf.integrations.mlflow.spark._spark_output import prepare_spark_output

    frame = pd.DataFrame({"reason": pd.Series([pd.NA, "excluded"], dtype="string")})
    with pytest.raises(ValueError, match="mixed-null"):
        prepare_spark_output(frame, (ColumnSpec(name="reason", dtype="string"),), True)
    assert str(frame.reason.dtype) == "string" and pd.isna(frame.reason.iloc[0])


def test_spark_worker_bounds_model_calls_and_preserves_duplicate_indices(
    artifact, tmp_path, monkeypatch
):
    """Serverless model-call bounds cannot rely on an unavailable Spark Arrow setting."""
    from skyulf.integrations.mlflow.models import local_model

    model = local_model.SkyulfLocalPythonModel(
        None,
        spark_model.partition_safety_certificate(artifact),
        spark_model.runtime_source_digest(),
    )
    model.load_context(SimpleNamespace(artifacts={"local_pipeline": str(tmp_path / "model")}))
    query = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}, index=[19, 4, 19, -1, 50, 4, 9]
    )
    original = local_model.score_local_pipeline
    calls = []

    def observe(frame, artifact):
        """Record actual scorer populations rather than inferring a bound from configuration."""
        calls.append(len(frame))
        return original(frame, artifact)

    monkeypatch.setattr(local_model, "score_local_pipeline", observe)
    expected = model.predict(None, query, params={"skyulf_spark_batch_rows": 3})
    assert calls == [7]
    calls.clear()
    result = model.predict(
        None, query, params={"skyulf_spark_output": True, "skyulf_spark_batch_rows": 3}
    )
    assert calls == [3, 3, 1]
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("rows", [False, True, 0, -1, 1.5, "3"])
def test_spark_batch_bound_rejects_invalid_values_before_udf(artifact, monkeypatch, rows):
    """Invalid limits cannot fall through to unbounded or accidental one-row model calls."""
    udf = Mock()
    monkeypatch.setattr(mlflow.pyfunc, "spark_udf", udf)
    with pytest.raises(ValueError, match="prediction_batch_rows"):
        spark_model.predict_spark_pyfunc(
            None,
            None,
            model_uri="models:/model/1",
            artifact=artifact,
            record_key_columns=("id",),
            env_manager="virtualenv",
            prediction_batch_rows=rows,
        )
    udf.assert_not_called()


def test_set_spark_worker_bounds_calls_preserving_keys_nulls_and_indices(
    artifact, tmp_path, monkeypatch
):
    """Independent components must remain keyed correctly across worker model-call slices."""
    from skyulf.inference._manifest import ColumnSpec
    from skyulf.inference._model_set_manifest import ComponentReference
    from skyulf.inference.model_set import save_model_set
    from skyulf.integrations.mlflow.models import model_set as adapter

    reference = ComponentReference(
        name="model", version="1", digest=artifact.manifest.pipeline_sha256
    )
    artifact_set = save_model_set(
        tmp_path / "set",
        {"first": (reference, tmp_path / "model")},
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    model = adapter.SkyulfModelSetPythonModel(
        None,
        spark_model.partition_safety_certificate(artifact_set),
        spark_model.runtime_source_digest(),
    )
    model.load_context(SimpleNamespace(artifacts={"model_set": str(tmp_path / "set")}))
    query = pd.DataFrame(
        {"id": [2**53 + offset for offset in range(7)], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]},
        index=[19, 4, 19, -1, 50, 4, 9],
    )
    original = adapter.predict_model_set
    calls = []

    def observe(frame, artifact):
        """Capture actual full-set scorer invocation sizes and preserve its real computation."""
        calls.append(len(frame))
        return original(frame, artifact)

    monkeypatch.setattr(adapter, "predict_model_set", observe)
    expected = model.predict(None, query)
    assert calls == [7]
    calls.clear()
    result = model.predict(
        None, query, params={"skyulf_spark_output": True, "skyulf_spark_batch_rows": 3}
    )
    assert calls == [3, 3, 1]
    assert result.index.equals(query.index) and result.id.tolist() == query.id.tolist()
    for name in expected:
        pd.testing.assert_series_equal(result[name].isna(), expected[name].isna())
        pd.testing.assert_series_equal(
            result[name].dropna(), expected[name].dropna(), check_dtype=False
        )
    assert result["first__exclusion_reason"].isna().all()


def test_empty_prediction_chunk_preserves_existing_scorer_behavior():
    """Empty input must not reach an empty concat or fabricate a different output schema."""
    from skyulf.integrations.mlflow.spark._spark_output import score_prediction_batches

    query = pd.DataFrame({"x": pd.Series(dtype=float)}, index=pd.Index([], name="source_index"))
    expected = pd.DataFrame({"prediction": pd.Series(dtype=float)}, index=query.index)
    score = Mock(return_value=expected)
    result = score_prediction_batches(query, score, {"skyulf_spark_batch_rows": 3}, True)
    score.assert_called_once_with(query)
    assert result is expected


def test_prediction_chunk_rejects_changed_row_identity():
    """A scorer cannot silently realign predictions with the wrong external record keys."""
    from skyulf.integrations.mlflow.spark._spark_output import score_prediction_batches

    query = pd.DataFrame({"x": [1.0, 2.0]}, index=[7, 3])
    with pytest.raises(ValueError, match="row identity"):
        score_prediction_batches(query, lambda frame: frame.reset_index(drop=True), None, True)
