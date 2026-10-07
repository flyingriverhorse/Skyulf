"""Real local MLflow packaging through an injected, simulated FE boundary."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
yaml = pytest.importorskip("yaml")

from skyulf.data.dataset import SplitDataset  # noqa: E402
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline  # noqa: E402
from skyulf.integrations.databricks.feature_store.config import (  # noqa: E402
    FeatureLookupSpec,
    FeatureTrainingSpec,
)
from skyulf.integrations.databricks.feature_store.lifecycle_config import (  # noqa: E402
    binding_json,
    serialize_feature_spec,
)
from skyulf.integrations.mlflow.models.local_feature_model import (  # noqa: E402
    FEATURE_STORE_KEY,
    RAW_MODEL_PATH_KEY,
    copy_feature_package,
    feature_package_models,
    log_feature_model_set,
    log_local_feature_model,
)
from skyulf.integrations.mlflow.models.local_model import SkyulfLocalPythonModel  # noqa: E402
from skyulf.integrations.mlflow.registration.registry import (  # noqa: E402
    load_registered_local_pipeline,
    load_run_local_pipeline,
    register_model,
    resolve_model,
)
from skyulf.pipeline import SkyulfPipeline  # noqa: E402


class SimulatedFeatureClient:
    """Mirror the documented wrapper layout using real local MLflow serializers."""

    def __init__(self, directory, training_set):
        """Retain the original native-boundary object for identity assertions."""
        self.directory = directory
        self.training_set = training_set
        self.calls = []

    def log_model(self, *, model, flavor, artifact_path, training_set, **kwargs):
        """Save a genuine raw pyfunc plus loader wrapper, without importing the SDK."""
        self.calls.append((model, flavor, training_set))
        assert training_set is self.training_set
        assert flavor is mlflow.pyfunc
        output_schema = kwargs.pop("output_schema")
        params = kwargs.pop("params")
        raw = self.directory / "feature_store" / "raw_model"
        flavor.save_model(
            path=str(raw),
            python_model=model,
            mlflow_model=mlflow.models.Model(
                signature=mlflow.models.ModelSignature(outputs=output_schema)
            ),
            **kwargs,
        )
        (raw.parent / "feature_spec.yaml").write_text(
            yaml.safe_dump(training_set.saved_spec), encoding="utf-8"
        )
        logged = flavor.log_model(
            name=artifact_path,
            loader_module="databricks.feature_store.mlflow_model",
            data_path=str(raw.parent),
            conda_env=str(raw / "conda.yaml"),
            signature=mlflow.models.ModelSignature(
                inputs=mlflow.types.Schema([mlflow.types.ColSpec("long", "id")]),
                outputs=output_schema,
                params=mlflow.models.infer_signature(params=params).params if params else None,
            ),
        )
        return logged


def training_set_for(spec, inputs):
    """Represent SDK public properties and its actual feature-spec YAML format."""
    feature = spec.lookups[0]
    columns: list[dict[str, Any]] = [
        {
            name: {
                "source": "feature_store",
                "data_type": "double",
                "table_name": feature.table_name,
                "feature_name": name,
                "lookup_key": list(feature.lookup_key),
                "timestamp_lookup_key": list(feature.timestamp_columns),
            }
        }
        if name in feature.feature_names
        else {
            name: {"source": "training_data", "data_type": "bigint" if name == "id" else "double"}
        }
        for name in inputs
    ]
    columns += [
        {
            name: {
                "source": "training_data",
                "include": False,
                "data_type": feature.timestamp_type
                if name in feature.timestamp_columns
                else "bigint",
            }
        }
        for name in spec.exclude_columns
    ]
    saved = {
        "input_columns": columns,
        "input_tables": [
            {
                feature.table_name: {
                    "table_id": "sdk-uc-id",
                    "lookback_window": None
                    if feature.lookback_window is None
                    else feature.lookback_window.total_seconds(),
                }
            }
        ],
        "serialization_version": 9,
    }
    native_columns = []
    for column in columns:
        name, data = next(iter(column.items()))
        info = SimpleNamespace(output_name=name)
        if data["source"] == "feature_store":
            info = SimpleNamespace(
                output_name=name,
                table_name=feature.table_name,
                feature_name=name,
                lookup_key=list(feature.lookup_key),
                timestamp_lookup_key=list(feature.timestamp_columns),
                default_value_str=None,
            )
        native_columns.append(
            SimpleNamespace(
                info=info,
                output_name=name,
                include=data.get("include", True),
                data_type=data.get("data_type", "bigint"),
            )
        )
    return SimpleNamespace(
        saved_spec=saved,
        feature_spec=SimpleNamespace(
            column_infos=native_columns,
            function_infos=[],
            table_infos=[
                SimpleNamespace(
                    table_name=feature.table_name,
                    lookback_window=None
                    if feature.lookback_window is None
                    else feature.lookback_window.total_seconds(),
                )
            ],
        ),
        get_output_columns=lambda: [*inputs, *([spec.label] if spec.label else [])],
    )


@pytest.fixture
def package_inputs(tmp_path, monkeypatch):
    """Fit a real pandas pipeline and create an explicit isolated tracking run."""
    monkeypatch.chdir(tmp_path)
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    tracking = mlflow.MlflowClient(tracking_uri=uri)
    experiment = tracking.create_experiment(
        "features", artifact_location=(tmp_path / "runs").as_uri()
    )
    run_id = tracking.create_run(experiment).info.run_id
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    path = tmp_path / "pipeline"
    save_local_pipeline(pipeline, path)
    spec = FeatureTrainingSpec(
        lookups=(
            FeatureLookupSpec(
                table_name="main.features.values", lookup_key=("id",), feature_names=("x",)
            ),
        ),
        label="target",
        exclude_columns=("id",),
    )
    binding = {
        "version": 1,
        "lookup_spec": serialize_feature_spec(spec),
        "lookup_evidence": {
            "policy": "training_snapshot",
            "feature_tables": [
                {"table_name": "main.features.values", "table_id": "delta-id", "version": 4}
            ],
        },
    }
    native = training_set_for(spec, ["x"])
    client = SimulatedFeatureClient(tmp_path / "sdk", native)
    options = {
        "training_set": native,
        "lookup_spec": spec,
        "lookup_binding": binding,
        "run_id": run_id,
        "artifact_path": "model",
        "tracking_uri": uri,
        "client": client,
    }
    yield path, options, tracking
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(previous)


def test_feature_package_preserves_real_raw_model_and_registry_loading(package_inputs):
    """FE envelopes must retain the fitted pyfunc and explicit registry identity."""
    path, options, tracking = package_inputs
    previous = mlflow.get_tracking_uri()
    uri = log_local_feature_model(path, **options)
    local = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=uri, tracking_uri=options["tracking_uri"])
    )
    outer, raw, raw_path = feature_package_models(local)
    loaded = mlflow.pyfunc.load_model(str(raw_path))
    assert isinstance(loaded.unwrap_python_model(), SkyulfLocalPythonModel)
    assert loaded.predict(pd.DataFrame({"x": [5.0]}))["prediction"].tolist() == pytest.approx(
        [10.0]
    )
    assert raw.signature.inputs.input_names() == ["x"]
    assert raw.signature.outputs.input_names() == ["prediction"]
    assert outer.metadata[FEATURE_STORE_KEY] == options["lookup_binding"]
    assert outer.metadata[RAW_MODEL_PATH_KEY] == raw_path.relative_to(local).as_posix()
    assert outer.metadata["skyulf_partition_safety"] == raw.metadata["skyulf_partition_safety"]
    assert raw.signature.params.to_dict()[1]["type"] == "long"
    wheel_pins = [
        line
        for line in (local / "requirements.txt").read_text().splitlines()
        if line.startswith("code/")
    ]
    assert wheel_pins
    assert all((local / pin).is_file() for pin in wheel_pins)
    assert options["client"].calls[0][2] is options["training_set"]
    assert mlflow.get_tracking_uri() == previous
    assert mlflow.active_run() is None
    digest = load_local_pipeline(path).manifest.pipeline_sha256
    from_run = load_run_local_pipeline(uri, digest=digest, tracking_uri=options["tracking_uri"])
    assert from_run.feature_lookup_json == binding_json(options["lookup_binding"])
    registered = register_model(
        uri,
        "feature-model",
        tracking_uri=options["tracking_uri"],
        registry_uri=options["tracking_uri"],
    )
    resolved = resolve_model(
        "feature-model",
        version=registered.version,
        tracking_uri=options["tracking_uri"],
        registry_uri=options["tracking_uri"],
    )
    from_registry = load_registered_local_pipeline(
        resolved, tracking_uri=options["tracking_uri"], registry_uri=options["tracking_uri"]
    )
    assert from_registry.feature_lookup_json == from_run.feature_lookup_json
    assert tracking.get_run(options["run_id"]).info.status == "RUNNING"


def test_feature_model_set_preserves_complete_raw_package(package_inputs, tmp_path):
    """One feature envelope must retain both fitted branches and record-key outputs."""
    from skyulf.inference.bundle import ColumnSpec
    from skyulf.inference.model_set import ComponentReference, save_model_set
    from skyulf.integrations.mlflow.models.model_set import load_registered_model_set

    path, options, _ = package_inputs
    artifact = load_local_pipeline(path)
    set_path = tmp_path / "set"
    saved = save_model_set(
        set_path,
        {
            name: (
                ComponentReference(
                    name=name, version="1", digest=artifact.manifest.pipeline_sha256
                ),
                path,
            )
            for name in ("left", "right")
        },
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    spec = replace(options["lookup_spec"], label=None, exclude_columns=())
    options["lookup_spec"] = spec
    options["lookup_binding"] = {
        **options["lookup_binding"],
        "lookup_spec": serialize_feature_spec(spec),
    }
    native = training_set_for(spec, [c.name for c in saved.manifest.input_schema])
    options["training_set"] = native
    options["client"] = SimulatedFeatureClient(tmp_path / "set-sdk", native)
    uri = log_feature_model_set(set_path, **options)
    local = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=uri, tracking_uri=options["tracking_uri"])
    )
    _, _, raw_path = feature_package_models(local)
    result = mlflow.pyfunc.load_model(str(raw_path)).predict(pd.DataFrame({"id": [2], "x": [5.0]}))
    assert result["id"].tolist() == [2]
    assert result["left__prediction"].tolist() == pytest.approx([10.0])
    assert result["right__prediction"].tolist() == pytest.approx([10.0])
    registered = register_model(
        uri,
        "feature-set",
        tracking_uri=options["tracking_uri"],
        registry_uri=options["tracking_uri"],
    )
    resolved = resolve_model(
        "feature-set",
        version=registered.version,
        tracking_uri=options["tracking_uri"],
        registry_uri=options["tracking_uri"],
    )
    loaded = load_registered_model_set(
        resolved, tracking_uri=options["tracking_uri"], registry_uri=options["tracking_uri"]
    )
    assert loaded.feature_lookup_json == binding_json(options["lookup_binding"])


def test_nullable_integer_lookup_rejected_before_sdk(package_inputs, tmp_path):
    """FE has no pre-Arrow hook to preserve nullable integers larger than 2**53."""
    _, options, _ = package_inputs
    frame = pd.DataFrame(
        {"x": pd.Series([1, 2, 3, 4], dtype="Int64"), "target": [2.0, 4.0, 6.0, 8.0]}
    )
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    path = tmp_path / "nullable"
    save_local_pipeline(pipeline, path)
    with pytest.raises(ValueError, match="nullable primitive transport"):
        log_local_feature_model(path, **options)
    assert options["client"].calls == []


def test_matching_active_run_and_copy_adoption_preserve_context(package_inputs):
    """Native logging and competition adoption keep the existing run selected."""
    path, options, tracking = package_inputs
    mlflow.set_tracking_uri(options["tracking_uri"])
    with mlflow.start_run(run_id=options["run_id"]):
        uri = log_local_feature_model(path, **options)
        assert mlflow.active_run().info.run_id == options["run_id"]
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
        parent = tracking.create_run(
            tracking.get_run(options["run_id"]).info.experiment_id
        ).info.run_id
        copied = copy_feature_package(
            local, run_id=parent, artifact_path="adopted", tracking_uri=options["tracking_uri"]
        )
        loaded = load_run_local_pipeline(
            copied,
            digest=load_local_pipeline(path).manifest.pipeline_sha256,
            tracking_uri=options["tracking_uri"],
        )
        assert mlflow.active_run().info.run_id == options["run_id"]
    assert loaded.feature_lookup_json == binding_json(options["lookup_binding"])


def test_sdk_failure_restores_context_and_status(package_inputs, monkeypatch):
    """A failed native SDK call must not strand a temporary fluent run."""
    path, options, tracking = package_inputs
    previous = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_RUN_ID", "caller-environment-run")

    class FailingClient:
        """Raise at the explicitly injected external boundary."""

        def log_model(self, **kwargs):
            """Exercise cleanup after native logging has started."""
            raise RuntimeError("native failure")

    options["client"] = FailingClient()
    with pytest.raises(RuntimeError, match="native failure"):
        log_local_feature_model(path, **options)
    assert mlflow.active_run() is None
    assert mlflow.get_tracking_uri() == previous
    assert tracking.get_run(options["run_id"]).info.status == "RUNNING"
    import os

    environment_run = os.environ.get("MLFLOW_RUN_ID")
    assert environment_run == "caller-environment-run"


def test_same_table_conflicting_lookback_rejected_before_native_creation():
    """One native table-level window cannot represent two different requested windows."""
    from datetime import timedelta

    first = FeatureLookupSpec(
        table_name="main.features.values",
        lookup_key=("id",),
        feature_names=("x",),
        timestamp_lookup_key="event_time",
        lookback_window=timedelta(hours=1),
    )
    second = replace(first, feature_names=("z",), lookback_window=timedelta(hours=2))
    with pytest.raises(ValueError, match="lookback_window"):
        FeatureTrainingSpec(lookups=(first, second), label="target")


def test_training_contract_mismatch_fails_before_sdk(package_inputs):
    """A native TrainingSet with a different lookup cannot be relabeled by metadata."""
    path, options, _ = package_inputs
    options["training_set"].feature_spec.column_infos[0].info.table_name = "main.features.other"
    with pytest.raises(ValueError, match="lookup"):
        log_local_feature_model(path, **options)
    assert options["client"].calls == []


@pytest.mark.parametrize(
    "mutation", ["path", "metadata", "feature_spec", "missing_certificate", "wheel"]
)
def test_feature_envelope_rejects_tampering(package_inputs, mutation):
    """Saved paths, raw certificates and native lookup instructions remain bound."""
    path, options, _ = package_inputs
    uri = log_local_feature_model(path, **options)
    local = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=uri, tracking_uri=options["tracking_uri"])
    )
    outer, raw, raw_path = feature_package_models(local)
    if mutation == "path":
        outer.metadata[RAW_MODEL_PATH_KEY] = "../outside"
        outer.save(str(local / "MLmodel"))
    elif mutation == "metadata":
        raw.metadata["local_pipeline_digest"] = "0" * 64
        raw.save(str(raw_path / "MLmodel"))
    elif mutation == "missing_certificate":
        raw.metadata.pop("skyulf_partition_safety")
        raw.save(str(raw_path / "MLmodel"))
    elif mutation == "wheel":
        next((local / "code").glob("*.whl")).write_bytes(b"changed")
    else:
        spec_path = raw_path.parent / "feature_spec.yaml"
        content = spec_path.read_text(encoding="utf-8").replace(
            "main.features.values", "main.features.other"
        )
        spec_path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        feature_package_models(local)


def test_different_active_run_rejected_without_ending_it(package_inputs):
    """Logging cannot take over another caller's fluent run."""
    path, options, tracking = package_inputs
    mlflow.set_tracking_uri(options["tracking_uri"])
    experiment = tracking.get_run(options["run_id"]).info.experiment_id
    with mlflow.start_run(experiment_id=experiment) as active:
        with pytest.raises(ValueError, match="active run"):
            log_local_feature_model(path, **options)
        assert mlflow.active_run().info.run_id == active.info.run_id
    assert options["client"].calls == []


@pytest.mark.parametrize(
    "mutation", [None, "keys", "time", "dtype", "lookback", "defaults", "excluded"]
)
def test_native_and_saved_lookup_contract_validation(tmp_path, mutation):
    """Validate actual lookup instructions independently of metadata and column order."""
    from datetime import timedelta

    from skyulf.integrations.mlflow.models._feature_package_spec import (
        validate_native_training_set,
        validate_saved_feature_spec,
    )

    spec = FeatureTrainingSpec(
        lookups=(
            FeatureLookupSpec(
                table_name="main.features.values",
                lookup_key=("id",),
                feature_names=("x",),
                timestamp_lookup_key="event_time",
                lookback_window=timedelta(hours=1),
            ),
        ),
        label="target",
        exclude_columns=("id", "event_time"),
    )
    native = training_set_for(spec, ["z", "x"])
    feature = native.feature_spec.column_infos[1]
    if mutation == "keys":
        feature.info.lookup_key = ["other_id"]
        native.saved_spec["input_columns"][1]["x"]["lookup_key"] = ["other_id"]
    elif mutation == "time":
        native.feature_spec.column_infos[-1].data_type = "date"
        native.saved_spec["input_columns"][-1]["event_time"]["data_type"] = "date"
    elif mutation == "dtype":
        feature.data_type = "bigint"
        native.saved_spec["input_columns"][1]["x"]["data_type"] = "bigint"
    elif mutation == "lookback":
        native.feature_spec.table_infos[0].lookback_window = 12.0
        native.saved_spec["input_tables"][0]["main.features.values"]["lookback_window"] = 12.0
    elif mutation == "defaults":
        feature.info.default_value_str = "0"
        native.saved_spec["input_columns"][1]["x"]["default_value"] = "0"
    elif mutation == "excluded":
        native.feature_spec.column_infos[-1].output_name = "unapproved"
        native.saved_spec["input_columns"][-1] = {
            "unapproved": {"source": "training_data", "include": False}
        }
    path = tmp_path / "feature_spec.yaml"
    path.write_text(yaml.safe_dump(native.saved_spec), encoding="utf-8")
    if mutation is None:
        validate_native_training_set(native, spec, {"x": "double", "z": "double"})
        assert len(validate_saved_feature_spec(path, spec, {"x": "double", "z": "double"})) == 64
    else:
        with pytest.raises(ValueError):
            validate_native_training_set(native, spec, {"x": "double", "z": "double"})
        with pytest.raises(ValueError):
            validate_saved_feature_spec(path, spec, {"x": "double", "z": "double"})
