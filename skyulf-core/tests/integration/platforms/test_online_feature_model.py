"""Real saved online guards through both fitted engines and native envelopes."""

import time
from pathlib import Path

import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
pl = pytest.importorskip("polars")
pytest.importorskip("yaml")

from tests.integration.platforms.test_feature_pipeline_model import (
    SimulatedFeatureClient,
    training_set_for,
)

from skyulf.data.dataset import SplitDataset
from skyulf.inference.bundle import ColumnSpec
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.model_set import ComponentReference, save_model_set
from skyulf.integrations.databricks.feature_store.config import (
    FeatureLookupSpec,
    FeatureTrainingSpec,
)
from skyulf.integrations.databricks.feature_store.lifecycle_config import serialize_feature_spec
from skyulf.integrations.databricks.feature_store.online_policy import (
    ONLINE_FEATURES_KEY,
    OnlineFeaturePolicy,
)
from skyulf.integrations.mlflow.models.feature_model import (
    feature_package_models,
    log_feature_model_set,
    log_feature_pipeline_model,
)
from skyulf.pipeline import SkyulfPipeline


class NativeSignatureClient(SimulatedFeatureClient):
    """Mirror native required source and optional fetched feature signature semantics."""

    def log_model(self, **kwargs):
        """Keep real raw serialization and replace only the simulated native signature."""
        result = super().log_model(**kwargs)
        path = Path(mlflow.artifacts.download_artifacts(artifact_uri=result.model_uri))
        model = mlflow.models.Model.load(path)
        types = {"bigint": "long", "double": "double", "timestamp": "datetime", "string": "string"}
        model.signature.inputs = mlflow.types.Schema(
            [
                mlflow.types.ColSpec(
                    types[column.data_type],
                    column.output_name,
                    required=not hasattr(column.info, "table_name"),
                )
                for column in self.training_set.feature_spec.column_infos
            ]
        )
        model.save(str(path / "MLmodel"))
        return result


@pytest.fixture(params=["pandas", "polars"])
def online_artifact(tmp_path, request):
    """Fit an imputer and remove freshness only after the saved raw input boundary."""
    frame = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "fresh_at": [1000.0] * 4, "target": [2.0, 4.0, 6.0, 8.0]}
    )
    if request.param.split("_")[0] == "polars":
        frame = pl.from_pandas(frame)
    preprocessing = [
        {
            "name": "fill",
            "transformer": "SimpleImputer",
            "params": {"columns": ["x"], "strategy": "mean"},
        },
        {
            "name": "drop_time",
            "transformer": "DropMissingColumns",
            "params": {"columns": ["fresh_at"]},
        },
    ]
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [] if request.param.endswith("_plain") else preprocessing,
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    path = tmp_path / "pipeline"
    save_pipeline(pipeline, path)
    return path


@pytest.fixture(params=["pipeline", "model_set"])
def online_package(online_artifact, tmp_path, monkeypatch, request):
    """Use genuine MLflow serialization and an isolated simulated native FE envelope."""
    monkeypatch.chdir(tmp_path)
    previous = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.create_experiment("online", artifact_location=(tmp_path / "runs").as_uri())
    run_id = client.create_run(experiment).info.run_id
    artifact = load_pipeline(online_artifact)
    original = (online_artifact / "pipeline.pkl").read_bytes()
    path = online_artifact
    inputs = list(artifact.manifest.input_columns)
    logger = log_feature_pipeline_model
    if request.param == "model_set":
        path = tmp_path / "set"
        saved = save_model_set(
            path,
            {
                name: (
                    ComponentReference(
                        name=name, version="1", digest=artifact.manifest.pipeline_sha256
                    ),
                    online_artifact,
                )
                for name in ("left", "right")
            },
            record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        )
        inputs = [column.name for column in saved.manifest.input_schema]
        logger = log_feature_model_set
    spec = FeatureTrainingSpec(
        lookups=(
            FeatureLookupSpec(
                table_name="main.features.values",
                lookup_key=("id",),
                feature_names=("x", "fresh_at"),
            ),
        ),
        label=None,
        exclude_columns=("id",) if request.param == "pipeline" else (),
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
    native = training_set_for(spec, inputs)
    policy = OnlineFeaturePolicy(
        required_features=("x", "fresh_at"), freshness=(("fresh_at", 60.0),)
    )
    try:
        uri = logger(
            path,
            training_set=native,
            lookup_spec=spec,
            lookup_binding=binding,
            run_id=run_id,
            artifact_path="model",
            tracking_uri=tracking_uri,
            client=NativeSignatureClient(tmp_path / "sdk", native),
            online_policy=policy,
        )
        local = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=uri, tracking_uri=tracking_uri)
        )
        assert (online_artifact / "pipeline.pkl").read_bytes() == original
        yield local, policy, binding, request.param
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)


def test_real_saved_guard_precedes_imputer_and_preserves_fresh_prediction(online_package):
    """Saved guards reject null/stale/future inputs before imputation for every component."""
    local, policy, binding, kind = online_package
    outer, raw, raw_path = feature_package_models(local)
    assert outer.metadata["skyulf_feature_store"] == binding
    assert (
        outer.metadata[ONLINE_FEATURES_KEY] == raw.metadata[ONLINE_FEATURES_KEY] == policy.to_dict()
    )
    model = mlflow.pyfunc.load_model(str(raw_path))
    frame = pd.DataFrame({"x": [5.0], "fresh_at": [time.time() - 1]})
    if kind == "model_set":
        frame["id"] = [2]
    result = model.predict(frame)
    field = "prediction" if kind == "pipeline" else "left__prediction"
    assert result[field].tolist() == pytest.approx([10.0])
    for column, value in [
        ("x", float("nan")),
        ("fresh_at", 0.0),
        ("fresh_at", time.time() + 1000),
        ("fresh_at", float("inf")),
    ]:
        invalid = frame.copy()
        invalid[column] = value
        with pytest.raises(ValueError, match="Online"):
            model.predict(invalid)
    invalid = frame.copy()
    invalid["fresh_at"] = 0.0
    with pytest.raises(ValueError, match="Online"):
        model.unwrap_python_model().predict(None, invalid, params={"now": 0.0})
    raw.metadata[ONLINE_FEATURES_KEY]["freshness"][0][1] = 999.0
    raw.save(str(raw_path / "MLmodel"))
    with pytest.raises(ValueError, match="metadata|policy"):
        feature_package_models(local)
    raw.flavors["python_function"]["config"][ONLINE_FEATURES_KEY]["freshness"][0][1] = 999.0
    raw.save(str(raw_path / "MLmodel"))
    with pytest.raises(ValueError, match="policy"):
        mlflow.pyfunc.load_model(str(raw_path))


def test_online_request_schema_uses_all_native_source_columns(tmp_path):
    """Excluded source metadata remains required while fetched features stay optional."""
    import yaml

    from skyulf.integrations.databricks.serving.online_endpoints import online_request_schema

    rows = [
        {"id": {"source": "training_data", "data_type": "bigint", "include": False}},
        {"segment": {"source": "training_data", "data_type": "string", "include": False}},
        {"x": {"source": "feature_store", "data_type": "double"}},
        {"fresh_at": {"source": "feature_store", "data_type": "double"}},
    ]
    path = tmp_path / "feature_spec.yaml"
    path.write_text(yaml.safe_dump({"input_columns": rows}), encoding="utf-8")
    signature = mlflow.models.ModelSignature(
        inputs=mlflow.types.Schema(
            [
                mlflow.types.ColSpec("long", "id"),
                mlflow.types.ColSpec("string", "segment"),
                mlflow.types.ColSpec("double", "x", required=False),
                mlflow.types.ColSpec("double", "fresh_at", required=False),
            ]
        )
    )
    assert online_request_schema(path, signature) == (("id", "int64"), ("segment", "string"))
    signature.inputs.inputs[1] = mlflow.types.ColSpec("string", "segment", required=False)
    with pytest.raises(ValueError, match="signature"):
        online_request_schema(path, signature)


def test_online_request_schema_rejects_temporal_source_with_guidance(tmp_path):
    """Required temporal inputs cannot silently disappear from entity-only serving requests."""
    import yaml

    from skyulf.integrations.databricks.serving.online_endpoints import online_request_schema

    path = tmp_path / "feature_spec.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input_columns": [
                    {
                        "event_at": {
                            "source": "training_data",
                            "data_type": "timestamp",
                            "include": False,
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    signature = mlflow.models.ModelSignature(
        inputs=mlflow.types.Schema([mlflow.types.ColSpec("datetime", "event_at")])
    )
    with pytest.raises(ValueError, match="non-temporal"):
        online_request_schema(path, signature)


@pytest.mark.parametrize("online_artifact", ["pandas_plain", "polars_plain"], indirect=True)
def test_prepare_online_endpoint_inspects_real_native_package(online_package, monkeypatch, request):
    """Admission checks the actual saved raw certificate and native source signature."""
    from skyulf.integrations.databricks.serving import online_endpoints
    from skyulf.integrations.databricks.serving.contracts import PinnedEndpointSpec
    from skyulf.integrations.mlflow.registration.registry import ResolvedModel

    local, policy, binding, kind = online_package
    _, raw, raw_path = feature_package_models(local)
    digest_key = "local_pipeline_digest" if kind == "pipeline" else "model_set_digest"
    spec = PinnedEndpointSpec("online-test", "main.models.online", "1", "main", "logs", "online")
    resolved = ResolvedModel(
        name=spec.model_name,
        version="1",
        model_uri=spec.model_uri,
        digest=raw.metadata[digest_key],
        signature=raw.signature,
    )
    monkeypatch.setattr(online_endpoints, "resolve_model", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(
        online_endpoints, "download_registered_package", lambda *args, **kwargs: str(local)
    )
    if request.node.callspec.params["online_artifact"].startswith("polars"):
        with pytest.raises(ValueError, match="fitted on pandas"):
            online_endpoints.prepare_online_endpoint(
                spec, tracking_uri="sqlite:///:memory:", registry_uri="sqlite:///:memory:"
            )
        return
    plan = online_endpoints.prepare_online_endpoint(
        spec, tracking_uri="sqlite:///:memory:", registry_uri="sqlite:///:memory:"
    )
    assert plan.online_contract is not None
    assert plan.input_columns == ("id",)
    assert plan.input_schema == (("id", "int64"),)
    assert plan.config["config"]["served_entities"][0]["entity_version"] == "1"
    raw.flavors["python_function"]["config"][ONLINE_FEATURES_KEY]["freshness"][0][1] = 999.0
    raw.save(str(raw_path / "MLmodel"))
    with pytest.raises(ValueError, match="policy"):
        online_endpoints.prepare_online_endpoint(
            spec, tracking_uri="sqlite:///:memory:", registry_uri="sqlite:///:memory:"
        )


def test_online_signature_cannot_omit_composite_lookup_key(tmp_path):
    """Every key referenced by saved lookup instructions must remain a source input."""
    import yaml

    from skyulf.integrations.databricks.serving.online_endpoints import online_request_schema

    rows = [
        {"id": {"source": "training_data", "data_type": "bigint"}},
        {"x": {"source": "feature_store", "data_type": "double", "lookup_key": ["id", "tenant"]}},
    ]
    path = tmp_path / "feature_spec.yaml"
    path.write_text(yaml.safe_dump({"input_columns": rows}), encoding="utf-8")
    signature = mlflow.models.ModelSignature(
        inputs=mlflow.types.Schema(
            [
                mlflow.types.ColSpec("long", "id"),
                mlflow.types.ColSpec("double", "x", required=False),
            ]
        )
    )
    with pytest.raises(ValueError, match="lookup keys"):
        online_request_schema(path, signature)
