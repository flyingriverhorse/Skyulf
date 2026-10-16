"""Pinned Databricks serving enrollment and named-record request contracts."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

pytest.importorskip("mlflow")
pytest.importorskip("databricks.sdk")

from skyulf.data.dataset import SplitDataset  # noqa: E402
from skyulf.inference.fitted_pipeline import (
    load_pipeline,
    save_pipeline,  # noqa: E402
)
from skyulf.integrations.databricks.serving import (  # noqa: E402
    PinnedEndpointSpec,
    build_pinned_endpoint,
    create_pinned_endpoint,
    endpoint_ready,
    prepare_pinned_endpoint,
    query_named_records,
    require_pinned_endpoint_ready,
)
from skyulf.integrations.mlflow.models.pipeline_model import _signature  # noqa: E402
from skyulf.integrations.mlflow.registration.registry import ResolvedModel  # noqa: E402
from skyulf.integrations.mlflow.shared._nullable_transport import (  # noqa: E402
    TRANSPORT_KEY,
    transport_spec,
)
from skyulf.integrations.mlflow.spark import spark_model  # noqa: E402
from skyulf.pipeline import SkyulfPipeline  # noqa: E402


@pytest.fixture
def artifact(tmp_path):
    """A fitted model gives admission an independently inspected contract."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [2.0, 4.0, 6.0]})
    pipeline = SkyulfPipeline({"modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    save_pipeline(pipeline, tmp_path / "model")
    return load_pipeline(tmp_path / "model")


@pytest.fixture
def package(artifact):
    """The package fixture mirrors a certified saved MLflow model."""
    certificate = spark_model.partition_safety_certificate(artifact)
    inputs, _ = spark_model._contract(artifact)
    return SimpleNamespace(
        metadata={
            spark_model.SAFETY_KEY: certificate,
            spark_model.SOURCE_KEY: spark_model.runtime_source_digest(),
            TRANSPORT_KEY: transport_spec(inputs),
            "skyulf_artifact_kind": "local_pipeline",
            "skyulf_execution_scope": "whole_frame_local",
            "local_pipeline_digest": certificate["pipeline_sha256"],
        },
        signature=_signature(artifact, spark_certified=True),
    )


@pytest.fixture
def spec():
    """A concrete UC version keeps serving enrollment immutable."""
    return PinnedEndpointSpec("sm23b-test", "main.ml.model", "7", "main", "observability", "sm23b")


@pytest.fixture
def resolved(spec, artifact):
    """A pinned registry read binds the model selector to the fitted digest."""
    return ResolvedModel(
        spec.model_name,
        spec.model_version,
        f"models:/{spec.model_name}/{spec.model_version}",
        None,
        artifact.manifest.pipeline_sha256,
    )


def test_config_pins_certified_model_and_inference_logging(spec, resolved, artifact, package):
    """Provisioning must log the exact model identity on a small isolated endpoint."""
    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    assert plan.output_schema == (("prediction", "float64"),)
    assert plan.config == {
        "name": "sm23b-test",
        "config": {
            "served_entities": [
                {
                    "entity_name": "main.ml.model",
                    "entity_version": "7",
                    "workload_type": "CPU",
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                }
            ],
        },
        "telemetry_config": {
            "table_names": {
                "logs_table": "main.observability.sm23b_otel_logs",
                "traces_table": "main.observability.sm23b_otel_spans",
                "metrics_table": "main.observability.sm23b_otel_metrics",
            },
            "inference_table_config": {"sampling_fraction": 1.0},
            "enabled_telemetry_features": ["TELEMETRY_FEATURE_INFERENCE_TABLE"],
        },
    }
    assert plan.input_columns == ("x",)


def test_endpoint_settings_keep_existing_defaults(spec):
    """Old callers retain the existing compute choice and a bounded setup wait."""
    assert (spec.workload_type, spec.workload_size, spec.scale_to_zero_enabled) == (
        "CPU",
        "Small",
        True,
    )
    assert spec.ready_timeout_s == 3600


@pytest.mark.parametrize("kind", ["CPU", "CPU_MEDIUM", "CPU_LARGE"])
@pytest.mark.parametrize("size", ["Small", "Medium", "Large"])
def test_endpoint_compute_overrides_reach_native_config(
    spec, resolved, artifact, package, kind, size
):
    """Explicit compute values must survive admission into the native create body."""
    configured = replace(
        spec,
        workload_type=kind,
        workload_size=size,
        scale_to_zero_enabled=False,
        ready_timeout_s=120,
    )
    plan = build_pinned_endpoint(
        configured, resolved=resolved, artifact=artifact, package_info=package
    )
    entity = plan.config["config"]["served_entities"][0]
    assert entity["workload_type"] == kind
    assert entity["workload_size"] == size
    assert entity["scale_to_zero_enabled"] is False
    assert "ready_timeout_s" not in entity


@pytest.mark.parametrize(
    "field,value",
    [
        ("workload_type", "GPU_SMALL"),
        ("workload_type", "cpu"),
        ("workload_type", None),
        ("workload_type", []),
        ("workload_size", "small"),
        ("workload_size", "XLarge"),
        ("workload_size", False),
        ("workload_size", []),
        ("scale_to_zero_enabled", "false"),
        ("scale_to_zero_enabled", 0),
        ("scale_to_zero_enabled", None),
        ("ready_timeout_s", True),
        ("ready_timeout_s", 0),
        ("ready_timeout_s", -1),
        ("ready_timeout_s", 1.5),
        ("ready_timeout_s", "3600"),
        ("ready_timeout_s", None),
    ],
)
def test_endpoint_settings_reject_invalid_values_before_native_io(spec, field, value):
    """Configuration mistakes must fail clearly, including truthy strings and booleans as integers."""
    with pytest.raises(ValueError, match=field):
        replace(spec, **{field: value})


def _setup_clock(monkeypatch):
    """Advance setup time deterministically without making the tests sleep."""
    from skyulf.integrations.databricks.serving import endpoints

    clock = [0.0]

    def advance(seconds):
        """Simulate only the requested polling delay."""
        clock[0] += seconds

    monkeypatch.setattr(endpoints, "monotonic", lambda: clock[0], raising=False)
    monkeypatch.setattr(endpoints, "sleep", advance, raising=False)
    return clock


def test_create_waits_for_settled_ready_configuration(
    spec, resolved, artifact, package, monkeypatch
):
    """A submitted create or READY with a pending update is not a successful setup."""
    from databricks.sdk.errors import NotFound

    configured = replace(spec, logging_mode="disabled")
    plan = build_pinned_endpoint(
        configured, resolved=resolved, artifact=artifact, package_info=package
    )
    ready = deepcopy(plan.config) | {"state": {"ready": "READY", "config_update": "NOT_UPDATING"}}
    pending = deepcopy(ready)
    pending["state"]["config_update"] = "IN_PROGRESS"
    client = SimpleNamespace(serving_endpoints=Mock())
    client.serving_endpoints.get.side_effect = [NotFound("absent"), pending, ready]
    clock = _setup_clock(monkeypatch)
    assert create_pinned_endpoint(client, plan) == ready
    client.serving_endpoints.create.assert_called_once()
    assert clock[0] > 0


def test_create_timeout_never_retries_or_deletes_endpoint(
    spec, resolved, artifact, package, monkeypatch
):
    """Timeout must leave the submitted resource available for inspection and explicit recovery."""
    from databricks.sdk.errors import NotFound

    configured = replace(spec, logging_mode="disabled", ready_timeout_s=2)
    plan = build_pinned_endpoint(
        configured, resolved=resolved, artifact=artifact, package_info=package
    )
    pending = {"state": {"ready": "NOT_READY", "config_update": "IN_PROGRESS"}}
    client = SimpleNamespace(serving_endpoints=Mock())
    client.serving_endpoints.get.side_effect = [NotFound("absent"), pending, pending]
    clock = _setup_clock(monkeypatch)
    with pytest.raises(TimeoutError, match="sm23b-test.*2"):
        create_pinned_endpoint(client, plan)
    client.serving_endpoints.create.assert_called_once()
    client.serving_endpoints.delete.assert_not_called()
    assert clock[0] == 2


@pytest.mark.parametrize("update", ["UPDATE_FAILED", "UPDATE_CANCELED"])
def test_create_stops_on_failed_deployment(spec, resolved, artifact, package, monkeypatch, update):
    """A terminal platform failure must surface immediately instead of waiting until timeout."""
    from databricks.sdk.errors import NotFound

    configured = replace(spec, logging_mode="disabled")
    plan = build_pinned_endpoint(
        configured, resolved=resolved, artifact=artifact, package_info=package
    )
    client = SimpleNamespace(serving_endpoints=Mock())
    client.serving_endpoints.get.side_effect = [
        NotFound("absent"),
        {"state": {"ready": "NOT_READY", "config_update": update}},
    ]
    clock = _setup_clock(monkeypatch)
    with pytest.raises(ValueError, match=update):
        create_pinned_endpoint(client, plan)
    assert clock[0] == 0


def test_wait_rejects_wrong_ready_config_immediately(
    spec, resolved, artifact, package, monkeypatch
):
    """A settled endpoint serving a different model cannot pass by waiting longer."""
    from skyulf.integrations.databricks.serving import wait_for_pinned_endpoint_ready

    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    ready = deepcopy(_ready_telemetry_response(plan))
    ready["config"]["served_entities"][0]["entity_version"] = "99"
    client = SimpleNamespace(api_client=Mock())
    client.api_client.do.return_value = ready
    clock = _setup_clock(monkeypatch)
    with pytest.raises(ValueError, match="selector"):
        wait_for_pinned_endpoint_ready(client, plan)
    assert clock[0] == 0


def test_wait_propagates_api_failure_without_retry(spec, resolved, artifact, package, monkeypatch):
    """Permission errors must not be disguised as an hour-long readiness timeout."""
    from databricks.sdk.errors import PermissionDenied

    from skyulf.integrations.databricks.serving import wait_for_pinned_endpoint_ready

    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    client = SimpleNamespace(api_client=Mock())
    client.api_client.do.side_effect = PermissionDenied("denied")
    clock = _setup_clock(monkeypatch)
    with pytest.raises(PermissionDenied):
        wait_for_pinned_endpoint_ready(client, plan)
    client.api_client.do.assert_called_once()
    assert clock[0] == 0


def test_inference_only_rollout_matches_native_persisted_sinks(spec, resolved, artifact, package):
    """Native inference telemetry persists only its enabled log sink, not unused sinks."""
    from skyulf.integrations.databricks.serving import (
        build_rollout_endpoint,
        rollout_endpoint_ready,
    )

    champion = build_pinned_endpoint(
        spec, resolved=resolved, artifact=artifact, package_info=package
    )
    challenger_spec = replace(spec, model_version="8")
    challenger = build_pinned_endpoint(
        challenger_spec,
        resolved=replace(resolved, version="8", model_uri=challenger_spec.model_uri),
        artifact=artifact,
        package_info=package,
    )
    plan = build_rollout_endpoint(champion, challenger)
    readback = deepcopy(plan.config)
    readback["state"] = {"ready": "READY", "config_update": "NOT_UPDATING"}
    readback["telemetry_config"] = {
        "table_names": {"logs_table": spec.telemetry_logs_table},
        "inference_table_config": {"name": spec.inference_table, "sampling_fraction": 1.0},
        "enabled_telemetry_features": ["TELEMETRY_FEATURE_INFERENCE_TABLE"],
    }
    assert rollout_endpoint_ready(readback, plan, challenger_percentage=0)


@pytest.mark.parametrize("model_version", ["latest", "@champion", "0", "-1", "1.0"])
def test_alias_or_nonconcrete_version_rejected(model_version):
    """An endpoint cannot drift with an alias or accept an invalid version."""
    with pytest.raises(ValueError, match="version"):
        PinnedEndpointSpec("safe", "main.ml.model", model_version, "main", "obs", "prefix")


@pytest.mark.parametrize("name", ["", "bad name", "a/b", "_hidden", "a" * 64])
def test_invalid_endpoint_name_rejected(name):
    """Unsafe or ambiguous endpoint names must fail before SDK calls."""
    with pytest.raises(ValueError, match="endpoint"):
        PinnedEndpointSpec(name, "main.ml.model", "7", "main", "obs", "prefix")


def test_invalid_uc_identifiers_rejected():
    """Logging targets and models must resolve to concrete UC identifiers."""
    with pytest.raises(ValueError, match="model"):
        PinnedEndpointSpec("safe", "model", "7", "main", "obs", "prefix")
    with pytest.raises(ValueError, match="schema"):
        PinnedEndpointSpec("safe", "main.ml.model", "7", "main", "bad.name", "prefix")


def test_invalid_logging_mode_rejected(spec):
    """Unknown logging transports cannot silently select a provisioning path."""
    with pytest.raises(ValueError, match="logging_mode"):
        replace(spec, logging_mode="unknown")


def test_uncertified_or_mismatched_package_rejected(spec, resolved, artifact, package):
    """A pyfunc without matching certificate, source and signature is inadmissible."""
    for field in (spark_model.SAFETY_KEY, spark_model.SOURCE_KEY, "local_pipeline_digest"):
        changed = deepcopy(package)
        changed.metadata[field] = "wrong"
        with pytest.raises(ValueError):
            build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=changed)
    changed = deepcopy(package)
    changed.signature = None
    with pytest.raises(ValueError, match="signature"):
        build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=changed)


def test_resolved_identity_must_match_spec_and_artifact(spec, resolved, artifact, package):
    """A downloaded package cannot be repointed at another registered version."""
    changed = ResolvedModel(resolved.name, "8", "models:/main.ml.model/8", None, resolved.digest)
    with pytest.raises(ValueError, match="resolved"):
        build_pinned_endpoint(spec, resolved=changed, artifact=artifact, package_info=package)
    changed = ResolvedModel(resolved.name, resolved.version, resolved.model_uri, None, "0" * 64)
    with pytest.raises(ValueError, match="digest"):
        build_pinned_endpoint(spec, resolved=changed, artifact=artifact, package_info=package)


def test_existing_endpoint_is_never_overwritten(spec, resolved, artifact, package):
    """Creation must fail when a name is already in use, without an update call."""
    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    client = SimpleNamespace(serving_endpoints=Mock())
    client.serving_endpoints.get.return_value = object()
    with pytest.raises(ValueError, match="exists"):
        create_pinned_endpoint(client, plan)
    client.serving_endpoints.create.assert_not_called()


def test_readiness_requires_both_states_and_matching_config(spec, resolved, artifact, package):
    """A ready flag alone cannot admit a pending or changed served version."""
    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    endpoint = SimpleNamespace(
        name=spec.endpoint_name,
        state=SimpleNamespace(ready="READY", config_update="NOT_UPDATING"),
        config=SimpleNamespace(
            served_entities=[SimpleNamespace(**plan.config["config"]["served_entities"][0])]
        ),
        telemetry_config=SimpleNamespace(
            table_names=SimpleNamespace(logs_table=spec.telemetry_logs_table),
            inference_table_config=SimpleNamespace(name=spec.inference_table, sampling_fraction=1),
            enabled_telemetry_features=["TELEMETRY_FEATURE_INFERENCE_TABLE"],
        ),
    )
    assert endpoint_ready(endpoint, plan)
    endpoint.state.config_update = "IN_PROGRESS"
    endpoint.config.served_entities = []
    assert not endpoint_ready(endpoint, plan)
    endpoint.config.served_entities = [
        SimpleNamespace(**plan.config["config"]["served_entities"][0])
    ]
    endpoint.state.config_update = "UPDATE_FAILED"
    with pytest.raises(ValueError, match="failed"):
        endpoint_ready(endpoint, plan)
    endpoint.state.config_update = "NOT_UPDATING"
    endpoint.telemetry_config.table_names.logs_table = "other"
    with pytest.raises(ValueError, match="config"):
        endpoint_ready(endpoint, plan)
    endpoint.telemetry_config.table_names.logs_table = spec.telemetry_logs_table
    endpoint.telemetry_config.inference_table_config.name = "other"
    with pytest.raises(ValueError, match="config"):
        endpoint_ready(endpoint, plan)
    endpoint.telemetry_config.inference_table_config.name = spec.inference_table
    endpoint.telemetry_config.inference_table_config.sampling_fraction = 0.5
    with pytest.raises(ValueError, match="config"):
        endpoint_ready(endpoint, plan)
    endpoint.telemetry_config.inference_table_config.sampling_fraction = 1
    endpoint.telemetry_config.enabled_telemetry_features = []
    with pytest.raises(ValueError, match="config"):
        endpoint_ready(endpoint, plan)
    endpoint.telemetry_config.enabled_telemetry_features = ["TELEMETRY_FEATURE_INFERENCE_TABLE"]
    endpoint.config.served_entities[0].entity_version = "8"
    with pytest.raises(ValueError, match="config"):
        endpoint_ready(endpoint, plan)


def test_named_records_preserve_null_and_numeric_types(spec, resolved, artifact, package):
    """The SDK transport sends exact rows and returns typed predictions on old SDKs."""
    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    client = SimpleNamespace(
        serving_endpoints=Mock(), serving_endpoints_data_plane=Mock(), api_client=Mock()
    )
    client.api_client.do.side_effect = [
        _ready_telemetry_response(plan),
        {"predictions": [{"prediction": 1}, {"prediction": 2}]},
        _ready_telemetry_response(plan),
        {"predictions": [{"prediction": 1}]},
    ]
    records = [{"x": None}, {"x": 7.25}]
    response = query_named_records(client, plan, records)
    assert response.predictions == [{"prediction": 1}, {"prediction": 2}]
    assert client.api_client.do.call_args_list[1].kwargs["path"] == (
        "/serving-endpoints/sm23b-test/invocations"
    )
    assert client.api_client.do.call_args_list[1].kwargs["body"] == {"dataframe_records": records}
    assert client.api_client.do.call_args_list[1].kwargs["headers"] == {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    client.serving_endpoints_data_plane.query.assert_not_called()
    with pytest.raises(ValueError, match="columns"):
        query_named_records(client, plan, [{"wrong": 2}])
    with pytest.raises(ValueError, match="finite"):
        query_named_records(client, plan, [{"x": float("nan")}])
    with pytest.raises(ValueError, match="schema"):
        query_named_records(client, plan, [{"x": "7.25"}])
    with pytest.raises(ValueError, match="schema"):
        query_named_records(client, plan, [{"x": True}])
    query_named_records(client, plan, [{"x": 1.5}], client_request_id="capture-1")
    assert client.api_client.do.call_args.kwargs["body"] == {
        "dataframe_records": [{"x": 1.5}],
        "client_request_id": "capture-1",
    }


def test_preparation_uses_exact_registry_version(spec, resolved, artifact, package, monkeypatch):
    """A registry read must supply all evidence from the same immutable version."""
    from skyulf.integrations.databricks.serving import endpoints

    resolve = Mock(return_value=resolved)
    download = Mock(return_value="package")
    loader = Mock(return_value=artifact)
    monkeypatch.setattr(endpoints, "resolve_model", resolve)
    monkeypatch.setattr(endpoints, "download_registered_package", download)
    monkeypatch.setattr(endpoints, "packaged_artifact_path", lambda *args: "payload")
    monkeypatch.setattr(endpoints, "load_pipeline", loader)
    monkeypatch.setattr(endpoints, "Path", lambda value: value)
    monkeypatch.setattr(endpoints, "build_pinned_endpoint", Mock(return_value="plan"))
    import mlflow

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", Mock(return_value=object()))
    monkeypatch.setattr(mlflow.models.Model, "load", Mock(return_value=package))
    package.flavors = {"python_function": {"artifacts": {"local_pipeline": {"path": "payload"}}}}
    assert prepare_pinned_endpoint(spec, tracking_uri="tracking", registry_uri="registry") == "plan"
    resolve.assert_called_once_with(
        spec.model_name, version="7", tracking_uri="tracking", registry_uri="registry"
    )
    assert download.call_args.args[2:4] == (spec.model_name, "7")
    loader.assert_called_once_with("payload")
    with pytest.raises(ValueError, match="explicit"):
        prepare_pinned_endpoint(spec)


def test_create_uses_sdk_config_without_replacing_existing_endpoint(
    spec, resolved, artifact, package
):
    """Telemetry creation preserves exact fields through the raw SDK transport."""
    from databricks.sdk.errors import NotFound

    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    client = SimpleNamespace(serving_endpoints=Mock(), api_client=Mock())
    client.serving_endpoints.get.side_effect = NotFound("absent")
    client.api_client.do.side_effect = [{}, _ready_telemetry_response(plan)]
    create_pinned_endpoint(client, plan)
    client.api_client.do.assert_any_call(
        method="POST", path="/api/2.0/serving-endpoints", body=plan.config
    )
    assert client.api_client.do.call_count == 2
    client.serving_endpoints.create.assert_not_called()


def test_explicit_gateway_mode_keeps_legacy_contract(spec, resolved, artifact, package):
    """Older workspaces can deliberately select Gateway without an automatic retry."""
    from databricks.sdk.errors import NotFound

    legacy = replace(spec, logging_mode="ai_gateway")
    plan = build_pinned_endpoint(legacy, resolved=resolved, artifact=artifact, package_info=package)
    assert "telemetry_config" not in plan.config
    assert plan.config["ai_gateway"]["usage_tracking_config"] == {"enabled": True}
    client = SimpleNamespace(serving_endpoints=Mock())
    ready = deepcopy(plan.config) | {"state": {"ready": "READY", "config_update": "NOT_UPDATING"}}
    client.serving_endpoints.get.side_effect = [NotFound("absent"), ready]
    create_pinned_endpoint(client, plan)
    assert client.serving_endpoints.create.call_args.kwargs[
        "ai_gateway"
    ].inference_table_config.enabled
    assert "telemetry_config" not in client.serving_endpoints.create.call_args.kwargs


def test_telemetry_create_failure_does_not_retry_gateway(spec, resolved, artifact, package):
    """A rejected telemetry create cannot mutate resources through a second mode."""
    from databricks.sdk.errors import NotFound

    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    client = SimpleNamespace(serving_endpoints=Mock(), api_client=Mock())
    client.serving_endpoints.get.side_effect = NotFound("absent")
    client.api_client.do.side_effect = ValueError("workspace rejected telemetry")
    with pytest.raises(ValueError, match="rejected telemetry"):
        create_pinned_endpoint(client, plan)
    client.api_client.do.assert_called_once()
    client.serving_endpoints.create.assert_not_called()


def _ready_telemetry_response(plan):
    """Model the complete raw endpoint response independently of SDK dataclasses."""
    return {
        "name": plan.spec.endpoint_name,
        "state": {"ready": "READY", "config_update": "NOT_UPDATING"},
        "config": {"served_entities": plan.config["config"]["served_entities"]},
        "telemetry_config": {
            "table_names": {"logs_table": plan.spec.telemetry_logs_table},
            "inference_table_config": {
                "name": plan.spec.inference_table,
                "sampling_fraction": 1,
            },
            "enabled_telemetry_features": ["TELEMETRY_FEATURE_INFERENCE_TABLE"],
        },
    }


def test_raw_readback_survives_sdk_telemetry_field_loss(spec, resolved, artifact, package):
    """Readiness must attest raw telemetry even when old SDK models omit its fields."""
    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    client = SimpleNamespace(serving_endpoints=Mock(), api_client=Mock())
    client.serving_endpoints.get.return_value = SimpleNamespace(name=spec.endpoint_name)
    client.api_client.do.return_value = _ready_telemetry_response(plan)
    assert require_pinned_endpoint_ready(client, plan) == client.api_client.do.return_value
    client.api_client.do.assert_called_once_with(
        method="GET", path=f"/api/2.0/serving-endpoints/{spec.endpoint_name}"
    )
    client.serving_endpoints.get.assert_not_called()
    client.api_client.do.return_value["telemetry_config"]["table_names"]["logs_table"] = "wrong"
    with pytest.raises(ValueError, match="config"):
        require_pinned_endpoint_ready(client, plan)


def test_shared_string_aliases_and_output_boolean_signature():
    """Saved string aliases and output booleans must keep their MLflow scalar types."""
    from mlflow.types import DataType

    from skyulf.integrations.databricks.serving import endpoints

    signature = SimpleNamespace(
        inputs=SimpleNamespace(inputs=[SimpleNamespace(name="text", type=DataType.string)]),
        outputs=SimpleNamespace(
            inputs=[
                SimpleNamespace(name="result", type=DataType.string),
                SimpleNamespace(name="flag", type=DataType.boolean),
            ]
        ),
    )
    outputs = (
        SimpleNamespace(name="result", dtype="str"),
        SimpleNamespace(name="flag", dtype="boolean"),
    )
    endpoints._validate_signature(signature, [("text", "utf8")], outputs)


def test_float32_request_rejects_finite_json_overflow(spec, resolved, artifact, package):
    """A JSON finite number cannot silently overflow the model's float32 input."""
    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    plan = replace(plan, input_schema=(("x", "float32"),))
    client = SimpleNamespace(serving_endpoints=Mock(), serving_endpoints_data_plane=Mock())
    with pytest.raises(ValueError, match="schema"):
        query_named_records(client, plan, [{"x": 1e100}])
    with pytest.raises(ValueError, match="schema"):
        query_named_records(client, plan, [{"x": 10**1000}])
    client.serving_endpoints.get.assert_not_called()


@pytest.mark.parametrize("field,value", [("entity_version", "8"), ("entity_name", "main.ml.other")])
def test_sql_deployment_rejects_mutated_config_identity(
    spec, resolved, artifact, package, field, value
):
    """A mutable request dictionary cannot redefine the model selected by the frozen spec."""
    from skyulf.integrations.databricks.serving import (
        build_serving_sql_function,
        create_serving_sql_function,
    )

    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=package)
    function = build_serving_sql_function(plan, "main.api.score")
    plan.config["config"]["served_entities"][0][field] = value
    client = SimpleNamespace(api_client=Mock(return_value=None))
    client.api_client.do.return_value = _ready_telemetry_response(plan)
    spark = SimpleNamespace(sql=Mock())
    with pytest.raises(ValueError, match="selector"):
        create_serving_sql_function(spark, client, function)
    spark.sql.assert_not_called()


def test_model_set_sql_preserves_record_keys_and_component_outcomes(tmp_path, artifact, spec):
    """A served model set must retain its keyed multi-output schema in the SQL contract."""
    from skyulf.inference.bundle import ColumnSpec
    from skyulf.inference.model_set import ComponentReference, save_model_set
    from skyulf.integrations.databricks.serving import build_serving_sql_function
    from skyulf.integrations.mlflow.models.model_set import _signature as set_signature

    components = {
        branch: (
            ComponentReference(
                name=f"main.ml.{branch}", version="1", digest=artifact.manifest.pipeline_sha256
            ),
            tmp_path / "model",
        )
        for branch in ("left", "right")
    }
    model_set = save_model_set(
        tmp_path / "set",
        components,
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    certificate = spark_model.partition_safety_certificate(model_set)
    inputs, _ = spark_model._contract(model_set)
    package = SimpleNamespace(
        metadata={
            spark_model.SAFETY_KEY: certificate,
            spark_model.SOURCE_KEY: spark_model.runtime_source_digest(),
            TRANSPORT_KEY: transport_spec(inputs),
            "skyulf_artifact_kind": "model_set",
            "skyulf_execution_scope": "whole_frame_local",
            "model_set_digest": certificate["model_set_sha256"],
        },
        signature=set_signature(model_set, spark_certified=True),
    )
    resolved = ResolvedModel(
        spec.model_name, spec.model_version, spec.model_uri, None, model_set.manifest.set_sha256
    )
    endpoint = build_pinned_endpoint(
        spec, resolved=resolved, artifact=model_set, package_info=package
    )
    function = build_serving_sql_function(endpoint, "main.api.score_set")
    assert function.response_type == (
        "STRUCT<`id`: BIGINT, `left__prediction`: DOUBLE, `left__scoring_status`: STRING, "
        "`left__exclusion_reason`: STRING, `right__prediction`: DOUBLE, "
        "`right__scoring_status`: STRING, `right__exclusion_reason`: STRING>"
    )


def test_disabled_logging_creates_an_endpoint_without_inference_tables(resolved, artifact, package):
    """Callers writing their own ai_query tables can serve without endpoint payload logging."""
    from databricks.sdk.errors import NotFound

    quiet = PinnedEndpointSpec(
        "sm23b-quiet", "main.ml.model", "7", "", "", "", logging_mode="disabled"
    )
    plan = build_pinned_endpoint(quiet, resolved=resolved, artifact=artifact, package_info=package)
    assert set(plan.config) == {"name", "config"}
    with pytest.raises(ValueError, match="logging is disabled"):
        _ = quiet.inference_table
    client = SimpleNamespace(serving_endpoints=Mock())
    ready = deepcopy(plan.config) | {"state": {"ready": "READY", "config_update": "NOT_UPDATING"}}
    client.serving_endpoints.get.side_effect = [NotFound("absent"), ready]
    create_pinned_endpoint(client, plan)
    kwargs = client.serving_endpoints.create.call_args.kwargs
    assert "ai_gateway" not in kwargs and "telemetry_config" not in kwargs
    ready = {
        "name": "sm23b-quiet",
        "state": {"ready": "READY", "config_update": "NOT_UPDATING"},
        "config": {"served_entities": plan.config["config"]["served_entities"]},
    }
    assert endpoint_ready(ready, plan)
    logged = {**ready, "ai_gateway": {"inference_table_config": {"enabled": True}}}
    with pytest.raises(ValueError, match="differs"):
        endpoint_ready(logged, plan)


def test_logged_modes_still_require_logging_destinations():
    """Only the explicit disabled mode may omit the payload table destination."""
    with pytest.raises(ValueError, match="logging_catalog"):
        PinnedEndpointSpec("sm23b-test", "main.ml.model", "7", "", "observability", "sm23b")
