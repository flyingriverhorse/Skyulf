"""Optional Databricks serving controls with inspected model-package admission."""

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ....inference.fitted_pipeline import FittedPipelineArtifact, load_pipeline
from ....inference.model_set import ModelSetArtifact, load_model_set
from ....inference.model_set_scoring import model_set_output_schema
from ....inference.pipeline_scoring import scoring_output_schema
from ...mlflow.registration.registry import (
    ResolvedModel,
    download_registered_package,
    packaged_artifact_path,
    resolve_model,
)
from ...mlflow.shared._model_metadata import mlflow_dtype, normalized_dtype
from ...mlflow.spark.spark_model import (
    SAFETY_KEY,
    SOURCE_KEY,
    partition_safety_certificate,
    runtime_source_digest,
)
from .contracts import PinnedEndpointPlan, PinnedEndpointSpec


def _artifact_schema(
    artifact: FittedPipelineArtifact | ModelSetArtifact,
) -> tuple[list[tuple[str, str]], tuple[Any, ...]]:
    """Read exact saved input and output fields from a fitted artifact."""
    if isinstance(artifact, FittedPipelineArtifact):
        return (
            list(zip(artifact.manifest.input_columns, artifact.manifest.input_dtypes, strict=True)),
            tuple(scoring_output_schema(artifact)),
        )
    if isinstance(artifact, ModelSetArtifact):
        return (
            [(column.name, column.dtype) for column in artifact.manifest.input_schema],
            tuple(model_set_output_schema(artifact)),
        )
    raise TypeError("Serving requires a loaded local pipeline or model-set artifact.")


def _validate_identity(
    spec: PinnedEndpointSpec,
    resolved: ResolvedModel,
    artifact: FittedPipelineArtifact | ModelSetArtifact,
    metadata: Mapping[str, Any],
    certificate: dict[str, Any],
) -> None:
    """Bind the saved artifact, registry read and package metadata to one digest."""
    if not isinstance(resolved, ResolvedModel) or (
        resolved.name,
        resolved.version,
        resolved.model_uri,
    ) != (spec.model_name, spec.model_version, spec.model_uri):
        raise ValueError("resolved registry model differs from the endpoint selector.")
    local = isinstance(artifact, FittedPipelineArtifact)
    digest_key = "pipeline_sha256" if local else "model_set_sha256"
    metadata_key = "local_pipeline_digest" if local else "model_set_digest"
    digest = certificate[digest_key]
    if resolved.digest != digest:
        raise ValueError("resolved registry digest differs from the inspected artifact.")
    expected = {
        "skyulf_artifact_kind": "local_pipeline" if local else "model_set",
        "skyulf_execution_scope": "whole_frame_local",
        metadata_key: digest,
        SAFETY_KEY: certificate,
        SOURCE_KEY: runtime_source_digest(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Pinned MLflow package identity, certificate or runtime source differs.")


def _validate_signature(signature: Any, inputs: list, outputs: tuple) -> None:
    """Require exact named scalar inputs and all declared prediction fields."""
    if signature is None or signature.inputs is None or signature.outputs is None:
        raise ValueError("Pinned MLflow package requires an explicit complete signature.")
    expected_inputs = [(name, _mlflow_type(dtype, input_value=True)) for name, dtype in inputs]
    expected_outputs = [
        (column.name, _mlflow_type(column.dtype, input_value=False)) for column in outputs
    ]
    actual_inputs = [(column.name, column.type) for column in signature.inputs.inputs]
    actual_outputs = [(column.name, column.type) for column in signature.outputs.inputs]
    if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
        raise ValueError("Pinned MLflow package signature differs from its artifact schema.")


def _mlflow_type(dtype: str, *, input_value: bool) -> Any:
    """Resolve one artifact dtype to its exact saved MLflow signature type."""
    normalized = (
        "string"
        if input_value and dtype in {"Int32", "Int64", "boolean", "Boolean"}
        else normalized_dtype(dtype)
    )
    return mlflow_dtype(normalized)


def _validate_transport(metadata: Mapping[str, Any], inputs: list) -> None:
    """Match the nullable primitive codec declared by the package."""
    encoded = {
        name: dtype for name, dtype in inputs if dtype in {"Int32", "Int64", "boolean", "Boolean"}
    }
    expected = {"codec": "nullable_primitives_v1", "columns": encoded} if encoded else None
    if metadata.get("skyulf_input_transport") != expected:
        raise ValueError("Pinned MLflow package nullable transport differs from artifact.")


def build_pinned_endpoint(
    spec: PinnedEndpointSpec,
    *,
    resolved: ResolvedModel,
    artifact: FittedPipelineArtifact | ModelSetArtifact,
    package_info: Any,
) -> PinnedEndpointPlan:
    """Build a credential-free config from one inspected concrete registry package.

    ``resolved``, ``artifact`` and ``package_info`` must be read from the same
    pinned registry version. Use ``resolve_model`` and the registered-package
    loader for that version; never supply package metadata from an alias.
    """
    certificate = partition_safety_certificate(artifact)
    inputs, outputs = _artifact_schema(artifact)
    metadata = getattr(package_info, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("Pinned MLflow package requires model metadata.")
    _validate_identity(spec, resolved, artifact, metadata, certificate)
    _validate_transport(metadata, inputs)
    _validate_signature(getattr(package_info, "signature", None), inputs, outputs)
    config = {
        "name": spec.endpoint_name,
        "config": {
            "served_entities": [
                {
                    "entity_name": spec.model_name,
                    "entity_version": spec.model_version,
                    "workload_type": "CPU",
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                }
            ],
        },
    }
    if spec.logging_mode == "telemetry":
        config["telemetry_config"] = {
            # Native CREATE requires all sinks; inference-only GET retains just logs.
            "table_names": {
                "logs_table": spec.telemetry_logs_table,
                "traces_table": spec.telemetry_traces_table,
                "metrics_table": spec.telemetry_metrics_table,
            },
            "inference_table_config": {"sampling_fraction": 1.0},
            "enabled_telemetry_features": ["TELEMETRY_FEATURE_INFERENCE_TABLE"],
        }
    else:
        config["ai_gateway"] = {
            "inference_table_config": {
                "enabled": True,
                "catalog_name": spec.logging_catalog,
                "schema_name": spec.logging_schema,
                "table_name_prefix": spec.logging_table_prefix,
            },
            "usage_tracking_config": {"enabled": True},
        }
    return PinnedEndpointPlan(
        spec,
        config,
        tuple(name for name, _ in inputs),
        tuple(inputs),
        tuple((column.name, column.dtype) for column in outputs),
    )


def prepare_pinned_endpoint(
    spec: PinnedEndpointSpec,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> PinnedEndpointPlan:
    """Read one concrete registry snapshot and inspect its packaged fitted artifact.

    Registry and tracking URI selection is explicit. This function performs
    read-only MLflow operations and never changes process-global stores.
    """
    import mlflow  # noqa: PLC0415  # ty: ignore[unresolved-import]

    if not tracking_uri or not registry_uri:
        raise ValueError("Serving preparation requires explicit tracking_uri and registry_uri.")
    resolved = resolve_model(
        spec.model_name,
        version=spec.model_version,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    package_path = Path(
        download_registered_package(
            mlflow, client, spec.model_name, spec.model_version, tracking_uri
        )
    )
    package_info = mlflow.models.Model.load(package_path)
    metadata = package_info.metadata or {}
    kind = metadata.get("skyulf_artifact_kind")
    if kind == "local_pipeline":
        artifact_path = packaged_artifact_path(package_path, package_info.flavors, kind)
        artifact = load_pipeline(artifact_path)
    elif kind == "model_set":
        artifact_path = packaged_artifact_path(package_path, package_info.flavors, kind)
        artifact = load_model_set(artifact_path)
    else:
        raise ValueError("Pinned MLflow package is not a supported Skyulf fitted artifact.")
    return build_pinned_endpoint(
        spec, resolved=resolved, artifact=artifact, package_info=package_info
    )


def _field(value: Any, name: str) -> Any:
    """Read SDK dataclasses and raw API dictionaries through one contract."""
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _value(value: Any) -> Any:
    """Compare SDK enum values with raw API strings."""
    return getattr(value, "value", value)


def _require_config(endpoint: Any, plan: PinnedEndpointPlan) -> None:
    """Fail when remote model, workload or logging differs from the pinned plan."""
    expected = plan.config
    entities = _field(_field(endpoint, "config"), "served_entities") or []
    if _field(endpoint, "name") != plan.spec.endpoint_name or len(entities) != 1:
        raise ValueError("Serving endpoint config differs from the pinned plan.")
    identity = (_field(entities[0], "entity_name"), _field(entities[0], "entity_version"))
    if identity != (plan.spec.model_name, plan.spec.model_version):
        raise ValueError("Serving endpoint config differs from the pinned model selector.")
    for key, wanted in expected["config"]["served_entities"][0].items():
        if _value(_field(entities[0], key)) != wanted:
            raise ValueError("Serving endpoint config differs from the pinned plan.")
    if plan.spec.logging_mode == "telemetry":
        _require_telemetry(endpoint, plan.spec)
    else:
        _require_gateway(endpoint, expected["ai_gateway"])


def _require_telemetry(endpoint: Any, spec: PinnedEndpointSpec) -> None:
    """Verify the native log table, payload view, sampling and feature readback."""
    telemetry = _field(endpoint, "telemetry_config")
    names = _field(telemetry, "table_names")
    inference = _field(telemetry, "inference_table_config")
    features = _field(telemetry, "enabled_telemetry_features") or []
    if (
        _field(names, "logs_table") != spec.telemetry_logs_table
        or _field(inference, "name") != spec.inference_table
        or _field(inference, "sampling_fraction") != 1.0
        or [_value(feature) for feature in features] != ["TELEMETRY_FEATURE_INFERENCE_TABLE"]
    ):
        raise ValueError("Serving endpoint config differs from the pinned plan.")


def _require_gateway(endpoint: Any, expected: dict[str, Any]) -> None:
    """Verify an explicitly selected legacy AI Gateway logging configuration."""
    gateway = _field(endpoint, "ai_gateway")
    logging = _field(gateway, "inference_table_config")
    usage = _field(gateway, "usage_tracking_config")
    for key, wanted in expected["inference_table_config"].items():
        if _field(logging, key) != wanted:
            raise ValueError("Serving endpoint config differs from the pinned plan.")
    if _field(usage, "enabled") is not True:
        raise ValueError("Serving endpoint config differs from the pinned plan.")


def endpoint_ready(endpoint: Any, plan: PinnedEndpointPlan) -> bool:
    """Require exact remote config and both settled Databricks state fields."""
    state = _field(endpoint, "state")
    update = _value(_field(state, "config_update"))
    if update in {"UPDATE_FAILED", "UPDATE_CANCELED"}:
        raise ValueError(f"Serving endpoint config update failed: {update}.")
    if _value(_field(state, "ready")) != "READY" or update != "NOT_UPDATING":
        return False
    _require_config(endpoint, plan)
    return True


def require_pinned_endpoint_ready(client: Any, plan: PinnedEndpointPlan) -> Any:
    """Read and admit only an exact, fully settled endpoint through an injected SDK client."""
    if plan.spec.logging_mode == "telemetry":
        endpoint = _telemetry_request(
            client, "GET", f"/api/2.0/serving-endpoints/{plan.spec.endpoint_name}"
        )
    else:
        endpoint = client.serving_endpoints.get(plan.spec.endpoint_name)
    if not endpoint_ready(endpoint, plan):
        raise RuntimeError("Serving endpoint is not ready or its config update is pending.")
    return endpoint


def _telemetry_request(client: Any, method: str, path: str, body: dict | None = None) -> Any:
    """Use the injected SDK transport without losing newer response fields."""
    transport = api_transport(client)
    if body is None:
        return transport(method=method, path=path)
    return transport(method=method, path=path, body=body)


def api_transport(client: Any) -> Any:
    """Require the authenticated transport supplied by the caller's SDK client."""
    transport = getattr(getattr(client, "api_client", None), "do", None)
    if not callable(transport):
        raise RuntimeError("Native telemetry requires client.api_client.do transport.")
    return transport


def create_pinned_endpoint(client: Any, plan: PinnedEndpointPlan) -> Any:
    """Create a new endpoint only; an existing name is never overwritten."""
    from databricks.sdk.errors import NotFound  # noqa: PLC0415
    from databricks.sdk.service import serving  # noqa: PLC0415

    try:
        client.serving_endpoints.get(plan.spec.endpoint_name)
    except NotFound:
        pass
    else:
        raise ValueError("Serving endpoint already exists; refusing to overwrite it.")
    if plan.spec.logging_mode == "telemetry":
        return _telemetry_request(client, "POST", "/api/2.0/serving-endpoints", plan.config)
    kwargs = {
        "name": plan.spec.endpoint_name,
        "config": serving.EndpointCoreConfigInput.from_dict(plan.config["config"]),
    }
    kwargs["ai_gateway"] = serving.AiGatewayConfig.from_dict(plan.config["ai_gateway"])
    return client.serving_endpoints.create(**kwargs)


def query_named_records(
    client: Any,
    plan: PinnedEndpointPlan,
    records: Sequence[Mapping[str, Any]],
    *,
    client_request_id: str | None = None,
) -> Any:
    """Send finite JSON named rows unchanged after exact schema and state checks."""
    from databricks.sdk.service import serving  # noqa: PLC0415

    rows = validate_named_rows(records, plan)
    if client_request_id is not None and (
        not isinstance(client_request_id, str) or not client_request_id.strip()
    ):
        raise ValueError("client_request_id must be a nonempty string.")
    require_pinned_endpoint_ready(client, plan)
    body = {"dataframe_records": rows}
    if client_request_id is not None:
        body["client_request_id"] = client_request_id
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    workspace_id = _field(getattr(client, "config", None), "workspace_id")
    if workspace_id:
        headers["X-Databricks-Workspace-Id"] = workspace_id
    response = api_transport(client)(
        method="POST",
        path=f"/serving-endpoints/{plan.spec.endpoint_name}/invocations",
        body=body,
        headers=headers,
        response_headers=["served-model-name"],
    )
    return serving.QueryEndpointResponse.from_dict(response)


def validate_named_rows(
    records: Sequence[Mapping[str, Any]], plan: PinnedEndpointPlan
) -> list[dict[str, Any]]:
    """Copy named rows after JSON and artifact-schema validation."""
    if not records or any(
        not isinstance(row, Mapping) or set(row) != set(plan.input_columns) for row in records
    ):
        raise ValueError("Serving dataframe_records columns differ from artifact inputs.")
    rows = [dict(row) for row in records]
    validate_json_rows(rows)
    validate_schema_rows(rows, plan.input_schema)
    return rows


def validate_json_rows(rows: list[dict[str, Any]]) -> None:
    """Require finite JSON scalars before the SDK sees an inference request."""
    try:
        json.dumps(rows, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Serving dataframe_records require finite JSON scalar values.") from exc
    if any(any(isinstance(value, (dict, list)) for value in row.values()) for row in rows):
        raise ValueError("Serving dataframe_records require scalar values.")


def validate_schema_rows(
    rows: list[dict[str, Any]], input_schema: tuple[tuple[str, str], ...]
) -> None:
    """Check every named value against its saved artifact input type."""
    for row in rows:
        for name, dtype in input_schema:
            if not _valid_scalar(row[name], dtype):
                raise ValueError(
                    f"Serving dataframe_records field {name!r} violates {dtype} schema."
                )


def _valid_scalar(value: Any, dtype: str) -> bool:
    """Validate JSON scalars against the saved MLflow named-input transport."""
    if value is None:
        return True
    if dtype in {"Int32", "Int64"}:
        return _valid_integer(value, 32 if dtype == "Int32" else 64, encoded=True)
    if dtype in {"boolean", "Boolean"}:
        return value in {"true", "false"} if isinstance(value, str) else False
    return _valid_plain_scalar(value, dtype.lower())


def _valid_plain_scalar(value: Any, normalized: str) -> bool:
    """Validate ordinary JSON primitives against exact fitted scalar types."""
    dtype = normalized_dtype(normalized)
    if dtype == "bool":
        return isinstance(value, bool)
    if dtype in {"int32", "int64"}:
        return _valid_integer(value, 32 if dtype == "int32" else 64, encoded=False)
    if dtype in {"float32", "float64"}:
        return _valid_float(value, dtype)
    if dtype == "string":
        return isinstance(value, str)
    return False


def _valid_float(value: Any, dtype: str) -> bool:
    """Reject nonfinite and out-of-range JSON numbers before MLflow coercion."""
    if type(value) not in {int, float}:
        return False
    try:
        number = float(value)
    except OverflowError:
        return False
    limit = 3.4028234663852886e38 if dtype == "float32" else float("inf")
    return math.isfinite(number) and abs(number) <= limit


def _valid_integer(value: Any, bits: int, *, encoded: bool) -> bool:
    """Reject ambiguous integers and enforce the fitted signed width."""
    if encoded:
        if not isinstance(value, str) or not re.fullmatch(r"(?:0|-?[1-9][0-9]*)", value):
            return False
        number = int(value)
    else:
        if type(value) is not int:
            return False
        number = value
    return -(2 ** (bits - 1)) <= number < 2 ** (bits - 1)
