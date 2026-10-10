"""Admission of native latest-feature envelopes using their actual request schema."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from ....inference.fitted_pipeline import load_pipeline
from ....inference.model_set import load_model_set
from ...mlflow.models._feature_package_spec import SPARK_MLFLOW_TYPES, feature_spec_entries
from ...mlflow.registration.registry import (
    download_registered_package,
    packaged_artifact_path,
    resolve_model,
)
from ..feature_store.online_policy import ONLINE_FEATURES_KEY, OnlineFeaturePolicy
from ..shared.json_contracts import finite_json_digest
from .contracts import PinnedEndpointPlan, PinnedEndpointSpec
from .endpoints import build_pinned_endpoint

_REQUEST_TYPES = {
    "integer": "int32",
    "long": "int64",
    "float": "float32",
    "double": "float64",
    "boolean": "bool",
    "string": "string",
}


def online_request_schema(path: Path, signature: Any) -> tuple[tuple[str, str], ...]:
    """Validate the entire native signature and return required source inputs.

    Use only after feature_package_models validates this saved FeatureSpec and
    its digest. Native FE includes excluded source fields in its signature.
    Temporal/binary source inputs are rejected until their REST transport is
    supported; this helper never promises temporal entity-only lookup.
    """
    import yaml  # noqa: PLC0415 - optional MLflow dependency

    if signature is None or signature.inputs is None:
        raise ValueError("Online feature model requires the native input signature.")
    saved = yaml.safe_load(path.read_bytes())
    entries = feature_spec_entries(saved.get("input_columns"))
    _validate_source_inputs(entries)
    expected = [
        (name, SPARK_MLFLOW_TYPES.get(info.get("data_type")), info.get("source") == "training_data")
        for name, info in entries
    ]
    actual = [(column.name, column.type.name, column.required) for column in signature.inputs]
    if actual != expected:
        raise ValueError("Online native signature differs from saved FeatureSpec source inputs.")
    return _required_schema(expected)


def _required_schema(columns: list[tuple[str, Any, bool]]) -> tuple[tuple[str, str], ...]:
    """Retain every SDK-required input and reject unsupported scalar transport."""
    result = []
    for name, dtype, required in columns:
        if not required:
            continue
        if dtype not in _REQUEST_TYPES:
            raise ValueError(
                "Online endpoint requires non-temporal scalar source inputs; retrain with a narrow non-temporal source or use an explicitly verified native REST contract."
            )
        result.append((name, _REQUEST_TYPES[dtype]))
    if not result:
        raise ValueError("Online native lookup requires explicit source input keys.")
    return tuple(result)


def prepare_online_endpoint(
    spec: PinnedEndpointSpec,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> PinnedEndpointPlan:
    """Inspect one pinned native package and admit its verified source request schema.

    Reuses fitted identity, raw signature, runtime certificate and logging
    admission. The returned plan supports query_named_records and excludes
    fetched feature overrides. Direct native REST calls still permit overrides;
    the saved freshness guard checks values, not their online-store provenance.
    Current pinned certification admits pandas fitted artifacts only. Raw Polars
    packages can retain this policy but are not certified for endpoint admission.
    Registry reads only: this function never creates or changes an endpoint.
    """
    import mlflow  # noqa: PLC0415  # ty: ignore[unresolved-import]

    from ...mlflow.models.feature_model import (  # noqa: PLC0415
        FEATURE_STORE_KEY,
        feature_package_models,
    )

    if not tracking_uri or not registry_uri:
        raise ValueError("Serving preparation requires explicit tracking_uri and registry_uri.")
    resolved = resolve_model(
        spec.model_name,
        version=spec.model_version,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    path = Path(
        download_registered_package(
            mlflow, client, spec.model_name, spec.model_version, tracking_uri
        )
    )
    outer, raw, raw_path = feature_package_models(path)
    value = (raw.metadata or {}).get(ONLINE_FEATURES_KEY)
    if value is None:
        raise ValueError("Online serving requires a saved latest OnlineFeaturePolicy.")
    policy = OnlineFeaturePolicy.from_dict(value)
    kind = raw.metadata["skyulf_artifact_kind"]
    artifact_path = packaged_artifact_path(raw_path, raw.flavors, kind)
    artifact = (
        load_pipeline(artifact_path) if kind == "local_pipeline" else load_model_set(artifact_path)
    )
    plan = build_pinned_endpoint(spec, resolved=resolved, artifact=artifact, package_info=raw)
    mlflow.pyfunc.load_model(str(raw_path))
    schema = online_request_schema(raw_path.parent / "feature_spec.yaml", outer.signature)
    return replace(
        plan,
        input_columns=tuple(name for name, _ in schema),
        input_schema=schema,
        online_contract=finite_json_digest(
            {
                "policy": policy.to_dict(),
                "lookup_spec": outer.metadata[FEATURE_STORE_KEY]["lookup_spec"],
            }
        ),
    )


def _validate_source_inputs(entries: list[tuple[str, dict[str, Any]]]) -> None:
    """Require all referenced composite and temporal keys in the native source schema."""
    sources = {name for name, info in entries if info.get("source") == "training_data"}
    for _, info in entries:
        if info.get("source") == "feature_store":
            keys = set(info.get("lookup_key", [])) | set(info.get("timestamp_lookup_key", []))
            if not keys.issubset(sources):
                raise ValueError("Online native source schema is missing declared lookup keys.")
