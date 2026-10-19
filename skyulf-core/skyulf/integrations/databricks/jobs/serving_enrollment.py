"""Enroll verified endpoint versions without replacing unrelated monitoring inventory."""

import json
from dataclasses import replace
from typing import Any

from ...mlflow.registration.registry import resolve_model
from ..observability.monitoring import monitoring_store
from ..observability.monitoring.monitoring_config import (
    MonitorConfig,
    parse_drift_thresholds,
    parse_performance_policies,
    store_namespace,
)
from ..observability.monitoring.serving import serving_registration
from ..serving.contracts import PinnedEndpointSpec
from .shared import reference_mutations


def _monitoring_enabled(values: dict[str, str]) -> bool:
    """Require an explicit target opt-in without accepting truthy or malformed flags."""
    enabled = values.get("monitoring_enabled", "false")
    if enabled not in ("true", "false"):
        raise ValueError("monitoring_enabled must be true or false.")
    return enabled == "true"


def serving_monitoring_namespace(logging_mode: str, values: dict[str, str]) -> str | None:
    """Require an explicitly configured shared store before changing a logged endpoint."""
    if not _monitoring_enabled(values) or logging_mode == "disabled":
        return None
    catalog, schema = values.get("monitoring_catalog", ""), values.get("monitoring_schema", "")
    if not catalog or not schema:
        raise ValueError(
            "Logged serving requires both monitoring_catalog and monitoring_schema; "
            "configure the shared monitoring store before running the serving job."
        )
    reference_mutations.notebook_reference_holder(values)
    return store_namespace(catalog, schema)


def build_serving_monitoring_configs(
    spec: PinnedEndpointSpec,
    settings: dict,
    namespace: str | None,
    values: dict[str, str],
    metadata: dict,
) -> list[MonitorConfig]:
    """Preflight immutable package identities and existing monitoring policies before writes."""
    if not _monitoring_enabled(values) or namespace is None:
        return []
    identities = _served_identities(spec, settings, metadata)
    policies = parse_performance_policies(values.get("monitoring_performance_policies", "{}"))
    if set(policies) - {identity["model_name"] for identity in identities}:
        raise ValueError("Performance policy model name does not match a served model component.")
    common: dict[str, Any] = {
        "environment": settings["ownership"]["target"],
        "project": settings["ownership"]["project"],
        "source_table": spec.inference_table,
        "prediction_table": spec.inference_table,
        "serving_endpoint": spec.endpoint_name,
        "execution_engine": "spark",
        "reference_namespace": namespace,
        "label_table": values.get("monitoring_label_table") or None,
        "result_available_at_column": values.get("monitoring_result_available_at_column") or None,
        "expected_interval_hours": float(values.get("monitoring_expected_interval_hours", "24")),
        "thresholds": parse_drift_thresholds(values.get("monitoring_drift_thresholds", "{}")),
    }
    configs = [
        MonitorConfig(**common, **identity, performance_policy=policies.get(identity["model_name"]))
        for identity in identities
    ]
    request = _enrollment_request(configs)
    return serving_registration.parse_serving_enrollments(
        request["monitoring_serving_enrollments"], namespace, request
    )


def _served_identities(spec: PinnedEndpointSpec, settings: dict, metadata: dict) -> list[dict]:
    """Use the loaded package kind and verified parent manifest, never component aliases."""
    kind = metadata.get("skyulf_artifact_kind")
    if kind == "local_pipeline":
        return [{"model_name": spec.model_name, "model_version": spec.model_version}]
    if kind != "model_set":
        raise ValueError("Serving monitoring requires supported fitted package metadata.")
    from ...mlflow.models.model_set import read_registered_model_set_manifest  # noqa: PLC0415

    options = {
        "tracking_uri": settings.get("tracking_uri", "databricks"),
        "registry_uri": settings.get("registry_uri", "databricks-uc"),
    }
    resolved = resolve_model(spec.model_name, version=spec.model_version, **options)
    if resolved.digest != metadata.get("model_set_digest"):
        raise ValueError("Serving monitoring model-set digest differs from the loaded package.")
    manifest = read_registered_model_set_manifest(resolved, **options)
    return [
        {
            "model_name": component.reference.name,
            "model_version": component.reference.version,
            "model_set_name": spec.model_name,
            "model_set_version": spec.model_version,
            "model_set_branch": component.branch,
        }
        for component in manifest.components
    ]


def enroll_verified_serving(
    spark: Any,
    namespace: str | None,
    configs: list[MonitorConfig],
    values: dict[str, str],
    *,
    tracking_uri: str | None = "databricks",
    registry_uri: str | None = "databricks-uc",
) -> dict:
    """Publish new monitors first, then disable superseded entries under the shared guard."""
    if not _monitoring_enabled(values):
        return {"status": "skipped", "reason": "monitoring_disabled"}
    if namespace is None:
        return {"status": "skipped", "reason": "logging_disabled"}
    request = _enrollment_request(configs)
    reference_mutations.prepare_reference_mutations(spark, namespace)
    with reference_mutations.reference_mutation_scope(
        spark, namespace, **reference_mutations.notebook_reference_holder(values)
    ):
        result = serving_registration.enroll_serving_configs(
            spark, namespace, request, tracking_uri=tracking_uri, registry_uri=registry_uri
        )
        retired = _retire_previous(spark, namespace, configs)
    return {**result, "retired_monitor_ids": retired}


def _enrollment_request(configs: list[MonitorConfig]) -> dict[str, str]:
    """Share the existing enrollment parser's project binding and bounded batch contract."""
    return {
        "monitoring_serving_enrollments": json.dumps([config.payload() for config in configs]),
        "monitoring_environment": configs[0].environment,
        "monitoring_project": configs[0].project,
    }


def _retire_previous(spark: Any, namespace: str, configs: list[MonitorConfig]) -> list[str]:
    """Keep historical rows and every other endpoint, project, environment and batch entry."""
    first = configs[0]
    owner = (first.environment, first.project, first.serving_endpoint)
    current = {config.monitor_id for config in configs}
    retired = []
    for payload in monitoring_store.load_enrolled_models(spark, namespace):
        config = MonitorConfig.from_dict(payload)
        if (
            config.enabled
            and (config.environment, config.project, config.serving_endpoint) == owner
            and config.monitor_id not in current
        ):
            monitoring_store.enroll_monitor(
                spark, namespace, replace(config, enabled=False), preserve_activation=True
            )
            retired.append(config.monitor_id)
    return retired
