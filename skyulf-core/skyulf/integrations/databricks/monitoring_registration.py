"""Register each model project's actual scored version in the shared Delta inventory."""

from typing import Any

from ._contracts import input_budget_bytes
from .monitoring_config import MonitorConfig, parse_drift_thresholds, store_namespace
from .monitoring_store import enroll_monitor
from .prediction_output import scoring_target


def monitoring_destination(values: dict[str, str]) -> str | None:
    """Require an independent monitoring destination whenever registration is configured."""
    if values.get("monitoring_deployment_mode") == "development":
        return None
    enabled = values.get("monitoring_enabled", "false")
    if enabled not in {"true", "false"}:
        raise ValueError("monitoring_enabled must be true or false.")
    catalog = values.get("monitoring_catalog", "")
    schema = values.get("monitoring_schema", "")
    if enabled == "false" and not catalog and not schema:
        return None
    if not catalog or not schema:
        raise ValueError("Set both monitoring_catalog and monitoring_schema.")
    return store_namespace(catalog, schema)


def _enrollment_config(
    workflow: dict[str, Any], values: dict[str, str], version: str
) -> MonitorConfig:
    """Use resolved producer names and the physical generation of the selected version."""
    return MonitorConfig(
        environment=values.get("monitoring_environment", ""),
        project=values.get("monitoring_project", ""),
        model_name=workflow["model_name"],
        model_version=version,
        source_table=workflow["score_source_table"],
        prediction_table=scoring_target({**workflow, "model_version": version}),
        label_table=values.get("monitoring_label_table") or None,
        result_available_at_column=values.get("monitoring_result_available_at_column") or None,
        expected_interval_hours=float(values.get("monitoring_expected_interval_hours", "24")),
        max_rows=workflow.get("max_rows", 10000),
        max_bytes=workflow.get("max_bytes", input_budget_bytes(workflow.get("max_input_mb", 64))),
        enabled=values.get("monitoring_enabled", "true") == "true",
        thresholds=parse_drift_thresholds(values.get("monitoring_drift_thresholds", "{}")),
    )


def validate_monitoring_settings(values: dict[str, str], workflow: dict[str, Any]) -> None:
    """Validate configured registration before any scoring write or recovery request."""
    if monitoring_destination(values) is None:
        return
    _enrollment_config(workflow, values, "1")


def register_scoring_monitor(
    spark: Any, workflow: dict[str, Any], values: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Upsert successful scoring, including no-ops; never enroll pending CDF recovery."""
    validate_monitoring_settings(values, workflow)
    namespace = monitoring_destination(values)
    if namespace is None or payload.get("recovery_required") is True:
        return None
    if workflow.get("training_layout") == "multi_target":
        return payload.get("monitoring")
    result = payload.get("result", {})
    if result.get("selected_model_name") != workflow["model_name"]:
        raise ValueError("Monitoring requires the scored model from this project.")
    version = result.get("selected_model_version")
    config = _enrollment_config(workflow, values, version)
    enroll_monitor(spark, namespace, config, preserve_activation=True)
    return {
        "monitor_id": config.monitor_id,
        "inventory_table": f"{namespace}.model_inventory",
        "config": config.payload(),
    }


def register_deployed_monitor(
    spark: Any,
    workflow: dict[str, Any],
    values: dict[str, str],
    payload: dict[str, Any],
    *,
    activation_started_ms: int,
) -> dict[str, Any] | None:
    """Enroll only a completed activation, including approvals and rollbacks, before scoring."""
    namespace = monitoring_destination(values)
    if namespace is None:
        return None
    result = payload.get("result", {})
    receipt = result.get("alias_change") if payload.get("action") == "train" else result
    if not receipt or receipt.get("kind") not in {"initial", "promotion", "rollback"}:
        return None
    validate_monitoring_settings(values, workflow)
    if receipt["model_name"] != workflow["model_name"]:
        raise ValueError("Activated model differs from the frozen workflow.")
    config = _enrollment_config(workflow, values, receipt["new_version"])
    enroll_monitor(spark, namespace, config, activation_started_ms=activation_started_ms)
    return {"monitor_id": config.monitor_id, "inventory_table": f"{namespace}.model_inventory"}


def publish_monitoring_request(dbutils: Any, payload: dict[str, Any]) -> None:
    """Pass the exact scored version and Delta commit to the independent monitoring task."""
    registration = payload.get("monitoring")
    if registration is None:
        return
    result = payload.get("result", payload)
    bindings = (
        {"configs": registration["configs"]}
        if "configs" in registration
        else {"config": registration["config"]}
    )
    dbutils.jobs.taskValues.set(
        key="monitoring_request",
        value={
            "namespace": registration["inventory_table"].rsplit(".", 1)[0],
            **bindings,
            "commit_version": result.get("commit_version"),
            "noop": result.get("noop", False),
            "has_saved_batch": bool(result.get("manifest")),
        },
    )
