"""Register each model project's actual scored version in the shared Delta inventory."""

from typing import Any

from ._contracts import input_budget_bytes
from .monitoring_config import (
    MAX_MONITOR_BYTES,
    MAX_MONITOR_ROWS,
    MonitorConfig,
    parse_drift_thresholds,
    parse_performance_policies,
    store_namespace,
)
from .monitoring_store import _validate_activation_order, enroll_monitor, ensure_monitoring_store
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


def _monitor_budget(value: Any, maximum: int) -> Any:
    """Cap producer allowances while leaving invalid values for contract validation."""
    return min(value, maximum) if type(value) is int else value


def build_monitor_enrollment_config(
    workflow: dict[str, Any], values: dict[str, str], version: str
) -> MonitorConfig:
    """Use resolved producer names and the physical generation of the selected version."""
    policies = parse_performance_policies(values.get("monitoring_performance_policies", "{}"))
    execution = values.get("monitoring_execution_engine", "local")
    budget = {"max_rows": 10000, "max_input_mb": 64} if execution == "spark" else workflow
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
        max_rows=_monitor_budget(budget.get("max_rows", 10000), MAX_MONITOR_ROWS),
        max_bytes=_monitor_budget(
            budget.get("max_bytes", input_budget_bytes(budget.get("max_input_mb", 64))),
            MAX_MONITOR_BYTES,
        ),
        enabled=values.get("monitoring_enabled", "true") == "true",
        thresholds=parse_drift_thresholds(values.get("monitoring_drift_thresholds", "{}")),
        performance_policy=policies.get(workflow["model_name"]),
        execution_engine=execution,
        reference_namespace=(
            store_namespace(
                values.get("monitoring_catalog", ""), values.get("monitoring_schema", "")
            )
            if execution == "spark"
            else None
        ),
    )


def validate_monitoring_settings(values: dict[str, str], workflow: dict[str, Any]) -> None:
    """Validate configured registration before any scoring write or recovery request."""
    if monitoring_destination(values) is None:
        return
    policies = parse_performance_policies(values.get("monitoring_performance_policies", "{}"))
    if workflow.get("training_layout", "single_model") != "multi_target" and set(policies) - {
        workflow["model_name"]
    }:
        raise ValueError("Performance policy model name does not match the workflow model.")
    build_monitor_enrollment_config(workflow, values, "1")


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
    config = build_monitor_enrollment_config(workflow, values, version)
    ensure_monitoring_store(spark, *namespace.split("."))
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
    config = build_monitor_enrollment_config(workflow, values, receipt["new_version"])
    _validate_activation_order(activation_started_ms)
    ensure_monitoring_store(spark, *namespace.split("."))
    enroll_monitor(spark, namespace, config, activation_started_ms=activation_started_ms)
    if config.execution_engine == "spark" and config.enabled:
        from .spark_monitoring_reference import prepare_spark_monitoring_reference  # noqa: PLC0415

        prepare_spark_monitoring_reference(
            spark,
            config,
            tracking_uri=workflow.get("tracking_uri", "databricks"),
            registry_uri=workflow.get("registry_uri", "databricks-uc"),
        )
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
