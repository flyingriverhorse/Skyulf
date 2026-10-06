"""Validate and enroll pinned serving monitors through the shared Spark store."""

import json
from typing import Any

from ..monitoring_config import MonitorConfig
from ..monitoring_store import enroll_monitor, ensure_monitoring_store
from ..spark.spark_monitoring_reference import prepare_spark_monitoring_reference


def parse_serving_enrollments(raw: str, namespace: str, values: dict) -> list[MonitorConfig]:
    """Reject an invalid enrollment batch before any store or reference write."""
    try:
        items = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("monitoring_serving_enrollments must be a JSON list.") from exc
    if type(items) is not list or not 1 <= len(items) <= 100:
        raise ValueError("monitoring_serving_enrollments must be a nonempty list of at most 100.")
    configs = [MonitorConfig.from_dict(item) for item in items]
    if len({config.monitor_id for config in configs}) != len(configs):
        raise ValueError("Serving enrollments require distinct model identities.")
    for config in configs:
        _validate_job_binding(config, namespace, values)
    return configs


def _validate_job_binding(config: MonitorConfig, namespace: str, values: dict) -> None:
    """Bind an online model to this job's project and shared store."""
    if config.serving_endpoint is None:
        raise ValueError("Serving enrollment requires serving_endpoint.")
    if config.environment != values["monitoring_environment"]:
        raise ValueError("Serving enrollment environment differs from job environment.")
    if config.project != values["monitoring_project"]:
        raise ValueError("Serving enrollment project differs from job project.")
    if config.reference_namespace != namespace:
        raise ValueError(
            "Serving enrollment reference_namespace differs from monitoring namespace."
        )


def enroll_serving_configs(spark: Any, namespace: str, values: dict) -> dict:
    """Prepare verified references first, then publish each online enrollment."""
    configs = parse_serving_enrollments(
        values.get("monitoring_serving_enrollments", "[]"), namespace, values
    )
    ensure_monitoring_store(spark, *namespace.split("."))
    for config in configs:
        if config.enabled:
            prepare_spark_monitoring_reference(spark, config)
    for config in configs:
        enroll_monitor(spark, namespace, config, preserve_activation=True)
    return {
        "status": "enrolled",
        "models": len(configs),
        "monitor_ids": [config.monitor_id for config in configs],
        "inventory_table": f"{namespace}.model_inventory",
    }
