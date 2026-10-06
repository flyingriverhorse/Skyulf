"""Run Spark monitoring separately from producer scoring and revisit delayed outcomes."""

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from ...observability.monitoring.local.monitoring import run_monitoring
from ...observability.monitoring.monitoring_config import MonitorConfig, json_digest
from ...observability.monitoring.monitoring_registration import monitoring_destination
from ...observability.monitoring.monitoring_sources import observation_window
from ...observability.monitoring.monitoring_store import (
    ensure_monitoring_store,
    load_enrolled_models,
)
from ...observability.monitoring.serving.serving_registration import enroll_serving_configs
from ...observability.monitoring.spark.spark_monitoring_reference import (
    prepare_spark_monitoring_reference,
)
from ...observability.monitoring.spark.spark_monitoring_windows import revisit_performance_windows
from ..shared.notebook_diagnostics import result_summary


def dispatch_monitoring_job(request: dict, values: dict, *, workspace: Any = None) -> dict:
    """Submit a receipt-bound independent job without waiting for its metric computation."""
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    job = values.get("monitoring_job_id", "")
    if not isinstance(job, str) or not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("Spark monitoring requires a resolved monitoring_job_id.")
    invocation = values.get("monitoring_invocation_id")
    if invocation is not None and not re.fullmatch(r"[1-9][0-9]*", str(invocation)):
        raise ValueError("monitoring_invocation_id must be a resolved producer run ID.")
    if request.get("noop") and invocation is None:
        raise ValueError("No-op Spark monitoring requires monitoring_invocation_id.")
    client = workspace or WorkspaceClient()
    response = client.jobs.run_now(
        job_id=int(job),
        idempotency_token=json_digest(
            {"job_id": job, "request": request, "invocation": invocation}
        ),
        job_parameters={"monitoring_request": json.dumps(request, sort_keys=True)},
    )
    return {"status": "queued", "job_id": int(job), "run_id": response.run_id}


def project_configs(spark: Any, namespace: str, values: dict) -> list[MonitorConfig]:
    """Select the project's current active Spark enrollments without resolving new aliases."""
    configs = [MonitorConfig.from_dict(item) for item in load_enrolled_models(spark, namespace)]
    return [
        config
        for config in configs
        if (
            config.environment == values["monitoring_environment"]
            and config.project == values["monitoring_project"]
            and config.enabled
            and config.execution_engine == "spark"
        )
    ]


def prepare_project_references(
    spark: Any, values: dict, *, model_ids: set[str] | None = None
) -> dict:
    """Prepare at activation or explicit legacy migration, never during ordinary observation."""
    namespace = monitoring_destination(values)
    if namespace is None:
        return {"status": "disabled"}
    configs = project_configs(spark, namespace, values)
    selected = [config for config in configs if model_ids is None or config.monitor_id in model_ids]
    for config in selected:
        prepare_spark_monitoring_reference(spark, config)
    return {"status": "prepared", "models": len(selected)}


def _requested_configs(
    request: dict, current: list[MonitorConfig], namespace: str
) -> list[MonitorConfig]:
    """Prevent superseded score requests from replacing a newer activation's monitoring."""
    if request.get("namespace") != namespace:
        raise ValueError("Monitoring request belongs to another namespace.")
    configs = [
        MonitorConfig.from_dict(item) for item in request.get("configs", [request.get("config")])
    ]
    if not configs or len({config.monitor_id for config in configs}) != len(configs):
        raise ValueError("Monitoring request requires distinct model identities.")
    live = {config.monitor_id: config for config in current}
    return [
        config
        for config in configs
        if config.monitor_id in live and config.payload() == live[config.monitor_id].payload()
    ]


def _publish_references(
    dbutils: Any, namespace: str, configs: list, start: datetime, end: datetime
) -> None:
    """Keep the existing report and guarded retraining tasks bound to exact observations."""
    references = [
        {
            "status": "ready",
            "namespace": namespace,
            "monitor_id": config.monitor_id,
            "config_digest": json_digest(config.payload()),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
        }
        for config in configs
    ]
    value = (
        references[0] if len(references) == 1 else {"status": "ready", "observations": references}
    )
    dbutils.jobs.taskValues.set(key="monitoring_reference", value=value)


def run_project_monitoring_notebook(spark: Any, dbutils: Any) -> dict:
    """Observe a score receipt or the scheduled project inventory on dedicated Spark compute."""
    values = dbutils.widgets.getAll()
    dbutils.jobs.taskValues.set(key="monitoring_reference", value={"status": "disabled"})
    namespace = monitoring_destination(values)
    if namespace is None or values.get("monitoring_enabled", "true") != "true":
        return {"status": "disabled"}
    action = values.get("monitoring_action", "observe")
    if action == "enroll_serving":
        return enroll_serving_configs(spark, namespace, values)
    if action == "prepare_references":
        return prepare_project_references(spark, values)
    if action != "observe":
        raise ValueError("monitoring_action must be observe, prepare_references or enroll_serving.")
    return _observe_project(spark, dbutils, values, namespace)


def _observe_project(spark: Any, dbutils: Any, values: dict, namespace: str) -> dict:
    """Bind one observation cutoff to active enrollments and optional scoring receipts."""
    from .monitoring_tasks import scoring_observation_window  # noqa: PLC0415

    request = _monitoring_request(values)
    now = _observation_time(values)
    observation_window(now, None, None)
    windows = _validate_observation_settings(values)
    if request is None:
        ensure_monitoring_store(spark, *namespace.split("."))
    else:
        _requested_configs(request, [], namespace)
    configs = project_configs(spark, namespace, values)
    if request is not None:
        configs = _requested_configs(request, configs, namespace)
    if not configs:
        return {"status": "no_active_models"}
    start, end = now - timedelta(days=1), now
    if request is not None and request.get("commit_version") is not None:
        start, end = scoring_observation_window(spark, configs[0], request["commit_version"])
    revisit_performance_windows(spark, namespace, configs, now, windows=windows)
    _publish_references(dbutils, namespace, configs, start, end)
    result = run_monitoring(
        spark,
        namespace,
        [config.payload() for config in configs],
        as_of=now,
        window_start=start,
        window_end=end,
        enroll_models=False,
        experiment_name=values.get("monitoring_experiment_name") or None,
    )
    if result["failed"]:
        print(result_summary(result))
        raise RuntimeError("Spark monitoring failed; durable failure reports were saved.")
    return result


def _monitoring_request(values: dict) -> dict | None:
    """Distinguish an absent schedule receipt from malformed explicit JSON values."""
    raw = values.get("monitoring_request", "")
    if not raw:
        return None
    request = json.loads(raw)
    if not isinstance(request, dict):
        raise ValueError("monitoring_request must be a JSON object.")
    return request


def _validate_observation_settings(values: dict) -> int:
    """Reject invalid schedule controls before provisioning an empty shared store."""
    for key in ("monitoring_environment", "monitoring_project"):
        value = values.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
            raise ValueError("Monitoring environment/project must be bounded identifiers.")
    windows = int(values.get("monitoring_revisit_windows", "3"))
    if not 1 <= windows <= 100:
        raise ValueError("monitoring_revisit_windows must be an integer between 1 and 100.")
    return windows


def _observation_time(values: dict) -> datetime:
    """Bind retries to the UTC job start without trusting timezone-free display strings."""
    if values.get("as_of"):
        return datetime.fromisoformat(values["as_of"].replace("Z", "+00:00"))
    if values.get("as_of_unix_ms"):
        return datetime.fromtimestamp(int(values["as_of_unix_ms"]) / 1000, UTC)
    return datetime.now(UTC)
