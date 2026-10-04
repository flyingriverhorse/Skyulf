"""Visible deployment enrollment and post-scoring monitoring tasks for model repositories."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from ._lifecycle_state import LifecycleContext, PhaseStore
from .delta import history
from .job_runtime import lifecycle_widget_context, read_notebook_config, saved_notebook_request
from .monitoring import run_monitoring
from .monitoring_config import MonitorConfig, json_digest, qualified_name
from .monitoring_output import render_monitor_output
from .monitoring_registration import monitoring_destination, register_deployed_monitor


def deployment_store(values: dict[str, str], *, expected_phase: str = "result") -> PhaseStore:
    """Bind the completed lifecycle receipt to this invocation before using frozen config."""
    for name, pattern in (("repair_count", r"[0-9]+"), ("execution_count", r"[1-9][0-9]*")):
        if not re.fullmatch(pattern, values.get(name, "")):
            raise ValueError("Enrollment repair metadata must be resolved nonnegative counters.")
    # This task only reads a completed receipt and performs an ordered upsert.
    # Preserve the original invocation identity while permitting this task's repair.
    context = LifecycleContext(
        **lifecycle_widget_context(values | {"repair_count": "0", "execution_count": "1"})
    )
    request = saved_notebook_request(values)
    if request["reference"].get("phase") != expected_phase:
        raise ValueError("Monitor enrollment requires a completed lifecycle result.")
    store = PhaseStore(request["tracking_uri"], context)
    store.bind(request["reference"])
    return store


def run_monitor_enrollment_notebook(spark: Any, dbutils: Any) -> dict:
    """Register the activated version without needing a prediction table or running inference."""
    values = dbutils.widgets.getAll()
    if monitoring_destination(values) is None:
        return {"status": "disabled"}
    store = deployment_store(values)
    result = register_deployed_monitor(
        spark,
        store.request["config"],
        values,
        store.receipt("result")["output"],
        activation_started_ms=store.client.get_run(store.run_id).info.start_time,
    )
    output = {"status": "enrolled", **result} if result else {"status": "no_activation"}
    print(json.dumps(output, sort_keys=True))
    return output


def scoring_observation_window(
    spark: Any, config: MonitorConfig, commit_version: int
) -> tuple[datetime, datetime]:
    """Select the exact scoring commit, including repairs long after the default daily window."""
    if type(commit_version) is not int or commit_version < 0:
        raise ValueError("Monitoring requires a committed prediction version.")
    row = (
        history(spark, config.prediction_table)
        .where(f"version = {commit_version}")
        .selectExpr("unix_micros(timestamp) AS committed_us")
        .first()
    )
    if row is None:
        raise ValueError("Scoring commit is no longer available for monitoring.")
    start = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=row["committed_us"])
    return start, start + timedelta(microseconds=1)


def _scoring_request(dbutils: Any, workflow: dict) -> dict:
    """Read only the successful scoring branch, never an excluded recovery task."""
    task = "score"
    if workflow.get("auto_rebuild_on_cdf_expiry", False):
        recovery = dbutils.jobs.taskValues.get(taskKey="score", key="recovery_required")
        if type(recovery) is not bool:
            raise ValueError("Monitoring requires a completed scoring recovery decision.")
        task = "recover_predictions" if recovery else "score"
    request = dbutils.jobs.taskValues.get(taskKey=task, key="monitoring_request")
    if not isinstance(request, dict) or type(request.get("noop")) is not bool:
        raise ValueError("Monitoring requires a completed scoring request.")
    return request


def completed_observation(
    spark: Any, namespace: str, config: MonitorConfig, start: datetime, end: datetime
) -> bool:
    """Skip an already measured no-op, but allow a missing or failed observation to recover."""
    name = qualified_name(f"{namespace}.monitoring_results")
    predicate = (
        f"monitor_id = '{config.monitor_id}' AND config_digest = '{json_digest(config.payload())}' "
        f"AND window_start = TIMESTAMP '{start.isoformat()}' "
        f"AND window_end = TIMESTAMP '{end.isoformat()}' AND status <> 'failed'"
    )
    return bool(spark.table(name).where(predicate).limit(1).count())


def _run_scoring_monitor(spark: Any, dbutils: Any) -> dict:
    """Measure the pinned scoring batch and publish its durable report reference."""
    values = dbutils.widgets.getAll()
    dbutils.jobs.taskValues.set(key="monitoring_reference", value={"status": "disabled"})
    namespace = monitoring_destination(values)
    if namespace is None or values.get("monitoring_enabled", "true") != "true":
        return {"status": "disabled"}
    workflow = read_notebook_config(values)
    request = _scoring_request(dbutils, workflow)
    configs = [
        MonitorConfig.from_dict(item) for item in request.get("configs", [request.get("config")])
    ]
    if namespace != request["namespace"] or not configs:
        raise ValueError("Monitoring destination or model differs from the scoring request.")
    _validate_requested_models(configs, workflow, values)
    if request["noop"] and not request.get("has_saved_batch", False):
        dbutils.jobs.taskValues.set(
            key="monitoring_reference", value={"status": "no_new_predictions"}
        )
        return {"status": "no_new_predictions"}
    return _observe_configs(spark, dbutils, namespace, configs, request, workflow, values)


def _observe_configs(
    spark: Any,
    dbutils: Any,
    namespace: str,
    configs: list[MonitorConfig],
    request: dict,
    workflow: dict,
    values: dict,
) -> dict:
    """Measure only missing component reports and retain all saved-batch references."""
    start, end = scoring_observation_window(spark, configs[0], request["commit_version"])
    references = [_observation_reference(namespace, config, start, end) for config in configs]
    dbutils.jobs.taskValues.set(
        key="monitoring_reference",
        value=references[0]
        if len(references) == 1
        else {"status": "ready", "observations": references},
    )
    pending = [
        config.payload()
        for config in configs
        if not completed_observation(spark, namespace, config, start, end)
    ]
    if not pending:
        return {"status": "already_observed"}
    result = run_monitoring(
        spark,
        namespace,
        pending,
        as_of=end,
        window_start=start,
        window_end=end,
        tracking_uri=workflow.get("tracking_uri", "databricks"),
        registry_uri=workflow.get("registry_uri", "databricks-uc"),
        experiment_name=values.get("monitoring_experiment_name") or None,
        enroll_models=False,
    )
    print(json.dumps(result, sort_keys=True))
    return result


def _validate_requested_models(configs: list[MonitorConfig], workflow: dict, values: dict) -> None:
    """Reject unrelated producers and mixed physical batches before querying predictions."""
    if workflow.get("training_layout") == "multi_target":
        from .model_set_project import load_project_model_set  # noqa: PLC0415

        settings = load_project_model_set(values, workflow)
        if not settings or any(
            config.model_set_name != settings["model_name"]
            or config.prediction_table != settings["prediction_table"]
            for config in configs
        ):
            raise ValueError("Monitoring model set differs from the scoring request.")
        if len({config.model_set_version for config in configs}) != 1:
            raise ValueError("Monitoring requires one pinned model-set version.")
    elif len(configs) != 1 or configs[0].model_name != workflow["model_name"]:
        raise ValueError("Monitoring destination or model differs from the scoring request.")


def _observation_reference(
    namespace: str, config: MonitorConfig, start: datetime, end: datetime
) -> dict:
    """Publish a compact lookup for each component's durable monitoring result."""
    return {
        "status": "ready",
        "namespace": namespace,
        "monitor_id": config.monitor_id,
        "config_digest": json_digest(config.payload()),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
    }


def run_scoring_monitor_notebook(
    spark: Any, dbutils: Any, *, display_html: Callable[[str], Any] | None = None
) -> dict:
    """Measure this scored model, show dashboard navigation and retain visible failures."""
    result = _run_scoring_monitor(spark, dbutils)
    if display_html is not None:
        display_html(
            render_monitor_output(
                result, dbutils.widgets.getAll().get("monitoring_dashboard_url", "")
            )
        )
    if result.get("failed"):
        raise RuntimeError("Monitoring failed; results were saved in the monitoring store.")
    return result
