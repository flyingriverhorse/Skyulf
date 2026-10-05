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
from .prediction_output import scoring_target


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


def prepare_scoring_monitoring_notebook(dbutils: Any) -> dict:
    """Publish an exact validated Spark receipt for a visible native Run Job task."""
    values = dbutils.widgets.getAll()
    dbutils.jobs.taskValues.set(key="monitoring_ready", value=False)
    dbutils.jobs.taskValues.set(key="monitoring_request_json", value="")
    namespace = monitoring_destination(values)
    if namespace is None or values.get("monitoring_enabled", "true") != "true":
        return {"status": "disabled"}
    workflow = read_notebook_config(values)
    request = _scoring_request(dbutils, workflow)
    configs = _native_request_configs(request, namespace)
    _validate_requested_models(configs, workflow, values)
    for config in configs:
        _validate_native_config(config, namespace, workflow, values)
    if _empty_scoring_request(request, configs):
        return {"status": "no_new_predictions"}
    serialized = json.dumps(request, sort_keys=True, allow_nan=False)
    dbutils.jobs.taskValues.set(key="monitoring_request_json", value=serialized)
    dbutils.jobs.taskValues.set(key="monitoring_ready", value=True)
    return {"status": "ready"}


def _native_request_configs(request: dict, namespace: str) -> list[MonitorConfig]:
    """Reject ambiguous identities and invalid committed-batch metadata before dispatch."""
    _validate_native_receipt(request, namespace)
    configs = [
        MonitorConfig.from_dict(item) for item in request.get("configs", [request.get("config")])
    ]
    if not configs or len({config.monitor_id for config in configs}) != len(configs):
        raise ValueError("Monitoring request requires distinct model identities.")
    return configs


def _validate_native_receipt(request: dict, namespace: str) -> None:
    """Require one configuration form and explicit successful scoring commit metadata."""
    if ("configs" in request) == ("config" in request):
        raise ValueError("Monitoring requires exactly one configuration form.")
    if namespace != request.get("namespace"):
        raise ValueError("Monitoring destination differs from the scoring request.")
    if type(request.get("has_saved_batch", False)) is not bool:
        raise ValueError("Monitoring requires a boolean saved-batch marker.")
    commit = request.get("commit_version")
    requires_commit = not request["noop"] or request.get("has_saved_batch", False)
    if (requires_commit or commit is not None) and (type(commit) is not int or commit < 0):
        raise ValueError("Monitoring requires a committed prediction version.")


def _validate_native_config(
    config: MonitorConfig, namespace: str, workflow: dict, values: dict
) -> None:
    """Bind native Spark observation to this producer's pinned physical scoring output."""
    if config.execution_engine != "spark" or not config.model_version or not config.enabled:
        raise ValueError("Native monitoring requires an enabled pinned Spark configuration.")
    expected = (
        namespace,
        values.get("monitoring_environment"),
        values.get("monitoring_project"),
        workflow["score_source_table"],
    )
    actual = (config.reference_namespace, config.environment, config.project, config.source_table)
    if actual != expected:
        raise ValueError("Monitoring configuration differs from the scoring producer.")
    if workflow.get("training_layout") != "multi_target":
        prediction_table = scoring_target(workflow | {"model_version": config.model_version})
        if config.model_set_name or config.prediction_table != prediction_table:
            raise ValueError("Monitoring prediction table differs from the scoring request.")


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
    handoff = _spark_handoff(dbutils, configs, request, values)
    if handoff is not None:
        return handoff
    if _empty_scoring_request(request, configs):
        dbutils.jobs.taskValues.set(
            key="monitoring_reference", value={"status": "no_new_predictions"}
        )
        return {"status": "no_new_predictions"}
    return _observe_configs(spark, dbutils, namespace, configs, request, workflow, values)


def _empty_scoring_request(request: dict, configs: list[MonitorConfig]) -> bool:
    """Skip a no-op only when neither saved predictions nor delayed outcomes need observation."""
    return (
        request["noop"]
        and not request.get("has_saved_batch", False)
        and not any(_performance_enabled(config) for config in configs)
    )


def _spark_handoff(
    dbutils: Any, configs: list[MonitorConfig], request: dict, values: dict
) -> dict | None:
    """Dispatch homogeneous Spark requests and leave local observation inline."""
    if not any(config.execution_engine == "spark" for config in configs):
        return None
    from .spark_monitoring_job import dispatch_monitoring_job  # noqa: PLC0415

    if not all(config.execution_engine == "spark" for config in configs):
        raise ValueError("A monitoring request cannot mix local and Spark execution.")
    result = dispatch_monitoring_job(request, values)
    dbutils.jobs.taskValues.set(key="monitoring_reference", value=result)
    return result


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
    now = datetime.now(UTC)
    performance_only = request["noop"] and not request.get("has_saved_batch", False)
    if performance_only:
        start, end = now - timedelta(days=1), now
    else:
        start, end = scoring_observation_window(spark, configs[0], request["commit_version"])
    references = [_observation_reference(namespace, config, start, end) for config in configs]
    dbutils.jobs.taskValues.set(
        key="monitoring_reference",
        value=references[0]
        if len(references) == 1
        else {"status": "ready", "observations": references},
    )
    pending = _pending_configs(spark, namespace, configs, start, end)
    if not pending:
        return {"status": "already_observed"}
    result = run_monitoring(
        spark,
        namespace,
        pending,
        as_of=now if any(_performance_enabled(config) for config in configs) else end,
        window_start=start,
        window_end=end,
        tracking_uri=workflow.get("tracking_uri", "databricks"),
        registry_uri=workflow.get("registry_uri", "databricks-uc"),
        experiment_name=values.get("monitoring_experiment_name") or None,
        enroll_models=False,
        **({"performance_only": True} if performance_only else {}),
    )
    print(json.dumps(result, sort_keys=True))
    return result


def _pending_configs(
    spark: Any, namespace: str, configs: list[MonitorConfig], start: datetime, end: datetime
) -> list[dict]:
    """Keep immutable drift retries cached while revisiting delayed performance labels."""
    return [
        config.payload()
        for config in configs
        if _performance_enabled(config)
        or not completed_observation(spark, namespace, config, start, end)
    ]


def _performance_enabled(config: MonitorConfig) -> bool:
    """Revisit mature label windows even when scoring has no new committed predictions."""
    return bool(config.performance_policy and config.performance_policy.get("mode") != "off")


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
