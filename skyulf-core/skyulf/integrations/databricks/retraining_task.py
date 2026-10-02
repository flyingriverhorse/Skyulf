"""Request existing training workflows after fresh, usable feature drift."""

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from .job_output import output_table
from .job_runtime import read_notebook_config
from .monitoring_config import MonitorConfig, json_digest
from .monitoring_output import load_observation
from .monitoring_registration import monitoring_destination
from .monitoring_store import load_enrolled_models


def retraining_policy(values: dict[str, str]) -> dict:
    """Require an explicit opt-in and bounded finite submission controls."""
    mode = values.get("on_drift", "disabled")
    if mode not in {"disabled", "retrain"}:
        raise ValueError("on_drift must be disabled or retrain.")
    cooldown = float(values.get("on_drift_cooldown_hours", "24"))
    minimum = int(values.get("on_drift_min_new_training_rows", "1"))
    if not math.isfinite(cooldown) or cooldown < 0 or minimum < 1:
        raise ValueError("On-drift cooldown must be finite/nonnegative and minimum rows positive.")
    return {"mode": mode, "cooldown_hours": cooldown, "minimum_rows": minimum}


def _broken_input(item: dict) -> bool:
    """Identify corruption independently of performance and distribution changes."""
    return bool(item.get("has_issue")) and (
        item.get("category") == "quality"
        or item.get("metric_name") in {"schema_missing", "type_drift"}
    )


def _drift_metrics_decision(metrics: list[dict]) -> str:
    """Separate distribution changes from broken or unmeasured feature inputs."""
    if any(_broken_input(item) for item in metrics):
        return "quality_issue"
    checks = [item for item in metrics if item.get("category") == "drift"]
    if not checks or any(item.get("status") != "measured" for item in checks):
        return "incomplete_drift"
    return "ready" if any(item.get("has_issue") for item in checks) else "no_drift"


def observation_decision(row: dict, config: MonitorConfig, now: datetime) -> str:
    """Admit only a recent observation for the currently enrolled concrete model."""
    if not config.enabled:
        return "disabled"
    if (
        row["config_digest"] != json_digest(config.payload())
        or row["model_version"] != config.model_version
        or row["monitor_id"] != config.monitor_id
    ):
        return "superseded"
    observed = row.get("observed_at")
    if observed is None:
        return "stale_observation"
    # Spark TIMESTAMP values returned to Python are naive UTC in these notebooks.
    observed = observed.replace(tzinfo=UTC) if observed.tzinfo is None else observed
    age = (now - observed).total_seconds()
    if age < 0 or age > config.expected_interval_hours * 3600:
        return "stale_observation"
    if row["status"] != "drift":
        return "no_drift"
    return _drift_metrics_decision(json.loads(row["report_json"])["metrics"])


def _training_workflows(values: dict[str, str]) -> dict[str, dict]:
    """Load the same target-bound Python recipes used by the training job."""
    from .branch_notebook import load_training_branch_configs  # noqa: PLC0415
    from .project import load_project_workflow  # noqa: PLC0415

    base = read_notebook_config(values)
    if base.get("training_layout") == "multi_target":
        branches = load_training_branch_configs(values)
        return {config["model_name"]: config for config in branches.values()}
    path = Path(values["config_path"]).parent.parent / "src/features"
    config = load_project_workflow(base, path)
    return {config["model_name"]: config}


def _current_configs(spark: Any, namespace: str, values: dict[str, str]) -> dict:
    """Keep independent repositories and environments outside this job's ownership."""
    configs = [MonitorConfig.from_dict(item) for item in load_enrolled_models(spark, namespace)]
    return {
        item.monitor_id: item
        for item in configs
        if (item.environment, item.project)
        == (values["monitoring_environment"], values["monitoring_project"])
    }


def _candidate(
    spark: Any, row: dict, config: MonitorConfig, workflow: dict, now: datetime, minimum: int
) -> dict:
    """Require fresh rows in the actual training partition, never fit during scoring."""
    from .retraining_data import assess_training_data  # noqa: PLC0415

    status = observation_decision(row, config, now)
    result = {
        "model_name": config.model_name,
        "report_id": row["report_id"],
        "status": status,
        "monitor_id": config.monitor_id,
        "config_digest": row["config_digest"],
    }
    if status != "ready":
        return result
    evidence = assess_training_data(spark, config, workflow, now)
    result.update(evidence)
    if evidence["changed_rows"] < minimum:
        result["status"] = "no_new_training_data"
    return result


def _collect_candidates(
    spark: Any, references: list[dict], namespace: str, values: dict, now: datetime, minimum: int
) -> list[dict]:
    """Bind every observation to this project's current inventory and training recipe."""
    configs = _current_configs(spark, namespace, values)
    workflows = _training_workflows(values)
    results = []
    for reference in references:
        if reference.get("namespace") != namespace:
            raise ValueError("On-drift observation belongs to a different monitoring namespace.")
        row = load_observation(spark, reference)
        config = configs.get(row["monitor_id"])
        if config is None or config.model_name not in workflows:
            raise ValueError("On-drift observation does not belong to this training project.")
        results.append(_candidate(spark, row, config, workflows[config.model_name], now, minimum))
    return results


def _submit_candidates(
    spark: Any,
    workspace: Any,
    namespace: str,
    values: dict,
    policy: dict,
    results: list[dict],
    now: datetime,
    *,
    preview_only: bool = False,
) -> dict:
    """Submit the whole configured train job once even when several branches drift."""
    from .retraining_requests import submit_retraining  # noqa: PLC0415

    ready = [item for item in results if item["status"] == "ready"]
    if not ready:
        return {"status": "not_requested", "models": results}
    current = _current_configs(spark, namespace, values)
    if any(
        item["monitor_id"] not in current
        or json_digest(current[item["monitor_id"]].payload()) != item["config_digest"]
        for item in ready
    ):
        return {"status": "superseded", "models": results}
    job_id = _training_job_id(workspace, values)
    identity = sorted(
        (
            item["model_name"],
            item["baseline_model_version"],
            item["source_table"],
            item["content_sha256"],
        )
        for item in ready
    )
    request_id = json_digest({"job_id": job_id, "training_data": identity})
    result = submit_retraining(
        spark,
        workspace,
        namespace=namespace,
        job_id=job_id,
        request_id=request_id,
        evidence={"models": results},
        cooldown_hours=policy["cooldown_hours"],
        now=now,
        **({"preview_only": True} if preview_only else {}),
    )
    return {**result, "models": results}


def _training_job_name(values: dict[str, str]) -> str:
    """Retain the current target's actual prefix when locating its paired train job."""
    name = values["train_job_name"]
    score_name = values.get("score_job_name")
    if score_name is None:
        return name
    score_suffix = name.removesuffix("_train") + "_score"
    if not name.endswith("_train") or not score_name.endswith(score_suffix):
        raise ValueError("On-drift requires the paired generated score/train job names.")
    return score_name[: -len(score_suffix)] + name


def _training_job_id(workspace: Any, values: dict[str, str]) -> int:
    """Resolve a unique deployed name without making the Bundle job graph cyclic."""
    name = _training_job_name(values)
    matches = list(islice(workspace.jobs.list(name=name, limit=2), 2))
    if len(matches) != 1:
        raise ValueError("On-drift requires exactly one train job with the configured name.")
    job = matches[0]
    if (
        type(job.job_id) is not int
        or job.job_id <= 0
        or job.settings is None
        or job.settings.name != name
    ):
        raise ValueError("On-drift requires the exact train job name and a positive job ID.")
    return job.job_id


def run_retraining_notebook(
    spark: Any,
    dbutils: Any,
    *,
    workspace: Any = None,
    display_html: Callable[[str], Any] | None = None,
) -> dict:
    """Expose optional retraining as a visible, asynchronous scoring-job task."""
    values = dbutils.widgets.getAll()
    policy = retraining_policy(values)
    result: dict = {"status": "disabled"}
    if policy["mode"] == "retrain":
        result = _run_enabled(spark, dbutils, workspace, values, policy)
    dbutils.jobs.taskValues.set(key="retraining_result", value=result)
    if display_html is not None:
        display_html(
            "<h2>Retraining after drift</h2>"
            + output_table(
                ("Decision", "Training run"), [(result["status"], result.get("run_id", ""))]
            )
            + "<p>Training uses the existing quality and approval policy. "
            "Actual targets must already be present in the configured training source.</p>"
        )
    return result


def run_retraining_check_notebook(
    spark: Any,
    dbutils: Any,
    *,
    workspace: Any = None,
    display_html: Callable[[str], Any] | None = None,
) -> dict:
    """Publish advisory eligibility for a visible If/else task without starting training."""
    values = dbutils.widgets.getAll()
    policy = retraining_policy(values)
    result: dict = {"status": "disabled"}
    if policy["mode"] == "retrain":
        result = _run_enabled(spark, dbutils, workspace, values, policy, preview_only=True)
    dbutils.jobs.taskValues.set(key="retraining_needed", value=result["status"] == "ready")
    dbutils.jobs.taskValues.set(key="retraining_check", value=result)
    if display_html is not None:
        rows = [
            (item["model_name"], item["status"], item.get("changed_rows", ""))
            for item in result.get("models", [])
        ]
        display_html(
            "<h2>Retraining eligibility</h2>"
            + output_table(("Decision",), [(result["status"],)])
            + output_table(("Model", "Reason", "New training rows"), rows)
            + "<p>The true branch rechecks these guards before submitting training.</p>"
        )
    return result


def _run_enabled(
    spark: Any,
    dbutils: Any,
    workspace: Any,
    values: dict,
    policy: dict,
    *,
    preview_only: bool = False,
) -> dict:
    """Validate the completed monitoring task before any remote training request."""
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    namespace = monitoring_destination(values)
    if namespace is None or values.get("monitoring_enabled") != "true":
        raise ValueError("on_drift=retrain requires enabled monitoring and a central namespace.")
    reference = dbutils.jobs.taskValues.get(taskKey="monitor_model", key="monitoring_reference")
    if reference.get("status") != "ready":
        return {"status": reference.get("status", "no_observation")}
    now = datetime.now(UTC)
    results = _collect_candidates(
        spark,
        reference.get("observations", [reference]),
        namespace,
        values,
        now,
        policy["minimum_rows"],
    )
    return _submit_candidates(
        spark,
        workspace or WorkspaceClient(),
        namespace,
        values,
        policy,
        results,
        now,
        preview_only=preview_only,
    )
