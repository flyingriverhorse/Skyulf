"""Publish a separate serving request only from a completed champion transition."""

import hashlib
import re
from typing import Any

from ..projects.yaml_config import read_yaml_mapping


def _automatic_serving_enabled(values: dict[str, str]) -> bool:
    """Read dispatch gates before requiring endpoint deployment bindings.

    Development targets follow the champion too; their endpoint and SQL function
    names carry the per-user resource suffix, so they never touch shared ones.
    """
    document = read_yaml_mapping(values["serving_config_path"])
    settings = document.get("serving_endpoint")
    if not isinstance(settings, dict):
        raise ValueError("serving_endpoint must be a mapping.")
    for key, default in (("enabled", None), ("follow_champion", True)):
        value = settings.get(key, default)
        if type(value) is not bool:
            raise ValueError(f"serving_endpoint.{key} must be a boolean.")
        if not value:
            return False
    return True


def active_serving_settings(
    values: dict[str, str], *, model_name: str | None = None
) -> dict[str, Any] | None:
    """Return validated endpoint settings only when this run hands off to serving.

    Initialization calls this too, so a bad config/serving.yml stops the run
    before training or any champion change instead of after the decision.
    """
    if values.get("deployed_serving_endpoint") != "true" or not _automatic_serving_enabled(values):
        return None
    from .serving_endpoint_job import read_serving_endpoint_settings  # noqa: PLC0415

    settings = read_serving_endpoint_settings(values)
    require_serving_model(settings, model_name)
    return settings


def require_serving_model(settings: dict[str, Any] | None, model_name: str | None) -> None:
    """Reject a mismatched endpoint identity before any lifecycle mutation."""
    if settings is not None and model_name is not None and settings["model_name"] != model_name:
        raise ValueError("serving_endpoint.model_name must match the lifecycle model_name.")


def serving_transition(receipt: dict[str, Any] | None, *, model_name: str) -> str:
    """Return the concrete changed champion version, or no request for other outcomes."""
    if not receipt or receipt.get("kind") not in {"initial", "promotion", "rollback"}:
        return ""
    if receipt.get("alias") != "champion":
        return ""
    if receipt.get("model_name") != model_name:
        raise ValueError("Serving handoff requires the configured model's champion transition.")
    version = receipt.get("new_version")
    if not isinstance(version, str) or not re.fullmatch(r"[1-9][0-9]*", version):
        raise ValueError("Serving handoff requires a concrete model version.")
    return "" if receipt.get("prior_version") == version else version


def publish_serving_handoff(
    dbutils: Any, receipt: dict[str, Any] | None, values: dict[str, str]
) -> None:
    """Set child-job values without making cloud calls or changing the score policy.

    Old projects have no serving deployment contract and remain inert. Manual
    endpoint runs are independent; automatic handoff follows serving.yml in every target.
    """
    if values.get("deployed_serving_endpoint") != "true":
        return
    settings = active_serving_settings(values)
    version = (
        "" if settings is None else serving_transition(receipt, model_name=settings["model_name"])
    )
    dbutils.jobs.taskValues.set(key="serving_requested", value=bool(version))
    dbutils.jobs.taskValues.set(key="serving_model_version", value=version)
    event_id = receipt.get("event_id", "") if version and receipt is not None else ""
    dbutils.jobs.taskValues.set(key="serving_event_id", value=event_id)


def submit_serving_run(client: Any, values: dict[str, str]) -> dict[str, Any]:
    """Submit once per promotion event, with a parent-run fallback for older projects."""
    for key in ("serving_job_id", "job_run_id", "model_version"):
        value = values.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
            raise ValueError(f"{key} must be a resolved positive integer.")
    event_id = values.get("serving_event_id", "")
    if not isinstance(event_id, str) or (event_id and not re.fullmatch(r"[\w-]{1,256}", event_id)):
        raise ValueError("serving_event_id must be a resolved event identity or empty.")
    origin = f"event:{event_id}" if event_id else values["job_run_id"]
    identity = ":".join((values["serving_job_id"], origin, values["model_version"]))
    submitted = client.jobs.run_now(
        job_id=int(values["serving_job_id"]),
        job_parameters={"model_version": values["model_version"]},
        idempotency_token=hashlib.sha256(identity.encode()).hexdigest(),
    )
    run_id = submitted.response.run_id
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("Serving submission did not return a concrete run_id.")
    return {"status": "SUBMITTED", "run_id": run_id, "model_version": values["model_version"]}


def run_serving_dispatch_notebook(dbutils: Any) -> dict[str, Any]:
    """Publish the child run identity, explicitly separating submission from serving success."""
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    result = submit_serving_run(WorkspaceClient(), dbutils.widgets.getAll())
    dbutils.jobs.taskValues.set(key="serving_run_id", value=result["run_id"])
    return result
