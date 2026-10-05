"""Route optional CDF recovery through explicit scoring tasks and a shared report."""

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from ...data.delta_io.cdf_recovery import CdfRecoveryRequired, validate_recovery_request
from ...jobs.shared.job_output import render_bundle_output
from ...jobs.shared.job_runtime import notebook_output, read_notebook_config, run_bundle_action
from ...observability.monitoring.monitoring_registration import (
    publish_monitoring_request,
    register_scoring_monitor,
    validate_monitoring_settings,
)

_LARGE_RECEIPT_FIELDS = {
    "temporal_history",
    "set_history",
    "cdf_recovery_request",
}


def _compact_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Exclude continuation state from task values while preserving the raw notebook result."""
    compact = dict(payload)
    result = dict(compact.get("result", compact))
    manifest = result.get("manifest")
    if isinstance(manifest, dict):
        result["manifest"] = {
            key: value for key, value in manifest.items() if key not in _LARGE_RECEIPT_FIELDS
        }
    if "result" in compact:
        compact["result"] = result
    else:
        compact = result
    if len(json.dumps(compact, default=str, allow_nan=False).encode()) > 48 * 1024:
        raise ValueError("Scoring report exceeds the 48 KiB task-value limit.")
    return compact


def _pending_result(request: dict[str, Any]) -> dict[str, Any]:
    """Report a requested recovery without claiming predictions were committed."""
    return {
        "action": "score",
        "recovery_required": True,
        "source_table": request["source_table"],
        "prediction_table": request["target_table"],
        "result": {
            "recovery_required": True,
            "selected_model_name": request["model_name"],
            "selected_model_version": request["model_version"],
            "source_end_version": request["source_end_version"],
        },
    }


def run_scoring_step(
    config: dict[str, Any],
    dbutils: Any,
    score: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Request the recovery branch only for opted-in, explicitly identified CDF loss."""
    enabled = config.get("auto_rebuild_on_cdf_expiry", False)
    if type(enabled) is not bool:
        raise ValueError("auto_rebuild_on_cdf_expiry must be a boolean.")
    try:
        payload = score()
    except CdfRecoveryRequired as error:
        if not enabled:
            raise
        dbutils.jobs.taskValues.set(key="cdf_recovery_request", value=error.request)
        dbutils.jobs.taskValues.set(key="recovery_required", value=True)
        return _pending_result(error.request)
    if enabled:
        dbutils.jobs.taskValues.set(key="scoring_result", value=_compact_result(payload))
        dbutils.jobs.taskValues.set(key="recovery_required", value=False)
    return payload


def _enabled_config(dbutils: Any) -> tuple[dict[str, str], dict[str, Any]]:
    """Refuse stale recovery nodes after the project opts out."""
    values = dbutils.widgets.getAll()
    config = read_notebook_config(values)
    if config.get("auto_rebuild_on_cdf_expiry") is not True:
        raise ValueError("CDF recovery must be enabled in the deployed project.")
    return values, config


def _recovery_needed(dbutils: Any) -> bool:
    """Require a real predecessor decision instead of coercing arbitrary task values."""
    needed = dbutils.jobs.taskValues.get(taskKey="score", key="recovery_required")
    if type(needed) is not bool:
        raise ValueError("Score task recovery_required must be a boolean.")
    return needed


def _recover(
    spark: Any,
    config: dict[str, Any],
    values: dict[str, str],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Reload only the concrete saved model selected by the failed CDF read."""
    model_set = config.get("training_layout") == "multi_target"
    expected = "model_set" if model_set else "single_model"
    if request["layout"] != expected:
        raise ValueError("CDF recovery layout differs from the deployed project.")
    if model_set:
        from ...model_sets.model_set_project import score_model_set_payload  # noqa: PLC0415

        return score_model_set_payload(spark, config, values, recovery_request=request)
    outcome = run_bundle_action(
        spark,
        config,
        {"score_model_version": request["model_version"]},
        task_role="score",
        recovery_request=request,
    )
    return {
        **asdict(outcome),
        "source_table": config["score_source_table"],
        "prediction_table": request["target_table"],
    }


def _render(payload: dict[str, Any]) -> str:
    """Use the existing single-model or model-set HTML summary."""
    if "model_set_name" in payload:
        from ...model_sets.model_set_project import render_model_set_result  # noqa: PLC0415

        return render_model_set_result(payload)
    return render_bundle_output(payload)


def run_cdf_recovery_notebook(
    spark: Any,
    dbutils: Any,
    *,
    display_html: Callable[[str], Any] | None = None,
    exit_notebook: bool = True,
) -> str:
    """Recompute a pinned full snapshot in the visible, explicitly enabled recovery node."""
    values, config = _enabled_config(dbutils)
    validate_monitoring_settings(values, config)
    if not _recovery_needed(dbutils):
        raise ValueError("Score task did not request CDF recovery.")
    request = validate_recovery_request(
        dbutils.jobs.taskValues.get(taskKey="score", key="cdf_recovery_request")
    )
    payload = _recover(spark, config, values, request)
    registration = register_scoring_monitor(spark, config, values, payload)
    if registration is not None:
        payload["monitoring"] = registration
        publish_monitoring_request(dbutils, payload)
    dbutils.jobs.taskValues.set(key="scoring_result", value=_compact_result(payload))
    return notebook_output(
        payload,
        dbutils,
        render=_render,
        display_html=display_html,
        exit_notebook=exit_notebook,
    )


def run_scoring_report_notebook(
    spark: Any,
    dbutils: Any,
    *,
    display_html: Callable[[str], Any] | None = None,
    exit_notebook: bool = True,
) -> str:
    """Render the successful branch only, without querying skipped tasks or writing data."""
    _enabled_config(dbutils)
    task = "recover_predictions" if _recovery_needed(dbutils) else "score"
    payload = dbutils.jobs.taskValues.get(taskKey=task, key="scoring_result")
    if not isinstance(payload, dict):
        raise ValueError("Scoring report requires a completed result object.")
    return notebook_output(
        payload,
        dbutils,
        render=_render,
        display_html=display_html,
        exit_notebook=exit_notebook,
    )
