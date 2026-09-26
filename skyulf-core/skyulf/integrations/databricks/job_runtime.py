"""Adapt Bundle parameters and task values to the existing local workflow services."""

import json
import logging
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..mlflow.promotion import AliasChangeReceipt
from .job_output import render_bundle_output
from .local_approval import resolve_candidate_comparison_digest
from .local_retraining import LocalCandidateResult
from .local_workflow import (
    AutoTrainingOutcome,
    _workflow_policies,
    resolve_target_config,
    run_action,
)
from .workflow_config import validate_deployed_contract, validate_workflow_config

_OPERATOR_FIELDS = {
    "candidate_version",
    "comparison_sha256",
    "expected_champion_version",
    "rejection_reason",
    "promotion_receipt_json",
}


@dataclass(frozen=True, slots=True)
class BundleActionResult:
    """Expose Core output, optional score handoff and copyable operator inputs."""

    action: str
    result: Any
    score_requested: bool
    next_actions: dict[str, dict[str, str]]


def _version(value: str, *, allow_none: bool = False) -> str | None:
    """Require a concrete version or an explicitly chosen bootstrap sentinel."""
    if allow_none and value == "none":
        return None
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError("Expected a concrete model version; use none only for bootstrap.")
    return value


def _operator_options(action: str, values: dict[str, str]) -> dict[str, Any]:
    """Reject incomplete and irrelevant operator inputs before any Core side effects."""
    allowed = {
        "approve": {"candidate_version", "comparison_sha256", "expected_champion_version"},
        "reject": {
            "candidate_version",
            "comparison_sha256",
            "expected_champion_version",
            "rejection_reason",
        },
        "rollback": {"promotion_receipt_json", "expected_champion_version"},
    }.get(action, set())
    if any(values.get(key, "") for key in _OPERATOR_FIELDS - allowed):
        raise ValueError(f"Unexpected operator parameters for {action}.")
    if not allowed:
        return {}
    expected = _version(
        values.get("expected_champion_version", ""), allow_none=action != "rollback"
    )
    options: dict[str, Any] = {"expected_champion_version": expected}
    if action == "rollback":
        try:
            payload = json.loads(values.get("promotion_receipt_json", ""))
            if not isinstance(payload, dict) or any(
                value is not None and not isinstance(value, str) for value in payload.values()
            ):
                raise ValueError
            receipt = AliasChangeReceipt(**payload)
            if receipt.kind != "promotion" or receipt.alias != "champion":
                raise ValueError
            if receipt.new_version != expected or receipt.prior_version is None:
                raise ValueError
            _version(receipt.prior_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Rollback needs a complete promotion_receipt_json matching the expected champion."
            ) from exc
        options["promotion_receipt"] = receipt
        return options
    options["candidate_version"] = _version(values.get("candidate_version", ""))
    digest = values.get("comparison_sha256", "")
    if digest and not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("comparison_sha256 must be empty or an exact 64-character digest.")
    options["comparison_sha256"] = digest
    if action == "reject":
        reason = values.get("rejection_reason", "")
        if not reason.strip() or len(reason.encode("utf-8")) > 256:
            raise ValueError("Rejection needs a reason of at most 256 UTF-8 bytes.")
        options["rejection_reason"] = reason
    return options


def _next_actions(result: Any, policy: str) -> dict[str, dict[str, str]]:
    """Expose the exact evidence accepted by existing Core approval and rollback APIs."""
    actions: dict[str, dict[str, str]] = {}
    candidate = result.candidate if isinstance(result, AutoTrainingOutcome) else result
    if isinstance(candidate, LocalCandidateResult) and policy == "manual_approval":
        for action in ("approve", "reject"):
            actions[action] = {
                "lifecycle_action": action,
                "candidate_version": candidate.model_version,
                "expected_champion_version": candidate.comparison.champion_version or "none",
            }
    receipt = result.alias_change if isinstance(result, AutoTrainingOutcome) else result
    if isinstance(receipt, AliasChangeReceipt) and receipt.kind == "promotion":
        actions["rollback"] = {
            "lifecycle_action": "rollback",
            "expected_champion_version": receipt.new_version,
            "promotion_receipt_json": json.dumps(
                asdict(receipt), sort_keys=True, separators=(",", ":")
            ),
        }
    return actions


def run_bundle_action(
    spark: Any,
    config: dict[str, Any],
    parameters: dict[str, str],
    *,
    task_role: str,
    experiment_name: str | None = None,
    artifact_path: str | Path | None = None,
) -> BundleActionResult:
    """Run one role-bound action; score handoff follows only a successful champion transition.

    Role is fixed by the deployed notebook, never by a run parameter. This is
    input isolation, not a substitute for workspace permissions or serialized
    writer ownership. A failed Core action produces no success task value.
    """
    if "score_model_selection" not in config or "promotion_policy" not in config:
        raise ValueError(
            "Regenerate the Bundle config and job graph with both independent policies."
        )
    _, policy = _workflow_policies(config)
    if config.get("score_handoff") not in {"disabled", "after_alias_change"}:
        raise ValueError("score_handoff must be disabled or after_alias_change.")
    for key in _OPERATOR_FIELDS | {
        "lifecycle_action",
        "task_role",
        "action",
        "score_model_version",
    }:
        if key in parameters and not isinstance(parameters[key], str):
            raise ValueError("Job parameters must be strings.")
    if parameters.get("task_role") or parameters.get("action"):
        raise ValueError("Notebook role and score action cannot be overridden by job parameters.")
    if task_role == "score":
        if parameters.get("lifecycle_action"):
            raise ValueError("Score job refuses lifecycle actions.")
        action = "score"
    elif task_role == "lifecycle":
        action = parameters.get("lifecycle_action", "")
        if action not in {"train", "train_monthly", "approve", "reject", "rollback"}:
            raise ValueError(
                "Lifecycle action must be train, train_monthly, approve, reject or rollback."
            )
    else:
        raise ValueError("Notebook task_role must be lifecycle or score.")
    override = parameters.get("score_model_version", "")
    if override:
        if action != "score":
            raise ValueError("score_model_version is only valid on the score job.")
        config = {
            **config,
            "score_model_selection": "pinned_version",
            "model_version": _version(override),
        }
    if "config_version" in config:
        config = validate_workflow_config(config, action=action)
    options = _operator_options(action, parameters)
    if action in {"approve", "reject"} and not options["comparison_sha256"]:
        options["comparison_sha256"] = resolve_candidate_comparison_digest(
            config, options["candidate_version"], action=action
        )
    if action in {"train", "train_monthly"}:
        options.update(experiment_name=experiment_name, artifact_path=artifact_path)
    result = run_action(spark, config, action, **options)
    receipt = result.alias_change if isinstance(result, AutoTrainingOutcome) else result
    score_requested = (
        config["score_handoff"] == "after_alias_change"
        and action in {"train", "train_monthly", "approve", "rollback"}
        and isinstance(receipt, AliasChangeReceipt)
        and receipt.kind in {"initial", "promotion", "rollback"}
    )
    return BundleActionResult(action, result, score_requested, _next_actions(result, policy))


def run_notebook(
    spark: Any,
    dbutils: Any,
    *,
    task_role: str,
    display_html: Callable[[str], Any] | None = None,
    exit_notebook: bool = True,
    preprocessing_path: str | Path | None = None,
) -> str:
    """Run the fixed role and render results, optionally deferring the notebook exit.

    Generated notebooks defer exit to a separate cell: Databricks otherwise
    replaces the readable report with the exit value in the same cell.
    A preprocessing path is relative to the config directory and used only for
    training. Scoring and operator actions retain the saved artifact's code.
    """
    values = dbutils.widgets.getAll()
    # Databricks pushes parent job parameters into Run Job children. The score
    # entrypoint ignores inherited lifecycle evidence and always dispatches score.
    # Keep role/action override guards; only lifecycle fields are discarded here.
    parameters = (
        {
            key: value
            for key, value in values.items()
            if key not in _OPERATOR_FIELDS | {"lifecycle_action"}
        }
        if task_role == "score"
        else values
    )
    config = json.loads(Path(values["config_path"]).read_text(encoding="utf-8"))
    required = {"training_table", "score_source_table", "prediction_table", "model_name"}
    if not isinstance(config, dict) or any(
        not isinstance(config.get(key), str) for key in required
    ):
        raise ValueError(
            "Workflow configuration must be an object with training_table, score_source_table, prediction_table and model_name bindings."
        )
    config = resolve_target_config(
        config,
        {
            name: values[name]
            for name in (
                "catalog",
                "input_schema",
                "output_schema",
                "metadata_schema",
                "resource_suffix",
            )
        },
    )
    validate_deployed_contract(config, parameters)
    if config.get("config_version") != 1:
        raise ValueError("config_version must be 1; migrate and regenerate/redeploy this Bundle.")
    if (
        preprocessing_path is not None
        and task_role == "lifecycle"
        and parameters.get("lifecycle_action") in {"train", "train_monthly"}
    ):
        from .project import load_project_workflow  # noqa: PLC0415 - project code is training-only

        config = load_project_workflow(
            config, Path(values["config_path"]).parent / preprocessing_path
        )
    with tempfile.TemporaryDirectory(prefix="skyulf-bundle-") as directory:
        outcome = run_bundle_action(
            spark,
            config,
            parameters,
            task_role=task_role,
            experiment_name=values.get("experiment_name"),
            artifact_path=Path(directory) / "artifact",
        )
    payload = asdict(outcome)
    output = json.dumps(payload, default=str, allow_nan=False)
    if task_role == "lifecycle":
        dbutils.jobs.taskValues.set(key="score_requested", value=outcome.score_requested)
    if display_html is not None:
        try:
            display_html(render_bundle_output(payload))
        except Exception:  # noqa: BLE001 - a display failure must not invite alias mutation retries
            logging.getLogger(__name__).warning("Readable output unavailable; see JSON result.")
            print(json.dumps(payload, indent=2, default=str, allow_nan=False))
    else:
        print(json.dumps(payload, indent=2, default=str, allow_nan=False))
    if exit_notebook:
        dbutils.notebook.exit(output)
    return output
