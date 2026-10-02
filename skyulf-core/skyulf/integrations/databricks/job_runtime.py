"""Adapt Bundle parameters and task values to the existing local workflow services."""

import json
import logging
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..mlflow.promotion import AliasChangeReceipt
from .job_output import render_bundle_output, render_lifecycle_output
from .local_approval import resolve_candidate_comparison_digest
from .local_workflow import BundleActionResult as BundleActionResult
from .local_workflow import (
    build_bundle_result,
    resolve_target_config,
    run_action,
    workflow_policies,
)
from .monitoring_registration import (
    publish_monitoring_request,
    register_scoring_monitor,
    validate_monitoring_settings,
)
from .workflow_config import validate_deployed_contract, validate_workflow_config

OPERATOR_FIELDS = {
    "candidate_version",
    "comparison_sha256",
    "expected_champion_version",
    "rejection_reason",
    "promotion_receipt_json",
}


def parse_model_version(value: str, *, allow_none: bool = False) -> str | None:
    """Require a concrete version or an explicitly chosen bootstrap sentinel."""
    if allow_none and value == "none":
        return None
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError("Expected a concrete model version; use none only for bootstrap.")
    return value


def operator_options(action: str, values: dict[str, str]) -> dict[str, Any]:
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
    if any(values.get(key, "") for key in OPERATOR_FIELDS - allowed):
        raise ValueError(f"Unexpected operator parameters for {action}.")
    if not allowed:
        return {}
    expected = parse_model_version(
        values.get("expected_champion_version", ""), allow_none=action != "rollback"
    )
    options: dict[str, Any] = {"expected_champion_version": expected}
    if action == "rollback":
        receipt = _rollback_receipt(values, expected)
        options["promotion_receipt"] = receipt
        return options
    options["candidate_version"] = parse_model_version(values.get("candidate_version", ""))
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


def _rollback_receipt(values: dict[str, str], expected: str | None) -> AliasChangeReceipt:
    """Parse a complete promotion receipt and preserve a single operator-facing failure."""
    try:
        payload = json.loads(values.get("promotion_receipt_json", ""))
        if not isinstance(payload, dict) or any(
            value is not None and not isinstance(value, str) for value in payload.values()
        ):
            raise ValueError
        receipt = AliasChangeReceipt(**payload)
        _validate_rollback_receipt(receipt, expected)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Rollback needs a complete promotion_receipt_json matching the expected champion."
        ) from exc
    return receipt


def _validate_rollback_receipt(receipt: AliasChangeReceipt, expected: str | None) -> None:
    """Require a champion promotion matching both sides of the rollback guard."""
    if receipt.kind != "promotion" or receipt.alias != "champion":
        raise ValueError
    if receipt.new_version != expected or receipt.prior_version is None:
        raise ValueError
    parse_model_version(receipt.prior_version)


def run_bundle_action(
    spark: Any,
    config: dict[str, Any],
    parameters: dict[str, str],
    *,
    task_role: str,
    experiment_name: str | None = None,
    artifact_path: str | Path | None = None,
    recovery_request: dict[str, Any] | None = None,
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
    _, policy = workflow_policies(config)
    if config.get("score_handoff") not in {"disabled", "after_alias_change"}:
        raise ValueError("score_handoff must be disabled or after_alias_change.")
    validate_job_parameters(parameters)
    action = _role_action(task_role, parameters)
    override = parameters.get("score_model_version", "")
    if override:
        if action != "score":
            raise ValueError("score_model_version is only valid on the score job.")
        config = {
            **config,
            "score_model_selection": "pinned_version",
            "model_version": parse_model_version(override),
        }
    if "config_version" in config:
        config = validate_workflow_config(config, action=action)
    options = _bundle_action_options(
        action, parameters, config, experiment_name, artifact_path, recovery_request
    )
    result = run_action(spark, config, action, **options)
    return build_bundle_result(config, action, result)


def _bundle_action_options(
    action: str,
    parameters: dict[str, str],
    config: dict[str, Any],
    experiment_name: str | None,
    artifact_path: str | Path | None,
    recovery_request: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate recovery scope before resolving lifecycle evidence or training settings."""
    if recovery_request is not None and action != "score":
        raise ValueError("CDF recovery is only available for scoring.")
    options = operator_options(action, parameters)
    if recovery_request is not None:
        options["recovery_request"] = recovery_request
    if action in {"approve", "reject"} and not options["comparison_sha256"]:
        options["comparison_sha256"] = resolve_candidate_comparison_digest(
            config, options["candidate_version"], action=action
        )
    if action == "train":
        options.update(experiment_name=experiment_name, artifact_path=artifact_path)
    return options


def validate_job_parameters(parameters: dict[str, str]) -> None:
    """Reject malformed or role-overriding job inputs before selecting an action."""
    for key in OPERATOR_FIELDS | {
        "lifecycle_action",
        "task_role",
        "action",
        "score_model_version",
    }:
        if key in parameters and not isinstance(parameters[key], str):
            raise ValueError("Job parameters must be strings.")
    if parameters.get("task_role") or parameters.get("action"):
        raise ValueError("Notebook role and score action cannot be overridden by job parameters.")


def _role_action(task_role: str, parameters: dict[str, str]) -> str:
    """Map a deployed notebook role to its allowed requested action."""
    if task_role == "score":
        if parameters.get("lifecycle_action"):
            raise ValueError("Score job refuses lifecycle actions.")
        action = "score"
    elif task_role == "lifecycle":
        action = parameters.get("lifecycle_action", "")
        if action not in {"train", "approve", "reject", "rollback"}:
            raise ValueError("Lifecycle action must be train, approve, reject or rollback.")
    else:
        raise ValueError("Notebook task_role must be lifecycle or score.")
    return action


def read_notebook_config(values: dict[str, str]) -> dict[str, Any]:
    """Load and bind the project once at the notebook's configuration boundary."""
    config = json.loads(Path(values["config_path"]).read_text(encoding="utf-8"))
    required = {"training_table", "score_source_table", "prediction_table", "model_name"}
    if not isinstance(config, dict) or any(
        not isinstance(config.get(key), str) for key in required
    ):
        raise ValueError(
            "Workflow configuration must be an object with training_table, "
            "score_source_table, prediction_table and model_name bindings."
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
    validate_deployed_contract(config, values)
    if config.get("config_version") != 1:
        raise ValueError("config_version must be 1; migrate and regenerate/redeploy this Bundle.")
    return config


def notebook_output(
    payload: dict[str, Any],
    dbutils: Any,
    *,
    render: Callable[[dict[str, Any]], str],
    display_html: Callable[[str], Any] | None,
    exit_notebook: bool,
    explanation_tracking_uri: str | None = None,
) -> str:
    """Publish readable and machine output without retrying completed side effects."""
    output = json.dumps(payload, default=str, allow_nan=False)
    if display_html is not None:
        try:
            display_html(render(payload))
        except Exception:  # noqa: BLE001 - display failure must not invite mutation retries
            logging.getLogger(__name__).warning("Readable output unavailable; see JSON result.")
            print(json.dumps(payload, indent=2, default=str, allow_nan=False))
    else:
        print(json.dumps(payload, indent=2, default=str, allow_nan=False))
    if explanation_tracking_uri and display_html is not None:
        _display_training_explanations(payload, explanation_tracking_uri, display_html)
    if exit_notebook:
        dbutils.notebook.exit(output)
    return output


def _display_training_explanations(
    payload: dict, tracking_uri: str, display_html: Callable
) -> None:
    """Keep optional report retrieval outside the committed training operation."""
    from skyulf.integrations.mlflow._client import (  # noqa: PLC0415 - preserve lazy dependency boundary
        make_tracking_client,
    )

    from .explanation_report import display_explanation_reports  # noqa: PLC0415

    try:
        client = make_tracking_client(tracking_uri)
        display_explanation_reports(payload, client, display_html)
    except Exception:  # noqa: BLE001 - never retry training because optional display failed
        logging.getLogger(__name__).warning("SHAP reports unavailable; inspect MLflow artifacts.")


def lifecycle_widget_context(values: dict[str, str]) -> dict[str, Any]:
    """Reject unresolved invocation values and unsupported repairs before loading files."""
    if values.get("workflow_contract") not in {"2", "3"}:
        raise ValueError(
            "Lifecycle tasks require graph contract 2 or 3; regenerate/redeploy together."
        )
    if any(values.get(key) for key in ("phase", "task_role", "action", "score_model_version")):
        raise ValueError("Notebook phase and lifecycle role cannot be overridden by parameters.")
    for key in ("job_id", "job_run_id"):
        if not isinstance(values.get(key), str) or not re.fullmatch(r"[1-9][0-9]*", values[key]):
            raise ValueError(f"{key} must be a resolved Databricks job identity.")
    if values.get("repair_count") != "0" or values.get("execution_count") != "1":
        raise ValueError(
            "Lifecycle repair/retry is unsupported. Inspect prior effects before starting a fresh run."
        )
    return {
        "job_id": values["job_id"],
        "job_run_id": values["job_run_id"],
        "repair_count": 0,
        "execution_count": 1,
    }


def _prepared_notebook_request(
    values: dict[str, str], preprocessing_path: str | Path | None
) -> dict[str, Any]:
    """Validate a new invocation and freeze project Python only for training."""
    action = values.get("lifecycle_action", "")
    if action not in {"train", "approve", "reject", "rollback"}:
        raise ValueError("Lifecycle action must be train, approve, reject or rollback.")
    config = read_notebook_config(values)
    if action == "train" and preprocessing_path is not None:
        from .project import load_project_workflow  # noqa: PLC0415 - training-only project code

        config = load_project_workflow(
            config, Path(values["config_path"]).parent / preprocessing_path
        )
    config = validate_workflow_config(config, action=action)
    validate_monitoring_settings(values, config)
    if action == "train" and "competition" in config:
        from .training_node_notebook import validate_model_task_names  # noqa: PLC0415

        validate_model_task_names(values, set(config["competition"]["candidates"]))
    options = operator_options(action, values)
    if action in {"approve", "reject"} and not options["comparison_sha256"]:
        options["comparison_sha256"] = resolve_candidate_comparison_digest(
            config, options["candidate_version"], action=action
        )
    return {
        "config": config,
        "action": action,
        "operator_options": options,
        "experiment_name": values.get("experiment_name"),
        "tracking_uri": config.get("tracking_uri", "databricks"),
    }


def saved_notebook_request(values: dict[str, str]) -> dict[str, Any]:
    """Validate the frozen predecessor reference and prepared tracking URI."""
    try:
        reference = json.loads(values["reference_json"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Lifecycle task needs its saved predecessor reference.") from exc
    if not isinstance(reference, dict):
        raise ValueError("Lifecycle predecessor reference must be a JSON object.")
    tracking_uri = values.get("tracking_uri")
    if not isinstance(tracking_uri, str) or not tracking_uri or "{{" in tracking_uri:
        raise ValueError("Lifecycle task needs the prepared tracking URI.")
    return {"reference": reference, "tracking_uri": tracking_uri}


def _validate_notebook_phase_contract(phase: str, values: dict[str, str]) -> None:
    """Reject mismatched old/new notebooks before preparing or reading saved state."""
    new_phases = {
        "initialize",
        "load_data",
        "prepare_dataset",
        "select_best_model",
        "model_decision",
    }
    old_phases = {"prepare", "train_register", "compare_decide", "operator"}
    contract = values["workflow_contract"]
    if (phase in new_phases and contract != "3") or (phase in old_phases and contract != "2"):
        raise ValueError("Notebook and job graph contract differ; regenerate/redeploy together.")


def _notebook_task_states(phase: str, values: dict[str, str]) -> dict[str, str] | None:
    """Keep graph-specific completion states separate from the pinned action."""
    if phase != "complete":
        return None
    if values["workflow_contract"] == "3":
        return {"decision": values.get("decision_result_state", "")}
    return {
        "training": values.get("training_result_state", ""),
        "operator": values.get("operator_result_state", ""),
    }


def run_lifecycle_notebook(
    spark: Any,
    dbutils: Any,
    *,
    phase: str,
    display_html: Callable[[str], Any] | None = None,
    exit_notebook: bool = True,
    preprocessing_path: str | Path | None = None,
    separate_shap: bool = False,
) -> str:
    """Execute a fixed lifecycle phase using durable evidence from its predecessor.

    Only prepare/initialize read editable configuration and project Python. Later tasks
    receive references from this job invocation and load the frozen MLflow
    evidence. Notebook metadata is a misuse guard, not workspace authorization.
    """
    values = dbutils.widgets.getAll()
    context_values = lifecycle_widget_context(values)
    _validate_notebook_phase_contract(phase, values)
    from .lifecycle_tasks import (  # noqa: PLC0415 - shared runtime helpers avoid a module cycle
        LifecycleContext,
        run_lifecycle_phase,
    )

    options = (
        _prepared_notebook_request(values, preprocessing_path)
        if phase in {"prepare", "initialize"}
        else saved_notebook_request(values)
    )
    task_states = _notebook_task_states(phase, values)
    outcome = run_lifecycle_phase(
        spark,
        phase=phase,
        context=LifecycleContext(**context_values),
        task_states=task_states,
        **options,
    )
    dbutils.jobs.taskValues.set(
        key="reference_json", value=json.dumps(outcome.reference, sort_keys=True)
    )
    if phase in {"prepare", "initialize"}:
        dbutils.jobs.taskValues.set(key="tracking_uri", value=options["tracking_uri"])
        dbutils.jobs.taskValues.set(
            key="training_requested", value=outcome.output["training_requested"]
        )
    if phase in {"result", "complete"}:
        dbutils.jobs.taskValues.set(key="score_requested", value=outcome.output["score_requested"])
    render = _lifecycle_notebook_renderer(phase, outcome, options["tracking_uri"], separate_shap)
    return notebook_output(
        outcome.output,
        dbutils,
        render=render,
        display_html=display_html,
        exit_notebook=exit_notebook,
        explanation_tracking_uri=(
            options["tracking_uri"]
            if phase in {"train", "train_register"}
            and not separate_shap
            and outcome.output.get("explanations", {}).get("status") in {"completed", "unavailable"}
            else None
        ),
    )


def _lifecycle_notebook_renderer(
    phase: str, outcome: Any, tracking_uri: str, separate_shap: bool
) -> Callable:
    """Keep training settings visible when SHAP is displayed by a separate task."""
    if phase in {"result", "complete"}:
        return render_bundle_output
    if phase == "select_best_model" and outcome.output.get("selection_mode") == "single_candidate":
        return lambda payload: render_lifecycle_output("validate_model", payload)
    if phase == "train" and separate_shap:
        from skyulf.integrations.mlflow._client import (  # noqa: PLC0415 - preserve lazy dependency boundary
            make_tracking_client,
        )

        from .training_node_output import render_training_node  # noqa: PLC0415

        client = make_tracking_client(tracking_uri)
        return lambda payload: render_training_node(
            client, outcome.reference["run_id"], {"training": payload}
        )
    return lambda payload: render_lifecycle_output(phase, payload)


def run_score_notebook(
    spark: Any,
    dbutils: Any,
    *,
    display_html: Callable[[str], Any] | None = None,
    exit_notebook: bool = True,
) -> str:
    """Score with a fixed role, ignoring inherited parent lifecycle evidence.

    Generated notebooks defer exit to a separate cell: Databricks otherwise
    replaces the readable report with the exit value in the same cell.
    Scoring uses saved model code and creates no training artifact directory.
    """
    from .scoring_recovery import run_scoring_step  # noqa: PLC0415 - shared notebook routing

    values = dbutils.widgets.getAll()
    # Databricks pushes parent job parameters into Run Job children. The score
    # entrypoint ignores inherited lifecycle evidence and always dispatches score.
    # Keep role/action override guards; only lifecycle fields are discarded here.
    parameters = {
        key: value
        for key, value in values.items()
        if key not in OPERATOR_FIELDS | {"lifecycle_action"}
    }
    config = read_notebook_config(values)

    def score() -> dict[str, Any]:
        """Keep the original outcome intact while allowing typed recovery routing."""
        validate_monitoring_settings(values, config)
        outcome = run_bundle_action(spark, config, parameters, task_role="score")
        payload = {
            **asdict(outcome),
            "source_table": config["score_source_table"],
            "prediction_table": config["prediction_table"],
        }
        registration = register_scoring_monitor(spark, config, values, payload)
        if registration is not None:
            payload["monitoring"] = registration
            publish_monitoring_request(dbutils, payload)
        return payload

    return notebook_output(
        run_scoring_step(config, dbutils, score),
        dbutils,
        render=render_bundle_output,
        display_html=display_html,
        exit_notebook=exit_notebook,
    )


def _run_legacy_lifecycle_notebook(
    spark: Any,
    dbutils: Any,
    *,
    display_html: Callable[[str], Any] | None,
    exit_notebook: bool,
    preprocessing_path: str | Path | None,
) -> str:
    """Preserve the sequential notebook API for callers outside the phased graph."""
    values = dbutils.widgets.getAll()
    config = read_notebook_config(values)
    if preprocessing_path is not None and values.get("lifecycle_action") == "train":
        from .project import load_project_workflow  # noqa: PLC0415 - project code is training-only

        config = load_project_workflow(
            config, Path(values["config_path"]).parent / preprocessing_path
        )
    with tempfile.TemporaryDirectory(prefix="skyulf-bundle-") as directory:
        outcome = run_bundle_action(
            spark,
            config,
            values,
            task_role="lifecycle",
            experiment_name=values.get("experiment_name"),
            artifact_path=Path(directory) / "artifact",
        )
    payload = asdict(outcome)
    dbutils.jobs.taskValues.set(key="score_requested", value=outcome.score_requested)
    return notebook_output(
        payload,
        dbutils,
        render=render_bundle_output,
        display_html=display_html,
        exit_notebook=exit_notebook,
        explanation_tracking_uri=(
            (config.get("tracking_uri") or "databricks")
            if outcome.action == "train" and config.get("pipeline", {}).get("explainability")
            else None
        ),
    )


def run_notebook(
    spark: Any,
    dbutils: Any,
    *,
    task_role: str,
    display_html: Callable[[str], Any] | None = None,
    exit_notebook: bool = True,
    preprocessing_path: str | Path | None = None,
) -> str:
    """Keep the role-based notebook API compatible for existing direct callers.

    Generated projects use run_score_notebook or run_lifecycle_notebook instead.
    The lifecycle role here retains sequential execution, not durable phases.
    preprocessing_path is relative to the config directory and only used for
    training; scoring and operator actions use the saved model's code.
    """
    if task_role == "score":
        return run_score_notebook(
            spark, dbutils, display_html=display_html, exit_notebook=exit_notebook
        )
    if task_role != "lifecycle":
        raise ValueError("Notebook task_role must be lifecycle or score.")
    return _run_legacy_lifecycle_notebook(
        spark,
        dbutils,
        display_html=display_html,
        exit_notebook=exit_notebook,
        preprocessing_path=preprocessing_path,
    )
