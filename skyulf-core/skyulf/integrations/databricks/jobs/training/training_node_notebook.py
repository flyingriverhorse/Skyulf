"""Thin notebook boundaries for named training tasks and independent report leaves."""

import json
from typing import Any

from ...lifecycle._lifecycle_state import LifecycleContext, PhaseStore
from ...model_sets.model_set_project import (
    capture_set_rules,
    load_project_model_set,
    render_model_set_result,
)
from ...observability.reports.training_node_output import render_training_node
from ..shared.job_output import render_lifecycle_output
from ..shared.job_runtime import (
    lifecycle_widget_context,
    notebook_output,
    operator_options,
    read_notebook_config,
    saved_notebook_request,
)
from .branch_notebook import load_training_branch_configs, render_branch_result
from .branch_tasks import (
    initialize_branch_training,
    run_branch_training,
)
from .training_nodes import run_competition_training


def _saved(values: dict) -> tuple[PhaseStore, dict]:
    """Resolve immutable invocation evidence before dispatching a named task."""
    options: dict[str, Any] = {
        **saved_notebook_request(values),
        "context": LifecycleContext(**lifecycle_widget_context(values)),
    }
    store = PhaseStore(options["tracking_uri"], options["context"])
    store.bind(options["reference"])
    return store, options


def _emit(dbutils: Any, outcome: Any, display_html: Any, render: Any) -> str:
    """Publish the small verified reference separately from readable notebook output."""
    dbutils.jobs.taskValues.set(
        key="reference_json", value=json.dumps(outcome.reference, sort_keys=True)
    )
    return notebook_output(
        outcome.output, dbutils, render=render, display_html=display_html, exit_notebook=False
    )


def run_model_training_notebook(spark: Any, dbutils: Any, *, display_html: Any = None) -> str:
    """Fit the named frozen candidate or branch and display training evidence only."""
    values = dbutils.widgets.getAll()
    store, options = _saved(values)
    train = run_branch_training if "branch_plan" in store.request else run_competition_training
    outcome = train(spark, name=values["model_key"], **options)
    return _emit(
        dbutils,
        outcome,
        display_html,
        lambda payload: render_training_node(store.client, payload["run_id"], payload),
    )


def run_initialize_models_notebook(spark: Any, dbutils: Any, *, display_html: Any = None) -> str:
    """Read editable multi-model configuration once before the graph branches."""
    values = dbutils.widgets.getAll()
    context = LifecycleContext(**lifecycle_widget_context(values))
    action = values.get("lifecycle_action", "train")
    operator_options(action, values)
    settings = load_project_model_set(values)
    source = ""
    if action == "train":
        configs = load_training_branch_configs(values)
        validate_model_task_names(values, set(configs))
        if settings is not None:
            settings, source = capture_set_rules(values, settings)
    else:
        if settings is None:
            raise ValueError("Enable a model set before running model-set operations.")
        configs = {"operator": read_notebook_config(values)}
    tracking_uri = next(iter(configs.values())).get("tracking_uri", "databricks")
    outcome = initialize_branch_training(
        spark,
        configs=configs,
        settings=settings,
        composition_source=source,
        context=context,
        tracking_uri=tracking_uri,
        experiment_name=values["experiment_name"],
        action=action,
        operator_values=values,
        score_handoff=read_notebook_config(values)["score_handoff"],
    )
    dbutils.jobs.taskValues.set(key="tracking_uri", value=tracking_uri)
    dbutils.jobs.taskValues.set(
        key="training_requested", value=outcome.output["training_requested"]
    )
    return _emit(
        dbutils,
        outcome,
        display_html,
        lambda payload: render_lifecycle_output("initialize", payload),
    )


def run_model_set_stage_notebook(
    spark: Any, dbutils: Any, *, phase: str, display_html: Any = None
) -> str:
    """Show each real set registration, quality check or alias decision in its own node."""
    from ...model_sets.model_set_stages import run_model_set_phase  # noqa: PLC0415

    store, options = _saved(dbutils.widgets.getAll())
    outcome = run_model_set_phase(spark, phase=phase, **options)
    render = (
        render_model_set_result if store.request["settings"] is not None else render_branch_result
    )
    return _emit(dbutils, outcome, display_html, render)


def run_models_report_notebook(dbutils: Any, *, display_html: Any = None) -> str:
    """Close a failed branch invocation even when the complete-only join was skipped."""
    values = dbutils.widgets.getAll()
    store, _ = _saved(values)
    if values.get("decision_result_state", "").lower() != "success":
        store.client.set_terminated(store.run_id, status="FAILED")
        store.run.set_tags({"skyulf.training.status": "failed"})
        raise ValueError(
            "Multi-model training or model-set action failed; inspect the task outputs."
        )
    phase = "model_decision" if store.request["action"] == "train" else "operator"
    output = dict(store.receipt(phase)["output"])
    output["score_requested"] = _set_score_requested(store.request, output)
    dbutils.jobs.taskValues.set(key="score_requested", value=output["score_requested"])
    return notebook_output(
        output,
        dbutils,
        render=render_model_set_result,
        display_html=display_html,
        exit_notebook=False,
    )


def _set_score_requested(request: dict, output: dict) -> bool:
    """Use the common handoff policy only for a verified whole-set champion transition."""
    from ....mlflow.lifecycle.promotion import AliasChangeReceipt  # noqa: PLC0415
    from ...lifecycle.workflow import build_bundle_result  # noqa: PLC0415

    action = request["action"]
    receipt = output.get("alias_change" if action == "train" else "receipt")
    if action not in {"train", "approve", "rollback"} or receipt is None:
        return False
    transition = AliasChangeReceipt(**receipt)
    if transition.model_name != request["settings"]["model_name"] or transition.alias != "champion":
        raise ValueError("Score handoff requires the model set's champion transition.")
    config = {**request["config"], "score_handoff": request.get("score_handoff", "disabled")}
    return build_bundle_result(config, action, transition).score_requested


def run_shap_notebook(dbutils: Any, *, display_html: Any = None) -> str:
    """Display only the exact predecessor training run, without registering or selecting."""
    from ...observability.reports.explanation_report import (  # noqa: PLC0415
        display_explanation_reports,
    )

    store, options = _saved(dbutils.widgets.getAll())
    phase = options["reference"]["phase"]
    if phase != "train" and not phase.startswith(("candidate_", "branch_")):
        raise ValueError("SHAP report requires a completed training task reference.")
    output = store.receipt(phase)["output"]
    run_id = output.get("run_id", store.run_id)
    paths = {item.path for item in store.client.list_artifacts(run_id)}
    payload = {"run_id": run_id, "status": "disabled"}
    if "explanations.json" in paths:
        from ...observability.reports.training_node_output import (  # noqa: PLC0415 - preserve lazy dependency boundary
            training_report_document,
        )

        payload.update(training_report_document(store.client, run_id, "explanations.json"))
        if display_html is not None:
            display_explanation_reports({"run_id": run_id}, store.client, display_html)
    elif display_html is not None:
        display_html("<h2>SHAP report</h2><p>SHAP was disabled for this model.</p>")
    # Keep exit small; detailed evidence and charts live in MLflow and the display cell.
    return json.dumps(
        {key: payload.get(key) for key in ("run_id", "status", "report_status", "reason")}
    )


def validate_model_task_names(values: dict, names: set[str]) -> None:
    """Reject an outdated deployed graph before training any changed model list."""
    encoded = values.get("model_keys_json")
    if encoded is None:
        return
    deployed = json.loads(encoded)
    if not isinstance(deployed, list) or len(deployed) != len(names) or set(deployed) != names:
        raise ValueError(
            "Model list differs from the job graph; run src/tools/refresh_training_graph.py and redeploy."
        )
