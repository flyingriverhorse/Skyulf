"""Independent reporting leaf bound to a completed training invocation."""

import json
from typing import Any

from ..lifecycle._lifecycle_state import LifecycleContext, LifecyclePhaseResult, PhaseStore
from ..observability.charts.evaluation_chart_data import chart_settings
from ..observability.charts.evaluation_chart_runs import chart_run
from .shared.job_runtime import lifecycle_widget_context, saved_notebook_request


def generate_evaluation_charts(
    *, context: LifecycleContext, tracking_uri: str, reference: dict
) -> LifecyclePhaseResult:
    """Generate optional images without changing training status, aliases or full-holdout metrics."""
    context.validate()
    store = PhaseStore(tracking_uri, context)
    store.bind(reference)
    if (
        reference["phase"] != store.predecessor("generate_charts")
        or store.request["action"] != "train"
    ):
        raise ValueError("Evaluation charts require the completed evaluation task reference.")
    settings = chart_settings(store.request["config"].get("evaluation_charts"))
    store.begin("generate_charts")
    try:
        output = {
            "status": "disabled",
            "run_id": store.run_id,
            "experiment_id": store.client.get_run(store.run_id).info.experiment_id,
            "reports": {},
            "image_keys": [],
        }
        if settings is not None:
            output.update(_generate(store, tracking_uri), status="complete")
        store.log("evaluation_charts/report.json", output)
        store.client.set_tag(store.run_id, "skyulf.charts.status", output["status"])
        return store.complete("generate_charts", output, reference)
    except Exception:  # noqa: BLE001 - chart failure must not rewrite a successful lifecycle
        store.client.set_tag(store.run_id, "skyulf.charts.status", "failed")
        raise


def _generate(store: PhaseStore, tracking_uri: str) -> dict:
    """Select the report strategy from the immutable invocation, not editable project files."""
    with chart_run(store.client, store.run_id) as destination:
        output = _generate_in_run(store, tracking_uri, destination)
    return {**output, "charts_run_id": destination}


def _generate_in_run(store: PhaseStore, tracking_uri: str, destination: str) -> dict:
    """Keep chart destinations separate from source model and metric identities."""
    from ..observability.charts.evaluation_chart_report import (  # noqa: PLC0415 - plotting remains optional
        competition_charts,
        model_set_charts,
        publish_figures,
        report_model,
    )

    if "branch_plan" in store.request:
        components = store.receipt("register_model_set")["output"]["components"]
        reports = {}
        for name, identity in components.items():
            verified = store.receipt(f"branch_{name}")["output"]
            if {key: value for key, value in verified.items() if key != "name"} != identity:
                raise ValueError("Chart component differs from its completed training receipt.")
            with chart_run(store.client, identity["run_id"]) as component_destination:
                reports[name] = report_model(
                    store.client,
                    tracking_uri,
                    identity,
                    destination=component_destination,
                    prefix=f"model_set_{name}",
                )
        keys = publish_figures(
            store.client,
            destination,
            model_set_charts(components),
            prefix="model_set",
            caption="Each component has its own target, units and heldout population.",
        )
        return {"layout": "multi_target", "reports": reports, "image_keys": keys}
    identity, selection = _single_identity(store)
    prefix = "competition_winner" if selection else "single"
    report = report_model(
        store.client, tracking_uri, identity, destination=destination, prefix=prefix
    )
    keys = list(report["image_keys"])
    if selection:
        keys += publish_figures(
            store.client,
            destination,
            competition_charts(selection),
            prefix="competition",
            caption="Winner selected using training CV; only its final holdout is plotted.",
        )
    return {
        "layout": "model_competition" if selection else "single_model",
        "reports": {prefix: report},
        "image_keys": keys,
    }


def _single_identity(store: PhaseStore) -> tuple[dict, dict | None]:
    """Bind parent evaluation outputs to the verified single fit or competition winner."""
    from ..training.competition.local_competition import selected_request  # noqa: PLC0415
    from .lifecycle.lifecycle_tasks import phase_training_spec  # noqa: PLC0415

    config, _ = selected_request(store)
    training = store.receipt("train")["output"]
    spec = phase_training_spec(training["spec"], config["pipeline"].get("project_python_source"))
    identity = {
        "run_id": store.run_id,
        "model_digest": training["model_digest"],
        "dataset_id": spec.dataset_id,
        "holdout_key_sha256": spec.holdout_key_sha256,
        "holdout_rows": training["holdout_rows"],
    }
    selection = training.get("competition")
    if selection:
        winner = store.receipt(f"candidate_{selection['winner']}")["output"]
        if winner["training"]["model_digest"] != identity["model_digest"]:
            raise ValueError("Chart winner differs from the selected model.")
        # Evaluation outputs belong to the parent after the winning model is adopted.
    return identity, selection


def render_chart_report(payload: dict, *, workspace_host: str = "") -> str:
    """List only created image keys and their MLflow Charts locations as plain text."""
    if payload["status"] == "disabled":
        return "Evaluation charts are disabled."
    destinations = {payload.get("charts_run_id", payload["run_id"]): list(payload["image_keys"])}
    for report in payload["reports"].values():
        keys = destinations.setdefault(report["destination_run_id"], [])
        keys.extend(key for key in report["image_keys"] if key not in keys)
    sections = []
    for run_id, keys in destinations.items():
        if not keys:
            continue
        location = f"MLflow run {run_id} > Charts"
        if workspace_host:
            location = (
                f"MLflow Charts: {workspace_host.rstrip('/')}/ml/experiments/"
                f"{payload['experiment_id']}/runs/{run_id}/model-metrics"
            )
        sections.append("\n".join([location, *(f"- {key}" for key in keys)]))
    return "\n\n".join(sections) or "No evaluation charts were generated."


def run_evaluation_charts_notebook(dbutils: Any) -> str:
    """Keep the notebook boundary thin and publish only small references to task values."""
    values = dbutils.widgets.getAll()
    outcome = generate_evaluation_charts(
        context=LifecycleContext(**lifecycle_widget_context(values)),
        **saved_notebook_request(values),
    )
    dbutils.jobs.taskValues.set(
        key="reference_json", value=json.dumps(outcome.reference, sort_keys=True)
    )
    print(render_chart_report(outcome.output, workspace_host=values.get("workspace_host", "")))
    return json.dumps(outcome.output, default=str, allow_nan=False)
