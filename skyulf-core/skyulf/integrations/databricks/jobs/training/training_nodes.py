"""Independent competition fits joined through the existing durable lifecycle."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from skyulf.integrations.mlflow.shared._client import require_mlflow

from ....mlflow.registration.registry import load_run_local_pipeline, packaged_artifact_path
from ...lifecycle._lifecycle_data import training_partitions
from ...lifecycle._lifecycle_state import LifecycleContext, LifecyclePhaseResult, PhaseStore
from ...observability.reports.explanation_report import copy_winner_explanations
from ...training.competition.competition_training import fit_training_pipeline
from ...training.competition.local_competition import choose_winner
from ...training.fitting import local_retraining as training
from ...training.shared.local_training_evidence import evidence_digest


def _candidate_phase(store: PhaseStore, name: str) -> str:
    """Use only the frozen candidate identifiers as durable task names."""
    if name not in store.request.get("competition", {}).get("candidates", {}):
        raise ValueError("Training node is not a candidate in this invocation.")
    return f"candidate_{name}"


def _fitted_output(fitted: Any, engine: str) -> dict[str, Any]:
    """Retain the same evidence used by serial registration without pickling task state."""
    return {
        "model_digest": fitted.artifact.manifest.pipeline_sha256,
        "project_source_sha256": fitted.artifact.manifest.project_source_sha256,
        "spec": training.training_spec_payload(fitted.spec, engine),
        "training_rows": fitted.training_rows,
        "holdout_rows": fitted.holdout_rows,
        "unavailable_labels": fitted.unavailable_labels,
        "tags": fitted.tags,
    }


def run_competition_training(
    spark: Any,
    *,
    name: str,
    context: LifecycleContext,
    tracking_uri: str,
    reference: dict[str, str],
) -> LifecyclePhaseResult:
    """Train exactly one pinned candidate; leave registration and selection to the join."""
    from ..lifecycle.lifecycle_tasks import (  # noqa: PLC0415 - lazy dependency  # noqa: PLC0415 - lazy dependency  # noqa: PLC0415 - lazy dependency
        phase_training_spec,
        record_phase_failure,
        validate_active_phase,
    )

    context.validate()
    store = PhaseStore(tracking_uri, context)
    store.bind(reference)
    phase = _candidate_phase(store, name)
    if reference["phase"] != "prepare_dataset":
        raise ValueError("Candidate training requires the prepared dataset reference.")
    validate_active_phase(store, "train")
    store.begin(phase)
    try:
        recipe = store.request["competition"]["candidates"][name]
        spec = phase_training_spec(
            store.request["spec"], recipe["pipeline"].get("project_python_source")
        )
        with TemporaryDirectory(prefix="skyulf-candidate-task-") as directory:
            fitted, row = fit_training_pipeline(
                spark,
                store,
                spec,
                training_partitions(store),
                Path(directory) / "model",
                name,
                recipe,
            )
            output = {
                "name": name,
                "run_id": row["run_id"],
                "evaluation": row,
                "training": _fitted_output(fitted, store.request["config"]["engine"]),
            }
        return store.complete(phase, output, reference)
    except BaseException:
        record_phase_failure(store, phase)
        raise


def _completed_candidates(store: PhaseStore) -> dict[str, dict]:
    """Require every expected fit to retain its own finished child run and recipe."""
    completed = {}
    for name, recipe in store.request["competition"]["candidates"].items():
        output = store.receipt(_candidate_phase(store, name))["output"]
        row = output["evaluation"]
        child = store.client.get_run(output["run_id"])
        if (
            output["name"] != name
            or row["candidate"] != name
            or row["run_id"] != child.info.run_id
            or row["model_uri"] != f"runs:/{child.info.run_id}/model"
            or child.info.status != "FINISHED"
            or child.data.tags.get("mlflow.parentRunId") != store.run_id
            or row["config_sha256"] != evidence_digest(recipe["effective_config"])
            or row["model_digest"] != output["training"]["model_digest"]
        ):
            raise ValueError("Candidate result differs from its pinned training task.")
        completed[name] = output
    return completed


def _adopt_model(store: PhaseStore, winner: dict, recipe: dict) -> str:
    """Repackage the original fitted bytes under the parent run without fitting again."""
    config = store.request["config"]
    row = winner["evaluation"]
    artifact = load_run_local_pipeline(
        row["model_uri"],
        digest=row["model_digest"],
        tracking_uri=config["tracking_uri"],
    )
    expected_config = recipe["effective_config"]
    if store.request["spec"].get("weight_column") is not None:
        prepared = store.receipt("prepare_dataset")["output"]
        expected_config = {
            **expected_config,
            "training_weights": prepared["holdout"]["attrs"]["training_weights"],
        }
    if artifact.pipeline.config != expected_config:
        raise ValueError("Winning model configuration differs from its frozen recipe.")
    with TemporaryDirectory(prefix="skyulf-winner-adoption-") as directory:
        package = Path(store.client.download_artifacts(winner["run_id"], "model", directory))
        model = require_mlflow().models.Model.load(package)
        path = packaged_artifact_path(package, model.flavors, "local_pipeline")
        return training.log_local_model(
            path, run_id=store.run_id, tracking_uri=config["tracking_uri"]
        )


def _copy_fit_evidence(store: PhaseStore, child_run_id: str) -> None:
    """Carry the winner's original receipts and experiment settings onto the parent."""
    child = store.client.get_run(child_run_id)
    store.run.log_params(child.data.params)
    store.run.log_metrics(child.data.metrics)
    with TemporaryDirectory(prefix="skyulf-winner-evidence-") as directory:
        for item in store.client.list_artifacts(child_run_id):
            if not item.is_dir:
                path = store.client.download_artifacts(child_run_id, item.path, directory)
                store.client.log_artifact(store.run_id, path)


def join_competition_training(store: PhaseStore, reference: dict[str, str]) -> LifecyclePhaseResult:
    """Verify a full fan-out and create the ordinary train receipt for existing phases."""
    from ..lifecycle.lifecycle_tasks import (  # noqa: PLC0415 - lazy dependency  # noqa: PLC0415 - lazy dependency  # noqa: PLC0415 - lazy dependency
        record_phase_failure,
        training_summary,
        validate_active_phase,
    )

    validate_active_phase(store, "train")
    completed = _completed_candidates(store)
    selection = choose_winner([value["evaluation"] for value in completed.values()], set(completed))
    store.begin("train")
    try:
        winner = completed[selection["winner"]]
        recipe = store.request["competition"]["candidates"][selection["winner"]]
        uri = _adopt_model(store, winner, recipe)
        _copy_fit_evidence(store, winner["run_id"])
        store.log("competition/selection.json", selection)
        if recipe["pipeline"].get("explainability"):
            copy_winner_explanations(store, selection)
        tags = {**winner["training"]["tags"], "competition_winner": selection["winner"]}
        store.run.set_tags({**tags, "competition_count": str(len(completed))})
        artifact = load_run_local_pipeline(
            uri,
            digest=winner["training"]["model_digest"],
            tracking_uri=store.request["config"]["tracking_uri"],
        )
        output = {
            **winner["training"],
            "model_uri": uri,
            "tags": tags,
            "competition": selection,
            "competition_sha256": evidence_digest(selection),
            **training_summary(
                store, artifact, config={**store.request["config"], "pipeline": recipe["pipeline"]}
            ),
        }
        return store.complete("train", output, reference)
    except BaseException:
        record_phase_failure(store, "train")
        raise
