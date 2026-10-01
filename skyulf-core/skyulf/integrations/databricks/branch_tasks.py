"""Pinned multi-target training across independent Databricks notebook tasks."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from skyulf.integrations.mlflow._client import get_or_create_experiment

from ..mlflow.validation import ModelComparisonReport
from ._lifecycle_state import LifecycleContext, LifecyclePhaseResult, PhaseStore
from .lifecycle_tasks import record_phase_failure, validate_active_phase
from .local_branches import (
    BranchTrainingResult,
    branch_training_payload,
    log_progress,
    prepare_training_branches,
    restore_training_branches,
    train_branch,
)
from .local_retraining import LocalCandidateResult
from .local_training_evidence import evidence_digest
from .model_set_project import package_training_model_set, project_endpoints
from .model_set_release import pin_model_set_baseline


def _training_request(spark: Any, configs: dict, settings: dict | None) -> dict:
    """Capture all source versions, baselines and recipes before any branch runs."""
    champions = None
    base = next(iter(configs.values()))
    if settings is not None:
        settings, champions = pin_model_set_baseline(settings, configs, project_endpoints(base))
    branches = prepare_training_branches(spark, configs, champion_versions=champions)
    plan = branch_training_payload(branches)
    digest = hashlib.sha256(
        json.dumps(
            plan,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return {"branch_plan": plan, "plan_sha256": digest, "settings": settings}


def initialize_branch_training(
    spark: Any,
    *,
    configs: dict,
    settings: dict | None,
    composition_source: str,
    context: LifecycleContext,
    tracking_uri: str,
    experiment_name: str,
    action: str = "train",
    operator_values: dict | None = None,
    score_handoff: str = "disabled",
) -> LifecyclePhaseResult:
    """Create one invocation that freezes branch policy or an explicit set operation."""
    context.validate()
    if action not in {"train", "approve", "reject", "rollback"}:
        raise ValueError("Unsupported multi-model lifecycle action.")
    if score_handoff not in {"disabled", "after_alias_change"}:
        raise ValueError("score_handoff must be disabled or after_alias_change.")
    if score_handoff == "after_alias_change" and settings is None:
        raise ValueError("Enable a model set before requesting score handoff.")
    base = next(iter(configs.values()))
    request: dict[str, Any] = {
        "version": 1,
        "context": context.identity(),
        "action": action,
        "config": base,
        "settings": settings,
        "composition_source": composition_source,
        "experiment_name": experiment_name,
        "operator_values": operator_values or {},
        "score_handoff": score_handoff,
    }
    if action == "train":
        request.update(_training_request(spark, configs, settings))
    store = PhaseStore(tracking_uri, context)
    _create_parent(store, request, experiment_name)
    if action == "train":
        store.log("branch_training_plan.json", request["branch_plan"])
        store.run.set_tags(
            {
                "skyulf.training.plan_sha256": request["plan_sha256"],
                "skyulf.training.status": "running",
            }
        )
    return store.complete(
        "prepare", {"training_requested": action == "train", "action": action}, None
    )


def _create_parent(store: PhaseStore, request: dict, experiment_name: str) -> None:
    """Reject duplicate job attempts before writing an immutable request."""
    experiment = get_or_create_experiment(store.client, experiment_name)
    tags = {
        "skyulf.lifecycle.job_id": store.context.job_id,
        "skyulf.lifecycle.job_run_id": store.context.job_run_id,
    }
    query = " AND ".join(f"tags.`{key}` = '{value}'" for key, value in tags.items())
    if store.client.search_runs([experiment], filter_string=query):
        raise ValueError("Lifecycle invocation already attempted; start a fresh run.")
    store.run_id = store.client.create_run(
        experiment, run_name="multi_model", tags=tags
    ).info.run_id
    request["run_id"] = store.run_id
    store.request = json.loads(json.dumps(request, allow_nan=False))
    store.request_digest = evidence_digest(store.request)
    store.log("lifecycle/request.json", store.request)
    store.run.set_tags({"skyulf.lifecycle.request": store.request_digest})
    store.begin("prepare")


def _bound_store(context: LifecycleContext, tracking_uri: str, reference: dict) -> PhaseStore:
    """Accept only the initialization receipt of the same unrepaired invocation."""
    context.validate()
    store = PhaseStore(tracking_uri, context)
    store.bind(reference)
    if reference["phase"] != "prepare":
        raise ValueError("Multi-model task requires its initialization reference.")
    return store


def run_branch_training(
    spark: Any,
    *,
    name: str,
    context: LifecycleContext,
    tracking_uri: str,
    reference: dict,
) -> LifecyclePhaseResult:
    """Train, register and compare one branch using its frozen independent policy."""
    store = _bound_store(context, tracking_uri, reference)
    validate_active_phase(store, "train")
    branches = {
        branch.name: branch for branch in restore_training_branches(store.request["branch_plan"])
    }
    if name not in branches:
        raise ValueError("Training node is not a branch in this invocation.")
    phase = f"branch_{name}"
    store.begin(phase)
    try:
        with TemporaryDirectory(prefix="skyulf-branch-task-") as directory:
            result = train_branch(
                spark,
                branches[name],
                run=store.run,
                digest=store.request["plan_sha256"],
                path=Path(directory) / "model",
                experiment_name=store.request["experiment_name"],
                evaluation_charts=store.request["config"].get("evaluation_charts"),
                **project_endpoints(store.request["config"]),
            )
        return store.complete(phase, {"name": name, **asdict(result)}, reference)
    except BaseException:
        record_phase_failure(store, phase)
        raise


def _completed_branches(store: PhaseStore, branches: tuple) -> BranchTrainingResult:
    """Require exact completed children before creating a publishable parent result."""
    completed = {}
    for branch in branches:
        payload = dict(store.receipt(f"branch_{branch.name}")["output"])
        name = payload.pop("name")
        child = store.client.get_run(payload["run_id"])
        if (
            name != branch.name
            or payload["model_name"] != branch.model_name
            or child.info.status != "FINISHED"
            or child.data.tags.get("mlflow.parentRunId") != store.run_id
            or child.data.tags.get("skyulf.training.plan_sha256") != store.request["plan_sha256"]
        ):
            raise ValueError("Branch result differs from its pinned training task.")
        payload["comparison"] = ModelComparisonReport(**payload["comparison"])
        completed[name] = LocalCandidateResult(**payload)
    first = branches[0].spec
    return BranchTrainingResult(
        store.run_id, first.table, first.version, completed, store.request["plan_sha256"]
    )


def register_branch_set(
    spark: Any,
    *,
    context: LifecycleContext,
    tracking_uri: str,
    reference: dict,
) -> LifecyclePhaseResult:
    """Package and nominate a complete model set without evaluating or promoting it."""
    store = _bound_store(context, tracking_uri, reference)
    validate_active_phase(store, "train")
    branches = restore_training_branches(store.request["branch_plan"])
    outcome = _completed_branches(store, branches)
    store.begin("register_model_set")
    try:
        store.log("branch_training_result.json", asdict(outcome))
        log_progress(store.run, outcome.components, status="complete")
        store.client.set_terminated(store.run_id, status="FINISHED")
        payload = _register_set(spark, store, branches, outcome)
        return store.complete("register_model_set", payload, reference)
    except BaseException:
        record_phase_failure(store, "register_model_set")
        store.client.set_terminated(store.run_id, status="FAILED")
        raise


def _register_set(
    spark: Any, store: PhaseStore, branches: tuple, outcome: BranchTrainingResult
) -> dict:
    """Expose the new immutable set and its challenger nomination as one visible step."""
    from ..mlflow.model_set_challenger import nominate_model_set  # noqa: PLC0415
    from ..mlflow.promotion import ExclusiveAliasWriterAdmission  # noqa: PLC0415

    payload = asdict(outcome)
    settings = store.request["settings"]
    if settings is None:
        return payload
    config = store.request["config"]
    candidate = package_training_model_set(
        spark,
        branches,
        outcome,
        settings,
        composition_source=store.request["composition_source"],
        **project_endpoints(config),
    )
    payload["model_set_candidate"] = {
        "name": candidate.name,
        "version": candidate.version,
        "digest": candidate.digest,
    }
    nominate_model_set(
        candidate,
        expected_champion_version=settings["expected_champion_version"],
        admission=ExclusiveAliasWriterAdmission(),
        **project_endpoints(config),
    )
    payload["challenger_version"] = candidate.version
    return payload


def assemble_branch_training(
    spark: Any, *, context: LifecycleContext, tracking_uri: str, reference: dict
) -> LifecyclePhaseResult:
    """Run all visible set stages sequentially for direct callers of the task service."""
    from .model_set_stages import run_model_set_phase  # noqa: PLC0415

    registered = register_branch_set(
        spark, context=context, tracking_uri=tracking_uri, reference=reference
    )
    run_model_set_phase(
        spark,
        phase="evaluate_model_set",
        context=context,
        tracking_uri=tracking_uri,
        reference=registered.reference,
    )
    return run_model_set_phase(
        spark,
        phase="model_decision",
        context=context,
        tracking_uri=tracking_uri,
        reference=reference,
    )


def run_branch_operator(
    spark: Any,
    *,
    context: LifecycleContext,
    tracking_uri: str,
    reference: dict,
) -> LifecyclePhaseResult:
    """Run an explicit set action from pinned settings without loading editable recipes."""
    from .job_runtime import operator_options  # noqa: PLC0415  # noqa: PLC0415
    from .model_set_project import run_model_set_operator  # noqa: PLC0415

    store = _bound_store(context, tracking_uri, reference)
    validate_active_phase(store, "operator")
    store.begin("operator")
    try:
        values = store.request["operator_values"]
        output = run_model_set_operator(
            spark,
            values,
            store.request["settings"],
            operator_options(store.request["action"], values),
            config=store.request["config"],
        )
        result = store.complete("operator", output, reference)
        store.client.set_terminated(store.run_id, status="FINISHED")
        return result
    except BaseException:
        record_phase_failure(store, "operator")
        raise
