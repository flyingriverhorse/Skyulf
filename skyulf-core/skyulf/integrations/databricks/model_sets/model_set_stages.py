"""Visible, durable registration, quality and alias decision stages for model sets."""

from dataclasses import asdict
from typing import Any

from skyulf.integrations.mlflow.shared._client import make_registry_client, require_mlflow

from ...mlflow.models.model_set import load_registered_model_set
from ...mlflow.registration.registry import resolve_model
from ..jobs.lifecycle.lifecycle_tasks import record_phase_failure
from ..jobs.training.branch_notebook import set_next_actions
from ..jobs.training.branch_tasks import register_branch_set, run_branch_operator
from ..lifecycle._lifecycle_state import LifecycleContext, LifecyclePhaseResult, PhaseStore
from ..shared._contracts import input_budget_bytes
from .model_set_project import project_endpoints
from .model_set_quality import evaluate_model_set_quality
from .model_set_release import (
    ModelSetQualityError,
    approve_project_model_set,
    champion_artifact,
    persist_decision,
)


def _registered_set(store: PhaseStore) -> Any:
    """Bind quality and decisions to the exact set registered by this invocation."""
    saved = store.receipt("register_model_set")["output"]["model_set_candidate"]
    candidate = resolve_model(
        saved["name"], version=saved["version"], **project_endpoints(store.request["config"])
    )
    if (
        candidate.digest != saved["digest"]
        or candidate.name != store.request["settings"]["model_name"]
    ):
        raise ValueError("Model set identity differs from the registered task receipt.")
    client = make_registry_client(require_mlflow(), **project_endpoints(store.request["config"]))
    version = client.get_model_version(candidate.name, candidate.version)
    if version.run_id != store.run_id:
        raise ValueError("Model set producing run differs from the invocation.")
    return candidate


def _evaluate(spark: Any, store: PhaseStore) -> dict:
    """Show per-model absolute and champion-comparison gates without moving aliases."""
    registered = store.receipt("register_model_set")["output"]
    settings = store.request["settings"]
    if settings is None:
        return {**registered, "quality": {"status": "model_set_disabled"}}
    candidate = _registered_set(store)
    config = store.request["config"]
    endpoints = project_endpoints(config)
    baseline = settings["expected_champion_version"]
    quality = evaluate_model_set_quality(
        spark,
        load_registered_model_set(candidate, **endpoints),
        champion_artifact(candidate.name, baseline, endpoints),
        expected_champion_version=baseline,
        max_rows=config["max_rows"],
        max_bytes=input_budget_bytes(config.get("max_input_mb")),
        **endpoints,
    )
    persist_decision(
        candidate, quality, settings.get("promotion_policy", "manual_approval"), endpoints
    )
    return {**registered, "quality": quality}


def _decide(spark: Any, store: PhaseStore) -> dict:
    """Apply policy after visible quality evaluation, rechecking evidence before promotion."""
    output = dict(store.receipt("evaluate_model_set")["output"])
    settings = store.request["settings"]
    if settings is None:
        return output
    candidate = _registered_set(store)
    policy = settings.get("promotion_policy", "manual_approval")
    output.update(promotion_policy=policy, alias_change=None)
    if policy == "automatic" and output["quality"]["passed"]:
        try:
            receipt = approve_project_model_set(
                spark,
                candidate,
                store.request["config"],
                expected_champion_version=settings["expected_champion_version"],
                policy=policy,
            )
            output.update(alias_change=asdict(receipt), quality_passed=True)
        except ModelSetQualityError as exc:
            output["quality"] = exc.decision
    output["next_actions"] = set_next_actions(output)
    return output


def _active(store: PhaseStore) -> None:
    """Allow the completed training parent while refusing failed or operator invocations."""
    if (
        store.request["action"] != "train"
        or store.client.get_run(store.run_id).info.status != "FINISHED"
        or any(
            key.endswith(".attempt") and value == "failed" for key, value in store.tags().items()
        )
    ):
        raise ValueError(
            "Model-set lifecycle is no longer active; inspect evidence and start a fresh run."
        )


def run_model_set_phase(
    spark: Any,
    *,
    phase: str,
    context: LifecycleContext,
    tracking_uri: str,
    reference: dict,
) -> LifecyclePhaseResult:
    """Execute one named set stage with integrity-checked predecessor evidence."""
    context.validate()
    if phase == "register_model_set":
        return register_branch_set(
            spark, context=context, tracking_uri=tracking_uri, reference=reference
        )
    if phase not in {"evaluate_model_set", "model_decision"}:
        raise ValueError("Unsupported model-set lifecycle stage.")
    store = PhaseStore(tracking_uri, context)
    store.bind(reference)
    expected = "prepare" if phase == "model_decision" else "register_model_set"
    if reference["phase"] != expected:
        raise ValueError("Model-set stage received the wrong predecessor reference.")
    if phase == "model_decision" and store.request["action"] != "train":
        return run_branch_operator(
            spark, context=context, tracking_uri=tracking_uri, reference=reference
        )
    _active(store)
    if phase == "model_decision":
        reference = store.reference(store.receipt("evaluate_model_set"))
    return _execute(spark, store, phase, reference)


def _execute(spark: Any, store: PhaseStore, phase: str, reference: dict) -> LifecyclePhaseResult:
    """Record attempt intent and never retry an uncertain alias mutation."""
    store.begin(phase)
    try:
        action = _evaluate if phase == "evaluate_model_set" else _decide
        return store.complete(phase, action(spark, store), reference)
    except BaseException:
        record_phase_failure(store, phase)
        store.client.set_terminated(store.run_id, status="FAILED")
        raise
