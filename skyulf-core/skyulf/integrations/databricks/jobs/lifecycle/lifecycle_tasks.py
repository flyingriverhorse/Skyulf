"""Fixed lifecycle phases backed by MLflow artifacts and existing Core computations.

The caller serializes these tasks in the lifecycle job. An attempt marker is
durable evidence of possible work, not an exactly-once registration guarantee.
Repairs and retries are rejected; uncertain outcomes require operator inspection.
"""

import json
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl

from skyulf.integrations.mlflow.shared._client import get_or_create_experiment

from .....inference.project_code import load_project_module, project_source_digest
from ....mlflow.lifecycle.challenger import ChallengerLifecycle
from ....mlflow.lifecycle.promotion import AliasChangeReceipt, ExclusiveAliasWriterAdmission
from ....mlflow.lifecycle.validation import ModelComparisonReport
from ....mlflow.registration.registry import load_run_pipeline
from ...lifecycle import _lifecycle_data as data_stages
from ...lifecycle import workflow as workflow
from ...lifecycle._lifecycle_state import (
    PHASE_PREDECESSORS,
    LifecycleContext,
    LifecyclePhaseResult,
    PhaseStore,
)
from ...training.competition.competition import (
    prepare_competition,
    selected_request,
    validate_competition_budget,
)
from ...training.fitting import candidate as training
from ...training.shared.training_evidence import evidence_digest, validate_training_evidence
from ...training.tuning.cv import CVSpec

__all__ = ["LifecycleContext", "LifecyclePhaseResult", "run_lifecycle_phase"]

_GROUPED_PHASES = {
    "train_register": ("train", "evaluate_register"),
    "compare_decide": ("compare", "decide"),
}


@dataclass(frozen=True)
class _ReplayEvidence:
    """Keep verified fit metadata within one phase; never reuse it across phase calls."""

    artifact: Any
    spec: training.TrainingSpec
    fitted: dict[str, Any]
    filter_evidence: dict[str, Any]


def phase_training_spec(payload: dict[str, Any], source: str | None) -> training.TrainingSpec:
    """Restore pinned source settings after loading the saved custom-step identities."""
    if source is not None:
        module = load_project_module(source)
        # custom_step registers inside builders, not when their classes are imported.
        # Keep the saved steps authoritative; these calls restore process-local IDs.
        for name in ("build_preprocessing", "build_pre_split_steps"):
            factory = getattr(module, name, None)
            if factory is not None:
                factory()
    return training.TrainingSpec.from_payload(payload)


def _prepare_training_request(
    spark: Any,
    request: dict[str, Any],
    config: dict[str, Any],
    options: dict[str, Any],
    policy: str,
    now: datetime | None,
) -> None:
    """Pin source, champion and effective recipe before creating a lifecycle run."""
    if options:
        raise ValueError("Training cannot accept operator options.")
    if config.get("training_layout") == "model_competition" or "competition" in config:
        from ...projects.competition_project import validate_competition_config  # noqa: PLC0415

        validate_competition_config(config)
        validate_competition_budget(config)
    spec, cv, champion = workflow.prepare_training(spark, config, policy=policy, now=now)
    if config.get("champion_version") is not None and str(config["champion_version"]) != champion:
        raise ValueError("champion_version does not match the current champion.")
    effective = training.candidate_config(
        spec,
        config["pipeline"],
        engine=config["engine"],
        cv=cv,
        metric=config["metric"],
        min_improvement=config["min_improvement"],
        champion_version=champion,
        quality_threshold=config.get("quality_threshold"),
        quality_gates=config.get("quality_gates"),
        risk_category=config.get("risk_category"),
    )
    request.update(
        spec=training.training_spec_payload(spec, config["engine"]),
        champion_version=champion,
        effective_config=effective,
    )
    if "competition" in config:
        request["competition"] = prepare_competition(config, spec, champion)


def _prepared_output(
    request: dict[str, Any],
    config: dict[str, Any],
    action: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Describe the pinned training source or the selected operator request."""
    output = {
        "action": action,
        "training_requested": action == "train",
        "model_name": config["model_name"],
    }
    if action == "train":
        pinned = request["spec"]
        output.update(
            source_table=pinned["table"],
            source_version=pinned["version"],
            engine=pinned["engine"],
            split_strategy=pinned["split_strategy"],
            expected_champion_version=request["champion_version"],
        )
        output |= {
            field: pinned[field]
            for field in ("start", "holdout_start", "cutoff", "result_cutoff")
            if pinned[field] is not None
        }
    else:
        output |= {
            field: options[field]
            for field in ("candidate_version", "expected_champion_version")
            if field in options
        }
    return output


def _prepare(
    spark: Any,
    store: PhaseStore,
    config: dict[str, Any],
    action: str,
    experiment_name: str,
    operator_options: dict[str, Any],
    now: datetime | None,
    *,
    graph_version: int = 2,
) -> LifecyclePhaseResult:
    """Resolve source and champion once, then persist the complete immutable invocation."""
    if action not in {"train", "approve", "reject", "rollback"}:
        raise ValueError("Unsupported lifecycle action.")
    _, policy = workflow.workflow_policies(config)
    options = {
        key: asdict(value) if isinstance(value, AliasChangeReceipt) else value
        for key, value in operator_options.items()
    }
    request: dict[str, Any] = {
        "version": 1,
        "context": store.context.identity(),
        "config": config,
        "action": action,
        "operator_options": options,
    }
    if graph_version == 3:
        request["graph_version"] = 3
    if action == "train":
        _prepare_training_request(spark, request, config, options, policy, now)
    # JSON normalization also detaches all caller-owned editable dictionaries.
    request = json.loads(json.dumps(request, allow_nan=False))
    experiment = get_or_create_experiment(store.client, experiment_name)
    if store.client.search_runs(
        [experiment],
        filter_string=(
            f"tags.`skyulf.lifecycle.job_id` = '{store.context.job_id}' AND "
            f"tags.`skyulf.lifecycle.job_run_id` = '{store.context.job_run_id}'"
        ),
    ):
        raise ValueError(
            "Lifecycle invocation already attempted; inspect evidence and start a fresh run."
        )
    created = store.client.create_run(
        experiment,
        run_name="candidate_training" if action == "train" else f"lifecycle_{action}",
        tags={
            "skyulf.lifecycle.job_id": store.context.job_id,
            "skyulf.lifecycle.job_run_id": store.context.job_run_id,
            "skyulf.lifecycle.status": "incomplete",
        },
    )
    store.run_id = created.info.run_id
    try:
        request["run_id"] = store.run_id
        store.request = request
        store.request_digest = evidence_digest(request)
        store.log("lifecycle/request.json", request)
        store.client.set_tag(store.run_id, "skyulf.lifecycle.request", store.request_digest)
        store.begin("prepare")
        output = _prepared_output(request, config, action, options)
        return store.complete("prepare", output, None)
    except BaseException:
        with suppress(Exception):
            store.client.set_terminated(store.run_id, status="FAILED")
        raise


def _train(spark: Any, store: PhaseStore) -> dict[str, Any]:
    """Run the SDK's shared fitting computation and upload its fitted package."""
    request = store.request
    config: dict[str, Any] = request["config"]
    source = config["pipeline"].get("project_python_source")
    spec = phase_training_spec(request["spec"], source)
    selection = None
    with TemporaryDirectory(prefix="skyulf-phase-fit-") as directory:
        path = Path(directory) / "artifact"
        if "competition" in request:
            from ...training.competition.competition_training import (  # noqa: PLC0415
                fit_competition,
            )

            fitted, pipeline, path, selection = fit_competition(
                spark, store, spec, Path(directory), **_prepared_fit_options(store)
            )
            config = {**config, "pipeline": pipeline}
        else:
            fitted = _fit_single_candidate(spark, store, spec, path)
        training.log_fitted_candidate(
            store.run,
            fitted,
            config["pipeline"],
            engine=config["engine"],
            risk_category=config.get("risk_category"),
        )
        if selection is not None:
            fitted.tags["competition_winner"] = selection["winner"]
        model_uri = training.log_pipeline_model(
            path,
            run_id=store.run_id,
            tracking_uri=config["tracking_uri"],
            **training.feature_log_options(spark, fitted.spec),
        )
    output = {
        "model_uri": model_uri,
        "model_digest": fitted.artifact.manifest.pipeline_sha256,
        "project_source_sha256": fitted.artifact.manifest.project_source_sha256,
        "spec": training.training_spec_payload(fitted.spec, config["engine"]),
        "training_rows": fitted.training_rows,
        "holdout_rows": fitted.holdout_rows,
        "unavailable_labels": fitted.unavailable_labels,
        "tags": fitted.tags,
        **training_summary(store, fitted.artifact, config=config),
    }
    if selection is not None:
        output.update(competition=selection, competition_sha256=evidence_digest(selection))
    return output


def _fit_single_candidate(spark: Any, store: PhaseStore, spec: Any, path: Path) -> Any:
    """Keep the original single-model fit path independent from competition orchestration."""
    config = store.request["config"]
    return training.fit_candidate(
        spark,
        spec,
        config["pipeline"],
        run=store.run,
        pipeline_config=store.request["effective_config"],
        artifact_path=path,
        engine=config["engine"],
        cv=CVSpec.from_workflow(config),
        risk_category=config.get("risk_category"),
        **_prepared_fit_options(store),
    )


def _prepared_fit_options(store: PhaseStore) -> dict[str, Any]:
    """Use staged partitions only for invocations pinned to the readable graph."""
    if store.request.get("graph_version", 2) == 3:
        return {"prepared_data": data_stages.training_partitions(store)}
    return {}


def _load_data(spark: Any, store: PhaseStore) -> dict[str, Any]:
    """Read the pinned source in its own visible data-loading stage."""
    source = store.request["config"]["pipeline"].get("project_python_source")
    return data_stages.load_source(spark, store, phase_training_spec(store.request["spec"], source))


def _prepare_dataset(spark: Any, store: PhaseStore) -> dict[str, Any]:
    """Persist fixed-cleanup results and split membership before learned transforms."""
    source = store.request["config"]["pipeline"].get("project_python_source")
    return data_stages.prepare_dataset(store, phase_training_spec(store.request["spec"], source))


def _select_best_model(spark: Any, store: PhaseStore) -> dict[str, Any]:
    """Verify the selected fitted candidate and expose its training-side leaderboard."""
    verified = _load_training_evidence(store)
    if "competition" in store.request:
        return {
            **verified.fitted["competition"],
            "selection_mode": "model_competition",
            "model_uri": verified.fitted["model_uri"],
            "model_digest": verified.fitted["model_digest"],
        }
    return {
        "candidate_count": 1,
        "selection_mode": "single_candidate",
        "selection_reason": "Only one model was requested; multi-model competition is not enabled.",
        "model_uri": verified.fitted["model_uri"],
        "model_digest": verified.fitted["model_digest"],
    }


def _validate_pinned_spec(fitted_spec: dict[str, Any], pinned_spec: dict[str, Any]) -> None:
    """Allow only membership evidence to enrich the original source specification."""
    enriched = {
        "holdout_key_sha256",
        "sample_key_sha256",
        "survivor_key_sha256",
        "training_evidence_sha256",
    }
    defaults = {
        "weight_column": None,
        "reserved_weight_columns": [],
        "weights_python_source": None,
        "weights_python_sha256": None,
    }
    if {key: value for key, value in (defaults | fitted_spec).items() if key not in enriched} != {
        key: value for key, value in (defaults | pinned_spec).items() if key not in enriched
    }:
        raise ValueError("Saved training source differs from pinned invocation.")


def _load_training_evidence(store: PhaseStore) -> _ReplayEvidence:
    """Verify saved fit, invocation and filter evidence before any source replay."""
    fitted = store.receipt("train")["output"]
    config, effective_config = selected_request(store)
    if fitted["model_uri"] != f"runs:/{store.run_id}/model":
        raise ValueError("Fitted model source differs from lifecycle invocation.")
    artifact = load_run_pipeline(
        fitted["model_uri"], digest=fitted["model_digest"], tracking_uri=config["tracking_uri"]
    )
    source = config["pipeline"].get("project_python_source")
    source_sha = None if source is None else project_source_digest(source)
    if store.read("candidate_training_spec.json") != fitted["spec"]:
        raise ValueError("Saved candidate training spec differs from phase receipt.")
    spec = phase_training_spec(fitted["spec"], source)
    _validate_pinned_spec(fitted["spec"], store.request["spec"])
    evidence = store.read("training_filter_evidence.json")
    validate_training_evidence(evidence, spec, project_source_sha256=source_sha)
    if spec.weight_column is not None:
        effective_config = {
            **effective_config,
            "training_weights": evidence.get("training_weights"),
        }
    if (
        artifact.manifest.fitted_engine != config["engine"]
        or artifact.manifest.project_source_sha256 != source_sha
        or fitted["project_source_sha256"] != source_sha
        or artifact.pipeline.config != effective_config
    ):
        raise ValueError(
            "Fitted model engine, configuration or project source differs from invocation."
        )
    return _ReplayEvidence(artifact, spec, fitted, evidence)


def _replay(spark: Any, store: PhaseStore) -> tuple[_ReplayEvidence, Any]:
    """Reconstruct exact heldout membership from freshly verified phase evidence."""
    verified = _load_training_evidence(store)
    engine = store.request["config"]["engine"]
    frame = training.read_training_snapshot(spark, verified.spec)
    _, holdout, _ = training.split_labeled_snapshot(frame, verified.spec, engine=engine)
    validate_training_evidence(
        verified.filter_evidence,
        verified.spec,
        project_source_sha256=verified.filter_evidence["project_source_sha256"],
        heldout=holdout,
    )
    native = pl.from_pandas(holdout) if engine == "polars" else holdout
    return verified, native


def _lifecycle(store: PhaseStore) -> ChallengerLifecycle:
    """Bind lifecycle status handling to the request's exact expected champion."""
    config = store.request["config"]
    return ChallengerLifecycle(
        config["model_name"],
        expected_champion_version=store.request["champion_version"],
        admission=ExclusiveAliasWriterAdmission(),
        tracking_uri=config["tracking_uri"],
        registry_uri=config.get("registry_uri", "databricks-uc"),
    )


def _registered(store: PhaseStore) -> Any:
    """Resolve the persisted version and verify its source run and package identity."""
    config = store.request["config"]
    payload = store.read("lifecycle/registration.json")
    if evidence_digest(payload) != store.tags().get("skyulf.lifecycle.registration"):
        raise ValueError("Registration receipt differs from durable identity.")
    if (
        payload["run_id"] != store.run_id
        or payload["model_name"] != config["model_name"]
        or payload["requested_source"] != f"runs:/{store.run_id}/model"
        or not isinstance(payload["registered_source"], str)
        or not payload["registered_source"]
    ):
        raise ValueError("Registration receipt belongs to another invocation.")
    candidate = training.resolve_model(
        config["model_name"],
        version=payload["model_version"],
        tracking_uri=config["tracking_uri"],
        registry_uri=config.get("registry_uri", "databricks-uc"),
    )
    client = workflow.make_registry_client(
        workflow.require_mlflow(),
        config["tracking_uri"],
        config.get("registry_uri", "databricks-uc"),
    )
    registered = client.get_model_version(candidate.name, candidate.version)
    if (
        candidate.digest != payload["model_digest"]
        or registered.run_id != store.run_id
        or registered.source != payload["registered_source"]
    ):
        raise ValueError("Registered candidate source or digest differs from durable receipt.")
    return candidate


def _evaluate_register(spark: Any, store: PhaseStore) -> dict[str, Any]:
    """Evaluate first and record mutation intent before registering and nominating."""
    verified, holdout = _replay(spark, store)
    spec = verified.spec
    config = store.request["config"]
    metrics = training.evaluate_candidate(
        verified.artifact,
        holdout,
        spec=spec,
        metric=config["metric"],
        chart_run=store.run,
        evaluation_charts=config.get("evaluation_charts"),
    )
    store.run.log_metrics(metrics)
    store.log(
        "lifecycle/initial_evaluation.json", {"metrics": metrics, "dataset_id": spec.dataset_id}
    )
    # Recheck the chain after evaluation and immediately before registration intent.
    fitted = store.receipt("train")["output"]
    store.client.set_tag(store.run_id, "skyulf.lifecycle.registration_intent", "started")
    registered = training.register_candidate(
        fitted["model_uri"],
        config["model_name"],
        tracking_uri=config["tracking_uri"],
        registry_uri=config.get("registry_uri", "databricks-uc"),
        tags=fitted["tags"],
    )
    receipt = {
        "run_id": store.run_id,
        "model_name": config["model_name"],
        "model_version": str(registered.version),
        "model_digest": fitted["model_digest"],
        "requested_source": fitted["model_uri"],
        "registered_source": registered.source,
    }
    store.log("lifecycle/registration.json", receipt)
    store.client.set_tag(store.run_id, "skyulf.lifecycle.registration", evidence_digest(receipt))
    candidate = _registered(store)
    _lifecycle(store).registered(candidate)
    output = {
        "candidate_version": candidate.version,
        "model_digest": candidate.digest,
        "metrics": metrics,
        "dataset_id": spec.dataset_id,
    }
    return output | training_summary(store, verified.artifact)


def training_summary(
    store: PhaseStore, artifact: Any, *, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Expose the same fitted search and explanation evidence in training and registry reports."""
    if config is None:
        config, _ = selected_request(store)
    output: dict[str, Any] = {}
    if config["pipeline"]["modeling"]["type"] == "hyperparameter_tuner":
        evidence = training.tuning_evidence(artifact)
        if evidence is None:
            raise ValueError("Search artifact lacks tuning evidence.")
        output["tuning"] = {
            key: value for key, value in evidence.items() if key not in {"trials", "modeling"}
        }
        output["tuning"]["strategy"] = artifact.pipeline.config["modeling"]["strategy"]
        output["tuning"]["artifact"] = "tuning.json"
    if config["pipeline"].get("explainability"):
        explanation = store.read("explanations.json")
        output["explanations"] = {
            key: explanation[key]
            for key in ("status", "reason", "sample_count", "report_status", "report_reason")
            if key in explanation
        }
        output["explanations"]["artifact"] = "explanations.json"
        output["explanations"]["report"] = "explanations.html"
    else:
        output["explanations"] = {"status": "disabled"}
    return output


def _compare(spark: Any, store: PhaseStore) -> dict[str, Any]:
    """Compare pinned versions through the same registered-model computation as the SDK."""
    verified, holdout = _replay(spark, store)
    spec = verified.spec
    config = store.request["config"]
    candidate = _registered(store)
    champion_version = store.request["champion_version"]
    champion = (
        None
        if champion_version is None
        else training.resolve_model(
            config["model_name"],
            version=champion_version,
            tracking_uri=config["tracking_uri"],
            registry_uri=config.get("registry_uri", "databricks-uc"),
        )
    )
    fitted = verified.fitted
    result = training.compare_candidate(
        candidate,
        champion,
        holdout,
        run=store.run,
        spec=spec,
        model_name=config["model_name"],
        metric=config["metric"],
        min_improvement=config["min_improvement"],
        quality_threshold=config.get("quality_threshold"),
        quality_gates=config.get("quality_gates"),
        tracking_uri=config["tracking_uri"],
        registry_uri=config.get("registry_uri", "databricks-uc"),
        engine=config["engine"],
        training_rows=fitted["training_rows"],
        holdout_rows=fitted["holdout_rows"],
        unavailable=fitted["unavailable_labels"],
    )
    return asdict(result)


def _candidate(payload: dict[str, Any]) -> training.CandidateResult:
    """Restore the SDK result type from verified JSON comparison evidence."""
    values = dict(payload)
    values["comparison"] = ModelComparisonReport(**payload["comparison"])
    return training.CandidateResult(**values)


def _decide(spark: Any, store: PhaseStore) -> dict[str, Any]:
    """Reuse strict automatic replay for promotion or the saved manual-review decision."""
    config = store.request["config"]
    candidate = _candidate(store.receipt("compare")["output"])
    _, policy = workflow.workflow_policies(config)
    # Verify invocation/package pins here; the decision service replays the source
    # and checks membership against registered evidence before any alias mutation.
    verified = _load_training_evidence(store)
    receipt = workflow.automatic_promotion(
        spark, config, verified.spec, candidate, promote=policy == "automatic"
    )
    return {
        "candidate": asdict(candidate),
        "alias_change": None if receipt is None else asdict(receipt),
        "promotion_policy": policy,
    }


def _operator(spark: Any, store: PhaseStore) -> dict[str, Any]:
    """Execute a saved operator action without reading current project Python or fitting."""
    options = dict(store.request["operator_options"])
    if "promotion_receipt" in options:
        options["promotion_receipt"] = AliasChangeReceipt(**options["promotion_receipt"])
    result = workflow.run_action(spark, store.request["config"], store.request["action"], **options)
    return {"action": store.request["action"], "result": asdict(result)}


def _finalize(store: PhaseStore) -> dict[str, Any]:
    """Treat missing or failed training phases as failure independently of later scoring."""
    try:
        store.receipt("decide")
    except Exception:  # noqa: BLE001 - absent or corrupt phase evidence is unsuccessful training
        tags = store.tags()
        registration = "not_attempted"
        if tags.get("skyulf.lifecycle.registration_intent"):
            registration = "unknown"
        if tags.get("skyulf.lifecycle.registration"):
            lifecycle = _lifecycle(store)
            lifecycle.restore(_registered(store))
            lifecycle.failed()
            registration = "recorded"
        return {"status": "FAILED", "registration_outcome": registration}
    return {"status": "FINISHED", "registration_outcome": "recorded"}


def _result(store: PhaseStore) -> dict[str, Any]:
    """Publish a Bundle result only from successful finalized branch receipts."""
    request = store.request
    if (
        store.tags().get("skyulf.lifecycle.status") != "FINISHED"
        or store.client.get_run(store.run_id).info.status != "FINISHED"
    ):
        raise ValueError("Lifecycle has no successful finalized result.")
    if request["action"] == "train":
        if store.receipt("finalize")["output"]["status"] != "FINISHED":
            raise ValueError("Lifecycle has no successful training finalization.")
        decision = store.receipt("decide")["output"]
        candidate = _candidate(decision["candidate"])
        receipt = decision["alias_change"]
        outcome = (
            workflow.AutoTrainingOutcome(
                candidate, None if receipt is None else AliasChangeReceipt(**receipt)
            )
            if decision["promotion_policy"] == "automatic"
            else candidate
        )
    else:
        outcome = AliasChangeReceipt(**store.receipt("operator")["output"]["result"])
    result = asdict(workflow.build_bundle_result(request["config"], request["action"], outcome))
    if request["action"] == "train" and "competition" in request:
        selected_request(store)
        result["competition"] = store.receipt("train")["output"]["competition"]
    return result


def _complete_invocation(
    spark: Any,
    store: PhaseStore,
    tracking_uri: str,
    reference: dict[str, str],
    task_states: dict[str, str] | None,
) -> LifecyclePhaseResult:
    """Finalize training before checking task outcomes and publishing the branch result."""
    training_action = store.request["action"] == "train"
    if training_action:
        run_lifecycle_phase(
            spark,
            phase="finalize",
            context=store.context,
            tracking_uri=tracking_uri,
            reference=reference,
        )
    expected_states = (
        {"training": "success", "operator": "excluded"}
        if training_action
        else {"training": "excluded", "operator": "success"}
    )
    if store.request.get("graph_version", 2) == 3:
        expected_states = {"decision": "success"}
    if task_states != expected_states:
        raise ValueError("Lifecycle task outcomes do not allow publishing a result.")
    return run_lifecycle_phase(
        spark,
        phase="result",
        context=store.context,
        tracking_uri=tracking_uri,
        reference=reference,
    )


def validate_active_phase(store: PhaseStore, phase: str) -> None:
    """Reject the wrong branch or inactive attempts before starting durable work."""
    training_action = store.request["action"] == "train"
    if (
        phase
        in {
            "load_data",
            "prepare_dataset",
            "train",
            "select_best_model",
            "evaluate_register",
            "compare",
            "decide",
            "finalize",
        }
        and not training_action
        or phase == "operator"
        and training_action
    ):
        raise ValueError("Lifecycle phase does not belong to the pinned action branch.")
    if phase not in {"result", "finalize"} and (
        store.client.get_run(store.run_id).info.status != "RUNNING"
        or any(
            key.endswith(".attempt") and value == "failed" for key, value in store.tags().items()
        )
    ):
        raise ValueError("Lifecycle is no longer active; inspect evidence and start a fresh run.")


def record_phase_failure(store: PhaseStore, phase: str) -> None:
    """Best-effort cleanup preserves the original error and any committed alias change."""
    with suppress(Exception):
        store.client.set_tag(store.run_id, f"skyulf.lifecycle.{phase}.attempt", "failed")
    if phase in {"compare", "decide"}:
        with suppress(Exception):
            lifecycle = _lifecycle(store)
            lifecycle.restore(_registered(store))
            lifecycle.failed()
    if phase == "operator":
        with suppress(Exception):
            store.client.set_terminated(store.run_id, status="FAILED")


def _execute_phase(
    spark: Any, store: PhaseStore, phase: str, reference: dict[str, str]
) -> LifecyclePhaseResult:
    """Execute one phase and persist its receipt, termination or failure evidence."""
    store.begin(phase)
    try:
        if phase == "finalize":
            output = _finalize(store)
        elif phase == "result":
            output = _result(store)
        else:
            output = {
                "load_data": _load_data,
                "prepare_dataset": _prepare_dataset,
                "select_best_model": _select_best_model,
                "train": _train,
                "evaluate_register": _evaluate_register,
                "compare": _compare,
                "decide": _decide,
                "operator": _operator,
            }[phase](spark, store)
        completed = store.complete(phase, output, reference)
        if phase in {"finalize", "operator"}:
            status = output["status"] if phase == "finalize" else "FINISHED"
            store.client.set_terminated(store.run_id, status=status)
            store.client.set_tag(store.run_id, "skyulf.lifecycle.status", status)
        return completed
    except BaseException:
        record_phase_failure(store, phase)
        raise


def _validate_phase_inputs(
    phase: str,
    task_states: dict[str, str] | None,
    config: dict[str, Any] | None,
    action: str | None,
    experiment_name: str | None,
    operator_options: dict[str, Any] | None,
    now: datetime | None,
) -> None:
    """Reject inputs that do not belong to the selected fixed phase."""
    if phase not in {
        "prepare",
        "initialize",
        "model_decision",
        "complete",
        *PHASE_PREDECESSORS,
        *_GROUPED_PHASES,
    }:
        raise ValueError("Unsupported fixed lifecycle phase.")
    if phase != "complete" and task_states is not None:
        raise ValueError("Task states are accepted only by complete.")
    if phase not in {"prepare", "initialize"} and any(
        value is not None for value in (config, action, experiment_name, operator_options, now)
    ):
        raise ValueError("Downstream lifecycle phases must use only the pinned invocation.")


def _run_prepare_phase(
    spark: Any,
    store: PhaseStore,
    reference: dict[str, str] | None,
    config: dict[str, Any] | None,
    action: str | None,
    experiment_name: str | None,
    operator_options: dict[str, Any] | None,
    now: datetime | None,
    tracking_uri: str,
    graph_version: int = 2,
) -> LifecyclePhaseResult:
    """Validate prepare inputs before persisting the pinned invocation."""
    if reference is not None or config is None or action is None or not experiment_name:
        raise ValueError("Prepare requires config, action and experiment with no predecessor.")
    if config.get("tracking_uri", "databricks") != tracking_uri:
        raise ValueError("Tracking URI differs from invocation configuration.")
    return _prepare(
        spark,
        store,
        {**config, "tracking_uri": tracking_uri},
        action,
        experiment_name,
        operator_options or {},
        now,
        graph_version=graph_version,
    )


def run_lifecycle_phase(
    spark: Any,
    *,
    phase: str,
    context: LifecycleContext,
    tracking_uri: str,
    reference: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
    action: str | None = None,
    experiment_name: str | None = None,
    operator_options: dict[str, Any] | None = None,
    now: datetime | None = None,
    task_states: dict[str, str] | None = None,
) -> LifecyclePhaseResult:
    """Run fixed notebook phases with durable references and no cross-task local state.

    Prepare (legacy) and initialize accept configuration and operator inputs.
    Initialize pins graph 3 with saved data preparation stages. Groups retain
    each internal phase's receipts and return the last phase's reference. Complete
    takes the prepare reference, finalizes training when requested, then requires
    successful selected-branch task outcomes before publishing the result.
    Model_decision routes the prepared action to policy or operator handling.
    Finalize and result also take the prepare reference; other phases
    take their immediate predecessor's reference. This adapter requires the
    lifecycle job's existing serialization and never runs scoring.
    """
    context.validate()
    _validate_phase_inputs(
        phase, task_states, config, action, experiment_name, operator_options, now
    )
    if phase in _GROUPED_PHASES:
        for internal_phase in _GROUPED_PHASES[phase]:
            completed = run_lifecycle_phase(
                spark,
                phase=internal_phase,
                context=context,
                tracking_uri=tracking_uri,
                reference=reference,
            )
            reference = completed.reference
        return completed
    store = PhaseStore(tracking_uri, context)
    if phase in {"prepare", "initialize"}:
        return _run_prepare_phase(
            spark,
            store,
            reference,
            config,
            action,
            experiment_name,
            operator_options,
            now,
            tracking_uri,
            graph_version=3 if phase == "initialize" else 2,
        )
    if reference is None:
        raise ValueError("Lifecycle phase requires its predecessor reference.")
    store.bind(reference)
    reference = _selection_reference(store, phase, reference)
    expected_predecessor = (
        "prepare" if phase in {"complete", "model_decision"} else store.predecessor(phase)
    )
    if reference["phase"] != expected_predecessor:
        raise ValueError("Lifecycle phase received the wrong predecessor reference.")
    if phase == "model_decision":
        return _run_model_decision(spark, store, tracking_uri, reference)
    if phase == "complete":
        return _complete_invocation(spark, store, tracking_uri, reference, task_states)
    validate_active_phase(store, phase)
    return _execute_phase(spark, store, phase, reference)


def _selection_reference(
    store: PhaseStore, phase: str, reference: dict[str, str]
) -> dict[str, str]:
    """Join named candidate tasks only at the competition selection boundary."""
    if (
        phase != "select_best_model"
        or reference["phase"] != "prepare_dataset"
        or "competition" not in store.request
    ):
        return reference
    from ..training.training_nodes import join_competition_training  # noqa: PLC0415

    return join_competition_training(store, reference).reference


def _run_model_decision(
    spark: Any, store: PhaseStore, tracking_uri: str, reference: dict[str, str]
) -> LifecyclePhaseResult:
    """Route saved policy/operator intent through one visible decision task."""
    phase = "operator"
    if store.request["action"] == "train":
        phase = "decide"
        reference = store.reference(store.receipt("compare"))
    return run_lifecycle_phase(
        spark,
        phase=phase,
        context=store.context,
        tracking_uri=tracking_uri,
        reference=reference,
    )
