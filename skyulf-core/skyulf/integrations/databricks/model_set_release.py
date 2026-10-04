"""Apply saved component quality gates to manual and automatic model-set releases."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from skyulf.integrations.mlflow._client import make_registry_client, require_mlflow

from ...inference.model_set import ModelSetArtifact
from ..mlflow.model_set import load_registered_model_set
from ..mlflow.model_set_challenger import nominate_model_set
from ..mlflow.model_set_lifecycle import approve_model_set
from ..mlflow.promotion import ExclusiveAliasWriterAdmission, controlled_champion_version
from ..mlflow.registry import ResolvedModel, resolve_model
from ._contracts import input_budget_bytes
from .model_set_quality import evaluate_model_set_quality


class ModelSetQualityError(ValueError):
    """Preserve the full failed decision while leaving every registry alias unchanged."""

    def __init__(self, decision: dict) -> None:
        """Expose actionable component failures to automatic and operator callers."""
        self.decision = decision
        super().__init__(
            "Model set quality gates failed: " + ", ".join(decision["failed_components"])
        )


def champion_artifact(name: str, version: str | None, endpoints: dict) -> ModelSetArtifact | None:
    """Load the one captured set version instead of resolving component champion aliases."""
    if version is None:
        return None
    return load_registered_model_set(resolve_model(name, version=version, **endpoints), **endpoints)


def pin_model_set_baseline(
    settings: dict, configs: dict[str, dict], endpoints: dict
) -> tuple[dict, dict]:
    """Pin one champion set before training and require compatible component identities."""
    _automatic_thresholds(settings, configs)
    version = controlled_champion_version(settings["model_name"], **endpoints)
    champion = champion_artifact(settings["model_name"], version, endpoints)
    previous = {c.branch: c.reference for c in champion.manifest.components} if champion else {}
    if champion and set(previous) != set(configs):
        raise ValueError(
            "Model set replacement requires the same branch names; use a new set for a new layout."
        )
    for name, config in configs.items():
        if name in previous and previous[name].name != config["model_name"]:
            raise ValueError(f"Branch {name} model_name differs from the champion set.")
    versions = {name: previous[name].version if name in previous else None for name in configs}
    return {**settings, "expected_champion_version": version}, versions


def _automatic_thresholds(settings: dict, configs: dict[str, dict]) -> None:
    """Reject incomplete automatic policies before opening a registry or Spark reader."""
    if settings.get("promotion_policy") != "automatic":
        return
    for name, config in configs.items():
        if config.get("quality_threshold") is None:
            raise ValueError(
                f"Automatic model-set promotion requires quality_threshold for branch {name}."
            )


def persist_decision(resolved: ResolvedModel, decision: dict, policy: str, endpoints: dict) -> None:
    """Keep inspectable passing or failed decisions without using tags as metric evidence."""
    client = make_registry_client(
        require_mlflow(), endpoints["tracking_uri"], endpoints["registry_uri"]
    )
    run_id = client.get_model_version(resolved.name, resolved.version).run_id
    if not run_id:
        raise ValueError("Model-set quality decision requires a producing run.")
    payload = {"model_set_digest": resolved.digest, "promotion_policy": policy, **decision}
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(content.encode()).hexdigest()
    path = f"model_set_quality/{resolved.version}/{digest}/decision.json"
    client.log_dict(run_id, payload, path)
    downloaded = client.download_artifacts(run_id, path)
    if json.loads(Path(downloaded).read_text(encoding="utf-8")) != payload:
        raise ValueError("Saved model-set quality decision could not be verified.")
    tags = {
        "model_set_quality_status": "passed" if decision["passed"] else "failed",
        "model_set_quality_sha256": digest,
        "model_set_quality_artifact": f"runs:/{run_id}/{path}",
        "model_set_promotion_policy": policy,
        "validation_status": "passed" if decision["passed"] else "rejected",
        "validation_reason": "Set quality passed"
        if decision["passed"]
        else "Set quality gates failed",
    }
    for key, value in tags.items():
        client.set_model_version_tag(resolved.name, resolved.version, key, value)


def approve_project_model_set(
    spark: Any,
    resolved: ResolvedModel,
    config: dict,
    *,
    expected_champion_version: str | None,
    policy: str,
) -> Any:
    """Run functional and saved holdout checks inside the common alias admission."""
    from .model_set_project import (  # noqa: PLC0415 - lazy dependency  # noqa: PLC0415 - lazy dependency
        approval_frame,
        project_endpoints,
    )

    endpoints = project_endpoints(config)
    artifact = load_registered_model_set(resolved, **endpoints)
    champion = champion_artifact(resolved.name, expected_champion_version, endpoints)

    def validate(saved: ModelSetArtifact) -> dict:
        """Reject failed gates before the alias writer records any transition intent."""
        decision = evaluate_model_set_quality(
            spark,
            saved,
            champion,
            expected_champion_version=expected_champion_version,
            max_rows=config["max_rows"],
            max_bytes=input_budget_bytes(config.get("max_input_mb")),
            **endpoints,
        )
        persist_decision(resolved, decision, policy, endpoints)
        if not decision["passed"]:
            raise ModelSetQualityError(decision)
        return decision

    return approve_model_set(
        resolved,
        approval_frame(spark, artifact, config),
        expected_champion_version=expected_champion_version,
        admission=ExclusiveAliasWriterAdmission(),
        max_rows=config["max_rows"],
        max_bytes=input_budget_bytes(config.get("max_input_mb")),
        quality_validator=validate,
        **endpoints,
    )


def automatic_model_set_release(
    spark: Any, candidate: ResolvedModel, settings: dict, config: dict
) -> dict:
    """Nominate every complete set and automatically activate it only when configured."""
    from .model_set_project import (  # noqa: PLC0415 - lazy dependency
        project_endpoints,
    )

    nominate_model_set(
        candidate,
        expected_champion_version=settings["expected_champion_version"],
        admission=ExclusiveAliasWriterAdmission(),
        **project_endpoints(config),
    )
    if settings.get("promotion_policy", "manual_approval") != "automatic":
        return {"promotion_policy": "manual_approval", "alias_change": None}
    try:
        receipt = approve_project_model_set(
            spark,
            candidate,
            config,
            expected_champion_version=settings["expected_champion_version"],
            policy="automatic",
        )
    except ModelSetQualityError as exc:
        return {"promotion_policy": "automatic", "alias_change": None, "quality": exc.decision}
    return {
        "promotion_policy": "automatic",
        "alias_change": asdict(receipt),
        "quality_passed": True,
    }
