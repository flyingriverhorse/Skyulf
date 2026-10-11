"""Promote only an explicitly enabled, durably completed pipeline rollout."""

import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import partial
from typing import Any, cast

from ...mlflow.lifecycle.promotion import AliasAdmission, AliasChangeReceipt
from ...mlflow.registration.registry import load_registered_pipeline, resolve_model
from ...mlflow.shared._client import make_registry_client, require_mlflow
from ..data.admission import PublishAdmission
from ..lifecycle.approval import approve_saved_candidate
from ..shared.json_contracts import finite_json_digest
from ..training.shared.training_evidence import load_candidate_evidence
from .rollout import hold_completed_rollout
from .rollout_endpoints import RolloutEndpointPlan
from .rollout_policy import DailyRolloutPolicy, RolloutEvidence, RolloutState, decide_rollout
from .rollout_store import MLflowRolloutStore

_APPROVAL_FIELDS = (
    "model_name",
    "metric",
    "min_improvement",
    "quality_threshold",
    "quality_gates",
    "max_rows",
    "max_input_mb",
    "tracking_uri",
    "registry_uri",
)
_PIN_FIELDS = {
    "auto_promote",
    "model_name",
    "candidate_version",
    "expected_champion_version",
    "comparison_sha256",
    "config_sha256",
}


def _approval_config(config: dict[str, Any]) -> dict[str, Any]:
    """Retain only approval settings, without saving arbitrary job configuration."""
    if not isinstance(config, dict):
        raise TypeError("Approval config must be a dictionary.")
    return {key: deepcopy(config[key]) for key in _APPROVAL_FIELDS if key in config}


def approval_config_digest(config: dict[str, Any]) -> str:
    """Bind approval policy, read budgets and registry selection without storing them."""
    return finite_json_digest(_approval_config(config))


def _digest(value: Any) -> None:
    """Require complete lowercase SHA256 evidence identifiers."""
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("Promotion requires a concrete SHA256 evidence digest.")


def _validate_pipeline_pair(plan: RolloutEndpointPlan, config: dict[str, Any]) -> None:
    """Reject model sets and foreign registry names before rollout initialization."""
    champion, challenger = plan.champion.spec, plan.challenger.spec
    if (
        champion.model_name != challenger.model_name
        or config.get("model_name") != champion.model_name
    ):
        raise ValueError("Automatic promotion requires the same registered model name.")
    options = {
        "tracking_uri": config.get("tracking_uri", "databricks"),
        "registry_uri": config.get("registry_uri", "databricks-uc"),
    }
    for spec in (champion, challenger):
        resolved = resolve_model(spec.model_name, version=spec.model_version, **options)
        load_registered_pipeline(resolved, **options)


def build_rollout_promotion(
    plan: RolloutEndpointPlan,
    config: dict[str, Any],
    *,
    auto_promote: bool,
    comparison_sha256: str,
) -> dict[str, Any] | None:
    """Prepare a saved pipeline-only approval pin before initializing traffic.

    Both concrete packages must load as fitted pipelines. Model-set promotion
    uses a different lifecycle and is not admitted by this automatic path.
    Only a digest of relevant configuration is persisted, never credentials.
    """
    if type(auto_promote) is not bool:
        raise ValueError("auto_promote must be an explicit bool.")
    if not auto_promote:
        return None
    _digest(comparison_sha256)
    config = _approval_config(config)
    _validate_pipeline_pair(plan, config)
    tracking_uri = config.get("tracking_uri", "databricks")
    registry_uri = config.get("registry_uri", "databricks-uc")
    client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
    spec = plan.challenger.spec
    report, _, _, _ = load_candidate_evidence(
        client, spec.model_name, spec.model_version, comparison_sha256, registry_uri=registry_uri
    )
    if report.champion_version != plan.champion.spec.model_version or not report.eligible:
        raise ValueError("Saved comparison does not approve this rollout model pair.")
    return {
        "auto_promote": True,
        "model_name": spec.model_name,
        "candidate_version": spec.model_version,
        "expected_champion_version": plan.champion.spec.model_version,
        "comparison_sha256": comparison_sha256,
        "config_sha256": approval_config_digest(config),
    }


def _promotion_pin(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Bind explicit opt-in and the original evidence to the verified completed pair."""
    pin = record.get("promotion")
    if type(pin) is not dict or set(pin) != _PIN_FIELDS or pin.get("auto_promote") is not True:
        raise ValueError("Rollout requires an explicitly enabled saved automatic promotion pin.")
    _digest(pin["comparison_sha256"])
    if pin["config_sha256"] != approval_config_digest(config):
        raise ValueError("Approval configuration differs from the initialized rollout.")
    state = RolloutState(**record["state"])
    if state.phase != "COMPLETE" or state.challenger_percentage != 100:
        raise ValueError("Promotion requires a verified completed 100-percent rollout.")
    if state.champion_model_name != state.challenger_model_name:
        raise ValueError("Automatic promotion requires the same registered model name.")
    expected = (
        state.champion_model_name,
        state.challenger_model_name,
        state.champion_model_version,
        state.challenger_model_version,
    )
    actual = (
        pin["model_name"],
        config.get("model_name"),
        pin["expected_champion_version"],
        pin["candidate_version"],
    )
    if expected != actual:
        raise ValueError("Automatic promotion pin differs from the completed model pair.")
    return deepcopy(pin)


def _fresh_completion(record: dict[str, Any], clock: Callable[[], datetime]) -> None:
    """Recheck final-stage evidence only before a new alias mutation, never on replay."""
    if record.get("promotion_receipt") is not None:
        raise ValueError("Saved rollout promotion is no longer current; refusing a new mutation.")
    state = replace(RolloutState(**record["state"]), phase="ACTIVE")
    evidence = RolloutEvidence(**record["evidence"])
    result = decide_rollout(DailyRolloutPolicy(**record["policy"]), state, evidence, now=clock())
    if result.action != "COMPLETE":
        raise ValueError(f"Final rollout evidence does not permit promotion: {result.reason}")


def _receipt_matches(receipt: AliasChangeReceipt, pin: dict[str, Any]) -> None:
    """Do not acknowledge a different lifecycle operation as this rollout's promotion."""
    expected = (
        "promotion",
        "champion",
        pin["model_name"],
        pin["expected_champion_version"],
        pin["candidate_version"],
        pin["comparison_sha256"],
    )
    actual = (
        receipt.kind,
        receipt.alias,
        receipt.model_name,
        receipt.prior_version,
        receipt.new_version,
        receipt.comparison_sha256,
    )
    if actual != expected:
        raise ValueError("Alias receipt differs from the initialized rollout promotion.")


def _now() -> datetime:
    """Use server UTC time for the mandatory fresh-completion guard."""
    return datetime.now(UTC)


def promote_completed_rollout(
    client: Any,
    spark: Any,
    config: dict[str, Any],
    *,
    store: MLflowRolloutStore,
    admission: PublishAdmission,
    clock: Callable[[], datetime] = _now,
) -> AliasChangeReceipt:
    """Verify durable completion and native endpoint state before guarded alias approval.

    The endpoint claim stays held through approval and receipt persistence.
    Saved approval acquires the same authority's distinct alias resource key;
    no same-key nested acquisition occurs. Uncertain alias outcomes preserve the
    existing pending-event reconciliation requirement. Snapshot errors propagate
    without weakening the original training feature-history checks.
    """
    config = _approval_config(config)
    with hold_completed_rollout(client, store=store, admission=admission) as record:
        pin = _promotion_pin(record, config)
        receipt = approve_saved_candidate(
            spark,
            config,
            candidate_version=pin["candidate_version"],
            comparison_sha256=pin["comparison_sha256"],
            expected_champion_version=pin["expected_champion_version"],
            # Both authorities call hold positionally; only protocol parameter names differ.
            admission=cast(AliasAdmission, admission),
            fresh_approval_guard=partial(_fresh_completion, record, clock),
        )
        _receipt_matches(receipt, pin)
        saved = record.get("promotion_receipt")
        if saved is not None:
            if saved != asdict(receipt):
                raise ValueError("Active alias receipt differs from the saved rollout receipt.")
            return receipt
        store.write(
            record | {"promotion_receipt": asdict(receipt)},
            expected_receipt_id=record["receipt_id"],
        )
        return receipt
