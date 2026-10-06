"""Nomination and explicit rejection of complete, immutable model sets."""

from typing import Any
from uuid import uuid4

from skyulf.integrations.mlflow.shared._client import make_registry_client, require_mlflow

from ..models.model_set import load_registered_model_set
from ..registration.registry import ResolvedModel
from .challenger import ChallengerLifecycle
from .promotion import (
    AliasAdmission,
    AliasChangeReceipt,
    AliasConflictError,
    active_marker,
    alias_resource_id,
    checked_challenger_event,
    commit_change,
    controlled_champion_version,
    read_optional_alias,
    validate_admission,
)
from .rejection import replayed_rejection


def _saved_baseline(resolved: ResolvedModel, expected: str | None, endpoints: dict) -> None:
    """Require a genuine set and preserve the baseline frozen during its training."""
    artifact = load_registered_model_set(resolved, **endpoints)
    evidence = artifact.manifest.quality_evidence
    if evidence is not None and evidence["expected_champion_version"] != expected:
        raise AliasConflictError("Model set baseline differs from expected champion.")


def nominate_model_set(
    resolved: ResolvedModel,
    *,
    expected_champion_version: str | None,
    admission: AliasAdmission,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> None:
    """Make a complete candidate challenger and archive the displaced contender."""
    endpoints = {"tracking_uri": tracking_uri, "registry_uri": registry_uri}
    _saved_baseline(resolved, expected_champion_version, endpoints)
    lifecycle = ChallengerLifecycle(
        resolved.name,
        expected_champion_version=expected_champion_version,
        admission=admission,
        **endpoints,
    )
    lifecycle.registered(resolved)


def verify_model_set_challenger(client: Any, resolved: ResolvedModel, *, required: bool) -> None:
    """Reject stale or manually altered contenders before quality checks or alias changes."""
    current = read_optional_alias(client, resolved.name, "challenger")
    if current is None and not required:
        return
    if current != resolved.version:
        raise AliasConflictError("Challenger alias differs from model-set candidate version.")
    event = checked_challenger_event(client, resolved.name, resolved.version)
    if event["k"] != "evaluation_error" and event["h"] != resolved.digest:
        raise AliasConflictError("Challenger receipt differs from the model-set digest.")


def reject_model_set(
    resolved: ResolvedModel,
    *,
    reason: str,
    expected_champion_version: str | None,
    admission: AliasAdmission,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> AliasChangeReceipt:
    """Reject the exact controlled set contender without running scoring or moving aliases.

    Repeated rejection returns the original receipt only while candidate, champion,
    artifact and reason still match. Rejected sets cannot subsequently be approved.
    """
    if not isinstance(reason, str) or not reason.strip() or len(reason.encode("utf-8")) > 256:
        raise ValueError("Rejection reason must be nonempty text of at most 256 UTF-8 bytes.")
    validate_admission(admission, registry_uri)
    endpoints = {"tracking_uri": tracking_uri, "registry_uri": registry_uri}
    _saved_baseline(resolved, expected_champion_version, endpoints)
    client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
    with admission.hold(alias_resource_id(resolved.name)):
        current = controlled_champion_version(resolved.name, **endpoints)
        if current != expected_champion_version or current == resolved.version:
            raise AliasConflictError("Champion changed before model-set rejection.")
        verify_model_set_challenger(client, resolved, required=True)
        return _reject_current(client, resolved, reason)


def _reject_current(client: Any, resolved: ResolvedModel, reason: str) -> AliasChangeReceipt:
    """Persist or replay the rejection using the common alias-event protocol."""
    event = checked_challenger_event(client, resolved.name, resolved.version)
    if event["k"] == "rejection":
        return replayed_rejection(
            client, resolved.name, resolved.version, event, str(resolved.digest), reason
        )
    receipt = AliasChangeReceipt(
        event_id=uuid4().hex,
        kind="rejection",
        model_name=resolved.name,
        alias="challenger",
        prior_version=resolved.version,
        new_version=resolved.version,
        comparison_sha256=resolved.digest,
        parent_event_id=active_marker(client, resolved.name, resolved.version, "challenger"),
    )
    commit_change(
        client,
        receipt,
        [],
        version_tags={
            "approval_status": "rejected",
            "approval_reason": reason,
            "promotion_status": "not_promoted",
        },
    )
    return receipt
