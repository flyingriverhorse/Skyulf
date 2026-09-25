"""Explicit operator rejection of an evaluated, registered challenger."""

import hashlib
import json
from dataclasses import asdict
from uuid import uuid4

from .promotion import (
    AliasAdmission,
    AliasChangeReceipt,
    AliasConflictError,
    _active_marker,
    _admission,
    _assert_no_pending,
    _checked_challenger_event,
    _commit_change,
    _read_optional_alias,
    _verify_original_receipt,
    _verify_staged_challenger,
    alias_resource_id,
)
from .registry import _make_client, _require_mlflow
from .validation import ModelComparisonReport


def reject_candidate(
    report: ModelComparisonReport,
    *,
    reason: str,
    expected_champion_version: str | None,
    admission: AliasAdmission,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> AliasChangeReceipt:
    """Record a manual decision without replacing aliases or altering evaluation metrics.

    Rejection requires the exact controlled staging evidence. It does not
    reread data or require a passing quality gate. A repeated request reuses
    its receipt only while champion, challenger, reason and proof still match.
    """
    if not isinstance(reason, str) or not reason.strip() or len(reason.encode("utf-8")) > 256:
        raise ValueError("Rejection reason must be nonempty text of at most 256 UTF-8 bytes.")
    if not isinstance(report, ModelComparisonReport):
        raise TypeError("Rejection requires a ModelComparisonReport.")
    if expected_champion_version != report.champion_version:
        raise ValueError("Expected champion must match the rejected comparison.")
    _admission(admission, registry_uri)
    digest = hashlib.sha256(
        json.dumps(asdict(report), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    client = _make_client(_require_mlflow(), tracking_uri, registry_uri)
    name = report.model_name
    version = report.candidate_version
    with admission.hold(alias_resource_id(name)):
        _assert_no_pending(client, name)
        if _read_optional_alias(client, name, "champion") != expected_champion_version:
            raise AliasConflictError("Champion changed before rejection.")
        if (
            _read_optional_alias(client, name, "challenger") != version
            or version == expected_champion_version
        ):
            raise AliasConflictError("Challenger changed before rejection.")
        event = _checked_challenger_event(client, name, version)
        if event["k"] == "rejection":
            tags = client.get_model_version(name, version).tags or {}
            if (
                event.get("h") != digest
                or tags.get("approval_status") != "rejected"
                or tags.get("approval_reason") != reason
            ):
                raise AliasConflictError(
                    "Rejection proof, status or reason differs from the recorded decision."
                )
            event_id = _active_marker(client, name, version, "challenger")
            if event_id is None:
                raise AliasConflictError("Rejected challenger lacks an active decision receipt.")
            receipt = AliasChangeReceipt(
                event_id=event_id,
                kind="rejection",
                model_name=name,
                alias="challenger",
                prior_version=version,
                new_version=version,
                comparison_sha256=digest,
                parent_event_id=event.get("e"),
            )
            _verify_original_receipt(client, receipt)
            return receipt
        stage_event = _verify_staged_challenger(client, report, digest)
        receipt = AliasChangeReceipt(
            event_id=uuid4().hex,
            kind="rejection",
            model_name=name,
            alias="challenger",
            prior_version=version,
            new_version=version,
            comparison_sha256=digest,
            parent_event_id=stage_event,
        )
        _commit_change(
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
