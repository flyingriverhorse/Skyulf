"""Approve or reject a fitted pipeline using its pinned evaluation evidence.

The caller must serialize this operation with every other lifecycle writer.
Approval never fits, registers or uploads a model and never changes a scoring pin.
"""

import re
from dataclasses import replace
from typing import Any

import polars as pl

from skyulf.integrations.mlflow.shared._client import make_registry_client, require_mlflow

from ...mlflow.lifecycle.promotion import (
    AliasChangeReceipt,
    AliasConflictError,
    ExclusiveAliasWriterAdmission,
    active_marker,
    controlled_champion_version,
    event_tag,
    initialize_champion,
    promote_candidate,
    read_event,
    verify_original_receipt,
    verify_staged_challenger,
)
from ...mlflow.lifecycle.rejection import reject_candidate
from ...mlflow.lifecycle.validation import ModelComparisonReport, validate_quality_policy
from ...mlflow.registration.registry import resolve_model
from ..shared._contracts import input_budget_bytes
from ..training.fitting import candidate
from ..training.shared.training_evidence import (
    load_candidate_evidence,
    validate_training_evidence,
)


def resolve_candidate_comparison_digest(
    config: dict[str, Any], candidate_version: str, *, action: str
) -> str:
    """Resolve a named candidate's proof from its active committed lifecycle receipt.

    This is the Bundle convenience path, not a latest-candidate lookup. The
    strict approval/rejection service still verifies the downloaded comparison,
    expected champion and active receipt against the full returned digest.
    """
    if config.get("promotion_policy") != "manual_approval" or action not in {"approve", "reject"}:
        raise ValueError("Evidence lookup requires manual_approval and approve/reject.")
    if not isinstance(candidate_version, str) or not re.fullmatch(
        r"[1-9][0-9]*", candidate_version
    ):
        raise ValueError("Evidence lookup requires a concrete candidate_version.")
    tracking_uri = config.get("tracking_uri", "databricks")
    registry_uri = config.get("registry_uri", "databricks-uc")
    name = config["model_name"]
    current = controlled_champion_version(
        name, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
    replay = action == "approve" and current == candidate_version
    alias = "champion" if replay else "challenger"
    event_id = active_marker(client, name, candidate_version, alias)
    if event_id is None:
        raise AliasConflictError("Candidate has no active saved comparison receipt.")
    raw = (client.get_model_version(name, candidate_version).tags or {}).get(event_tag(event_id))
    try:
        event = read_event(raw)
    except (TypeError, ValueError) as exc:
        raise AliasConflictError("Saved comparison receipt is malformed.") from exc
    return _comparison_receipt_digest(event, action, replay)


def _completed_approval(
    client: Any, report: ModelComparisonReport, digest: str
) -> AliasChangeReceipt:
    """Replay only the same still-active committed transition, never a later rollback."""
    event_id = active_marker(client, report.model_name, report.candidate_version)
    if event_id is None:
        raise AliasConflictError("Candidate lacks an active approval receipt.")
    raw = client.get_model_version(report.model_name, report.candidate_version).tags.get(
        event_tag(event_id)
    )
    event = read_event(raw)
    if not isinstance(event, dict) or (
        event.get("k") not in {"initial", "promotion"}
        or event.get("h") != digest
        or event.get("p") != report.champion_version
    ):
        raise AliasConflictError("Active champion receipt differs from requested approval.")
    receipt = AliasChangeReceipt(
        event_id=event_id,
        kind=event["k"],
        model_name=report.model_name,
        alias="champion",
        prior_version=event["p"],
        new_version=report.candidate_version,
        comparison_sha256=digest,
        parent_event_id=event.get("e"),
        previous_champion_version=event.get("o"),
    )
    verify_original_receipt(client, receipt)
    return receipt


def _validate_candidate_request(
    candidate_version: str | None,
    comparison_sha256: str | None,
    expected_champion_version: str | None,
) -> tuple[str, str]:
    """Reject ambiguous candidate and evidence pins before accessing the registry."""
    if not isinstance(candidate_version, str) or not re.fullmatch(
        r"[1-9][0-9]*", candidate_version
    ):
        raise ValueError("Approval requires a concrete candidate_version.")
    if not isinstance(comparison_sha256, str) or not re.fullmatch(
        r"[a-f0-9]{64}", comparison_sha256
    ):
        raise ValueError("Approval requires a comparison_sha256 evidence digest.")
    if expected_champion_version is not None and (
        not isinstance(expected_champion_version, str)
        or not re.fullmatch(r"[1-9][0-9]*", expected_champion_version)
    ):
        raise ValueError("Expected champion must be a concrete version or None for bootstrap.")
    return candidate_version, comparison_sha256


def reject_workflow_candidate(
    config: dict[str, Any],
    *,
    candidate_version: str | None,
    comparison_sha256: str | None,
    expected_champion_version: str | None,
    rejection_reason: str,
) -> AliasChangeReceipt:
    """Reject saved candidate evidence without fitting, reading rows or changing a scoring pin."""
    if config.get("promotion_policy") != "manual_approval":
        raise ValueError("Rejection requires explicit promotion_policy=manual_approval.")
    candidate_version, comparison_sha256 = _validate_candidate_request(
        candidate_version, comparison_sha256, expected_champion_version
    )
    tracking_uri = config.get("tracking_uri", "databricks")
    registry_uri = config.get("registry_uri", "databricks-uc")
    client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
    report, _, _, _ = load_candidate_evidence(
        client,
        config["model_name"],
        candidate_version,
        comparison_sha256,
        registry_uri=registry_uri,
    )
    if (
        controlled_champion_version(
            config["model_name"], tracking_uri=tracking_uri, registry_uri=registry_uri
        )
        != expected_champion_version
    ):
        raise AliasConflictError("Champion changed before rejection.")
    return reject_candidate(
        report,
        reason=rejection_reason,
        expected_champion_version=expected_champion_version,
        admission=ExclusiveAliasWriterAdmission(),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )


def approve_candidate(
    spark: Any,
    config: dict[str, Any],
    *,
    candidate_version: str | None,
    comparison_sha256: str | None,
    expected_champion_version: str | None,
) -> AliasChangeReceipt:
    """Re-evaluate the saved holdout and approve without retraining or changing score selection.

    The current metric policy must match the saved comparison. Read limits may
    be tightened; the source version and time window come from the candidate's
    saved training specification. A repeated committed request returns its
    original receipt only while that transition remains active.
    """
    if config.get("promotion_policy") != "manual_approval":
        raise ValueError("Approval requires explicit promotion_policy=manual_approval.")
    candidate_version, comparison_sha256 = _validate_candidate_request(
        candidate_version, comparison_sha256, expected_champion_version
    )
    if type(config.get("max_rows")) is not int or config["max_rows"] <= 0:
        raise ValueError("Approval max_rows must be a positive integer.")
    max_bytes = input_budget_bytes(config.get("max_input_mb"))
    tracking_uri = config.get("tracking_uri", "databricks")
    registry_uri = config.get("registry_uri", "databricks-uc")
    name = config["model_name"]
    client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
    report, spec, engine, filter_evidence = load_candidate_evidence(
        client, name, candidate_version, comparison_sha256, registry_uri=registry_uri
    )
    _validate_approval_policy(config, report, expected_champion_version)
    resolved_candidate = resolve_model(
        name, version=candidate_version, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    if resolved_candidate.digest != report.candidate_digest:
        raise ValueError("Candidate model digest differs from saved comparison.")
    current = controlled_champion_version(
        name, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    if current == candidate_version:
        return _completed_approval(client, report, comparison_sha256)
    if current != expected_champion_version:
        raise AliasConflictError("Champion changed since the requested comparison.")
    verify_staged_challenger(client, report, comparison_sha256)
    bounded_spec = replace(
        spec,
        max_rows=min(spec.max_rows, config["max_rows"]),
        max_bytes=min(spec.max_bytes, max_bytes),
    )
    frame = candidate.read_training_snapshot(spark, bounded_spec)
    _, heldout, _ = candidate.split_labeled_snapshot(frame, bounded_spec, engine=engine)
    if filter_evidence is not None:
        validate_training_evidence(
            filter_evidence,
            spec,
            project_source_sha256=filter_evidence["project_source_sha256"],
            heldout=heldout,
        )
    native = pl.from_pandas(heldout) if engine == "polars" else heldout
    options = {
        "target_column": spec.target_column,
        "admission": ExclusiveAliasWriterAdmission(),
        "max_rows": bounded_spec.max_rows,
        "max_bytes": bounded_spec.max_bytes,
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
    }
    if expected_champion_version is None:
        return initialize_champion(report, native, **options)
    return promote_candidate(
        report, native, expected_champion_version=expected_champion_version, **options
    )


def _comparison_receipt_digest(event: Any, action: str, replay: bool) -> str:
    """Validate the committed receipt kind and return its concrete comparison digest."""
    kinds = {"initial", "promotion"} if replay else {"challenger"}
    if action == "approve" and isinstance(event, dict) and event.get("k") == "rejection":
        raise AliasConflictError("Candidate was explicitly rejected.")
    if action == "reject":
        kinds.add("rejection")
    if not isinstance(event, dict) or event.get("s") != "committed" or event.get("k") not in kinds:
        raise AliasConflictError("Candidate has no committed comparison for this action.")
    return _receipt_evidence_digest(event)


def _receipt_evidence_digest(event: dict[str, Any]) -> str:
    """Require a concrete comparison digest from a validated committed receipt."""
    digest = event.get("h")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise AliasConflictError("Saved comparison receipt has no valid digest.")
    return digest


def _validate_approval_policy(
    config: dict[str, Any], report: ModelComparisonReport, expected_champion_version: str | None
) -> None:
    """Bind current approval settings to the saved champion and absolute quality policy."""
    if report.champion_version != expected_champion_version:
        raise ValueError("Expected champion differs from the saved comparison.")
    validate_quality_policy(
        config.get("metric", ""), config.get("quality_threshold"), config.get("quality_gates")
    )
    if (
        any(
            config.get(field) != getattr(report, field)
            for field in ("metric", "min_improvement", "quality_threshold")
        )
        or (config.get("quality_gates") or None) != report.quality_gates
        or report.quality_threshold is None
    ):
        raise ValueError("Approval policy must match the saved, absolute-quality-gated comparison.")
