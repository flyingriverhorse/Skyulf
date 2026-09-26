"""Approve or reject an existing local model using its pinned evaluation evidence.

The caller must serialize this operation with every other lifecycle writer.
Approval never fits, registers or uploads a model and never changes a scoring pin.
"""

import hashlib
import json
import re
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl

from ...inference.project_code import load_project_module, project_source_digest
from ..mlflow.promotion import (
    AliasChangeReceipt,
    AliasConflictError,
    ExclusiveAliasWriterAdmission,
    _active_marker,
    _event_tag,
    _read_event,
    _verify_original_receipt,
    _verify_staged_challenger,
    controlled_champion_version,
    initialize_champion,
    promote_candidate,
)
from ..mlflow.registry import (
    _make_client,
    _require_mlflow,
    load_registered_local_pipeline,
    resolve_model,
)
from ..mlflow.rejection import reject_candidate
from ..mlflow.validation import ModelComparisonReport
from . import local_retraining
from ._contracts import input_budget_bytes
from .local_retraining import LocalTrainingSpec
from .local_training_evidence import validate_training_evidence
from .training_dates import training_date_spec


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
    client = _make_client(_require_mlflow(), tracking_uri, registry_uri)
    replay = action == "approve" and current == candidate_version
    alias = "champion" if replay else "challenger"
    event_id = _active_marker(client, name, candidate_version, alias)
    if event_id is None:
        raise AliasConflictError("Candidate has no active saved comparison receipt.")
    raw = (client.get_model_version(name, candidate_version).tags or {}).get(_event_tag(event_id))
    try:
        event = _read_event(raw)
    except (TypeError, ValueError) as exc:
        raise AliasConflictError("Saved comparison receipt is malformed.") from exc
    kinds = {"initial", "promotion"} if replay else {"challenger"}
    if action == "approve" and isinstance(event, dict) and event.get("k") == "rejection":
        raise AliasConflictError("Candidate was explicitly rejected.")
    if action == "reject":
        kinds.add("rejection")
    if not isinstance(event, dict) or event.get("s") != "committed" or event.get("k") not in kinds:
        raise AliasConflictError("Candidate has no committed comparison for this action.")
    digest = event.get("h")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise AliasConflictError("Saved comparison receipt has no valid digest.")
    return digest


def _load_evidence(
    client: Any, name: str, version: str, digest: str, *, registry_uri: str | None = None
) -> tuple[ModelComparisonReport, LocalTrainingSpec, str, dict[str, Any] | None]:
    """Read only the named version's run artifacts and verify the operator's evidence pin."""
    model = client.get_model_version(name, version)
    if not model.run_id:
        raise ValueError("Candidate has no training run with approval evidence.")
    with TemporaryDirectory(prefix="skyulf-approval-") as directory:
        report_path = client.download_artifacts(
            model.run_id, "candidate_comparison.json", directory
        )
        spec_path = client.download_artifacts(
            model.run_id, "candidate_training_spec.json", directory
        )
        report = ModelComparisonReport(**json.loads(Path(report_path).read_text(encoding="utf-8")))
        saved_spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        saved_filter_evidence = None
        if saved_spec.get("training_evidence_sha256") is not None:
            evidence_path = client.download_artifacts(
                model.run_id, "training_filter_evidence.json", directory
            )
            saved_filter_evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
            if not isinstance(saved_filter_evidence, dict):
                raise ValueError("Saved training filter evidence must be a JSON object.")
    actual = hashlib.sha256(
        json.dumps(asdict(report), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if actual != digest:
        raise ValueError("Saved comparison digest differs from requested approval evidence.")
    if report.model_name != name or report.candidate_version != version:
        raise ValueError("Saved comparison does not identify the requested candidate.")
    engine = saved_spec.pop("engine")
    if engine not in ("pandas", "polars"):
        raise ValueError("Saved approval engine must be pandas or polars.")
    source_sha = None
    recipe = None
    if saved_filter_evidence is not None:
        tracking_uri = getattr(client, "tracking_uri", None)
        reference = resolve_model(
            name,
            version=version,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri or tracking_uri,
        )
        artifact = load_registered_local_pipeline(
            reference, tracking_uri=tracking_uri, registry_uri=registry_uri or tracking_uri
        )
        if engine != artifact.manifest.fitted_engine:
            raise ValueError("Saved approval engine differs from fitted model engine.")
        source_sha = artifact.manifest.project_source_sha256
        if source_sha is not None:
            source = artifact.pipeline.config["project_python_source"]
            if project_source_digest(source) != source_sha:
                raise ValueError("Saved project source differs from model manifest.")
            factory = getattr(load_project_module(source), "build_pre_split_steps", None)
            recipe = factory() if factory is not None else []
    for field in ("start", "holdout_start", "cutoff", "result_cutoff"):
        value = saved_spec[field]
        saved_spec[field] = None if value is None else datetime.fromisoformat(value)
    for field in ("record_key_columns", "input_columns"):
        saved_spec[field] = tuple(saved_spec[field])
    saved_spec["pre_split_steps"] = tuple(saved_spec.get("pre_split_steps", ()))
    for field in ("event_time_parsing", "result_time_parsing"):
        saved_spec[field] = training_date_spec(saved_spec[field])
    spec = LocalTrainingSpec(**saved_spec)
    if spec.holdout_key_sha256 is None:
        raise ValueError("Saved training evidence requires holdout membership proof.")
    if spec.dataset_id != report.dataset_id:
        raise ValueError("Saved training snapshot differs from comparison evidence.")
    if saved_filter_evidence is not None:
        validate_training_evidence(saved_filter_evidence, spec, project_source_sha256=source_sha)
        if source_sha is not None and recipe != list(spec.pre_split_steps):
            raise ValueError("Saved project source recipe differs from training evidence.")
    return report, spec, engine, saved_filter_evidence


def _completed_approval(
    client: Any, report: ModelComparisonReport, digest: str
) -> AliasChangeReceipt:
    """Replay only the same still-active committed transition, never a later rollback."""
    event_id = _active_marker(client, report.model_name, report.candidate_version)
    if event_id is None:
        raise AliasConflictError("Candidate lacks an active approval receipt.")
    raw = client.get_model_version(report.model_name, report.candidate_version).tags.get(
        _event_tag(event_id)
    )
    event = _read_event(raw)
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
    _verify_original_receipt(client, receipt)
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


def reject_local_candidate(
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
    client = _make_client(_require_mlflow(), tracking_uri, registry_uri)
    report, _, _, _ = _load_evidence(
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


def approve_local_candidate(
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
    client = _make_client(_require_mlflow(), tracking_uri, registry_uri)
    report, spec, engine, filter_evidence = _load_evidence(
        client, name, candidate_version, comparison_sha256, registry_uri=registry_uri
    )
    if report.champion_version != expected_champion_version:
        raise ValueError("Expected champion differs from the saved comparison.")
    if (
        any(
            config.get(field) != getattr(report, field)
            for field in ("metric", "min_improvement", "quality_threshold")
        )
        or report.quality_threshold is None
    ):
        raise ValueError("Approval policy must match the saved, absolute-quality-gated comparison.")
    candidate = resolve_model(
        name, version=candidate_version, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    if candidate.digest != report.candidate_digest:
        raise ValueError("Candidate model digest differs from saved comparison.")
    current = controlled_champion_version(
        name, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    if current == candidate_version:
        return _completed_approval(client, report, comparison_sha256)
    if current != expected_champion_version:
        raise AliasConflictError("Champion changed since the requested comparison.")
    _verify_staged_challenger(client, report, comparison_sha256)
    bounded_spec = replace(
        spec,
        max_rows=min(spec.max_rows, config["max_rows"]),
        max_bytes=min(spec.max_bytes, max_bytes),
    )
    frame = local_retraining.read_training_snapshot(spark, bounded_spec)
    _, heldout, _ = local_retraining.split_labeled_snapshot(frame, bounded_spec, engine=engine)
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
