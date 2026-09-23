"""Version-checked MLflow champion promotion with shared admission and receipts.

Alias writes are not compare-and-swap. Every writer must use the same non-expiring
admission and registry writes must be restricted to those writers. A prepared
receipt survives an uncertain mutation, which requires operator reconciliation.
"""

import hashlib
import json
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import pandas as pd
import polars as pl

from ...integrations.databricks.admission import BatchConflictError, LocalTableLock
from ...integrations.databricks.delta_admission import DeltaTableAdmission
from .registry import (
    RegistryError,
    RegistryOperationError,
    _error_code,
    _make_client,
    _require_mlflow,
    _translate_error,
    resolve_model,
)
from .validation import ModelComparisonReport, compare_registered_local_models

_ALIAS = "champion"
_CHALLENGER = "challenger"
_PREVIOUS = "previous_champion"
_ACTIVE_TAG = "champion_current_event"
_CHALLENGER_TAG = "challenger_current_event"


class AliasConflictError(RegistryError):
    """The champion alias or shared admission disagrees with the expected state."""


class AliasOutcomeUnknownError(RegistryOperationError):
    """Alias mutation may have succeeded but its final receipt was not verified."""


class AliasAdmission(Protocol):
    """Hold one non-expiring shared resource claim through alias verification."""

    local_only: bool

    def hold(self, resource_id: str) -> AbstractContextManager[None]:
        """Reject a competing alias writer and release after the receipt is durable."""
        ...


def alias_resource_id(model_name: str, alias: str = _ALIAS) -> str:
    """Derive a stable, unambiguous control-row key for one model alias."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be nonempty.")
    if alias != _ALIAS:
        raise ValueError("Only the champion alias is supported.")
    encoded = json.dumps([model_name, alias], separators=(",", ":")).encode()
    return "mlflow-alias:" + hashlib.sha256(encoded).hexdigest()


class LocalAliasAdmission:
    """Coordinate test/development writers sharing one host and lock directory."""

    local_only = True

    def __init__(self, directory: str | Path) -> None:
        """Reuse the tested non-expiring OS file-lock implementation."""
        self._lock = LocalTableLock(directory)

    @contextmanager
    def hold(self, resource_id: str) -> Iterator[None]:
        """Translate local lock contention to an alias conflict."""
        try:
            with self._lock.hold(resource_id):
                yield
        except BatchConflictError as exc:
            raise AliasConflictError("Another writer holds alias admission.") from exc


class DeltaAliasAdmission:
    """Coordinate Databricks writers through a preprovisioned Delta control row.

    Provision exactly one target_id/owner row with target_id equal to
    alias_resource_id(model_name). Keep this authority separate from prediction
    table admission. The underlying Delta claim never expires automatically.
    """

    local_only = False

    def __init__(self, spark: Any, control_table: str) -> None:
        """Bind a pre-existing control table without creating resources."""
        self._lock = DeltaTableAdmission(spark, control_table)

    @contextmanager
    def hold(self, resource_id: str) -> Iterator[None]:
        """Translate distributed Delta contention to an alias conflict."""
        try:
            with self._lock.hold(resource_id):
                yield
        except BatchConflictError as exc:
            raise AliasConflictError("Another writer holds alias admission.") from exc


@dataclass(frozen=True, slots=True)
class AliasChangeReceipt:
    """Record a verified alias transition and its durable registry event."""

    event_id: str
    kind: str
    model_name: str
    alias: str
    prior_version: str | None
    new_version: str
    comparison_sha256: str | None
    parent_event_id: str | None
    previous_champion_version: str | None = None


def _admission(admission: AliasAdmission, registry_uri: str | None) -> AliasAdmission:
    """Reject missing or host-local admission for a Unity Catalog writer."""
    if not callable(getattr(admission, "hold", None)) or not isinstance(
        getattr(admission, "local_only", None), bool
    ):
        raise ValueError("A shared alias admission is required.")
    effective_registry_uri = registry_uri or _require_mlflow().get_registry_uri()
    if effective_registry_uri.startswith("databricks-uc") and admission.local_only:
        raise ValueError("Unity Catalog promotion requires distributed alias admission.")
    return admission


def _read_optional_alias(client: Any, name: str, alias: str) -> str | None:
    """Resolve one alias, distinguishing a missing alias from registry failures."""
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except Exception as exc:  # noqa: BLE001 - MLflow backends expose different error classes
        missing_alias = _error_code(exc) in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"} or (
            _error_code(exc) == "INVALID_PARAMETER_VALUE"
            and "alias" in str(exc).lower()
            and "not found" in str(exc).lower()
        )
        if missing_alias:
            return None
        raise _translate_error(exc, name=name, version=alias) from exc


def _read_alias(client: Any, name: str) -> str:
    """Read the current champion without creating a missing alias."""
    current = _read_optional_alias(client, name, _ALIAS)
    if current is None:
        raise AliasConflictError("Champion alias is missing; initialize it explicitly.")
    return current


def _active_marker(client: Any, name: str, current: str, alias: str = _ALIAS) -> str | None:
    """Reject a bypass write that left the active receipt on another version."""
    tag = _ACTIVE_TAG if alias == _ALIAS else _CHALLENGER_TAG
    try:
        raw = (client.get_registered_model(name).tags or {}).get(tag)
    except Exception as exc:  # noqa: BLE001 - registry transport boundary
        raise _translate_error(exc, name=name, version=current) from exc
    if raw is None:
        return None
    try:
        marker = json.loads(raw)
        if type(marker) is not dict or marker["version"] != current:
            raise ValueError
        event_id = marker["event_id"]
        if type(event_id) is not str or not event_id:
            raise ValueError
    except (TypeError, ValueError, KeyError) as exc:
        raise AliasConflictError(f"Active receipt disagrees with {alias} alias.") from exc
    return event_id


def _event_tag(event_id: str) -> str:
    """Name an event without overwriting another promotion or rollback."""
    return f"promotion_{event_id}"


def _event_payload(receipt: AliasChangeReceipt, status: str) -> dict[str, str | None]:
    """Keep UC tag values under 256 bytes; model, version and event ID are in the tag address."""
    return {
        "k": receipt.kind,
        "p": receipt.prior_version,
        "h": receipt.comparison_sha256,
        "e": receipt.parent_event_id,
        "s": status,
        "o": receipt.previous_champion_version,
    }


def _write_event(client: Any, receipt: AliasChangeReceipt, status: str) -> None:
    """Persist a prepared or committed transition on its destination version."""
    value = json.dumps(_event_payload(receipt, status), sort_keys=True, separators=(",", ":"))
    if len(value.encode("utf-8")) > 256:
        raise ValueError("UC model-version receipt exceeds the 256-byte tag value limit.")
    client.set_model_version_tag(
        receipt.model_name, receipt.new_version, _event_tag(receipt.event_id), value
    )


def _verify_original_receipt(client: Any, receipt: AliasChangeReceipt) -> bool:
    """Refuse forged receipts; return true for a pre-multi-alias promotion."""
    try:
        raw = client.get_model_version(receipt.model_name, receipt.new_version).tags.get(
            _event_tag(receipt.event_id)
        )
    except Exception as exc:  # noqa: BLE001 - registry transport boundary
        raise _translate_error(exc, name=receipt.model_name, version=receipt.new_version) from exc
    expected = _event_payload(receipt, "committed")
    try:
        stored = json.loads(raw) if raw is not None else None
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Promotion receipt is malformed.") from exc
    if stored == expected:
        return False
    legacy = {key: value for key, value in expected.items() if key != "o"}
    if receipt.previous_champion_version is None and stored == legacy:
        return True
    raise ValueError("Promotion receipt is missing or differs from registry state.")


def _commit_change(
    client: Any,
    receipt: AliasChangeReceipt,
    updates: list[tuple[str, str | None, str | None]],
) -> None:
    """Prepare, apply verified alias updates, then persist their common receipt.

    MLflow alias calls are not atomic. Once any call may have changed an alias,
    failures are reported as unknown and require reconciliation before retry.
    """
    try:
        _write_event(client, receipt, "prepared")
    except Exception as exc:  # noqa: BLE001 - registry transport boundary
        raise _translate_error(exc, name=receipt.model_name, version=receipt.new_version) from exc
    changed = False
    for alias, new_version, old_version in updates:
        try:
            if new_version is None:
                client.delete_registered_model_alias(receipt.model_name, alias)
            else:
                client.set_registered_model_alias(receipt.model_name, alias, new_version)
        except Exception as exc:  # noqa: BLE001 - alias write can have an unknown outcome
            try:
                current = _read_optional_alias(client, receipt.model_name, alias)
            except RegistryError:
                current = object()
            if not changed and current == old_version:
                raise _translate_error(
                    exc, name=receipt.model_name, version=receipt.new_version
                ) from exc
            raise AliasOutcomeUnknownError(
                f"Alias outcome unknown; inspect prepared event {receipt.event_id}."
            ) from exc
        changed = True
        try:
            if _read_optional_alias(client, receipt.model_name, alias) != new_version:
                raise AliasOutcomeUnknownError("Alias verification failed.")
        except Exception as exc:  # noqa: BLE001 - a mutation has already occurred
            raise AliasOutcomeUnknownError(
                f"Alias outcome unknown; inspect prepared event {receipt.event_id}."
            ) from exc
    try:
        if receipt.kind == "promotion":
            client.delete_registered_model_tag(receipt.model_name, _CHALLENGER_TAG)
        _write_event(client, receipt, "committed")
        tag = _ACTIVE_TAG if receipt.alias == _ALIAS else _CHALLENGER_TAG
        client.set_registered_model_tag(
            receipt.model_name,
            tag,
            json.dumps({"event_id": receipt.event_id, "version": receipt.new_version}),
        )
        if (
            _active_marker(client, receipt.model_name, receipt.new_version, receipt.alias)
            != receipt.event_id
        ):
            raise AliasOutcomeUnknownError("Active receipt verification failed.")
    except Exception as exc:  # noqa: BLE001 - never imply rollback after a possible alias write
        raise AliasOutcomeUnknownError(
            f"Alias may have changed; inspect event {receipt.event_id} before retry."
        ) from exc


def _validated_report(
    report: ModelComparisonReport,
    heldout: pd.DataFrame | pl.DataFrame,
    *,
    target_column: str,
    expected_champion_version: str,
    max_rows: int,
    max_bytes: int,
    tracking_uri: str | None,
    registry_uri: str | None,
) -> str:
    """Re-evaluate the pinned candidate and return its comparison digest."""
    if not isinstance(report, ModelComparisonReport):
        raise TypeError("report must be a ModelComparisonReport.")
    if (
        type(expected_champion_version) is not str
        or expected_champion_version != report.champion_version
        or not expected_champion_version.isascii()
        or not expected_champion_version.isdigit()
    ):
        raise ValueError("Expected champion version must match the comparison report.")
    if not report.eligible or report.reason != "candidate_improved":
        raise ValueError("Comparison does not approve candidate promotion.")
    candidate = resolve_model(
        report.model_name,
        version=report.candidate_version,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    champion = resolve_model(
        report.model_name,
        version=expected_champion_version,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    fresh = compare_registered_local_models(
        candidate,
        champion,
        heldout,
        target_column=target_column,
        dataset_id=report.dataset_id,
        metric=report.metric,
        min_improvement=report.min_improvement,
        quality_threshold=report.quality_threshold,
        max_rows=max_rows,
        max_bytes=max_bytes,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    if fresh != report or not fresh.eligible:
        raise ValueError("Pinned comparison no longer matches the supplied report.")
    return hashlib.sha256(
        json.dumps(asdict(fresh), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def stage_challenger(
    report: ModelComparisonReport,
    heldout: pd.DataFrame | pl.DataFrame,
    *,
    target_column: str,
    expected_champion_version: str,
    admission: AliasAdmission,
    max_rows: int,
    max_bytes: int,
    expected_challenger_version: str | None = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> AliasChangeReceipt:
    """Assign a validated candidate to challenger without moving champion."""
    _admission(admission, registry_uri)
    digest = _validated_report(
        report,
        heldout,
        target_column=target_column,
        expected_champion_version=expected_champion_version,
        max_rows=max_rows,
        max_bytes=max_bytes,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    client = _make_client(_require_mlflow(), tracking_uri, registry_uri)
    with admission.hold(alias_resource_id(report.model_name)):
        if _read_alias(client, report.model_name) != expected_champion_version:
            raise AliasConflictError("Champion alias differs from expected version.")
        existing = _read_optional_alias(client, report.model_name, _CHALLENGER)
        if existing != expected_challenger_version:
            raise AliasConflictError("Challenger alias differs from expected version.")
        if existing == report.candidate_version:
            raise AliasConflictError("Candidate already holds challenger alias.")
        receipt = AliasChangeReceipt(
            event_id=uuid4().hex,
            kind="challenger",
            model_name=report.model_name,
            alias=_CHALLENGER,
            prior_version=existing,
            new_version=report.candidate_version,
            comparison_sha256=digest,
            parent_event_id=_active_marker(client, report.model_name, expected_champion_version),
        )
        _commit_change(client, receipt, [(_CHALLENGER, report.candidate_version, existing)])
        return receipt


def _verify_staged_challenger(
    client: Any, report: ModelComparisonReport, comparison_sha256: str
) -> str:
    """Require a committed staging event for this exact comparison."""
    if _read_optional_alias(client, report.model_name, _CHALLENGER) != report.candidate_version:
        raise AliasConflictError("Challenger alias differs from candidate version.")
    event_id = _active_marker(client, report.model_name, report.candidate_version, _CHALLENGER)
    if event_id is None:
        raise AliasConflictError("Challenger alias lacks a controlled staging event.")
    try:
        raw = client.get_model_version(report.model_name, report.candidate_version).tags.get(
            _event_tag(event_id)
        )
        event = json.loads(raw) if raw is not None else None
    except Exception as exc:  # noqa: BLE001 - registry transport boundary
        raise AliasConflictError("Challenger staging event cannot be verified.") from exc
    if not isinstance(event, dict) or (
        event.get("k") != "challenger"
        or event.get("s") != "committed"
        or event.get("h") != comparison_sha256
    ):
        raise AliasConflictError("Challenger staging event differs from comparison.")
    return event_id


def promote_candidate(
    report: ModelComparisonReport,
    heldout: pd.DataFrame | pl.DataFrame,
    *,
    target_column: str,
    expected_champion_version: str,
    admission: AliasAdmission,
    max_rows: int,
    max_bytes: int,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> AliasChangeReceipt:
    """Re-evaluate a pinned report and explicitly promote only a better candidate."""
    _admission(admission, registry_uri)
    digest = _validated_report(
        report,
        heldout,
        target_column=target_column,
        expected_champion_version=expected_champion_version,
        max_rows=max_rows,
        max_bytes=max_bytes,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    mlflow = _require_mlflow()
    client = _make_client(mlflow, tracking_uri, registry_uri)
    with admission.hold(alias_resource_id(report.model_name)):
        current = _read_alias(client, report.model_name)
        if current != expected_champion_version:
            raise AliasConflictError("Champion alias differs from expected version.")
        _verify_staged_challenger(client, report, digest)
        prior_event = _active_marker(client, report.model_name, current)
        previous = _read_optional_alias(client, report.model_name, _PREVIOUS)
        receipt = AliasChangeReceipt(
            event_id=uuid4().hex,
            kind="promotion",
            model_name=report.model_name,
            alias=_ALIAS,
            prior_version=current,
            new_version=report.candidate_version,
            comparison_sha256=digest,
            parent_event_id=prior_event,
            previous_champion_version=previous,
        )
        _commit_change(
            client,
            receipt,
            [
                (_PREVIOUS, current, previous),
                (_ALIAS, report.candidate_version, current),
                (_CHALLENGER, None, report.candidate_version),
            ],
        )
        return receipt


def rollback_promotion(
    receipt: AliasChangeReceipt,
    *,
    expected_current_version: str,
    admission: AliasAdmission,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> AliasChangeReceipt:
    """Restore the prior version only while the matching promotion remains active."""
    if not isinstance(receipt, AliasChangeReceipt) or receipt.kind != "promotion":
        raise ValueError("Rollback requires a completed promotion receipt.")
    if receipt.prior_version is None:
        raise ValueError("Promotion receipt must name its prior champion version.")
    if expected_current_version != receipt.new_version:
        raise ValueError("Expected current version must match the promotion receipt.")
    _admission(admission, registry_uri)
    mlflow = _require_mlflow()
    client = _make_client(mlflow, tracking_uri, registry_uri)
    with admission.hold(alias_resource_id(receipt.model_name)):
        current = _read_alias(client, receipt.model_name)
        if current != expected_current_version:
            raise AliasConflictError("Champion alias differs from expected current version.")
        if _active_marker(client, receipt.model_name, current) != receipt.event_id:
            raise AliasConflictError("A newer promotion superseded this rollback receipt.")
        legacy = _verify_original_receipt(client, receipt)
        if _read_optional_alias(client, receipt.model_name, _CHALLENGER) is not None:
            raise AliasConflictError("Clear the newer challenger before rollback.")
        if not legacy and (
            _read_optional_alias(client, receipt.model_name, _PREVIOUS) != receipt.prior_version
        ):
            raise AliasConflictError("Previous champion alias differs from promotion receipt.")
        # The previous version must still exist before changing the alias.
        resolve_model(
            receipt.model_name,
            version=receipt.prior_version,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        reversal = AliasChangeReceipt(
            event_id=uuid4().hex,
            kind="rollback",
            model_name=receipt.model_name,
            alias=_ALIAS,
            prior_version=current,
            new_version=receipt.prior_version,
            comparison_sha256=receipt.comparison_sha256,
            parent_event_id=receipt.event_id,
            previous_champion_version=receipt.previous_champion_version,
        )
        updates: list[tuple[str, str | None, str | None]] = [
            (_ALIAS, receipt.prior_version, current)
        ]
        if not legacy:
            updates.append((_PREVIOUS, receipt.previous_champion_version, receipt.prior_version))
        _commit_change(client, reversal, updates)
        return reversal
