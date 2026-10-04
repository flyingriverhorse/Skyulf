"""Functional validation and explicitly approved coherent model-set releases.

Representative predictions certify executability and exercised outputs. Sets
with saved quality pins additionally require a policy-aware validator under the
same admission. Legacy packages retain their explicit functional-only API.
"""

import hashlib
import json
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from skyulf.integrations.mlflow._client import make_registry_client, require_mlflow

from ...inference.model_set import ModelSetArtifact
from ...inference.model_set_scoring import model_set_output_schema, predict_model_set
from .model_set import load_registered_model_set
from .model_set_challenger import verify_model_set_challenger
from .promotion import (
    AliasAdmission,
    AliasChangeReceipt,
    AliasConflictError,
    active_marker,
    alias_resource_id,
    assert_not_rejected,
    commit_change,
    controlled_champion_version,
    event_tag,
    read_event,
    read_optional_alias,
    rollback_promotion,
    validate_admission,
)
from .registry import ResolvedModel, resolve_model

_PROOF_TAG = "model_set_validation_sha256"
_PATH_TAG = "model_set_validation_artifact"


def approve_model_set(
    resolved: ResolvedModel,
    validation_frame: pd.DataFrame,
    *,
    expected_champion_version: str | None,
    admission: AliasAdmission,
    max_rows: int,
    max_bytes: int,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    quality_validator: Callable[[ModelSetArtifact], dict[str, Any]] | None = None,
) -> AliasChangeReceipt:
    """Validate a bounded representative frame and explicitly activate one complete set.

    At least one prediction from every component and every composition rule is
    required. Every writer must share the same non-expiring alias admission.
    Packages with saved quality pins also require a passing quality validator;
    representative predictions alone cannot approve those packages.
    """
    validate_admission(admission, registry_uri)
    _bounded_frame(validation_frame, max_rows, max_bytes)
    artifact = load_registered_model_set(
        resolved, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    evidence = _validation_evidence(resolved, artifact, validation_frame, max_bytes)
    client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
    with admission.hold(alias_resource_id(resolved.name)):
        assert_not_rejected(client, resolved.name, resolved.version)
        current = controlled_champion_version(
            resolved.name, tracking_uri=tracking_uri, registry_uri=registry_uri
        )
        if current != expected_champion_version:
            raise AliasConflictError("Champion alias differs from expected version.")
        verify_model_set_challenger(
            client, resolved, required=artifact.manifest.quality_evidence is not None
        )
        quality = _quality_validation(artifact, expected_champion_version, quality_validator)
        if quality is not None:
            evidence["quality"] = quality
        if current == resolved.version:
            raise AliasConflictError("Model set is already the current champion.")
        _validate_set_roles(client, resolved.name, current)
        if current is not None:
            _validated_version(client, resolved.name, current, tracking_uri, registry_uri)
        digest = _persist_evidence(client, resolved, evidence, tracking_uri)
        receipt, updates = _release_change(client, resolved, current, digest)
        commit_change(
            client,
            receipt,
            updates,
            version_tags={
                "validation_status": "passed",
                "validation_reason": "Set validation passed",
            },
        )
        return receipt


def _quality_validation(
    artifact: ModelSetArtifact, expected: str | None, validator: Callable | None
) -> dict | None:
    """Require policy-aware validation for quality-bound sets before any alias intent."""
    saved = artifact.manifest.quality_evidence
    if saved is not None and validator is None:
        raise ValueError("Model set requires a quality validator for saved component policies.")
    if validator is None:
        return None
    result = validator(artifact)
    _validate_quality_result(result, artifact, expected)
    return result


def _validate_quality_result(result: Any, artifact: ModelSetArtifact, expected: str | None) -> None:
    """Reject incomplete or contradictory aggregate decisions."""
    branches = {c.branch for c in artifact.manifest.components}
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise ValueError("Model set quality validation failed.")
    if (
        result.get("expected_champion_version") != expected
        or set(result.get("components", {})) != branches
    ):
        raise ValueError("Model set quality validation differs from baseline or components.")
    if any(value.get("passed") is not True for value in result["components"].values()):
        raise ValueError("Model set component quality validation failed.")


def _validate_set_roles(client: Any, name: str, current: str | None) -> None:
    """Require champion history to match the previous committed set release."""
    previous = read_optional_alias(client, name, "previous_champion")
    if current is None and previous is not None:
        raise AliasConflictError("Model set has a previous champion without a current champion.")
    if current is not None:
        event_id = active_marker(client, name, current)
        tags = client.get_model_version(name, current).tags or {}
        event = read_event(tags.get(event_tag(str(event_id))))
        expected = event.get("p") if event["k"] == "promotion" else event.get("o")
        if previous != expected:
            raise AliasConflictError(
                "Model set previous champion differs from its committed receipt."
            )


def _release_change(
    client: Any, resolved: ResolvedModel, current: str | None, digest: str
) -> tuple[AliasChangeReceipt, list[tuple[str, str | None, str | None]]]:
    """Plan one complete-set activation using the shared durable receipt format."""
    previous = read_optional_alias(client, resolved.name, "previous_champion")
    receipt = AliasChangeReceipt(
        event_id=uuid4().hex,
        kind="initial" if current is None else "promotion",
        model_name=resolved.name,
        alias="champion",
        prior_version=current,
        new_version=resolved.version,
        comparison_sha256=digest,
        parent_event_id=active_marker(client, resolved.name, current) if current else None,
        previous_champion_version=previous,
    )
    updates: list[tuple[str, str | None, str | None]] = []
    if current is not None:
        updates.append(("previous_champion", current, previous))
    updates.append(("champion", resolved.version, current))
    if read_optional_alias(client, resolved.name, "challenger") is not None:
        updates.append(("challenger", None, resolved.version))
    return receipt, updates


def _bounded_frame(frame: pd.DataFrame, max_rows: int, max_bytes: int) -> None:
    """Reject empty or oversized validation data before any model is loaded."""
    if type(max_rows) is not int or type(max_bytes) is not int or min(max_rows, max_bytes) <= 0:
        raise ValueError("Validation max_rows and max_bytes must be positive integers.")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Model set validation requires a nonempty pandas DataFrame.")
    if len(frame) > max_rows or int(frame.memory_usage(index=True, deep=True).sum()) > max_bytes:
        raise ValueError("Model set validation frame exceeds its explicit bounds.")


def _artifact_identity(resolved: ResolvedModel, artifact: ModelSetArtifact) -> dict[str, Any]:
    """Describe the exact executable set without storing producer paths or source data."""
    source = artifact.directory / "composition.py"
    identity = {
        "kind": "model_set_functional_validation_v1",
        "model_name": resolved.name,
        "model_version": resolved.version,
        "set_sha256": artifact.manifest.set_sha256,
        "components": [
            {"branch": c.branch, "reference": c.reference.model_dump()}
            for c in artifact.manifest.components
        ],
        "input_schema": [c.model_dump() for c in artifact.manifest.input_schema],
        "output_schema": [c.model_dump() for c in model_set_output_schema(artifact)],
        "composition_config": artifact.manifest.composition_config,
        "composition_source_sha256": hashlib.sha256(
            source.read_bytes() if source.is_file() else b""
        ).hexdigest(),
    }
    if artifact.manifest.quality_evidence is not None:
        identity["quality_evidence"] = artifact.manifest.quality_evidence
    return identity


def _validation_evidence(
    resolved: ResolvedModel, artifact: ModelSetArtifact, frame: pd.DataFrame, max_bytes: int
) -> dict[str, Any]:
    """Exercise every output path and bind success to frame and executable identities."""
    result = predict_model_set(frame, artifact, max_rows=len(frame), max_bytes=max_bytes)
    if int(result.memory_usage(index=True, deep=True).sum()) > max_bytes:
        raise ValueError("Model set validation output exceeds its explicit byte bound.")
    branches = [component.branch for component in artifact.manifest.components]
    rules = [rule["name"] for rule in artifact.manifest.composition_config["outputs"]]
    exercised = {}
    for name in [*branches, *rules]:
        count = int((result[f"{name}__scoring_status"] == "predicted").sum())
        if count == 0:
            raise ValueError(f"Validation data did not exercise model set output {name}.")
        exercised[name] = count
    return {
        **_artifact_identity(resolved, artifact),
        "row_count": len(frame),
        "dataset_sha256": _frame_digest(frame),
        "output_sha256": _frame_digest(result),
        "exercised_predictions": exercised,
    }


def _frame_digest(frame: pd.DataFrame) -> str:
    """Hash ordered values, columns and dtypes without persisting representative rows."""
    schema = json.dumps(list(zip(frame.columns, map(str, frame.dtypes), strict=True))).encode()
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    return hashlib.sha256(schema + values).hexdigest()


def _evidence_bytes(evidence: dict[str, Any]) -> bytes:
    """Serialize deterministic JSON for artifact integrity checks."""
    return json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _persist_evidence(
    client: Any, resolved: ResolvedModel, evidence: dict[str, Any], tracking_uri: str | None
) -> str:
    """Upload and read back durable validation before writing any alias intent."""
    version = client.get_model_version(resolved.name, resolved.version)
    if not version.run_id:
        raise ValueError("Model set validation requires a producing run ID.")
    content = _evidence_bytes(evidence)
    digest = hashlib.sha256(content).hexdigest()
    folder = f"model_set_validation/{resolved.version}/{digest}"
    with tempfile.TemporaryDirectory(prefix="skyulf-set-validation-") as directory:
        path = Path(directory) / "validation.json"
        path.write_bytes(content)
        client.log_artifact(version.run_id, str(path), artifact_path=folder)
    artifact_uri = f"runs:/{version.run_id}/{folder}/validation.json"
    _download_evidence(artifact_uri, digest, tracking_uri)
    client.set_model_version_tag(resolved.name, resolved.version, _PROOF_TAG, digest)
    client.set_model_version_tag(resolved.name, resolved.version, _PATH_TAG, artifact_uri)
    tags = client.get_model_version(resolved.name, resolved.version).tags or {}
    if tags.get(_PROOF_TAG) != digest or tags.get(_PATH_TAG) != artifact_uri:
        raise ValueError("Model set validation evidence tags could not be verified.")
    return digest


def _download_evidence(uri: str, digest: str, tracking_uri: str | None) -> dict[str, Any]:
    """Read evidence only when its complete stored bytes match the recorded digest."""
    path = require_mlflow().artifacts.download_artifacts(
        artifact_uri=uri, tracking_uri=tracking_uri
    )
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Model set validation evidence digest differs from recorded proof.")
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Model set validation evidence must be an object.")
    return result


def _verify_set_evidence(
    client: Any, resolved: ResolvedModel, artifact: ModelSetArtifact, tracking_uri: str | None
) -> None:
    """Require durable successful validation bound to this complete package identity."""
    tags = client.get_model_version(resolved.name, resolved.version).tags or {}
    if not tags.get(_PROOF_TAG) or not tags.get(_PATH_TAG):
        raise ValueError("Model set version has no durable validation evidence.")
    evidence = _download_evidence(tags[_PATH_TAG], tags[_PROOF_TAG], tracking_uri)
    expected = _artifact_identity(resolved, artifact)
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError("Model set validation evidence differs from current package identity.")
    if artifact.manifest.quality_evidence is not None:
        _validate_quality_result(
            evidence.get("quality"),
            artifact,
            artifact.manifest.quality_evidence["expected_champion_version"],
        )
    _verify_exercised_predictions(evidence)


def _verify_exercised_predictions(evidence: dict[str, Any]) -> None:
    """Require recorded representative rows and successful output execution counts."""
    counts = evidence.get("exercised_predictions", {})
    if not evidence.get("row_count") or not counts or any(value <= 0 for value in counts.values()):
        raise ValueError("Model set validation evidence has no exercised predictions.")


def _validated_version(
    client: Any, name: str, version: str, tracking_uri: str | None, registry_uri: str | None
) -> None:
    """Verify the set kind, complete bytes and functional evidence for a pinned version."""
    resolved = resolve_model(
        name, version=version, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    artifact = load_registered_model_set(
        resolved, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    _verify_set_evidence(client, resolved, artifact, tracking_uri)


class _ValidationAdmission:
    """Run set validation inside the same shared admission used by rollback."""

    def __init__(self, admission: AliasAdmission, validate: Callable[[], None]) -> None:
        """Wrap an existing admission without nesting or weakening its lock."""
        self._admission = admission
        self.local_only = admission.local_only
        self._validate = validate

    @contextmanager
    def hold(self, resource_id: str) -> Iterator[None]:
        """Keep the shared lock through validation and the complete alias mutation."""
        with self._admission.hold(resource_id):
            self._validate()
            yield


def rollback_model_set(
    receipt: AliasChangeReceipt,
    *,
    expected_current_version: str,
    admission: AliasAdmission,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> AliasChangeReceipt:
    """Restore a complete prior set only with valid artifacts and persisted validation."""
    if not isinstance(receipt, AliasChangeReceipt) or receipt.kind != "promotion":
        raise ValueError("Model set rollback requires a completed promotion receipt.")
    if receipt.prior_version is None:
        raise ValueError("Model set rollback requires a prior version.")
    client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
    validate_admission(admission, registry_uri)

    def validate() -> None:
        """Recheck both executable sets while rollback holds its shared admission."""
        for version in (receipt.new_version, receipt.prior_version):
            _validated_version(client, receipt.model_name, str(version), tracking_uri, registry_uri)

    return rollback_promotion(
        receipt,
        expected_current_version=expected_current_version,
        admission=_ValidationAdmission(admission, validate),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
