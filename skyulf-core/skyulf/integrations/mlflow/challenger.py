"""Nominate registered contenders independently of promotion eligibility."""

from uuid import uuid4

from .promotion import (
    AliasAdmission,
    AliasChangeReceipt,
    AliasConflictError,
    _active_marker,
    _admission,
    _assert_no_pending,
    _assert_not_rejected,
    _checked_challenger_event,
    _commit_change,
    _read_optional_alias,
    alias_resource_id,
)
from .registry import ResolvedModel, _make_client, _require_mlflow, resolve_model


class ChallengerLifecycle:
    """Track one explicit training orchestrator's registered contender.

    Pass registered as the training service's on_registered callback. Generic
    training remains alias-free unless the caller explicitly opts into this
    externally serialized lifecycle. Evaluation and promotion stay separate.
    """

    def __init__(
        self,
        model_name: str,
        *,
        expected_champion_version: str | None,
        admission: AliasAdmission,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        """Bind the expected champion and one shared alias writer."""
        self.model_name = model_name
        self.expected_champion_version = expected_champion_version
        self.admission = admission
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.candidate: ResolvedModel | None = None

    def registered(self, candidate: ResolvedModel) -> None:
        """Publish a registered contender as pending before its comparison runs."""
        if not isinstance(candidate, ResolvedModel) or candidate.name != self.model_name:
            raise ValueError("Registered candidate must belong to the lifecycle model.")
        _admission(self.admission, self.registry_uri)
        fresh = resolve_model(
            candidate.name,
            version=candidate.version,
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri,
        )
        if fresh.digest != candidate.digest or not candidate.digest:
            raise ValueError("Candidate artifact digest changed before nomination.")
        client = _make_client(_require_mlflow(), self.tracking_uri, self.registry_uri)
        with self.admission.hold(alias_resource_id(candidate.name)):
            _assert_not_rejected(client, candidate.name, candidate.version)
            champion = _read_optional_alias(client, candidate.name, "champion")
            if champion != self.expected_champion_version:
                raise AliasConflictError("Champion changed before challenger nomination.")
            if candidate.version == champion:
                raise AliasConflictError("Champion cannot also be its own challenger.")
            existing = _read_optional_alias(client, candidate.name, "challenger")
            if existing is not None:
                _checked_challenger_event(client, candidate.name, existing)
                if int(existing) > int(candidate.version):
                    raise AliasConflictError("A newer challenger already exists.")
                if existing == candidate.version:
                    self.candidate = candidate
                    return
            receipt = AliasChangeReceipt(
                event_id=uuid4().hex,
                kind="nomination",
                model_name=candidate.name,
                alias="challenger",
                prior_version=existing,
                new_version=candidate.version,
                comparison_sha256=candidate.digest,
                parent_event_id=(
                    _active_marker(client, candidate.name, champion) if champion else None
                ),
            )
            _commit_change(
                client,
                receipt,
                [("challenger", candidate.version, existing)],
                version_tags={
                    "validation_status": "pending",
                    "validation_reason": "Waiting for evaluation",
                    "promotion_status": "not_promoted",
                },
            )
            self.candidate = candidate

    def failed(self) -> None:
        """Retain the contender and record failure without copying raw exception text."""
        candidate = self.candidate
        if candidate is None:
            return
        client = _make_client(_require_mlflow(), self.tracking_uri, self.registry_uri)
        with self.admission.hold(alias_resource_id(candidate.name)):
            version = client.get_model_version(candidate.name, candidate.version)
            if (version.tags or {}).get("validation_status") != "pending":
                return
            _assert_no_pending(client, candidate.name)
            if _read_optional_alias(client, candidate.name, "challenger") != candidate.version:
                raise AliasConflictError("Challenger changed before failure recording.")
            _checked_challenger_event(client, candidate.name, candidate.version)
            receipt = AliasChangeReceipt(
                event_id=uuid4().hex,
                kind="evaluation_error",
                model_name=candidate.name,
                alias="challenger",
                prior_version=candidate.version,
                new_version=candidate.version,
                comparison_sha256=None,
                parent_event_id=_active_marker(
                    client, candidate.name, candidate.version, "challenger"
                ),
            )
            _commit_change(
                client,
                receipt,
                [],
                version_tags={
                    "validation_status": "error",
                    "validation_reason": "Evaluation could not be completed",
                    "promotion_status": "not_promoted",
                },
            )
