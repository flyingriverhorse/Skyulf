"""Explicit alias promotion must preserve pinned comparisons and rollback receipts."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import save_local_pipeline
from skyulf.integrations.mlflow.local_model import log_local_model
from skyulf.integrations.mlflow.promotion import (
    AliasChangeReceipt,
    AliasConflictError,
    AliasOutcomeUnknownError,
    ExclusiveAliasWriterAdmission,
    LocalAliasAdmission,
    _admission,
    alias_resource_id,
    promote_candidate,
    rollback_promotion,
    stage_challenger,
)
from skyulf.integrations.mlflow.registry import (
    RegistryAccessError,
    RegistryModelNotFoundError,
    register_model,
    resolve_model,
)
from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run
from skyulf.integrations.mlflow.validation import (
    ModelComparisonReport,
    compare_registered_local_models,
)
from skyulf.pipeline import SkyulfPipeline

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def case(tmp_path: Path):
    """Use real local MLflow versions so alias and receipt behavior cannot be mocked away."""
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    name = "sm22-promotion"
    tracking = TrackingConfig(enabled=True, tracking_uri=uri, experiment_name=name)
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment(name, artifact_location=(tmp_path / "mlruns").as_uri())
    x = np.arange(12, dtype="float64")
    holdout_x = np.arange(20, 25, dtype="float64")
    heldout = pd.DataFrame({"x": holdout_x, "target": 2.0 * holdout_x})
    for offset in (10.0, 0.0):
        train = pd.DataFrame({"x": x, "target": 2.0 * x + offset})
        pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
        pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="target")
        artifact_path = tmp_path / f"model-{int(offset)}"
        save_local_pipeline(pipeline, artifact_path)
        with track_run(tracking, run_name=f"fit-{offset}") as run:
            assert run.run_id is not None
            model_uri = log_local_model(
                artifact_path, run_id=run.run_id, artifact_path="model", tracking_uri=uri
            )
        register_model(model_uri, name, tracking_uri=uri, registry_uri=uri)
    client.set_registered_model_alias(name, "champion", "1")
    candidate = resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri)
    champion = resolve_model(name, version="1", tracking_uri=uri, registry_uri=uri)
    report = compare_registered_local_models(
        candidate,
        champion,
        heldout,
        target_column="target",
        dataset_id="labels@5/heldout",
        metric="heldout_mse",
        min_improvement=1.0,
        quality_threshold=1.0,
        max_rows=10,
        max_bytes=10_000,
        tracking_uri=uri,
        registry_uri=uri,
    )
    assert report.eligible
    return client, uri, name, heldout, report, LocalAliasAdmission(tmp_path / "locks")


def _promote(case, **changes):
    """Call the real promotion path with one validated local comparison."""
    _, uri, _, heldout, report, admission = case
    options: dict[str, Any] = {
        "target_column": "target",
        "expected_champion_version": "1",
        "admission": admission,
        "max_rows": 10,
        "max_bytes": 10_000,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    options.update(changes)
    selected = cast(ModelComparisonReport, options.pop("report", report))
    return promote_candidate(selected, heldout, **options)


def _stage(case, **changes):
    """Publish a verified challenger before an explicit champion promotion."""
    _, uri, _, heldout, report, admission = case
    options: dict[str, Any] = {
        "target_column": "target",
        "expected_champion_version": "1",
        "admission": admission,
        "max_rows": 10,
        "max_bytes": 10_000,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    options.update(changes)
    selected = cast(ModelComparisonReport, options.pop("report", report))
    return stage_challenger(selected, heldout, **options)


def _reject(case, reason="Business review declined deployment", **changes):
    """Reject the staged comparison through the same explicit alias admission."""
    from skyulf.integrations.mlflow.rejection import reject_candidate

    _, uri, _, _, report, admission = case
    options = {
        "expected_champion_version": report.champion_version,
        "admission": admission,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    options.update(changes)
    return reject_candidate(report, reason=reason, **options)


def test_manual_rejection_retains_aliases_and_cannot_be_erased_by_recheck(case):
    """An operator rejection must survive quality rechecks without changing objective metrics."""
    client, _, name, _, _, _ = case
    _stage(case)
    before = client.get_registered_model(name).aliases
    receipt = _reject(case)
    assert receipt.kind == "rejection"
    assert client.get_registered_model(name).aliases == before
    tags = client.get_model_version(name, "2").tags
    assert tags["validation_status"] == "passed"
    assert tags["approval_status"] == "rejected"
    assert tags["approval_reason"] == "Business review declined deployment"
    assert _reject(case) == receipt
    with pytest.raises(AliasConflictError, match="reason|decision"):
        _reject(case, reason="Different decision")
    with pytest.raises(AliasConflictError, match="rejected"):
        _promote(case)
    with pytest.raises(AliasConflictError, match="rejected"):
        _stage(case, expected_challenger_version="2")
    assert client.get_registered_model(name).aliases == before


def test_rejected_first_candidate_cannot_initialize_or_renominate(case):
    """Bootstrap and nomination must not bypass a recorded operator rejection."""
    from skyulf.integrations.mlflow.challenger import ChallengerLifecycle
    from skyulf.integrations.mlflow.promotion import initialize_champion
    from skyulf.integrations.mlflow.rejection import reject_candidate

    client, uri, name, heldout, _, admission = case
    client.delete_registered_model_alias(name, "champion")
    candidate = resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri)
    evaluation: dict[str, Any] = {
        "target_column": "target",
        "max_rows": 10,
        "max_bytes": 10_000,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    report = compare_registered_local_models(
        candidate,
        None,
        heldout,
        dataset_id="labels@5/heldout",
        metric="heldout_mse",
        min_improvement=1.0,
        quality_threshold=1.0,
        **evaluation,
    )
    _stage(case, report=report, expected_champion_version=None)
    reject_candidate(
        report,
        reason="Review declined first deployment",
        expected_champion_version=None,
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    with pytest.raises(AliasConflictError, match="rejected"):
        initialize_champion(report, heldout, admission=admission, **evaluation)
    lifecycle = ChallengerLifecycle(
        name,
        expected_champion_version=None,
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    with pytest.raises(AliasConflictError, match="rejected"):
        lifecycle.registered(candidate)
    aliases = client.get_registered_model(name).aliases
    assert set(aliases) == {"challenger"}
    assert str(aliases["challenger"]) == "2"


@pytest.mark.parametrize("reason", ["", "   ", "a" * 257, "\u00fc" * 129])
def test_invalid_rejection_reason_cannot_write_a_decision(case, reason):
    """Empty and oversized UTF-8 reasons must fail before changing model metadata."""
    client, _, name, _, _, _ = case
    _stage(case)
    tags = client.get_model_version(name, "2").tags
    with pytest.raises(ValueError, match="reason"):
        _reject(case, reason=reason)
    assert client.get_model_version(name, "2").tags == tags


def test_partial_rejection_retains_pending_event_and_blocks_retry(case, monkeypatch):
    """A lost decision-tag write cannot be treated as a completed or safe-to-repeat rejection."""
    client, _, name, _, _, _ = case
    _stage(case)
    original = mlflow.MlflowClient.set_model_version_tag

    def fail_reason(self, model_name, version, key, value):
        """Fail after status was written but before its human-readable reason."""
        if key == "approval_reason":
            raise RuntimeError("lost decision response")
        return original(self, model_name, version, key, value)

    monkeypatch.setattr(mlflow.MlflowClient, "set_model_version_tag", fail_reason)
    with pytest.raises(AliasOutcomeUnknownError):
        _reject(case)
    with pytest.raises(AliasConflictError, match="pending"):
        _reject(case)
    assert client.get_registered_model(name).tags["pending_alias_event"]


def test_rollback_retry_returns_original_receipt_without_alias_writes(case, monkeypatch):
    """A repeated logical rollback must be idempotent even though its expected source moved."""
    client, uri, name, _, _, admission = case
    _stage(case)
    receipt = _promote(case)
    options: dict[str, Any] = {
        "expected_current_version": "2",
        "admission": admission,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    reversal = rollback_promotion(receipt, **options)
    before = client.get_registered_model(name).tags

    def unexpected_write(*args, **kwargs):
        """Replaying a committed rollback may read state but must not mutate it."""
        raise AssertionError("rollback replay attempted a write")

    monkeypatch.setattr(mlflow.MlflowClient, "set_registered_model_alias", unexpected_write)
    assert rollback_promotion(receipt, **options) == reversal
    assert client.get_registered_model(name).tags == before


@pytest.mark.parametrize("changed", ["pending", "previous", "receipt"])
def test_rollback_retry_refuses_changed_control_state(case, changed):
    """A completed rollback is not permission to ignore later corruption or pending writes."""
    client, uri, name, _, _, admission = case
    _stage(case)
    receipt = _promote(case)
    options: dict[str, Any] = {
        "expected_current_version": "2",
        "admission": admission,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    reversal = rollback_promotion(receipt, **options)
    if changed == "pending":
        client.set_registered_model_tag(name, "pending_alias_event", "uncertain")
    elif changed == "previous":
        client.set_registered_model_alias(name, "previous_champion", "2")
    else:
        client.set_model_version_tag(name, "1", f"promotion_{reversal.event_id}", "{}")
    with pytest.raises((AliasConflictError, ValueError)):
        rollback_promotion(receipt, **options)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


def test_stage_challenger_requires_validation_and_preserves_champion(case) -> None:
    """Staging verifies evidence while leaving the production version unchanged."""
    client, _, name, _, report, _ = case
    staged = _stage(case)
    assert staged.kind == "challenger"
    assert staged.new_version == "2"
    assert str(client.get_model_version_by_alias(name, "challenger").version) == "2"
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"
    assert json.loads(client.get_registered_model(name).tags["challenger_current_event"]) == {
        "event_id": staged.event_id,
        "version": "2",
    }

    with pytest.raises(AliasConflictError, match="[Cc]hallenger"):
        _stage(case)
    with pytest.raises(ValueError, match="comparison"):
        _stage(case, report=replace(report, eligible=True, candidate_metrics={"heldout_mse": -1.0}))


def test_rejected_challenger_is_visible_but_cannot_promote(case) -> None:
    """A contender's identity must not imply approval to replace champion."""
    client, uri, name, heldout, _, _ = case
    report = compare_registered_local_models(
        resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri),
        resolve_model(name, version="1", tracking_uri=uri, registry_uri=uri),
        heldout,
        target_column="target",
        dataset_id="labels@5/heldout",
        metric="heldout_mse",
        min_improvement=1_000_000,
        max_rows=10,
        max_bytes=10_000,
        tracking_uri=uri,
        registry_uri=uri,
    )
    _stage(case, report=report)
    assert str(client.get_model_version_by_alias(name, "challenger").version) == "2"
    assert client.get_model_version(name, "2").tags["validation_status"] == "rejected"
    assert (
        client.get_model_version(name, "2").tags["validation_reason"]
        == "Improvement below required minimum"
    )
    with pytest.raises(ValueError, match="approve"):
        _promote(case, report=report)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


def test_nomination_precedes_evaluation_and_failure_keeps_challenger(case) -> None:
    """An evaluation failure remains inspectable without moving production aliases."""
    from skyulf.integrations.mlflow.challenger import ChallengerLifecycle

    client, uri, name, _, _, admission = case
    lifecycle = ChallengerLifecycle(
        name,
        expected_champion_version="1",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    candidate = resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri)
    lifecycle.registered(candidate)
    first = client.get_registered_model(name).tags["challenger_current_event"]
    lifecycle.registered(candidate)
    assert client.get_registered_model(name).tags["challenger_current_event"] == first
    assert client.get_model_version(name, "2").tags["validation_status"] == "pending"
    lifecycle.failed()
    assert client.get_model_version(name, "2").tags["validation_status"] == "error"
    assert str(client.get_model_version_by_alias(name, "challenger").version) == "2"
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


def test_tied_v3_remains_challenger_through_rollback_and_v4_replaces_it(case) -> None:
    """Retaining an unsuccessful contender must not block safe rollback or later training."""
    from skyulf.integrations.mlflow.challenger import ChallengerLifecycle

    client, uri, name, heldout, _, admission = case
    _stage(case)
    promoted = _promote(case)
    source = client.get_model_version(name, "2").source
    register_model(source, name, tracking_uri=uri, registry_uri=uri)
    candidate = resolve_model(name, version="3", tracking_uri=uri, registry_uri=uri)
    lifecycle = ChallengerLifecycle(
        name,
        expected_champion_version="2",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    lifecycle.registered(candidate)
    report = compare_registered_local_models(
        candidate,
        resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri),
        heldout,
        target_column="target",
        dataset_id="labels@5/heldout",
        metric="heldout_mse",
        min_improvement=0,
        max_rows=10,
        max_bytes=10_000,
        tracking_uri=uri,
        registry_uri=uri,
    )
    _stage(case, report=report, expected_champion_version="2", expected_challenger_version="3")
    assert not report.eligible
    rollback_promotion(
        promoted,
        expected_current_version="2",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"
    assert str(client.get_model_version_by_alias(name, "challenger").version) == "3"
    register_model(source, name, tracking_uri=uri, registry_uri=uri)
    next_lifecycle = ChallengerLifecycle(
        name,
        expected_champion_version="1",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    next_lifecycle.registered(resolve_model(name, version="4", tracking_uri=uri, registry_uri=uri))
    with pytest.raises(AliasConflictError, match="newer"):
        next_lifecycle.registered(candidate)
    assert str(client.get_model_version_by_alias(name, "challenger").version) == "4"
    assert client.get_model_version(name, "3").tags["validation_status"] == "rejected"


def test_first_nominee_is_removed_from_challenger_when_initialized(case) -> None:
    """The first champion must still pass quality and stop being its own challenger."""
    from skyulf.integrations.mlflow.challenger import ChallengerLifecycle
    from skyulf.integrations.mlflow.promotion import initialize_champion

    client, uri, name, heldout, _, admission = case
    client.delete_registered_model_alias(name, "champion")
    candidate = resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri)
    lifecycle = ChallengerLifecycle(
        name,
        expected_champion_version=None,
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    lifecycle.registered(candidate)
    report = compare_registered_local_models(
        candidate,
        None,
        heldout,
        target_column="target",
        dataset_id="labels@5/heldout",
        metric="heldout_mse",
        min_improvement=0,
        quality_threshold=1.0,
        max_rows=10,
        max_bytes=10_000,
        tracking_uri=uri,
        registry_uri=uri,
    )
    _stage(case, report=report, expected_champion_version=None, expected_challenger_version="2")
    assert client.get_model_version(name, "2").tags["validation_status"] == "passed"
    initialize_champion(
        report,
        heldout,
        target_column="target",
        admission=admission,
        max_rows=10,
        max_bytes=10_000,
        tracking_uri=uri,
        registry_uri=uri,
    )
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias(name, "challenger")
    assert client.get_model_version(name, "2").tags["promotion_status"] == "promoted"


def test_partial_challenger_status_write_keeps_pending_receipt(case, monkeypatch) -> None:
    """Partial UI status must never be reported as a completed registry transition."""
    from skyulf.integrations.mlflow.promotion import controlled_champion_version

    client, uri, name, _, _, _ = case
    original = client.set_model_version_tag

    def fail_reason(model_name, version, key, value):
        """Simulate loss of the status write after the alias and first tag changed."""
        if key == "validation_reason":
            raise RuntimeError("tag transport failed")
        return original(model_name, version, key, value)

    monkeypatch.setattr(client, "set_model_version_tag", fail_reason)
    monkeypatch.setattr("skyulf.integrations.mlflow.promotion._make_client", lambda *args: client)
    with pytest.raises(AliasOutcomeUnknownError, match="Alias may have changed"):
        _stage(case)
    with pytest.raises(AliasConflictError, match="pending"):
        controlled_champion_version(name, tracking_uri=uri, registry_uri=uri)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


def test_promotion_requires_staged_challenger(case) -> None:
    """A direct or tampered challenger alias must not bypass the validated staging step."""
    client, _, name, _, _, _ = case
    with pytest.raises(AliasConflictError, match="[Cc]hallenger"):
        _promote(case)
    _stage(case)
    client.set_registered_model_alias(name, "challenger", "1")
    with pytest.raises(AliasConflictError, match="[Cc]hallenger"):
        _promote(case)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


@pytest.mark.parametrize("receipt_format", ["current", "readable_v1"])
def test_promote_and_rollback_persist_receipts(case, receipt_format) -> None:
    """An accepted candidate moves the alias and leaves auditable forward and reverse receipts."""
    client, uri, name, _, _, admission = case

    _stage(case)
    receipt = _promote(case)
    assert receipt.prior_version == "1"
    assert receipt.new_version == "2"
    assert str(client.get_model_version_by_alias(name, "champion").version) == "2"
    assert str(client.get_model_version_by_alias(name, "previous_champion").version) == "1"
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias(name, "challenger")
    assert "challenger_current_event" not in client.get_registered_model(name).tags
    tag = client.get_model_version(name, "2").tags[f"promotion_{receipt.event_id}"]
    assert len(tag.encode("utf-8")) <= 256
    assert json.loads(tag)["state"] == "committed"
    assert json.loads(tag)["action"] == "promotion"
    assert json.loads(tag)["from_version"] == "1"
    assert "from" not in json.loads(tag)
    if receipt_format == "readable_v1":
        old_payload = json.loads(tag)
        old_payload["from"] = old_payload.pop("from_version")
        client.set_model_version_tag(
            name, "2", f"promotion_{receipt.event_id}", json.dumps(old_payload)
        )
    assert json.loads(client.get_registered_model(name).tags["champion_current_event"]) == {
        "event_id": receipt.event_id,
        "version": "2",
    }

    reversal = rollback_promotion(
        receipt,
        expected_current_version="2",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    assert reversal.prior_version == "2"
    assert reversal.new_version == "1"
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias(name, "previous_champion")
    assert json.loads(client.get_registered_model(name).tags["champion_current_event"]) == {
        "event_id": reversal.event_id,
        "version": "1",
    }


def test_rollback_restores_prior_previous_champion_pointer(case) -> None:
    """Rollback restores the preceding rollback pointer rather than leaving a stale one."""
    client, uri, name, _, _, admission = case
    client.set_registered_model_alias(name, "previous_champion", "2")
    _stage(case)
    receipt = _promote(case)
    assert receipt.previous_champion_version == "2"
    assert str(client.get_model_version_by_alias(name, "previous_champion").version) == "1"

    rollback_promotion(
        receipt,
        expected_current_version="2",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    assert str(client.get_model_version_by_alias(name, "previous_champion").version) == "2"


def test_rollback_rejects_changed_previous_pointer(case) -> None:
    """A bypass edit to the rollback target cannot silently be accepted."""
    client, uri, name, _, _, admission = case
    _stage(case)
    receipt = _promote(case)
    client.set_registered_model_alias(name, "previous_champion", "2")
    with pytest.raises(AliasConflictError, match="Previous champion"):
        rollback_promotion(
            receipt,
            expected_current_version="2",
            admission=admission,
            tracking_uri=uri,
            registry_uri=uri,
        )
    assert str(client.get_model_version_by_alias(name, "champion").version) == "2"


def test_legacy_champion_receipt_can_still_roll_back(case) -> None:
    """A receipt written before the challenger aliases existed stays reversible."""
    client, uri, name, _, _, admission = case
    legacy = AliasChangeReceipt(
        event_id="legacy",
        kind="promotion",
        model_name=name,
        alias="champion",
        prior_version="1",
        new_version="2",
        comparison_sha256="old-comparison",
        parent_event_id=None,
    )
    client.set_model_version_tag(
        name,
        "2",
        "promotion_legacy",
        json.dumps(
            {"k": "promotion", "p": "1", "h": "old-comparison", "e": None, "s": "committed"},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    client.set_registered_model_alias(name, "champion", "2")
    client.set_registered_model_tag(
        name, "champion_current_event", json.dumps({"event_id": "legacy", "version": "2"})
    )

    reversal = rollback_promotion(
        legacy,
        expected_current_version="2",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    assert reversal.new_version == "1"
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


def test_partial_promotion_reports_unknown_outcome(case, monkeypatch) -> None:
    """A failure after changing the previous pointer requires reconciliation."""
    client, _, name, _, _, _ = case
    _stage(case)
    original = client.set_registered_model_alias
    error = mlflow.exceptions.MlflowException("denied")
    error.error_code = "PERMISSION_DENIED"

    def deny_champion(model_name, alias, version):
        """Allow the first alias update, then deny the champion move."""
        if alias == "champion":
            raise error
        return original(model_name, alias, version)

    monkeypatch.setattr(client, "set_registered_model_alias", deny_champion)
    monkeypatch.setattr("skyulf.integrations.mlflow.promotion._make_client", lambda *args: client)
    with pytest.raises(AliasOutcomeUnknownError, match="inspect prepared event"):
        _promote(case)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"
    assert str(client.get_model_version_by_alias(name, "previous_champion").version) == "1"
    assert str(client.get_model_version_by_alias(name, "challenger").version) == "2"


def test_stale_alias_and_modified_report_cannot_promote(case) -> None:
    """The lock path must reject stale alias state and forged comparison fields."""
    client, _, name, _, report, _ = case
    client.set_registered_model_alias(name, "champion", "2")
    with pytest.raises(AliasConflictError, match="expected"):
        _promote(case)
    client.set_registered_model_alias(name, "champion", "1")

    tampered = replace(report, eligible=True, candidate_metrics={"heldout_mse": -1.0})
    with pytest.raises(ValueError, match="comparison"):
        _promote(case, report=tampered)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


def test_missing_champion_and_contention_refuse_promotion(case) -> None:
    """The first alias must be initialized separately, and a second writer cannot enter."""
    client, _, name, _, _, admission = case
    client.delete_registered_model_alias(name, "champion")
    with pytest.raises(AliasConflictError, match="missing"):
        _promote(case)
    client.set_registered_model_alias(name, "champion", "1")
    with (
        admission.hold(alias_resource_id(name, "champion")),
        pytest.raises(AliasConflictError, match="admission"),
    ):
        _promote(case)


def test_first_champion_needs_absolute_quality_and_verified_receipt(case) -> None:
    """An unseeded registry must select only a rechecked, good first model."""
    from skyulf.integrations.mlflow.promotion import (
        controlled_champion_version,
        initialize_champion,
    )

    client, uri, name, heldout, _, admission = case
    client.delete_registered_model_alias(name, "champion")
    with pytest.raises(RegistryModelNotFoundError):
        resolve_model(name, alias="champion", tracking_uri=uri, registry_uri=uri)
    assert controlled_champion_version(name, tracking_uri=uri, registry_uri=uri) is None
    candidate = resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri)
    report = compare_registered_local_models(
        candidate,
        None,
        heldout,
        target_column="target",
        dataset_id="labels@5/heldout",
        metric="heldout_mse",
        min_improvement=0.0,
        quality_threshold=1.0,
        max_rows=10,
        max_bytes=10_000,
        tracking_uri=uri,
        registry_uri=uri,
    )
    receipt = initialize_champion(
        report,
        heldout,
        target_column="target",
        admission=admission,
        max_rows=10,
        max_bytes=10_000,
        tracking_uri=uri,
        registry_uri=uri,
    )
    assert receipt.kind == "initial"
    assert receipt.new_version == "2"
    assert str(client.get_model_version_by_alias(name, "champion").version) == "2"
    assert controlled_champion_version(name, tracking_uri=uri, registry_uri=uri) == "2"
    with pytest.raises(AliasConflictError):
        initialize_champion(
            report,
            heldout,
            target_column="target",
            admission=admission,
            max_rows=10,
            max_bytes=10_000,
            tracking_uri=uri,
            registry_uri=uri,
        )


def test_first_champion_rejects_missing_or_failed_quality_threshold(case) -> None:
    """Automatic bootstrap cannot use improvement alone without a baseline."""
    from skyulf.integrations.mlflow.promotion import initialize_champion

    client, uri, name, heldout, _, admission = case
    client.delete_registered_model_alias(name, "champion")
    candidate = resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri)
    for threshold in (None, -1.0):
        report = compare_registered_local_models(
            candidate,
            None,
            heldout,
            target_column="target",
            dataset_id="labels@5/heldout",
            metric="heldout_mse",
            min_improvement=0.0,
            quality_threshold=threshold,
            max_rows=10,
            max_bytes=10_000,
            tracking_uri=uri,
            registry_uri=uri,
        )
        with pytest.raises(ValueError, match="quality threshold"):
            initialize_champion(
                report,
                heldout,
                target_column="target",
                admission=admission,
                max_rows=10,
                max_bytes=10_000,
                tracking_uri=uri,
                registry_uri=uri,
            )
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias(name, "champion")


def test_controlled_champion_rejects_alias_without_receipt(case) -> None:
    """An alias set outside the guarded lifecycle must not feed automatic scoring."""
    from skyulf.integrations.mlflow.promotion import controlled_champion_version

    client, uri, name, _, _, _ = case
    with pytest.raises(AliasConflictError, match="committed receipt"):
        controlled_champion_version(name, tracking_uri=uri, registry_uri=uri)
    client.set_registered_model_tag(name, "pending_alias_event", "unresolved")
    with pytest.raises(AliasConflictError, match="pending"):
        controlled_champion_version(name, tracking_uri=uri, registry_uri=uri)


def test_exclusive_alias_writer_requires_external_serialization() -> None:
    """A table-free UC writer opts into an explicit alias resource contract."""
    admission = ExclusiveAliasWriterAdmission()
    assert _admission(admission, "databricks-uc") is admission
    with admission.hold(alias_resource_id("catalog.schema.model")):
        assert admission.local_only is False
    with pytest.raises(ValueError, match="alias resource ID"), admission.hold("not-an-alias"):
        pass


def test_rollback_rejects_newer_alias_state(case) -> None:
    """A receipt cannot roll back a different current model version."""
    client, uri, name, _, _, admission = case
    _stage(case)
    receipt = _promote(case)
    client.set_registered_model_alias(name, "champion", "1")
    with pytest.raises(AliasConflictError, match="receipt disagrees"):
        rollback_promotion(
            receipt,
            expected_current_version="2",
            admission=admission,
            tracking_uri=uri,
            registry_uri=uri,
        )


def test_permission_denial_does_not_move_alias(case, monkeypatch) -> None:
    """A denied registry write is typed and leaves the previous alias selected."""
    client, _, name, _, _, _ = case
    error = mlflow.exceptions.MlflowException("denied")
    error.error_code = "PERMISSION_DENIED"

    def denied(*args, **kwargs):
        """Simulate a restricted registry principal at the mutation boundary."""
        raise error

    _stage(case)
    monkeypatch.setattr(client, "set_registered_model_alias", denied)
    monkeypatch.setattr("skyulf.integrations.mlflow.promotion._make_client", lambda *args: client)
    with pytest.raises(RegistryAccessError):
        _promote(case)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


def test_lost_alias_write_response_is_reported_as_unknown(case, monkeypatch) -> None:
    """A committed alias move with a lost response must never be reported as safely denied."""
    client, _, name, _, _, _ = case
    _stage(case)
    original = client.set_registered_model_alias

    def move_then_fail(model_name, alias, version):
        """Simulate a transport failure after the champion alias moved."""
        original(model_name, alias, version)
        if alias == "champion":
            raise mlflow.exceptions.MlflowException("response lost")

    monkeypatch.setattr(client, "set_registered_model_alias", move_then_fail)
    monkeypatch.setattr("skyulf.integrations.mlflow.promotion._make_client", lambda *args: client)
    with pytest.raises(AliasOutcomeUnknownError, match="inspect prepared event"):
        _promote(case)
    assert str(client.get_model_version_by_alias(name, "champion").version) == "2"
    assert client.get_registered_model(name).tags.get("pending_alias_event")
    from skyulf.integrations.mlflow.promotion import controlled_champion_version

    with pytest.raises(AliasConflictError, match="pending"):
        controlled_champion_version(name, tracking_uri=case[1], registry_uri=case[1])


def test_old_receipt_cannot_rollback_newer_promotion_to_same_version(case) -> None:
    """An old receipt must not undo a later promotion even when its version numbers match."""
    client, uri, name, _, _, admission = case
    _stage(case)
    old_receipt = _promote(case)
    rollback_promotion(
        old_receipt,
        expected_current_version="2",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    _stage(case)
    new_receipt = _promote(case)
    assert new_receipt.event_id != old_receipt.event_id
    with pytest.raises(AliasConflictError, match="superseded"):
        rollback_promotion(
            old_receipt,
            expected_current_version="2",
            admission=admission,
            tracking_uri=uri,
            registry_uri=uri,
        )
    assert str(client.get_model_version_by_alias(name, "champion").version) == "2"


def test_global_uc_registry_rejects_local_admission(tmp_path, monkeypatch) -> None:
    """Implicit UC registry configuration must still require distributed admission."""
    monkeypatch.setattr(mlflow, "get_registry_uri", lambda: "databricks-uc")
    with pytest.raises(ValueError, match="distributed"):
        _admission(LocalAliasAdmission(tmp_path / "locks"), None)


def test_receipt_rejects_conflicting_version_field_names() -> None:
    """Legacy and current field names must never silently choose different prior versions."""
    from skyulf.integrations.mlflow.promotion import _read_event

    with pytest.raises(ValueError, match="[Dd]uplicate|[Aa]mbiguous"):
        _read_event(json.dumps({"action": "promotion", "from": "1", "from_version": "2"}))
