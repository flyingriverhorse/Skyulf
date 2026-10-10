"""Completed rollouts retain the saved comparison and shared alias writer checks."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

pytest.importorskip("mlflow")

from skyulf.integrations.databricks.lifecycle import approval  # noqa: E402
from skyulf.integrations.databricks.serving.rollout_policy import (  # noqa: E402
    DailyRolloutPolicy,
    RolloutEvidence,
    RolloutState,
)
from skyulf.integrations.mlflow.lifecycle.promotion import (  # noqa: E402
    AliasChangeReceipt,
    ExclusiveAliasWriterAdmission,
)


@pytest.fixture
def rollout_api():
    """Require the guarded integration before exercising any promotion behavior."""
    name = "skyulf.integrations.databricks.serving.rollout_promotion"
    assert find_spec(name) is not None, "Guarded rollout promotion is missing"
    return import_module(name)


@pytest.fixture
def completed(rollout_api, monkeypatch):
    """Represent the controller's verified durable result and record lock ownership."""
    now = datetime(2026, 10, 10, tzinfo=UTC)
    start = (now - timedelta(days=1)).isoformat()
    state = RolloutState(
        "roll",
        "endpoint",
        "main.ml.model",
        "1",
        "main.ml.model",
        "2",
        100,
        start,
        start,
        "COMPLETE",
    )
    evidence = RolloutEvidence(
        "roll",
        "endpoint",
        "main.ml.model",
        "1",
        "main.ml.model",
        "2",
        100,
        start,
        start,
        now.isoformat(),
        now.isoformat(),
        "PASS",
        "LIVE",
        "passed",
    )
    config = {"model_name": "main.ml.model", "max_rows": 10, "max_input_mb": 1}
    pin = {
        "auto_promote": True,
        "model_name": state.challenger_model_name,
        "candidate_version": "2",
        "expected_champion_version": "1",
        "comparison_sha256": "a" * 64,
        "config_sha256": rollout_api.approval_config_digest(config),
    }
    record = {
        "receipt_id": "receipt",
        "state": asdict(state),
        "policy": asdict(DailyRolloutPolicy()),
        "evidence": asdict(evidence),
        "promotion": pin,
        "promotion_receipt": None,
    }
    scope = {"held": False}

    @contextmanager
    def hold(client, *, store, admission):
        """Model the real controller's lock while detecting out-of-scope writes."""
        scope["held"] = True
        try:
            yield deepcopy(record)
        finally:
            scope["held"] = False

    monkeypatch.setattr(rollout_api, "hold_completed_rollout", hold)
    receipt = AliasChangeReceipt(
        "event", "promotion", "main.ml.model", "champion", "1", "2", "a" * 64, "prior"
    )

    def approve(spark, config, **kwargs):
        """Require endpoint admission and invoke the actual fresh final-stage guard."""
        assert scope["held"]
        kwargs["fresh_approval_guard"]()
        return receipt

    operation = Mock(side_effect=approve)
    monkeypatch.setattr(rollout_api, "approve_saved_candidate", operation)
    store = Mock()
    store.write.side_effect = lambda value, **kwargs: value
    return config, record, operation, store, now, receipt


def test_completed_rollout_promotes_under_endpoint_admission_and_saves_receipt(
    rollout_api, completed
):
    """Only controller-verified completion may perform and durably acknowledge an alias change."""
    config, record, operation, store, now, receipt = completed
    authority = ExclusiveAliasWriterAdmission()
    result = rollout_api.promote_completed_rollout(
        None, None, config, store=store, admission=authority, clock=lambda: now
    )
    assert result == receipt
    assert operation.call_args.kwargs["admission"] is authority
    assert store.write.call_args.kwargs["expected_receipt_id"] == "receipt"
    assert store.write.call_args.args[0]["promotion_receipt"] == asdict(receipt)


@pytest.mark.parametrize("change", ["disabled", "comparison", "config", "model", "phase"])
def test_changed_completion_pin_rejects_before_alias_operation(rollout_api, completed, change):
    """Promotion cannot substitute new evidence, policy, model identity or an incomplete stage."""
    config, record, operation, store, now, _ = completed
    if change == "disabled":
        record["promotion"]["auto_promote"] = False
    if change == "comparison":
        record["promotion"]["comparison_sha256"] = "invalid"
    if change == "config":
        config["max_rows"] = 11
    if change == "model":
        record["state"]["champion_model_name"] = "other.ml.model"
    if change == "phase":
        record["state"]["phase"] = "ACTIVE"
    with pytest.raises(ValueError):
        rollout_api.promote_completed_rollout(
            None,
            None,
            config,
            store=store,
            admission=ExclusiveAliasWriterAdmission(),
            clock=lambda: now,
        )
    operation.assert_not_called()
    store.write.assert_not_called()


def test_stale_complete_refuses_first_alias_mutation(rollout_api, completed):
    """A completed traffic stage must not authorize an alias change days after observation."""
    config, _, _, store, now, _ = completed
    with pytest.raises(ValueError, match="stale"):
        rollout_api.promote_completed_rollout(
            None,
            None,
            config,
            store=store,
            admission=ExclusiveAliasWriterAdmission(),
            clock=lambda: now + timedelta(days=2),
        )
    store.write.assert_not_called()


def test_committed_replay_verifies_original_receipt_even_after_evidence_ages(
    rollout_api, completed
):
    """Acknowledging a completed alias operation never starts a new promotion or stage."""
    config, record, operation, store, now, receipt = completed
    record["promotion_receipt"] = asdict(receipt)
    operation.side_effect = None
    operation.return_value = receipt
    assert (
        rollout_api.promote_completed_rollout(
            None,
            None,
            config,
            store=store,
            admission=ExclusiveAliasWriterAdmission(),
            clock=lambda: now + timedelta(days=2),
        )
        == receipt
    )
    assert operation.call_count == 1
    store.write.assert_not_called()


def test_saved_promotion_cannot_start_a_second_mutation(rollout_api, completed):
    """A reverted or superseded alias must never trigger another promotion on a COMPLETE retry."""
    config, record, _, store, now, receipt = completed
    record["promotion_receipt"] = asdict(receipt)
    with pytest.raises(ValueError, match="already|superseded|current"):
        rollout_api.promote_completed_rollout(
            None,
            None,
            config,
            store=store,
            admission=ExclusiveAliasWriterAdmission(),
            clock=lambda: now,
        )
    store.write.assert_not_called()


@pytest.fixture
def promotion_builder(rollout_api, monkeypatch):
    """Resolve concrete packages without requiring a live registry in builder tests."""
    plan = SimpleNamespace(
        champion=SimpleNamespace(
            spec=SimpleNamespace(model_name="main.ml.model", model_version="1")
        ),
        challenger=SimpleNamespace(
            spec=SimpleNamespace(model_name="main.ml.model", model_version="2")
        ),
    )
    loader = Mock()
    monkeypatch.setattr(
        rollout_api, "resolve_model", Mock(side_effect=lambda name, version, **kw: version)
    )
    monkeypatch.setattr(rollout_api, "load_registered_pipeline", loader)
    monkeypatch.setattr(rollout_api, "require_mlflow", Mock())
    monkeypatch.setattr(rollout_api, "make_registry_client", Mock())
    evidence = Mock(
        return_value=(SimpleNamespace(champion_version="1", eligible=True), None, None, None)
    )
    monkeypatch.setattr(rollout_api, "load_candidate_evidence", evidence)
    return plan, loader, evidence


def test_builder_checks_both_pipeline_packages_and_avoids_saving_job_secrets(
    rollout_api, promotion_builder
):
    """Auto promotion is pipeline-only and persists only relevant configuration digests."""
    plan, loader, _ = promotion_builder
    config = {"model_name": "main.ml.model", "secret": "not-persisted"}
    pin = rollout_api.build_rollout_promotion(
        plan, config, auto_promote=True, comparison_sha256="a" * 64
    )
    assert loader.call_count == 2
    assert pin["config_sha256"] == rollout_api.approval_config_digest(
        {"model_name": "main.ml.model"}
    )
    assert "not-persisted" not in str(pin)


@pytest.mark.parametrize("option", [True, False, "true", 1, None])
def test_builder_explicit_boolean_and_modelset_rejection(rollout_api, promotion_builder, option):
    """Unsupported model sets fail during setup, while disabled promotion performs no registry work."""
    plan, loader, evidence = promotion_builder
    loader.side_effect = ValueError("Not a pipeline artifact: model_set")
    if option is False:
        assert (
            rollout_api.build_rollout_promotion(plan, {}, auto_promote=option, comparison_sha256="")
            is None
        )
        loader.assert_not_called()
    else:
        with pytest.raises(ValueError, match="bool|pipeline"):
            rollout_api.build_rollout_promotion(
                plan,
                {"model_name": "main.ml.model"},
                auto_promote=option,
                comparison_sha256="a" * 64,
            )
    evidence.assert_not_called()


@dataclass
class SavedSpec:
    """Carry only bounded saved snapshot fields consumed by the approval boundary."""

    max_rows: int = 20
    max_bytes: int = 1024 * 1024
    target_column: str = "target"


@pytest.fixture
def saved_approval(monkeypatch):
    """Isolate registry and source I/O while retaining real saved-approval dispatch."""
    report = SimpleNamespace(candidate_digest="digest")
    config = {"model_name": "main.ml.model", "max_rows": 10, "max_input_mb": 1}
    reader = Mock(return_value=pd.DataFrame({"x": [1.0, 2.0], "target": [2.0, 4.0]}))
    promoter = Mock(return_value="committed-receipt")
    monkeypatch.setattr(approval, "require_mlflow", Mock())
    monkeypatch.setattr(approval, "make_registry_client", Mock())
    monkeypatch.setattr(
        approval,
        "load_candidate_evidence",
        Mock(return_value=(report, SavedSpec(), "pandas", None)),
    )
    monkeypatch.setattr(approval, "_validate_approval_policy", Mock())
    monkeypatch.setattr(
        approval, "resolve_model", Mock(return_value=SimpleNamespace(digest="digest"))
    )
    monkeypatch.setattr(approval, "controlled_champion_version", Mock(return_value="1"))
    monkeypatch.setattr(approval, "verify_staged_challenger", Mock())
    monkeypatch.setattr(approval.candidate, "read_training_snapshot", reader)
    monkeypatch.setattr(
        approval.candidate,
        "split_labeled_snapshot",
        lambda frame, spec, engine: (None, frame, None),
    )
    monkeypatch.setattr(approval, "promote_candidate", promoter)
    return config, reader, promoter


def test_shared_saved_approval_uses_injected_alias_authority(saved_approval):
    """Automatic approval must use the same distributed authority as other alias writers."""
    operation = getattr(approval, "approve_saved_candidate", None)
    assert callable(operation), "Saved approval does not expose shared alias admission"
    config, reader, promoter = saved_approval
    authority = ExclusiveAliasWriterAdmission()
    result = operation(
        None,
        config,
        candidate_version="2",
        comparison_sha256="a" * 64,
        expected_champion_version="1",
        admission=authority,
    )
    assert result == "committed-receipt"
    assert reader.call_args.args[1].max_rows == 10
    assert promoter.call_args.kwargs["admission"] is authority


def test_manual_wrapper_retains_policy_gate_and_exclusive_default(monkeypatch):
    """Extracting shared approval must not grant automatic callers manual approval rights."""
    operation = getattr(approval, "approve_saved_candidate", None)
    assert callable(operation), "Saved approval extraction is missing"
    shared = Mock(return_value="receipt")
    monkeypatch.setattr(approval, "approve_saved_candidate", shared)
    arguments = {
        "candidate_version": "2",
        "comparison_sha256": "a" * 64,
        "expected_champion_version": "1",
    }
    with pytest.raises(ValueError, match="manual_approval"):
        approval.approve_candidate(None, {"promotion_policy": "automatic"}, **arguments)
    shared.assert_not_called()
    assert (
        approval.approve_candidate(None, {"promotion_policy": "manual_approval"}, **arguments)
        == "receipt"
    )
    assert isinstance(shared.call_args.kwargs["admission"], ExclusiveAliasWriterAdmission)


def test_shared_approval_preserves_snapshot_failure_without_mutation(saved_approval):
    """A changed native feature source must never bypass the original snapshot guard."""
    operation = getattr(approval, "approve_saved_candidate", None)
    assert callable(operation), "Saved approval extraction is missing"
    config, reader, promoter = saved_approval
    reader.side_effect = ValueError("Feature snapshot changed")
    with pytest.raises(ValueError, match="snapshot changed"):
        operation(
            None,
            config,
            candidate_version="2",
            comparison_sha256="a" * 64,
            expected_champion_version="1",
            admission=ExclusiveAliasWriterAdmission(),
        )
    promoter.assert_not_called()


def test_shared_approval_replays_committed_receipt_without_reading_rows(
    saved_approval, monkeypatch
):
    """A crash after registry commit must recover the exact active receipt without another write."""
    operation = getattr(approval, "approve_saved_candidate", None)
    assert callable(operation), "Saved approval extraction is missing"
    config, reader, promoter = saved_approval
    monkeypatch.setattr(approval, "controlled_champion_version", Mock(return_value="2"))
    verifier = Mock(return_value="original-receipt")
    monkeypatch.setattr(approval, "_completed_approval", verifier)
    result = operation(
        None,
        config,
        candidate_version="2",
        comparison_sha256="a" * 64,
        expected_champion_version="1",
        admission=ExclusiveAliasWriterAdmission(),
    )
    reader.assert_not_called()
    promoter.assert_not_called()
    assert result == "original-receipt"


def test_fresh_guard_blocks_new_mutation_but_not_committed_replay(saved_approval, monkeypatch):
    """Expired completion evidence prevents new alias writes but cannot hide a committed receipt."""
    config, reader, promoter = saved_approval
    guard = Mock(side_effect=ValueError("Final evidence is stale"))
    options = {
        "candidate_version": "2",
        "comparison_sha256": "a" * 64,
        "expected_champion_version": "1",
        "admission": ExclusiveAliasWriterAdmission(),
        "fresh_approval_guard": guard,
    }
    with pytest.raises(ValueError, match="stale"):
        approval.approve_saved_candidate(None, config, **options)
    promoter.assert_not_called()
    monkeypatch.setattr(approval, "controlled_champion_version", Mock(return_value="2"))
    monkeypatch.setattr(approval, "_completed_approval", Mock(return_value="original"))
    guard.reset_mock()
    reader.reset_mock()
    assert approval.approve_saved_candidate(None, config, **options) == "original"
    guard.assert_not_called()
    reader.assert_not_called()
