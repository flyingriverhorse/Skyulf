"""Recover only fully applied or unchanged lifecycle transactions in a real registry."""

import json
from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import pytest

mlflow = pytest.importorskip("mlflow")

from skyulf.integrations.mlflow.lifecycle import promotion
from skyulf.integrations.mlflow.registration.registry import RegistryOperationError


@pytest.fixture
def registry(tmp_path):
    """Use real SQLite aliases and tags without unrelated model training."""
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    name = "alias-recovery"
    client.create_registered_model(name)
    for _ in range(3):
        client.create_model_version(name, source=tmp_path.as_uri())
    initial = promotion.AliasChangeReceipt(
        uuid4().hex, "initial", name, "champion", None, "1", None, None
    )
    promotion.commit_change(client, initial, [("champion", "1", None)])
    receipt = promotion.AliasChangeReceipt(
        uuid4().hex, "promotion", name, "champion", "1", "2", "a" * 64, initial.event_id
    )
    return client, uri, receipt, promotion.LocalAliasAdmission(tmp_path / "locks")


def _updates():
    """Include a second role so interrupted writes are distinguishable."""
    return [("previous_champion", "1", None), ("champion", "2", "1")]


def _interrupt(registry, monkeypatch, applied, *, updates=None, version_tags=None):
    """Stop before a chosen alias call or before final metadata without undoing real writes."""
    client, _, receipt, _ = registry
    changes = _updates() if updates is None else updates
    with monkeypatch.context() as patch:
        original = client.set_registered_model_alias
        calls = 0

        def write(name, alias, version):
            """Inject a transport failure after the requested number of real calls."""
            nonlocal calls
            if calls == applied:
                raise RuntimeError("injected registry interruption")
            calls += 1
            return original(name, alias, version)

        patch.setattr(client, "set_registered_model_alias", write)
        patch.setattr(
            promotion,
            "_finish_alias_change",
            Mock(side_effect=promotion.AliasOutcomeUnknownError("injected metadata interruption")),
        )
        with pytest.raises(promotion.AliasOutcomeUnknownError):
            promotion.commit_change(client, receipt, changes, version_tags=version_tags)
    assert (
        client.get_registered_model(receipt.model_name).tags["pending_alias_event"]
        == receipt.event_id
    )


def _reconcile(registry):
    """Call the public recovery API with the original admission authority."""
    _, uri, receipt, admission = registry
    return promotion.reconcile_alias_change(
        receipt.model_name,
        version=receipt.new_version,
        event_id=receipt.event_id,
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )


@pytest.mark.parametrize("applied,expected", [(0, "aborted"), (2, "committed")])
def test_reconcile_complete_states_without_alias_writes(registry, monkeypatch, applied, expected):
    """An interrupted operation must unblock safely without reapplying any alias mutation."""
    client, _, receipt, _ = registry
    _interrupt(registry, monkeypatch, applied, version_tags={"validation_reason": "checked"})
    before = dict(client.get_registered_model(receipt.model_name).aliases)
    with monkeypatch.context() as patch:
        set_alias = Mock(side_effect=AssertionError("repair cannot move aliases"))
        patch.setattr(mlflow.MlflowClient, "set_registered_model_alias", set_alias)
        patch.setattr(mlflow.MlflowClient, "delete_registered_model_alias", set_alias)
        assert _reconcile(registry) == expected
        set_alias.assert_not_called()
    model = client.get_registered_model(receipt.model_name)
    tags = client.get_model_version(receipt.model_name, receipt.new_version).tags
    assert dict(model.aliases) == before
    assert "pending_alias_event" not in model.tags
    assert promotion.read_event(tags[promotion.event_tag(receipt.event_id)])["s"] == expected
    assert tags.get("validation_reason") == ("checked" if expected == "committed" else None)


def test_partial_aliases_refuse_and_report_every_role(registry, monkeypatch):
    """A mixed transaction must remain blocked with enough facts for operator repair."""
    client, _, receipt, _ = registry
    _interrupt(registry, monkeypatch, 1)
    before = client.get_model_version(receipt.model_name, receipt.new_version).tags
    with pytest.raises(promotion.AliasConflictError) as error:
        _reconcile(registry)
    for role in ("champion", "challenger", "previous_champion", "previous_challenger"):
        assert role in str(error.value)
    assert client.get_model_version(receipt.model_name, receipt.new_version).tags == before
    assert (
        client.get_registered_model(receipt.model_name).tags["pending_alias_event"]
        == receipt.event_id
    )


def test_full_intent_is_durable_before_pending(registry, monkeypatch):
    """Recovery evidence must include custom status targets before mutation is authorized."""
    client, _, receipt, _ = registry
    _interrupt(registry, monkeypatch, 0, version_tags={"validation_reason": "checked"})
    tags = client.get_model_version(receipt.model_name, receipt.new_version).tags
    assert "alias_intent" in tags
    assert all(
        len(value.encode()) <= 256 for key, value in tags.items() if key.startswith("alias_intent_")
    )


@pytest.mark.parametrize("applied", [0, 2])
def test_reconciliation_retry_is_read_only(registry, monkeypatch, applied):
    """Retry after a lost successful response must not rewrite event timestamps or metadata."""
    client, _, receipt, _ = registry
    _interrupt(registry, monkeypatch, applied)
    expected = _reconcile(registry)
    before = client.get_model_version(receipt.model_name, receipt.new_version).tags
    with monkeypatch.context() as patch:
        mutation = Mock(side_effect=AssertionError("terminal retry must be read-only"))
        for method in (
            "set_model_version_tag",
            "set_registered_model_tag",
            "delete_registered_model_tag",
        ):
            patch.setattr(mlflow.MlflowClient, method, mutation)
        assert _reconcile(registry) == expected
        mutation.assert_not_called()
    assert client.get_model_version(receipt.model_name, receipt.new_version).tags == before


@pytest.mark.parametrize(
    "corruption", ["missing", "chunk", "foreign", "role", "marker", "event", "terminal"]
)
def test_inconsistent_recovery_evidence_is_read_only(registry, monkeypatch, corruption):
    """Legacy, overwritten and contradictory evidence must not authorize cleanup."""
    client, _, receipt, _ = registry
    _interrupt(registry, monkeypatch, 0)
    name, version = receipt.model_name, receipt.new_version
    if corruption == "missing":
        client.delete_model_version_tag(name, version, "alias_intent")
    elif corruption == "chunk":
        client.set_model_version_tag(name, version, "alias_intent_0000", "{}")
    elif corruption == "foreign":
        client.set_registered_model_tag(name, "pending_alias_event", "another-event")
    elif corruption == "role":
        client.set_registered_model_alias(name, "challenger", "3")
    elif corruption == "marker":
        client.set_registered_model_tag(name, "champion_current_event", "{}")
    elif corruption == "event":
        client.set_model_version_tag(name, version, promotion.event_tag(receipt.event_id), "[]")
    else:
        promotion._write_event(client, receipt, "committed")
    model_before = client.get_registered_model(name)
    version_before = client.get_model_version(name, version).tags
    with pytest.raises(promotion.AliasConflictError):
        _reconcile(registry)
    assert client.get_registered_model(name).tags == model_before.tags
    assert client.get_model_version(name, version).tags == version_before


@pytest.mark.parametrize("status", ["prepared", "committed"])
def test_tag_only_recovery_requires_unambiguous_metadata(registry, monkeypatch, status):
    """Empty alias lists cannot accidentally count as a fully applied transaction."""
    client, uri, receipt, admission = registry
    receipt = replace(receipt, kind="evaluation_error", alias="challenger", prior_version="2")
    client.set_registered_model_alias(receipt.model_name, "challenger", "2")
    registry = client, uri, receipt, admission
    _interrupt(registry, monkeypatch, 0, updates=[], version_tags={"validation_status": "error"})
    if status == "committed":
        client.set_model_version_tag(receipt.model_name, "2", "validation_status", "error")
        promotion._write_event(client, receipt, "committed")
    assert _reconcile(registry) == ("committed" if status == "committed" else "aborted")


def test_partial_tag_only_operation_requires_manual_inspection(registry, monkeypatch):
    """Partly written rejection metadata must never be mistaken for an unchanged operation."""
    client, uri, receipt, admission = registry
    receipt = replace(receipt, kind="rejection", alias="challenger", prior_version="2")
    client.set_registered_model_alias(receipt.model_name, "challenger", "2")
    registry = client, uri, receipt, admission
    _interrupt(
        registry,
        monkeypatch,
        0,
        updates=[],
        version_tags={"approval_status": "rejected", "approval_reason": "review"},
    )
    client.set_model_version_tag(receipt.model_name, "2", "approval_status", "rejected")
    with pytest.raises(promotion.AliasConflictError, match="manual"):
        _reconcile(registry)
    assert (
        client.get_registered_model(receipt.model_name).tags["pending_alias_event"]
        == receipt.event_id
    )


@pytest.mark.parametrize("applied", [0, 2])
def test_repair_never_clears_another_pending_owner(registry, monkeypatch, applied):
    """Ownership must be reread immediately before both abort and completion cleanup."""
    client, _, receipt, _ = registry
    _interrupt(registry, monkeypatch, applied)
    original = mlflow.MlflowClient.set_model_version_tag

    def write(self, name, version, key, value):
        """Replace ownership when the terminal event is written, before cleanup."""
        result = original(self, name, version, key, value)
        if key == promotion.event_tag(receipt.event_id):
            client.set_registered_model_tag(name, "pending_alias_event", "another-event")
        return result

    monkeypatch.setattr(mlflow.MlflowClient, "set_model_version_tag", write)
    with pytest.raises(promotion.AliasOutcomeUnknownError):
        _reconcile(registry)
    assert (
        client.get_registered_model(receipt.model_name).tags["pending_alias_event"]
        == "another-event"
    )


@pytest.mark.parametrize("target", ["model", "version"])
def test_intent_capacity_fails_before_preparation(registry, target):
    """UC tag exhaustion must be detected before publishing any pending or alias mutation."""
    client, _, receipt, _ = registry
    for index in range(49):
        if target == "model":
            client.set_registered_model_tag(receipt.model_name, f"external_{index}", "value")
        else:
            client.set_model_version_tag(receipt.model_name, "2", f"external_{index}", "value")
    model_before = client.get_registered_model(receipt.model_name)
    version_before = client.get_model_version(receipt.model_name, "2").tags
    with pytest.raises(RegistryOperationError) as error:
        promotion.commit_change(client, receipt, _updates())
    assert isinstance(error.value.__cause__, ValueError)
    assert "50-tag" in str(error.value.__cause__)
    assert client.get_registered_model(receipt.model_name).tags == model_before.tags
    assert client.get_registered_model(receipt.model_name).aliases == model_before.aliases
    assert client.get_model_version(receipt.model_name, "2").tags == version_before


def test_recovery_intent_reuses_slots_after_a_completed_operation(registry, monkeypatch):
    """Repeated events retain compact receipts without accumulating full intent tags."""
    client, uri, receipt, admission = registry
    _interrupt(registry, monkeypatch, 0, version_tags={"review": "a" * 256})
    assert _reconcile(registry) == "aborted"
    before = client.get_model_version(receipt.model_name, "2").tags
    next_receipt = replace(receipt, event_id=uuid4().hex)
    promotion.commit_change(client, next_receipt, _updates())
    after = client.get_model_version(receipt.model_name, "2").tags
    assert promotion.event_tag(receipt.event_id) in after
    assert promotion.event_tag(next_receipt.event_id) in after
    assert len([key for key in after if key.startswith("alias_intent")]) <= len(
        [key for key in before if key.startswith("alias_intent")]
    )
    with pytest.raises(promotion.AliasConflictError, match="intent"):
        _reconcile((client, uri, receipt, admission))
    assert _reconcile((client, uri, next_receipt, admission)) == "committed"


def test_admission_conflict_prevents_recovery(registry, monkeypatch):
    """Recovery must participate in the same lock as normal alias writers."""
    client, _, receipt, admission = registry
    _interrupt(registry, monkeypatch, 2)
    before = client.get_model_version(receipt.model_name, "2").tags
    with (
        admission.hold(promotion.alias_resource_id(receipt.model_name)),
        pytest.raises(promotion.AliasConflictError, match="admission"),
    ):
        _reconcile(registry)
    assert client.get_model_version(receipt.model_name, "2").tags == before


@pytest.mark.parametrize("fail_at", ["status", "event", "active", "clear", "clear_response"])
def test_interrupted_completion_can_be_retried(registry, monkeypatch, fail_at):
    """Every uncertain metadata boundary must converge without moving aliases again."""
    client, _, receipt, _ = registry
    _interrupt(registry, monkeypatch, 2, version_tags={"validation_status": "passed"})
    if fail_at in {"status", "event"}:
        method = "set_model_version_tag"
        target = (
            "validation_status" if fail_at == "status" else promotion.event_tag(receipt.event_id)
        )
    else:
        method = (
            "set_registered_model_tag" if fail_at == "active" else "delete_registered_model_tag"
        )
        target = "champion_current_event" if fail_at == "active" else "pending_alias_event"
    original = getattr(mlflow.MlflowClient, method)

    def fail(self, *args, **kwargs):
        """Fail only the selected write, optionally after its server-side effect."""
        key = args[2] if method == "set_model_version_tag" else args[1]
        if key != target:
            return original(self, *args, **kwargs)
        if fail_at == "clear_response":
            original(self, *args, **kwargs)
        raise RuntimeError("injected recovery transport failure")

    with monkeypatch.context() as patch:
        patch.setattr(mlflow.MlflowClient, method, fail)
        with pytest.raises(promotion.AliasOutcomeUnknownError):
            _reconcile(registry)
    if fail_at != "clear_response":
        assert (
            client.get_registered_model(receipt.model_name).tags["pending_alias_event"]
            == receipt.event_id
        )
    assert _reconcile(registry) == "committed"
    assert "pending_alias_event" not in client.get_registered_model(receipt.model_name).tags


@pytest.mark.parametrize("applied,expected", [(0, "aborted"), (2, "committed")])
def test_rollback_recovery(registry, monkeypatch, applied, expected):
    """Rollback uses reversed roles and must obey the same two-state recovery contract."""
    client, uri, receipt, admission = registry
    promotion.commit_change(client, receipt, _updates())
    rollback = replace(
        receipt,
        event_id=uuid4().hex,
        kind="rollback",
        prior_version="2",
        new_version="1",
        parent_event_id=receipt.event_id,
    )
    registry = client, uri, rollback, admission
    _interrupt(
        registry,
        monkeypatch,
        applied,
        updates=[("champion", "1", "2"), ("previous_champion", "2", "1")],
    )
    assert _reconcile(registry) == expected
    assert promotion.read_optional_alias(client, receipt.model_name, "champion") == (
        "1" if applied else "2"
    )


@pytest.mark.parametrize("applied", [1, 2])
def test_displaced_challenger_history_recovery(registry, monkeypatch, applied):
    """History is a guarded fourth role, and its final receipt must be recoverable."""
    client, uri, receipt, admission = registry
    nominee = replace(receipt, kind="nomination", alias="challenger", prior_version=None)
    promotion.commit_change(client, nominee, [("challenger", "2", None)])
    next_nominee = replace(
        nominee,
        event_id=uuid4().hex,
        prior_version="2",
        new_version="3",
        parent_event_id=nominee.event_id,
    )
    registry = client, uri, next_nominee, admission
    _interrupt(registry, monkeypatch, applied, updates=[("challenger", "3", "2")])
    if applied == 1:
        with pytest.raises(promotion.AliasConflictError, match="previous_challenger"):
            _reconcile(registry)
    else:
        assert _reconcile(registry) == "committed"
        marker = json.loads(
            client.get_registered_model(receipt.model_name).tags[
                "previous_challenger_current_event"
            ]
        )
        assert marker == {"event_id": next_nominee.event_id, "version": "3"}
    assert promotion.read_optional_alias(client, receipt.model_name, "previous_challenger") == (
        "2" if applied == 2 else None
    )


def test_intent_write_failure_never_publishes_pending(registry, monkeypatch):
    """An incomplete snapshot must not authorize even the first alias call."""
    client, _, receipt, _ = registry
    original = client.set_model_version_tag

    def fail(name, version, key, value):
        """Interrupt storage after one real chunk has already been written."""
        if key == "alias_intent_0001":
            raise RuntimeError("lost intent write")
        return original(name, version, key, value)

    monkeypatch.setattr(client, "set_model_version_tag", fail)
    with pytest.raises(RegistryOperationError) as error:
        promotion.commit_change(client, receipt, _updates())
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "intent write" in str(error.value.__cause__)
    assert "pending_alias_event" not in client.get_registered_model(receipt.model_name).tags
    assert promotion.read_optional_alias(client, receipt.model_name, "champion") == "1"
    assert (
        promotion.event_tag(receipt.event_id)
        not in client.get_model_version(receipt.model_name, "2").tags
    )


@pytest.mark.parametrize("field", ["aliases", "model_tags", "version_tags"])
def test_malformed_intent_mapping_is_a_typed_conflict(registry, monkeypatch, field):
    """Malformed stored mappings must remain blocked through the documented conflict API."""
    from skyulf.integrations.mlflow.lifecycle.alias_intent import (
        read_alias_intent,
        write_alias_intent,
    )

    client, _, receipt, _ = registry
    _interrupt(registry, monkeypatch, 0)
    payload = read_alias_intent(client.get_model_version(receipt.model_name, "2").tags)
    payload[field] = list(payload[field])
    write_alias_intent(client, receipt.model_name, "2", payload, set())
    with pytest.raises(promotion.AliasConflictError, match="intent"):
        _reconcile(registry)
    assert (
        client.get_registered_model(receipt.model_name).tags["pending_alias_event"]
        == receipt.event_id
    )


@pytest.mark.parametrize("applied", [0, 2])
def test_corrupted_unchanged_history_refuses_before_any_recovery_write(
    registry, monkeypatch, applied
):
    """An unrelated broken history receipt must not be discovered after clearing pending."""
    client, uri, receipt, admission = registry
    first = replace(receipt, kind="nomination", alias="challenger", prior_version=None)
    promotion.commit_change(client, first, [("challenger", "2", None)])
    next_nominee = replace(
        first,
        event_id=uuid4().hex,
        new_version="3",
        prior_version="2",
        parent_event_id=first.event_id,
    )
    promotion.commit_change(client, next_nominee, [("challenger", "3", "2")])
    candidate = replace(receipt, event_id=uuid4().hex, new_version="3")
    registry = client, uri, candidate, admission
    _interrupt(
        registry,
        monkeypatch,
        applied,
        updates=[
            ("previous_champion", "1", None),
            ("champion", "3", "1"),
            ("challenger", None, "3"),
        ],
    )
    client.set_model_version_tag(
        receipt.model_name, "3", f"challenger_history_{next_nominee.event_id}", "{}"
    )
    before = client.get_model_version(receipt.model_name, "3").tags
    with pytest.raises(promotion.AliasConflictError, match="history|intent"):
        _reconcile(registry)
    assert client.get_model_version(receipt.model_name, "3").tags == before
    assert (
        client.get_registered_model(receipt.model_name).tags["pending_alias_event"]
        == candidate.event_id
    )


def test_corrupted_changed_history_is_not_overwritten_by_recovery(registry, monkeypatch):
    """Conflicting prepared history evidence requires inspection even when aliases all moved."""
    client, uri, receipt, admission = registry
    nominee = replace(receipt, kind="nomination", alias="challenger", prior_version=None)
    promotion.commit_change(client, nominee, [("challenger", "2", None)])
    candidate = replace(
        nominee,
        event_id=uuid4().hex,
        new_version="3",
        prior_version="2",
        parent_event_id=nominee.event_id,
    )
    registry = client, uri, candidate, admission
    _interrupt(registry, monkeypatch, 2, updates=[("challenger", "3", "2")])
    client.set_model_version_tag(
        receipt.model_name, "3", f"challenger_history_{candidate.event_id}", "{}"
    )
    before = client.get_model_version(receipt.model_name, "3").tags
    with pytest.raises(promotion.AliasConflictError, match="history|intent"):
        _reconcile(registry)
    assert client.get_model_version(receipt.model_name, "3").tags == before
    assert (
        client.get_registered_model(receipt.model_name).tags["pending_alias_event"]
        == candidate.event_id
    )


def test_aborting_history_change_validates_the_previous_history_receipt(registry, monkeypatch):
    """A planned history replacement cannot hide corruption in the retained old pointer."""
    client, uri, receipt, admission = registry
    first = replace(receipt, kind="nomination", alias="challenger", prior_version=None)
    promotion.commit_change(client, first, [("challenger", "2", None)])
    second = replace(
        first,
        event_id=uuid4().hex,
        new_version="3",
        prior_version="2",
        parent_event_id=first.event_id,
    )
    promotion.commit_change(client, second, [("challenger", "3", "2")])
    candidate = replace(
        first, event_id=uuid4().hex, prior_version="3", parent_event_id=second.event_id
    )
    registry = client, uri, candidate, admission
    _interrupt(registry, monkeypatch, 0, updates=[("challenger", "2", "3")])
    client.set_model_version_tag(
        receipt.model_name, "3", f"challenger_history_{second.event_id}", "{}"
    )
    before = client.get_model_version(receipt.model_name, "2").tags
    with pytest.raises(promotion.AliasConflictError, match="history"):
        _reconcile(registry)
    assert client.get_model_version(receipt.model_name, "2").tags == before
    assert (
        client.get_registered_model(receipt.model_name).tags["pending_alias_event"]
        == candidate.event_id
    )


def test_intent_permission_failure_preserves_registry_error_type(registry, monkeypatch):
    """The new pre-pending writes must preserve existing actionable permission errors."""
    from skyulf.integrations.mlflow.registration.registry import RegistryAccessError

    client, _, receipt, _ = registry
    denied = mlflow.exceptions.MlflowException("intent permission denied")
    denied.error_code = "PERMISSION_DENIED"
    monkeypatch.setattr(client, "set_model_version_tag", Mock(side_effect=denied))
    with pytest.raises(RegistryAccessError):
        promotion.commit_change(client, receipt, _updates())
    assert "pending_alias_event" not in client.get_registered_model(receipt.model_name).tags


def test_silent_terminal_event_loss_keeps_pending(registry, monkeypatch):
    """An acknowledged but absent event must not clear the recovery marker."""
    client, _, receipt, _ = registry
    _interrupt(registry, monkeypatch, 2)
    original = mlflow.MlflowClient.set_model_version_tag

    def omit(self, name, version, key, value):
        """Simulate a backend acknowledging a terminal receipt without persisting it."""
        if key != promotion.event_tag(receipt.event_id):
            return original(self, name, version, key, value)
        return None

    monkeypatch.setattr(mlflow.MlflowClient, "set_model_version_tag", omit)
    with pytest.raises(promotion.AliasOutcomeUnknownError):
        _reconcile(registry)
    assert (
        client.get_registered_model(receipt.model_name).tags["pending_alias_event"]
        == receipt.event_id
    )
