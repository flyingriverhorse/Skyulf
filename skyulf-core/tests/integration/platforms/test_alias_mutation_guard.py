"""Optional platform guards run at the actual MLflow alias network boundary."""

from unittest.mock import Mock

import pytest

pytest.importorskip("mlflow")

from skyulf.integrations.mlflow.lifecycle import promotion


def _receipt():
    """Use one concrete transition for the shared alias-write implementation."""
    return promotion.AliasChangeReceipt(
        "event", "promotion", "c.s.model", "champion", "1", "2", "digest", None
    )


@pytest.mark.parametrize("version", [None, "2"])
def test_guard_runs_before_each_network_mutation(monkeypatch, version):
    """Both deletion and assignment must mark uncertain writes before sending them."""
    events = []
    client = Mock()
    client.delete_registered_model_alias.side_effect = lambda *args: events.append("write")
    client.set_registered_model_alias.side_effect = lambda *args: events.append("write")
    monkeypatch.setattr(promotion, "read_optional_alias", lambda *args: version)
    with promotion.guard_alias_mutations(lambda: events.append("guard")):
        promotion._apply_alias_updates(
            client, _receipt(), [("champion", version, "1"), ("challenger", version, "1")]
        )
    assert events == ["guard", "write", "guard", "write"]


def test_guard_refusal_prevents_alias_request(monkeypatch):
    """Lost ownership must fail before even the first alias request is sent."""
    client = Mock()
    callback = Mock(side_effect=RuntimeError("ownership lost"))
    with (
        promotion.guard_alias_mutations(callback),
        pytest.raises(RuntimeError, match="ownership lost"),
    ):
        promotion._apply_alias_updates(client, _receipt(), [("champion", "2", "1")])
    client.set_registered_model_alias.assert_not_called()
    client.delete_registered_model_alias.assert_not_called()


def test_nested_guards_restore_outer_scope_and_clear_after_failure(monkeypatch):
    """A nested platform operation must not leak or erase its caller's admission callback."""
    events = []
    client = Mock()
    monkeypatch.setattr(promotion, "read_optional_alias", lambda *args: "2")

    def write():
        """Exercise the common network sink without higher-level registry mocks."""
        promotion._apply_alias_updates(client, _receipt(), [("champion", "2", "1")])

    with promotion.guard_alias_mutations(lambda: events.append("outer")):
        write()
        with (
            pytest.raises(ValueError, match="validation"),
            promotion.guard_alias_mutations(lambda: events.append("inner")),
        ):
            write()
            raise ValueError("validation")
        write()
    write()
    assert events == ["outer", "inner", "outer"]
    assert client.set_registered_model_alias.call_count == 4


def test_without_guard_existing_sdk_call_is_unchanged(monkeypatch):
    """Direct SDK callers keep their existing explicit admission contract."""
    client = Mock()
    monkeypatch.setattr(promotion, "read_optional_alias", lambda *args: "2")
    promotion._apply_alias_updates(client, _receipt(), [("champion", "2", "1")])
    client.set_registered_model_alias.assert_called_once_with("c.s.model", "champion", "2")


def test_validation_failure_does_not_mark_alias_mutation(monkeypatch):
    """An ordinary pre-write conflict must not dirty the platform's store owner."""
    callback, client = Mock(), Mock()
    monkeypatch.setattr(
        promotion, "assert_no_pending", Mock(side_effect=promotion.AliasConflictError("pending"))
    )
    with (
        promotion.guard_alias_mutations(callback),
        pytest.raises(promotion.AliasConflictError, match="pending"),
    ):
        promotion.commit_change(client, _receipt(), [("champion", "2", "1")])
    callback.assert_not_called()
    client.set_registered_model_alias.assert_not_called()


@pytest.mark.parametrize(
    "outcome", ["no_alias", "validation_error", "success", "after_write_error"]
)
def test_real_lifecycle_scope_acquires_only_at_alias_write(monkeypatch, outcome):
    """Evaluation has no authority I/O; uncertain actual alias writes retain the admitted owner."""
    from types import SimpleNamespace

    from skyulf.integrations.databricks.jobs.lifecycle import lifecycle_tasks
    from skyulf.integrations.databricks.lifecycle._lifecycle_state import PhaseStore
    from skyulf.integrations.databricks.observability.monitoring import reference_admission

    owners = []
    authority = Mock()

    def acquire(owner, holder, timeout):
        """Represent only Delta transport while exercising the real admission contexts."""
        owners.append(owner)

    def require(owner):
        """Every network boundary must still own the same control row."""
        assert owners == [owner]

    authority.acquire.side_effect = acquire
    authority.require.side_effect = require
    authority.release.side_effect = lambda owner: owners.remove(owner)
    factory = Mock(return_value=authority)
    monkeypatch.setattr(reference_admission, "_Authority", factory)
    store = Mock(
        spec=PhaseStore,
        request={"monitoring_namespace": "c.monitoring", "graph_version": 3},
        context=SimpleNamespace(job_run_id="123"),
        run_id="run",
        begin=Mock(),
        complete=Mock(return_value="completed"),
        client=Mock(),
    )

    def operation(*args):
        """Perform expensive validation before reaching the real shared alias-write sink."""
        factory.assert_not_called()
        if outcome == "validation_error":
            raise ValueError("validation failed")
        if outcome == "no_alias":
            return {}
        client = Mock()

        def network(*args):
            """Observe actual ownership exactly when the registry request is issued."""
            assert len(owners) == 1

        client.set_registered_model_alias.side_effect = network
        promotion._apply_alias_updates(client, _receipt(), [("champion", "2", "1")])
        assert len(owners) == 1
        if outcome == "after_write_error":
            raise RuntimeError("completion failed")
        return {}

    monkeypatch.setattr(lifecycle_tasks, "_operator", operation)
    monkeypatch.setattr(lifecycle_tasks, "record_phase_failure", Mock())
    monkeypatch.setattr(promotion, "read_optional_alias", lambda *args: "2")
    if outcome.endswith("error"):
        with pytest.raises((ValueError, RuntimeError), match="failed"):
            lifecycle_tasks._execute_phase("spark", store, "operator", {})
    else:
        assert lifecycle_tasks._execute_phase("spark", store, "operator", {}) == "completed"
    assert factory.call_count == (1 if outcome in {"success", "after_write_error"} else 0)
    assert bool(owners) is (outcome == "after_write_error")
    # The callback is removed even when a dirty remote owner remains for recovery.
    assert promotion._BEFORE_ALIAS_WRITE.get() is None
