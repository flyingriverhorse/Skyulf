"""Retraining submission retains durable intent across retries and competing observations."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def submission(monkeypatch):
    """Replace Delta IO with shared durable rows while exercising the submission protocol."""
    from skyulf.integrations.databricks import retraining_requests as module

    rows = {}

    class Store:
        """Model atomic job claims and separate immutable request evidence."""

        def __init__(self, spark, namespace):
            """Share rows across independent invocations, like the Delta table."""

        def read(self, key):
            """Return an isolated snapshot of the persisted record."""
            return dict(rows[key]) if key in rows else None

        def claim(self, row, cutoff_ms):
            """Allow one unresolved job intent or a new request after cooldown."""
            key = f"job:{row['job_id']}"
            old = rows.get(key)
            if old is None or (
                old["request_id"] != row["request_id"]
                and old["status"] == "submitted"
                and old["submitted_at_ms"] <= cutoff_ms
            ):
                rows[key] = row | {"row_id": key}
            return dict(rows[key])

        def record(self, row):
            """Retain the original observation even if a retry sees later evidence."""
            rows.setdefault(row["row_id"], dict(row))

        def complete(self, row):
            """Save the accepted run before releasing its gate into cooldown."""
            rows[row["row_id"]] = dict(row)
            key = f"job:{row['job_id']}"
            if rows[key]["request_id"] == row["request_id"]:
                rows[key] = row | {"row_id": key}

    monkeypatch.setattr(module, "_RequestStore", Store)
    workspace = Mock()
    workspace.jobs.get.return_value.settings.max_concurrent_runs = 1
    workspace.jobs.list_runs.return_value = []
    workspace.jobs.run_now.return_value = SimpleNamespace(response=SimpleNamespace(run_id=42))
    arguments = {
        "namespace": "ops.monitoring",
        "job_id": 17,
        "request_id": "a" * 64,
        "evidence": {"model_version": "7", "prediction_commit": 3},
        "cooldown_hours": 24,
        "now": datetime(2026, 10, 1, tzinfo=UTC),
    }
    return module, workspace, arguments, rows, Store


def test_submission_persists_intent_then_submits_with_queue_and_without_wait(submission):
    """Jobs must not start until both the job gate and auditable intent exist."""
    module, workspace, arguments, rows, _ = submission

    def run_now(**kwargs):
        """Inspect durable state at the irreversible API boundary."""
        assert len(rows) == 2
        assert all(row["status"] == "intent" for row in rows.values())
        assert kwargs["job_parameters"] == {"lifecycle_action": "train"}
        assert kwargs["idempotency_token"] == arguments["request_id"]
        assert kwargs["queue"].enabled is True
        return SimpleNamespace(response=SimpleNamespace(run_id=42))

    workspace.jobs.run_now.side_effect = run_now
    result = module.submit_retraining(Mock(), workspace, **arguments)
    assert result["status"] == "submitted" and result["run_id"] == 42
    assert all(row["status"] == "submitted" for row in rows.values())
    assert all(row["run_id"] == 42 for row in rows.values())


def test_completed_request_never_submits_twice_even_after_cooldown(submission):
    """Replaying old model and data evidence cannot retrain again on a later schedule."""
    module, workspace, arguments, rows, _ = submission
    module.submit_retraining(Mock(), workspace, **arguments)
    result = module.submit_retraining(
        Mock(), workspace, **(arguments | {"now": arguments["now"] + timedelta(days=3)})
    )
    assert result["status"] == "already_submitted" and result["run_id"] == 42
    assert workspace.jobs.run_now.call_count == 1
    assert len(rows) == 2


def test_preview_does_not_claim_or_submit_and_submission_rechecks_active_run(submission):
    """The visible true branch is advisory and cannot bypass a later active training run."""
    module, workspace, arguments, rows, _ = submission
    result = module.submit_retraining(Mock(), workspace, **arguments, preview_only=True)
    assert result["status"] == "ready"
    assert rows == {}
    workspace.jobs.run_now.assert_not_called()
    workspace.jobs.list_runs.return_value = [SimpleNamespace(run_id=8)]
    result = module.submit_retraining(Mock(), workspace, **arguments)
    assert result["status"] == "active_run"
    assert rows == {}
    workspace.jobs.run_now.assert_not_called()


@pytest.mark.parametrize("case", ["already_submitted", "cooldown", "active_run", "pending_request"])
def test_preview_exposes_blocking_reason_without_mutating_receipts(submission, case):
    """False branches must reflect durable request guards without claiming new work."""
    module, workspace, arguments, rows, _ = submission
    if case == "active_run":
        workspace.jobs.list_runs.return_value = [SimpleNamespace(run_id=8)]
    elif case == "pending_request":
        workspace.jobs.run_now.side_effect = TimeoutError("response lost")
        with pytest.raises(TimeoutError):
            module.submit_retraining(Mock(), workspace, **arguments)
        arguments = arguments | {"request_id": "b" * 64}
    else:
        module.submit_retraining(Mock(), workspace, **arguments)
        if case == "cooldown":
            arguments = arguments | {"request_id": "b" * 64}
    saved = {key: dict(row) for key, row in rows.items()}
    workspace.jobs.run_now.reset_mock()
    result = module.submit_retraining(Mock(), workspace, **arguments, preview_only=True)
    assert result["status"] == case
    assert rows == saved
    workspace.jobs.run_now.assert_not_called()


def test_ambiguous_api_error_keeps_intent_and_retries_same_token(submission):
    """A lost API response must not authorize a different request or duplicate a run."""
    module, workspace, arguments, rows, _ = submission
    workspace.jobs.run_now.side_effect = TimeoutError("response lost")
    with pytest.raises(TimeoutError):
        module.submit_retraining(Mock(), workspace, **arguments)
    assert all(row["status"] == "intent" for row in rows.values())
    blocked = module.submit_retraining(Mock(), workspace, **(arguments | {"request_id": "b" * 64}))
    assert blocked["status"] == "pending_request"
    workspace.jobs.list_runs.return_value = [SimpleNamespace(run_id=42)]
    workspace.jobs.run_now.side_effect = None
    result = module.submit_retraining(
        Mock(), workspace, **(arguments | {"evidence": {"later_report": True}})
    )
    assert result["status"] == "submitted"
    assert workspace.jobs.run_now.call_count == 2
    assert all('"model_version": "7"' in row["evidence_json"] for row in rows.values())


def test_job_cooldown_blocks_distinct_request_then_allows_new_data(submission):
    """Cooldown belongs to the training job across all monitored components."""
    module, workspace, arguments, rows, _ = submission
    module.submit_retraining(Mock(), workspace, **arguments)
    next_request = arguments | {"request_id": "b" * 64}
    result = module.submit_retraining(Mock(), workspace, **next_request)
    assert result["status"] == "cooldown"
    result = module.submit_retraining(
        Mock(), workspace, **(next_request | {"now": arguments["now"] + timedelta(hours=24)})
    )
    assert result["status"] == "submitted"
    assert workspace.jobs.run_now.call_count == 2
    assert len(rows) == 3


def test_active_training_run_blocks_new_request_without_intent(submission):
    """A scheduled or manually started run must suppress automatic retraining."""
    module, workspace, arguments, rows, _ = submission
    workspace.jobs.list_runs.return_value = [SimpleNamespace(run_id=8)]
    result = module.submit_retraining(Mock(), workspace, **arguments)
    assert result["status"] == "active_run"
    assert rows == {}
    workspace.jobs.run_now.assert_not_called()


def test_manual_start_race_retains_one_queued_request_and_blocks_later_data(submission):
    """An intervening manual run must queue this token instead of consuming it as skipped."""
    module, workspace, arguments, rows, _ = submission

    def queued_submission(**kwargs):
        """Simulate the manual run winning just after the initial active-run query."""
        assert kwargs["queue"].enabled is True
        workspace.jobs.list_runs.return_value = [
            SimpleNamespace(run_id=42, state=SimpleNamespace(life_cycle_state="QUEUED"))
        ]
        return SimpleNamespace(response=SimpleNamespace(run_id=42))

    workspace.jobs.run_now.side_effect = queued_submission
    first = module.submit_retraining(Mock(), workspace, **arguments)
    later = module.submit_retraining(
        Mock(),
        workspace,
        **(arguments | {"request_id": "b" * 64, "now": arguments["now"] + timedelta(days=2)}),
    )
    assert first["status"] == "submitted" and first["run_id"] == 42
    assert later["status"] == "active_run"
    assert workspace.jobs.run_now.call_count == 1
    assert len(rows) == 2


def test_competing_request_claim_cannot_submit(submission, monkeypatch):
    """A concurrent winner between the initial read and CAS must prevent API submission."""
    module, workspace, arguments, rows, store = submission
    claim = store.claim

    def competing_claim(self, row, cutoff_ms):
        """Commit a different observer's intent immediately before this claim."""
        claim(self, row | {"request_id": "b" * 64}, cutoff_ms)
        return claim(self, row, cutoff_ms)

    monkeypatch.setattr(store, "claim", competing_claim)
    result = module.submit_retraining(Mock(), workspace, **arguments)
    assert result["status"] == "pending_request"
    assert rows["job:17"]["request_id"] == "b" * 64
    workspace.jobs.run_now.assert_not_called()


def test_retry_repairs_gate_after_saved_receipt_before_gate_update(submission, monkeypatch):
    """A partial Delta failure must not leave a completed request blocking its job forever."""
    module, workspace, arguments, rows, store = submission
    complete = store.complete

    def partial_complete(self, row):
        """Persist the request receipt then simulate a failed gate update."""
        rows[row["row_id"]] = dict(row)
        raise OSError("Delta connection lost")

    monkeypatch.setattr(store, "complete", partial_complete)
    with pytest.raises(OSError):
        module.submit_retraining(Mock(), workspace, **arguments)
    assert rows["job:17"]["status"] == "intent"
    monkeypatch.setattr(store, "complete", complete)
    result = module.submit_retraining(Mock(), workspace, **arguments)
    assert result["status"] == "already_submitted"
    assert rows["job:17"]["status"] == "submitted"
    assert workspace.jobs.run_now.call_count == 1


def test_missing_run_id_keeps_submission_pending(submission):
    """An incomplete API response cannot release the job gate or fabricate a run receipt."""
    module, workspace, arguments, rows, _ = submission
    workspace.jobs.run_now.return_value.response.run_id = None
    with pytest.raises(ValueError, match="run_id"):
        module.submit_retraining(Mock(), workspace, **arguments)
    assert all(row["status"] == "intent" for row in rows.values())


@pytest.mark.parametrize("concurrency", [None, 0, 2])
def test_job_must_disable_concurrent_runs_before_intent(submission, concurrency):
    """The platform concurrency contract is required even when the local gate is clear."""
    module, workspace, arguments, rows, _ = submission
    workspace.jobs.get.return_value.settings.max_concurrent_runs = concurrency
    with pytest.raises(ValueError, match="max_concurrent_runs"):
        module.submit_retraining(Mock(), workspace, **arguments)
    assert rows == {}
    workspace.jobs.run_now.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"job_id": True},
        {"job_id": 0},
        {"job_id": 2**63},
        {"request_id": "unsafe'"},
        {"cooldown_hours": -1},
        {"cooldown_hours": float("nan")},
        {"cooldown_hours": 1e308},
        {"now": datetime(2026, 10, 1)},
        {"evidence": {"bad": float("nan")}},
    ],
)
def test_invalid_submission_fails_before_cloud_access(submission, changes):
    """Only finite bounded identities and aware UTC timestamps may reach persistence."""
    module, workspace, arguments, rows, _ = submission
    with pytest.raises(ValueError):
        module.submit_retraining(Mock(), workspace, **(arguments | changes))
    assert rows == {}
    workspace.jobs.get.assert_not_called()


def test_store_rejects_foreign_table_before_writes():
    """An unrelated table cannot be adopted as the durable retraining gate."""
    from skyulf.integrations.databricks.retraining_requests import _RequestStore

    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.first.return_value = {"value": "foreign"}
    with pytest.raises(ValueError, match="ownership"):
        _RequestStore(spark, "ops.monitoring")
    assert all(call.args[0].startswith("SHOW TBLPROPERTIES") for call in spark.sql.call_args_list)


def test_store_requires_serializable_isolation():
    """Snapshot isolation alone cannot serialize two first claims for the same job."""
    from skyulf.integrations.databricks.monitoring_store import OWNER
    from skyulf.integrations.databricks.retraining_requests import _RequestStore

    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.first.side_effect = [
        {"value": OWNER},
        {"value": OWNER},
        {"value": "WriteSerializable"},
    ]
    with pytest.raises(ValueError, match="Serializable"):
        _RequestStore(spark, "ops.monitoring")
    spark.createDataFrame.assert_not_called()


def test_store_claim_sql_cannot_replace_an_unresolved_intent(monkeypatch):
    """The job-wide conflict predicate must be enforced inside the Delta transaction."""
    from skyulf.integrations.databricks import retraining_requests as module

    store = object.__new__(module._RequestStore)
    store.spark = Mock()
    store.name = "ops.monitoring.retraining_requests"
    row = {"row_id": "request:17:" + "a" * 64, "job_id": 17, "request_id": "a" * 64}
    monkeypatch.setattr(store, "read", lambda key: row | {"row_id": key})
    statements = []
    monkeypatch.setattr(module, "merge_with_retry", lambda spark, sql: statements.append(sql))
    result = store.claim(row, 123)
    assert result["row_id"] == "job:17"
    assert "ON t.row_id = s.row_id" in statements[0]
    assert "t.request_id <> s.request_id AND t.status = 'submitted'" in statements[0]
    assert "t.submitted_at_ms <= 123" in statements[0]
    assert "WHEN NOT MATCHED THEN INSERT *" in statements[0]
    store.spark.catalog.dropTempView.assert_called_once()
