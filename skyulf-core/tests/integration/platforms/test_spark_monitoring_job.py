"""Spark monitoring handoff schedules independent work without inline population reads."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.jobs.monitoring.spark_monitoring_job import (
    dispatch_monitoring_job,
)


def test_handoff_is_nonblocking_and_idempotent_for_same_scoring_receipt():
    """Score repair must reuse one independent monitoring job invocation."""
    workspace = Mock()
    workspace.jobs.run_now.return_value = SimpleNamespace(run_id=42)
    request = {
        "namespace": "a.monitoring",
        "commit_version": 7,
        "noop": False,
        "config": {"model_name": "a.b.model", "model_version": "2"},
    }
    values = {"monitoring_job_id": "123"}
    first = dispatch_monitoring_job(request, values, workspace=workspace)
    token = workspace.jobs.run_now.call_args.kwargs["idempotency_token"]
    second = dispatch_monitoring_job(request, values, workspace=workspace)
    assert first == second == {"status": "queued", "job_id": 123, "run_id": 42}
    assert workspace.jobs.run_now.call_args.kwargs["idempotency_token"] == token
    assert len(token) == 64
    assert "monitoring_request" in workspace.jobs.run_now.call_args.kwargs["job_parameters"]


def test_missing_monitoring_job_fails_before_submission():
    """An invalid deployment cannot silently fall back to local monitoring."""
    workspace = Mock()
    with pytest.raises(ValueError, match="monitoring_job_id"):
        dispatch_monitoring_job({}, {}, workspace=workspace)
    workspace.jobs.run_now.assert_not_called()


def test_databricks_epoch_start_time_keeps_utc_cutoff():
    """Databricks iso_datetime omits timezone; the job must use its UTC epoch timestamp."""
    from skyulf.integrations.databricks.jobs.monitoring.spark_monitoring_job import (
        _observation_time,
    )

    expected = datetime(2026, 10, 5, 6, 0, tzinfo=UTC)
    result = _observation_time({"as_of_unix_ms": str(int(expected.timestamp() * 1000))})
    assert result == expected and result.tzinfo is UTC


def test_distinct_noop_scoring_runs_can_revisit_late_labels():
    """Idempotency must deduplicate repairs without suppressing a later scoring invocation."""
    workspace = Mock()
    workspace.jobs.run_now.return_value = SimpleNamespace(run_id=42)
    request = {"noop": True, "commit_version": None}
    values = {"monitoring_job_id": "123", "monitoring_invocation_id": "100"}
    dispatch_monitoring_job(request, values, workspace=workspace)
    first = workspace.jobs.run_now.call_args.kwargs["idempotency_token"]
    dispatch_monitoring_job(request, values, workspace=workspace)
    assert workspace.jobs.run_now.call_args.kwargs["idempotency_token"] == first
    dispatch_monitoring_job(
        request, values | {"monitoring_invocation_id": "101"}, workspace=workspace
    )
    assert workspace.jobs.run_now.call_args.kwargs["idempotency_token"] != first
