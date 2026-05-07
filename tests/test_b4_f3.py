"""
Tests for:
  B4 — run_pipeline_batch_task (one Celery task per pipeline)
  F3 — Sentry span no-op when sentry-sdk is absent
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session():
    """Return a mock SQLAlchemy session that is closeable."""
    s = MagicMock()
    s.close = MagicMock()
    return s


# ---------------------------------------------------------------------------
# B4 — run_pipeline_batch_task
# ---------------------------------------------------------------------------


class TestRunPipelineBatchTask:
    """Unit tests for the batch Celery task (B4)."""

    def test_single_branch_calls_execute_once(self):
        """A single-branch batch delegates straight to execute_pipeline."""
        from backend.ml_pipeline.tasks import run_pipeline_batch_task

        session = _make_session()
        with (
            patch("backend.ml_pipeline.tasks.get_db_session", return_value=session),
            patch("backend.ml_pipeline.tasks.execute_pipeline") as mock_exec,
        ):
            run_pipeline_batch_task([("job-1", {"pipeline_id": "p1", "nodes": []})])

        mock_exec.assert_called_once_with("job-1", {"pipeline_id": "p1", "nodes": []}, session)
        session.close.assert_called_once()

    def test_multi_branch_calls_execute_for_each_branch(self):
        """Multi-branch batch calls execute_pipeline once per branch."""
        from backend.ml_pipeline.tasks import run_pipeline_batch_task

        branches = [
            ("job-a", {"pipeline_id": "pa", "nodes": []}),
            ("job-b", {"pipeline_id": "pb", "nodes": []}),
            ("job-c", {"pipeline_id": "pc", "nodes": []}),
        ]

        sessions = [_make_session() for _ in branches]
        session_iter = iter(sessions)

        def _next_session():
            return next(session_iter)

        with (
            patch("backend.ml_pipeline.tasks.get_db_session", side_effect=_next_session),
            patch("backend.ml_pipeline.tasks.execute_pipeline") as mock_exec,
        ):
            run_pipeline_batch_task(branches)

        assert mock_exec.call_count == 3
        called_job_ids = {c.args[0] for c in mock_exec.call_args_list}
        assert called_job_ids == {"job-a", "job-b", "job-c"}
        for s in sessions:
            s.close.assert_called_once()

    def test_session_closed_even_if_execute_raises(self):
        """DB session is always closed, even when execute_pipeline raises."""
        from backend.ml_pipeline.tasks import run_pipeline_batch_task

        session = _make_session()
        with (
            patch("backend.ml_pipeline.tasks.get_db_session", return_value=session),
            patch(
                "backend.ml_pipeline.tasks.execute_pipeline",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            run_pipeline_batch_task([("job-err", {"pipeline_id": "p", "nodes": []})])

        session.close.assert_called_once()

    def test_multi_branch_propagates_first_exception(self):
        """If any branch raises, the exception propagates out of the task."""
        from backend.ml_pipeline.tasks import run_pipeline_batch_task

        def _fail_on_b(job_id, config, session):
            if job_id == "job-b":
                raise ValueError("branch b failed")

        sessions = [_make_session(), _make_session()]
        session_iter = iter(sessions)

        with (
            patch(
                "backend.ml_pipeline.tasks.get_db_session", side_effect=lambda: next(session_iter)
            ),
            patch("backend.ml_pipeline.tasks.execute_pipeline", side_effect=_fail_on_b),
            pytest.raises(ValueError, match="branch b failed"),
        ):
            run_pipeline_batch_task(
                [
                    ("job-a", {"pipeline_id": "pa", "nodes": []}),
                    ("job-b", {"pipeline_id": "pb", "nodes": []}),
                ]
            )

    def test_run_pipeline_task_still_works(self):
        """Legacy single-job task still executes correctly (backward compat)."""
        from backend.ml_pipeline.tasks import run_pipeline_task

        session = _make_session()
        with (
            patch("backend.ml_pipeline.tasks.get_db_session", return_value=session),
            patch("backend.ml_pipeline.tasks.execute_pipeline") as mock_exec,
        ):
            run_pipeline_task("job-legacy", {"pipeline_id": "p0", "nodes": []})

        mock_exec.assert_called_once_with("job-legacy", {"pipeline_id": "p0", "nodes": []}, session)
        session.close.assert_called_once()


# ---------------------------------------------------------------------------
# F3 — Sentry span no-op / active
# ---------------------------------------------------------------------------


class TestPipelineSpan:
    """Unit tests for the _pipeline_span context manager (F3)."""

    def test_noop_when_sentry_absent(self):
        """_pipeline_span returns a nullcontext when sentry_sdk is not installed."""
        from backend.ml_pipeline.tasks import _pipeline_span

        # Ensure sentry_sdk is absent at call-time by hiding it in sys.modules.
        real = sys.modules.pop("sentry_sdk", None)
        try:
            with _pipeline_span("job-noop"):
                pass  # must not raise
        finally:
            if real is not None:
                sys.modules["sentry_sdk"] = real

    def test_span_active_when_sentry_present(self):
        """_pipeline_span calls start_transaction when sentry_sdk is available."""
        from backend.ml_pipeline.tasks import _pipeline_span

        fake_tx = MagicMock()
        fake_tx.__enter__ = MagicMock(return_value=fake_tx)
        fake_tx.__exit__ = MagicMock(return_value=False)
        fake_sentry = MagicMock()
        fake_sentry.start_transaction.return_value = fake_tx

        # Inject fake sentry_sdk at call-time (runtime import inside _pipeline_span).
        real = sys.modules.get("sentry_sdk")
        sys.modules["sentry_sdk"] = fake_sentry  # type: ignore[assignment]
        try:
            with _pipeline_span("job-traced"):
                pass
        finally:
            if real is not None:
                sys.modules["sentry_sdk"] = real
            else:
                sys.modules.pop("sentry_sdk", None)

        fake_sentry.start_transaction.assert_called_once_with(
            op="pipeline", name="run_pipeline/job-traced"
        )
        fake_tx.set_tag.assert_called_once_with("job_id", "job-traced")


# ---------------------------------------------------------------------------
# B4 — api.py uses batch task for the Celery path
# ---------------------------------------------------------------------------


class TestApiUsesB4:
    """Verify api.py routes Celery submissions through run_pipeline_batch_task."""

    def test_celery_path_uses_batch_task(self, client_with_celery):
        """When USE_CELERY=True, api.py calls run_pipeline_batch_task.delay once."""
        client, mock_delay = client_with_celery
        payload = {
            "pipeline_id": "batch-test",
            "nodes": [
                {
                    "node_id": "node_1",
                    "step_type": "data_loader",
                    "params": {"source_id": "dummy"},
                    "inputs": [],
                }
            ],
        }
        resp = client.post("/api/pipeline/run", json=payload)
        assert resp.status_code == 200
        mock_delay.assert_called_once()
        # The single delay call should carry a list of (job_id, payload) tuples
        batched_arg = mock_delay.call_args.args[0]
        assert isinstance(batched_arg, list)
        assert len(batched_arg) == 1
        job_id, branch_payload = batched_arg[0]
        assert isinstance(job_id, str)
        assert branch_payload["pipeline_id"] == "batch-test"


@pytest.fixture
def client_with_celery():
    """TestClient with USE_CELERY=True and run_pipeline_batch_task.delay mocked."""
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.config import get_settings

    mock_task_result = MagicMock()
    mock_task_result.id = "celery-task-id-123"
    mock_batch_task = MagicMock()
    mock_batch_task.delay = MagicMock(return_value=mock_task_result)

    settings = get_settings()

    with (
        patch.object(settings, "USE_CELERY", True),
        # Patch the names in the run_pipeline sub-router's namespace so the
        # route handler sees the mocks. (E9 phase 2 moved the handler out of
        # api.py into backend.ml_pipeline._internal._routers.run_pipeline.)
        patch(
            "backend.ml_pipeline._internal._routers.run_pipeline.run_pipeline_batch_task",
            mock_batch_task,
        ),
        patch(
            "backend.ml_pipeline._internal._routers.run_pipeline.JobManager.find_active_job",
            return_value=None,
        ),
        patch(
            "backend.ml_pipeline._internal._routers.run_pipeline.JobManager.create_job",
            return_value="test-job-id",
        ),
        patch(
            "backend.ml_pipeline._internal._routers.run_pipeline.JobManager.attach_celery_task_id"
        ),
        patch("backend.ml_pipeline._internal._routers.run_pipeline.publish_job_event"),
        patch(
            "backend.ml_pipeline._internal._routers.run_pipeline.resolve_pipeline_nodes",
            return_value=None,
        ),
    ):
        with TestClient(app, base_url="http://localhost") as c:
            yield c, mock_batch_task.delay
