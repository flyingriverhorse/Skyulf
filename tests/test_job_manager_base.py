"""Unit tests for TrainingJobManagerBase shared helpers and both concrete managers.

Tests are deliberately free of any database: all SQLAlchemy objects are
replaced by lightweight MagicMock / SimpleNamespace instances.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.ml_pipeline._execution.advanced_tuning_manager import AdvancedTuningManager
from backend.ml_pipeline._execution.basic_training_manager import BasicTrainingManager
from backend.ml_pipeline._execution.job_manager_base import TrainingJobManagerBase
from backend.ml_pipeline._execution.schemas import JobStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(status: str, logs: list[str] | None = None, task_id: str | None = None):
    """Return a simple namespace that mimics a SQLAlchemy job row."""
    meta = {"celery_task_id": task_id} if task_id else {}
    job = SimpleNamespace(
        id="job-123",
        status=status,
        error_message=None,
        finished_at=None,
        logs=list(logs) if logs else None,
        job_metadata=meta,
    )
    return job


def _make_sync_session(job=None):
    """Return a mock sync Session whose .query().filter()...first() returns *job*.

    `_update_status_sync` may chain a second `.filter()` call for the
    `run_mode` scoping (see `job_manager_base.py`), so both the single- and
    double-filter chains are wired to resolve to the same *job*.
    """
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = job
    session.query.return_value.filter.return_value.filter.return_value.first.return_value = job
    return session


def _make_async_session(job=None):
    """Return a mock AsyncSession whose .execute() scalar resolves to *job*."""
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = job
    execute_result = AsyncMock(return_value=scalar_result)
    session = MagicMock()
    session.execute = execute_result
    session.commit = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# TrainingJobManagerBase direct tests
# ---------------------------------------------------------------------------


class TestAppendJobLogs(unittest.TestCase):
    """_append_job_logs appends without clobbering existing logs."""

    def test_appends_to_existing_logs(self):
        job = SimpleNamespace(logs=["line1", "line2"])
        TrainingJobManagerBase._append_job_logs(job, ["line3"])
        self.assertEqual(job.logs, ["line1", "line2", "line3"])

    def test_appends_when_logs_is_none(self):
        job = SimpleNamespace(logs=None)
        TrainingJobManagerBase._append_job_logs(job, ["first"])
        self.assertEqual(job.logs, ["first"])

    def test_appends_empty_list_is_noop(self):
        job = SimpleNamespace(logs=["existing"])
        TrainingJobManagerBase._append_job_logs(job, [])
        self.assertEqual(job.logs, ["existing"])

    def test_does_not_mutate_original_list(self):
        original = ["a", "b"]
        job = SimpleNamespace(logs=original)
        TrainingJobManagerBase._append_job_logs(job, ["c"])
        # A new list is assigned; the original list object is unchanged
        self.assertEqual(original, ["a", "b"])
        self.assertEqual(job.logs, ["a", "b", "c"])


class TestHandleCancelledStatusUpdate(unittest.TestCase):
    """_handle_cancelled_status_update appends logs and returns True."""

    def test_appends_logs_and_commits(self):
        job = SimpleNamespace(logs=["prev"])
        session = MagicMock()
        result = TrainingJobManagerBase._handle_cancelled_status_update(session, job, ["new"])
        self.assertTrue(result)
        session.commit.assert_called_once()
        self.assertEqual(job.logs, ["prev", "new"])

    def test_no_commit_when_no_logs(self):
        job = SimpleNamespace(logs=None)
        session = MagicMock()
        result = TrainingJobManagerBase._handle_cancelled_status_update(session, job, None)
        self.assertTrue(result)
        session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# cancel_training_job
# ---------------------------------------------------------------------------


class TestCancelTrainingJob(unittest.IsolatedAsyncioTestCase):
    """BasicTrainingManager.cancel_training_job — cancels QUEUED/RUNNING jobs."""

    async def _run_cancel(self, job):
        session = _make_async_session(job)
        mock_stmt = MagicMock()
        with patch("backend.ml_pipeline._execution.job_manager_base.select", return_value=mock_stmt):
            result = await BasicTrainingManager.cancel_training_job(session, "job-123")
        return result, session

    async def test_cancels_queued_job(self):
        job = _make_job(JobStatus.QUEUED.value)
        result, session = await self._run_cancel(job)
        self.assertTrue(result)
        self.assertEqual(job.status, JobStatus.CANCELLED.value)
        self.assertEqual(job.error_message, "Job cancelled by user.")
        self.assertIsNotNone(job.finished_at)
        session.commit.assert_called_once()

    async def test_cancels_running_job(self):
        job = _make_job(JobStatus.RUNNING.value)
        result, _ = await self._run_cancel(job)
        self.assertTrue(result)
        self.assertEqual(job.status, JobStatus.CANCELLED.value)

    async def test_returns_false_for_already_cancelled(self):
        job = _make_job(JobStatus.CANCELLED.value)
        result, session = await self._run_cancel(job)
        self.assertFalse(result)
        session.commit.assert_not_called()

    async def test_returns_false_for_completed_job(self):
        job = _make_job(JobStatus.COMPLETED.value)
        result, _ = await self._run_cancel(job)
        self.assertFalse(result)

    async def test_returns_false_when_job_not_found(self):
        session = _make_async_session(None)
        mock_stmt = MagicMock()
        with patch("backend.ml_pipeline._execution.job_manager_base.select", return_value=mock_stmt):
            result = await BasicTrainingManager.cancel_training_job(session, "missing-id")
        self.assertFalse(result)

    async def test_revokes_celery_task(self):
        job = _make_job(JobStatus.RUNNING.value, task_id="celery-abc")
        session = _make_async_session(job)
        mock_celery = MagicMock()
        mock_stmt = MagicMock()
        with patch("backend.ml_pipeline._execution.job_manager_base.select", return_value=mock_stmt), \
             patch("backend.celery_app.celery_app", mock_celery):
            await BasicTrainingManager.cancel_training_job(session, "job-123")
        mock_celery.control.revoke.assert_called_once_with(
            "celery-abc", terminate=True, signal="SIGTERM"
        )

    async def test_celery_revoke_failure_does_not_raise(self):
        """A broken Celery broker must not prevent the cancel from completing."""
        job = _make_job(JobStatus.RUNNING.value, task_id="celery-abc")
        session = _make_async_session(job)
        mock_celery = MagicMock()
        mock_celery.control.revoke.side_effect = RuntimeError("broker down")
        mock_stmt = MagicMock()
        with patch("backend.ml_pipeline._execution.job_manager_base.select", return_value=mock_stmt), \
             patch("backend.celery_app.celery_app", mock_celery):
            result = await BasicTrainingManager.cancel_training_job(session, "job-123")
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# cancel_tuning_job
# ---------------------------------------------------------------------------


class TestCancelTuningJob(unittest.IsolatedAsyncioTestCase):
    """AdvancedTuningManager.cancel_tuning_job mirrors the training cancel."""

    async def _run_cancel(self, job):
        session = _make_async_session(job)
        mock_stmt = MagicMock()
        with patch("backend.ml_pipeline._execution.job_manager_base.select", return_value=mock_stmt):
            result = await AdvancedTuningManager.cancel_tuning_job(session, "job-123")
        return result, session

    async def test_cancels_queued_job(self):
        job = _make_job(JobStatus.QUEUED.value)
        result, _ = await self._run_cancel(job)
        self.assertTrue(result)
        self.assertEqual(job.status, JobStatus.CANCELLED.value)

    async def test_cancels_running_job(self):
        job = _make_job(JobStatus.RUNNING.value)
        result, _ = await self._run_cancel(job)
        self.assertTrue(result)

    async def test_returns_false_when_already_cancelled(self):
        job = _make_job(JobStatus.CANCELLED.value)
        result, _ = await self._run_cancel(job)
        self.assertFalse(result)

    async def test_returns_false_when_job_not_found(self):
        session = _make_async_session(None)
        mock_stmt = MagicMock()
        with patch("backend.ml_pipeline._execution.job_manager_base.select", return_value=mock_stmt):
            result = await AdvancedTuningManager.cancel_tuning_job(session, "missing")
        self.assertFalse(result)

    async def test_revokes_celery_task(self):
        job = _make_job(JobStatus.RUNNING.value, task_id="celery-xyz")
        session = _make_async_session(job)
        mock_celery = MagicMock()
        mock_stmt = MagicMock()
        with patch(
            "backend.ml_pipeline._execution.job_manager_base.select", return_value=mock_stmt
        ), patch("backend.celery_app.celery_app", mock_celery):
            await AdvancedTuningManager.cancel_tuning_job(session, "job-123")
        mock_celery.control.revoke.assert_called_once_with(
            "celery-xyz", terminate=True, signal="SIGTERM"
        )

    async def test_celery_revoke_failure_does_not_raise(self):
        job = _make_job(JobStatus.RUNNING.value, task_id="celery-xyz")
        session = _make_async_session(job)
        mock_celery = MagicMock()
        mock_celery.control.revoke.side_effect = OSError("no broker")
        mock_stmt = MagicMock()
        with patch(
            "backend.ml_pipeline._execution.job_manager_base.select", return_value=mock_stmt
        ), patch("backend.celery_app.celery_app", mock_celery):
            result = await AdvancedTuningManager.cancel_tuning_job(session, "job-123")
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# update_status_sync — BasicTrainingManager
# ---------------------------------------------------------------------------


class TestBasicUpdateStatusSync(unittest.TestCase):
    """BasicTrainingManager.update_status_sync behaviour."""

    def test_returns_false_when_job_not_found(self):
        session = _make_sync_session(None)
        result = BasicTrainingManager.update_status_sync(session, "missing")
        self.assertFalse(result)

    def test_normal_update_sets_status(self):
        job = _make_job(JobStatus.RUNNING.value)
        # _apply_status_update_fields only reads .logs so add that attr too
        job.logs = []
        # patch _apply_status_update_fields to a simple spy
        applied: list = []

        def spy(j, s, e, lg, r):
            j.status = s.value if s else j.status
            applied.append((s, e, lg, r))

        session = _make_sync_session(job)
        with patch.object(BasicTrainingManager, "_apply_status_update_fields", side_effect=spy):
            result = BasicTrainingManager.update_status_sync(
                session, "job-123", status=JobStatus.COMPLETED
            )
        self.assertTrue(result)
        session.commit.assert_called_once()
        self.assertEqual(applied[0][0], JobStatus.COMPLETED)

    def test_cancelled_job_only_appends_logs(self):
        """Regression guard: a CANCELLED job must not have its status revived."""
        job = _make_job(JobStatus.CANCELLED.value, logs=["prev"])
        session = _make_sync_session(job)
        result = BasicTrainingManager.update_status_sync(
            session, "job-123", status=JobStatus.COMPLETED, logs=["worker-log"]
        )
        self.assertTrue(result)
        # Status must remain CANCELLED
        self.assertEqual(job.status, JobStatus.CANCELLED.value)
        # Log was still appended
        self.assertIn("worker-log", job.logs)
        session.commit.assert_called_once()

    def test_cancelled_job_no_logs_no_commit(self):
        job = _make_job(JobStatus.CANCELLED.value)
        session = _make_sync_session(job)
        BasicTrainingManager.update_status_sync(session, "job-123", logs=None)
        session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# update_status_sync — AdvancedTuningManager
# ---------------------------------------------------------------------------


class TestAdvancedUpdateStatusSync(unittest.TestCase):
    """AdvancedTuningManager.update_status_sync mirrors BasicTrainingManager."""

    def test_returns_false_when_job_not_found(self):
        session = _make_sync_session(None)
        result = AdvancedTuningManager.update_status_sync(session, "missing")
        self.assertFalse(result)

    def test_normal_update_commits(self):
        job = _make_job(JobStatus.RUNNING.value)
        job.logs = []

        def spy(j, s, e, lg, r):
            j.status = s.value if s else j.status

        session = _make_sync_session(job)
        with patch.object(AdvancedTuningManager, "_apply_status_update_fields", side_effect=spy):
            result = AdvancedTuningManager.update_status_sync(
                session, "job-123", status=JobStatus.COMPLETED
            )
        self.assertTrue(result)
        session.commit.assert_called_once()

    def test_cancelled_job_only_appends_logs(self):
        """Regression guard: a CANCELLED tuning job must not have its status revived."""
        job = _make_job(JobStatus.CANCELLED.value, logs=["existing"])
        session = _make_sync_session(job)
        result = AdvancedTuningManager.update_status_sync(
            session, "job-123", status=JobStatus.COMPLETED, logs=["late-write"]
        )
        self.assertTrue(result)
        self.assertEqual(job.status, JobStatus.CANCELLED.value)
        self.assertIn("late-write", job.logs)


if __name__ == "__main__":
    unittest.main()
