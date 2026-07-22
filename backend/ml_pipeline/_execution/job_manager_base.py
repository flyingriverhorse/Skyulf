"""Shared base for BasicTrainingManager and AdvancedTuningManager.

Extracts the cancel-job, log-append, cancelled-guard, and status-update
skeleton so the logic lives in exactly one place.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.ml_pipeline._execution.schemas import JobStatus


class TrainingJobManagerBase:
    """Mixin base providing shared job-management helpers.

    Concrete managers inherit this class and delegate the duplicated
    cancel/log/status-update flows to these methods.
    """

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_job_logs(job: Any, logs: list[str]) -> None:
        """Append *logs* to ``job.logs`` without clobbering existing entries."""
        current_logs: list[str] = job.logs or []
        job.logs = current_logs + logs

    # ------------------------------------------------------------------
    # Cancelled-state guard
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_cancelled_status_update(session: Session, job: Any, logs: list[str] | None) -> bool:
        """Append logs to a cancelled job; never revive its status.

        Returns True so callers can propagate the result unchanged.
        """
        if logs:
            TrainingJobManagerBase._append_job_logs(job, logs)
            session.commit()
        return True

    # ------------------------------------------------------------------
    # Celery revoke
    # ------------------------------------------------------------------

    @staticmethod
    def _revoke_celery_task(job_metadata: dict[str, Any]) -> None:
        """Best-effort revoke of the Celery task recorded in *job_metadata*.

        Silences any exception so a broker/network failure never blocks the
        user-visible cancel response.
        """
        task_id = job_metadata.get("celery_task_id")
        if task_id:
            try:
                from backend.celery_app import celery_app  # noqa: PLC0415

                celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
            except Exception:
                # Best-effort: never let revoke errors block the user-visible
                # cancel.  The status guard in update_status_sync still keeps
                # the row at CANCELLED even if the worker writes back.
                pass  # nosec B110

    # ------------------------------------------------------------------
    # Shared cancel
    # ------------------------------------------------------------------

    @staticmethod
    async def _cancel_job(
        session: AsyncSession, model: type[Any], job_id: str, run_mode: str | None = None
    ) -> bool:
        """Cancel a QUEUED/RUNNING job row and revoke its Celery task.

        `run_mode`, when given, scopes the lookup to that mode so a caller
        scoped to "fixed" jobs (e.g. BasicTrainingManager) never cancels a
        "tuned" row that happens to share the same underlying table.

        Returns True if the job was found and cancelled, False otherwise.
        """
        stmt = select(model).where(model.id == job_id)
        if run_mode is not None:
            stmt = stmt.where(model.run_mode == run_mode)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if job and job.status in [JobStatus.QUEUED.value, JobStatus.RUNNING.value]:
            job.status = JobStatus.CANCELLED.value
            job.error_message = "Job cancelled by user."
            job.finished_at = datetime.now(UTC)
            meta = (job.job_metadata or {}) if isinstance(job.job_metadata, dict) else {}
            TrainingJobManagerBase._revoke_celery_task(meta)
            await session.commit()
            return True
        return False

    # ------------------------------------------------------------------
    # Shared status-update skeleton
    # ------------------------------------------------------------------

    @staticmethod
    def _update_status_sync(
        session: Session,
        model: type[Any],
        job_id: str,
        status: JobStatus | None,
        error: str | None,
        result: dict[str, Any] | None,
        logs: list[str] | None,
        apply_fields_fn: Callable[
            [Any, JobStatus | None, str | None, list[str] | None, dict[str, Any] | None],
            None,
        ],
        run_mode: str | None = None,
    ) -> bool:
        """Update a job row; guard against overwriting a CANCELLED status.

        *apply_fields_fn* is the model-specific hook that writes
        status/error/logs/result onto the concrete job row. `run_mode`, when
        given, scopes the lookup to that mode (see `_cancel_job` for why this
        matters now that both modes share one table). Returns True if the
        job was found, False otherwise.
        """
        query = session.query(model).filter(model.id == job_id)
        if run_mode is not None:
            query = query.filter(model.run_mode == run_mode)
        job = query.first()
        if not job:
            return False

        # Guard: once a job is CANCELLED by the user the worker may still be
        # mid-fit and try to flip it back to RUNNING/COMPLETED.  Refuse those
        # overwrites so the user-visible state stays accurate; logs are still
        # appended so cancellation traces are preserved for debugging.
        if job.status == JobStatus.CANCELLED.value:
            return TrainingJobManagerBase._handle_cancelled_status_update(session, job, logs)

        apply_fields_fn(job, status, error, logs, result)
        session.commit()
        return True
