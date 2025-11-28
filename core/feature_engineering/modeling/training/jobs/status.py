"""Status management helpers for training jobs."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import core.database.engine as db_engine
from core.database.models import TrainingJob
from core.feature_engineering.schemas import TrainingJobStatus
from core.utils.datetime import utcnow


def _set_job_attribute(job: TrainingJob, attr: str, value: Any) -> None:
    setattr(job, attr, value)


def update_job_progress_sync(job_id: str, progress: int, current_step: Optional[str] = None) -> None:
    """Synchronously update job progress (for use in blocking tasks)."""
    if not db_engine.sync_session_factory:
        # This can happen if init_db() hasn't been called in this process
        import sys
        sys.stderr.write(f"⚠️ update_job_progress_sync: sync_session_factory is None for job {job_id}\n")
        return

    try:
        with db_engine.sync_session_factory() as session:
            job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                job.progress = progress
                if current_step:
                    job.current_step = current_step
                session.commit()
            else:
                import sys
                sys.stderr.write(f"⚠️ update_job_progress_sync: Job {job_id} not found\n")
    except Exception as e:
        # Fail silently to avoid crashing the training job, but log to stderr
        import sys
        sys.stderr.write(f"❌ update_job_progress_sync error: {e}\n")
        pass


async def update_job_status(
    session: AsyncSession,
    job: TrainingJob,
    *,
    status: TrainingJobStatus,
    metrics: Optional[dict] = None,
    artifact_uri: Optional[str] = None,
    error_message: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> TrainingJob:
    now = utcnow()
    _set_job_attribute(job, "status", status.value)

    if status == TrainingJobStatus.RUNNING:
        _set_job_attribute(job, "started_at", now)
    if status in {TrainingJobStatus.SUCCEEDED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED}:
        _set_job_attribute(job, "finished_at", now)

    if metrics is not None:
        _set_job_attribute(job, "metrics", metrics)
    if artifact_uri is not None:
        _set_job_attribute(job, "artifact_uri", artifact_uri)
    if error_message is not None:
        _set_job_attribute(job, "error_message", error_message)
    if metadata is not None:
        merged = dict(job.job_metadata or {})
        merged.update(metadata)
        _set_job_attribute(job, "job_metadata", merged)

    await session.commit()
    await session.refresh(job)
    return job


async def bulk_mark_cancelled(session: AsyncSession, job_ids: Iterable[str]) -> None:
    if not job_ids:
        return

    stmt = select(TrainingJob).where(TrainingJob.id.in_(list(job_ids)))
    results = await session.execute(stmt)
    jobs = list(results.scalars().all())

    for job in jobs:
        await update_job_status(session, job, status=TrainingJobStatus.CANCELLED)


__all__ = [
    "bulk_mark_cancelled",
    "update_job_status",
]
