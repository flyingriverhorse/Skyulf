"""Status management utilities for hyperparameter tuning jobs."""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.models import HyperparameterTuningJob
from core.feature_engineering.schemas import HyperparameterTuningJobStatus
from core.utils.datetime import utcnow

logger = logging.getLogger(__name__)


def _set_job_attribute(job: HyperparameterTuningJob, attr: str, value: Any) -> None:
    """Assign a value to a SQLAlchemy model attribute without mypy issues."""

    setattr(job, attr, value)


async def update_tuning_job_status(
    session: AsyncSession,
    job: HyperparameterTuningJob,
    *,
    status: HyperparameterTuningJobStatus,
    metrics: Optional[dict] = None,
    results: Optional[List[dict]] = None,
    best_params: Optional[dict] = None,
    best_score: Optional[float] = None,
    artifact_uri: Optional[str] = None,
    error_message: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> HyperparameterTuningJob:
    """Persist status transitions for a tuning job."""

    now = utcnow()
    _set_job_attribute(job, "status", status.value)

    if status == HyperparameterTuningJobStatus.RUNNING:
        _set_job_attribute(job, "started_at", now)
    if status in {
        HyperparameterTuningJobStatus.SUCCEEDED,
        HyperparameterTuningJobStatus.FAILED,
        HyperparameterTuningJobStatus.CANCELLED,
    }:
        _set_job_attribute(job, "finished_at", now)

    if metrics is not None:
        _set_job_attribute(job, "metrics", metrics)
    if results is not None:
        _set_job_attribute(job, "results", results)
    if best_params is not None:
        _set_job_attribute(job, "best_params", best_params)
    if best_score is not None:
        _set_job_attribute(job, "best_score", best_score)
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


async def bulk_mark_tuning_cancelled(
    session: AsyncSession,
    job_ids: Iterable[str],
) -> None:
    """Convenience helper to mark tuning jobs as cancelled."""

    if not job_ids:
        return

    stmt = select(HyperparameterTuningJob).where(HyperparameterTuningJob.id.in_(list(job_ids)))
    results = await session.execute(stmt)
    jobs = list(results.scalars().all())

    for job in jobs:
        await update_tuning_job_status(session, job, status=HyperparameterTuningJobStatus.CANCELLED)

    logger.info("Cancelled %s tuning job(s)", len(jobs))


__all__ = [
    "_set_job_attribute",
    "update_tuning_job_status",
    "bulk_mark_tuning_cancelled",
]