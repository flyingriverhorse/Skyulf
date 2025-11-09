"""Utilities for creating and managing background model training jobs."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Iterable, List, Optional, Sequence, Union, cast

from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.engine import CursorResult

from core.database.models import TrainingJob
from core.feature_engineering.schemas import (
    TrainingJobCreate,
    TrainingJobStatus,
)

logger = logging.getLogger(__name__)


def _set_job_attribute(job: TrainingJob, attr: str, value: Any) -> None:
    """Assign a value to a SQLAlchemy model attribute without mypy complaints."""

    setattr(job, attr, value)


async def _resolve_next_version(
    session: AsyncSession,
    *,
    dataset_source_id: str,
    node_id: str,
    model_type: Optional[str] = None,
) -> int:
    """Return the next model version scoped to dataset/node/model-type."""

    filters = [
        TrainingJob.dataset_source_id == dataset_source_id,
        TrainingJob.node_id == node_id,
    ]

    if model_type:
        filters.append(TrainingJob.model_type == model_type)

    query: Select[tuple[int]] = select(func.max(TrainingJob.version)).where(*filters)
    result = await session.execute(query)
    current_max: Optional[int] = result.scalar()
    return (current_max or 0) + 1


async def create_training_job(
    session: AsyncSession,
    payload: TrainingJobCreate,
    *,
    user_id: Optional[int] = None,
    model_type_override: Optional[str] = None,
) -> TrainingJob:
    """Persist a new training job and compute its semantic version."""

    node_id_value = payload.node_id or payload.target_node_id
    if not node_id_value:
        raise ValueError("Training job payload missing node identifier")

    model_type_value = (model_type_override or payload.model_type or "").strip()
    if not model_type_value:
        raise ValueError("Training job payload missing model type")

    job_id = uuid.uuid4().hex
    version = await _resolve_next_version(
        session,
        dataset_source_id=payload.dataset_source_id,
        node_id=node_id_value,
        model_type=model_type_value,
    )

    job_metadata = dict(payload.metadata or {})
    if payload.target_node_id:
        job_metadata.setdefault("target_node_id", payload.target_node_id)

    job = TrainingJob(
        id=job_id,
        dataset_source_id=payload.dataset_source_id,
        pipeline_id=payload.pipeline_id,
        node_id=node_id_value,
        user_id=user_id,
        status=TrainingJobStatus.QUEUED.value,
        version=version,
        model_type=model_type_value,
        hyperparameters=payload.hyperparameters or {},
        job_metadata=job_metadata,
        graph=payload.graph.model_dump(),
    )

    session.add(job)
    await session.commit()
    await session.refresh(job)

    logger.info(
        "Enqueued training job %s (pipeline=%s node=%s version=%s)",
        job.id,
        job.pipeline_id,
        job.node_id,
        job.version,
    )

    return job


async def get_training_job(session: AsyncSession, job_id: str) -> Optional[TrainingJob]:
    """Fetch a single training job by primary key."""

    return await session.get(TrainingJob, job_id)


async def list_training_jobs(
    session: AsyncSession,
    *,
    user_id: Optional[int] = None,
    dataset_source_id: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    node_id: Optional[str] = None,
    limit: int = 20,
) -> List[TrainingJob]:
    """Return recent training jobs filtered by optional parameters."""

    stmt = select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit)

    if user_id is not None:
        stmt = stmt.where(TrainingJob.user_id == user_id)
    if dataset_source_id is not None:
        stmt = stmt.where(TrainingJob.dataset_source_id == dataset_source_id)
    if pipeline_id is not None:
        stmt = stmt.where(TrainingJob.pipeline_id == pipeline_id)
    if node_id is not None:
        stmt = stmt.where(TrainingJob.node_id == node_id)

    results = await session.execute(stmt)
    return list(results.scalars().all())


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
    """Persist status transitions for a training job."""

    now = datetime.utcnow()
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


async def bulk_mark_cancelled(
    session: AsyncSession,
    job_ids: Iterable[str],
) -> None:
    """Convenience helper to cancel multiple jobs (best-effort)."""

    if not job_ids:
        return

    stmt = select(TrainingJob).where(TrainingJob.id.in_(list(job_ids)))
    results = await session.execute(stmt)
    jobs = list(results.scalars().all())

    for job in jobs:
        await update_job_status(session, job, status=TrainingJobStatus.CANCELLED)

    logger.info("Cancelled %s training job(s)", len(jobs))


async def purge_training_jobs(
    session: AsyncSession,
    *,
    statuses: Optional[Sequence[Union[str, TrainingJobStatus]]] = None,
    older_than: Optional[datetime] = None,
    dataset_source_id: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> int:
    """Remove training jobs that match the supplied filters.

    Args:
        session: Active async SQLAlchemy session.
        statuses: Optional iterable that limits deletions to the provided
            lifecycle states. Accepts either :class:`TrainingJobStatus` members
            or their string representations.
        older_than: Optional UTC timestamp; jobs created on or before this
            value are eligible when provided.
        dataset_source_id: Optional dataset identifier to scope deletions.
        pipeline_id: Optional pipeline identifier to scope deletions.
        limit: Optional maximum number of jobs to delete in one call.
        dry_run: When ``True``, rows are not deleted; the function only
            reports how many would match.

    Returns:
        Count of deleted rows, or the number of rows that *would* be deleted
        when ``dry_run`` is enabled.
    """

    filters: List[ColumnElement[bool]] = []

    if statuses:
        resolved_statuses: List[str] = []
        for status in statuses:
            if isinstance(status, TrainingJobStatus):
                resolved_statuses.append(status.value)
            elif isinstance(status, str):
                try:
                    resolved_statuses.append(TrainingJobStatus(status.lower()).value)
                except ValueError:
                    logger.warning("Ignoring unknown training job status '%s'", status)
            else:
                logger.warning("Unsupported status type %s; ignoring entry", type(status))

        if resolved_statuses:
            filters.append(TrainingJob.status.in_(resolved_statuses))

    if older_than is not None:
        filters.append(TrainingJob.created_at <= older_than)

    if dataset_source_id is not None:
        filters.append(TrainingJob.dataset_source_id == dataset_source_id)

    if pipeline_id is not None:
        filters.append(TrainingJob.pipeline_id == pipeline_id)

    selection = select(TrainingJob.id).order_by(TrainingJob.created_at.asc())
    if filters:
        selection = selection.where(*filters)
    if isinstance(limit, int) and limit > 0:
        selection = selection.limit(limit)

    result = await session.execute(selection)
    job_ids = [row[0] for row in result.all()]

    if not job_ids:
        logger.info("No training jobs matched the purge filters")
        return 0

    if dry_run:
        logger.info("Dry run: %s training job(s) would be deleted", len(job_ids))
        return len(job_ids)

    deletion = delete(TrainingJob).where(TrainingJob.id.in_(job_ids))
    outcome = await session.execute(deletion)
    cursor = cast(CursorResult[Any], outcome)
    await session.commit()

    deleted_count = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else len(job_ids)
    logger.info("Deleted %s training job(s)", deleted_count)
    return deleted_count
