"""Persistence helpers for training jobs."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, List, Optional, Sequence, Union, cast

from sqlalchemy import Select, delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from core.database.models import TrainingJob
from core.feature_engineering.schemas import TrainingJobCreate, TrainingJobStatus

logger = logging.getLogger(__name__)


async def _resolve_next_version(
    session: AsyncSession,
    *,
    dataset_source_id: str,
    node_id: str,
    model_type: Optional[str] = None,
) -> int:
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


__all__ = [
    "create_training_job",
    "get_training_job",
    "list_training_jobs",
    "purge_training_jobs",
]
