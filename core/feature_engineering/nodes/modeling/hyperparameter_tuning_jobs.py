"""Utilities for managing hyperparameter tuning jobs."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Union

from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.models import HyperparameterTuningJob
from core.feature_engineering.schemas import HyperparameterTuningJobCreate, HyperparameterTuningJobStatus

logger = logging.getLogger(__name__)


async def _resolve_next_run_number(
    session: AsyncSession,
    *,
    pipeline_id: str,
    node_id: str,
) -> int:
    """Return the next sequential run number for a tuning node."""

    query: Select[tuple[int]] = select(func.max(HyperparameterTuningJob.run_number)).where(
        HyperparameterTuningJob.pipeline_id == pipeline_id,
        HyperparameterTuningJob.node_id == node_id,
    )
    result = await session.execute(query)
    current_max: Optional[int] = result.scalar()
    return (current_max or 0) + 1


async def create_tuning_job(
    session: AsyncSession,
    payload: HyperparameterTuningJobCreate,
    *,
    user_id: Optional[int] = None,
    model_type_override: Optional[str] = None,
) -> HyperparameterTuningJob:
    """Persist a new hyperparameter tuning job request."""

    node_id_value = payload.node_id or payload.target_node_id
    if not node_id_value:
        raise ValueError("Tuning job payload missing node identifier")

    job_id = uuid.uuid4().hex
    run_number = await _resolve_next_run_number(
        session,
        pipeline_id=payload.pipeline_id,
        node_id=node_id_value,
    )

    job_metadata = dict(payload.metadata or {})
    if payload.target_node_id:
        job_metadata.setdefault("target_node_id", payload.target_node_id)

    model_type_value = (model_type_override or payload.model_type).strip()

    job = HyperparameterTuningJob(
        id=job_id,
        dataset_source_id=payload.dataset_source_id,
        pipeline_id=payload.pipeline_id,
        node_id=node_id_value,
        user_id=user_id,
        status=HyperparameterTuningJobStatus.QUEUED.value,
        run_number=run_number,
        model_type=model_type_value,
        search_strategy=payload.search_strategy,
        search_space=payload.search_space,
        baseline_hyperparameters=payload.baseline_hyperparameters or {},
        n_iterations=payload.n_iterations,
        scoring=payload.scoring,
        random_state=payload.random_state,
        cross_validation=payload.cross_validation,
        job_metadata=job_metadata,
        graph=payload.graph.model_dump(),
    )

    session.add(job)
    await session.commit()
    await session.refresh(job)

    logger.info(
        "Enqueued tuning job %s (pipeline=%s node=%s run=%s)",
        job.id,
        job.pipeline_id,
        job.node_id,
        job.run_number,
    )

    return job


async def get_tuning_job(session: AsyncSession, job_id: str) -> Optional[HyperparameterTuningJob]:
    """Fetch a single tuning job by identifier."""

    return await session.get(HyperparameterTuningJob, job_id)


async def list_tuning_jobs(
    session: AsyncSession,
    *,
    user_id: Optional[int] = None,
    dataset_source_id: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    node_id: Optional[str] = None,
    limit: int = 20,
) -> List[HyperparameterTuningJob]:
    """Return recent tuning jobs filtered by optional parameters."""

    stmt = select(HyperparameterTuningJob).order_by(HyperparameterTuningJob.created_at.desc()).limit(limit)

    if user_id is not None:
        stmt = stmt.where(HyperparameterTuningJob.user_id == user_id)
    if dataset_source_id is not None:
        stmt = stmt.where(HyperparameterTuningJob.dataset_source_id == dataset_source_id)
    if pipeline_id is not None:
        stmt = stmt.where(HyperparameterTuningJob.pipeline_id == pipeline_id)
    if node_id is not None:
        stmt = stmt.where(HyperparameterTuningJob.node_id == node_id)

    results = await session.execute(stmt)
    return list(results.scalars().all())


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

    now = datetime.utcnow()
    job.status = status.value

    if status == HyperparameterTuningJobStatus.RUNNING:
        job.started_at = now
    if status in {
        HyperparameterTuningJobStatus.SUCCEEDED,
        HyperparameterTuningJobStatus.FAILED,
        HyperparameterTuningJobStatus.CANCELLED,
    }:
        job.finished_at = now

    if metrics is not None:
        job.metrics = metrics
    if results is not None:
        job.results = results
    if best_params is not None:
        job.best_params = best_params
    if best_score is not None:
        job.best_score = best_score
    if artifact_uri is not None:
        job.artifact_uri = artifact_uri
    if error_message is not None:
        job.error_message = error_message
    if metadata is not None:
        merged = dict(job.job_metadata or {})
        merged.update(metadata)
        job.job_metadata = merged

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


async def purge_tuning_jobs(
    session: AsyncSession,
    *,
    statuses: Optional[Sequence[Union[str, HyperparameterTuningJobStatus]]] = None,
    older_than: Optional[datetime] = None,
    dataset_source_id: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> int:
    """Remove tuning jobs that match the supplied filters."""

    filters = []

    if statuses:
        resolved_statuses: List[str] = []
        for status in statuses:
            if isinstance(status, HyperparameterTuningJobStatus):
                resolved_statuses.append(status.value)
            elif isinstance(status, str):
                try:
                    resolved_statuses.append(HyperparameterTuningJobStatus(status.lower()).value)
                except ValueError:
                    logger.warning("Ignoring unknown tuning job status '%s'", status)
            else:
                logger.warning("Unsupported status type %s; ignoring entry", type(status))

        if resolved_statuses:
            filters.append(HyperparameterTuningJob.status.in_(resolved_statuses))

    if older_than is not None:
        filters.append(HyperparameterTuningJob.created_at <= older_than)

    if dataset_source_id is not None:
        filters.append(HyperparameterTuningJob.dataset_source_id == dataset_source_id)

    if pipeline_id is not None:
        filters.append(HyperparameterTuningJob.pipeline_id == pipeline_id)

    selection = select(HyperparameterTuningJob.id).order_by(HyperparameterTuningJob.created_at.asc())
    if filters:
        selection = selection.where(*filters)
    if isinstance(limit, int) and limit > 0:
        selection = selection.limit(limit)

    result = await session.execute(selection)
    job_ids = [row[0] for row in result.all()]

    if not job_ids:
        logger.info("No tuning jobs matched the purge filters")
        return 0

    if dry_run:
        logger.info("Dry run: %s tuning job(s) would be deleted", len(job_ids))
        return len(job_ids)

    deletion = delete(HyperparameterTuningJob).where(HyperparameterTuningJob.id.in_(job_ids))
    outcome = await session.execute(deletion)
    await session.commit()

    deleted_count = outcome.rowcount if outcome.rowcount and outcome.rowcount > 0 else len(job_ids)
    logger.info("Deleted %s tuning job(s)", deleted_count)
    return deleted_count
