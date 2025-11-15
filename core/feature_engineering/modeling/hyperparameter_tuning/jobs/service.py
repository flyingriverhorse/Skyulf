"""Service-layer helpers for hyperparameter tuning jobs."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, List, Optional, Sequence, Union, cast

from sqlalchemy import ColumnElement, delete, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.models import HyperparameterTuningJob
from core.feature_engineering.schemas import HyperparameterTuningJobCreate, HyperparameterTuningJobStatus

from .repository import _resolve_next_run_number

logger = logging.getLogger(__name__)


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

    model_type_value = (model_type_override or payload.model_type or "").strip()
    if not model_type_value:
        raise ValueError("Tuning job payload missing model type")

    job_id = uuid.uuid4().hex
    run_number = await _resolve_next_run_number(
        session,
        dataset_source_id=payload.dataset_source_id,
        node_id=node_id_value,
        model_type=model_type_value,
    )

    job_metadata = dict(payload.metadata or {})
    if payload.target_node_id:
        job_metadata.setdefault("target_node_id", payload.target_node_id)

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

    filters: List[ColumnElement[bool]] = []

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
    cursor = cast(CursorResult[Any], outcome)
    await session.commit()

    deleted_count = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else len(job_ids)
    logger.info("Deleted %s tuning job(s)", deleted_count)
    return deleted_count


__all__ = [
    "create_tuning_job",
    "purge_tuning_jobs",
]
