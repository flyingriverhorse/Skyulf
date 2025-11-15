"""Database access helpers for hyperparameter tuning jobs."""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.models import HyperparameterTuningJob


async def _resolve_next_run_number(
    session: AsyncSession,
    *,
    dataset_source_id: str,
    node_id: str,
    model_type: Optional[str] = None,
) -> int:
    """Return the next sequential run number for a dataset/node/model-type trio."""

    filters = [
        HyperparameterTuningJob.dataset_source_id == dataset_source_id,
        HyperparameterTuningJob.node_id == node_id,
    ]

    if model_type:
        filters.append(HyperparameterTuningJob.model_type == model_type)

    query: Select[tuple[int]] = select(func.max(HyperparameterTuningJob.run_number)).where(*filters)
    result = await session.execute(query)
    current_max: Optional[int] = result.scalar()
    return (current_max or 0) + 1


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


__all__ = [
    "_resolve_next_run_number",
    "get_tuning_job",
    "list_tuning_jobs",
]
