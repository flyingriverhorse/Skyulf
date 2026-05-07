"""Job-management endpoints (E9 phase 2).

`/jobs/node-summaries`, `/jobs/{job_id}` (status/cancel/promote/unpromote),
`/jobs/{job_id}/evaluation`, `/jobs` (list), `/jobs/tuning/...`.

All handlers delegate to `JobManager` / `EvaluationService`; this module
is a pure HTTP veneer.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.engine import get_async_session
from backend.exceptions.core import SkyulfException
from backend.ml_pipeline._execution.jobs import JobInfo, JobManager
from backend.ml_pipeline._services.evaluation_service import EvaluationService
from backend.realtime.events import JobEvent, publish_job_event

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Pipeline"])


@router.get("/jobs/node-summaries", response_model=Dict[str, List[Dict[str, Any]]])
async def get_node_summaries(limit: int = 200, session: AsyncSession = Depends(get_async_session)):
    """Per-node card summaries from the latest completed run group.

    Returns ``{ node_id: [entry, ...] }`` where each entry carries a
    ``summary`` string plus parallel-branch metadata (``branch_index``,
    ``pipeline_id``, ``parent_pipeline_id``, ``finished_at``). For
    canvases with a parallel terminal (one training node fed by N
    branches), the array contains one entry per branch so the card can
    render Path A / Path B / … on separate lines. Older run groups are
    dropped per node id so a fresh single-branch run never inherits
    stale per-branch entries from a previous parallel run.

    Lets the canvas render the same one-liner on trainer cards that the
    engine produces inline for every other node — trainer/tuner jobs
    run via Celery and the engine's per-node ``metadata.summary``
    never reaches the FE store through the regular ``/preview`` path
    (which strips trainers).
    """
    return await JobManager.get_node_summaries(session, limit=limit)


@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """Return the status of a background job."""
    job = await JobManager.get_job(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """Cancel a running or queued job."""
    success = await JobManager.cancel_job(session, job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job could not be cancelled (maybe it's already finished or doesn't exist)",
        )
    publish_job_event(JobEvent(event="status", job_id=job_id, status="cancelled"))
    return {"message": "Job cancelled successfully"}


@router.post("/jobs/{job_id}/promote")
async def promote_job(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """Mark a completed job as the promoted winner."""
    success = await JobManager.promote_job(session, job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job could not be promoted (must be completed and exist)",
        )
    return {"message": "Job promoted successfully"}


@router.delete("/jobs/{job_id}/promote")
async def unpromote_job(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """Remove promotion from a job."""
    success = await JobManager.unpromote_job(session, job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job unPromoted successfully"}


@router.get("/jobs/{job_id}/evaluation")
async def get_job_evaluation(  # noqa: C901
    job_id: str, session: AsyncSession = Depends(get_async_session)
):
    """Retrieve the raw evaluation data (y_true, y_pred) for a job."""
    try:
        return await EvaluationService.get_job_evaluation(session, job_id)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Failed to retrieve evaluation for job %s", job_id)
        raise SkyulfException(message="Failed to retrieve evaluation data")


@router.get("/jobs", response_model=List[JobInfo])
async def list_jobs(
    limit: int = 50,
    skip: int = 0,
    job_type: Optional[Literal["training", "tuning"]] = None,
    session: AsyncSession = Depends(get_async_session),
):
    """List recent jobs."""
    return await JobManager.list_jobs(session, limit, skip, job_type)


@router.get("/jobs/tuning/latest/{node_id}", response_model=Optional[JobInfo])
async def get_latest_tuning_job(node_id: str, session: AsyncSession = Depends(get_async_session)):
    """Latest completed tuning job for a specific node."""
    return await JobManager.get_latest_tuning_job_for_node(session, node_id)


@router.get("/jobs/tuning/best/{model_type}", response_model=Optional[JobInfo])
async def get_best_tuning_job_model(
    model_type: str, session: AsyncSession = Depends(get_async_session)
):
    """Best (latest completed) tuning job for a specific model type."""
    return await JobManager.get_best_tuning_job_for_model(session, model_type)


@router.get("/jobs/tuning/history/{model_type}", response_model=List[JobInfo])
async def get_tuning_jobs_history(
    model_type: str, session: AsyncSession = Depends(get_async_session)
):
    """History of completed tuning jobs for a specific model type."""
    return await JobManager.get_tuning_jobs_for_model(session, model_type)


__all__ = ["router"]
