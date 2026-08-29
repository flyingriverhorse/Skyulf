"""Job-management endpoints (E9 phase 2).

`/jobs/node-summaries`, `/jobs/{job_id}` (status/cancel/promote/unpromote),
`/jobs/{job_id}/evaluation`, `/jobs/{job_id}/thresholds/...`
(preview/save/toggle/clear tuned thresholds), `/jobs` (list),
`/jobs/tuning/...`.

All handlers delegate to `JobManager` / `EvaluationService`; this module
is a pure HTTP veneer.
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database.engine import get_async_session
from backend.exceptions.core import SkyulfException
from backend.ml_pipeline._execution.jobs import JobInfo, JobManager
from backend.ml_pipeline._internal._routers.run_pipeline import resubmit_job_from_graph
from backend.ml_pipeline._services.evaluation_service import EvaluationService
from backend.ml_pipeline._services.threshold_tuning_service import (
    ThresholdTuningError,
    ThresholdTuningService,
)
from backend.realtime.events import JobEvent, publish_job_event
from backend.realtime.trial_buffer import get_iterations, get_trials

# Retry (OPS-001) only makes sense for jobs that persisted a resubmittable
# pipeline graph snapshot and have reached a state that will never mutate
# again on its own.
RETRIABLE_JOB_TYPES = {"training", "tuning"}
RETRIABLE_STATUSES = {"failed", "cancelled"}

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Pipeline"])


class ThresholdTuningPreviewRequest(BaseModel):
    """Request body for previewing tuned thresholds."""

    metric: str


class ThresholdTuningPreviewResponse(BaseModel):
    """Response body for a threshold-tuning preview."""

    thresholds: dict[str, float]
    classes: list
    metric: str
    split_used: str


class ThresholdTuningSaveRequest(BaseModel):
    """Request body for saving a previously previewed threshold set."""

    thresholds: dict[str, float]
    classes: list
    metric: str
    split_used: str


class ThresholdTuningToggleRequest(BaseModel):
    """Request body for enabling/disabling a job's saved tuned thresholds."""

    enabled: bool


class ThresholdTuningGetResponse(BaseModel):
    """Response body describing a job's currently saved tuned thresholds, if any."""

    thresholds: dict[str, float] | None = None
    classes: list | None = None
    metric: str | None = None
    split_used: str | None = None
    computed_at: str | None = None
    # "training" when seeded by training-time threshold tuning; None for
    # manually saved/legacy sets.
    source: str | None = None
    enabled: bool = False


@router.get("/jobs/node-summaries", response_model=dict[str, list[dict[str, Any]]])
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


@router.get("/jobs/{job_id}/trials")
async def get_job_trials(job_id: str):
    """Snapshot of completed trials/iterations so far, for charts opened mid-run.

    The ``/ws/jobs`` broadcast only reaches already-connected clients; this
    backfills the trials (tuning) or boosting iterations a late opener
    missed. Empty once the job's buffer has been evicted — terminal jobs
    redraw from persisted ``metrics.trials`` / ``metrics.iterations``.
    """
    trials = get_trials(job_id)
    metric = next((t["metric"] for t in reversed(trials) if t["metric"]), None)
    iterations = get_iterations(job_id)
    iteration_metric = next((t["metric"] for t in reversed(iterations) if t["metric"]), None)
    return {
        "trials": trials,
        "metric": metric,
        "iterations": iterations,
        "iteration_metric": iteration_metric,
    }


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


class RetryJobResponse(BaseModel):
    """Response body for a successful job retry submission."""

    job_id: str
    message: str


@router.post("/jobs/{job_id}/retry", response_model=RetryJobResponse)
async def retry_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
):
    """Resubmit a failed or cancelled training/tuning job from its stored graph.

    Only training/tuning jobs that reached a terminal, non-successful state
    and still have a stored pipeline graph snapshot can be retried; EDA and
    ingestion jobs, and jobs missing a graph snapshot, return 400 so the
    frontend can explain why retry isn't offered instead of silently no-oping.
    """
    job = await JobManager.get_job(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.job_type not in RETRIABLE_JOB_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Retry is not supported for {job.job_type} jobs",
        )
    if job.status not in RETRIABLE_STATUSES:
        raise HTTPException(
            status_code=400,
            detail="Only failed or cancelled jobs can be retried",
        )
    if not job.graph or not job.graph.get("nodes"):
        raise HTTPException(
            status_code=400,
            detail="Job has no stored pipeline graph to retry",
        )

    try:
        new_job_id = await resubmit_job_from_graph(session, job, background_tasks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return RetryJobResponse(job_id=new_job_id, message="Retry submitted")


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
            raise HTTPException(status_code=404, detail=str(e)) from e
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception:
        logger.exception("Failed to retrieve evaluation for job %s", job_id)
        raise SkyulfException(message="Failed to retrieve evaluation data") from None


@router.get("/jobs/{job_id}/thresholds", response_model=ThresholdTuningGetResponse)
async def get_thresholds(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """Return the job's currently saved tuned thresholds (if any) and whether they're enabled.

    Lets the Evaluation tab restore its "Use tuned thresholds" toggle/preview
    on load instead of always starting unchecked, and lets the Inference page
    show what's already active for a deployment before the user runs a
    prediction.
    """
    try:
        result = await ThresholdTuningService.get_saved(session, job_id)
    except ThresholdTuningError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ThresholdTuningGetResponse(**result)


@router.post("/jobs/{job_id}/thresholds/preview", response_model=ThresholdTuningPreviewResponse)
async def preview_thresholds(
    job_id: str,
    request: ThresholdTuningPreviewRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Compute (without saving) tuned per-class thresholds for a job's evaluation data."""
    try:
        result = await ThresholdTuningService.preview(session, job_id, request.metric)
    except ThresholdTuningError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ThresholdTuningPreviewResponse(**result)


@router.post("/jobs/{job_id}/thresholds/save")
async def save_thresholds(
    job_id: str,
    request: ThresholdTuningSaveRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Persist a previewed threshold set on the job, enabling it by default."""
    try:
        await ThresholdTuningService.save(
            session,
            job_id,
            thresholds=request.thresholds,
            classes=request.classes,
            metric=request.metric,
            split_used=request.split_used,
        )
    except ThresholdTuningError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"status": "saved"}


@router.post("/jobs/{job_id}/thresholds/toggle")
async def toggle_thresholds(
    job_id: str,
    request: ThresholdTuningToggleRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Enable or disable use of the job's saved tuned thresholds at predict time."""
    try:
        await ThresholdTuningService.toggle(session, job_id, request.enabled)
    except ThresholdTuningError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"status": "toggled", "enabled": request.enabled}


@router.delete("/jobs/{job_id}/thresholds")
async def clear_thresholds(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """Remove any saved tuned thresholds from the job."""
    try:
        await ThresholdTuningService.clear(session, job_id)
    except ThresholdTuningError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"status": "cleared"}


@router.get("/jobs", response_model=list[JobInfo])
async def list_jobs(
    limit: int | None = None,
    skip: int = 0,
    job_type: Literal["training", "tuning"] | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    """List recent jobs."""
    effective_limit = limit if limit is not None else get_settings().DEFAULT_PAGE_SIZE
    return await JobManager.list_jobs(session, effective_limit, skip, job_type)


@router.get("/jobs/tuning/latest/{node_id}", response_model=JobInfo | None)
async def get_latest_tuning_job(node_id: str, session: AsyncSession = Depends(get_async_session)):
    """Latest completed tuning job for a specific node."""
    return await JobManager.get_latest_tuning_job_for_node(session, node_id)


@router.get("/jobs/tuning/best/{model_type}", response_model=JobInfo | None)
async def get_best_tuning_job_model(
    model_type: str, session: AsyncSession = Depends(get_async_session)
):
    """Best (latest completed) tuning job for a specific model type."""
    return await JobManager.get_best_tuning_job_for_model(session, model_type)


@router.get("/jobs/tuning/history/{model_type}", response_model=list[JobInfo])
async def get_tuning_jobs_history(
    model_type: str, session: AsyncSession = Depends(get_async_session)
):
    """History of completed tuning jobs for a specific model type."""
    return await JobManager.get_tuning_jobs_for_model(session, model_type)


__all__ = ["router"]
