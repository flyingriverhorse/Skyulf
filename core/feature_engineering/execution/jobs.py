"""Background job management for full dataset execution."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from core.utils.datetime import utcnow
from core.feature_engineering.schemas import (
    FullExecutionSignal,
    PipelinePreviewRequest,
)
from core.feature_engineering.execution.engine import run_pipeline_execution
from core.feature_engineering.execution.data import load_dataset_frame
from core.feature_engineering.execution.graph import DATASET_NODE_ID

logger = logging.getLogger(__name__)

FullExecutionJobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
FULL_DATASET_EXECUTION_ROW_LIMIT = 200_000


@dataclass
class FullExecutionJob:
    id: str
    dataset_source_id: str
    status: FullExecutionJobStatus
    created_at: datetime
    updated_at: datetime
    signal: Optional[FullExecutionSignal] = None
    task: Optional[asyncio.Task] = None
    session: Optional[AsyncSession] = None
    
    # Execution context
    execution_order: List[str] = field(default_factory=list)
    node_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pipeline_id: Optional[str] = None
    applied_steps: List[str] = field(default_factory=list)
    preview_total_rows: int = 0
    
    # Request payload for re-execution context
    payload: Optional[PipelinePreviewRequest] = None
    graph_nodes: List[Dict[str, Any]] = field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = field(default_factory=list)


class FullExecutionJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, FullExecutionJob] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, job_id: str) -> asyncio.Lock:
        if job_id not in self._locks:
            self._locks[job_id] = asyncio.Lock()
        return self._locks[job_id]

    async def ensure_job(
        self,
        *,
        dataset_source_id: str,
        execution_order: List[str],
        node_map: Dict[str, Dict[str, Any]],
        target_node_id: Optional[str],
        total_rows_estimate: Optional[int],
        graph_nodes: Optional[List[Dict[str, Any]]] = None,
        graph_edges: Optional[List[Dict[str, Any]]] = None,
        defer_reason: Optional[str] = None,
    ) -> Tuple[FullExecutionJob, FullExecutionSignal, bool]:
        # Simple job ID strategy: dataset_source_id + target_node_id (if any)
        # In a real system, this might need to be more robust (e.g. hash of graph)
        job_id = f"full_exec_{dataset_source_id}"
        if target_node_id:
            job_id += f"_{target_node_id}"

        async with self._get_lock(job_id):
            existing = self._jobs.get(job_id)
            if existing:
                # If running or queued, return existing signal
                if existing.status in ("queued", "running"):
                    return existing, existing.signal or FullExecutionSignal(
                        status=existing.status,
                        job_id=job_id,
                        progress=0.0,
                        message="Job is already running",
                    ), False
                
                # If succeeded recently, maybe return cached? 
                # For now, we always restart if not running/queued.
            
            # Create new job
            signal = FullExecutionSignal(
                status="queued",
                job_id=job_id,
                progress=0.0,
                message=defer_reason or "Full execution queued",
                total_rows=total_rows_estimate,
            )
            
            job = FullExecutionJob(
                id=job_id,
                dataset_source_id=dataset_source_id,
                status="queued",
                created_at=utcnow(),
                updated_at=utcnow(),
                signal=signal,
                execution_order=execution_order,
                node_map=node_map,
                graph_nodes=graph_nodes or [],
                graph_edges=graph_edges or [],
            )
            self._jobs[job_id] = job
            return job, signal, True

    async def _update(
        self,
        job_id: str,
        *,
        status: Optional[FullExecutionJobStatus] = None,
        signal_updates: Optional[Dict[str, Any]] = None,
    ) -> Optional[FullExecutionJob]:
        job = self._jobs.get(job_id)
        if not job:
            return None
            
        if status:
            job.status = status
        
        job.updated_at = utcnow()
        
        if signal_updates and job.signal:
            current_data = job.signal.dict()
            current_data.update(signal_updates)
            job.signal = FullExecutionSignal(**current_data)
            
        return job

    async def mark_running(
        self,
        job_id: str,
        *,
        reason: Optional[str] = None,
        poll_after_seconds: Optional[int] = None,
    ) -> Optional[FullExecutionSignal]:
        updates: Dict[str, Any] = {"status": "running"}
        if reason:
            updates["message"] = reason
        if poll_after_seconds is not None:
            updates["poll_after"] = poll_after_seconds
            
        job = await self._update(job_id, status="running", signal_updates=updates)
        return job.signal if job else None

    async def mark_completed(
        self,
        job_id: str,
        *,
        status: FullExecutionJobStatus,
        signal_updates: Dict[str, Any],
    ) -> Optional[FullExecutionSignal]:
        updates = {**signal_updates, "status": status}
        job = await self._update(job_id, status=status, signal_updates=updates)
        return job.signal if job else None

    async def get_signal(self, dataset_source_id: str, job_id: str) -> Optional[FullExecutionSignal]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        return job.signal


full_execution_job_store = FullExecutionJobStore()


def _log_background_exception(task: asyncio.Task) -> None:
    try:
        exc = task.exception()
        if exc:
            logger.error(f"Background full execution task failed: {exc}", exc_info=exc)
    except Exception:
        pass


async def run_full_dataset_execution(
    *,
    session: AsyncSession,
    dataset_source_id: str,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    pipeline_id: str,
    applied_steps: List[str],
    preview_total_rows: int,
) -> Tuple[FullExecutionSignal, int]:
    
    # Load full dataset
    frame, meta = await load_dataset_frame(
        session,
        dataset_source_id,
        sample_size=0,
        execution_mode="full",
    )
    
    if frame.empty:
        raise ValueError(f"Could not load full dataset for {dataset_source_id}")
        
    total_rows = len(frame)
    
    # Run pipeline
    transformed_frame, steps, signals, _ = run_pipeline_execution(
        frame,
        execution_order,
        node_map,
        pipeline_id=pipeline_id,
        collect_signals=True,
    )
    
    # Build signal
    # We don't return the full frame in the signal, just stats
    signal = FullExecutionSignal(
        status="succeeded",
        job_id=pipeline_id, # Placeholder, will be updated by caller
        progress=1.0,
        message="Full execution completed",
        total_rows=total_rows,
        preview_rows=len(transformed_frame),
        columns=list(transformed_frame.columns),
        signals=signals,
    )
    
    return signal, len(transformed_frame)


async def _run_full_execution_job(job: FullExecutionJob) -> None:
    await full_execution_job_store.mark_running(
        job.id,
        reason="Full dataset execution running in background.",
        poll_after_seconds=4,
    )
    try:
        if not job.session:
            # In a real app, we'd need to create a new session here
            # For now, assuming session is passed or we can't run
            logger.error("No session available for background job")
            await full_execution_job_store.mark_completed(
                job.id,
                status="failed",
                signal_updates={"message": "Internal error: No database session"},
            )
            return

        signal, rows = await run_full_dataset_execution(
            session=job.session,
            dataset_source_id=job.dataset_source_id,
            execution_order=job.execution_order,
            node_map=job.node_map,
            pipeline_id=job.id,
            applied_steps=job.applied_steps,
            preview_total_rows=job.preview_total_rows,
        )
        
        await full_execution_job_store.mark_completed(
            job.id,
            status="succeeded",
            signal_updates={
                "progress": 1.0,
                "message": "Execution completed successfully",
                "total_rows": signal.total_rows,
                "preview_rows": signal.preview_rows,
                "columns": signal.columns,
                "signals": signal.signals,
            },
        )
        
    except MemoryError:
        logger.error("Memory error during full execution")
        await full_execution_job_store.mark_completed(
            job.id,
            status="failed",
            signal_updates={
                "message": "Dataset too large for memory. Try sampling or reducing columns.",
                "error_code": "MEMORY_LIMIT_EXCEEDED",
            },
        )
    except Exception as exc:
        logger.exception("Full execution job failed")
        await full_execution_job_store.mark_completed(
            job.id,
            status="failed",
            signal_updates={"message": str(exc)},
        )


def schedule_full_execution_job(job: FullExecutionJob) -> None:
    if job.task and not job.task.done():
        return
    task = asyncio.create_task(_run_full_execution_job(job))
    job.task = task
    task.add_done_callback(_log_background_exception)


def build_failed_full_execution_signal(
    *,
    status: FullExecutionJobStatus,
    reason: str,
    dataset_source_id: str,
    total_rows_estimate: Optional[int],
    warnings: Optional[List[str]] = None,
) -> FullExecutionSignal:
    return FullExecutionSignal(
        status=status,
        job_id="",
        progress=0.0,
        message=reason,
        total_rows=total_rows_estimate,
        warnings=warnings,
    )


def should_defer_full_execution(total_rows_estimate: Optional[int]) -> bool:
    if total_rows_estimate is None:
        return False
    return total_rows_estimate > FULL_DATASET_EXECUTION_ROW_LIMIT


async def defer_full_execution_for_limit(
    *,
    applied_steps: List[str],
    dataset_source_id: str,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    payload: PipelinePreviewRequest,
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    total_rows_estimate: int,
    preview_total_rows: int,
) -> Tuple[FullExecutionSignal, int]:
    
    job, signal, created = await full_execution_job_store.ensure_job(
        dataset_source_id=dataset_source_id,
        execution_order=execution_order,
        node_map=node_map,
        target_node_id=payload.target_node_id,
        total_rows_estimate=total_rows_estimate,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        defer_reason=f"Dataset size ({total_rows_estimate} rows) exceeds interactive limit.",
    )
    
    if created:
        # We need a session for the background job. 
        # This is tricky because the current session might close.
        # In a real app, the job runner should create its own session.
        # For now, we'll skip scheduling here and let the caller handle it 
        # or assume the job store can handle session creation (it can't yet).
        # TODO: Fix session management for background jobs
        pass

    return signal, preview_total_rows
