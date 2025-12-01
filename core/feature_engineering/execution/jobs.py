"""Background job management for full dataset execution."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from celery.result import AsyncResult
from sqlalchemy.ext.asyncio import AsyncSession

from core.utils.datetime import utcnow
from core.feature_engineering.schemas import (
    FullExecutionSignal,
    PipelinePreviewRequest,
)
from core.feature_engineering.execution.engine import run_pipeline_execution
from core.feature_engineering.execution.data import load_dataset_frame
from core.feature_engineering.execution.graph import DATASET_NODE_ID
from core.feature_engineering.execution.tasks import run_full_dataset_execution_task

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
    task_id: Optional[str] = None
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
        applied_steps: Optional[List[str]] = None,
        preview_total_rows: int = 0,
    ) -> Tuple[FullExecutionJob, FullExecutionSignal, bool]:
        # Simple job ID strategy: dataset_source_id + target_node_id (if any)
        # In a real system, this might need to be more robust (e.g. hash of graph)
        job_id = f"full_exec_{dataset_source_id}"
        if target_node_id:
            job_id += f"_{target_node_id}"

        async with self._get_lock(job_id):
            existing = self._jobs.get(job_id)
            if existing:
                # Check if existing job is still valid/running via Celery
                if existing.task_id:
                    task_result = AsyncResult(existing.task_id)
                    if task_result.state in ("PENDING", "STARTED", "RETRY"):
                         return existing, existing.signal or FullExecutionSignal(
                            status="deferred",
                            job_status="running",
                            job_id=job_id,
                            reason="Job is already running",
                        ), False
                
                # If succeeded recently, maybe return cached? 
                # For now, we always restart if not running/queued.
            
            # Create new job
            signal = FullExecutionSignal(
                status="deferred",
                job_status="queued",
                job_id=job_id,
                reason=defer_reason or "Full execution queued",
                total_rows=total_rows_estimate,
            )
            
            # Trigger Celery Task
            task = run_full_dataset_execution_task.delay(
                dataset_source_id=dataset_source_id,
                execution_order=execution_order,
                node_map=node_map,
                pipeline_id=job_id,
                applied_steps=applied_steps or [],
                preview_total_rows=preview_total_rows,
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
                task_id=task.id,
                applied_steps=applied_steps or [],
                preview_total_rows=preview_total_rows,
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
            
        if job.task_id:
            task_result = AsyncResult(job.task_id)
            if task_result.state == "SUCCESS":
                result_data = task_result.result
                if isinstance(result_data, dict):
                    job.signal = FullExecutionSignal(**result_data)
                    if job.signal.job_status:
                        job.status = job.signal.job_status
                    elif job.signal.status == "failed":
                        job.status = "failed"
                    elif job.signal.status == "skipped":
                        job.status = "cancelled"
                    else:
                        job.status = "succeeded"
            elif task_result.state == "FAILURE":
                job.status = "failed"
                job.signal = FullExecutionSignal(
                    status="failed",
                    job_status="failed",
                    job_id=job_id,
                    reason=str(task_result.result),
                )
            elif task_result.state in ("PENDING", "STARTED", "RETRY"):
                job.status = "running"
                # Update signal status if needed
                if job.signal:
                    job.signal.status = "deferred"
                    job.signal.job_status = "running"
        
        return job.signal


full_execution_job_store = FullExecutionJobStore()



def build_failed_full_execution_signal(
    *,
    status: FullExecutionJobStatus,
    reason: str,
    dataset_source_id: str,
    total_rows_estimate: Optional[int],
    warnings: Optional[List[str]] = None,
) -> FullExecutionSignal:
    signal_status = "failed"
    if status in ("queued", "running"):
        signal_status = "deferred"
    elif status == "cancelled":
        signal_status = "skipped"
    elif status == "succeeded":
        signal_status = "succeeded"

    return FullExecutionSignal(
        status=signal_status,
        job_status=status,
        job_id="",
        reason=reason,
        total_rows=total_rows_estimate,
        warnings=warnings or [],
        dataset_source_id=dataset_source_id,
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
        applied_steps=applied_steps,
        preview_total_rows=preview_total_rows,
    )
    
    if created:
        # Job is already scheduled via Celery in ensure_job
        pass

    return signal, preview_total_rows
