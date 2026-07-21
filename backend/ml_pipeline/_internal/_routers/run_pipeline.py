"""`POST /run` — async pipeline submission (E9 phase 2).

Owns the per-key submit lock, the partitioning into parallel branches,
the idempotency check that deduplicates double-clicks, and the Celery /
BackgroundTasks fan-out. All ML logic still lives in
`_execution/graph_utils` and `tasks`.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.data_ingestion.service import DataIngestionService
from backend.database.engine import get_async_session
from backend.middleware.rate_limiter import limiter
from backend.ml_pipeline._execution.jobs import JobManager
from backend.ml_pipeline._execution.schemas import (
    NodeConfig,
    PipelineConfig,
    coerce_step_type,
)
from backend.ml_pipeline._internal._schemas import (
    PipelineConfigModel,
    RunPipelineResponse,
)
from backend.ml_pipeline.constants import StepType
from backend.ml_pipeline.resolution import resolve_pipeline_nodes
from backend.ml_pipeline.tasks import run_pipeline_batch_task, run_pipeline_task
from backend.realtime.events import JobEvent, publish_job_event

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Pipeline"])

# Per-key asyncio locks prevent two concurrent requests from racing through
# the find_active_job + create_job check simultaneously (same event loop).
_submit_locks: dict[str, asyncio.Lock] = {}
_submit_locks_guard = asyncio.Lock()


async def _get_submit_lock(key: str) -> asyncio.Lock:
    async with _submit_locks_guard:
        if key not in _submit_locks:
            _submit_locks[key] = asyncio.Lock()
        return _submit_locks[key]


async def _release_submit_lock(key: str) -> None:
    """Evict the lock entry once no submission is actively holding it.

    Any waiter that already has a reference to the lock object will acquire
    it normally; the eviction only prevents the dict from growing unbounded.
    """
    async with _submit_locks_guard:
        _submit_locks.pop(key, None)


def _build_sub_pipelines(
    config: PipelineConfigModel, internal_config: PipelineConfig
) -> list[PipelineConfig]:
    """Split disconnected subgraphs into experiment groups, then partition each for parallel branches.

    When a specific target node was requested and the partitioner split by
    multiple terminals, only the branch containing that node is returned —
    this ensures clicking Train on node A doesn't also execute node B.
    """
    from backend.ml_pipeline._execution.graph_utils import (
        _split_connected_components,
        partition_parallel_pipeline,
    )

    components = _split_connected_components(internal_config)
    sub_pipelines: list[PipelineConfig] = []
    for comp in components:
        sub_pipelines.extend(partition_parallel_pipeline(comp))

    if config.target_node_id and len(sub_pipelines) > 1:
        filtered = [
            sub
            for sub in sub_pipelines
            if any(n.node_id == config.target_node_id for n in sub.nodes)
        ]
        if filtered:
            sub_pipelines = filtered

    return sub_pipelines


def _detect_dataset_id(config_nodes: list[Any]) -> str:
    """Detect the dataset_id from the first DATA_LOADER node in the request."""
    for node in config_nodes:
        if node.step_type == StepType.DATA_LOADER:
            return node.params.get("dataset_id", "unknown")
    return "unknown"


def _resolve_branch_target_node_id(
    sub: PipelineConfig, requested_target_node_id: str | None
) -> str | None:
    """Identify the terminal node for a sub-pipeline (training/tuning/preview leaf)."""
    target_node_id = requested_target_node_id
    terminal_types = {
        StepType.TRAINING,
        "data_preview",
    }
    for n in reversed(sub.nodes):
        if n.step_type in terminal_types:
            target_node_id = n.node_id
            break
    if not target_node_id and sub.nodes:
        target_node_id = sub.nodes[-1].node_id
    return target_node_id


def _resolve_model_and_job_type(
    sub: PipelineConfig, target_node_id: str | None, requested_job_type: Any
) -> tuple[str, Any]:
    """Determine model type and job type from the sub-pipeline's terminal node."""
    model_type = "unknown"
    job_type = requested_job_type or "training"
    for n in sub.nodes:
        if n.node_id != target_node_id:
            continue
        if n.step_type == StepType.TRAINING:
            model_type = n.params.get("algorithm", n.params.get("model_type", "unknown"))
            job_type = "tuning" if n.params.get("run_mode", "fixed") == "tuned" else "training"
        elif n.step_type == "data_preview":
            model_type = "preview"
            job_type = "preview"
        break
    return model_type, job_type


def _build_branch_graph(sub: PipelineConfig) -> dict[str, Any]:
    """Build the per-branch graph snapshot persisted to the Job's `graph` column.

    We persist *this branch's* nodes (with the terminal's `inputs` already
    rewritten by the partitioner to point only at this branch's parent in
    parallel mode) instead of the full original config — otherwise the
    Experiments comparison view walks back from a shared terminal whose
    `inputs` still list every branch's parent and ends up showing both
    branches' preprocessing chains in every column (reported as "Path A and
    Path B both show every Encoding").
    """
    return {
        "pipeline_id": sub.pipeline_id,
        "nodes": [
            {
                "node_id": n.node_id,
                "step_type": n.step_type,
                "params": n.params,
                "inputs": n.inputs,
            }
            for n in sub.nodes
        ],
        "metadata": sub.metadata,
    }


async def _submit_or_dedupe_branch_job(
    db: AsyncSession,
    dataset_id: str,
    target_node_id: str | None,
    branch_index: int,
    sub: PipelineConfig,
    job_type: Any,
    model_type: str,
    branch_graph: dict[str, Any],
) -> tuple[str, bool]:
    """Return an existing job id if this branch is already queued/running, else create one.

    Idempotency: if this exact node is already queued/running from a recent
    submission, the existing job id is returned instead of spawning a
    duplicate Celery task (e.g. accidental double-click). The asyncio lock
    serialises concurrent requests so the check+create pair is atomic within
    the event loop — no two coroutines race through it at the same time.

    Returns ``(job_id, was_existing)``.
    """
    submit_key = f"{dataset_id}:{target_node_id}:{branch_index}"
    lock = await _get_submit_lock(submit_key)
    async with lock:
        existing_job_id = await JobManager.find_active_job(
            db, dataset_id, target_node_id or "unknown", branch_index
        )
        if existing_job_id:
            logger.info("Deduplicating submission: returning existing job %s", existing_job_id)
            await _release_submit_lock(submit_key)
            return existing_job_id, True

        # Create Job in DB (commits immediately, visible to next waiter)
        job_id = await JobManager.create_job(
            session=db,
            pipeline_id=sub.pipeline_id,
            node_id=target_node_id or "unknown",
            job_type=cast(Literal["training", "tuning", "preview"], job_type),
            dataset_id=dataset_id,
            model_type=model_type,
            graph=branch_graph,
            branch_index=branch_index,
        )
    await _release_submit_lock(submit_key)
    return job_id, False


async def _submit_branch_jobs(
    db: AsyncSession,
    sub_pipelines: list[PipelineConfig],
    config: PipelineConfigModel,
    dataset_id: str,
    resolved_s3_options: Any,
) -> tuple[list[str], list[tuple]]:
    """Create/dedupe a Job row per branch and build Celery/BackgroundTasks payloads.

    Returns ``(all_job_ids, task_payloads)`` where ``task_payloads`` only
    contains newly-created jobs (deduped/existing jobs are not re-submitted).
    """
    all_job_ids: list[str] = []
    task_payloads: list[tuple] = []

    for sub in sub_pipelines:
        target_node_id = _resolve_branch_target_node_id(sub, config.target_node_id)
        model_type, job_type = _resolve_model_and_job_type(sub, target_node_id, config.job_type)
        branch_graph = _build_branch_graph(sub)

        branch_index: int = sub.metadata.get("branch_index", 0)
        job_id, was_existing = await _submit_or_dedupe_branch_job(
            db, dataset_id, target_node_id, branch_index, sub, job_type, model_type, branch_graph
        )
        all_job_ids.append(job_id)
        if was_existing:
            continue

        publish_job_event(JobEvent(event="created", job_id=job_id, status="queued", progress=0))

        # Reuse the same dict shape for the Celery payload (storage_options
        # is added below; we don't persist it into the DB graph snapshot).
        sub_payload: dict[str, Any] = dict(branch_graph)
        if resolved_s3_options:
            sub_payload["storage_options"] = resolved_s3_options

        # Collect all branches; submitted as a single batch after the loop.
        task_payloads.append((job_id, sub_payload))

    return all_job_ids, task_payloads


def _run_branches_concurrently(payloads: list[tuple], max_parallel_workers: int) -> None:
    """Run each branch's pipeline task in its own thread (BackgroundTasks, non-Celery path)."""
    max_workers = min(len(payloads), max_parallel_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_pipeline_task, jid, pl) for jid, pl in payloads]
        for f in futures:
            f.result()  # propagate exceptions per-branch via logging


async def _dispatch_branch_tasks(
    task_payloads: list[tuple],
    settings: Any,
    background_tasks: BackgroundTasks,
    db: AsyncSession,
) -> None:
    """Dispatch branch execution via Celery (batched) or BackgroundTasks (single/concurrent).

    Submits all branches in one Celery task (B4: one round-trip vs N) when
    Celery is enabled. For BackgroundTasks (non-Celery), a single branch runs
    directly while multiple branches run concurrently via a thread pool.
    """
    if not task_payloads:
        return

    if settings.USE_CELERY:
        task = run_pipeline_batch_task.delay(task_payloads)
        # Attach the same Celery task id to every job so cancel_job can revoke it.
        for jid, _ in task_payloads:
            try:
                await JobManager.attach_celery_task_id(db, jid, task.id)
            except Exception:
                logger.warning("Failed to attach celery task id for job %s", jid)
    elif len(task_payloads) == 1:
        background_tasks.add_task(run_pipeline_task, *task_payloads[0])
    else:
        background_tasks.add_task(
            _run_branches_concurrently, task_payloads, settings.MAX_PARALLEL_BRANCH_WORKERS
        )


@router.post("/run", response_model=RunPipelineResponse)
@limiter.limit("20/minute")
async def run_pipeline(
    config: PipelineConfigModel,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
):
    """Submit a pipeline for asynchronous execution via Celery or BackgroundTasks.

    When the graph contains multiple training nodes or a training node with
    ``execution_mode == "parallel"``, the pipeline is automatically partitioned
    into independent sub-pipelines. Each sub-pipeline gets its own job and
    runs concurrently. The response includes all ``job_ids``.
    """
    pipeline_id = config.pipeline_id

    if not config.nodes:
        raise HTTPException(status_code=400, detail="Pipeline has no nodes")

    # --- Path Resolution Logic ---
    ingestion_service = DataIngestionService(db)
    resolved_s3_options = await resolve_pipeline_nodes(config.nodes, ingestion_service)

    # Convert API models to internal dataclasses for partitioning
    internal_nodes = [
        NodeConfig(
            node_id=n.node_id,
            step_type=coerce_step_type(n.step_type),
            params=n.params,
            inputs=n.inputs,
        )
        for n in config.nodes
    ]
    internal_config = PipelineConfig(
        pipeline_id=pipeline_id,
        nodes=internal_nodes,
        metadata=config.metadata,
    )

    sub_pipelines = _build_sub_pipelines(config, internal_config)
    dataset_id = _detect_dataset_id(config.nodes)
    settings = get_settings()

    all_job_ids, task_payloads = await _submit_branch_jobs(
        db, sub_pipelines, config, dataset_id, resolved_s3_options
    )

    await _dispatch_branch_tasks(task_payloads, settings, background_tasks, db)

    is_parallel = len(all_job_ids) > 1
    message = (
        f"Parallel execution started: {len(all_job_ids)} branches"
        if is_parallel
        else "Pipeline execution started"
    )

    if not all_job_ids:
        raise HTTPException(status_code=400, detail="Pipeline produced no runnable sub-pipelines")

    return RunPipelineResponse(
        message=message,
        pipeline_id=pipeline_id,
        job_id=all_job_ids[0],
        job_ids=all_job_ids,
    )


__all__ = ["router"]
