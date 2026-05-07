"""`POST /run` — async pipeline submission (E9 phase 2).

Owns the per-key submit lock, the partitioning into parallel branches,
the idempotency check that deduplicates double-clicks, and the Celery /
BackgroundTasks fan-out. All ML logic still lives in
`_execution/graph_utils` and `tasks`.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, cast

from fastapi import APIRouter, BackgroundTasks, Depends, Request
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
_submit_locks: Dict[str, asyncio.Lock] = {}
_submit_locks_guard = asyncio.Lock()


async def _get_submit_lock(key: str) -> asyncio.Lock:
    async with _submit_locks_guard:
        if key not in _submit_locks:
            _submit_locks[key] = asyncio.Lock()
        return _submit_locks[key]


@router.post("/run", response_model=RunPipelineResponse)
@limiter.limit("20/minute")
async def run_pipeline(  # noqa: C901
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
    from backend.ml_pipeline._execution.graph_utils import (
        _split_connected_components,
        partition_parallel_pipeline,
    )

    pipeline_id = config.pipeline_id

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

    # Split disconnected subgraphs into separate experiment groups first,
    # then partition each group for parallel branches.
    components = _split_connected_components(internal_config)
    sub_pipelines: list[PipelineConfig] = []
    for comp in components:
        sub_pipelines.extend(partition_parallel_pipeline(comp))

    # When a specific target node was requested and the partitioner split
    # by multiple terminals, only run the branch containing that node.
    # This ensures clicking Train on node A doesn't also execute node B.
    if config.target_node_id and len(sub_pipelines) > 1:
        filtered = [
            sub
            for sub in sub_pipelines
            if any(n.node_id == config.target_node_id for n in sub.nodes)
        ]
        if filtered:
            sub_pipelines = filtered

    # Detect dataset_id from the first DATA_LOADER node
    dataset_id = "unknown"
    for node in config.nodes:
        if node.step_type == StepType.DATA_LOADER:
            dataset_id = node.params.get("dataset_id", "unknown")
            break

    settings = get_settings()
    all_job_ids: List[str] = []
    task_payloads: List[tuple] = []

    for sub in sub_pipelines:
        # Identify the terminal node for this sub-pipeline
        target_node_id = config.target_node_id
        terminal_types = {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING, "data_preview"}
        for n in reversed(sub.nodes):
            if n.step_type in terminal_types:
                target_node_id = n.node_id
                break
        if not target_node_id and sub.nodes:
            target_node_id = sub.nodes[-1].node_id

        # Determine model type and job type from the terminal node
        model_type = "unknown"
        job_type = config.job_type or StepType.BASIC_TRAINING
        for n in sub.nodes:
            if n.node_id == target_node_id:
                if n.step_type == StepType.BASIC_TRAINING:
                    model_type = n.params.get("model_type", n.params.get("algorithm", "unknown"))
                    job_type = StepType.BASIC_TRAINING
                elif n.step_type == StepType.ADVANCED_TUNING:
                    model_type = n.params.get("algorithm", n.params.get("model_type", "unknown"))
                    job_type = StepType.ADVANCED_TUNING
                elif n.step_type == "data_preview":
                    model_type = "preview"
                    job_type = "preview"
                break

        # Build the per-branch graph snapshot. We persist *this branch's*
        # nodes (with the terminal's `inputs` already rewritten by the
        # partitioner to point only at this branch's parent in parallel
        # mode) instead of the full original config — otherwise the
        # Experiments comparison view walks back from a shared terminal
        # whose `inputs` still list every branch's parent and ends up
        # showing both branches' preprocessing chains in every column
        # (reported as "Path A and Path B both show every Encoding").
        branch_graph: Dict[str, Any] = {
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

        # Idempotency: if this exact node is already queued/running from a
        # recent submission, return the existing job id instead of spawning
        # a duplicate Celery task (e.g. accidental double-click).
        # The asyncio lock serialises concurrent requests so the check+create
        # pair is atomic within the event loop — no two coroutines race through
        # it at the same time.
        branch_index: int = sub.metadata.get("branch_index", 0)
        _submit_key = f"{dataset_id}:{target_node_id}:{branch_index}"
        _lock = await _get_submit_lock(_submit_key)
        async with _lock:
            existing_job_id = await JobManager.find_active_job(
                db, dataset_id, target_node_id or "unknown", branch_index
            )
            if existing_job_id:
                logger.info("Deduplicating submission: returning existing job %s", existing_job_id)
                all_job_ids.append(existing_job_id)
                continue

            # Create Job in DB (commits immediately, visible to next waiter)
            job_id = await JobManager.create_job(
                session=db,
                pipeline_id=sub.pipeline_id,
                node_id=target_node_id or "unknown",
                job_type=cast(Literal["basic_training", "advanced_tuning", "preview"], job_type),
                dataset_id=dataset_id,
                model_type=model_type,
                graph=branch_graph,
                branch_index=branch_index,
            )

        all_job_ids.append(job_id)
        publish_job_event(JobEvent(event="created", job_id=job_id, status="queued", progress=0))

        # Reuse the same dict shape for the Celery payload (storage_options
        # is added below; we don't persist it into the DB graph snapshot).
        sub_payload: Dict[str, Any] = dict(branch_graph)
        if resolved_s3_options:
            sub_payload["storage_options"] = resolved_s3_options

        # Collect all branches; submitted as a single batch after the loop.
        task_payloads.append((job_id, sub_payload))

    # Submit all branches in one Celery task (B4: one round-trip vs N).
    # For BackgroundTasks (non-Celery), keep per-branch concurrency.
    if task_payloads:
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

            def _run_branches_concurrently(payloads: List[tuple]) -> None:
                with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
                    futures = [pool.submit(run_pipeline_task, jid, pl) for jid, pl in payloads]
                    for f in futures:
                        f.result()  # propagate exceptions per-branch via logging

            background_tasks.add_task(_run_branches_concurrently, task_payloads)

    is_parallel = len(all_job_ids) > 1
    message = (
        f"Parallel execution started: {len(all_job_ids)} branches"
        if is_parallel
        else "Pipeline execution started"
    )

    return RunPipelineResponse(
        message=message,
        pipeline_id=pipeline_id,
        job_id=all_job_ids[0],
        job_ids=all_job_ids,
    )


__all__ = ["router"]
