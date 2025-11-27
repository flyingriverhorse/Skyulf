import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

import core.database.engine as db_engine
from core.feature_engineering.execution.data import coerce_int, load_dataset_frame
from core.feature_engineering.execution.engine import run_pipeline_execution
from core.feature_engineering.execution.graph import (
    DATASET_NODE_ID,
    execution_order,
    generate_pipeline_id,
    resolve_catalog_type,
)
from core.feature_engineering.execution.jobs import (
    FULL_DATASET_EXECUTION_ROW_LIMIT,
    FullExecutionJob,
    full_execution_job_store,
)
from core.feature_engineering.schemas import (
    FullExecutionJobStatus,
    FullExecutionSignal,
    PipelinePreviewRequest,
    PipelinePreviewSignals,
)
from core.feature_engineering.eda_fast.service import DEFAULT_SAMPLE_CAP
from core.utils.datetime import utcnow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreviewSamplingConfig:
    execution_order: List[str]
    target_catalog_type: str
    include_preview_rows: bool
    effective_sample_size: int
    metrics_requested_sample_size: int
    load_mode: Literal["auto", "sample", "full"]


def append_unique_step(applied_steps: List[str], message: str) -> None:
    normalized = (message or "").strip()
    if not normalized:
        return
    if normalized not in applied_steps:
        applied_steps.append(normalized)


def build_preview_node_map(graph_nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    node_map: Dict[str, Dict[str, Any]] = {}
    for node in graph_nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        data = node.get("data")
        safe_data = {key: value for key, value in data.items() if not callable(value)} if isinstance(data, dict) else {}
        node_map[str(node_id)] = {**node, "data": safe_data}

    if DATASET_NODE_ID not in node_map:
        node_map[DATASET_NODE_ID] = {
            "id": DATASET_NODE_ID,
            "data": {"catalogType": "dataset", "label": "Dataset input", "isDataset": True},
        }

    return node_map


def resolve_preview_sampling(
    payload: PipelinePreviewRequest,
    node_map: Dict[str, Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
) -> PreviewSamplingConfig:
    order = execution_order(node_map, graph_edges, payload.target_node_id)
    if not order:
        order = [DATASET_NODE_ID]

    target_node = node_map.get(payload.target_node_id) if payload.target_node_id else None
    target_catalog_type = resolve_catalog_type(target_node) if target_node else ""

    include_preview_rows = bool(payload.include_preview_rows) or target_catalog_type == "data_preview"

    try:
        requested_sample_size = int(payload.sample_size)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        requested_sample_size = 1000
    if requested_sample_size < 0:
        requested_sample_size = 0

    effective_sample_size = requested_sample_size if include_preview_rows else 0
    if include_preview_rows and target_catalog_type == "data_preview" and effective_sample_size <= 0:
        effective_sample_size = DEFAULT_SAMPLE_CAP

    metrics_requested_sample_size = max(effective_sample_size, 0)
    if effective_sample_size > 0:
        load_mode: Literal["auto", "sample", "full"] = "sample"
    else:
        load_mode = "auto"

    return PreviewSamplingConfig(
        execution_order=order,
        target_catalog_type=target_catalog_type,
        include_preview_rows=include_preview_rows,
        effective_sample_size=effective_sample_size,
        metrics_requested_sample_size=metrics_requested_sample_size,
        load_mode=load_mode,
    )


async def load_preview_dataset(
    session: AsyncSession,
    dataset_source_id: str,
    sampling: PreviewSamplingConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any], int, int, int]:
    frame, preview_meta = await load_dataset_frame(
        session,
        dataset_source_id,
        sample_size=sampling.effective_sample_size,
        execution_mode=sampling.load_mode,
        allow_empty_sample=not sampling.include_preview_rows,
    )

    preview_rows = int(frame.shape[0])
    preview_total_rows = coerce_int(
        preview_meta.get("total_rows") if isinstance(preview_meta, dict) else None,
        preview_rows if sampling.include_preview_rows else 0,
    )
    metrics_requested_sample_size = coerce_int(
        preview_meta.get("sample_size") if isinstance(preview_meta, dict) else None,
        sampling.metrics_requested_sample_size if sampling.metrics_requested_sample_size > 0 else preview_rows,
    )

    return frame, preview_meta, preview_rows, preview_total_rows, metrics_requested_sample_size


def build_full_preview_signal(
    working_frame: pd.DataFrame,
    applied_steps: List[str],
    dataset_source_id: str,
    preview_total_rows: int,
) -> Tuple[FullExecutionSignal, int]:
    total_rows_actual = preview_total_rows if preview_total_rows > 0 else int(working_frame.shape[0])
    signal = FullExecutionSignal(
        status="succeeded",
        reason="Preview executed against full dataset.",
        total_rows=total_rows_actual,
        processed_rows=int(working_frame.shape[0]),
        applied_steps=list(applied_steps),
        dataset_source_id=dataset_source_id,
        last_updated=utcnow(),
    )
    return signal, total_rows_actual


def should_defer_full_execution(total_rows_estimate: Optional[int]) -> bool:
    return bool(total_rows_estimate and total_rows_estimate > FULL_DATASET_EXECUTION_ROW_LIMIT)


def _log_background_exception(task: asyncio.Task) -> None:  # pragma: no cover - background logging
    try:
        task.result()
    except Exception:
        logger.exception("Full dataset background execution task failed")


async def _run_full_execution_job(job: FullExecutionJob) -> None:
    await full_execution_job_store.mark_running(
        job.id,
        reason="Full dataset execution running in background.",
        poll_after_seconds=4,
    )
    try:
        session_factory = db_engine.async_session_factory
        if session_factory is None:
            raise RuntimeError("Async session factory is not initialized")

        async with session_factory() as job_session:
            full_frame, full_meta = await load_dataset_frame(
                job_session,
                job.dataset_source_id,
                sample_size=0,
                execution_mode="full",
            )

        # Generate pipeline ID for transformer storage
        pipeline_id = generate_pipeline_id(job.dataset_source_id, job.graph_nodes, job.graph_edges)

        _, applied_steps, _, _ = run_pipeline_execution(
            full_frame,
            job.execution_order,
            job.node_map,
            pipeline_id=pipeline_id,
            collect_signals=False,
        )

        processed_rows = int(full_frame.shape[0])
        total_rows_full = coerce_int(full_meta.get("total_rows"), processed_rows)

        await full_execution_job_store.mark_completed(
            job.id,
            status="succeeded",
            signal_updates={
                "status": "succeeded",
                "reason": "Full dataset execution completed in background.",
                "processed_rows": processed_rows,
                "total_rows": total_rows_full,
                "applied_steps": applied_steps,
                "warnings": [],
                "poll_after_seconds": None,
            },
        )
    except MemoryError:
        await full_execution_job_store.mark_completed(
            job.id,
            status="failed",
            signal_updates={
                "status": "failed",
                "reason": "Full dataset execution failed due to insufficient memory.",
                "warnings": ["memory_error"],
                "poll_after_seconds": None,
            },
        )
    except Exception as exc:
        logger.exception(
            "Full dataset background execution failed for dataset %s (job %s)",
            job.dataset_source_id,
            job.id,
        )
        await full_execution_job_store.mark_completed(
            job.id,
            status="failed",
            signal_updates={
                "status": "failed",
                "reason": f"Full dataset execution failed: {exc}",
                "warnings": ["background_failure"],
                "poll_after_seconds": None,
            },
        )


def schedule_full_execution_job(job: FullExecutionJob) -> None:
    if job.task and not job.task.done():
        return
    task = asyncio.create_task(_run_full_execution_job(job))
    job.task = task
    task.add_done_callback(_log_background_exception)


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
    defer_reason = (
        f"Full dataset execution deferred – {total_rows_estimate:,} rows exceeds limit "
        f"{FULL_DATASET_EXECUTION_ROW_LIMIT:,}."
    )
    append_unique_step(applied_steps, defer_reason)

    job, job_signal, _ = await full_execution_job_store.ensure_job(
        dataset_source_id=dataset_source_id,
        execution_order=execution_order,
        node_map=node_map,
        target_node_id=payload.target_node_id,
        total_rows_estimate=total_rows_estimate,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        defer_reason=defer_reason,
    )

    if job_signal.reason and job_signal.reason.strip() != defer_reason.strip():
        append_unique_step(applied_steps, job_signal.reason)

    schedule_full_execution_job(job)
    return job_signal, preview_total_rows


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
    full_frame, full_meta = await load_dataset_frame(
        session,
        dataset_source_id,
        sample_size=0,
        execution_mode="full",
    )

    _, full_applied_steps, _, _ = run_pipeline_execution(
        full_frame,
        execution_order,
        node_map,
        pipeline_id=pipeline_id,
        collect_signals=False,
    )

    processed_rows = int(full_frame.shape[0])
    total_rows_full = coerce_int(full_meta.get("total_rows"), processed_rows)
    signal = FullExecutionSignal(
        status="succeeded",
        total_rows=total_rows_full,
        processed_rows=processed_rows,
        applied_steps=full_applied_steps,
        dataset_source_id=dataset_source_id,
        last_updated=utcnow(),
    )

    if total_rows_full:
        preview_total_rows = total_rows_full

    append_unique_step(applied_steps, f"Full dataset run processed {processed_rows:,} row(s).")
    return signal, preview_total_rows


def build_failed_full_execution_signal(
    *,
    status: FullExecutionJobStatus,
    reason: str,
    dataset_source_id: str,
    total_rows_estimate: Optional[int],
    warnings: Optional[List[str]] = None,
) -> FullExecutionSignal:
    if status == "succeeded":
        signal_status: Literal["succeeded", "deferred", "skipped", "failed"] = "succeeded"
    elif status in {"queued", "running"}:
        signal_status = "deferred"
    elif status == "cancelled":
        signal_status = "skipped"
    else:
        signal_status = "failed"

    return FullExecutionSignal(
        status=signal_status,
        reason=reason,
        total_rows=total_rows_estimate,
        warnings=warnings or [],
        dataset_source_id=dataset_source_id,
        last_updated=utcnow(),
    )


async def maybe_collect_full_execution(
    session: AsyncSession,
    include_preview_rows: bool,
    effective_sample_size: int,
    preview_total_rows: int,
    working_frame: pd.DataFrame,
    applied_steps: List[str],
    dataset_source_id: str,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    pipeline_id: str,
    payload: PipelinePreviewRequest,
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
) -> Tuple[Optional[FullExecutionSignal], int]:
    if not include_preview_rows:
        return None, preview_total_rows

    dataset_already_full = effective_sample_size == 0
    if dataset_already_full:
        signal, updated_total_rows = build_full_preview_signal(
            working_frame,
            applied_steps,
            dataset_source_id,
            preview_total_rows,
        )
        return signal, updated_total_rows

    total_rows_estimate = preview_total_rows if preview_total_rows > 0 else None
    if should_defer_full_execution(total_rows_estimate):
        signal, updated_total_rows = await defer_full_execution_for_limit(
            applied_steps=applied_steps,
            dataset_source_id=dataset_source_id,
            execution_order=execution_order,
            node_map=node_map,
            payload=payload,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            total_rows_estimate=total_rows_estimate or 0,
            preview_total_rows=preview_total_rows,
        )
        return signal, updated_total_rows

    try:
        signal, updated_total_rows = await run_full_dataset_execution(
            session=session,
            dataset_source_id=dataset_source_id,
            execution_order=execution_order,
            node_map=node_map,
            pipeline_id=pipeline_id,
            applied_steps=applied_steps,
            preview_total_rows=preview_total_rows,
        )
        return signal, updated_total_rows
    except MemoryError:
        reason = "Full dataset execution failed due to insufficient memory."
        signal = build_failed_full_execution_signal(
            status="failed",
            reason=reason,
            dataset_source_id=dataset_source_id,
            total_rows_estimate=total_rows_estimate,
            warnings=["memory_error"],
        )
        append_unique_step(applied_steps, reason)
        return signal, preview_total_rows
    except Exception as exc:
        message = f"Full dataset execution failed: {exc}"
        signal = build_failed_full_execution_signal(
            status="failed",
            reason=str(exc),
            dataset_source_id=dataset_source_id,
            total_rows_estimate=total_rows_estimate,
        )
        append_unique_step(applied_steps, message)
        return signal, preview_total_rows
