import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.engine import get_async_session
from core.database.models import FeatureEngineeringPipeline
from core.feature_engineering.execution.data import (
    build_eda_service,
    coerce_int,
    load_dataset_frame,
)
from core.feature_engineering.execution.engine import run_pipeline_execution
from core.feature_engineering.execution.graph import (
    determine_node_split_type,
    generate_pipeline_id,
)
from core.feature_engineering.execution.jobs import full_execution_job_store
from core.feature_engineering.execution.preview import (
    append_unique_step,
    build_preview_node_map,
    maybe_collect_full_execution,
    resolve_preview_sampling,
)
from core.feature_engineering.preprocessing.inspection import build_data_snapshot_response
from core.feature_engineering.preprocessing.split import SPLIT_TYPE_COLUMN
from core.feature_engineering.schemas import (
    FeaturePipelineCreate,
    FeaturePipelineResponse,
    FullExecutionSignal,
    PipelinePreviewMetrics,
    PipelinePreviewRequest,
    PipelinePreviewResponse,
    PipelinePreviewRowsResponse,
    PipelinePreviewSignals,
)
from core.utils.datetime import utcnow

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/preview",
    response_model=PipelinePreviewResponse,
)
async def preview_pipeline(
    payload: PipelinePreviewRequest,
    session: AsyncSession = Depends(get_async_session),
) -> PipelinePreviewResponse:
    dataset_source_id = (payload.dataset_source_id or "").strip()
    if not dataset_source_id:
        raise HTTPException(status_code=400, detail="dataset_source_id must not be empty")

    graph_nodes = payload.graph.nodes or []
    graph_edges = payload.graph.edges or []

    node_map = build_preview_node_map(graph_nodes)
    sampling_config = resolve_preview_sampling(payload, node_map, graph_edges)

    # We need to load the preview dataset here because we need 'preview_rows' and 'preview_total_rows'
    # which are returned by load_preview_dataset in execution/preview.py
    # But wait, I didn't export load_preview_dataset in api/pipeline.py imports.
    # I should import it.
    from core.feature_engineering.execution.preview import load_preview_dataset

    frame, preview_meta, preview_rows, preview_total_rows, metrics_requested_sample_size = await load_preview_dataset(
        session,
        dataset_source_id,
        sampling_config,
    )

    # Generate stable pipeline ID from dataset + graph structure
    pipeline_id = generate_pipeline_id(dataset_source_id, graph_nodes, graph_edges)

    collect_signals = bool(payload.include_signals)

    working_frame, applied_steps, preview_signals, modeling_metadata = run_pipeline_execution(
        frame,
        sampling_config.execution_order,
        node_map,
        pipeline_id=pipeline_id,
        collect_signals=collect_signals,
    )

    full_execution_signal, preview_total_rows = await maybe_collect_full_execution(
        session=session,
        include_preview_rows=sampling_config.include_preview_rows,
        effective_sample_size=sampling_config.effective_sample_size,
        preview_total_rows=preview_total_rows,
        working_frame=working_frame,
        applied_steps=applied_steps,
        dataset_source_id=dataset_source_id,
        execution_order=sampling_config.execution_order,
        node_map=node_map,
        pipeline_id=pipeline_id,
        payload=payload,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
    )

    if full_execution_signal and collect_signals:
        if preview_signals is None:
            preview_signals = PipelinePreviewSignals()
        preview_signals.full_execution = full_execution_signal

    baseline_sample_rows = coerce_int(
        preview_meta.get("sample_size") if isinstance(preview_meta, dict) else None,
        metrics_requested_sample_size if metrics_requested_sample_size > 0 else preview_rows,
    )

    # Filter DataFrame by split type if target node is connected to a specific train/test/validation output
    if payload.target_node_id and SPLIT_TYPE_COLUMN in working_frame.columns:
        split_type = determine_node_split_type(payload.target_node_id, graph_edges, node_map)

        if split_type:
            # Filter to show only the requested split
            original_rows = len(working_frame)
            working_frame = working_frame[working_frame[SPLIT_TYPE_COLUMN] == split_type].copy()
            filtered_rows = len(working_frame)

            if filtered_rows < original_rows:
                filter_msg = f"Showing {split_type} split: {filtered_rows:,} of {original_rows:,} rows"
                append_unique_step(applied_steps, filter_msg)

            # Update preview metrics
            preview_rows = filtered_rows
            if preview_total_rows > 0:
                # Estimate total rows for this split based on the ratio
                ratio = filtered_rows / original_rows if original_rows > 0 else 0
                preview_total_rows = int(preview_total_rows * ratio)

    response = build_data_snapshot_response(
        working_frame,
        target_node_id=payload.target_node_id,
        preview_rows=preview_rows,
        preview_total_rows=preview_total_rows,
        initial_sample_rows=baseline_sample_rows,
        applied_steps=applied_steps,
        metrics_requested_sample_size=metrics_requested_sample_size,
        modeling_signals=modeling_metadata,
        signals=preview_signals if collect_signals else None,
        include_signals=collect_signals,
    )

    if not sampling_config.include_preview_rows:
        response.sample_rows = []

    return response


@router.get(
    "/{dataset_source_id}/preview/rows",
    response_model=PipelinePreviewRowsResponse,
)
async def preview_pipeline_rows(
    dataset_source_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    mode: str = Query("head"),
    session: AsyncSession = Depends(get_async_session),
) -> PipelinePreviewRowsResponse:
    dataset_source = (dataset_source_id or "").strip()
    if not dataset_source:
        raise HTTPException(status_code=400, detail="dataset_source_id must not be empty")

    eda_service = build_eda_service(session, limit)
    window_payload = await eda_service.preview_rows_window(
        dataset_source,
        offset=offset,
        limit=limit,
        mode=mode,
    )

    if not window_payload.get("success"):
        detail = window_payload.get("error") or window_payload.get("message") or "Unable to load preview rows"
        raise HTTPException(status_code=400, detail=detail)

    preview = window_payload.get("preview") or {}

    rows = preview.get("rows") or []
    columns = preview.get("columns") or []

    preview_offset = coerce_int(preview.get("offset"), offset)
    preview_limit = coerce_int(preview.get("limit"), limit)
    returned_rows = coerce_int(preview.get("returned_rows"), len(rows))

    raw_total_rows = preview.get("total_rows")
    total_rows_value: Optional[int] = None
    if isinstance(raw_total_rows, (int, float)):
        total_rows_value = coerce_int(raw_total_rows, returned_rows)

    next_offset = preview.get("next_offset")
    if next_offset is not None:
        next_offset = coerce_int(next_offset, preview_offset + returned_rows)

    return PipelinePreviewRowsResponse(
        columns=[str(column) for column in columns],
        rows=rows,
        offset=preview_offset,
        limit=preview_limit,
        returned_rows=returned_rows,
        total_rows=total_rows_value,
        next_offset=next_offset,
        has_more=bool(preview.get("has_more", False)),
        sampling_mode=str(preview.get("mode") or "window"),
        sampling_adjustments=preview.get("sampling_adjustments") or [],
        large_dataset=bool(preview.get("large_dataset", False)),
    )


@router.get(
    "/{dataset_source_id}/full-execution/{job_id}",
    response_model=FullExecutionSignal,
)
async def get_full_execution_status(
    dataset_source_id: str,
    job_id: str,
) -> FullExecutionSignal:
    normalized_dataset = (dataset_source_id or "").strip()
    if not normalized_dataset:
        raise HTTPException(status_code=400, detail="dataset_source_id must not be empty")

    normalized_job_id = (job_id or "").strip()
    if not normalized_job_id:
        raise HTTPException(status_code=400, detail="job_id must not be empty")

    signal = await full_execution_job_store.get_signal(normalized_dataset, normalized_job_id)
    if not signal:
        raise HTTPException(status_code=404, detail="Full dataset execution job not found")
    return signal


def _build_pipeline_response(pipeline: FeatureEngineeringPipeline) -> FeaturePipelineResponse:
    """Coerce a SQLAlchemy pipeline row into a response model without validation failures."""

    graph_payload: Dict[str, Any]
    raw_graph = getattr(pipeline, "graph", None)
    if isinstance(raw_graph, dict):
        graph_payload = raw_graph
    else:
        graph_payload = {"nodes": [], "edges": []}

    metadata_payload = pipeline.pipeline_metadata if isinstance(pipeline.pipeline_metadata, dict) else None

    response_payload = {
        "id": pipeline.id,
        "dataset_source_id": pipeline.dataset_source_id,
        "name": pipeline.name,
        "description": pipeline.description,
        "graph": graph_payload,
        "metadata": metadata_payload,
        "is_active": pipeline.is_active,
        "created_at": pipeline.created_at,
        "updated_at": pipeline.updated_at,
    }

    return FeaturePipelineResponse.model_validate(response_payload)


@router.get(
    "/{dataset_source_id}",
    response_model=Optional[FeaturePipelineResponse],
)
async def get_pipeline(
    dataset_source_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> Optional[FeaturePipelineResponse]:
    """Fetch the latest saved pipeline for a dataset, if one exists."""

    result = await session.execute(
        select(FeatureEngineeringPipeline)
        .where(FeatureEngineeringPipeline.dataset_source_id == dataset_source_id)
        .order_by(FeatureEngineeringPipeline.updated_at.desc())
    )
    pipeline = result.scalars().first()

    return _build_pipeline_response(pipeline) if pipeline else None


@router.get(
    "/{dataset_source_id}/history",
    response_model=List[FeaturePipelineResponse],
)
async def get_pipeline_history(  # pragma: no cover - fastapi route
    dataset_source_id: str,
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_async_session),
) -> List[FeaturePipelineResponse]:
    """Return the most recent pipeline revisions for a dataset."""

    limit_value = max(1, min(limit, 50))

    result = await session.execute(
        select(FeatureEngineeringPipeline)
        .where(FeatureEngineeringPipeline.dataset_source_id == dataset_source_id)
        .order_by(
            FeatureEngineeringPipeline.updated_at.desc(),
            FeatureEngineeringPipeline.id.desc(),
        )
        .limit(limit_value)
    )

    pipelines = result.scalars().all()

    return [_build_pipeline_response(item) for item in pipelines]


@router.post(
    "/{dataset_source_id}",
    response_model=FeaturePipelineResponse,
    status_code=status.HTTP_200_OK,
)
async def upsert_pipeline(
    dataset_source_id: str,
    payload: FeaturePipelineCreate,
    session: AsyncSession = Depends(get_async_session),
) -> FeaturePipelineResponse:
    """Create or update the pipeline associated with a dataset."""

    result = await session.execute(
        select(FeatureEngineeringPipeline)
        .where(FeatureEngineeringPipeline.dataset_source_id == dataset_source_id)
        .order_by(FeatureEngineeringPipeline.updated_at.desc())
    )
    pipeline = result.scalars().first()

    graph_payload = payload.graph.model_dump()

    if pipeline:
        setattr(pipeline, "name", payload.name)
        setattr(pipeline, "description", payload.description)
        setattr(pipeline, "graph", graph_payload)
        setattr(pipeline, "pipeline_metadata", payload.metadata)
    else:
        pipeline = FeatureEngineeringPipeline(
            dataset_source_id=dataset_source_id,
            name=payload.name,
            description=payload.description,
            graph=graph_payload,
            pipeline_metadata=payload.metadata,
        )

    session.add(pipeline)
    await session.commit()
    await session.refresh(pipeline)

    return _build_pipeline_response(pipeline)
