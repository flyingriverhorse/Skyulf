import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import joblib
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.engine import get_async_session
from core.database.models import TrainingJob
from core.feature_engineering.export import export_project_bundle
from core.feature_engineering.execution.data import load_dataset_frame
from core.feature_engineering.execution.engine import collect_pipeline_signals
from core.feature_engineering.execution.preview import build_preview_node_map
from core.feature_engineering.execution.graph import execution_order as resolve_execution_order
from core.feature_engineering.modeling.training.jobs import (
    create_training_job as create_training_job_record,
)
from core.feature_engineering.modeling.training.jobs import (
    get_training_job as fetch_training_job,
)
from core.feature_engineering.modeling.training.jobs import (
    list_training_jobs as fetch_training_jobs,
)
from core.feature_engineering.modeling.training.jobs import update_job_status
from core.feature_engineering.modeling.training.tasks import (
    _prepare_training_data,
    _resolve_training_inputs,
    dispatch_training_job,
)
from core.feature_engineering.modeling.training.evaluation import (
    build_classification_split_report,
    build_regression_split_report,
)
from core.feature_engineering.schemas import (
    ModelEvaluationReport,
    ModelEvaluationRequest,
    TrainingJobBatchResponse,
    TrainingJobCreate,
    TrainingJobListResponse,
    TrainingJobResponse,
    TrainingJobStatus,
    TrainingJobSummary,
)
from core.utils.datetime import utcnow

logger = logging.getLogger(__name__)

class TrainingJobExportRequest(BaseModel):
    sample_size: Optional[int] = 100
    project_name: str
    project_description: Optional[str] = None

def _slugify(text: str) -> str:
    return text.lower().replace(" ", "-")

router = APIRouter()


@router.post(
    "/training-jobs",
    response_model=TrainingJobBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_training_job(
    payload: TrainingJobCreate,
    session: AsyncSession = Depends(get_async_session),
) -> TrainingJobBatchResponse:
    """Create one or more background training jobs and optionally dispatch them to Celery."""

    created_jobs: List[TrainingJob] = []

    try:
        for model_type in payload.model_types:
            scoped_payload = payload.model_copy(update={"model_types": [model_type]})
            job = await create_training_job_record(
                session,
                scoped_payload,
                user_id=None,
                model_type_override=model_type,
            )
            created_jobs.append(job)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if payload.run_training:
        for job in created_jobs:
            try:
                dispatch_training_job(str(job.id))
            except Exception as exc:  # pragma: no cover - Celery connection issues
                await update_job_status(
                    session,
                    job,
                    status=TrainingJobStatus.FAILED,
                    error_message="Failed to enqueue training job",
                )
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to enqueue training job",
                ) from exc

    job_payloads = [TrainingJobResponse.model_validate(job, from_attributes=True) for job in created_jobs]
    return TrainingJobBatchResponse(jobs=job_payloads)


@router.get(
    "/training-jobs/{job_id}",
    response_model=TrainingJobResponse,
)
async def get_training_job_detail(
    job_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> TrainingJobResponse:
    """Return a single training job (no authentication required)."""

    job = await fetch_training_job(session, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found")

    return TrainingJobResponse.model_validate(job, from_attributes=True)


@router.get(
    "/training-jobs",
    response_model=TrainingJobListResponse,
)
async def list_training_job_records(
    pipeline_id: Optional[str] = Query(default=None),
    node_id: Optional[str] = Query(default=None),
    dataset_source_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
) -> TrainingJobListResponse:
    """Return recent training jobs (no authentication required for viewing)."""

    logger.debug(
        "Listing training jobs (dataset_source_id=%s pipeline_id=%s node_id=%s limit=%s)",
        dataset_source_id,
        pipeline_id,
        node_id,
        limit,
    )

    # Return all jobs since authentication is not required
    jobs = await fetch_training_jobs(
        session,
        user_id=None,
        dataset_source_id=dataset_source_id,
        pipeline_id=pipeline_id,
        node_id=node_id,
        limit=limit,
    )

    summaries = [TrainingJobSummary.model_validate(job, from_attributes=True) for job in jobs]
    return TrainingJobListResponse(jobs=summaries, total=len(summaries))


@router.post(
    "/training-jobs/{job_id}/evaluate",
    response_model=ModelEvaluationReport,
    status_code=status.HTTP_200_OK,
)
async def evaluate_trained_model(
    job_id: str,
    payload: ModelEvaluationRequest = Body(default_factory=ModelEvaluationRequest),
    session: AsyncSession = Depends(get_async_session),
) -> ModelEvaluationReport:
    """Generate diagnostic plots and metrics for a completed training job."""

    job = await fetch_training_job(session, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Training job not found")

    if not job.artifact_uri:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Training job does not have a stored artifact yet. Re-run the job first.",
        )

    artifact_path = Path(job.artifact_uri)
    if not artifact_path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="Model artifact could not be located on disk.",
        )

    try:
        artifact_bundle = joblib.load(artifact_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to load model artifact for job %s", job_id)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load model artifact.",
        ) from exc

    dataset_frame, node_config, _, _ = await _resolve_training_inputs(session, job)

    node_config_map: Dict[str, Any] = node_config if isinstance(node_config, dict) else {}
    job_metadata: Dict[str, Any] = job.job_metadata if isinstance(job.job_metadata, dict) else {}

    def _node_value(key: str) -> Any:
        return node_config_map[key] if key in node_config_map else None

    target_column = _node_value("target_column") or _node_value("targetColumn") or job_metadata.get("target_column")
    if not target_column:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Evaluation node requires a configured target column.",
        )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
        target_meta,
    ) = _prepare_training_data(dataset_frame, target_column)

    problem_type = job_metadata.get("resolved_problem_type")
    if not isinstance(problem_type, str) or problem_type.lower() not in {"classification", "regression"}:
        artifact_problem_type = artifact_bundle.get("problem_type") if isinstance(artifact_bundle, dict) else None
        if isinstance(artifact_problem_type, str) and artifact_problem_type.lower() in {"classification", "regression"}:
            problem_type = artifact_problem_type.lower()
        else:
            problem_type = "classification"
    else:
        problem_type = problem_type.lower()

    raw_splits = payload.splits or ["test"]
    normalized_splits: List[str] = []
    seen: Set[str] = set()
    for entry in raw_splits:
        if entry is None:
            continue
        normalized = str(entry).strip().lower()
        if normalized in {"train", "training"}:
            key = "train"
        elif normalized in {"validation", "valid", "val"}:
            key = "validation"
        elif normalized in {"test", "testing"}:
            key = "test"
        else:
            continue
        if key not in seen:
            seen.add(key)
            normalized_splits.append(key)
    if not normalized_splits:
        normalized_splits = ["test"]

    model = artifact_bundle.get("model") if isinstance(artifact_bundle, dict) else None
    if model is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Model artifact is missing trained estimator.",
        )

    include_confusion = bool(payload.include_confusion)
    include_curves = bool(payload.include_curves)
    include_residuals = bool(payload.include_residuals)
    max_curve_points = payload.max_curve_points or 500
    max_scatter_points = payload.max_scatter_points or 750

    splits_payload: Dict[str, Any] = {}
    label_names = None
    if isinstance(target_meta, dict) and target_meta.get("dtype") == "categorical":
        label_names = target_meta.get("categories")

    split_map = {
        "train": (X_train, y_train),
        "validation": (X_validation, y_validation),
        "test": (X_test, y_test),
    }

    for split_name in normalized_splits:
        features, target = split_map.get(split_name, (None, None))
        if features is None or target is None or features.empty:
            continue

        if problem_type == "classification":
            splits_payload[split_name] = build_classification_split_report(
                model,
                split_name=split_name,
                features=features,
                target=target,
                label_names=label_names,
                include_confusion=include_confusion,
                include_curves=include_curves,
                max_curve_points=max_curve_points,
            )
        else:
            splits_payload[split_name] = build_regression_split_report(
                model,
                split_name=split_name,
                features=features,
                target=target,
                include_residuals=include_residuals,
                max_scatter_points=max_scatter_points,
            )

    report = ModelEvaluationReport(
        job_id=job_id,
        generated_at=utcnow(),
        problem_type=problem_type,
        splits=splits_payload,
        feature_columns=feature_columns,
    )

    # Persist evaluation results to the job record
    if job.metrics is None:
        job.metrics = {}
    
    # Ensure we trigger a change detection for SQLAlchemy JSON field
    metrics_update = dict(job.metrics)
    metrics_update["evaluation"] = report.model_dump(mode="json")
    job.metrics = metrics_update
    
    session.add(job)
    await session.commit()

    return report


@router.post("/training-jobs/{job_id}/export", status_code=201)
async def export_training_job(
    job_id: str,
    payload: TrainingJobExportRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Export a training job artifact bundle."""
    job = await fetch_training_job(session, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Training job not found")

    if not job.artifact_uri or not Path(job.artifact_uri).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Artifact not found")

    # Load dataset frame
    frame, _ = await load_dataset_frame(
        session,
        job.dataset_source_id,
        sample_size=payload.sample_size,
        execution_mode="sample",
        allow_empty_sample=False,
    )

    # Reconstruct graph execution
    graph = job.graph if isinstance(job.graph, dict) else {}
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    node_map = build_preview_node_map(nodes)
    exec_order = resolve_execution_order(node_map, edges)

    _, signals, modeling_snapshot, applied_steps = collect_pipeline_signals(
        frame,
        exec_order,
        node_map,
        pipeline_id=job.pipeline_id,
        preserve_split_column=True,
    )

    # Create a temporary directory for the export
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Prepare metadata
        if job.job_metadata is None:
            job.job_metadata = {}
        
        meta_update = dict(job.job_metadata)
        meta_update["last_export"] = utcnow().isoformat()
        meta_update["project_name"] = payload.project_name
        meta_update["pipeline_signals"] = signals.model_dump(mode="json") if signals else {}

        result = export_project_bundle(
            artifact_path=job.artifact_uri,
            output_directory=temp_dir,
            job_id=job_id,
            pipeline_id=job.pipeline_id,
            job_metadata=meta_update,
        )
        
        # Persist metadata update
        job.job_metadata = meta_update
        session.add(job)
        await session.commit()
        await session.refresh(job)

        return {
            "project_name": payload.project_name,
            "project_slug": _slugify(payload.project_name),
            "sample_size": payload.sample_size,
            "manifest": result.manifest_payload,
            "warnings": modeling_snapshot.warnings,
            "artefacts": result.artefact_entries,
        }
    except Exception as e:
        logger.exception("Export failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

