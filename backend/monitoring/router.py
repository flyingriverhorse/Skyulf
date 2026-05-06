from backend.exceptions.core import SkyulfException
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import Optional, List, Dict, Any, cast
from pydantic import BaseModel
import polars as pl
import pandas as pd
import io
import logging
import re
from datetime import datetime
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

logger = logging.getLogger(__name__)
from backend.config import get_settings
from backend.dependencies import get_db
from backend.database.models import BasicTrainingJob, AdvancedTuningJob, DriftCheckResult, ErrorEvent
from backend.ml_pipeline._execution.graph_utils import extract_job_details
from skyulf.profiling.drift import DriftCalculator

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


class DriftJobOption(BaseModel):
    job_id: str
    dataset_name: str
    filename: str
    created_at: Optional[str] = None
    model_type: Optional[str] = None
    target_column: Optional[str] = None
    n_features: Optional[int] = None
    n_rows: Optional[int] = None
    description: Optional[str] = None
    best_metric: Optional[str] = None


@router.get("/jobs", response_model=List[DriftJobOption])
async def list_drift_jobs(db: AsyncSession = Depends(get_db)):  # noqa: C901
    """
    List all jobs that have reference data available for drift calculation.
    Scans subdirectories in the artifact folder, enriched with DB metadata.
    """
    settings = get_settings()
    base_path = Path(settings.TRAINING_ARTIFACT_DIR)

    jobs: List[DriftJobOption] = []

    if not base_path.exists():
        return []

    # Collect job IDs found on disk
    found_jobs: List[DriftJobOption] = []

    for item_path in base_path.iterdir():
        if item_path.is_dir():
            folder_name = item_path.name
            created_at_str = "Unknown"

            try:
                match = re.search(r"(\d{8})_(\d{6})", folder_name)
                if match:
                    date_part = match.group(1)
                    time_part = match.group(2)
                    dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
                    created_at_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

            try:
                for file_path in item_path.glob("reference_data_*.joblib"):
                    filename = file_path.name
                    key = file_path.stem
                    remainder = key[15:]  # len("reference_data_")
                    parts = remainder.rsplit("_", 1)
                    if len(parts) == 2:
                        dataset_name = parts[0]
                        job_id = parts[1]
                        found_jobs.append(
                            DriftJobOption(
                                job_id=job_id,
                                dataset_name=dataset_name,
                                filename=filename,
                                created_at=created_at_str,
                            )
                        )
            except Exception:
                continue

    if not found_jobs:
        return []

    # Enrich from database
    job_ids = [j.job_id for j in found_jobs]
    db_jobs: dict[str, BasicTrainingJob | AdvancedTuningJob] = {}
    try:
        for model_cls in (BasicTrainingJob, AdvancedTuningJob):
            stmt = select(model_cls).where(model_cls.id.in_(job_ids))
            result = await db.execute(stmt)
            for row in result.scalars().all():
                db_jobs[str(row.id)] = row
    except Exception:
        logger.warning("Could not enrich drift jobs from DB", exc_info=True)

    for job in found_jobs:
        db_row = db_jobs.get(job.job_id)
        if db_row:
            job.model_type = db_row.model_type

            # Extract target column from graph
            try:
                graph: Dict[str, Any] = cast(Dict[str, Any], db_row.graph or {})
                node_id: str = cast(str, db_row.node_id or "")
                _, target_col, _ = extract_job_details(graph, node_id)
                job.target_column = target_col
            except Exception:
                pass

            # Description from job_metadata
            meta: Dict[str, Any] = cast(Dict[str, Any], db_row.job_metadata or {})
            if isinstance(meta, dict):
                job.description = meta.get("description")

            # Best metric from metrics dict
            metrics: Dict[str, Any] = cast(Dict[str, Any], db_row.metrics or {})
            if isinstance(metrics, dict):
                # Data shape
                if "n_rows" in metrics:
                    job.n_rows = int(metrics["n_rows"])
                if "n_features" in metrics:
                    job.n_features = int(metrics["n_features"])

                # Collect all test metrics into a compact summary
                metric_parts: List[str] = []
                for key, label in [
                    ("test_accuracy", "acc"),
                    ("test_f1_weighted", "f1"),
                    ("test_precision_weighted", "prec"),
                    ("test_recall_weighted", "recall"),
                    ("test_roc_auc", "auc"),
                    ("test_r2", "r2"),
                    ("test_rmse", "rmse"),
                    ("test_mae", "mae"),
                ]:
                    if key in metrics:
                        val = metrics[key]
                        if isinstance(val, (int, float)):
                            metric_parts.append(f"{label}: {val:.4f}")
                if metric_parts:
                    job.best_metric = " | ".join(metric_parts)

        jobs.append(job)

    jobs.sort(key=lambda x: x.created_at or "", reverse=True)
    return jobs


class JobDescriptionUpdate(BaseModel):
    description: str


@router.patch("/jobs/{job_id}/description")
async def update_job_description(
    job_id: str,
    body: JobDescriptionUpdate,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Update a training job's description via job_metadata JSON."""
    for model_cls in (BasicTrainingJob, AdvancedTuningJob):
        stmt = select(model_cls).where(model_cls.id == job_id)
        result = await db.execute(stmt)
        row = result.scalar_one_or_none()
        if row:
            meta_raw: Dict[str, Any] = cast(Dict[str, Any], row.job_metadata or {})
            if not isinstance(meta_raw, dict):
                meta_raw = {}
            meta_raw["description"] = body.description
            row.job_metadata = cast(Any, meta_raw)
            await db.commit()
            return {"status": "ok"}

    raise HTTPException(status_code=404, detail="Job not found")


class EnrichedDriftReport(BaseModel):
    """DriftReport with optional feature importance overlay."""

    reference_rows: int
    current_rows: int
    drifted_columns_count: int
    column_drifts: Dict[str, Any]
    missing_columns: List[str] = []
    new_columns: List[str] = []
    feature_importances: Optional[Dict[str, float]] = None


@router.post("/drift/calculate", response_model=EnrichedDriftReport)
async def calculate_drift(  # noqa: C901  # multi-stage handler: parse → load ref → load curr → compute → persist
    job_id: str = Form(...),
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None),
    threshold_psi: Optional[float] = Form(None),
    threshold_ks: Optional[float] = Form(None),
    threshold_wasserstein: Optional[float] = Form(None),
    threshold_kl: Optional[float] = Form(None),
    db: AsyncSession = Depends(get_db),
) -> EnrichedDriftReport:
    settings = get_settings()
    base_path = Path(settings.TRAINING_ARTIFACT_DIR)

    # 1. Find the job folder
    job_folder = None
    if base_path.exists():
        for item_path in base_path.iterdir():
            if item_path.is_dir() and item_path.name.endswith(f"_{job_id}"):
                job_folder = str(item_path)
                break

    if not job_folder:
        # Fallback: try root if not found in subfolders (backward compatibility)
        job_folder = str(base_path)

    # Use the job folder (or root) as the artifact store base path
    artifact_store = LocalArtifactStore(base_path=job_folder)

    # 2. Find Reference Data
    reference_key = None
    if dataset_name:
        # Sanitize as done in engine.py
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset_name)
        candidate = f"reference_data_{safe_name}_{job_id}"
        if artifact_store.exists(candidate):
            reference_key = candidate

    if not reference_key:
        # Search for reference_data_*{job_id}
        artifacts = artifact_store.list_artifacts()
        for key in artifacts:
            # Check if key matches pattern reference_data_.*_{job_id}
            # Note: list_artifacts returns keys without extension
            if key.startswith("reference_data_") and key.endswith(job_id):
                reference_key = key
                break

    if not reference_key:
        raise HTTPException(status_code=404, detail=f"Reference data not found for job {job_id}")

    # 3. Load Reference Data
    try:
        ref_data = artifact_store.load(reference_key)
        # Convert to Polars
        if isinstance(ref_data, pd.DataFrame):
            ref_df = pl.from_pandas(ref_data)
        else:
            # Assume it's already compatible or fail
            ref_df = pl.DataFrame(ref_data)
    except Exception:
        logger.exception("Failed to load reference data for job %s", job_id)
        raise SkyulfException(message="Failed to load reference data")

    # 3. Load Current Data
    try:
        content = await file.read()
        filename = (file.filename or "").lower()
        if filename.endswith(".csv"):
            curr_df = pl.read_csv(io.BytesIO(content))
        elif filename.endswith(".parquet"):
            curr_df = pl.read_parquet(io.BytesIO(content))
        else:
            # Default to CSV
            curr_df = pl.read_csv(io.BytesIO(content))
    except Exception as e:
        logger.warning("Failed to parse uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Failed to parse uploaded file")

    # 4. Calculate Drift
    try:
        custom_thresholds: Dict[str, float] = {}
        if threshold_psi is not None:
            custom_thresholds["psi"] = threshold_psi
        if threshold_ks is not None:
            custom_thresholds["ks"] = threshold_ks
        if threshold_wasserstein is not None:
            custom_thresholds["wasserstein"] = threshold_wasserstein
        if threshold_kl is not None:
            custom_thresholds["kl_divergence"] = threshold_kl
        calculator = DriftCalculator(ref_df, curr_df)
        report = calculator.calculate_drift(thresholds=custom_thresholds or None)
    except Exception:
        logger.exception("Drift calculation failed for job %s", job_id)
        raise SkyulfException(message="Drift calculation failed")

    # 5. Save drift check result to DB for history
    try:
        # Build per-column summary (PSI + Wasserstein, compact)
        col_summary: Dict[str, Any] = {}
        for col_name, col_drift in report.column_drifts.items():
            metrics_map: Dict[str, float] = {}
            for m in col_drift.metrics:
                metrics_map[m.metric] = m.value
            col_summary[col_name] = {
                "drifted": col_drift.drift_detected,
                "psi": metrics_map.get("psi"),
                "wasserstein": metrics_map.get("wasserstein_distance"),
                "ks_p_value": metrics_map.get("ks_test_p_value"),
            }

        check = DriftCheckResult(
            job_id=job_id,
            dataset_name=dataset_name,
            reference_rows=report.reference_rows,
            current_rows=report.current_rows,
            drifted_columns_count=report.drifted_columns_count,
            total_columns=len(report.column_drifts),
            summary=col_summary,
            column_drifts=report.model_dump(
                exclude={
                    "reference_rows",
                    "current_rows",
                    "drifted_columns_count",
                    "missing_columns",
                    "new_columns",
                }
            ),
        )
        db.add(check)
        await db.commit()
    except Exception:
        logger.warning("Failed to save drift check result", exc_info=True)

    # 6. Load feature importances from training job
    feature_importances: Optional[Dict[str, float]] = None
    try:
        for model_cls in (BasicTrainingJob, AdvancedTuningJob):
            stmt = select(model_cls).where(model_cls.id == job_id)
            result = await db.execute(stmt)
            row = result.scalar_one_or_none()
            if row:
                job_metrics: Dict[str, Any] = cast(Dict[str, Any], row.metrics or {})
                if "feature_importances" in job_metrics:
                    feature_importances = job_metrics["feature_importances"]
                break
    except Exception:
        logger.warning("Could not load feature importances for job %s", job_id)

    return EnrichedDriftReport(
        reference_rows=report.reference_rows,
        current_rows=report.current_rows,
        drifted_columns_count=report.drifted_columns_count,
        column_drifts={k: v.model_dump() for k, v in report.column_drifts.items()},
        missing_columns=report.missing_columns,
        new_columns=report.new_columns,
        feature_importances=feature_importances,
    )


class DriftHistoryEntry(BaseModel):
    id: int
    job_id: str
    dataset_name: Optional[str] = None
    reference_rows: Optional[int] = None
    current_rows: Optional[int] = None
    drifted_columns_count: Optional[int] = None
    total_columns: Optional[int] = None
    summary: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


@router.get("/drift/history/{job_id}", response_model=List[DriftHistoryEntry])
async def get_drift_history(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[DriftHistoryEntry]:
    """Return all drift check results for a given job, newest first."""
    stmt = (
        select(DriftCheckResult)
        .where(DriftCheckResult.job_id == job_id)
        .order_by(DriftCheckResult.created_at.desc())  # type: ignore[union-attr]
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [
        DriftHistoryEntry(
            id=r.id,
            job_id=r.job_id,
            dataset_name=r.dataset_name,
            reference_rows=r.reference_rows,
            current_rows=r.current_rows,
            drifted_columns_count=r.drifted_columns_count,
            total_columns=r.total_columns,
            summary=cast(Optional[Dict[str, Any]], r.summary),
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in rows
    ]


class DriftStatusSummary(BaseModel):
    has_drift: bool
    drifted_jobs: int
    latest_check: Optional[str] = None


@router.get("/drift/status", response_model=DriftStatusSummary)
async def get_drift_status(
    db: AsyncSession = Depends(get_db),
) -> DriftStatusSummary:
    """Return a lightweight summary of whether any recent drift was detected."""
    # Get the latest check per job, check if any have drifted columns
    stmt = (
        select(DriftCheckResult)
        .order_by(DriftCheckResult.created_at.desc())  # type: ignore[union-attr]
        .limit(50)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    if not rows:
        return DriftStatusSummary(has_drift=False, drifted_jobs=0)

    # Deduplicate: keep only the latest check per job_id
    seen_jobs: set[str] = set()
    drifted_count = 0
    latest_check: Optional[str] = None
    for r in rows:
        job_id = r.job_id
        if job_id in seen_jobs:
            continue
        seen_jobs.add(job_id)
        if latest_check is None and r.created_at:
            latest_check = r.created_at.isoformat()
        if cast(int, r.drifted_columns_count or 0) > 0:
            drifted_count += 1

    return DriftStatusSummary(
        has_drift=drifted_count > 0,
        drifted_jobs=drifted_count,
        latest_check=latest_check,
    )


# ---------------------------------------------------------------------------
# In-house error tracker endpoints
# ---------------------------------------------------------------------------


@router.get("/errors")
async def list_error_events(
    limit: int = 100,
    since: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Return the most recent error events (newest first, max 500).

    Pass `since` as an ISO-8601 datetime string to filter to events after that point.
    """
    from sqlalchemy import and_

    limit = min(max(1, limit), 500)
    filters = []
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            filters.append(ErrorEvent.created_at >= since_dt)
        except ValueError:
            pass  # ignore malformed since param

    stmt = (
        select(ErrorEvent)
        .where(and_(*filters) if filters else True)
        .order_by(ErrorEvent.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [e.to_dict() for e in result.scalars().all()]


@router.get("/errors/count")
async def get_error_count(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Return total unresolved error count — used for the sidebar badge."""
    from sqlalchemy import func

    stmt = select(func.count()).select_from(ErrorEvent)
    result = await db.execute(stmt)
    count = result.scalar() or 0
    return {"count": count}


@router.delete("/errors")
async def clear_error_events(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Delete all stored error events (admin cleanup)."""
    from sqlalchemy import delete

    result = await db.execute(delete(ErrorEvent))
    await db.commit()
    return {"deleted": result.rowcount}


@router.get("/errors/{error_id}")
async def get_error_event(
    error_id: int,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Return full detail for a single error event, including full traceback."""
    stmt = select(ErrorEvent).where(ErrorEvent.id == error_id)
    result = await db.execute(stmt)
    event = result.scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail=f"ErrorEvent {error_id} not found")
    return event.to_dict()

