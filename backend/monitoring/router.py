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
from datetime import datetime, timedelta, timezone
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

logger = logging.getLogger(__name__)
from backend.config import get_settings
from backend.dependencies import get_db
from backend.database.models import (
    BasicTrainingJob,
    AdvancedTuningJob,
    DriftCheckResult,
    ErrorEvent,
)
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


class ErrorEventResponse(BaseModel):
    id: int
    route: str
    error_type: str
    message: str
    traceback: Optional[str] = None
    job_id: Optional[str] = None
    status_code: int
    created_at: Optional[str] = None
    resolved_at: Optional[str] = None


class ErrorCountResponse(BaseModel):
    count: int


class ErrorDeleteResponse(BaseModel):
    deleted: int


class ErrorGroupedEntry(BaseModel):
    error_type: str
    route: str
    count: int
    last_seen: Optional[str] = None
    first_seen: Optional[str] = None
    sample_id: Optional[int] = None


class ErrorTimelineEntry(BaseModel):
    hour: str
    count: int


@router.get("/errors", response_model=List[ErrorEventResponse])
async def list_error_events(
    limit: int = 100,
    since: Optional[str] = None,
    show_resolved: bool = False,
    db: AsyncSession = Depends(get_db),
) -> List[ErrorEventResponse]:
    """Return the most recent error events (newest first, max 500).

    By default only unresolved events are returned. Pass ``show_resolved=true``
    to include resolved/dismissed events. Pass ``since`` as an ISO-8601 datetime
    string to filter to events after that point.
    """
    from sqlalchemy import and_

    limit = min(max(1, limit), 500)
    filters = []
    if not show_resolved:
        filters.append(ErrorEvent.resolved_at.is_(None))
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            filters.append(ErrorEvent.created_at >= since_dt)
        except ValueError:
            pass  # ignore malformed since param

    stmt = (
        select(ErrorEvent)
        .where(and_(*filters) if filters else True)  # ty: ignore[invalid-argument-type]
        .order_by(ErrorEvent.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [ErrorEventResponse(**e.to_dict()) for e in result.scalars().all()]


@router.get("/errors/count", response_model=ErrorCountResponse)
async def get_error_count(
    db: AsyncSession = Depends(get_db),
) -> ErrorCountResponse:
    """Return unresolved error count — used for the sidebar badge."""
    from sqlalchemy import func

    stmt = select(func.count()).select_from(ErrorEvent).where(ErrorEvent.resolved_at.is_(None))
    result = await db.execute(stmt)
    count = result.scalar() or 0
    return ErrorCountResponse(count=int(count))


@router.delete("/errors", response_model=ErrorDeleteResponse)
async def clear_error_events(
    db: AsyncSession = Depends(get_db),
) -> ErrorDeleteResponse:
    """Delete all stored error events (admin cleanup)."""
    from sqlalchemy import delete

    result = await db.execute(delete(ErrorEvent))
    await db.commit()
    return ErrorDeleteResponse(deleted=result.rowcount or 0)


@router.get("/errors/grouped", response_model=List[ErrorGroupedEntry])
async def get_errors_grouped(
    db: AsyncSession = Depends(get_db),
) -> List[ErrorGroupedEntry]:
    """Aggregate error events by (error_type, route) — unresolved only."""
    from sqlalchemy import func as sa_func

    stmt = (
        select(
            ErrorEvent.error_type,
            ErrorEvent.route,
            sa_func.count(ErrorEvent.id).label("error_count"),
            sa_func.max(ErrorEvent.created_at).label("last_seen"),
            sa_func.min(ErrorEvent.created_at).label("first_seen"),
            sa_func.min(ErrorEvent.id).label("sample_id"),
        )
        .where(ErrorEvent.resolved_at.is_(None))
        .group_by(ErrorEvent.error_type, ErrorEvent.route)
        .order_by(sa_func.count(ErrorEvent.id).desc())
    )
    result = await db.execute(stmt)
    rows = result.all()
    return [
        ErrorGroupedEntry(
            error_type=r.error_type,
            route=r.route,
            count=int(r.error_count),
            last_seen=r.last_seen.isoformat() if r.last_seen else None,
            first_seen=r.first_seen.isoformat() if r.first_seen else None,
            sample_id=r.sample_id,
        )
        for r in rows
    ]


@router.get("/errors/timeline", response_model=List[ErrorTimelineEntry])
async def get_error_timeline(
    hours: int = 24,
    db: AsyncSession = Depends(get_db),
) -> List[ErrorTimelineEntry]:
    """Return error count bucketed by hour for the last N hours.

    Returns a list of ``{ hour: <ISO string>, count: N }`` entries,
    one per hour slot, oldest first. Slots with zero events are included
    so the chart always has a complete x-axis.
    """
    hours = min(max(1, hours), 168)  # cap at 7 days
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    stmt = select(ErrorEvent.created_at).where(ErrorEvent.created_at >= cutoff)
    result = await db.execute(stmt)
    timestamps = [row[0] for row in result.all()]

    # Build a zero-filled bucket dict: { slot_iso: count }
    buckets: Dict[str, int] = {}
    for i in range(hours):
        slot = (cutoff + timedelta(hours=i)).replace(minute=0, second=0, microsecond=0)
        buckets[slot.strftime("%Y-%m-%dT%H:00")] = 0

    for ts in timestamps:
        if ts is None:
            continue
        # Normalise to UTC-aware
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        slot_key = ts.strftime("%Y-%m-%dT%H:00")
        if slot_key in buckets:
            buckets[slot_key] += 1

    return [ErrorTimelineEntry(hour=h, count=c) for h, c in sorted(buckets.items())]


@router.patch("/errors/{error_id}/resolve", response_model=ErrorEventResponse)
async def resolve_error_event(
    error_id: int,
    db: AsyncSession = Depends(get_db),
) -> ErrorEventResponse:
    """Mark an error event as resolved/dismissed."""
    stmt = select(ErrorEvent).where(ErrorEvent.id == error_id)
    result = await db.execute(stmt)
    event = result.scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail=f"ErrorEvent {error_id} not found")
    event.resolved_at = datetime.now(timezone.utc)
    await db.commit()
    return ErrorEventResponse(**event.to_dict())


@router.patch("/errors/{error_id}/unresolve", response_model=ErrorEventResponse)
async def unresolve_error_event(
    error_id: int,
    db: AsyncSession = Depends(get_db),
) -> ErrorEventResponse:
    """Clear the resolved flag on an error event."""
    stmt = select(ErrorEvent).where(ErrorEvent.id == error_id)
    result = await db.execute(stmt)
    event = result.scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail=f"ErrorEvent {error_id} not found")
    event.resolved_at = None
    await db.commit()
    return ErrorEventResponse(**event.to_dict())


@router.get("/errors/{error_id}", response_model=ErrorEventResponse)
async def get_error_event(
    error_id: int,
    db: AsyncSession = Depends(get_db),
) -> ErrorEventResponse:
    """Return full detail for a single error event, including full traceback."""
    stmt = select(ErrorEvent).where(ErrorEvent.id == error_id)
    result = await db.execute(stmt)
    event = result.scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail=f"ErrorEvent {error_id} not found")
    return ErrorEventResponse(**event.to_dict())


# ---------------------------------------------------------------------------
# Workspace-wide slow-node telemetry
# ---------------------------------------------------------------------------
# Reads `metrics.node_timings` (written by JobStrategy.handle_success) off
# every completed job in the lookback window and aggregates per `step_type`.
# Surfaces the same numbers the engine already collects, no extra
# instrumentation required. Legacy jobs without `node_timings` are simply
# skipped — the page just shows fewer entries until enough new runs land.


class SlowNodeAggregate(BaseModel):
    step_type: str
    count: int
    total_seconds: float
    avg_seconds: float
    p95_seconds: float
    max_seconds: float
    sample_node_id: Optional[str] = None


class SlowNodesResponse(BaseModel):
    days: int
    total_jobs_scanned: int
    total_node_runs: int
    aggregates: List[SlowNodeAggregate]


def _percentile(values: List[float], pct: float) -> float:
    """Cheap nearest-rank percentile — avoids pulling numpy into the route."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[rank]


@router.get("/slow-nodes", response_model=SlowNodesResponse)
async def list_slow_nodes(
    days: int = 7,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> SlowNodesResponse:
    """Aggregate per-step execution time across completed jobs in the window.

    Returns the top `limit` step_types sorted by total cumulative seconds —
    the most useful "where to invest in optimisation" signal.
    """
    days = max(1, min(days, 90))
    limit = max(1, min(limit, 50))

    cutoff = datetime.now() - timedelta(days=days)

    by_step: Dict[str, List[float]] = {}
    sample_node: Dict[str, str] = {}
    jobs_scanned = 0
    runs_seen = 0

    # Scan both job tables — same metrics shape, different rows.
    for model in (BasicTrainingJob, AdvancedTuningJob):
        stmt = select(model).where(
            model.status == "completed",
            model.finished_at.isnot(None),
            model.finished_at >= cutoff,
        )
        result = await db.execute(stmt)
        for job in result.scalars().all():
            jobs_scanned += 1
            metrics = job.metrics or {}
            timings = metrics.get("node_timings") if isinstance(metrics, dict) else None
            if not isinstance(timings, list):
                continue
            for entry in timings:
                if not isinstance(entry, dict):
                    continue
                step = str(entry.get("step_type") or "unknown")
                try:
                    secs = float(entry.get("execution_time") or 0.0)
                except (TypeError, ValueError):
                    continue
                if secs <= 0:
                    continue
                by_step.setdefault(step, []).append(secs)
                sample_node.setdefault(step, str(entry.get("node_id") or ""))
                runs_seen += 1

    aggregates: List[SlowNodeAggregate] = []
    for step, values in by_step.items():
        total = sum(values)
        aggregates.append(
            SlowNodeAggregate(
                step_type=step,
                count=len(values),
                total_seconds=round(total, 4),
                avg_seconds=round(total / len(values), 4),
                p95_seconds=round(_percentile(values, 95), 4),
                max_seconds=round(max(values), 4),
                sample_node_id=sample_node.get(step) or None,
            )
        )

    aggregates.sort(key=lambda a: a.total_seconds, reverse=True)
    return SlowNodesResponse(
        days=days,
        total_jobs_scanned=jobs_scanned,
        total_node_runs=runs_seen,
        aggregates=aggregates[:limit],
    )
