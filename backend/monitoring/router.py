import io
import logging
import re
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any, Literal, cast

import pandas as pd
import polars as pl
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.exceptions.core import SkyulfException
from backend.middleware.rate_limiter import limiter
from backend.ml_pipeline.artifacts.factory import ArtifactFactory

logger = logging.getLogger(__name__)
from backend.database.models import (
    Deployment,
    DriftCheckResult,
    DriftThresholdVersion,
    ErrorEvent,
    PipelineRunLog,
    TrainingJob,
)
from backend.dependencies import get_db
from backend.ml_pipeline._execution.graph_utils import (
    extract_job_details,
    model_registry_tags,
    resolve_model_family,
)
from backend.ml_pipeline._execution.utils import parse_branch_info, resolve_dataset_name
from backend.ml_pipeline.constants import StepType
from skyulf.profiling.drift import DriftCalculator
from skyulf.registry import NodeRegistry as SkyulfRegistry

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


class DriftJobOption(BaseModel):
    job_id: str
    dataset_name: str
    filename: str
    created_at: str | None = None
    model_type: str | None = None
    target_column: str | None = None
    n_features: int | None = None
    n_rows: int | None = None
    description: str | None = None
    best_metric: str | None = None


async def _fetch_drift_job_rows(db: AsyncSession, job_ids: list[str]) -> dict[str, TrainingJob]:
    """Look up training/tuning job rows (either run_mode) for the given ids."""
    db_jobs: dict[str, TrainingJob] = {}
    try:
        stmt = select(TrainingJob).where(TrainingJob.id.in_(job_ids))
        result = await db.execute(stmt)
        for row in result.scalars().all():
            db_jobs[str(row.id)] = row
    except Exception:  # noqa: BLE001 - enrichment failure returns partial result
        logger.warning("Could not enrich drift jobs from DB", exc_info=True)
    return db_jobs


def _extract_drift_target_column(db_row: TrainingJob) -> str | None:
    """Resolve the target column for a job row from its stored graph, if possible."""
    try:
        graph: dict[str, Any] = cast(dict[str, Any], db_row.graph or {})
        node_id: str = cast(str, db_row.node_id or "")
        _, target_col, _ = extract_job_details(graph, node_id)
        return target_col
    except Exception:  # noqa: BLE001 - optional metadata falls back to None
        return None  # nosec B110 - target column is optional metadata; job listing still succeeds


def _build_drift_metric_summary(metrics: dict[str, Any]) -> str | None:
    """Collapse the known test metrics present in `metrics` into a compact `key: value` summary."""
    metric_parts: list[str] = []
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
    return " | ".join(metric_parts) if metric_parts else None


def _enrich_drift_job(job: DriftJobOption, db_row: TrainingJob) -> None:
    """Fill in `job`'s model/target/description/metric fields from its DB row, in place."""
    job.model_type = db_row.model_type
    job.target_column = _extract_drift_target_column(db_row)

    meta: dict[str, Any] = cast(dict[str, Any], db_row.job_metadata or {})
    if isinstance(meta, dict):
        job.description = meta.get("description")

    metrics: dict[str, Any] = cast(dict[str, Any], db_row.metrics or {})
    if isinstance(metrics, dict):
        if "n_rows" in metrics:
            job.n_rows = int(metrics["n_rows"])
        if "n_features" in metrics:
            job.n_features = int(metrics["n_features"])
        job.best_metric = _build_drift_metric_summary(metrics)


@router.get("/jobs", response_model=list[DriftJobOption])
async def list_drift_jobs(db: AsyncSession = Depends(get_db)):
    """
    List all jobs that have reference data available for drift calculation.
    Scans subdirectories in the artifact folder, enriched with DB metadata.
    """
    jobs: list[DriftJobOption] = []

    # Discover reference artifacts via the storage seam (local today; UC/S3-ready).
    found_jobs: list[DriftJobOption] = [
        DriftJobOption(
            job_id=ref.job_id,
            dataset_name=ref.dataset_name,
            filename=ref.filename,
            created_at=ref.created_at or "Unknown",
        )
        for ref in ArtifactFactory.get_discovery().list_reference_artifacts()
    ]

    if not found_jobs:
        return []

    # Enrich from database
    job_ids = [j.job_id for j in found_jobs]
    db_jobs = await _fetch_drift_job_rows(db, job_ids)

    for job in found_jobs:
        db_row = db_jobs.get(job.job_id)
        if db_row:
            _enrich_drift_job(job, db_row)
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
    stmt = select(TrainingJob).where(TrainingJob.id == job_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row:
        meta_raw: dict[str, Any] = cast(dict[str, Any], row.job_metadata or {})
        if not isinstance(meta_raw, dict):
            meta_raw = {}
        meta_raw["description"] = body.description
        row.job_metadata = cast(Any, meta_raw)
        await db.commit()
        return {"status": "ok"}

    raise HTTPException(status_code=404, detail="Job not found")


class EnrichedDriftReport(BaseModel):
    """DriftReport with optional feature importance overlay plus alert identity.

    `alert_id`/`severity`/`threshold_version`/`deployment_id`/`model_version`
    let the UI immediately deep-link the freshly-created alert to its
    evidence, threshold snapshot, and related deployment without a second
    round trip.
    """

    reference_rows: int
    current_rows: int
    drifted_columns_count: int
    column_drifts: dict[str, Any]
    missing_columns: list[str] = []
    new_columns: list[str] = []
    feature_importances: dict[str, float] | None = None
    alert_id: int | None = None
    severity: str = "none"
    threshold_version: int | None = None
    deployment_id: int | None = None
    model_version: str | None = None


def _find_reference_key(artifact_store, dataset_name: str | None, job_id: str) -> str | None:
    """Locate the reference-data artifact key for a job, preferring an exact dataset_name match."""
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

    return reference_key


def _load_reference_dataframe(artifact_store, reference_key: str, job_id: str) -> pl.DataFrame:
    """Load and convert the reference-data artifact to a Polars DataFrame."""
    try:
        ref_data = artifact_store.load(reference_key)
        # Convert to Polars
        if isinstance(ref_data, pl.DataFrame):
            return ref_data
        if isinstance(ref_data, pd.DataFrame):
            return pl.from_pandas(ref_data)
        # Assume it's already compatible or fail
        return pl.DataFrame(ref_data)
    except Exception:
        logger.exception("Failed to load reference data for job %s", job_id)
        raise SkyulfException(message="Failed to load reference data") from None


async def _load_current_dataframe(file: UploadFile) -> pl.DataFrame:
    """Read the uploaded file (bounded by MAX_UPLOAD_SIZE) and parse it as CSV/Parquet into Polars."""
    try:
        from backend.config import get_settings as _get_settings

        _settings = _get_settings()
        _max_size = _settings.MAX_UPLOAD_SIZE
        content = await file.read(_max_size + 1)
        if len(content) > _max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {_max_size // (1024 * 1024)} MB.",
            )
        filename = (file.filename or "").lower()
        if filename.endswith(".parquet"):
            return pl.read_parquet(io.BytesIO(content))
        # Default to CSV (also used for .csv and any other extension)
        return pl.read_csv(io.BytesIO(content))
    except Exception as e:
        logger.warning("Failed to parse uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Failed to parse uploaded file") from e


# Mirrors `DriftCalculator._merge_thresholds`'s defaults (skyulf-core keeps
# that method private) so the backend can record the *effective* threshold
# set — including values the caller didn't override — in the durable
# threshold-version snapshot.
_DEFAULT_DRIFT_THRESHOLDS: dict[str, float] = {
    "psi": 0.2,
    "ks_statistic": 0.1,
    "wasserstein": 0.1,
    "kl_divergence": 0.1,
}

DriftDispositionAction = Literal["acknowledge", "resolve", "reopen"]

# Valid disposition transitions: {action: {allowed current statuses}}. Kept
# explicit so an invalid transition (e.g. resolving a check that was never
# acknowledged) is rejected with a clear error instead of silently applied.
_ALLOWED_DISPOSITION_TRANSITIONS: dict[str, set[str]] = {
    "acknowledge": {"new", "reopened"},
    "resolve": {"acknowledged"},
    "reopen": {"acknowledged", "resolved"},
}


def _build_drift_thresholds(
    threshold_psi: float | None,
    threshold_ks: float | None,
    threshold_wasserstein: float | None,
    threshold_kl: float | None,
) -> dict[str, float]:
    """Assemble the custom drift-metric thresholds dict from the individual per-metric overrides."""
    custom_thresholds: dict[str, float] = {}
    if threshold_psi is not None:
        custom_thresholds["psi"] = threshold_psi
    if threshold_ks is not None:
        custom_thresholds["ks_statistic"] = threshold_ks
    if threshold_wasserstein is not None:
        custom_thresholds["wasserstein"] = threshold_wasserstein
    if threshold_kl is not None:
        custom_thresholds["kl_divergence"] = threshold_kl
    return custom_thresholds


def _effective_drift_thresholds(custom_thresholds: dict[str, float]) -> dict[str, float]:
    """Merge user overrides over the calculator's defaults for durable recording."""
    return {**_DEFAULT_DRIFT_THRESHOLDS, **custom_thresholds}


async def _get_or_create_threshold_version(
    db: AsyncSession, effective_thresholds: dict[str, float]
) -> DriftThresholdVersion:
    """Return the threshold version matching `effective_thresholds`, creating one if needed.

    Comparison rounds to 6 decimals to avoid spurious new versions from
    float noise. A changed threshold set always gets a *new* version number
    rather than mutating the latest row, so alerts already pinned to an
    older version keep pointing at the values they were actually evaluated
    against.
    """
    stmt = select(DriftThresholdVersion).order_by(DriftThresholdVersion.version.desc()).limit(1)
    result = await db.execute(stmt)
    latest = result.scalar_one_or_none()

    def _matches(row: DriftThresholdVersion) -> bool:
        return (
            round(row.psi, 6) == round(effective_thresholds["psi"], 6)
            and round(row.ks, 6) == round(effective_thresholds["ks_statistic"], 6)
            and round(row.wasserstein, 6) == round(effective_thresholds["wasserstein"], 6)
            and round(row.kl_divergence, 6) == round(effective_thresholds["kl_divergence"], 6)
        )

    if latest is not None and _matches(latest):
        return latest

    next_version = (latest.version + 1) if latest is not None else 1
    new_version = DriftThresholdVersion(
        version=next_version,
        psi=effective_thresholds["psi"],
        ks=effective_thresholds["ks_statistic"],
        wasserstein=effective_thresholds["wasserstein"],
        kl_divergence=effective_thresholds["kl_divergence"],
    )
    db.add(new_version)
    await db.flush()
    return new_version


def _classify_drift_severity(report) -> str:
    """Derive a triage severity from the report — schema drift is always critical.

    A structural change (columns appearing/disappearing) breaks downstream
    assumptions regardless of how many value distributions also drifted, so
    it always classifies as critical. Otherwise severity scales with the
    fraction of drifted feature columns.
    """
    if report.missing_columns or report.new_columns:
        return "critical"
    total = len(report.column_drifts)
    if total == 0 or report.drifted_columns_count == 0:
        return "none"
    ratio = report.drifted_columns_count / total
    return "critical" if ratio > 0.3 else "warning"


def _build_drift_column_summary(report) -> dict[str, Any]:
    """Build a compact per-column drift summary (drifted flag + PSI/Wasserstein/KS p-value)."""
    col_summary: dict[str, Any] = {}
    for col_name, col_drift in report.column_drifts.items():
        metrics_map: dict[str, float] = {}
        for m in col_drift.metrics:
            metrics_map[m.metric] = m.value
        col_summary[col_name] = {
            "drifted": col_drift.drift_detected,
            "psi": metrics_map.get("psi"),
            "wasserstein": metrics_map.get("wasserstein_distance"),
            "ks_statistic": metrics_map.get("ks_statistic"),
            "ks_p_value": metrics_map.get("ks_test_p_value"),
        }
    return col_summary


async def _find_deployment_context(db: AsyncSession, job_id: str) -> tuple[int | None, str | None]:
    """Resolve the active deployment id and model-version label for a job, if any.

    Returns `(None, None)` when the job has never been deployed — the drift
    alert still records its evidence, just without a deployment link.
    """
    try:
        stmt = select(Deployment).where(Deployment.job_id == job_id, Deployment.is_active)
        result = await db.execute(stmt)
        deployment = result.scalar_one_or_none()

        job_stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        job_result = await db.execute(job_stmt)
        job_row = job_result.scalar_one_or_none()
        model_version = f"v{job_row.version}" if job_row is not None else None

        return (deployment.id if deployment is not None else None), model_version
    except Exception:  # noqa: BLE001 - deployment context is optional
        logger.warning("Could not resolve deployment context for job %s", job_id, exc_info=True)
        return None, None


async def _save_drift_alert(
    db: AsyncSession,
    *,
    job_id: str,
    dataset_name: str | None,
    evaluation_status: str,
    report=None,
    effective_thresholds: dict[str, float] | None = None,
    threshold_version: DriftThresholdVersion | None = None,
    deployment_id: int | None = None,
    model_version: str | None = None,
    error_message: str | None = None,
) -> DriftCheckResult | None:
    """Persist a durable drift alert row and return it; failures are logged but non-fatal.

    Covers every evaluation outcome explicitly: a completed report (with its
    evidence and severity), a missing reference baseline
    (`evaluation_status="no_baseline"`), and a calculation failure
    (`evaluation_status="failed"`, with `error_message`) all get a row so
    the history never conflates "nothing to show" with "we don't know".
    """
    try:
        check = DriftCheckResult(
            job_id=job_id,
            dataset_name=dataset_name,
            evaluation_status=evaluation_status,
            error_message=error_message,
            deployment_id=deployment_id,
            model_version=model_version,
            status="new",
            severity="none",
        )

        if report is not None:
            col_summary = _build_drift_column_summary(report)
            check.reference_rows = report.reference_rows
            check.current_rows = report.current_rows
            check.drifted_columns_count = report.drifted_columns_count
            check.total_columns = len(report.column_drifts)
            check.summary = col_summary
            # Flat `{column: ColumnDrift-dump}` — matches the shape returned by
            # `EnrichedDriftReport.column_drifts` so the alert-detail endpoint's
            # evidence exactly mirrors what the user saw at evaluation time.
            check.column_drifts = {k: v.model_dump() for k, v in report.column_drifts.items()}
            check.severity = _classify_drift_severity(report)

        if effective_thresholds is not None:
            check.threshold_psi = effective_thresholds["psi"]
            check.threshold_ks = effective_thresholds["ks_statistic"]
            check.threshold_wasserstein = effective_thresholds["wasserstein"]
            check.threshold_kl = effective_thresholds["kl_divergence"]
        if threshold_version is not None:
            check.threshold_version = threshold_version.version

        db.add(check)
        await db.commit()
        await db.refresh(check)
        return check
    except Exception:  # noqa: BLE001 - alert persistence is non-fatal
        logger.warning("Failed to save drift alert", exc_info=True)
        await db.rollback()
        return None


async def _load_feature_importances(db: AsyncSession, job_id: str) -> dict[str, float] | None:
    """Look up feature importances recorded on the training job's metrics, if any."""
    feature_importances: dict[str, float] | None = None
    try:
        stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await db.execute(stmt)
        row = result.scalar_one_or_none()
        if row:
            job_metrics: dict[str, Any] = cast(dict[str, Any], row.metrics or {})
            if "feature_importances" in job_metrics:
                feature_importances = job_metrics["feature_importances"]
    except Exception:  # noqa: BLE001 - feature importances are optional
        logger.warning("Could not load feature importances for job %s", job_id)
    return feature_importances


@router.post("/drift/calculate", response_model=EnrichedDriftReport)
@limiter.limit("20/minute")
async def calculate_drift(
    request: Request,
    job_id: str = Form(...),
    file: UploadFile = File(...),
    dataset_name: str | None = Form(None),
    threshold_psi: float | None = Form(None),
    threshold_ks: float | None = Form(None),
    threshold_wasserstein: float | None = Form(None),
    threshold_kl: float | None = Form(None),
    db: AsyncSession = Depends(get_db),
) -> EnrichedDriftReport:
    # 1. Find the job folder (via the storage seam) and its artifact store.
    artifact_store = ArtifactFactory.get_discovery().get_store_for_job(job_id)

    deployment_id, model_version = await _find_deployment_context(db, job_id)

    # 2. Find Reference Data — no baseline is an explicit, recorded outcome
    # (not a silent 404), so the alert history can distinguish "never
    # checked" from "checked, but there was nothing to compare against".
    reference_key = _find_reference_key(artifact_store, dataset_name, job_id)
    if not reference_key:
        await _save_drift_alert(
            db,
            job_id=job_id,
            dataset_name=dataset_name,
            evaluation_status="no_baseline",
            deployment_id=deployment_id,
            model_version=model_version,
            error_message=f"Reference data not found for job {job_id}",
        )
        raise HTTPException(status_code=404, detail=f"Reference data not found for job {job_id}")

    # 3. Load Reference Data
    ref_df = _load_reference_dataframe(artifact_store, reference_key, job_id)

    # 3. Load Current Data
    curr_df = await _load_current_dataframe(file)

    # 4. Calculate Drift
    custom_thresholds = _build_drift_thresholds(
        threshold_psi, threshold_ks, threshold_wasserstein, threshold_kl
    )
    effective_thresholds = _effective_drift_thresholds(custom_thresholds)
    try:
        calculator = DriftCalculator(ref_df, curr_df)
        report = calculator.calculate_drift(thresholds=custom_thresholds or None)
    except Exception as exc:
        logger.exception("Drift calculation failed for job %s", job_id)
        await _save_drift_alert(
            db,
            job_id=job_id,
            dataset_name=dataset_name,
            evaluation_status="failed",
            deployment_id=deployment_id,
            model_version=model_version,
            error_message=str(exc),
        )
        raise SkyulfException(message="Drift calculation failed") from None

    # 5. Pin the threshold version this check was evaluated against, then
    # save the durable alert row for history/lifecycle tracking.
    threshold_version = await _get_or_create_threshold_version(db, effective_thresholds)
    alert = await _save_drift_alert(
        db,
        job_id=job_id,
        dataset_name=dataset_name,
        evaluation_status="completed",
        report=report,
        effective_thresholds=effective_thresholds,
        threshold_version=threshold_version,
        deployment_id=deployment_id,
        model_version=model_version,
    )

    # 6. Load feature importances from training job
    feature_importances = await _load_feature_importances(db, job_id)

    return EnrichedDriftReport(
        reference_rows=report.reference_rows,
        current_rows=report.current_rows,
        drifted_columns_count=report.drifted_columns_count,
        column_drifts={k: v.model_dump() for k, v in report.column_drifts.items()},
        missing_columns=report.missing_columns,
        new_columns=report.new_columns,
        feature_importances=feature_importances,
        alert_id=alert.id if alert is not None else None,
        severity=alert.severity if alert is not None else _classify_drift_severity(report),
        threshold_version=threshold_version.version,
        deployment_id=deployment_id,
        model_version=model_version,
    )


class DriftHistoryEntry(BaseModel):
    """One row in a job's drift history — an alert's identity, evidence summary, and lifecycle."""

    id: int
    job_id: str
    dataset_name: str | None = None
    reference_rows: int | None = None
    current_rows: int | None = None
    drifted_columns_count: int | None = None
    total_columns: int | None = None
    summary: dict[str, Any] | None = None
    created_at: str | None = None
    severity: str = "none"
    status: str = "new"
    owner: str | None = None
    acknowledged_at: str | None = None
    resolved_at: str | None = None
    threshold_version: int | None = None
    threshold_psi: float | None = None
    threshold_ks: float | None = None
    threshold_wasserstein: float | None = None
    threshold_kl: float | None = None
    deployment_id: int | None = None
    model_version: str | None = None
    evaluation_status: str = "completed"
    error_message: str | None = None


def _to_drift_history_entry(r: DriftCheckResult) -> DriftHistoryEntry:
    """Map a `DriftCheckResult` row to its API representation."""
    return DriftHistoryEntry(
        id=r.id,
        job_id=r.job_id,
        dataset_name=r.dataset_name,
        reference_rows=r.reference_rows,
        current_rows=r.current_rows,
        drifted_columns_count=r.drifted_columns_count,
        total_columns=r.total_columns,
        summary=cast(dict[str, Any] | None, r.summary),
        created_at=r.created_at.isoformat() if r.created_at else None,
        severity=r.severity,
        status=r.status,
        owner=r.owner,
        acknowledged_at=r.acknowledged_at.isoformat() if r.acknowledged_at else None,
        resolved_at=r.resolved_at.isoformat() if r.resolved_at else None,
        threshold_version=r.threshold_version,
        threshold_psi=r.threshold_psi,
        threshold_ks=r.threshold_ks,
        threshold_wasserstein=r.threshold_wasserstein,
        threshold_kl=r.threshold_kl,
        deployment_id=r.deployment_id,
        model_version=r.model_version,
        evaluation_status=r.evaluation_status,
        error_message=r.error_message,
    )


@router.get("/drift/history/{job_id}", response_model=list[DriftHistoryEntry])
async def get_drift_history(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[DriftHistoryEntry]:
    """Return all drift check results for a given job, newest first."""
    stmt = (
        select(DriftCheckResult)
        .where(DriftCheckResult.job_id == job_id)
        .order_by(DriftCheckResult.created_at.desc())  # type: ignore[union-attr]
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [_to_drift_history_entry(r) for r in rows]


class DriftAlertDetail(DriftHistoryEntry):
    """Full alert detail, including the per-feature drift evidence."""

    column_drifts: dict[str, Any] | None = None
    disposition_history: list[dict[str, Any]] = []


@router.get("/drift/alerts/{alert_id}", response_model=DriftAlertDetail)
async def get_drift_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
) -> DriftAlertDetail:
    """Return full detail for one drift alert, including per-feature evidence."""
    stmt = select(DriftCheckResult).where(DriftCheckResult.id == alert_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Drift alert {alert_id} not found")
    base = _to_drift_history_entry(row)
    return DriftAlertDetail(
        **base.model_dump(),
        column_drifts=cast(dict[str, Any] | None, row.column_drifts),
        disposition_history=cast(list[dict[str, Any]] | None, row.disposition_history) or [],
    )


class DriftDispositionUpdate(BaseModel):
    """Request body to acknowledge, resolve, or reopen a drift alert."""

    action: DriftDispositionAction
    actor: str
    note: str | None = None


@router.patch("/drift/alerts/{alert_id}/disposition", response_model=DriftAlertDetail)
async def update_drift_alert_disposition(
    alert_id: int,
    body: DriftDispositionUpdate,
    db: AsyncSession = Depends(get_db),
) -> DriftAlertDetail:
    """Record an explicit acknowledge/resolve/reopen disposition, with actor and timestamp.

    Only the transitions in `_ALLOWED_DISPOSITION_TRANSITIONS` are accepted —
    e.g. an alert cannot be resolved before it has been acknowledged — so the
    lifecycle can't silently skip a step.
    """
    stmt = select(DriftCheckResult).where(DriftCheckResult.id == alert_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Drift alert {alert_id} not found")

    allowed_from = _ALLOWED_DISPOSITION_TRANSITIONS[body.action]
    if row.status not in allowed_from:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot {body.action} a drift alert with status '{row.status}'; "
                f"expected one of {sorted(allowed_from)}"
            ),
        )

    now = datetime.now(UTC)
    new_status = {"acknowledge": "acknowledged", "resolve": "resolved", "reopen": "reopened"}[
        body.action
    ]
    row.status = new_status
    row.owner = body.actor
    if body.action == "acknowledge":
        row.acknowledged_at = now
    elif body.action == "resolve":
        row.resolved_at = now
    elif body.action == "reopen":
        row.resolved_at = None

    history = cast(list[dict[str, Any]] | None, row.disposition_history) or []
    history = [
        *history,
        {"status": new_status, "actor": body.actor, "note": body.note, "at": now.isoformat()},
    ]
    row.disposition_history = cast(Any, history)

    await db.commit()
    await db.refresh(row)

    base = _to_drift_history_entry(row)
    return DriftAlertDetail(
        **base.model_dump(),
        column_drifts=cast(dict[str, Any] | None, row.column_drifts),
        disposition_history=history,
    )


class DriftStatusSummary(BaseModel):
    has_drift: bool
    drifted_jobs: int
    latest_check: str | None = None
    # Unacknowledged critical alerts — drives the sidebar badge's urgency,
    # distinct from `has_drift`/`drifted_jobs` which only look at the latest
    # check per job regardless of whether anyone has triaged it yet.
    unacknowledged_critical: int = 0


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
    latest_check: str | None = None
    unacknowledged_critical = 0
    for r in rows:
        job_id = r.job_id
        if job_id in seen_jobs:
            continue
        seen_jobs.add(job_id)
        if latest_check is None and r.created_at:
            latest_check = r.created_at.isoformat()
        if cast(int, r.drifted_columns_count or 0) > 0:
            drifted_count += 1
        if r.severity == "critical" and r.status in ("new", "reopened"):
            unacknowledged_critical += 1

    return DriftStatusSummary(
        has_drift=drifted_count > 0,
        drifted_jobs=drifted_count,
        latest_check=latest_check,
        unacknowledged_critical=unacknowledged_critical,
    )


# ---------------------------------------------------------------------------
# In-house error tracker endpoints
# ---------------------------------------------------------------------------


class ErrorEventResponse(BaseModel):
    id: int
    route: str
    error_type: str
    message: str
    traceback: str | None = None
    job_id: str | None = None
    status_code: int
    created_at: str | None = None
    resolved_at: str | None = None
    # Derived, not stored — see `_classify_error_severity`.
    severity: str


class ErrorCountResponse(BaseModel):
    count: int


class ErrorDeleteResponse(BaseModel):
    deleted: int


class ErrorGroupedEntry(BaseModel):
    error_type: str
    route: str
    count: int
    last_seen: str | None = None
    first_seen: str | None = None
    sample_id: int | None = None


class ErrorTimelineEntry(BaseModel):
    hour: str
    count: int


class ErrorFacets(BaseModel):
    """Every typed facet value present across the full unfiltered history.

    Computed before any filter is applied so the client's dropdowns list
    every value available, not only the ones visible on the current page.
    """

    severities: list[str]
    error_types: list[str]
    job_ids: list[str]


class ErrorEventFiltersEcho(BaseModel):
    since: str | None = None
    show_resolved: bool = False
    severity: str | None = None
    error_type: str | None = None
    job_id: str | None = None
    q: str | None = None


class ErrorEventSearchResponse(BaseModel):
    total: int
    total_unfiltered: int
    facets: ErrorFacets
    filters: ErrorEventFiltersEcho
    entries: list[ErrorEventResponse]


def _classify_error_severity(status_code: int) -> str:
    """Map an HTTP status code to one of the typed Error Log severities.

    5xx are the unhandled server failures the tracker exists to surface,
    4xx are client-caused but still worth triaging, everything else (the
    background-task sentinel `0`) is informational.
    """
    if status_code >= 500:
        return "critical"
    if status_code >= 400:
        return "warning"
    return "info"


def _normalize_since_for_naive_column(since_dt: datetime) -> datetime:
    """Convert a `since` filter value to the naive-UTC form stored in `created_at`/`run_at`.

    The timestamp columns are `DateTime` without timezone, so they are always
    naive (treated as UTC). A tz-aware `since` is converted to UTC and then
    stripped of tzinfo so the comparison is apples-to-apples in SQL.
    """
    if since_dt.tzinfo is not None:
        return since_dt.astimezone(UTC).replace(tzinfo=None)
    return since_dt


def _q_like_condition(q: str, *columns):
    """Build a case-insensitive, wildcard-escaped `LIKE` OR-condition across `columns`."""
    from sqlalchemy import func as sa_func
    from sqlalchemy import or_

    escaped = q.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"
    return or_(
        *(sa_func.lower(sa_func.coalesce(col, "")).like(pattern, escape="\\") for col in columns)
    )


@router.get("/errors", response_model=ErrorEventSearchResponse)
async def list_error_events(
    limit: int = 100,
    since: str | None = None,
    show_resolved: bool = False,
    severity: str | None = None,
    error_type: str | None = None,
    job_id: str | None = None,
    q: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> ErrorEventSearchResponse:
    """Return error events matching the given filters (newest first, max 500).

    By default only unresolved events are returned. Pass ``show_resolved=true``
    to include resolved/dismissed events. ``since`` is an ISO-8601 datetime
    string. ``severity`` (critical/warning/info), ``error_type``, and
    ``job_id`` are exact-match typed facets; ``q`` is the generic free-text
    search, matched case-insensitively against message, route, error type,
    and job id (so an exact HTTP `job_id` is still found by generic search).
    All filters, pagination, and facets are computed with SQL aggregates and a
    bounded ``LIMIT`` — the full table is never materialized into Python.
    """
    from sqlalchemy import case
    from sqlalchemy import func as sa_func

    limit = min(max(1, limit), 500)

    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            since_dt = None  # ignore malformed since param

    conditions = []
    if not show_resolved:
        conditions.append(ErrorEvent.resolved_at.is_(None))
    if since_dt is not None:
        conditions.append(ErrorEvent.created_at >= _normalize_since_for_naive_column(since_dt))
    if severity == "critical":
        conditions.append(ErrorEvent.status_code >= 500)
    elif severity == "warning":
        conditions.append(ErrorEvent.status_code.between(400, 499))
    elif severity == "info":
        conditions.append(ErrorEvent.status_code < 400)
    if error_type is not None:
        conditions.append(ErrorEvent.error_type == error_type)
    if job_id is not None:
        conditions.append(sa_func.coalesce(ErrorEvent.job_id, "") == job_id)
    if q:
        conditions.append(
            _q_like_condition(
                q, ErrorEvent.message, ErrorEvent.route, ErrorEvent.error_type, ErrorEvent.job_id
            )
        )

    total_unfiltered_result = await db.execute(select(sa_func.count()).select_from(ErrorEvent))
    total_unfiltered = int(total_unfiltered_result.scalar() or 0)

    severity_case = case(
        (ErrorEvent.status_code >= 500, "critical"),
        (ErrorEvent.status_code >= 400, "warning"),
        else_="info",
    )
    facet_severities_result = await db.execute(select(severity_case).distinct())
    facet_severities = sorted({row[0] for row in facet_severities_result.all()})

    facet_error_types_result = await db.execute(
        select(ErrorEvent.error_type).distinct().order_by(ErrorEvent.error_type)
    )
    facet_error_types = [row[0] for row in facet_error_types_result.all()]

    facet_job_ids_result = await db.execute(
        select(ErrorEvent.job_id)
        .where(ErrorEvent.job_id.isnot(None))
        .distinct()
        .order_by(ErrorEvent.job_id)
    )
    facet_job_ids = [row[0] for row in facet_job_ids_result.all()]

    total_result = await db.execute(
        select(sa_func.count()).select_from(ErrorEvent).where(*conditions)
    )
    total = int(total_result.scalar() or 0)

    entries_stmt = (
        select(ErrorEvent).where(*conditions).order_by(ErrorEvent.created_at.desc()).limit(limit)
    )
    entries_result = await db.execute(entries_stmt)
    entries = [
        ErrorEventResponse(**e.to_dict(), severity=_classify_error_severity(e.status_code))
        for e in entries_result.scalars().all()
    ]

    return ErrorEventSearchResponse(
        total=total,
        total_unfiltered=total_unfiltered,
        facets=ErrorFacets(
            severities=facet_severities,
            error_types=facet_error_types,
            job_ids=facet_job_ids,
        ),
        filters=ErrorEventFiltersEcho(
            since=since,
            show_resolved=show_resolved,
            severity=severity,
            error_type=error_type,
            job_id=job_id,
            q=q,
        ),
        entries=entries,
    )


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
    deleted = getattr(result, "rowcount", None) or 0
    return ErrorDeleteResponse(deleted=deleted)


@router.get("/errors/grouped", response_model=list[ErrorGroupedEntry])
async def get_errors_grouped(
    db: AsyncSession = Depends(get_db),
) -> list[ErrorGroupedEntry]:
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


def _build_zero_filled_hour_buckets(cutoff: datetime, hours: int) -> dict[str, int]:
    """Build a zero-filled dict keyed by hourly ISO slot strings starting at `cutoff`."""
    buckets: dict[str, int] = {}
    for i in range(hours):
        slot = (cutoff + timedelta(hours=i)).replace(minute=0, second=0, microsecond=0)
        buckets[slot.strftime("%Y-%m-%dT%H:00")] = 0
    return buckets


def _fill_error_buckets(buckets: dict[str, int], timestamps: list) -> None:
    """Increment each bucket's count in-place for timestamps that fall within a known hour slot."""
    for ts in timestamps:
        if ts is None:
            continue
        # Normalise to UTC-aware
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        slot_key = ts.strftime("%Y-%m-%dT%H:00")
        if slot_key in buckets:
            buckets[slot_key] += 1


@router.get("/errors/timeline", response_model=list[ErrorTimelineEntry])
async def get_error_timeline(
    hours: int = 24,
    db: AsyncSession = Depends(get_db),
) -> list[ErrorTimelineEntry]:
    """Return error count bucketed by hour for the last N hours.

    Returns a list of ``{ hour: <ISO string>, count: N }`` entries,
    one per hour slot, oldest first. Slots with zero events are included
    so the chart always has a complete x-axis.
    """
    from backend.config import get_settings as _get_settings

    hours = min(
        max(1, hours), _get_settings().MONITORING_MAX_TIMELINE_HOURS
    )  # cap at configured max
    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=hours)

    stmt = select(ErrorEvent.created_at).where(ErrorEvent.created_at >= cutoff)
    result = await db.execute(stmt)
    timestamps = [row[0] for row in result.all()]

    # Build a zero-filled bucket dict: { slot_iso: count }
    buckets = _build_zero_filled_hour_buckets(cutoff, hours)
    _fill_error_buckets(buckets, timestamps)

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
    event.resolved_at = datetime.now(UTC)
    await db.commit()
    return ErrorEventResponse(
        **event.to_dict(), severity=_classify_error_severity(event.status_code)
    )


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
    return ErrorEventResponse(
        **event.to_dict(), severity=_classify_error_severity(event.status_code)
    )


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
    return ErrorEventResponse(
        **event.to_dict(), severity=_classify_error_severity(event.status_code)
    )


# ---------------------------------------------------------------------------
# Workspace-wide slow-node telemetry
# ---------------------------------------------------------------------------
# Reads `metrics.node_timings` (written by JobStrategy.handle_success) off
# every completed job in the lookback window and aggregates per `step_type`.
# Surfaces the same numbers the engine already collects, no extra
# instrumentation required. Legacy jobs without `node_timings` are simply
# skipped — the page just shows fewer entries until enough new runs land.


class SlowNodeRun(BaseModel):
    """A single contributing run behind a `SlowNodeAggregate` row.

    Carries enough job/node/dataset identity for a `RecordLink` to open the
    measured job and Canvas node directly, rather than leaving an operator to
    guess which run produced the aggregate.
    """

    job_id: str
    pipeline_id: str
    node_id: str
    dataset_source_id: str
    execution_seconds: float
    finished_at: str | None = None
    is_outlier: bool = False


# Cap on how many contributing runs are echoed per aggregate — enough to show
# the slowest handful (including any outliers) without inflating the payload
# with every run behind a high-volume step type.
_CONTRIBUTING_RUNS_LIMIT = 5


class SlowNodeAggregate(BaseModel):
    step_type: str
    count: int
    total_seconds: float
    avg_seconds: float
    p95_seconds: float
    max_seconds: float
    sample_node_id: str | None = None
    # True when `count == 1` — the aggregate is one run's measurement, not a
    # statistical summary, and the UI must say so rather than implying a trend.
    is_single_run: bool = False
    # True when the sample node's run is not itself an outlier, so a caller
    # can trust it as roughly representative of the group.
    sample_is_representative: bool = True
    contributing_runs: list[SlowNodeRun] = []


class SlowNodesResponse(BaseModel):
    days: int
    unit: str = "seconds"
    total_jobs_scanned: int
    total_node_runs: int
    aggregates: list[SlowNodeAggregate]


def _percentile(values: list[float], pct: float) -> float:
    """Cheap nearest-rank percentile — avoids pulling numpy into the route."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[rank]


def _clamp_slow_nodes_params(days: int, limit: int) -> tuple[int, int]:
    """Clamp `days`/`limit` query params to the configured monitoring caps."""
    from backend.config import get_settings as _get_settings

    _settings = _get_settings()
    days = max(1, min(days, _settings.MONITORING_MAX_SLOWNODES_DAYS))
    limit = max(1, min(limit, _settings.MAX_PAGE_SIZE))
    return days, limit


# ---------------------------------------------------------------------------
# Training-family resolution — the engine dispatches every canvas
# Classification/Regression/Text Classification/Segmentation/Ensemble/generic
# Training node under the single canonical `step_type: 'training'` (see
# `pipelineConverter.ts`'s `ALL_TRAINING_DISPATCH_TYPES` collapse), which is
# load-bearing for engine dispatch and must not change. This section resolves
# the real model family purely for Slow Nodes *display*, mirroring the
# frontend's `getTaskForModelType` (jobMeta.ts) precedence so both surfaces
# agree on what each model_type means.
# ---------------------------------------------------------------------------

_TRAINING_FAMILY_LABELS: dict[str, str] = {
    "classification": "Classification",
    "regression": "Regression",
    "text_classification": "Text Classification",
    "segmentation": "Segmentation",
    "ensemble": "Ensemble",
}

# Grouping key/label used when a legacy `training` run's model family can't
# be resolved (malformed/missing graph node and no job.model_type) — an
# honest fallback rather than silently mislabeling it as any other family.
_UNRESOLVED_TRAINING_KEY = "training_unspecified"
_UNRESOLVED_TRAINING_LABEL = "Training (unspecified)"


@lru_cache(maxsize=1)
def _step_type_registry_labels() -> dict[str, str]:
    """Cache `{step_type_id: display_name}` for every known engine step, keyed by every alias.

    Unlike `model_registry_tags()` (deduped via `_build_node_registry()`),
    this reads `SkyulfRegistry.get_all_metadata()` directly so a
    backward-compatible alias (e.g. `FeatureMath` for `FeatureGenerationNode`)
    still resolves to its own registered display name rather than being
    dropped by the dedup pass.
    """
    labels = {
        node_id: str(meta.get("name") or node_id)
        for node_id, meta in SkyulfRegistry.get_all_metadata().items()
    }
    labels[StepType.DATA_LOADER] = "Data Loader"
    labels[_UNRESOLVED_TRAINING_KEY] = _UNRESOLVED_TRAINING_LABEL
    return labels


def _display_step_type(step: str) -> str:
    """Resolve a raw `step_type` id to the human-readable label shown on Slow Nodes.

    Prefers the skyulf-core registry's own display name — the same one the
    Canvas node palette uses — so a step's label never drifts from what the
    node is actually called. Falls back to a generic Title Case prettifier
    only for identifiers with no registered definition (e.g. the internal
    `feature_target_split` engine step or an already-resolved model family
    name, for which the prettifier is a no-op).
    """
    return _step_type_registry_labels().get(step) or _humanize_step_type(step)


def _resolve_training_step_type(job: Any, node_id: str, registry_tags: dict[str, list[str]]) -> str:
    """Resolve the display step_type for a `training`-step timing entry.

    First tries the node's own stored graph params (`algorithm`/`model_type`
    on the matching node), falling back to the job row's own `model_type`
    column when the graph lookup fails — this covers both new runs and
    historical runs whose `node_timings` were already persisted with the
    canonical `training` literal.
    """
    model_type: str | None = None
    graph: dict[str, Any] = cast(dict[str, Any], getattr(job, "graph", None) or {})
    node_map = _build_node_map(graph) if isinstance(graph, dict) else {}
    entry = node_map.get(node_id)
    if entry is not None:
        _, params, _ = entry
        candidate = params.get("algorithm") or params.get("model_type")
        model_type = candidate if isinstance(candidate, str) else None
    if not model_type:
        job_model_type = getattr(job, "model_type", None)
        model_type = job_model_type if isinstance(job_model_type, str) else None

    family = resolve_model_family(model_type, registry_tags)
    if family is None:
        return _UNRESOLVED_TRAINING_KEY
    return _TRAINING_FAMILY_LABELS[family]


def _accumulate_node_timing(
    entry: Any,
    job: Any,
    by_step: dict[str, list[SlowNodeRun]],
    sample_node: dict[str, str],
) -> bool:
    """Fold a single node-timing entry into the running per-step aggregates.

    `job` supplies the run-level context (job/pipeline/dataset identity, and
    when the run finished) that a `SlowNodeRun` needs to be independently
    addressable later, rather than only contributing a bare number. A raw
    `training` step_type (the canonical engine dispatch value shared by every
    training-family node) is resolved to its real model family here, purely
    for display — see `_resolve_training_step_type`. Every step_type — model
    families included — is then run through `_display_step_type()` so the
    grouping key is always a consistent human-readable label rather than a
    mix of raw PascalCase/snake_case ids and Title Case family names.

    Returns True if the entry contributed a run to the aggregates.
    """
    if not isinstance(entry, dict):
        return False
    step = str(entry.get("step_type") or "unknown")
    try:
        secs = float(entry.get("execution_time") or 0.0)
    except (TypeError, ValueError):
        return False
    if secs <= 0:
        return False
    node_id = str(entry.get("node_id") or "")
    if step == StepType.TRAINING:
        step = _resolve_training_step_type(job, node_id, model_registry_tags())
    step = _display_step_type(step)
    by_step.setdefault(step, []).append(
        SlowNodeRun(
            job_id=str(job.id),
            pipeline_id=str(job.pipeline_id),
            node_id=node_id,
            dataset_source_id=str(job.dataset_source_id),
            execution_seconds=round(secs, 4),
            finished_at=job.finished_at.isoformat() if job.finished_at else None,
        )
    )
    sample_node.setdefault(step, node_id)
    return True


async def _scan_slow_node_jobs(
    db: AsyncSession,
    cutoff: datetime,
) -> tuple[dict[str, list[SlowNodeRun]], dict[str, str], int, int]:
    """Scan completed jobs since `cutoff` and aggregate per-step node timings."""
    by_step: dict[str, list[SlowNodeRun]] = {}
    sample_node: dict[str, str] = {}
    jobs_scanned = 0
    runs_seen = 0

    # Scan the unified table — both run_modes share the same metrics shape.
    stmt = select(TrainingJob).where(
        TrainingJob.status == "completed",
        TrainingJob.finished_at.isnot(None),
        TrainingJob.finished_at >= cutoff,
    )
    result = await db.execute(stmt)
    for job in result.scalars().all():
        jobs_scanned += 1
        metrics = job.metrics or {}
        timings = metrics.get("node_timings") if isinstance(metrics, dict) else None
        if not isinstance(timings, list):
            continue
        for entry in timings:
            if _accumulate_node_timing(entry, job, by_step, sample_node):
                runs_seen += 1

    return by_step, sample_node, jobs_scanned, runs_seen


def _mark_outlier_runs(runs: list[SlowNodeRun], avg_seconds: float) -> None:
    """Flag runs materially slower than the group's average.

    Uses 1.5x average as a simple, explainable threshold rather than a formal
    statistical test — good enough to point an operator at the runs worth
    investigating first without over-claiming precision.
    """
    threshold = avg_seconds * 1.5
    for run in runs:
        run.is_outlier = run.execution_seconds > threshold


def _build_slow_node_aggregates(
    by_step: dict[str, list[SlowNodeRun]],
    sample_node: dict[str, str],
) -> list[SlowNodeAggregate]:
    """Turn per-step run lists into sorted `SlowNodeAggregate` rows."""
    aggregates: list[SlowNodeAggregate] = []
    for step, runs in by_step.items():
        values = [run.execution_seconds for run in runs]
        total = sum(values)
        avg = total / len(values)
        _mark_outlier_runs(runs, avg)

        sample_id = sample_node.get(step) or None
        sample_run = next((run for run in runs if run.node_id == sample_id), None)
        contributing_runs = sorted(runs, key=lambda run: run.execution_seconds, reverse=True)[
            :_CONTRIBUTING_RUNS_LIMIT
        ]

        aggregates.append(
            SlowNodeAggregate(
                step_type=step,
                count=len(values),
                total_seconds=round(total, 4),
                avg_seconds=round(avg, 4),
                p95_seconds=round(_percentile(values, 95), 4),
                max_seconds=round(max(values), 4),
                sample_node_id=sample_id,
                is_single_run=len(values) == 1,
                sample_is_representative=sample_run is None or not sample_run.is_outlier,
                contributing_runs=contributing_runs,
            )
        )
    aggregates.sort(key=lambda a: a.total_seconds, reverse=True)
    return aggregates


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
    days, limit = _clamp_slow_nodes_params(days, limit)
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)

    by_step, sample_node, jobs_scanned, runs_seen = await _scan_slow_node_jobs(db, cutoff)
    aggregates = _build_slow_node_aggregates(by_step, sample_node)

    return SlowNodesResponse(
        days=days,
        total_jobs_scanned=jobs_scanned,
        total_node_runs=runs_seen,
        aggregates=aggregates[:limit],
    )


# ---------------------------------------------------------------------------
# Pipeline run logs — node failures & warnings persisted from the frontend
# ---------------------------------------------------------------------------


class PipelineLogEntry(BaseModel):
    node_id: str | None = None
    node_type: str | None = None
    level: str = "error"
    logger: str | None = None
    message: str


class PipelineLogBatch(BaseModel):
    pipeline_id: str | None = None
    entries: list[PipelineLogEntry]


class PipelineRunLogResponse(BaseModel):
    id: int
    pipeline_id: str | None = None
    node_id: str | None = None
    node_type: str | None = None
    level: str
    logger: str | None = None
    message: str
    run_at: str | None = None


class PipelineLogFacets(BaseModel):
    """Every typed facet value present across the full unfiltered history."""

    levels: list[str]
    node_types: list[str]
    pipeline_ids: list[str]
    node_ids: list[str]


class PipelineLogFiltersEcho(BaseModel):
    since: str | None = None
    pipeline_id: str | None = None
    level: str | None = None
    node_type: str | None = None
    node_id: str | None = None
    q: str | None = None


class PipelineLogSearchResponse(BaseModel):
    total: int
    total_unfiltered: int
    facets: PipelineLogFacets
    filters: PipelineLogFiltersEcho
    entries: list[PipelineRunLogResponse]


@router.post("/pipeline-logs", response_model=list[PipelineRunLogResponse], status_code=201)
async def create_pipeline_logs(
    batch: PipelineLogBatch,
    db: AsyncSession = Depends(get_db),
) -> list[PipelineRunLogResponse]:
    """Persist a batch of node failures / warnings from a pipeline preview run."""
    if not batch.entries:
        return []
    rows = [
        PipelineRunLog(
            pipeline_id=batch.pipeline_id,
            node_id=e.node_id,
            node_type=e.node_type,
            level=e.level,
            logger=e.logger,
            message=e.message,
        )
        for e in batch.entries
    ]
    db.add_all(rows)
    await db.commit()
    for row in rows:
        await db.refresh(row)
    return [PipelineRunLogResponse(**r.to_dict()) for r in rows]


@router.get("/pipeline-logs", response_model=PipelineLogSearchResponse)
async def list_pipeline_logs(
    limit: int = 200,
    since: str | None = None,
    pipeline_id: str | None = None,
    level: str | None = None,
    node_type: str | None = None,
    node_id: str | None = None,
    q: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> PipelineLogSearchResponse:
    """Return pipeline run logs matching the given filters (newest first, max 500).

    ``level`` (error/warning/info), ``node_type``, and ``node_id`` are exact-match
    typed facets; ``q`` is the generic free-text search, matched case-insensitively
    against message, node type, and node id (so an exact pipeline `node_id` is
    still found by generic search). All filters, pagination, and facets are
    computed with SQL aggregates and a bounded ``LIMIT`` — the full table is
    never materialized into Python.
    """
    from sqlalchemy import func as sa_func

    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            since_dt = None  # ignore malformed since param

    limit = min(max(1, limit), 500)

    conditions = []
    if since_dt is not None:
        conditions.append(PipelineRunLog.run_at >= _normalize_since_for_naive_column(since_dt))
    if pipeline_id is not None:
        conditions.append(PipelineRunLog.pipeline_id == pipeline_id)
    if level is not None:
        conditions.append(PipelineRunLog.level == level)
    if node_type is not None:
        conditions.append(PipelineRunLog.node_type == node_type)
    if node_id is not None:
        conditions.append(sa_func.coalesce(PipelineRunLog.node_id, "") == node_id)
    if q:
        conditions.append(
            _q_like_condition(
                q, PipelineRunLog.message, PipelineRunLog.node_type, PipelineRunLog.node_id
            )
        )

    total_unfiltered_result = await db.execute(select(sa_func.count()).select_from(PipelineRunLog))
    total_unfiltered = int(total_unfiltered_result.scalar() or 0)

    facet_levels_result = await db.execute(
        select(PipelineRunLog.level)
        .where(PipelineRunLog.level.isnot(None))
        .distinct()
        .order_by(PipelineRunLog.level)
    )
    facet_levels = [row[0] for row in facet_levels_result.all()]

    facet_node_types_result = await db.execute(
        select(PipelineRunLog.node_type)
        .where(PipelineRunLog.node_type.isnot(None))
        .distinct()
        .order_by(PipelineRunLog.node_type)
    )
    facet_node_types = [row[0] for row in facet_node_types_result.all()]

    facet_pipeline_ids_result = await db.execute(
        select(PipelineRunLog.pipeline_id)
        .where(PipelineRunLog.pipeline_id.isnot(None))
        .distinct()
        .order_by(PipelineRunLog.pipeline_id)
    )
    facet_pipeline_ids = [row[0] for row in facet_pipeline_ids_result.all()]

    facet_node_ids_result = await db.execute(
        select(PipelineRunLog.node_id)
        .where(PipelineRunLog.node_id.isnot(None))
        .distinct()
        .order_by(PipelineRunLog.node_id)
    )
    facet_node_ids = [row[0] for row in facet_node_ids_result.all()]

    total_result = await db.execute(
        select(sa_func.count()).select_from(PipelineRunLog).where(*conditions)
    )
    total = int(total_result.scalar() or 0)

    entries_stmt = (
        select(PipelineRunLog)
        .where(*conditions)
        .order_by(PipelineRunLog.run_at.desc())
        .limit(limit)
    )
    entries_result = await db.execute(entries_stmt)
    entries = [PipelineRunLogResponse(**log.to_dict()) for log in entries_result.scalars().all()]

    return PipelineLogSearchResponse(
        total=total,
        total_unfiltered=total_unfiltered,
        facets=PipelineLogFacets(
            levels=facet_levels,
            node_types=facet_node_types,
            pipeline_ids=facet_pipeline_ids,
            node_ids=facet_node_ids,
        ),
        filters=PipelineLogFiltersEcho(
            since=since,
            pipeline_id=pipeline_id,
            level=level,
            node_type=node_type,
            node_id=node_id,
            q=q,
        ),
        entries=entries,
    )


@router.delete("/pipeline-logs", status_code=204)
async def clear_pipeline_logs(
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete all pipeline run log entries."""
    from sqlalchemy import delete as sa_delete

    await db.execute(sa_delete(PipelineRunLog))
    await db.commit()


# ---------------------------------------------------------------------------
# Node inspector — read-only detail from a job's stored graph snapshot
# ---------------------------------------------------------------------------
# Every job persists the graph exactly as it executed (`TrainingJob.graph`),
# so a single node can be investigated from its run record without the
# ML Canvas: this works even when the source pipeline was since edited or
# deleted, or was never a saved pipeline at all (a `preview_*`/`*__branch_N`
# synthetic run). Serves only from already-stored columns — never loads
# model artifacts.


class NodeNeighbor(BaseModel):
    node_id: str
    step_type: str
    label: str


class NodeInspectorDetail(BaseModel):
    node_id: str
    step_type: str
    label: str
    params: dict[str, Any]
    upstream: list[NodeNeighbor]
    downstream: list[NodeNeighbor]
    execution_seconds: float | None = None
    execution_status: str | None = None


class NodeInspectorLogEntry(BaseModel):
    level: str
    message: str
    run_at: str | None = None


class NodeInspectorResponse(BaseModel):
    job_id: str
    node_id: str
    node_found: bool
    node: NodeInspectorDetail | None = None
    pipeline_id: str
    dataset_source_id: str
    dataset_name: str | None = None
    branch_index: int | None = None
    run_mode: str
    model_type: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    # True only for a synthetic run (`preview_*` / `*__branch_N`) that was
    # never a saved, independently loadable pipeline.
    is_synthetic_pipeline: bool
    can_open_in_canvas: bool
    recent_logs: list[NodeInspectorLogEntry] = []


# A graph node entry as (step_type, params, upstream node ids).
_GraphNodeEntry = tuple[str, dict[str, Any], list[str]]


def _parse_inspector_node(node: dict[str, Any]) -> tuple[str, str, dict[str, Any], list[str]]:
    """Parse one stored graph node into (node_id, step_type, params, upstream_ids).

    Training-job graphs are always persisted in the internal
    `{"node_id", "step_type", "params", "inputs"}` shape written by
    `_build_branch_graph`; the legacy React Flow `{"id", "type", "data"}`
    shape is also tolerated for older/rescued rows.
    """
    if "step_type" in node:
        node_id = str(node.get("node_id") or "")
        step_type = str(node.get("step_type") or "unknown")
        params = node.get("params") or {}
        inputs = node.get("inputs") or []
    else:
        node_id = str(node.get("id") or "")
        step_type = str(node.get("type") or node.get("data", {}).get("catalogType") or "unknown")
        params = (
            node.get("data", {}).get("config")
            or node.get("parameters")
            or node.get("data", {})
            or {}
        )
        inputs = node.get("inputs") or []
    return (
        node_id,
        step_type,
        dict(params) if isinstance(params, dict) else {},
        list(inputs) if isinstance(inputs, list) else [],
    )


def _humanize_step_type(step_type: str) -> str:
    """Turn a snake_case step type into a human label, e.g. `train_test_split` -> `Train Test Split`."""
    label = step_type.replace("_", " ").replace("-", " ").strip().title()
    return label or step_type


def _build_node_map(graph: dict[str, Any]) -> dict[str, _GraphNodeEntry]:
    """Index every node in a stored graph by id, tolerating malformed entries."""
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    node_map: dict[str, _GraphNodeEntry] = {}
    if not isinstance(nodes, list):
        return node_map
    for raw in nodes:
        if not isinstance(raw, dict):
            continue
        node_id, step_type, params, inputs = _parse_inspector_node(raw)
        if node_id:
            node_map[node_id] = (step_type, params, inputs)
    return node_map


def _build_node_neighbor(node_id: str, node_map: dict[str, _GraphNodeEntry]) -> NodeNeighbor:
    """Build a `NodeNeighbor` for `node_id`, tolerating an id absent from the graph."""
    entry = node_map.get(node_id)
    step_type = entry[0] if entry else "unknown"
    return NodeNeighbor(node_id=node_id, step_type=step_type, label=_humanize_step_type(step_type))


def _find_downstream_ids(node_id: str, node_map: dict[str, _GraphNodeEntry]) -> list[str]:
    """Every node whose `inputs` names `node_id` — i.e. `node_id`'s downstream neighbours."""
    return [nid for nid, (_, _, inputs) in node_map.items() if node_id in inputs]


def _is_synthetic_pipeline(pipeline_id: str) -> bool:
    """True for a `preview_*` run or a `*__branch_N` split of one — never a saved pipeline."""
    parent, _ = parse_branch_info(pipeline_id)
    root = parent or pipeline_id
    return root.startswith("preview_")


def _extract_node_execution(
    metrics: dict[str, Any], node_id: str
) -> tuple[float | None, str | None]:
    """Look up `node_id`'s recorded execution time/status from `metrics.node_timings`."""
    timings = metrics.get("node_timings") if isinstance(metrics, dict) else None
    if not isinstance(timings, list):
        return None, None
    for entry in timings:
        if not isinstance(entry, dict) or str(entry.get("node_id")) != node_id:
            continue
        try:
            seconds = float(entry.get("execution_time"))
        except (TypeError, ValueError):
            seconds = None
        status = entry.get("status")
        return seconds, status if isinstance(status, str) else None
    return None, None


async def _build_node_inspector_response(
    db: AsyncSession, job: TrainingJob, node_id: str
) -> NodeInspectorResponse:
    """Build a read-only node-inspector payload from `job`'s stored graph snapshot.

    Sourced entirely from `job.graph`/`job.metrics` — the graph exactly as it
    executed — never from live canvas/pipeline state, so investigation works
    even when the source pipeline was edited, deleted, or never saved.
    """
    graph: dict[str, Any] = cast(dict[str, Any], job.graph or {})
    node_map = _build_node_map(graph)
    entry = node_map.get(node_id)

    node_detail: NodeInspectorDetail | None = None
    if entry is not None:
        step_type, params, inputs = entry
        metrics: dict[str, Any] = cast(dict[str, Any], job.metrics or {})
        execution_seconds, execution_status = _extract_node_execution(metrics, node_id)
        node_detail = NodeInspectorDetail(
            node_id=node_id,
            step_type=step_type,
            label=_humanize_step_type(step_type),
            params=params,
            upstream=[_build_node_neighbor(nid, node_map) for nid in inputs if nid],
            downstream=[
                _build_node_neighbor(nid, node_map)
                for nid in _find_downstream_ids(node_id, node_map)
            ],
            execution_seconds=execution_seconds,
            execution_status=execution_status,
        )

    dataset_name = await resolve_dataset_name(db, job.dataset_source_id)

    log_stmt = (
        select(PipelineRunLog)
        .where(PipelineRunLog.pipeline_id == job.pipeline_id, PipelineRunLog.node_id == node_id)
        .order_by(PipelineRunLog.run_at.desc())
        .limit(10)
    )
    log_rows = (await db.execute(log_stmt)).scalars().all()
    recent_logs = [
        NodeInspectorLogEntry(
            level=row.level,
            message=row.message,
            run_at=row.run_at.isoformat() if row.run_at else None,
        )
        for row in log_rows
    ]

    job_metadata: dict[str, Any] = cast(dict[str, Any], job.job_metadata or {})
    branch_index = job_metadata.get("branch_index") if isinstance(job_metadata, dict) else None
    is_synthetic = _is_synthetic_pipeline(job.pipeline_id)

    return NodeInspectorResponse(
        job_id=str(job.id),
        node_id=node_id,
        node_found=node_detail is not None,
        node=node_detail,
        pipeline_id=job.pipeline_id,
        dataset_source_id=job.dataset_source_id,
        dataset_name=dataset_name,
        branch_index=branch_index if isinstance(branch_index, int) else None,
        run_mode=job.run_mode,
        model_type=job.model_type,
        status=job.status,
        started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        is_synthetic_pipeline=is_synthetic,
        can_open_in_canvas=not is_synthetic,
        recent_logs=recent_logs,
    )


@router.get("/jobs/{job_id}/nodes/{node_id}", response_model=NodeInspectorResponse)
async def get_job_node(
    job_id: str,
    node_id: str,
    db: AsyncSession = Depends(get_db),
) -> NodeInspectorResponse:
    """Read-only inspector snapshot for one node from a job's executed graph.

    Serves entirely from the job's stored `graph`/`metrics` columns — never
    loads model artifacts — so a node can be investigated from any
    operational surface (Error Log, Slow Nodes, Jobs) without the pipeline
    being currently open (or even still existing) on the canvas.
    """
    stmt = select(TrainingJob).where(TrainingJob.id == job_id)
    job = (await db.execute(stmt)).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return await _build_node_inspector_response(db, job, node_id)


@router.get("/pipeline-runs/{pipeline_id}/nodes/{node_id}", response_model=NodeInspectorResponse)
async def get_pipeline_run_node(
    pipeline_id: str,
    node_id: str,
    db: AsyncSession = Depends(get_db),
) -> NodeInspectorResponse:
    """Same as `get_job_node`, resolved from the most recent job for `pipeline_id`.

    Used when the caller only has a `pipeline_id` (e.g. a pipeline-run log
    entry, which predates any job row) rather than a `job_id` directly.
    """
    stmt = (
        select(TrainingJob)
        .where(TrainingJob.pipeline_id == pipeline_id)
        .order_by(TrainingJob.created_at.desc())
        .limit(1)
    )
    job = (await db.execute(stmt)).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found for pipeline run {pipeline_id}")
    return await _build_node_inspector_response(db, job, node_id)
