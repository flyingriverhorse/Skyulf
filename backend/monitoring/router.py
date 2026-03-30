import os
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
from backend.database.models import BasicTrainingJob, AdvancedTuningJob
from backend.ml_pipeline.execution.graph_utils import extract_job_details
from skyulf.profiling.drift import DriftCalculator, DriftReport

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
async def list_drift_jobs(db: AsyncSession = Depends(get_db)):
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
                match = re.search(r'(\d{8})_(\d{6})', folder_name)
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
                    parts = remainder.rsplit('_', 1)
                    if len(parts) == 2:
                        dataset_name = parts[0]
                        job_id = parts[1]
                        found_jobs.append(DriftJobOption(
                            job_id=job_id,
                            dataset_name=dataset_name,
                            filename=filename,
                            created_at=created_at_str
                        ))
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
            job.model_type = cast(str, db_row.model_type)

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

                # Pick the most useful primary metric
                for key in ("test_score", "accuracy", "r2", "f1", "rmse"):
                    if key in metrics:
                        val = metrics[key]
                        if isinstance(val, (int, float)):
                            job.best_metric = f"{key}: {val:.4f}"
                        break

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

@router.post("/drift/calculate", response_model=DriftReport)
async def calculate_drift(
    job_id: str = Form(...),
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None)
):
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
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', dataset_name)
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
    except Exception as e:
        logger.exception("Failed to load reference data for job %s", job_id)
        raise HTTPException(status_code=500, detail="Failed to load reference data")

    # 3. Load Current Data
    try:
        content = await file.read()
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            curr_df = pl.read_csv(io.BytesIO(content))
        elif filename.endswith('.parquet'):
            curr_df = pl.read_parquet(io.BytesIO(content))
        else:
            # Default to CSV
            curr_df = pl.read_csv(io.BytesIO(content))
    except Exception as e:
        logger.warning("Failed to parse uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Failed to parse uploaded file")

    # 4. Calculate Drift
    try:
        calculator = DriftCalculator(ref_df, curr_df)
        report = calculator.calculate_drift()
        return report
    except Exception as e:
        logger.exception("Drift calculation failed for job %s", job_id)
        raise HTTPException(status_code=500, detail="Drift calculation failed")
