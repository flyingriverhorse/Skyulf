import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import Optional, List
from pydantic import BaseModel
import polars as pl
import pandas as pd
import io
import re
from datetime import datetime
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.config import get_settings
from skyulf.profiling.drift import DriftCalculator, DriftReport

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

class DriftJobOption(BaseModel):
    job_id: str
    dataset_name: str
    filename: str
    created_at: Optional[str] = None

@router.get("/jobs", response_model=List[DriftJobOption])
async def list_drift_jobs():
    """
    List all jobs that have reference data available for drift calculation.
    Scans subdirectories in the artifact folder.
    """
    settings = get_settings()
    base_path = Path(settings.TRAINING_ARTIFACT_DIR)
    
    jobs = []
    
    if not base_path.exists():
        return []

    # Iterate over all items in the base path (job folders)
    for item_path in base_path.iterdir():
        if item_path.is_dir():
            # Try to parse timestamp from folder name
            # Format: {dataset_name}_{date}_{time}_{uuid}
            # Example: Iris_csv_20260105_153129_9c30cc64...
            folder_name = item_path.name
            created_at_str = "Unknown"
            
            try:
                # Extract date/time parts using regex or splitting
                # We look for YYYYMMDD_HHMMSS pattern
                match = re.search(r'(\d{8})_(\d{6})', folder_name)
                if match:
                    date_part = match.group(1)
                    time_part = match.group(2)
                    dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
                    created_at_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

            # Look for reference_data_*.joblib inside this directory
            try:
                for file_path in item_path.glob("reference_data_*.joblib"):
                    filename = file_path.name
                    # filename is like reference_data_{dataset_name}_{job_id}.joblib
                    # Remove extension
                    key = file_path.stem
                    
                    # Remove prefix
                    remainder = key[15:] # len("reference_data_")
                    
                    # Split by last underscore to get job_id
                    parts = remainder.rsplit('_', 1)
                    if len(parts) == 2:
                        dataset_name = parts[0]
                        job_id = parts[1]
                        
                        jobs.append(DriftJobOption(
                            job_id=job_id,
                            dataset_name=dataset_name,
                            filename=filename,
                            created_at=created_at_str
                        ))
            except Exception:
                continue
                
    # Sort by created_at descending (newest first)
    jobs.sort(key=lambda x: x.created_at or "", reverse=True)
    return jobs

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
        raise HTTPException(status_code=500, detail=f"Failed to load reference data: {str(e)}")

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
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {str(e)}")

    # 4. Calculate Drift
    try:
        calculator = DriftCalculator(ref_df, curr_df)
        report = calculator.calculate_drift()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift calculation failed: {str(e)}")
