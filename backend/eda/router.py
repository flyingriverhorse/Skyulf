from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import Optional, List, Any

from backend.dependencies import get_db
from backend.database.models import EDAReport, DataSource
from backend.config import get_settings
from backend.eda.tasks import run_eda_background, generate_profile_celery
from backend.services.data_service import DataService
from backend.utils.file_utils import extract_file_path_from_source
from backend.celery_app import celery_app
from skyulf.profiling.analyzer import EDAAnalyzer
import polars as pl
import pandas as pd

router = APIRouter(prefix="/eda", tags=["EDA"])

class FilterRequest(BaseModel):
    column: str
    operator: str
    value: Any

class AnalyzeRequest(BaseModel):
    target_col: Optional[str] = None
    exclude_cols: Optional[List[str]] = None
    filters: Optional[List[FilterRequest]] = None

@router.post("/{dataset_id}/analyze")
async def trigger_analysis(
    dataset_id: int, 
    background_tasks: BackgroundTasks,
    request: Optional[AnalyzeRequest] = None,
    session: AsyncSession = Depends(get_db)
):
    """
    Triggers an EDA analysis job for the given dataset.
    """
    print(f"Triggering analysis for dataset {dataset_id}. Request: {request}")

    # Check if dataset exists
    ds = await session.get(DataSource, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    # Create Report entry
    config = {}
    if request:
        if request.target_col:
            print(f"Setting target_col to {request.target_col}")
            config["target_col"] = request.target_col
        if request.exclude_cols:
            print(f"Excluding columns: {request.exclude_cols}")
            config["exclude_cols"] = request.exclude_cols
        if request.filters:
            print(f"Applying filters: {request.filters}")
            config["filters"] = [f.dict() for f in request.filters]

    report = EDAReport(
        data_source_id=dataset_id,
        status="PENDING",
        config=config
    )
    session.add(report)
    await session.commit()
    await session.refresh(report)
    
    # Dispatch
    settings = get_settings()
    if settings.USE_CELERY:
        # Use Celery
        task = generate_profile_celery.delay(report.id)
        
        # Store task_id in config for cancellation
        new_config = dict(report.config) if report.config else {}
        new_config["celery_task_id"] = task.id
        report.config = new_config
        
        session.add(report)
        await session.commit()
    else:
        # Use BackgroundTasks
        background_tasks.add_task(run_eda_background, report.id)
        
    return {"job_id": report.id, "status": "PENDING"}

@router.post("/reports/{report_id}/cancel")
async def cancel_analysis(
    report_id: int,
    session: AsyncSession = Depends(get_db)
):
    """
    Cancels a running analysis job.
    """
    report = await session.get(EDAReport, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    if report.status not in ["PENDING", "STARTED", "RUNNING"]:
        return {"message": "Job is not running", "status": report.status}
        
    # Revoke Celery task if exists
    if report.config and "celery_task_id" in report.config:
        task_id = report.config["celery_task_id"]
        celery_app.control.revoke(task_id, terminate=True)
        
    report.status = "CANCELLED"
    report.error_message = "Cancelled by user"
    session.add(report)
    await session.commit()
    
    return {"message": "Job cancelled", "status": "CANCELLED"}

@router.get("/{dataset_id}/history")
async def get_report_history(
    dataset_id: int,
    session: AsyncSession = Depends(get_db)
):
    """
    Returns a list of past reports for a dataset.
    """
    query = (
        select(EDAReport.id, EDAReport.created_at, EDAReport.status, EDAReport.config)
        .where(EDAReport.data_source_id == dataset_id)
        .order_by(desc(EDAReport.created_at))
        .limit(20)
    )
    result = await session.execute(query)
    reports = result.all()
    
    return [
        {
            "id": r.id,
            "created_at": r.created_at,
            "status": r.status,
            "target_col": r.config.get("target_col") if r.config else None
        }
        for r in reports
    ]

@router.get("/{dataset_id}/latest")
async def get_latest_report(
    dataset_id: int,
    session: AsyncSession = Depends(get_db)
):
    """
    Returns the most recent report for a dataset, regardless of status.
    """
    query = (
        select(EDAReport)
        .where(EDAReport.data_source_id == dataset_id)
        .order_by(desc(EDAReport.created_at))
        .limit(1)
    )
    result = await session.execute(query)
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(status_code=404, detail="No analysis found for this dataset")
        
    return report.to_dict()

@router.get("/reports/{report_id}")
async def get_report(
    report_id: int,
    session: AsyncSession = Depends(get_db)
):
    """
    Get a specific report by ID.
    """
    report = await session.get(EDAReport, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report.to_dict()

class DecompositionRequest(BaseModel):
    measure_col: Optional[str] = None
    measure_agg: str = "count"
    split_col: Optional[str] = None
    filters: Optional[List[FilterRequest]] = None

@router.post("/{dataset_id}/decomposition")
async def get_decomposition(
    dataset_id: int,
    request: DecompositionRequest,
    session: AsyncSession = Depends(get_db)
):
    """
    Calculates the breakdown of a measure by a split column for Decomposition Trees.
    """
    try:
        # 1. Fetch DataSource
        ds = await session.get(DataSource, dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # 2. Resolve File Path
        source_data_dict = {
            "config": ds.config or {},
            "connection_info": ds.source_metadata or {},
            "file_path": (ds.config or {}).get("file_path"),
            "source_id": ds.source_id
        }
        file_path = extract_file_path_from_source(source_data_dict)
        
        if not file_path:
             # Fallback
             if ds.source_id and (str(ds.source_id).endswith('.csv') or str(ds.source_id).endswith('.parquet')):
                 file_path = ds.source_id
             else:
                 raise HTTPException(status_code=400, detail="File path not found")

        # 3. Prepare Storage Options (Simplified from tasks.py)
        storage_options = None
        if file_path and str(file_path).startswith("s3://"):
            creds = ds.credentials or {}
            if not creds:
                 config_creds = ds.config or {}
                 creds = {
                     "aws_access_key_id": config_creds.get("aws_access_key_id"),
                     "aws_secret_access_key": config_creds.get("aws_secret_access_key"),
                     "aws_session_token": config_creds.get("aws_session_token"),
                     "endpoint_url": config_creds.get("endpoint_url")
                 }
            
            if not creds.get("aws_access_key_id"):
                settings = get_settings()
                creds = {
                    "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
                    "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
                    "aws_session_token": settings.AWS_SESSION_TOKEN,
                    "endpoint_url": None
                }

            storage_options = {
                "key": creds.get("aws_access_key_id") or creds.get("key"),
                "secret": creds.get("aws_secret_access_key") or creds.get("secret"),
                "token": creds.get("aws_session_token") or creds.get("token"),
                "endpoint_url": creds.get("endpoint_url")
            }
            storage_options = {k: v for k, v in storage_options.items() if v is not None}

        # 4. Load Data
        data_service = DataService()
        try:
            df = await data_service.load_file(file_path, storage_options=storage_options)
            # Ensure Polars DataFrame
            if isinstance(df, pd.DataFrame):
                try:
                    df = pl.from_pandas(df)
                except Exception:
                    # Fallback: convert object cols to string to handle mixed types
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str)
                    df = pl.from_pandas(df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

        # 5. Run Analysis
        analyzer = EDAAnalyzer(df)
        
        filters_dict = [f.dict() for f in request.filters] if request.filters else []
        
        # Handle empty string split_col from frontend
        split_col_arg = request.split_col
        if split_col_arg == "":
            split_col_arg = None

        result = analyzer.get_decomposition_split(
            measure_col=request.measure_col,
            measure_agg=request.measure_agg,
            split_col=split_col_arg,
            filters=filters_dict
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR in get_decomposition: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

