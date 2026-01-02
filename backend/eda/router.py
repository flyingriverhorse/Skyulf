from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import Optional, List

from backend.dependencies import get_db
from backend.database.models import EDAReport, DataSource
from backend.config import get_settings
from backend.eda.tasks import run_eda_background, generate_profile_celery

router = APIRouter(prefix="/eda", tags=["EDA"])

class AnalyzeRequest(BaseModel):
    target_col: Optional[str] = None
    exclude_cols: Optional[List[str]] = None

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
        generate_profile_celery.delay(report.id)
    else:
        # Use BackgroundTasks
        background_tasks.add_task(run_eda_background, report.id)
        
    return {"job_id": report.id, "status": "PENDING"}

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
