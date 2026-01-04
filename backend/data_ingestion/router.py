from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from .dependencies import get_data_service
from .schemas.ingestion import (
    DataSourceCreate,
    DataSourceListResponse,
    DataSourceRead,
    DataSourceResponse,
    DataSourceSampleResponse,
    IngestionJobResponse,
    IngestionStatus,
)
from .service import DataIngestionService

router = APIRouter(prefix="/api/ingestion", tags=["Data Ingestion"])
sources_router = APIRouter(prefix="/data/api", tags=["Data Sources"])


@sources_router.get("/sources", response_model=DataSourceListResponse)
async def list_sources(
    limit: int = 50,
    skip: int = 0,
    service: DataIngestionService = Depends(get_data_service),
):
    """
    List all available data sources.
    """
    # TODO: Get real user ID from auth dependency. For now, list all sources.
    sources = await service.list_sources(user_id=None, limit=limit, skip=skip)
    return DataSourceListResponse(
        sources=[DataSourceRead.model_validate(s) for s in sources]
    )


@sources_router.get("/sources/usable", response_model=DataSourceListResponse)
async def list_usable_sources(
    service: DataIngestionService = Depends(get_data_service),
):
    """
    List only successfully ingested data sources.
    """
    sources = await service.list_usable_sources(user_id=None)
    return DataSourceListResponse(
        sources=[DataSourceRead.model_validate(s) for s in sources]
    )


@sources_router.get("/sources/{source_id}", response_model=DataSourceResponse)
async def get_source(
    source_id: str, service: DataIngestionService = Depends(get_data_service)
):
    """
    Get a specific data source.
    """
    source = await service.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return DataSourceResponse(source=source)


@sources_router.get(
    "/sources/{source_id}/sample", response_model=DataSourceSampleResponse
)
async def get_source_sample(
    source_id: str,
    limit: int = 5,
    service: DataIngestionService = Depends(get_data_service),
):
    """
    Get a sample of data from the source.
    """
    data = await service.get_sample(source_id, limit)
    return DataSourceSampleResponse(data=data)


@sources_router.delete("/sources/{source_id}")
async def delete_source(
    source_id: str, service: DataIngestionService = Depends(get_data_service)
):
    """
    Delete a data source.
    """
    success = await service.delete_source(source_id)
    if not success:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"message": "Source deleted successfully"}


@router.post("/database", response_model=IngestionJobResponse)
async def create_database_source(
    data: DataSourceCreate,
    background_tasks: BackgroundTasks,
    service: DataIngestionService = Depends(get_data_service),
):
    """
    Create a database source and start ingestion.
    """
    # TODO: Get real user ID from auth dependency
    user_id = 1
    return await service.create_database_source(data, user_id, background_tasks)


@router.post("/upload", response_model=IngestionJobResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service: DataIngestionService = Depends(get_data_service),
):
    """
    Upload a file and start ingestion process.
    """
    # TODO: Get real user ID from auth dependency
    user_id = 1
    return await service.handle_file_upload(file, user_id, background_tasks)


@router.get("/{source_id}/status", response_model=IngestionStatus)
async def get_status(
    source_id: int, service: DataIngestionService = Depends(get_data_service)
):
    """
    Get the status of an ingestion job.
    """
    status_data = await service.get_ingestion_status(source_id)
    return IngestionStatus(**status_data)


@router.post("/{source_id}/cancel")
async def cancel_ingestion(
    source_id: str, service: DataIngestionService = Depends(get_data_service)
):
    """
    Cancel an ingestion job.
    """
    success = await service.cancel_ingestion(source_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job could not be cancelled (maybe it's already finished or doesn't exist)",
        )
    return {"message": "Ingestion job cancelled successfully"}
