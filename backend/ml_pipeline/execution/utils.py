from typing import Optional, Dict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.models import DataSource

async def resolve_dataset_name(session: AsyncSession, dataset_source_id: Optional[str]) -> Optional[str]:
    """
    Resolves the dataset name from a dataset_source_id.
    Handles both integer IDs (as strings) and UUIDs.
    """
    if not dataset_source_id:
        return None

    # Try finding by ID (int) or source_id (UUID)
    is_digit = str(dataset_source_id).isdigit()
    ds_id_val = int(dataset_source_id) if is_digit else -1
    
    ds_stmt = select(DataSource.name).where(
        (DataSource.source_id == str(dataset_source_id)) | 
        (DataSource.id == ds_id_val)
    )
    ds_result = await session.execute(ds_stmt)
    dataset_name = ds_result.scalar_one_or_none()
    
    if not dataset_name:
        dataset_name = f"Dataset {dataset_source_id}"
        
    return dataset_name

async def get_dataset_map(session: AsyncSession) -> Dict[str, str]:
    """
    Returns a map of dataset IDs (and UUIDs) to dataset names.
    Useful for bulk resolution.
    """
    ds_result = await session.execute(select(DataSource.id, DataSource.source_id, DataSource.name))
    ds_map = {}
    for ds_id, ds_uuid, ds_name in ds_result.all():
        ds_map[str(ds_id)] = ds_name
        if ds_uuid:
            ds_map[str(ds_uuid)] = ds_name
    return ds_map
