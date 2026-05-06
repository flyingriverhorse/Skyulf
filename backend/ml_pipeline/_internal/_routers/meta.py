"""Read-only metadata endpoints (E9 phase 2).

`/registry`, `/stats`, `/datasets/list`, `/datasets/{id}/schema`,
`/hyperparameters/{model_type}`, `/hyperparameters/{model_type}/defaults`.

Pure read-side; no engine, no Celery, no locks.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from skyulf.modeling.hyperparameters import (
    get_default_search_space,
    get_hyperparameters,
)
from skyulf.registry import NodeRegistry as SkyulfRegistry
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.data.catalog import FileSystemCatalog
from backend.data_ingestion.service import DataIngestionService
from backend.database.engine import get_async_session
from backend.database.models import (
    AdvancedTuningJob,
    BasicTrainingJob,
    DataSource,
    Deployment,
)
from backend.exceptions.core import SkyulfException
from backend.ml_pipeline._internal._schemas import RegistryItem
from backend.ml_pipeline._internal._advisor import AnalysisProfile, DataProfiler
from backend.ml_pipeline.constants import StepType
from backend.utils.file_utils import extract_file_path_from_source

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Pipeline"])


@lru_cache(maxsize=1)
def _build_node_registry() -> List[RegistryItem]:
    """Merge the static DATA_LOADER entry with all skyulf-core registered nodes.

    Result is cached so the dict scan runs only once per process lifetime.
    """
    static: List[RegistryItem] = [
        RegistryItem(
            id=StepType.DATA_LOADER,
            name="Data Loader",
            category="Data Source",
            description="Loads data from a source.",
            params={"source_id": "string", "date_range": "optional[dict]"},
        ),
    ]
    dynamic: List[RegistryItem] = []
    for node_id, meta in SkyulfRegistry.get_all_metadata().items():
        item_data = dict(meta)
        if "id" not in item_data:
            item_data["id"] = node_id
        dynamic.append(RegistryItem(**item_data))

    dynamic_ids = {n.id for n in dynamic}
    return [n for n in static if n.id not in dynamic_ids] + dynamic


@router.get("/stats", response_model=Dict[str, int])
async def get_system_stats(session: AsyncSession = Depends(get_async_session)):
    """Return high-level system statistics for the dashboard."""
    training_count = await session.scalar(select(func.count(BasicTrainingJob.id)))
    tuning_count = await session.scalar(select(func.count(AdvancedTuningJob.id)))
    deployment_count = await session.scalar(
        select(func.count(Deployment.id)).where(Deployment.is_active)
    )
    datasource_count = await session.scalar(
        select(func.count(DataSource.id)).where(DataSource.test_status == "success")
    )
    return {
        "total_jobs": (training_count or 0) + (tuning_count or 0),
        "active_deployments": deployment_count or 0,
        "data_sources": datasource_count or 0,
        "training_jobs": training_count or 0,
        "tuning_jobs": tuning_count or 0,
    }


@router.get("/registry", response_model=List[RegistryItem])
def get_node_registry():
    """List available pipeline nodes (transformers, models, etc.)."""
    return _build_node_registry()


@router.get("/datasets/{dataset_id}/schema", response_model=AnalysisProfile)
async def get_dataset_schema(
    dataset_id: int, session: AsyncSession = Depends(get_async_session)
):
    """Return the schema (columns, types, stats) of a dataset.

    Prefers the cached profile in `DataSource.source_metadata['profile']`
    when present; falls back to a 1000-row sample profile.
    """
    ingestion_service = DataIngestionService(session)
    ds = await ingestion_service.get_source(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    if ds.source_metadata and "profile" in ds.source_metadata:
        try:
            cached_profile = ds.source_metadata["profile"]
            columns = {}
            for col_name, stats in cached_profile.get("columns", {}).items():
                dtype = str(stats.get("type", "unknown"))
                col_type = "unknown"
                if any(x in dtype for x in ["Int", "Float", "Decimal"]):
                    col_type = "numeric"
                elif any(x in dtype for x in ["Utf8", "String", "Categorical", "Object"]):
                    col_type = "categorical"
                elif "Date" in dtype or "Time" in dtype:
                    col_type = "datetime"
                elif "Bool" in dtype:
                    col_type = "boolean"
                columns[col_name] = {
                    "name": col_name,
                    "dtype": dtype,
                    "column_type": col_type,
                    "missing_count": stats.get("null_count", 0),
                    "missing_ratio": stats.get("null_percentage", 0) / 100.0,
                    "unique_count": stats.get("unique_count", 0),
                    "min_value": stats.get("min"),
                    "max_value": stats.get("max"),
                    "mean_value": stats.get("mean"),
                    "std_value": stats.get("std"),
                }
            return {
                "row_count": cached_profile.get("row_count", 0),
                "column_count": cached_profile.get("column_count", 0),
                "duplicate_row_count": cached_profile.get("duplicate_rows", 0),
                "columns": columns,
            }
        except Exception as e:
            logger.warning(f"Failed to parse cached profile for {dataset_id}: {e}")

    try:
        ds_dict = {
            "connection_info": ds.config,
            "file_path": ds.config.get("file_path") if ds.config else None,
        }
        path = extract_file_path_from_source(ds_dict)
        if not path:
            raise HTTPException(
                status_code=400,
                detail=f"Could not resolve path for dataset {dataset_id}",
            )
        catalog = FileSystemCatalog()
        df = catalog.load(str(path), limit=1000)
        return DataProfiler.generate_profile(df)
    except Exception as e:
        raise SkyulfException(message=f"Failed to profile dataset: {str(e)}")


@router.get("/hyperparameters/{model_type}")
def get_model_hyperparameters(model_type: str):
    """List tunable hyperparameters for a specific model type."""
    return get_hyperparameters(model_type)


@router.get("/hyperparameters/{model_type}/defaults")
def get_model_default_search_space(model_type: str, strategy: str = "random"):
    """Default search space for a model. `strategy` accepts random/grid/halving_grid."""
    return get_default_search_space(model_type, strategy=strategy)


@router.get("/datasets/list", response_model=List[Dict[str, Any]])
async def list_datasets(session: AsyncSession = Depends(get_async_session)):
    """Return a simple list of available datasets for filtering."""
    stmt = select(DataSource.source_id, DataSource.name).where(DataSource.is_active)
    result = await session.execute(stmt)
    return [{"id": row.source_id, "name": row.name} for row in result.all()]


__all__ = ["router", "_build_node_registry"]
