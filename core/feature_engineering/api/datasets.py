"""Dataset API endpoints."""

from datetime import datetime
from typing import List, Optional, cast

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.engine import get_async_session
from core.database.models import DataSource
from core.feature_engineering.schemas import DatasetSourceSummary

router = APIRouter()


@router.get("/api/datasets", response_model=List[DatasetSourceSummary])
async def list_active_datasets(
    limit: int = Query(8, ge=1, le=50),
    session: AsyncSession = Depends(get_async_session),
) -> List[DatasetSourceSummary]:
    """Return a trimmed list of active dataset sources for quick selection."""

    limit_value = max(1, min(limit, 50))

    result = await session.execute(
        select(DataSource)
        .where(DataSource.is_active.is_(True))
        .order_by(DataSource.updated_at.desc(), DataSource.id.desc())
        .limit(limit_value)
    )

    datasets = result.scalars().all()

    summaries: List[DatasetSourceSummary] = []
    for dataset in datasets:
        dataset_id = cast(Optional[int], dataset.id)
        if dataset_id is None:
            continue
        source_id_value = cast(Optional[str], dataset.source_id) or str(dataset_id)
        summaries.append(
            DatasetSourceSummary(
                id=int(dataset_id),
                source_id=source_id_value,
                name=cast(Optional[str], dataset.name),
                description=cast(Optional[str], dataset.description),
                created_at=cast(Optional[datetime], dataset.created_at),
            )
        )

    return summaries
