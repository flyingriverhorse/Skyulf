"""Pipeline save / load / versions endpoints (E9 phase 2).

Self-contained sub-router included by `backend.ml_pipeline.api` so the
public URL surface is unchanged. Owns no business logic of its own —
just translates HTTP requests into `PipelineVersionsService` calls
plus a small JSON-on-disk fallback for the "json" storage backend.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database.engine import get_async_session
from backend.database.models import FeatureEngineeringPipeline
from backend.exceptions.core import SkyulfException
from backend.ml_pipeline._internal._schemas import (
    PipelineVersionCreateModel,
    PipelineVersionPatchModel,
    SavedPipelineModel,
)
from backend.ml_pipeline._services.pipeline_versions_service import (
    PipelineVersionsService,
)

logger = logging.getLogger(__name__)

# No prefix — mounted by `api.py` under the same root as the legacy router
# so all paths (`/save/...`, `/load/...`, `/versions/...`) stay byte-identical.
router = APIRouter(tags=["ML Pipeline"])


@router.post("/save/{dataset_id}")
async def save_pipeline(
    dataset_id: str,
    payload: SavedPipelineModel,
    session: AsyncSession = Depends(get_async_session),
):
    """Save the pipeline configuration (DB or on-disk JSON per settings)."""
    settings = get_settings()

    if settings.PIPELINE_STORAGE_TYPE == "json":
        storage_dir = settings.PIPELINE_STORAGE_PATH
        os.makedirs(storage_dir, exist_ok=True)
        file_path = os.path.join(storage_dir, f"{dataset_id}.json")
        try:
            with open(file_path, "w") as f:
                json.dump(payload.model_dump(), f, indent=2)
            return {"status": "success", "id": dataset_id, "storage": "json"}
        except Exception as e:
            raise SkyulfException(message=f"Failed to save pipeline to JSON: {str(e)}")

    # Default: Database Storage
    try:
        stmt = select(FeatureEngineeringPipeline).where(
            FeatureEngineeringPipeline.dataset_source_id == dataset_id,
            FeatureEngineeringPipeline.is_active,
        )
        result = await session.execute(stmt)
        existing_pipeline = result.scalar_one_or_none()

        if existing_pipeline:
            cast(Any, existing_pipeline).graph = payload.graph
            cast(Any, existing_pipeline).name = payload.name
            if payload.description:
                cast(Any, existing_pipeline).description = payload.description
        else:
            new_pipeline = FeatureEngineeringPipeline(
                dataset_source_id=dataset_id,
                name=payload.name,
                description=payload.description,
                graph=payload.graph,
                is_active=True,
            )
            session.add(new_pipeline)

        await session.commit()

        # L7: stamp a server-side version snapshot every successful save.
        # Best-effort — version persistence must never break Save itself.
        try:
            await PipelineVersionsService.create_version(
                session=session,
                dataset_source_id=dataset_id,
                graph=payload.graph,
                name=payload.name,
                kind="manual",
                note=payload.note,
                dataset_name=payload.dataset_name,
            )
        except Exception as ver_err:  # noqa: BLE001
            logger.warning(
                "Failed to write pipeline_version snapshot for %s: %s",
                dataset_id,
                ver_err,
            )

        return {"status": "success", "id": dataset_id, "storage": "database"}
    except Exception as e:
        await session.rollback()
        raise SkyulfException(message=f"Failed to save pipeline: {str(e)}")


@router.get("/load/{dataset_id}")
async def load_pipeline(
    dataset_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Load the pipeline configuration (DB or on-disk JSON per settings)."""
    settings = get_settings()

    if settings.PIPELINE_STORAGE_TYPE == "json":
        storage_dir = settings.PIPELINE_STORAGE_PATH
        file_path = os.path.join(storage_dir, f"{dataset_id}.json")
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise SkyulfException(message=f"Failed to load pipeline from JSON: {str(e)}")

    # Default: Database Storage
    try:
        stmt = select(FeatureEngineeringPipeline).where(
            FeatureEngineeringPipeline.dataset_source_id == dataset_id,
            FeatureEngineeringPipeline.is_active,
        )
        result = await session.execute(stmt)
        pipeline = result.scalar_one_or_none()
        if not pipeline:
            return None
        return pipeline.to_dict()
    except Exception as e:
        raise SkyulfException(message=f"Failed to load pipeline: {str(e)}")


# --- L7: Server-side pipeline versioning ---
#
# Replaces the per-browser localStorage Recent ring buffer with a
# durable, cross-device history. Routes mirror the shape of
# `frontend/ml-canvas/src/core/utils/recentPipelines.ts` so the
# frontend swap is mechanical.


@router.get("/versions/{dataset_source_id}")
async def list_pipeline_versions(
    dataset_source_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> List[Dict[str, Any]]:
    """List all snapshots for a dataset (pinned first, newest first)."""
    versions = await PipelineVersionsService.list_versions(session, dataset_source_id)
    return [v.to_dict() for v in versions]


@router.post("/versions/{dataset_source_id}")
async def create_pipeline_version(
    dataset_source_id: str,
    payload: PipelineVersionCreateModel,
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    """Explicitly create a snapshot. `kind` defaults to 'manual'; pass
    'auto' from background callers (e.g. successful Run hooks)."""
    try:
        version = await PipelineVersionsService.create_version(
            session=session,
            dataset_source_id=dataset_source_id,
            graph=payload.graph,
            name=payload.name,
            kind=payload.kind,
            note=payload.note,
            dataset_name=payload.dataset_name,
            pinned=payload.pinned,
        )
        return version.to_dict()
    except Exception as e:  # noqa: BLE001
        await session.rollback()
        raise SkyulfException(message=f"Failed to create pipeline version: {str(e)}")


@router.patch("/versions/{dataset_source_id}/{version_id}")
async def update_pipeline_version(
    dataset_source_id: str,
    version_id: int,
    payload: PipelineVersionPatchModel,
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    """Toggle pin, rename, or edit the note on a snapshot."""
    version = await PipelineVersionsService.get_version(session, version_id)
    if version is None or version.dataset_source_id != dataset_source_id:
        raise HTTPException(status_code=404, detail="Version not found")
    updated = await PipelineVersionsService.update_version(
        session,
        version_id,
        name=payload.name,
        note=payload.note,
        pinned=payload.pinned,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return updated.to_dict()


@router.delete("/versions/{dataset_source_id}/{version_id}")
async def delete_pipeline_version(
    dataset_source_id: str,
    version_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    """Hard-delete a snapshot. Pinned rows are not protected from
    explicit user deletion (matches the localStorage behavior)."""
    version = await PipelineVersionsService.get_version(session, version_id)
    if version is None or version.dataset_source_id != dataset_source_id:
        raise HTTPException(status_code=404, detail="Version not found")
    ok = await PipelineVersionsService.delete_version(session, version_id)
    return {"status": "success" if ok else "not_found", "id": version_id}


__all__ = ["router"]
