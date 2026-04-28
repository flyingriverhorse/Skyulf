"""
Pipeline Versions Service (L7)
------------------------------

CRUD over `PipelineVersion` rows. Replaces the per-browser localStorage
"Recent" ring buffer with a durable, server-side history.

Versions are keyed by `dataset_source_id` (not `pipelines.id`) because
`FeatureEngineeringPipeline` is upserted-by-dataset today; one active
pipeline per dataset means dataset_source_id is the natural identity.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import PipelineVersion

logger = logging.getLogger(__name__)


def _count_graph(graph: Any) -> tuple[int, int]:
    """Best-effort node/edge counts. Tolerates either RF snapshot shape
    ({nodes, edges}) or engine config shape (list of nodes)."""
    if isinstance(graph, dict):
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        n = len(nodes) if isinstance(nodes, list) else 0
        e = len(edges) if isinstance(edges, list) else 0
        return n, e
    if isinstance(graph, list):
        return len(graph), 0
    return 0, 0


class PipelineVersionsService:
    """Async CRUD for pipeline_versions."""

    @staticmethod
    async def create_version(
        session: AsyncSession,
        *,
        dataset_source_id: str,
        graph: Any,
        name: str,
        kind: str = "manual",
        note: Optional[str] = None,
        dataset_name: Optional[str] = None,
        user_id: Optional[int] = None,
        pinned: bool = False,
    ) -> PipelineVersion:
        # Next version_int = max+1 for this dataset.
        stmt = select(PipelineVersion.version_int).where(
            PipelineVersion.dataset_source_id == dataset_source_id
        )
        result = await session.execute(stmt)
        existing = [row[0] for row in result.all()]
        next_int = (max(existing) + 1) if existing else 1

        node_count, edge_count = _count_graph(graph)
        version = PipelineVersion(
            dataset_source_id=dataset_source_id,
            version_int=next_int,
            name=name,
            note=note,
            kind=kind,
            pinned=pinned,
            graph=graph,
            node_count=node_count,
            edge_count=edge_count,
            dataset_name=dataset_name,
            user_id=user_id,
        )
        session.add(version)
        await session.commit()
        await session.refresh(version)
        return version

    @staticmethod
    async def list_versions(
        session: AsyncSession, dataset_source_id: str
    ) -> List[PipelineVersion]:
        # Pinned first, then newest first.
        stmt = (
            select(PipelineVersion)
            .where(PipelineVersion.dataset_source_id == dataset_source_id)
            .order_by(
                PipelineVersion.pinned.desc(),
                PipelineVersion.version_int.desc(),
            )
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_version(
        session: AsyncSession, version_id: int
    ) -> Optional[PipelineVersion]:
        return await session.get(PipelineVersion, version_id)

    @staticmethod
    async def update_version(
        session: AsyncSession,
        version_id: int,
        *,
        name: Optional[str] = None,
        note: Optional[str] = None,
        pinned: Optional[bool] = None,
    ) -> Optional[PipelineVersion]:
        version = await session.get(PipelineVersion, version_id)
        if version is None:
            return None
        if name is not None:
            trimmed = name.strip()
            if trimmed:
                version.name = trimmed
        if note is not None:
            version.note = note or None
        if pinned is not None:
            version.pinned = pinned
        await session.commit()
        await session.refresh(version)
        return version

    @staticmethod
    async def delete_version(session: AsyncSession, version_id: int) -> bool:
        version = await session.get(PipelineVersion, version_id)
        if version is None:
            return False
        await session.delete(version)
        await session.commit()
        return True
