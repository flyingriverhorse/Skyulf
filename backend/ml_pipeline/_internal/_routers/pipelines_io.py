"""Pipeline save / load / versions endpoints (E9 phase 2).

Self-contained sub-router included by `backend.ml_pipeline.api` so the
public URL surface is unchanged. Owns no business logic of its own —
just translates HTTP requests into `PipelineVersionsService` calls
plus a small JSON-on-disk fallback for the "json" storage backend.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import aiofiles
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

# `dataset_id` comes straight from the URL path and is used to build a
# filename for the on-disk "json" storage backend. Only allow the charset
# real dataset ids use (alphanumerics, dash, underscore) -- this rejects
# path separators, "..", null bytes, and any other filesystem-meaningful
# character outright, so the on-disk path can never leave `storage_dir`
# regardless of how it's later joined/formatted.
_SAFE_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Sentinel the audit-log `actor` filter uses to mean "saves with no user_id".
ANONYMOUS_ACTOR = "__anonymous__"


def _pipeline_json_path(storage_dir: str | Path, dataset_id: str) -> Path:
    """Return the on-disk JSON path for `dataset_id`, or raise ValueError.

    `dataset_id` must match `_SAFE_DATASET_ID_RE`; anything else (path
    separators, "..", absolute-path overrides, etc.) is rejected before any
    `Path` is built from it.
    """
    if not _SAFE_DATASET_ID_RE.fullmatch(dataset_id):
        raise ValueError(f"Invalid dataset_id: {dataset_id!r}")
    return Path(storage_dir) / f"{dataset_id}.json"


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
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        try:
            file_path = _pipeline_json_path(storage_dir, dataset_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid dataset_id") from e
        try:
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(payload.model_dump(), indent=2))
            return {"status": "success", "id": dataset_id, "storage": "json"}
        except Exception as e:
            raise SkyulfException(message=f"Failed to save pipeline to JSON: {str(e)}") from e

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
        raise SkyulfException(message=f"Failed to save pipeline: {str(e)}") from e


@router.get("/load/{dataset_id}")
async def load_pipeline(
    dataset_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Load the pipeline configuration (DB or on-disk JSON per settings)."""
    settings = get_settings()

    if settings.PIPELINE_STORAGE_TYPE == "json":
        storage_dir = settings.PIPELINE_STORAGE_PATH
        try:
            file_path = _pipeline_json_path(storage_dir, dataset_id)
        except ValueError:
            return None
        if not file_path.exists():
            return None
        try:
            async with aiofiles.open(file_path) as f:
                return json.loads(await f.read())
        except Exception as e:
            raise SkyulfException(message=f"Failed to load pipeline from JSON: {str(e)}") from e

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
        raise SkyulfException(message=f"Failed to load pipeline: {str(e)}") from e


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
) -> list[dict[str, Any]]:
    """List all snapshots for a dataset (pinned first, newest first)."""
    versions = await PipelineVersionsService.list_versions(session, dataset_source_id)
    return [v.to_dict() for v in versions]


@router.post("/versions/{dataset_source_id}")
async def create_pipeline_version(
    dataset_source_id: str,
    payload: PipelineVersionCreateModel,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
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
        raise SkyulfException(message=f"Failed to create pipeline version: {str(e)}") from e


@router.patch("/versions/{dataset_source_id}/{version_id}")
async def update_pipeline_version(
    dataset_source_id: str,
    version_id: int,
    payload: PipelineVersionPatchModel,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    """Hard-delete a snapshot. Pinned rows are not protected from
    explicit user deletion (matches the localStorage behavior)."""
    version = await PipelineVersionsService.get_version(session, version_id)
    if version is None or version.dataset_source_id != dataset_source_id:
        raise HTTPException(status_code=404, detail="Version not found")
    ok = await PipelineVersionsService.delete_version(session, version_id)
    return {"status": "success" if ok else "not_found", "id": version_id}


# ---------------------------------------------------------------------------
# Audit log (read-only).
#
# We already store every save as an append-only `PipelineVersion` row with
# `user_id` + `created_at`. The audit endpoint walks that history in order
# and returns a per-version diff against the immediately-prior version, so
# admins can answer "who broke this pipeline, and when?" without a separate
# audit table or DB migration.
# ---------------------------------------------------------------------------


def _node_set(graph: Any) -> dict[str, dict[str, Any]]:
    """Index a saved graph by node id for O(1) diff lookups.

    Tolerates both the canonical `{nodes: [...], edges: [...]}` shape and
    legacy graphs missing the wrapper. Unknown shapes return an empty dict
    so the diff degrades to "no changes" rather than raising.
    """
    if not isinstance(graph, dict):
        return {}
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict):
            nid = n.get("id")
            if isinstance(nid, str):
                indexed[nid] = n
    return indexed


def _diff_versions(prev: Any, curr: Any) -> dict[str, Any]:
    """Compute a compact node-level diff between two saved graphs."""
    a = _node_set(prev)
    b = _node_set(curr)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    modified: list[str] = [
        nid
        for nid in set(a) & set(b)
        # Cheap structural compare — sufficient because saved graphs are
        # JSON round-tripped (no datetimes, no sets, no class instances).
        if json.dumps(a[nid], sort_keys=True) != json.dumps(b[nid], sort_keys=True)
    ]
    return {
        "nodes_added": added,
        "nodes_removed": removed,
        "nodes_modified": sorted(modified),
        "delta_node_count": len(b) - len(a),
    }


@router.get("/versions/{dataset_source_id}/audit")
async def get_pipeline_audit_log(
    dataset_source_id: str,
    limit: int | None = None,
    actor: str | None = None,
    kind: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Return a chronological audit trail for one dataset's pipeline.

    Each entry is a saved `PipelineVersion` augmented with a per-node diff
    against its immediate predecessor. The first version has no predecessor
    so its diff lists every node as `added`.

    Args:
        dataset_source_id: Dataset whose pipeline history is returned.
        limit: Maximum entries to return, capped at 200.
        actor: Restrict to saves made by this `user_id`, or `ANONYMOUS_ACTOR`
            to select saves that have no `user_id`.
        kind: Restrict to saves of this kind (e.g. `save`, `autosave`).
        created_after: ISO-8601 lower bound (inclusive) on `created_at`.
        created_before: ISO-8601 upper bound (inclusive) on `created_at`.

    Returns:
        A mapping with the dataset id, `total` matching the applied filters,
        `total_unfiltered`, `facets` of every actor/kind in the full history,
        the echoed `filters`, and the capped `entries`.

    Raises:
        HTTPException: If `created_after`/`created_before` is not ISO-8601.
    """
    default_limit = get_settings().DEFAULT_PAGE_SIZE
    capped_limit = max(1, min(int(limit or default_limit), 200))

    def _parse_bound(raw: str | None, field: str) -> datetime | None:
        if raw is None:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"{field} must be an ISO-8601 datetime, got {raw!r}",
            ) from exc

    after = _parse_bound(created_after, "created_after")
    before = _parse_bound(created_before, "created_before")

    versions = await PipelineVersionsService.list_versions(session, dataset_source_id)
    # `list_versions` sorts pinned-first for the version picker, which is not a
    # chronological ordering — reversing it would diff a version against the
    # wrong predecessor. Sort on `version_int` so each diff sees true history.
    chronological = sorted(versions, key=lambda v: v.version_int)
    entries: list[dict[str, Any]] = []
    prev_graph: Any = None
    for v in chronological:
        diff = _diff_versions(prev_graph, v.graph)
        entries.append(
            {
                "id": v.id,
                "version_int": v.version_int,
                "name": v.name,
                "note": v.note,
                "kind": v.kind,
                "user_id": v.user_id,
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "node_count": v.node_count,
                "edge_count": v.edge_count,
                "diff": diff,
            }
        )
        # Advance the baseline for every version, including filtered-out ones,
        # so a filtered view still reports each entry's real change.
        prev_graph = v.graph

    total_unfiltered = len(entries)
    # Facets are computed before filtering so the client's dropdowns list every
    # actor/kind in the dataset's history, not only those on the current page.
    # Actor ids are stringified because they arrive back as query-string values.
    facet_actors = sorted({str(e["user_id"]) for e in entries if e["user_id"] is not None})
    facet_kinds = sorted({e["kind"] for e in entries if e["kind"] is not None})
    has_anonymous_actor = any(e["user_id"] is None for e in entries)

    def _matches(entry: dict[str, Any]) -> bool:
        if actor is not None:
            # Saves made outside an authenticated session have no `user_id`;
            # ANONYMOUS_ACTOR is the only way for a client to select them.
            # `user_id` may be an int, so compare on its string form.
            if actor == ANONYMOUS_ACTOR:
                if entry["user_id"] is not None:
                    return False
            elif entry["user_id"] is None or str(entry["user_id"]) != actor:
                return False
        if kind is not None and entry["kind"] != kind:
            return False
        if after is not None or before is not None:
            raw = entry["created_at"]
            if raw is None:
                return False
            stamp = datetime.fromisoformat(raw)
            if after is not None and stamp < after:
                return False
            if before is not None and stamp > before:
                return False
        return True

    entries = [e for e in entries if _matches(e)]
    # Newest-first for UI consumption; cap after the diff walk so each
    # `diff` is computed against its true predecessor, not a window edge.
    entries.reverse()
    return {
        "dataset_source_id": dataset_source_id,
        "total": len(entries),
        "total_unfiltered": total_unfiltered,
        "facets": {
            "actors": facet_actors,
            "kinds": facet_kinds,
            "has_anonymous_actor": has_anonymous_actor,
        },
        "filters": {
            "actor": actor,
            "kind": kind,
            "created_after": created_after,
            "created_before": created_before,
        },
        "entries": entries[:capped_limit],
    }


__all__ = ["router"]
