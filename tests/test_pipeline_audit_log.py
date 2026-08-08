"""Regression tests for the pipeline audit log endpoint.

The audit trail computes each version's node diff against its immediate
predecessor, so the endpoint depends on walking versions in true chronological
order. `PipelineVersionsService.list_versions` sorts pinned versions first for
the version-picker UI, which is *not* a chronological ordering — reversing that
list therefore does not reconstruct history.
"""

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from backend.ml_pipeline._internal._routers.pipelines_io import (
    ANONYMOUS_ACTOR,
    get_pipeline_audit_log,
)


def _version(
    version_int: int,
    nodes: list[str],
    *,
    pinned: bool = False,
    user_id: str = "tester",
    kind: str = "save",
    created_at: datetime | None = None,
) -> Any:
    """Build a minimal PipelineVersion stand-in with a graph of named nodes."""
    return SimpleNamespace(
        id=version_int,
        version_int=version_int,
        name=f"v{version_int}",
        note=None,
        kind=kind,
        user_id=user_id,
        created_at=created_at,
        node_count=len(nodes),
        edge_count=0,
        pinned=pinned,
        graph={"nodes": [{"id": n, "type": n, "data": {}} for n in nodes]},
    )


async def _run_audit(versions: list[Any], **params: Any) -> dict[str, Any]:
    with patch(
        "backend.ml_pipeline._internal._routers.pipelines_io.PipelineVersionsService.list_versions",
        new=AsyncMock(return_value=versions),
    ):
        return await get_pipeline_audit_log(
            "ds-1", limit=params.pop("limit", 50), session=AsyncMock(), **params
        )


@pytest.mark.asyncio
async def test_oldest_version_is_the_one_reported_as_all_added() -> None:
    """The first save has no predecessor, so only it may list every node as added."""
    # Service order: newest first (no pins).
    versions = [
        _version(3, ["a", "b", "c"]),
        _version(2, ["a", "b"]),
        _version(1, ["a"]),
    ]

    result = await _run_audit(versions)

    by_version = {e["version_int"]: e for e in result["entries"]}
    assert by_version[1]["diff"]["nodes_added"] == ["a"]
    assert by_version[2]["diff"]["nodes_added"] == ["b"]
    assert by_version[3]["diff"]["nodes_added"] == ["c"]


@pytest.mark.asyncio
async def test_pinned_version_does_not_corrupt_the_diff_walk() -> None:
    """A pinned version sorts first, which must not be mistaken for oldest-first.

    `list_versions` returns pinned-first, then newest-first. Reversing that list
    puts the pinned (here: oldest) version *last*, so it would be diffed against
    the newest graph and a mid-history version would be reported as the origin.
    """
    # Service order: v1 is pinned so it leads, then 3, 2 by recency.
    versions = [
        _version(1, ["a"], pinned=True),
        _version(3, ["a", "b", "c"]),
        _version(2, ["a", "b"]),
    ]

    result = await _run_audit(versions)

    by_version = {e["version_int"]: e for e in result["entries"]}
    assert by_version[1]["diff"]["nodes_added"] == ["a"]
    assert by_version[2]["diff"]["nodes_added"] == ["b"]
    assert by_version[3]["diff"]["nodes_added"] == ["c"]
    assert by_version[2]["diff"]["nodes_removed"] == []
    assert by_version[3]["diff"]["nodes_removed"] == []


@pytest.mark.asyncio
async def test_actor_filter_narrows_entries_without_losing_diff_context() -> None:
    """Filtering by actor must not change what each remaining entry's diff reports."""
    versions = [
        _version(3, ["a", "b", "c"], user_id="alice"),
        _version(2, ["a", "b"], user_id="bob"),
        _version(1, ["a"], user_id="alice"),
    ]

    result = await _run_audit(versions, actor="alice")

    assert [e["version_int"] for e in result["entries"]] == [3, 1]
    assert result["total"] == 2
    assert result["total_unfiltered"] == 3
    # v3's real change is "c" relative to v2, even though v2 is filtered out.
    by_version = {e["version_int"]: e for e in result["entries"]}
    assert by_version[3]["diff"]["nodes_added"] == ["c"]


@pytest.mark.asyncio
async def test_kind_filter_selects_only_matching_saves() -> None:
    versions = [
        _version(3, ["a", "b", "c"], kind="autosave"),
        _version(2, ["a", "b"], kind="save"),
        _version(1, ["a"], kind="save"),
    ]

    result = await _run_audit(versions, kind="save")

    assert [e["version_int"] for e in result["entries"]] == [2, 1]
    assert result["total"] == 2
    assert result["total_unfiltered"] == 3


@pytest.mark.asyncio
async def test_time_bounds_are_inclusive() -> None:
    versions = [
        _version(3, ["a", "b", "c"], created_at=datetime(2026, 3, 3)),
        _version(2, ["a", "b"], created_at=datetime(2026, 2, 2)),
        _version(1, ["a"], created_at=datetime(2026, 1, 1)),
    ]

    result = await _run_audit(
        versions, created_after="2026-02-02T00:00:00", created_before="2026-03-03T00:00:00"
    )

    assert [e["version_int"] for e in result["entries"]] == [3, 2]


@pytest.mark.asyncio
async def test_malformed_time_bound_is_rejected_rather_than_ignored() -> None:
    """A bad bound must fail loudly; silently returning everything would mislead."""
    versions = [_version(1, ["a"], created_at=datetime(2026, 1, 1))]

    with pytest.raises(HTTPException) as exc:
        await _run_audit(versions, created_after="last-tuesday")

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_filters_are_echoed_so_the_client_can_state_its_scope() -> None:
    versions = [_version(1, ["a"], user_id="alice")]

    result = await _run_audit(versions, actor="alice")

    assert result["filters"]["actor"] == "alice"
    assert result["filters"]["kind"] is None


@pytest.mark.asyncio
async def test_facets_cover_full_history_not_just_the_returned_page() -> None:
    """Dropdowns must offer every actor/kind, even when a filter hides them."""
    versions = [
        _version(3, ["a", "b", "c"], user_id="alice", kind="autosave"),
        _version(2, ["a", "b"], user_id="bob", kind="save"),
        _version(1, ["a"], user_id=None, kind="save"),
    ]

    result = await _run_audit(versions, actor="alice")

    assert result["facets"]["actors"] == ["alice", "bob"]
    assert result["facets"]["kinds"] == ["autosave", "save"]
    assert result["facets"]["has_anonymous_actor"] is True


@pytest.mark.asyncio
async def test_anonymous_sentinel_selects_saves_without_a_user_id() -> None:
    versions = [
        _version(2, ["a", "b"], user_id="bob"),
        _version(1, ["a"], user_id=None),
    ]

    result = await _run_audit(versions, actor=ANONYMOUS_ACTOR)

    assert [e["version_int"] for e in result["entries"]] == [1]


@pytest.mark.asyncio
async def test_integer_user_ids_match_their_query_string_form() -> None:
    """`user_id` is an int in the DB but arrives as a string from the query."""
    versions = [_version(2, ["a", "b"], user_id=7), _version(1, ["a"], user_id=9)]

    result = await _run_audit(versions, actor="7")

    assert [e["version_int"] for e in result["entries"]] == [2]
    assert result["facets"]["actors"] == ["7", "9"]
