"""Tests for the OPS-004 Error Log search/facets/severity behavior.

Covers `GET /monitoring/errors` and `GET /monitoring/pipeline-logs`: the
server-side generic-search (`q`) matching exact HTTP `job_id` / pipeline
`node_id` values, the typed severity/error-type/job-id/node-id facets
computed over the full unfiltered history, and the `total`/`total_unfiltered`
split that lets the UI distinguish "no history" from "no match".
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.monitoring.router import (
    _classify_error_severity,
    list_error_events,
    list_pipeline_logs,
)


def _make_error_event(**kwargs) -> MagicMock:
    """Build a lightweight mock standing in for an ErrorEvent row."""
    row = MagicMock()
    row.id = kwargs.get("id", 1)
    row.route = kwargs.get("route", "/api/pipeline/run")
    row.error_type = kwargs.get("error_type", "ValueError")
    row.message = kwargs.get("message", "boom")
    row.traceback = kwargs.get("traceback", "Traceback...")
    row.job_id = kwargs.get("job_id", "job-abc-123")
    row.status_code = kwargs.get("status_code", 500)
    row.created_at = kwargs.get("created_at", datetime(2026, 8, 7, 10, 0, tzinfo=UTC))
    row.resolved_at = kwargs.get("resolved_at")
    row.to_dict.return_value = {
        "id": row.id,
        "route": row.route,
        "error_type": row.error_type,
        "message": row.message,
        "traceback": row.traceback,
        "job_id": row.job_id,
        "status_code": row.status_code,
        "created_at": row.created_at.isoformat(),
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
    }
    return row


def _make_pipeline_log(**kwargs) -> MagicMock:
    """Build a lightweight mock standing in for a PipelineRunLog row."""
    row = MagicMock()
    row.id = kwargs.get("id", 1)
    row.pipeline_id = kwargs.get("pipeline_id", "pipeline-1")
    row.node_id = kwargs.get("node_id", "node-xyz")
    row.node_type = kwargs.get("node_type", "encoder")
    row.level = kwargs.get("level", "error")
    row.logger = kwargs.get("logger", "skyulf")
    row.message = kwargs.get("message", "node failed")
    row.run_at = kwargs.get("run_at", datetime(2026, 8, 7, 10, 5, tzinfo=UTC))
    row.to_dict.return_value = {
        "id": row.id,
        "pipeline_id": row.pipeline_id,
        "node_id": row.node_id,
        "node_type": row.node_type,
        "level": row.level,
        "logger": row.logger,
        "message": row.message,
        "run_at": row.run_at.isoformat(),
    }
    return row


def _make_db(rows: list) -> AsyncMock:
    """Build a mock `db.execute(...)` result whose `.scalars().all()` returns rows."""
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    db.execute.return_value = result
    return db


class TestClassifyErrorSeverity:
    """5xx/4xx/else -> critical/warning/info, used to unify the HTTP taxonomy."""

    @pytest.mark.parametrize(
        ("status_code", "expected"),
        [
            (500, "critical"),
            (503, "critical"),
            (404, "warning"),
            (400, "warning"),
            (0, "info"),
            (200, "info"),
        ],
    )
    def test_classification(self, status_code: int, expected: str) -> None:
        assert _classify_error_severity(status_code) == expected


class TestListErrorEventsSearch:
    """Generic `q` search must still match exact HTTP job_id values."""

    @pytest.mark.asyncio
    async def test_q_matches_exact_job_id(self) -> None:
        db = _make_db([_make_error_event(job_id="job-abc-123")])
        response = await list_error_events(q="job-abc-123", db=db)
        assert response.total == 1
        assert response.entries[0].job_id == "job-abc-123"

    @pytest.mark.asyncio
    async def test_q_matches_message_case_insensitively(self) -> None:
        db = _make_db([_make_error_event(message="Disk Full")])
        response = await list_error_events(q="disk full", db=db)
        assert response.total == 1

    @pytest.mark.asyncio
    async def test_q_excludes_non_matching_events(self) -> None:
        db = _make_db([_make_error_event(job_id="job-abc-123", message="boom")])
        response = await list_error_events(q="no-such-thing", db=db)
        assert response.total == 0
        assert response.total_unfiltered == 1


class TestListErrorEventsFacets:
    """Typed severity/error_type/job_id facets are computed over the *full* history."""

    @pytest.mark.asyncio
    async def test_facets_cover_full_unfiltered_history(self) -> None:
        rows = [
            _make_error_event(id=1, status_code=500, error_type="ValueError", job_id="job-1"),
            _make_error_event(id=2, status_code=404, error_type="KeyError", job_id="job-2"),
        ]
        db = _make_db(rows)
        response = await list_error_events(severity="critical", db=db)
        # Only one event matches the filter...
        assert response.total == 1
        # ...but facets still list every value across the whole history.
        assert response.facets.severities == ["critical", "warning"]
        assert response.facets.error_types == ["KeyError", "ValueError"]
        assert response.facets.job_ids == ["job-1", "job-2"]
        assert response.total_unfiltered == 2

    @pytest.mark.asyncio
    async def test_severity_filter_applies_server_side(self) -> None:
        rows = [
            _make_error_event(id=1, status_code=500),
            _make_error_event(id=2, status_code=404),
        ]
        db = _make_db(rows)
        response = await list_error_events(severity="warning", db=db)
        assert response.total == 1
        assert response.entries[0].id == 2
        assert response.entries[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_error_type_filter_applies_server_side(self) -> None:
        rows = [
            _make_error_event(id=1, error_type="ValueError"),
            _make_error_event(id=2, error_type="KeyError"),
        ]
        db = _make_db(rows)
        response = await list_error_events(error_type="KeyError", db=db)
        assert response.total == 1
        assert response.entries[0].error_type == "KeyError"

    @pytest.mark.asyncio
    async def test_job_id_filter_applies_server_side(self) -> None:
        rows = [
            _make_error_event(id=1, job_id="job-1"),
            _make_error_event(id=2, job_id="job-2"),
        ]
        db = _make_db(rows)
        response = await list_error_events(job_id="job-2", db=db)
        assert response.total == 1
        assert response.entries[0].job_id == "job-2"


class TestListErrorEventsEmptyStates:
    """total_unfiltered lets the UI distinguish "no history" from "no match"."""

    @pytest.mark.asyncio
    async def test_no_history_recorded(self) -> None:
        db = _make_db([])
        response = await list_error_events(db=db)
        assert response.total == 0
        assert response.total_unfiltered == 0

    @pytest.mark.asyncio
    async def test_no_match_for_active_filters(self) -> None:
        db = _make_db([_make_error_event()])
        response = await list_error_events(q="nothing-matches-this", db=db)
        assert response.total == 0
        assert response.total_unfiltered == 1


class TestListPipelineLogsSearch:
    """Generic `q` search must still match exact pipeline node_id values."""

    @pytest.mark.asyncio
    async def test_q_matches_exact_node_id(self) -> None:
        db = _make_db([_make_pipeline_log(node_id="node-xyz")])
        response = await list_pipeline_logs(q="node-xyz", db=db)
        assert response.total == 1
        assert response.entries[0].node_id == "node-xyz"

    @pytest.mark.asyncio
    async def test_node_id_facet_filters_server_side(self) -> None:
        rows = [
            _make_pipeline_log(id=1, node_id="node-a"),
            _make_pipeline_log(id=2, node_id="node-b"),
        ]
        db = _make_db(rows)
        response = await list_pipeline_logs(node_id="node-b", db=db)
        assert response.total == 1
        assert response.entries[0].node_id == "node-b"
        assert response.facets.node_ids == ["node-a", "node-b"]

    @pytest.mark.asyncio
    async def test_level_facet_filters_server_side(self) -> None:
        rows = [
            _make_pipeline_log(id=1, level="error"),
            _make_pipeline_log(id=2, level="warning"),
        ]
        db = _make_db(rows)
        response = await list_pipeline_logs(level="warning", db=db)
        assert response.total == 1
        assert response.entries[0].level == "warning"

    @pytest.mark.asyncio
    async def test_no_history_vs_no_match(self) -> None:
        empty_db = _make_db([])
        empty_response = await list_pipeline_logs(db=empty_db)
        assert empty_response.total_unfiltered == 0

        db = _make_db([_make_pipeline_log()])
        no_match_response = await list_pipeline_logs(q="nothing-matches-this", db=db)
        assert no_match_response.total == 0
        assert no_match_response.total_unfiltered == 1
