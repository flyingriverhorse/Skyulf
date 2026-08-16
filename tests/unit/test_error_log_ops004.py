"""Tests for the OPS-004 Error Log search/facets/severity behavior.

Covers `GET /monitoring/errors` and `GET /monitoring/pipeline-logs`: the
server-side generic-search (`q`) matching exact HTTP `job_id` / pipeline
`node_id` values, the typed severity/error-type/job-id/node-id facets
computed over the full unfiltered history, and the `total`/`total_unfiltered`
split that lets the UI distinguish "no history" from "no match".

These tests run against a real (in-memory SQLite) `AsyncSession` rather than a
mock, because the endpoints now issue several distinct SQL statements
(a bounded/filtered page query, a filtered `COUNT(*)`, an unfiltered
`COUNT(*)`, and per-facet `SELECT DISTINCT` queries) instead of a single
`SELECT *` — a single mocked `.scalars().all()` can no longer stand in for
every call.
"""

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, ErrorEvent, PipelineRunLog
from backend.monitoring.router import (
    _classify_error_severity,
    list_error_events,
    list_pipeline_logs,
)

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def async_session():
    """A fresh in-memory SQLite `AsyncSession` with the full ORM schema created."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


async def _seed_error_event(
    session: AsyncSession,
    *,
    route: str = "/api/pipeline/run",
    error_type: str = "ValueError",
    message: str = "boom",
    job_id: str | None = "job-abc-123",
    status_code: int = 500,
    created_at: datetime | None = None,
    resolved_at: datetime | None = None,
) -> ErrorEvent:
    row = ErrorEvent(
        route=route,
        error_type=error_type,
        message=message,
        traceback="Traceback...",
        job_id=job_id,
        status_code=status_code,
        created_at=created_at or datetime(2026, 8, 7, 10, 0),
        resolved_at=resolved_at,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def _seed_pipeline_log(
    session: AsyncSession,
    *,
    pipeline_id: str | None = "pipeline-1",
    node_id: str | None = "node-xyz",
    node_type: str | None = "encoder",
    level: str = "error",
    message: str = "node failed",
    run_at: datetime | None = None,
) -> PipelineRunLog:
    row = PipelineRunLog(
        pipeline_id=pipeline_id,
        node_id=node_id,
        node_type=node_type,
        level=level,
        logger="skyulf",
        message=message,
        run_at=run_at or datetime(2026, 8, 7, 10, 5),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


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
    async def test_q_matches_exact_job_id(self, async_session: AsyncSession) -> None:
        await _seed_error_event(async_session, job_id="job-abc-123")
        response = await list_error_events(q="job-abc-123", db=async_session)
        assert response.total == 1
        assert response.entries[0].job_id == "job-abc-123"

    @pytest.mark.asyncio
    async def test_q_matches_message_case_insensitively(self, async_session: AsyncSession) -> None:
        await _seed_error_event(async_session, message="Disk Full")
        response = await list_error_events(q="disk full", db=async_session)
        assert response.total == 1

    @pytest.mark.asyncio
    async def test_q_excludes_non_matching_events(self, async_session: AsyncSession) -> None:
        await _seed_error_event(async_session, job_id="job-abc-123", message="boom")
        response = await list_error_events(q="no-such-thing", db=async_session)
        assert response.total == 0
        assert response.total_unfiltered == 1

    @pytest.mark.asyncio
    async def test_q_with_wildcard_characters_is_treated_literally(
        self, async_session: AsyncSession
    ) -> None:
        """A literal `%`/`_` in `q` must not act as a SQL LIKE wildcard."""
        await _seed_error_event(async_session, message="disk 100%_full")
        matching = await list_error_events(q="100%_full", db=async_session)
        assert matching.total == 1
        non_matching = await list_error_events(q="100xfull", db=async_session)
        assert non_matching.total == 0


class TestListErrorEventsFacets:
    """Typed severity/error_type/job_id facets are computed over the *full* history."""

    @pytest.mark.asyncio
    async def test_facets_cover_full_unfiltered_history(self, async_session: AsyncSession) -> None:
        await _seed_error_event(
            async_session, status_code=500, error_type="ValueError", job_id="job-1"
        )
        await _seed_error_event(
            async_session, status_code=404, error_type="KeyError", job_id="job-2"
        )
        response = await list_error_events(severity="critical", db=async_session)
        # Only one event matches the filter...
        assert response.total == 1
        # ...but facets still list every value across the whole history.
        assert response.facets.severities == ["critical", "warning"]
        assert response.facets.error_types == ["KeyError", "ValueError"]
        assert response.facets.job_ids == ["job-1", "job-2"]
        assert response.total_unfiltered == 2

    @pytest.mark.asyncio
    async def test_severity_filter_applies_server_side(self, async_session: AsyncSession) -> None:
        row1 = await _seed_error_event(async_session, status_code=500)
        row2 = await _seed_error_event(async_session, status_code=404)
        response = await list_error_events(severity="warning", db=async_session)
        assert response.total == 1
        assert response.entries[0].id == row2.id
        assert response.entries[0].id != row1.id
        assert response.entries[0].severity == "warning"

    @pytest.mark.asyncio
    async def test_error_type_filter_applies_server_side(self, async_session: AsyncSession) -> None:
        await _seed_error_event(async_session, error_type="ValueError")
        await _seed_error_event(async_session, error_type="KeyError")
        response = await list_error_events(error_type="KeyError", db=async_session)
        assert response.total == 1
        assert response.entries[0].error_type == "KeyError"

    @pytest.mark.asyncio
    async def test_job_id_filter_applies_server_side(self, async_session: AsyncSession) -> None:
        await _seed_error_event(async_session, job_id="job-1")
        await _seed_error_event(async_session, job_id="job-2")
        response = await list_error_events(job_id="job-2", db=async_session)
        assert response.total == 1
        assert response.entries[0].job_id == "job-2"

    @pytest.mark.asyncio
    async def test_since_filter_applies_server_side(self, async_session: AsyncSession) -> None:
        await _seed_error_event(async_session, created_at=datetime(2026, 8, 1, 0, 0))
        await _seed_error_event(async_session, created_at=datetime(2026, 8, 10, 0, 0))
        response = await list_error_events(since="2026-08-05T00:00:00Z", db=async_session)
        assert response.total == 1
        assert response.total_unfiltered == 2

    @pytest.mark.asyncio
    async def test_show_resolved_defaults_to_unresolved_only(
        self, async_session: AsyncSession
    ) -> None:
        await _seed_error_event(async_session, resolved_at=None)
        await _seed_error_event(async_session, resolved_at=datetime(2026, 8, 7, 11, 0))
        default_response = await list_error_events(db=async_session)
        assert default_response.total == 1
        all_response = await list_error_events(show_resolved=True, db=async_session)
        assert all_response.total == 2


class TestListErrorEventsEmptyStates:
    """total_unfiltered lets the UI distinguish "no history" from "no match"."""

    @pytest.mark.asyncio
    async def test_no_history_recorded(self, async_session: AsyncSession) -> None:
        response = await list_error_events(db=async_session)
        assert response.total == 0
        assert response.total_unfiltered == 0

    @pytest.mark.asyncio
    async def test_no_match_for_active_filters(self, async_session: AsyncSession) -> None:
        await _seed_error_event(async_session)
        response = await list_error_events(q="nothing-matches-this", db=async_session)
        assert response.total == 0
        assert response.total_unfiltered == 1


class TestListErrorEventsScanIsBounded:
    """The endpoint must not materialize the full table into Python to page it."""

    @pytest.mark.asyncio
    async def test_limit_is_pushed_into_sql(self, async_session: AsyncSession) -> None:
        for i in range(20):
            await _seed_error_event(
                async_session,
                message=f"error {i}",
                created_at=datetime(2026, 8, 7, 10, 0) + timedelta(minutes=i),
            )

        select_statements: list[str] = []

        def _capture(conn, cursor, statement, parameters, context, executemany):
            if statement.strip().upper().startswith("SELECT"):
                select_statements.append(statement)

        sync_engine = async_session.bind.sync_engine
        event.listen(sync_engine, "before_cursor_execute", _capture)
        try:
            response = await list_error_events(limit=5, db=async_session)
        finally:
            event.remove(sync_engine, "before_cursor_execute", _capture)

        assert len(response.entries) == 5
        assert response.total_unfiltered == 20
        entries_selects = [s for s in select_statements if "error_events" in s and "LIMIT" in s]
        assert entries_selects, "expected a LIMIT-bound SELECT against error_events"

    @pytest.mark.asyncio
    async def test_total_unfiltered_is_a_count_not_a_full_materialization(
        self, async_session: AsyncSession
    ) -> None:
        for i in range(10):
            await _seed_error_event(async_session, message=f"error {i}")

        count_result = await async_session.execute(select(ErrorEvent.id))
        assert len(count_result.all()) == 10  # sanity check on seed data

        response = await list_error_events(limit=3, db=async_session)
        assert response.total_unfiltered == 10
        assert len(response.entries) == 3


class TestListPipelineLogsSearch:
    """Generic `q` search must still match exact pipeline node_id values."""

    @pytest.mark.asyncio
    async def test_q_matches_exact_node_id(self, async_session: AsyncSession) -> None:
        await _seed_pipeline_log(async_session, node_id="node-xyz")
        response = await list_pipeline_logs(q="node-xyz", db=async_session)
        assert response.total == 1
        assert response.entries[0].node_id == "node-xyz"

    @pytest.mark.asyncio
    async def test_node_id_facet_filters_server_side(self, async_session: AsyncSession) -> None:
        await _seed_pipeline_log(async_session, node_id="node-a")
        await _seed_pipeline_log(async_session, node_id="node-b")
        response = await list_pipeline_logs(node_id="node-b", db=async_session)
        assert response.total == 1
        assert response.entries[0].node_id == "node-b"
        assert response.facets.node_ids == ["node-a", "node-b"]

    @pytest.mark.asyncio
    async def test_level_facet_filters_server_side(self, async_session: AsyncSession) -> None:
        await _seed_pipeline_log(async_session, level="error")
        await _seed_pipeline_log(async_session, level="warning")
        response = await list_pipeline_logs(level="warning", db=async_session)
        assert response.total == 1
        assert response.entries[0].level == "warning"

    @pytest.mark.asyncio
    async def test_no_history_vs_no_match(self, async_session: AsyncSession) -> None:
        empty_response = await list_pipeline_logs(db=async_session)
        assert empty_response.total_unfiltered == 0

        await _seed_pipeline_log(async_session)
        no_match_response = await list_pipeline_logs(q="nothing-matches-this", db=async_session)
        assert no_match_response.total == 0
        assert no_match_response.total_unfiltered == 1


class TestListPipelineLogsScanIsBounded:
    """The endpoint must not materialize the full table into Python to page it."""

    @pytest.mark.asyncio
    async def test_limit_is_pushed_into_sql(self, async_session: AsyncSession) -> None:
        for i in range(15):
            await _seed_pipeline_log(
                async_session,
                message=f"node failed {i}",
                run_at=datetime(2026, 8, 7, 10, 0) + timedelta(minutes=i),
            )

        response = await list_pipeline_logs(limit=4, db=async_session)
        assert len(response.entries) == 4
        assert response.total_unfiltered == 15
