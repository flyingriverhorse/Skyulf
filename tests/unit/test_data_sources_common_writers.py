"""The shared ``data_sources`` writers, executed against a real SQLite session.

``test_data_sources_unscoped_write_guard`` pins what ``_common``'s writers refuse
to do; this module covers what they do. Nothing else in the suite runs their
bodies — the guard tests replace ``async_session_or_connection`` with a mock and
assert it is never called — so statement building, the commit, the reported
``affected_rows``, and the rollback-and-reraise path are exercised here against
an in-memory database holding the real ``data_sources`` table.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.config import get_settings
from backend.database.data_sources import _common
from backend.database.data_sources import async_postgres_queries as pg_q
from backend.database.data_sources import async_sqlite_queries as sqlite_q
from backend.database.models import Base, DataSource

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def session(monkeypatch):
    """Provide an in-memory SQLite session holding the real ``data_sources`` table.

    ``_common.async_session_or_connection`` is redirected to yield this same
    session, so the writers run their own statements against a live database.
    """
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:

        @asynccontextmanager
        async def _same_session(_settings, _config=None):
            yield s

        monkeypatch.setattr(_common, "async_session_or_connection", _same_session)
        yield s

    await engine.dispose()


async def _add_source(session: AsyncSession, source_id: str, name: str, type_: str = "postgres"):
    """Insert one committed ``data_sources`` row.

    Committed, so a writer's rollback cannot erase the fixture and make an
    "unchanged row" assertion pass for the wrong reason.
    """
    session.add(
        DataSource(source_id=source_id, name=name, type=type_, config={"host": "localhost"})
    )
    await session.commit()


async def _names_by_source_id(session: AsyncSession) -> dict[str, str]:
    """Return the table's contents as ``{source_id: name}``."""
    rows = (await session.execute(select(DataSource.source_id, DataSource.name))).all()
    return dict(rows)


def test_both_engine_modules_re_export_the_one_shared_implementation():
    """``_DB_PEERS`` dispatch and the ``sqlite_*``/``postgres_*`` names need one target."""
    for mod in (sqlite_q, pg_q):
        assert mod.update_data_source is _common.update_data_source
        assert mod.delete_data_source is _common.delete_data_source


async def test_update_rewrites_the_matching_row_and_reports_it(session):
    """A scoped update writes the row it names and reports exactly one affected row."""
    await _add_source(session, "a", "first")
    await _add_source(session, "b", "second")

    result = await _common.update_data_source(get_settings(), {"source_id": "a"}, {"name": "x"})

    assert result == {"affected_rows": 1}
    assert await _names_by_source_id(session) == {"a": "x", "b": "second"}


async def test_update_requires_every_filter_key_to_match(session):
    """The filter keys are ANDed, so a row satisfying only one of them is untouched."""
    await _add_source(session, "a", "first", type_="postgres")

    result = await _common.update_data_source(
        get_settings(), {"source_id": "a", "type": "snowflake"}, {"name": "x"}
    )

    assert result == {"affected_rows": 0}
    assert await _names_by_source_id(session) == {"a": "first"}


async def test_delete_removes_only_the_filtered_row(session):
    """A delete scoped to one ``source_id`` must leave the other row in the table."""
    await _add_source(session, "a", "first")
    await _add_source(session, "b", "second")

    result = await _common.delete_data_source(get_settings(), {"source_id": "b"})

    assert result == {"affected_rows": 1}
    assert await _names_by_source_id(session) == {"a": "first"}


async def test_a_failed_update_rolls_back_and_reraises(session):
    """A statement error surfaces to the caller and leaves the row as it was."""
    await _add_source(session, "a", "first")

    with pytest.raises(OperationalError):
        await _common.update_data_source(
            get_settings(), {"source_id": "a"}, {"no_such_column": "x"}
        )

    assert await _names_by_source_id(session) == {"a": "first"}


async def test_a_failed_delete_rolls_back_and_reraises(session):
    """Same on the delete path: the error surfaces and no row is lost."""
    await _add_source(session, "a", "first")
    await _add_source(session, "b", "second")

    with pytest.raises(OperationalError):
        await _common.delete_data_source(get_settings(), {"no_such_column": "x"})

    assert await _names_by_source_id(session) == {"a": "first", "b": "second"}


@pytest.mark.parametrize(
    ("call", "args"),
    [
        (_common.update_data_source, ({"source_id": "a"}, {"name": "x"})),
        (_common.delete_data_source, ({"source_id": "a"},)),
    ],
    ids=["update", "delete"],
)
async def test_the_error_handler_rolls_back_before_reraising(monkeypatch, call, args):
    """A statement error must roll the session back and let the original error through.

    The real-database tests above cannot tell "rolled back" from "never applied" —
    both leave the row intact — so the handler's own contract is asserted here.
    """
    failing = AsyncMock()
    failing.execute.side_effect = OperationalError("stmt", {}, Exception("no such column"))

    @asynccontextmanager
    async def _failing_session(_settings, _config=None):
        yield failing

    monkeypatch.setattr(_common, "async_session_or_connection", _failing_session)

    with pytest.raises(OperationalError):
        await call(get_settings(), *args)

    failing.rollback.assert_awaited_once()
    failing.commit.assert_not_awaited()
