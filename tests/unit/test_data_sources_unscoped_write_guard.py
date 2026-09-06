"""Guards against unscoped writes to ``data_sources`` (OC-159).

``delete_data_source`` and ``update_data_source`` build their WHERE clause by
looping over ``filter_dict``, so an empty dict produces a statement with no
predicate at all: a full-table DELETE/UPDATE that still returns
``affected_rows`` as though that had been the requested operation. Both the
SQLite and the PostgreSQL query modules had the identical shape — identical
enough that the two are now one implementation in ``_common``, re-exported by
each engine module, so the guard cannot be fixed in one engine and missed in
the other.

These tests pin the hazard itself (the compiled SQL) as well as the guard, so
the reason the ``ValueError`` exists stays visible if the builder is refactored.
The session patches target ``_common`` because that is the module whose globals
the shared writers resolve ``async_session_or_connection`` from; patching an
engine module would leave ``assert_not_called`` passing vacuously.
"""

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import column, delete, table, update

from backend.config import get_settings
from backend.database.data_sources import _common
from backend.database.data_sources import async_data_sources_crud as crud
from backend.database.data_sources import async_postgres_queries as pg_q
from backend.database.data_sources import async_sqlite_queries as sqlite_q


@pytest.fixture
def settings():
    """Return the shared test Settings instance."""
    return get_settings()


def test_an_empty_filter_compiles_to_a_statement_with_no_where_clause():
    """No filter keys means no WHERE, which in SQL means every row.

    Mirrors how the query modules build their statements: the column list is
    derived from ``filter_dict`` (plus ``update_data`` on the update path), so an
    empty filter leaves nothing to constrain on. Asserted against the compiled
    statement rather than a database, so the hazard stays documented without any
    backend being reachable.
    """
    filter_dict: dict[str, str] = {}

    # delete_data_source: table(TABLE, *[column(c) for c in filter_dict])
    delete_tbl = table("data_sources", *[column(c) for c in filter_dict])
    assert "WHERE" not in str(delete(delete_tbl))

    # update_data_source: columns come from update_data first, then filter_dict
    update_data = {"name": "x"}
    update_tbl = table(
        "data_sources",
        *[column(c) for c in update_data] + [column(c) for c in filter_dict],
    )
    assert "WHERE" not in str(update(update_tbl).values(**update_data))

    # A populated filter is the only thing that puts the WHERE back.
    scoped = {"id": "a"}
    scoped_tbl = table("data_sources", *[column(c) for c in scoped])
    scoped_stmt = delete(scoped_tbl)
    for key, value in scoped.items():
        scoped_stmt = scoped_stmt.where(column(key) == value)
    assert "WHERE" in str(scoped_stmt)


@pytest.mark.parametrize("mod", [sqlite_q, pg_q], ids=["sqlite", "postgres"])
async def test_delete_rejects_an_empty_filter_before_opening_a_session(mod, settings):
    """An empty filter must raise, and must not start a transaction to do it.

    Both parameters resolve to the one shared writer in ``_common``; what the
    parameter still pins is that each engine module re-exports it, since
    ``_DB_PEERS`` dispatch and the public ``sqlite_*``/``postgres_*`` names both
    reach the guard through those attributes.
    """
    with (
        patch.object(_common, "async_session_or_connection", new=AsyncMock()) as session,
        pytest.raises(ValueError, match="non-empty filter_dict"),
    ):
        await mod.delete_data_source(settings, {})
    session.assert_not_called()


@pytest.mark.parametrize("mod", [sqlite_q, pg_q], ids=["sqlite", "postgres"])
async def test_update_rejects_an_empty_filter_before_opening_a_session(mod, settings):
    """Same guard on the update path: unscoped means every row is rewritten."""
    with (
        patch.object(_common, "async_session_or_connection", new=AsyncMock()) as session,
        pytest.raises(ValueError, match="non-empty filter_dict"),
    ):
        await mod.update_data_source(settings, {}, {"name": "new"})
    session.assert_not_called()


async def test_crud_delete_with_an_empty_filter_never_reaches_the_database(settings):
    """The public entry point is safe too, but reports a wrapped RuntimeError.

    ``_delete_primary`` catches every exception and re-raises
    ``RuntimeError(...) from None``, so the guard's ``ValueError`` is masked at
    this layer and survives only in the ``logger.exception`` call. What matters
    here is that the delete does not run: assert the session was never opened.
    """
    with (
        patch.object(crud, "get_primary_database", return_value="sqlite"),
        patch.object(_common, "async_session_or_connection", new=AsyncMock()) as session,
        pytest.raises(RuntimeError, match="Failed to delete from primary database"),
    ):
        await crud.delete(settings, {})
    session.assert_not_called()


async def test_normalize_filter_turns_none_into_the_unscoped_empty_dict():
    """``None`` is silently normalized to ``{}`` — the shape that used to wipe the table.

    Pinned because ``crud.delete``'s annotation says ``dict[str, Any]`` while
    ``_normalize_filter`` accepts ``None``, so the annotation understates what
    actually reaches the query layer. The guard below it is what makes that
    mismatch safe rather than destructive.
    """
    assert crud._normalize_filter(None) == {}
    assert crud._normalize_filter({"source_id": "a"}) == {"id": "a"}
