"""Engine-agnostic ``data_sources`` writes, shared by the SQLite and PostgreSQL modules.

``async_sqlite_queries`` and ``async_postgres_queries`` are peer implementations of one
CRUD surface, but ``update_data_source`` and ``delete_data_source`` are built from
SQLAlchemy Core, which takes its dialect from the session — so the two copies were
byte-identical apart from a single comment. They are implemented here once and
re-exported by both modules, which leaves ``_DB_PEERS`` dispatch and the
``sqlite_*``/``postgres_*`` names in ``__init__`` working unchanged.

The empty-filter guard lives beside them because both functions derive their column
list from ``filter_dict``, and it is the guard that keeps an empty filter from
compiling into a full-table write.
"""

import logging
from typing import Any

from sqlalchemy import column, delete, table, update

from backend.config import Settings

from ..adapter import async_session_or_connection

logger = logging.getLogger(__name__)

TABLE = "data_sources"

__all__ = ["delete_data_source", "require_non_empty_filter", "update_data_source"]


def require_non_empty_filter(filter_dict: dict[str, Any], operation: str) -> None:
    """Reject an empty ``filter_dict`` rather than read it as "match every row".

    Both writers below derive their column list from ``filter_dict``, so with no keys
    nothing is left to constrain on and the statement compiles to a bare
    ``DELETE FROM data_sources`` or an unscoped ``UPDATE data_sources SET ...``. Either
    one touches every row and still reports ``affected_rows`` as though that had been
    the requested operation. Reaching this shape does not take a determined caller:
    ``_normalize_filter(None)`` returns ``{}``, so the public ``update``/``delete``
    entry points accept the table-wiping input.

    Call this before the session opens, so that no transaction is started for a call
    that cannot run.

    Args:
        filter_dict: The WHERE-clause keys the caller supplied.
        operation: Name of the calling function, used verbatim in the error message so
            a wrapped failure still names the operation.

    Raises:
        ValueError: If ``filter_dict`` is empty.
    """
    if not filter_dict:
        raise ValueError(f"{operation} requires a non-empty filter_dict")


async def update_data_source(
    settings: Settings, filter_dict: dict[str, Any], update_data: dict[str, Any]
):
    """Update the data source records matching every key in ``filter_dict``.

    Raises:
        ValueError: If ``filter_dict`` is empty; see ``require_non_empty_filter``.
    """
    require_non_empty_filter(filter_dict, "update_data_source")

    async with async_session_or_connection(settings) as session:
        try:
            tbl = table(
                TABLE,
                *[column(c) for c in update_data] + [column(c) for c in filter_dict],
            )

            stmt = update(tbl).values(**update_data)

            for k, v in filter_dict.items():
                stmt = stmt.where(column(k) == v)

            result = await session.execute(stmt)
            await session.commit()

            return {"affected_rows": result.rowcount}

        except Exception as e:
            logger.exception(f"Failed to update data source: {e}")
            await session.rollback()
            raise


async def delete_data_source(settings: Settings, filter_dict: dict[str, Any]):
    """Delete the data source records matching every key in ``filter_dict``.

    Raises:
        ValueError: If ``filter_dict`` is empty; see ``require_non_empty_filter``.
    """
    require_non_empty_filter(filter_dict, "delete_data_source")

    async with async_session_or_connection(settings) as session:
        try:
            tbl = table(TABLE, *[column(c) for c in filter_dict])
            stmt = delete(tbl)

            for k, v in filter_dict.items():
                stmt = stmt.where(column(k) == v)

            result = await session.execute(stmt)
            await session.commit()

            return {"affected_rows": result.rowcount}

        except Exception as e:
            logger.exception(f"Failed to delete data source: {e}")
            await session.rollback()
            raise
