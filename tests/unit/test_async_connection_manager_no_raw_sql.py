"""OC-152 — the shared connection managers expose no raw-SQL executor.

``execute_query``/``execute_update`` bound their *values* correctly but accepted
an unconstrained query string, making them a ready-made injection sink for the
first caller who interpolated a column or table name. They sat in a shared
manager class where they looked blessed, and had zero callers anywhere in the
repository — every live query path uses SQLAlchemy constructs or a parameterised
``sa_text`` with named binds.

The audit filed only the PostgreSQL pair; the SQLite manager carried the same
dead pair, so both were removed.
"""

import pytest

from backend.database.async_connection_manager import (
    AsyncPostgreSQLConnectionManager,
    AsyncSQLiteConnectionManager,
)

MANAGERS = [AsyncSQLiteConnectionManager, AsyncPostgreSQLConnectionManager]


@pytest.mark.parametrize("manager_cls", MANAGERS)
@pytest.mark.parametrize("method", ["execute_query", "execute_update"])
def test_no_raw_sql_executor(manager_cls, method):
    assert not hasattr(manager_cls, method), (
        f"{manager_cls.__name__}.{method} re-introduces an unconstrained raw-SQL "
        "sink; use SQLAlchemy constructs or a parameterised sa_text instead"
    )
