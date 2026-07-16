"""Additional coverage tests for backend/database/adapter.py.

Focused on the URL-based DB type detection, connection-config builder, the
per-backend async context manager factories (mocking out optional heavy
drivers that aren't installed in this environment), the factory dispatch
table, and the AsyncSnowflakeConnection thread-pool wrapper -- all of which
were split out during an extract-method refactor and lost direct test
coverage.
"""

import sys
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.database import adapter


def _fake_settings(**overrides) -> SimpleNamespace:
    """Build a minimal settings-like object with the attributes adapter.py needs."""
    defaults = {
        "DATABASE_URL": "sqlite+aiosqlite:///./test.db",
        "DB_ECHO": False,
        "DB_POOL_SIZE": 5,
        "DB_MAX_OVERFLOW": 10,
        "DB_SYNC_EXECUTOR_WORKERS": 1,
        "SNOWFLAKE_ACCOUNT": "acct",
        "SNOWFLAKE_USER": "user",
        "SNOWFLAKE_PASSWORD": "pw",
        "SNOWFLAKE_DATABASE": "db",
        "SNOWFLAKE_SCHEMA": "schema",
        "SNOWFLAKE_WAREHOUSE": "wh",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestGetDbTypeFromUrl:
    """Tests for get_db_type_from_url covering every recognized prefix + fallbacks."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("postgresql://u:p@host/db", adapter.DatabaseType.POSTGRES),
            ("postgresql+asyncpg://u:p@host/db", adapter.DatabaseType.POSTGRES),
            ("sqlite:///file.db", adapter.DatabaseType.SQLITE),
            ("sqlite+aiosqlite:///file.db", adapter.DatabaseType.SQLITE),
            ("mysql://u:p@host/db", adapter.DatabaseType.MYSQL),
            ("mysql+aiomysql://u:p@host/db", adapter.DatabaseType.MYSQL),
            ("mongodb://u:p@host/db", adapter.DatabaseType.MONGODB),
            ("mongodb+srv://u:p@host/db", adapter.DatabaseType.MONGODB),
        ],
    )
    def test_recognized_prefixes(self, url, expected):
        assert adapter.get_db_type_from_url(url) == expected

    def test_snowflake_keyword_fallback(self):
        assert (
            adapter.get_db_type_from_url("snowflake://account/db") == adapter.DatabaseType.SNOWFLAKE
        )

    def test_unrecognized_url_defaults_to_sqlite(self):
        assert adapter.get_db_type_from_url("some-random-string") == adapter.DatabaseType.SQLITE

    def test_get_db_type_uses_settings_database_url(self):
        settings = _fake_settings(DATABASE_URL="mysql://u:p@host/db")
        assert adapter.get_db_type(settings) == adapter.DatabaseType.MYSQL


class TestBuildConnectionConfig:
    def test_builds_expected_keys(self):
        settings = _fake_settings()
        cfg = adapter.build_connection_config(settings)
        assert cfg == {
            "database_url": settings.DATABASE_URL,
            "db_echo": settings.DB_ECHO,
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "snowflake_account": settings.SNOWFLAKE_ACCOUNT,
            "snowflake_user": settings.SNOWFLAKE_USER,
            "snowflake_password": settings.SNOWFLAKE_PASSWORD,
            "snowflake_database": settings.SNOWFLAKE_DATABASE,
            "snowflake_schema": settings.SNOWFLAKE_SCHEMA,
            "snowflake_warehouse": settings.SNOWFLAKE_WAREHOUSE,
        }


class TestPostgresSqliteSession:
    async def test_yields_session_and_closes(self, monkeypatch):
        session = AsyncMock()

        async def fake_get_async_session():
            yield session

        monkeypatch.setattr("backend.database.engine.get_async_session", fake_get_async_session)
        settings = _fake_settings()
        cfg = adapter.build_connection_config(settings)
        async with adapter._postgres_sqlite_session(settings, cfg) as resource:
            assert resource is session
        session.close.assert_awaited_once()
        session.rollback.assert_not_called()

    async def test_rolls_back_and_reraises_on_exception(self, monkeypatch):
        session = AsyncMock()

        async def fake_get_async_session():
            yield session

        monkeypatch.setattr("backend.database.engine.get_async_session", fake_get_async_session)
        settings = _fake_settings()
        cfg = adapter.build_connection_config(settings)
        with pytest.raises(ValueError, match="boom"):
            async with adapter._postgres_sqlite_session(settings, cfg):
                raise ValueError("boom")
        session.rollback.assert_awaited_once()
        session.close.assert_awaited_once()


class TestMysqlConnection:
    async def test_raises_runtime_error_when_aiomysql_missing(self):
        settings = _fake_settings()
        cfg = adapter.build_connection_config(settings)
        with pytest.raises(RuntimeError, match="aiomysql"):
            async with adapter._mysql_connection(settings, cfg):
                pass

    async def test_yields_connection_when_aiomysql_available(self, monkeypatch):
        fake_conn = MagicMock()
        fake_aiomysql = types.ModuleType("aiomysql")
        fake_aiomysql.connect = AsyncMock(return_value=fake_conn)
        monkeypatch.setitem(sys.modules, "aiomysql", fake_aiomysql)

        settings = _fake_settings(DATABASE_URL="mysql://myuser:mypass@myhost:3307/mydb")
        cfg = adapter.build_connection_config(settings)
        async with adapter._mysql_connection(settings, cfg) as conn:
            assert conn is fake_conn
        fake_aiomysql.connect.assert_awaited_once()
        _, kwargs = fake_aiomysql.connect.call_args
        assert kwargs["host"] == "myhost"
        assert kwargs["port"] == 3307
        assert kwargs["user"] == "myuser"
        assert kwargs["db"] == "mydb"
        fake_conn.close.assert_called_once()


class TestMongoDatabase:
    async def test_raises_runtime_error_when_motor_missing(self):
        settings = _fake_settings()
        cfg = adapter.build_connection_config(settings)
        with pytest.raises(RuntimeError, match="motor"):
            async with adapter._mongo_database(settings, cfg):
                pass

    async def test_yields_database_when_motor_available(self, monkeypatch):
        fake_db = MagicMock()
        fake_client = MagicMock()
        fake_client.__getitem__ = MagicMock(return_value=fake_db)
        fake_client.close = MagicMock()

        fake_motor_asyncio = types.ModuleType("motor.motor_asyncio")
        fake_motor_asyncio.AsyncIOMotorClient = MagicMock(return_value=fake_client)
        fake_motor = types.ModuleType("motor")
        fake_motor.motor_asyncio = fake_motor_asyncio
        monkeypatch.setitem(sys.modules, "motor", fake_motor)
        monkeypatch.setitem(sys.modules, "motor.motor_asyncio", fake_motor_asyncio)

        settings = _fake_settings(DATABASE_URL="mongodb://host/mydb")
        cfg = adapter.build_connection_config(settings)
        async with adapter._mongo_database(settings, cfg) as db:
            assert db is fake_db
        fake_client.close.assert_called_once()


class TestSnowflakeConnection:
    async def test_raises_runtime_error_when_snowflake_missing(self):
        settings = _fake_settings()
        cfg = adapter.build_connection_config(settings)
        with pytest.raises(RuntimeError, match="snowflake-connector-python"):
            async with adapter._snowflake_connection(settings, cfg):
                pass

    async def test_yields_wrapped_connection_when_snowflake_available(self, monkeypatch):
        fake_conn = MagicMock()
        fake_snowflake_connector = types.ModuleType("snowflake.connector")
        fake_snowflake_connector.connect = MagicMock(return_value=fake_conn)
        fake_snowflake = types.ModuleType("snowflake")
        fake_snowflake.connector = fake_snowflake_connector
        monkeypatch.setitem(sys.modules, "snowflake", fake_snowflake)
        monkeypatch.setitem(sys.modules, "snowflake.connector", fake_snowflake_connector)

        settings = _fake_settings()
        cfg = adapter.build_connection_config(settings)
        async with adapter._snowflake_connection(settings, cfg) as conn:
            assert isinstance(conn, adapter.AsyncSnowflakeConnection)
            assert conn.connection is fake_conn
        fake_conn.close.assert_called_once()
        fake_snowflake_connector.connect.assert_called_once_with(
            account="acct",
            user="user",
            password="pw",
            database="db",
            schema="schema",
            warehouse="wh",
        )


class TestAsyncSessionOrConnection:
    async def test_unsupported_db_type_raises(self, monkeypatch):
        monkeypatch.setattr(adapter, "get_db_type", lambda settings: "not-a-real-db-type")
        settings = _fake_settings()
        with pytest.raises(RuntimeError, match="Unsupported database type"):
            async with adapter.async_session_or_connection(settings):
                pass

    async def test_dispatches_to_matching_factory(self, monkeypatch):
        from contextlib import asynccontextmanager

        sentinel = object()

        @asynccontextmanager
        async def fake_factory(settings, cfg):
            yield sentinel

        monkeypatch.setitem(
            adapter._CONNECTION_FACTORIES, adapter.DatabaseType.SQLITE, fake_factory
        )
        settings = _fake_settings(DATABASE_URL="sqlite+aiosqlite:///./x.db")
        async with adapter.async_session_or_connection(settings) as resource:
            assert resource is sentinel

    async def test_uses_provided_config_override(self, monkeypatch):
        from contextlib import asynccontextmanager

        received_cfg = {}

        @asynccontextmanager
        async def fake_factory(settings, cfg):
            received_cfg.update(cfg)
            yield None

        monkeypatch.setitem(
            adapter._CONNECTION_FACTORIES, adapter.DatabaseType.SQLITE, fake_factory
        )
        settings = _fake_settings(DATABASE_URL="sqlite+aiosqlite:///./x.db")
        custom_cfg = {"database_url": "sqlite+aiosqlite:///./override.db"}
        async with adapter.async_session_or_connection(settings, config=custom_cfg):
            pass
        assert received_cfg == custom_cfg


class TestAsyncSnowflakeConnectionWrapper:
    """Tests for the AsyncSnowflakeConnection thread-pool wrapper class."""

    @pytest.fixture
    def wrapper(self):
        fake_connection = MagicMock()
        executor = ThreadPoolExecutor(max_workers=1)
        wrapper = adapter.AsyncSnowflakeConnection(fake_connection, executor)
        yield wrapper, fake_connection
        executor.shutdown(wait=True)

    async def test_execute_without_params(self, wrapper):
        conn_wrapper, fake_connection = wrapper
        fake_cursor = MagicMock()
        fake_cursor.fetchall.return_value = [("row1",)]
        fake_connection.cursor.return_value = fake_cursor

        result = await conn_wrapper.execute("SELECT 1")

        assert result == [("row1",)]
        fake_cursor.execute.assert_called_once_with("SELECT 1")
        fake_cursor.close.assert_called_once()

    async def test_execute_with_params(self, wrapper):
        conn_wrapper, fake_connection = wrapper
        fake_cursor = MagicMock()
        fake_cursor.fetchall.return_value = []
        fake_connection.cursor.return_value = fake_cursor

        await conn_wrapper.execute("SELECT * FROM t WHERE id = %s", (1,))

        fake_cursor.execute.assert_called_once_with("SELECT * FROM t WHERE id = %s", (1,))

    async def test_commit(self, wrapper):
        conn_wrapper, fake_connection = wrapper
        await conn_wrapper.commit()
        fake_connection.commit.assert_called_once()

    async def test_rollback(self, wrapper):
        conn_wrapper, fake_connection = wrapper
        await conn_wrapper.rollback()
        fake_connection.rollback.assert_called_once()

    def test_cursor_and_connection_property(self, wrapper):
        conn_wrapper, fake_connection = wrapper
        assert conn_wrapper.cursor() is fake_connection.cursor.return_value
        assert conn_wrapper.connection is fake_connection
