"""Tests for SQL connector: validation, pooling, and timeouts."""
import pytest

from backend.data_ingestion.connectors.sql import (
    DatabaseConnector,
    _engine_cache,
    _validate_query,
    _validate_table_name,
    get_pooled_engine,
)


# ---------- Table name validation ----------


class TestTableNameValidation:
    def test_simple_name(self) -> None:
        assert _validate_table_name("users") == "users"

    def test_schema_qualified(self) -> None:
        assert _validate_table_name("public.users") == "public.users"

    def test_quoted_name(self) -> None:
        assert _validate_table_name('"my_table"') == '"my_table"'

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name length"):
            _validate_table_name("")

    def test_rejects_none_like(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name length"):
            _validate_table_name("")

    def test_rejects_sql_injection_semicolon(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name("users; DROP TABLE users")

    def test_rejects_sql_injection_comment(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name("users--")

    def test_rejects_spaces(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name("users table")

    def test_rejects_parentheses(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name("users()")

    def test_rejects_starting_with_number(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name("123users")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name length"):
            _validate_table_name("a" * 256)


# ---------- Query validation ----------


class TestQueryValidation:
    def test_simple_select(self) -> None:
        assert _validate_query("SELECT * FROM users") == "SELECT * FROM users"

    def test_select_with_where(self) -> None:
        q = "SELECT id, name FROM users WHERE status = 'active'"
        assert _validate_query(q) == q

    def test_select_strips_whitespace(self) -> None:
        assert _validate_query("  SELECT 1  ") == "SELECT 1"

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="Empty query"):
            _validate_query("")

    def test_rejects_insert(self) -> None:
        with pytest.raises(ValueError, match="Multi-statement"):
            _validate_query("SELECT 1; INSERT INTO users VALUES (1)")

    def test_rejects_delete(self) -> None:
        with pytest.raises(ValueError, match="Only SELECT"):
            _validate_query("DELETE FROM users")

    def test_rejects_update(self) -> None:
        with pytest.raises(ValueError, match="Only SELECT"):
            _validate_query("UPDATE users SET name = 'x'")

    def test_rejects_drop(self) -> None:
        with pytest.raises(ValueError, match="Only SELECT"):
            _validate_query("DROP TABLE users")

    def test_rejects_multi_statement(self) -> None:
        with pytest.raises(ValueError, match="Multi-statement"):
            _validate_query("SELECT 1; SELECT 2")

    def test_rejects_select_with_insert_subquery(self) -> None:
        with pytest.raises(ValueError, match="Multi-statement"):
            _validate_query("SELECT 1; INSERT INTO x VALUES (1)")


# ---------- Connection pooling ----------


class TestConnectionPooling:
    def test_engine_cached(self) -> None:
        conn_str = "sqlite:///:memory:"
        _engine_cache.clear()
        e1 = get_pooled_engine(conn_str)
        e2 = get_pooled_engine(conn_str)
        assert e1 is e2
        _engine_cache.clear()

    def test_different_conn_strings_get_different_engines(self) -> None:
        _engine_cache.clear()
        e1 = get_pooled_engine("sqlite:///db1.sqlite")
        e2 = get_pooled_engine("sqlite:///db2.sqlite")
        assert e1 is not e2
        _engine_cache.clear()


# ---------- DatabaseConnector integration ----------


class TestDatabaseConnector:
    def test_build_query_from_table(self) -> None:
        c = DatabaseConnector("sqlite:///:memory:", table_name="users")
        assert c._build_query() == 'SELECT * FROM "users"'

    def test_build_query_from_custom_query(self) -> None:
        c = DatabaseConnector("sqlite:///:memory:", query="SELECT id FROM users")
        assert c._build_query() == "SELECT id FROM users"

    def test_build_query_no_table_no_query_raises(self) -> None:
        c = DatabaseConnector("sqlite:///:memory:")
        with pytest.raises(ValueError, match="No table_name or query"):
            c._build_query()

    def test_build_query_rejects_bad_table(self) -> None:
        c = DatabaseConnector("sqlite:///:memory:", table_name="users; DROP TABLE x")
        with pytest.raises(ValueError, match="Invalid table name"):
            c._build_query()

    def test_build_query_rejects_bad_custom_query(self) -> None:
        c = DatabaseConnector("sqlite:///:memory:", query="DROP TABLE users")
        with pytest.raises(ValueError, match="Only SELECT"):
            c._build_query()

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        c = DatabaseConnector("sqlite:///:memory:", table_name="t")
        result = await c.connect()
        assert result is True
        _engine_cache.clear()

    @pytest.mark.asyncio
    async def test_connect_bad_string_raises(self) -> None:
        c = DatabaseConnector("invalid://host/db", table_name="t")
        with pytest.raises(ConnectionError):
            await c.connect()

    @pytest.mark.asyncio
    async def test_fetch_data_rejects_bad_limit(self) -> None:
        c = DatabaseConnector("sqlite:///:memory:", table_name="t")
        with pytest.raises(ValueError, match="Invalid limit"):
            await c.fetch_data(limit=-1)
        _engine_cache.clear()
