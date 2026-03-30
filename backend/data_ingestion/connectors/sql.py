import hashlib
import logging
import re
from typing import Dict, Optional

import polars as pl
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.pool import QueuePool

from .base import BaseConnector

logger = logging.getLogger(__name__)

# ---------- Validation ----------

_SAFE_TABLE_RE = re.compile(r"^[a-zA-Z_\"][a-zA-Z0-9_.\"]*$")
_MAX_TABLE_LEN = 255

# Multi-statement and write keywords that should never appear in user queries
_DANGEROUS_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


def _validate_table_name(name: str) -> str:
    """Reject anything that isn't a plain SQL identifier (schema.table allowed)."""
    if not name or len(name) > _MAX_TABLE_LEN:
        raise ValueError(f"Invalid table name length: {len(name) if name else 0}")
    if not _SAFE_TABLE_RE.match(name):
        raise ValueError(f"Invalid table name: {name!r}")
    return name


def _validate_query(query: str) -> str:
    """Allow only single SELECT statements."""
    stripped = query.strip()
    if not stripped:
        raise ValueError("Empty query")
    if ";" in stripped:
        raise ValueError("Multi-statement queries are not allowed.")
    if not stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")
    if _DANGEROUS_KEYWORDS.search(stripped):
        raise ValueError("Query contains disallowed keywords.")
    return stripped


# ---------- Connection pooling ----------

_engine_cache: Dict[str, Engine] = {}

_POOL_SIZE = 5
_MAX_OVERFLOW = 10
_POOL_TIMEOUT = 30
_POOL_RECYCLE = 1800  # 30 min
_STATEMENT_TIMEOUT_MS = 30_000  # 30 s


def get_pooled_engine(connection_string: str) -> Engine:
    """Return a pooled SQLAlchemy engine (cached per connection string)."""
    key = hashlib.sha256(connection_string.encode()).hexdigest()
    if key not in _engine_cache:
        connect_args: Dict[str, object] = {}
        if "postgresql" in connection_string:
            connect_args["options"] = f"-c statement_timeout={_STATEMENT_TIMEOUT_MS}"
        elif "mysql" in connection_string:
            connect_args["read_timeout"] = _STATEMENT_TIMEOUT_MS // 1000
        elif "sqlite" in connection_string:
            connect_args["timeout"] = _STATEMENT_TIMEOUT_MS // 1000

        _engine_cache[key] = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=_POOL_SIZE,
            max_overflow=_MAX_OVERFLOW,
            pool_timeout=_POOL_TIMEOUT,
            pool_recycle=_POOL_RECYCLE,
            connect_args=connect_args,
        )
        logger.info("Created pooled engine for connection (hash=%s)", key[:8])
    return _engine_cache[key]


# ---------- Connector ----------


class DatabaseConnector(BaseConnector):
    """
    Connector for SQL databases (Postgres, MySQL, SQLite, Snowflake, etc.).
    Uses SQLAlchemy connection pooling and Polars for data fetching.
    """

    def __init__(
        self,
        connection_string: str,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self.query = query
        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        """Lazily obtain a pooled engine."""
        if self._engine is None:
            self._engine = get_pooled_engine(self.connection_string)
        return self._engine

    def _build_query(self) -> str:
        """Return a safe SQL string from either self.query or self.table_name."""
        if self.query:
            return _validate_query(self.query)
        if self.table_name:
            safe_name = _validate_table_name(self.table_name)
            return f'SELECT * FROM "{safe_name}"'
        raise ValueError("No table_name or query provided")

    async def connect(self) -> bool:
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    async def validate(self) -> bool:
        return await self.connect()

    async def get_schema(self) -> Dict[str, str]:
        engine = self._get_engine()
        q = self._build_query()
        schema_query = f"SELECT * FROM ({q}) AS subq LIMIT 0"

        try:
            df = pl.read_database(query=schema_query, connection=engine)
            return {col: str(dtype) for col, dtype in df.schema.items()}
        except Exception as e:
            raise RuntimeError(f"Failed to get schema: {e}")

    async def fetch_data(
        self, query: Optional[str] = None, limit: Optional[int] = None
    ) -> pl.DataFrame:
        engine = self._get_engine()

        if query:
            q = _validate_query(query)
        else:
            q = self._build_query()

        if limit:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"Invalid limit: {limit}")
            q = f"SELECT * FROM ({q}) AS subq LIMIT {limit}"

        try:
            return pl.read_database(query=q, connection=engine)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data: {e}")
