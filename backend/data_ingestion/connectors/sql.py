from typing import Dict, Optional

import polars as pl
from sqlalchemy import Engine, create_engine, text

from .base import BaseConnector


class DatabaseConnector(BaseConnector):
    """
    Connector for SQL databases (Postgres, MySQL, SQLite, Snowflake, etc.).
    Uses SQLAlchemy for connection and Polars for data fetching.
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

    async def connect(self) -> bool:
        try:
            # Create SQLAlchemy engine
            self._engine = create_engine(self.connection_string)
            # Test connection
            if self._engine:
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")

    async def validate(self) -> bool:
        return await self.connect()

    async def get_schema(self) -> Dict[str, str]:
        if not self._engine:
            await self.connect()

        if not self._engine:
            raise ConnectionError("Database engine not initialized")

        # Construct a query to fetch 0 rows just to get schema
        q = self.query
        if not q and self.table_name:
            q = f"SELECT * FROM {self.table_name}"  # nosec

        if not q:
            raise ValueError("No table_name or query provided")

        # Wrap in a limit 0 to avoid fetching data
        # Note: This syntax might vary by dialect, but works for Postgres/MySQL/SQLite
        schema_query = f"SELECT * FROM ({q}) as subq LIMIT 0"  # nosec

        try:
            # Polars read_database uses the engine to fetch schema
            df = pl.read_database(query=schema_query, connection=self._engine)
            return {col: str(dtype) for col, dtype in df.schema.items()}
        except Exception as e:
            raise RuntimeError(f"Failed to get schema: {str(e)}")

    async def fetch_data(
        self, query: Optional[str] = None, limit: Optional[int] = None
    ) -> pl.DataFrame:
        if not self._engine:
            await self.connect()

        if not self._engine:
            raise ConnectionError("Database engine not initialized")

        q = query or self.query
        if not q and self.table_name:
            q = f"SELECT * FROM {self.table_name}"  # nosec

        if not q:
            raise ValueError("No query or table specified")

        if limit:
            # Naive limit injection. For production, use a proper query builder or dialect-specific limit.
            # This assumes the user provided a SELECT statement.
            q = f"SELECT * FROM ({q}) as subq LIMIT {limit}"  # nosec

        try:
            return pl.read_database(query=q, connection=self._engine)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data: {str(e)}")
