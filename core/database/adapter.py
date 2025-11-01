"""
Async Database Adapter for FastAPI

Provides unified access to different database backends with async support.
This is the async equivalent of the Flask db/adapter.py module.

Supports:
- PostgreSQL (async via asyncpg)
- SQLite (async via aiosqlite)
- MySQL (async via aiomysql) 
- MongoDB (async via motor)
- Snowflake (async wrapper)
"""

from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Union
import logging

# Import settings with fallback
try:
    from config import Settings
except ImportError:
    # Fallback for standalone testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import Settings

logger = logging.getLogger(__name__)


class DatabaseType:
    """Database type constants."""
    POSTGRES = "postgres"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MYSQL = "mysql"
    MARIADB = "mariadb"
    MONGODB = "mongodb"
    MONGO = "mongo"
    SNOWFLAKE = "snowflake"


def get_db_type_from_url(database_url: str) -> str:
    """
    Extract database type from connection URL.
    
    Args:
        database_url: Database connection URL
        
    Returns:
        str: Database type identifier
    """
    if database_url.startswith("postgresql://") or database_url.startswith("postgresql+asyncpg://"):
        return DatabaseType.POSTGRES
    elif database_url.startswith("sqlite://") or database_url.startswith("sqlite+aiosqlite://"):
        return DatabaseType.SQLITE
    elif database_url.startswith("mysql://") or database_url.startswith("mysql+aiomysql://"):
        return DatabaseType.MYSQL
    elif database_url.startswith("mongodb://") or database_url.startswith("mongodb+srv://"):
        return DatabaseType.MONGODB
    elif "snowflake" in database_url.lower():
        return DatabaseType.SNOWFLAKE
    else:
        # Default fallback
        return DatabaseType.SQLITE


def get_db_type(settings: Settings) -> str:
    """
    Get database type from settings.
    
    Args:
        settings: Application settings
        
    Returns:
        str: Database type identifier
    """
    return get_db_type_from_url(settings.DATABASE_URL)


def build_connection_config(settings: Settings) -> Dict[str, Any]:
    """
    Build connection configuration from settings.
    
    Args:
        settings: Application settings
        
    Returns:
        Dict[str, Any]: Connection configuration
    """
    return {
        "database_url": settings.DATABASE_URL,
        "db_echo": settings.DB_ECHO,
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        # Snowflake specific
        "snowflake_account": settings.SNOWFLAKE_ACCOUNT,
        "snowflake_user": settings.SNOWFLAKE_USER,
        "snowflake_password": settings.SNOWFLAKE_PASSWORD,
        "snowflake_database": settings.SNOWFLAKE_DATABASE,
        "snowflake_schema": settings.SNOWFLAKE_SCHEMA,
        "snowflake_warehouse": settings.SNOWFLAKE_WAREHOUSE,
    }


@asynccontextmanager
async def async_session_or_connection(
    settings: Settings, 
    config: Optional[Dict[str, Any]] = None
):
    """
    Async context manager that yields appropriate database connection/session.
    
    Args:
        settings: Application settings
        config: Optional override configuration
        
    Yields:
        Union: Database-specific session or connection object
        
    Yields:
        - AsyncSession for PostgreSQL/SQLite (SQLAlchemy)
        - AsyncConnection for MySQL
        - AsyncIOMotorDatabase for MongoDB  
        - AsyncConnection for Snowflake
    """
    db_type = get_db_type(settings)
    cfg = config or build_connection_config(settings)
    
    if db_type in (DatabaseType.POSTGRES, DatabaseType.POSTGRESQL, DatabaseType.SQLITE):
        # Use our main async engine and session
        from .engine import get_async_session
        
        async for session in get_async_session():
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    elif db_type in (DatabaseType.MYSQL, DatabaseType.MARIADB):
        # MySQL async connection
        try:
            import aiomysql
        except ImportError:
            raise RuntimeError("aiomysql package required for MySQL async support")
        
        # Parse connection details from URL or config
        import urllib.parse as urlparse
        parsed = urlparse.urlparse(cfg["database_url"])
        
        conn = await aiomysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password,
            db=parsed.path.lstrip("/") if parsed.path else None,
            autocommit=False
        )
        
        try:
            yield conn
        finally:
            conn.close()
    
    elif db_type in (DatabaseType.MONGODB, DatabaseType.MONGO):
        # MongoDB async connection
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise RuntimeError("motor package required for MongoDB async support")
        
        client = AsyncIOMotorClient(cfg["database_url"])
        
        try:
            # Extract database name from URL or config
            import urllib.parse as urlparse
            parsed = urlparse.urlparse(cfg["database_url"])
            db_name = parsed.path.lstrip("/") if parsed.path else "default"
            
            database = client[db_name]
            yield database
        finally:
            client.close()
    
    elif db_type == DatabaseType.SNOWFLAKE:
        # Snowflake async wrapper (Snowflake doesn't have native async support)
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        try:
            import snowflake.connector
        except ImportError:
            raise RuntimeError("snowflake-connector-python package required for Snowflake support")
        
        # Create connection in thread pool since Snowflake is sync-only
        executor = ThreadPoolExecutor(max_workers=1)
        
        def create_snowflake_connection():
            return snowflake.connector.connect(
                account=cfg.get("snowflake_account"),
                user=cfg.get("snowflake_user"),
                password=cfg.get("snowflake_password"),
                database=cfg.get("snowflake_database"),
                schema=cfg.get("snowflake_schema"),
                warehouse=cfg.get("snowflake_warehouse"),
            )
        
        conn = await asyncio.get_event_loop().run_in_executor(
            executor, create_snowflake_connection
        )
        
        try:
            # Wrap connection with async interface
            yield AsyncSnowflakeConnection(conn, executor)
        finally:
            await asyncio.get_event_loop().run_in_executor(
                executor, conn.close
            )
            executor.shutdown(wait=True)
    
    else:
        raise RuntimeError(f"Unsupported database type: {db_type}")


class AsyncSnowflakeConnection:
    """
    Async wrapper for Snowflake connection.
    Since Snowflake doesn't support native async, we wrap it with ThreadPoolExecutor.
    """
    
    def __init__(self, connection, executor):
        self._connection = connection
        self._executor = executor
    
    async def execute(self, query: str, params: Optional[tuple] = None):
        """Execute a query asynchronously."""
        import asyncio
        
        def _execute():
            cursor = self._connection.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                return cursor.fetchall()
            finally:
                cursor.close()
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _execute
        )
    
    async def commit(self):
        """Commit transaction asynchronously."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, self._connection.commit
        )
    
    async def rollback(self):
        """Rollback transaction asynchronously."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, self._connection.rollback
        )
    
    def cursor(self):
        """Get cursor (still sync, use execute method for async operations)."""
        return self._connection.cursor()
    
    @property
    def connection(self):
        """Get underlying connection for direct access if needed."""
        return self._connection