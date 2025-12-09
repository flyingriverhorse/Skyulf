"""
Async Database Connection Manager
Handles async connection pooling and concurrent access optimization for SQLite and PostgreSQL
"""

import os
import asyncio
import logging
import atexit
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import aiosqlite
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from core.config import Settings

logger = logging.getLogger(__name__)


class AsyncSQLiteConnectionManager:
    """
    Async SQLite connection manager with optimized settings
    """

    def __init__(self, database_path: str, pool_size: int = 10, timeout: int = 30):
        self.database_path = database_path
        self.pool_size = pool_size
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(pool_size)
        self._initialized = False

        # Register cleanup on exit
        atexit.register(self._sync_close_all)

    async def initialize(self):
        """Initialize the async SQLite manager with optimized settings"""
        try:
            # Ensure database directory exists
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)

            # Enable WAL mode and optimizations
            await self._setup_database_optimizations()

            self._initialized = True
            logger.info(f"[OK] Async SQLite manager initialized for {self.database_path}")

        except Exception as e:
            logger.error(f"Failed to initialize async SQLite manager: {e}")
            raise

    async def _setup_database_optimizations(self):
        """Setup WAL mode and performance optimizations"""
        try:
            async with aiosqlite.connect(self.database_path, timeout=self.timeout) as conn:
                # Enable WAL mode (Write-Ahead Logging)
                await conn.execute("PRAGMA journal_mode=WAL")

                # Optimize for concurrent access
                await conn.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL
                await conn.execute("PRAGMA cache_size=10000")     # 10MB cache
                await conn.execute("PRAGMA temp_store=MEMORY")    # Use memory for temp
                await conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O

                # WAL checkpoint settings
                await conn.execute("PRAGMA wal_autocheckpoint=1000")  # Every 1000 pages

                # Connection-specific settings
                await conn.execute("PRAGMA busy_timeout=30000")      # 30 second timeout
                await conn.execute("PRAGMA foreign_keys=ON")         # Enable FK constraints

                await conn.commit()

            logger.info("[OK] Async SQLite WAL mode and optimizations enabled")

        except Exception as e:
            logger.error(f"Failed to setup SQLite optimizations: {e}")
            raise

    @asynccontextmanager
    async def get_connection(self):
        """Get an async SQLite connection with semaphore limiting"""
        if not self._initialized:
            raise RuntimeError("Async SQLite manager not initialized")

        async with self._semaphore:  # Limit concurrent connections
            conn = None
            try:
                conn = await aiosqlite.connect(
                    self.database_path,
                    timeout=self.timeout
                )

                # Set connection-specific pragmas
                await conn.execute("PRAGMA busy_timeout=30000")
                await conn.execute("PRAGMA foreign_keys=ON")

                yield conn

            except Exception:
                if conn:
                    try:
                        await conn.rollback()
                    except Exception:
                        pass
                raise
            finally:
                if conn:
                    try:
                        await conn.close()
                    except Exception:
                        pass

    async def execute_query(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
    ) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dicts"""
        async with self.get_connection() as conn:
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = await cursor.fetchall()
            await cursor.close()

            return [dict(zip(columns, row)) for row in rows]

    async def execute_update(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
    ) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows"""
        async with self.get_connection() as conn:
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)

            await conn.commit()
            rowcount = cursor.rowcount
            await cursor.close()
            return rowcount

    def _sync_close_all(self):
        """Synchronous close for atexit handler"""
        logger.info("[OK] Async SQLite manager cleanup registered")


class AsyncPostgreSQLConnectionManager:
    """
    Async PostgreSQL connection manager with SQLAlchemy async pooling
    """

    def __init__(self, connection_string: str, pool_size: int = 10):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self._engine = None
        self._session_maker = None
        self._initialized = False

    async def initialize(self):
        """Initialize async SQLAlchemy engine with connection pooling"""
        try:
            # Create async engine with optimized pool settings
            self._engine = create_async_engine(
                self.connection_string,
                pool_size=self.pool_size,
                max_overflow=20,
                pool_pre_ping=True,        # Validate connections before use
                pool_recycle=3600,         # Recycle connections every hour
                echo=False,                # Set to True for SQL debugging
                connect_args={
                    "server_settings": {
                        "application_name": "MLOps_FastAPI",
                        "statement_timeout": "300000"  # 5 minute timeout
                    }
                }
            )

            # Create session maker
            self._session_maker = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False
            )

            # Test connection
            async with self._engine.begin() as conn:
                await conn.execute("SELECT 1")

            self._initialized = True
            logger.info(f"[OK] Async PostgreSQL connection pool initialized with {self.pool_size} connections")

        except Exception as e:
            logger.error(f"Failed to initialize async PostgreSQL connection pool: {e}")
            raise

    @asynccontextmanager
    async def get_connection(self):
        """Get an async connection from the SQLAlchemy pool"""
        if not self._engine:
            raise RuntimeError("Async PostgreSQL engine not initialized")

        async with self._engine.begin() as conn:
            try:
                yield conn
            except Exception:
                await conn.rollback()
                raise

    @asynccontextmanager
    async def get_session(self):
        """Get an async session from the session maker"""
        if not self._session_maker:
            raise RuntimeError("Async PostgreSQL session maker not initialized")

        async with self._session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    async def execute_query(
        self,
        query: str,
        params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dicts"""
        async with self.get_connection() as conn:
            from sqlalchemy import text
            result = await conn.execute(text(query), params or {})
            columns = result.keys()
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    async def execute_update(
        self,
        query: str,
        params: Dict[str, Any] | None = None,
    ) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows"""
        async with self.get_connection() as conn:
            from sqlalchemy import text
            result = await conn.execute(text(query), params or {})
            await conn.commit()
            return result.rowcount

    async def close(self):
        """Close the async engine and all connections"""
        if self._engine:
            await self._engine.dispose()
            logger.info("[OK] Async PostgreSQL connection pool closed")


class AsyncDatabaseManager:
    """
    Unified async database manager that handles both SQLite and PostgreSQL
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.sqlite_manager = None
        self.postgres_manager = None
        self._initialized = False

    async def initialize(self):
        """Initialize database managers based on settings"""
        try:
            # Initialize SQLite if configured
            sqlite_path = getattr(self.settings, 'DB_PATH', None)
            if sqlite_path:
                if not os.path.isabs(sqlite_path):
                    sqlite_path = os.path.join(os.getcwd(), sqlite_path)

                pool_size = getattr(self.settings, 'sqlite_pool_size', 10)
                timeout = getattr(self.settings, 'sqlite_timeout', 30)

                self.sqlite_manager = AsyncSQLiteConnectionManager(sqlite_path, pool_size, timeout)
                await self.sqlite_manager.initialize()

            # Initialize PostgreSQL if configured
            postgres_url = getattr(self.settings, 'postgres_url', None)
            if postgres_url:
                pool_size = getattr(self.settings, 'postgres_pool_size', 10)
                self.postgres_manager = AsyncPostgreSQLConnectionManager(postgres_url, pool_size)
                await self.postgres_manager.initialize()

            self._initialized = True
            logger.info("[OK] Async database managers initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize async database managers: {e}")
            raise

    def get_primary_db_manager(self):
        """Get the primary database manager"""
        if not self._initialized:
            raise RuntimeError("Async database manager not initialized")

        primary = getattr(self.settings, 'DB_PRIMARY', 'sqlite').lower()

        if primary == 'sqlite':
            if not self.sqlite_manager:
                raise RuntimeError("SQLite manager not available")
            return self.sqlite_manager
        elif primary == 'postgres':
            if not self.postgres_manager:
                raise RuntimeError("PostgreSQL manager not available")
            return self.postgres_manager
        else:
            raise ValueError(f"Unknown database primary: {primary}")

    def get_sqlite_manager(self):
        """Get async SQLite connection manager"""
        if not self.sqlite_manager:
            raise RuntimeError("Async SQLite manager not available")
        return self.sqlite_manager

    def get_postgres_manager(self):
        """Get async PostgreSQL connection manager"""
        if not self.postgres_manager:
            raise RuntimeError("Async PostgreSQL manager not available")
        return self.postgres_manager

    async def close_all(self):
        """Close all async database connections"""
        tasks = []

        if self.postgres_manager:
            tasks.append(self.postgres_manager.close())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("[OK] All async database connections closed")


# Global async database manager instance
_async_db_manager: Optional[AsyncDatabaseManager] = None


async def initialize_async_database_manager(settings: Settings):
    """Initialize the global async database manager"""
    global _async_db_manager
    _async_db_manager = AsyncDatabaseManager(settings)
    await _async_db_manager.initialize()
    return _async_db_manager


def get_async_database_manager() -> Optional[AsyncDatabaseManager]:
    """Get the global async database manager"""
    return _async_db_manager


async def close_async_database_manager():
    """Close the global async database manager"""
    global _async_db_manager
    if _async_db_manager:
        await _async_db_manager.close_all()
        _async_db_manager = None
