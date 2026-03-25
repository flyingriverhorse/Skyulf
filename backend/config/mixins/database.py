"""Database and pipeline storage settings."""

from typing import Any, Dict, Optional


class DatabaseMixin:
    """Database URLs, connection tuning, and pipeline storage."""

    DATABASE_URL: str = "sqlite+aiosqlite:///./mlops_database.db"
    DB_ECHO: bool = False
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Pipeline storage
    PIPELINE_STORAGE_TYPE: str = "database"
    PIPELINE_STORAGE_PATH: str = "exports/pipelines"

    # Component fields (auto-construct DATABASE_URL in validator)
    DB_TYPE: str = "sqlite"
    DB_PRIMARY: str = "sqlite"
    DB_PATH: str = "mlops_database.db"
    DB_PROVIDER: Optional[str] = None
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = None
    DB_NAME: Optional[str] = None
    DB_SSLMODE: Optional[str] = None
    DB_SSLROOTCERT: Optional[str] = None
    DB_EXTRA_PARAMS: Optional[str] = None

    # ── Helper methods ────────────────────────────────────────────────

    def get_sqlite_url(self) -> str:
        """Get SQLite database URL."""
        db_path = self.DB_PATH or "mlops_database.db"  # type: ignore[attr-defined]
        return f"sqlite+aiosqlite:///./{db_path}"

    def get_postgresql_url(self) -> str:
        """Get PostgreSQL database URL from component fields."""
        if not all([self.DB_USER, self.DB_PASSWORD, self.DB_HOST, self.DB_NAME]):  # type: ignore[attr-defined]
            raise ValueError(
                "PostgreSQL configuration incomplete: "
                "missing DB_USER, DB_PASSWORD, DB_HOST, or DB_NAME"
            )
        port = self.DB_PORT or 5432  # type: ignore[attr-defined]
        url = (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@"  # type: ignore[attr-defined]
            f"{self.DB_HOST}:{port}/{self.DB_NAME}"  # type: ignore[attr-defined]
        )
        if self.DB_SSLMODE:  # type: ignore[attr-defined]
            url += f"?sslmode={self.DB_SSLMODE}"  # type: ignore[attr-defined]
        return url

    def get_database_url_for_type(self, db_type: str) -> str:
        """Get database URL for the specified database type."""
        if db_type.lower() == "sqlite":
            return self.get_sqlite_url()
        elif db_type.lower() in ("postgresql", "postgres"):
            return self.get_postgresql_url()
        raise ValueError(f"Unsupported database type: {db_type}")

    def get_database_config_info(self) -> Dict[str, Any]:
        """Get database configuration information for admin dashboard."""
        return {
            "primary_db": self.DB_PRIMARY,  # type: ignore[attr-defined]
            "current_db_type": (
                "sqlite" if self.DATABASE_URL.startswith("sqlite") else "postgresql"  # type: ignore[attr-defined]
            ),
            "sqlite_configured": True,
            "postgresql_configured": all([self.DB_USER, self.DB_HOST, self.DB_NAME]),  # type: ignore[attr-defined]
            "sqlite_label": "SQLite",
            "postgres_label": "PostgreSQL",
            "sqlite_path": self.DB_PATH,  # type: ignore[attr-defined]
            "postgres_host": self.DB_HOST,  # type: ignore[attr-defined]
            "postgres_port": self.DB_PORT or 5432,  # type: ignore[attr-defined]
            "postgres_database": self.DB_NAME,  # type: ignore[attr-defined]
        }
