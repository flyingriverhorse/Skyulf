"""
Base Settings Model

© 2025 Murat Unsal — Skyulf Project

Combines all domain mixins into one flat ``Settings`` class.
Each mixin lives in ``backend/config/mixins/`` and owns one domain's fields.
"""

import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, cast

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.config.mixins.aws import AWSMixin
from backend.config.mixins.cache import CacheMixin
from backend.config.mixins.celery import CeleryMixin
from backend.config.mixins.core import CoreMixin
from backend.config.mixins.database import DatabaseMixin
from backend.config.mixins.files import FilesMixin
from backend.config.mixins.llm import LLMMixin
from backend.config.mixins.logging import LoggingMixin
from backend.config.mixins.security import SecurityMixin
from backend.config.mixins.snowflake import SnowflakeMixin


class Settings(
    CoreMixin,
    AWSMixin,
    SecurityMixin,
    DatabaseMixin,
    CeleryMixin,
    FilesMixin,
    SnowflakeMixin,
    LoggingMixin,
    CacheMixin,
    LLMMixin,
    BaseSettings,
):
    """
    Application settings assembled from domain mixins.

    All fields stay flat — use ``settings.FIELD_NAME`` directly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Validators ────────────────────────────────────────────────────

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    @model_validator(mode="after")
    def validate_fallback_auth(self) -> "Settings":
        """Reject a blank fallback password when fallback auth is enabled."""
        if self.AUTH_FALLBACK_ENABLED and not self.AUTH_FALLBACK_PASSWORD:
            raise ValueError(
                "AUTH_FALLBACK_PASSWORD must be explicitly set when AUTH_FALLBACK_ENABLED=true. "
                "Set a strong password in your .env file."
            )
        return self

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                try:
                    return cast(list[str], json.loads(v))
                except json.JSONDecodeError:
                    v = v[1:-1]  # strip brackets, fall through to split
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return cast(list[str], v)

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                try:
                    return cast(list[str], json.loads(v))
                except json.JSONDecodeError:
                    v = v[1:-1]  # strip brackets, fall through to split
            return [host.strip() for host in v.split(",") if host.strip()]
        return cast(list[str], v)

    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                try:
                    return cast(list[str], json.loads(v))
                except json.JSONDecodeError:
                    v = v[1:-1]  # strip brackets, fall through to split
            return [ext.strip() for ext in v.split(",") if ext.strip()]
        return cast(list[str], v)

    @field_validator("API_DOCS_SERVERS", mode="before")
    @classmethod
    def parse_api_docs_servers(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return cast(list[str], v)

    @model_validator(mode="after")
    def validate_database_config(self) -> "Settings":
        """Auto-construct DATABASE_URL from component fields when needed."""
        # pylint: disable=access-member-before-definition
        if self.DB_TYPE == "postgres":
            if not self.DATABASE_URL.startswith(("postgresql", "postgres")):
                if not all([self.DB_USER, self.DB_PASSWORD, self.DB_HOST, self.DB_NAME]):
                    raise ValueError(
                        "For PostgreSQL, either provide DATABASE_URL or all of: "
                        "DB_USER, DB_PASSWORD, DB_HOST, DB_NAME"
                    )
                port = self.DB_PORT or 5432
                pwd = urllib.parse.quote_plus(self.DB_PASSWORD) if self.DB_PASSWORD else ""
                url = (
                    f"postgresql+asyncpg://{self.DB_USER}:{pwd}@"
                    f"{self.DB_HOST}:{port}/{self.DB_NAME}"
                )
                params = []
                if self.DB_SSLMODE:
                    params.append(f"sslmode={self.DB_SSLMODE}")
                if self.DB_SSLROOTCERT:
                    params.append(f"sslrootcert={self.DB_SSLROOTCERT}")
                if self.DB_EXTRA_PARAMS:
                    params.append(self.DB_EXTRA_PARAMS)

                if params:
                    url += f"?{'&'.join(params)}"

                self.DATABASE_URL = url
        elif self.DB_TYPE == "sqlite":
            if not self.DATABASE_URL.startswith("sqlite"):
                db_path = self.DB_PATH or "mlops_database.db"
                if Path(db_path).is_absolute():
                    self.DATABASE_URL = f"sqlite+aiosqlite:///{db_path}"
                else:
                    self.DATABASE_URL = f"sqlite+aiosqlite:///./{db_path}"
        return self

    @model_validator(mode="after")
    def validate_production_secret_key(self) -> "Settings":
        """Fail fast if SECRET_KEY was not explicitly set in production.

        The default value is a randomly generated token — different per process
        and per restart — which breaks JWT validation across workers and makes
        all existing tokens invalid after a restart.
        """
        env = os.getenv("FASTAPI_ENV", "development").lower()
        if env == "production" and not self.is_field_set("SECRET_KEY"):
            raise ValueError(
                "SECRET_KEY must be explicitly set in production environments. "
                "Add SECRET_KEY=<value> to your .env file or environment. "
                "Generate a strong key with: "
                'python -c "import secrets; print(secrets.token_urlsafe(64))"'
            )
        return self

    # ── Utility ───────────────────────────────────────────────────────

    def configure_pandas(self) -> None:
        """Configure Pandas with optimal settings for the ML platform."""
        _logger = logging.getLogger(__name__)
        try:
            import pandas as pd

            pd.options.mode.copy_on_write = True
            _logger.info("Pandas configured with optimized settings for ML workflows")
        except ImportError:
            _logger.warning("Pandas not available, skipping configuration")

    def is_field_set(self, field_name: str) -> bool:
        """Return True if the given settings field was explicitly provided by the user."""
        return field_name in getattr(self, "model_fields_set", set())
