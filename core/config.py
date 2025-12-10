"""
FastAPI Configuration Management

© 2025 Murat Unsal — Skyulf Project

Comprehensive configuration system migrated from Flask with modern Pydantic validation.
Includes ML platform settings, LLM configurations, and feature management.
"""

import os
import logging
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from logging import Handler
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def setup_universal_logging(
    log_file: str = "logs/fastapi_app.log",
    log_level: str = "INFO",
    rotation_type: str = "size",
    rotation_when: str | None = None,
    rotation_interval: int = 1,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 10,
    console_log_level: str = "WARNING",
) -> None:
    """
    Universal logging setup for FastAPI applications.
    Enhanced for async applications and modern Python practices.

    Args:
        log_file: Path to log file (creates directory if needed)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_log_level: Logging level for console output
    """
    # Create log directory with better error handling
    log_dir = os.path.dirname(log_file)
    if log_dir:  # Only create if there's actually a directory path
        os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove all existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Choose a file handler based on rotation_type (size or time)
    try:
        file_handler: Handler
        if rotation_type and rotation_type.lower() in ("time", "timed"):
            # Use TimedRotatingFileHandler for time-based rotation
            when = rotation_when or "midnight"
            file_handler = TimedRotatingFileHandler(
                filename=log_file,
                when=when,
                interval=rotation_interval,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            # Default to size-based rotation
            # On Windows, RotatingFileHandler can cause PermissionError due to file locking
            if os.name == 'nt':
                file_handler = logging.FileHandler(
                    log_file,
                    encoding="utf-8",
                )
            else:
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Enhanced formatter with more context for debugging
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s: %(message)s "
            "[%(filename)s:%(lineno)d in %(funcName)s()]"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except (OSError, IOError) as e:
        # Fallback if file logging fails
        print(f"Warning: Could not setup file logging to {log_file}: {e}")

    # Console handler with cleaner output for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        getattr(logging, console_log_level.upper(), logging.WARNING)
    )  # Only show warnings and errors in console
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Silence overly verbose loggers from dependencies
    verbose_loggers = [
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "urllib3",
        "requests",
        "aiohttp",
        "uvicorn.access",
        "snowflake.connector",
    ]

    for logger_name in verbose_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Success message with version info
    root_logger.info(
        f"Universal logging initialized. Log file: {log_file}, Level: {log_level}"
    )


class Settings(BaseSettings):
    """
    Comprehensive application settings with automatic environment variable loading.
    Migrated and enhanced from Flask configuration with Pydantic validation.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )

    # === CORE APPLICATION METADATA ===
    APP_NAME: str = "Skyulf"
    APP_VERSION: str = "0.0.3"
    APP_SUMMARY: str = "Skyulf MLops service surface for data, experimentation, and automation."
    APP_DESCRIPTION: str = (
        "Programmatic interface for Skyulf's MLops platform covering data ingestion, model lifecycle, "
        "feature engineering, and analysis workflows."
    )
    DEBUG: bool = False
    TESTING: bool = False

    # Server configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 1

    # === SECURITY ===
    SECRET_KEY: str = secrets.token_urlsafe(32)

    # JWT Configuration
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # 7 days
    ALGORITHM: str = "HS256"

    # Security Configuration
    MAX_LOGIN_ATTEMPTS: int = 5
    ACCOUNT_LOCKOUT_DURATION_MINUTES: int = 30

    # Developer fallback authentication (disabled in production)
    AUTH_FALLBACK_ENABLED: bool = True
    AUTH_FALLBACK_USERNAME: str = "admin"
    AUTH_FALLBACK_PASSWORD: str = "admin123"
    AUTH_FALLBACK_DISPLAY_NAME: str = "Fallback Admin"
    AUTH_FALLBACK_IS_ADMIN: bool = True

    # CORS and security
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    API_DOCS_ENABLED: bool | None = None
    API_DOCS_URL: str = "/docs"
    API_REDOC_URL: str = "/redoc"
    API_OPENAPI_URL: str = "/openapi.json"
    API_DOCS_DEFAULT_MODELS_EXPAND_DEPTH: int | None = -1
    API_DOCS_DEFAULT_DOC_EXPANSION: str | None = "list"
    API_DOCS_PERSIST_AUTH: bool = True
    API_DOCS_DISPLAY_REQUEST_DURATION: bool = True
    API_DOCS_ENABLE_FILTER: bool = True
    API_DOCS_ENABLE_TRY_IT_OUT: bool = True
    API_DOCS_SERVERS: List[str] = []
    API_CONTACT_NAME: str | None = "Skyulf Support"
    API_CONTACT_EMAIL: str | None = None
    API_CONTACT_URL: str | None = None
    API_TOS_URL: str | None = None
    API_LICENSE_NAME: str | None = "Apache 2.0"
    API_LICENSE_URL: str | None = "https://www.apache.org/licenses/LICENSE-2.0.html"
    API_LOGO_URL: str | None = None

    # Session configuration
    PERMANENT_SESSION_LIFETIME: int = 28800  # 8 hours
    SESSION_COOKIE_SECURE: bool = False  # Will be overridden in production
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"

    # === PAGE-LEVEL SECURITY CONFIGURATION ===
    # Redirect URL for unauthenticated users
    LOGIN_REDIRECT_URL: str = "/login"

    # Redirect URL after successful login
    LOGIN_SUCCESS_REDIRECT_URL: str = "/"

    # === USER REGISTRATION SETTINGS ===
    # Allow new user registration via the frontend
    ALLOW_USER_REGISTRATION: bool = True

    # === DATABASE CONFIGURATION ===
    DATABASE_URL: str = "sqlite+aiosqlite:///./mlops_database.db"
    DB_ECHO: bool = False  # Set to True for SQL query logging
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # === PIPELINE STORAGE ===
    # Options: "database" (default) or "json"
    PIPELINE_STORAGE_TYPE: str = "database"
    PIPELINE_STORAGE_PATH: str = "exports/pipelines"

    # Database configuration options (migrated from Flask)
    DB_TYPE: str = "sqlite"  # sqlite or postgres
    DB_PRIMARY: str = "sqlite"  # primary database choice
    DB_PATH: str = "mlops_database.db"  # SQLite database file path

    DB_PROVIDER: Optional[str] = None
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = None
    DB_NAME: Optional[str] = None
    DB_SSLMODE: Optional[str] = None  # e.g. require, verify-full
    DB_SSLROOTCERT: Optional[str] = None  # path to CA bundle for verify-full
    DB_EXTRA_PARAMS: Optional[str] = None  # e.g. application_name=mlops

    # === BACKGROUND TASKS / CELERY ===
    CELERY_BROKER_URL: str = "redis://127.0.0.1:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://127.0.0.1:6379/0"
    CELERY_TASK_DEFAULT_QUEUE: str = "mlops-training"

    # Model training artifact storage
    TRAINING_ARTIFACT_DIR: str = "uploads/models"

    # === FILE HANDLING (Enhanced for ML workflows) ===
    UPLOAD_DIR: str = "uploads/data"
    MAX_UPLOAD_SIZE: int = 1024 * 1024 * 1024  # 1GB for large ML datasets
    ALLOWED_EXTENSIONS: List[str] = [
        ".csv", ".xlsx", ".xls", ".parquet", ".json", ".txt",
        ".pkl", ".pickle", ".feather", ".h5", ".hdf5"
    ]

    # Data processing directories
    TEMP_DIR: str = "temp/processing"
    EXPORT_DIR: str = "exports/data"
    MODELS_DIR: str = "uploads/models"

    # === SNOWFLAKE CONFIGURATION (Enhanced) ===
    SNOWFLAKE_CONNECTION_TYPE: str = "native"  # Modern native connection
    SNOWFLAKE_ACCOUNT: Optional[str] = None
    SNOWFLAKE_USER: Optional[str] = None
    SNOWFLAKE_PASSWORD: Optional[str] = None
    SNOWFLAKE_WAREHOUSE: Optional[str] = None
    SNOWFLAKE_DATABASE: Optional[str] = None
    SNOWFLAKE_ROLE: Optional[str] = None
    SNOWFLAKE_SCHEMA: Optional[str] = None
    FEATURE_SNOWFLAKE: bool = False

    # === DATA INGESTION FEATURE TOGGLES ===
    ENABLE_LINEAGE: bool = True
    ENABLE_SCHEMA_DRIFT: bool = True
    ENABLE_RETENTION: bool = True

    # === LOGGING ===
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/fastapi_app.log"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s [%(filename)s:%(lineno)d in %(funcName)s()]"
    LOG_MAX_SIZE: int = 50 * 1024 * 1024  # 50MB
    LOG_BACKUP_COUNT: int = 5
    # Rotation strategy: 'size' (default) or 'time'
    LOG_ROTATION_TYPE: str = "size"
    # When using time-based rotation, this controls the 'when' argument
    # Accepts values like 'midnight', 'D', 'H', 'M', 'S', or 'W0'-'W6'
    LOG_ROTATION_WHEN: Optional[str] = "midnight"
    # Interval for time-based rotation (e.g., 1 day when when='midnight')
    LOG_ROTATION_INTERVAL: int = 1

    # === CACHE CONFIGURATION ===
    CACHE_TYPE: str = "filesystem"  # Good for ML model caching
    CACHE_DEFAULT_TIMEOUT: int = 3600  # 1 hour default
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 300  # 5 minutes default

    # === BACKGROUND TASKS ===
    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v

    @field_validator("API_DOCS_SERVERS", mode="before")
    @classmethod
    def parse_api_docs_servers(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @model_validator(mode="after")
    def validate_database_config(self):
        """Validate database configuration consistency and auto-construct DATABASE_URL."""
        # Auto-construct DATABASE_URL based on DB_TYPE if not explicitly set to PostgreSQL
        if self.DB_TYPE == "postgres":
            if not self.DATABASE_URL.startswith("postgresql"):
                # Construct PostgreSQL URL from individual settings
                if not all([self.DB_USER, self.DB_PASSWORD, self.DB_HOST, self.DB_NAME]):
                    raise ValueError(
                        "For PostgreSQL, either provide DATABASE_URL or all of: "
                        "DB_USER, DB_PASSWORD, DB_HOST, DB_NAME"
                    )
                # Auto-construct the DATABASE_URL
                port = self.DB_PORT or 5432
                self.DATABASE_URL = (
                    "postgresql+asyncpg://"
                    f"{self.DB_USER}:{self.DB_PASSWORD}@"
                    f"{self.DB_HOST}:{port}/{self.DB_NAME}"
                )
                if self.DB_SSLMODE:
                    self.DATABASE_URL += f"?sslmode={self.DB_SSLMODE}"
        elif self.DB_TYPE == "sqlite":
            # Ensure SQLite URL format
            if not self.DATABASE_URL.startswith("sqlite"):
                db_path = self.DB_PATH if self.DB_PATH else "mlops_database.db"
                self.DATABASE_URL = f"sqlite+aiosqlite:///./{db_path}"

        return self

    def create_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories: List[str | Path] = [
            self.UPLOAD_DIR,
            self.TEMP_DIR,
            self.EXPORT_DIR,
            self.MODELS_DIR,
            Path(self.LOG_FILE).parent,
            "logs",
            "exports/models",
            # "backups",
        ]

        for directory in directories:
            target_path = directory if isinstance(directory, Path) else Path(directory)
            target_path.mkdir(parents=True, exist_ok=True)

        print(f"Created {len(directories)} application directories")

    def validate_snowflake_config(self) -> bool:
        """Validate that Snowflake configuration is complete."""
        if not self.FEATURE_SNOWFLAKE:
            return True  # Skip validation if feature is disabled

        required_configs = [
            self.SNOWFLAKE_ACCOUNT,
            self.SNOWFLAKE_USER,
            self.SNOWFLAKE_PASSWORD,
            self.SNOWFLAKE_WAREHOUSE,
            self.SNOWFLAKE_DATABASE,
        ]

        missing_configs = []
        for i, config in enumerate(required_configs):
            if not config or config in ["x", "your-account", "your-user"]:
                config_names = [
                    "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
                    "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE"
                ]
                missing_configs.append(config_names[i])

        if missing_configs:
            error_msg = (
                f"Missing or placeholder Snowflake configurations: {', '.join(missing_configs)}\n"
                f"Please set these in environment variables"
            )
            raise ValueError(error_msg)

        return True

    def configure_pandas(self) -> None:
        """Configure Pandas with optimal settings for the ML platform."""
        try:
            import pandas as pd

            # Enable copy-on-write mode for better memory efficiency (Pandas 2.x feature)
            pd.options.mode.copy_on_write = True

            # Avoid non-ASCII to prevent Windows console encoding issues
            print("OK Pandas configured with optimized settings for ML workflows")

        except ImportError:
            print("WARNING: Pandas not available, skipping configuration")

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration as a dictionary for the LLM service."""
        return {
            # OpenAI
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "OPENAI_ORG_ID": self.OPENAI_ORG_ID,
            "OPENAI_DEFAULT_MODEL": self.OPENAI_DEFAULT_MODEL,

            # DeepSeek
            "DEEPSEEK_API_KEY": self.DEEPSEEK_API_KEY,
            "DEEPSEEK_API_URL": self.DEEPSEEK_API_URL,
            "DEEPSEEK_DEFAULT_MODEL": self.DEEPSEEK_DEFAULT_MODEL,
            "DEEPSEEK_CODE_MODEL": self.DEEPSEEK_CODE_MODEL,
            "DEEPSEEK_MATH_MODEL": self.DEEPSEEK_MATH_MODEL,
            "DEEPSEEK_TIMEOUT_SECONDS": self.DEEPSEEK_TIMEOUT_SECONDS,
            "DEEPSEEK_MAX_RETRIES": self.DEEPSEEK_MAX_RETRIES,
            "DEEPSEEK_RETRY_BACKOFF_SECONDS": self.DEEPSEEK_RETRY_BACKOFF_SECONDS,

            # Anthropic
            "ANTHROPIC_API_KEY": self.ANTHROPIC_API_KEY,
            "CLAUDE_DEFAULT_MODEL": self.CLAUDE_DEFAULT_MODEL,

            # Local LLM
            "LOCAL_LLM_URL": self.LOCAL_LLM_URL,
            "LOCAL_LLM_MODEL": self.LOCAL_LLM_MODEL,
            "LOCAL_LLM_TYPE": self.LOCAL_LLM_TYPE,

            # Defaults
            "DEFAULT_LLM_PROVIDER": self.DEFAULT_LLM_PROVIDER,
            "DEFAULT_LLM_MODEL": self.DEFAULT_LLM_MODEL,
            "LLM_MAX_HISTORY_MESSAGES": self.LLM_MAX_HISTORY_MESSAGES,
            "LLM_MAX_HISTORY_CHAR_LENGTH": self.LLM_MAX_HISTORY_CHAR_LENGTH,
            "LLM_SYSTEM_PROMPT_CHAR_LIMIT": self.LLM_SYSTEM_PROMPT_CHAR_LIMIT,
            "LLM_USER_HISTORY_MESSAGES": self.LLM_USER_HISTORY_MESSAGES,
            "LLM_USER_HISTORY_CHAR_LENGTH": self.LLM_USER_HISTORY_CHAR_LENGTH,
            "LLM_CELL_HISTORY_MESSAGES": self.LLM_CELL_HISTORY_MESSAGES,
            "LLM_CELL_HISTORY_CHAR_LENGTH": self.LLM_CELL_HISTORY_CHAR_LENGTH,
        }

    def get_sqlite_url(self) -> str:
        """Get SQLite database URL."""
        db_path = self.DB_PATH if self.DB_PATH else "mlops_database.db"
        # Ensure path is in app_data directory
        if not db_path.startswith("/"):
            db_path = f"{db_path}"
        return f"sqlite+aiosqlite:///./{db_path}"

    def get_postgresql_url(self) -> str:
        """Get PostgreSQL database URL from configuration."""
        if not all([self.DB_USER, self.DB_PASSWORD, self.DB_HOST, self.DB_NAME]):
            raise ValueError("PostgreSQL configuration incomplete: missing DB_USER, DB_PASSWORD, DB_HOST, or DB_NAME")

        port = self.DB_PORT or 5432
        url = f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{port}/{self.DB_NAME}"

        # Add SSL mode if specified
        if self.DB_SSLMODE:
            url += f"?sslmode={self.DB_SSLMODE}"

        return url

    def get_database_url_for_type(self, db_type: str) -> str:
        """Get database URL for the specified database type."""
        if db_type.lower() == "sqlite":
            return self.get_sqlite_url()
        elif db_type.lower() in ["postgresql", "postgres"]:
            return self.get_postgresql_url()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def get_database_config_info(self) -> Dict[str, Any]:
        """Get database configuration information for admin dashboard."""
        return {
            "primary_db": self.DB_PRIMARY,
            "current_db_type": "sqlite" if self.DATABASE_URL.startswith("sqlite") else "postgresql",
            "sqlite_configured": True,  # SQLite is always available
            "postgresql_configured": all([self.DB_USER, self.DB_HOST, self.DB_NAME]),
            "sqlite_label": "SQLite",
            "postgres_label": "PostgreSQL",
            "sqlite_path": self.DB_PATH,
            "postgres_host": self.DB_HOST,
            "postgres_port": self.DB_PORT or 5432,
            "postgres_database": self.DB_NAME
        }

    def setup_logging(self) -> None:
        """Initialize application logging."""
        setup_universal_logging(self.LOG_FILE, self.LOG_LEVEL)

    def is_field_set(self, field_name: str) -> bool:
        """Return True if the given settings field was explicitly provided by the user."""
        return field_name in getattr(self, "model_fields_set", set())



class DevelopmentSettings(Settings):
    """Development environment settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DB_ECHO: bool = False  # Disable SQL query logging
    HOST: str = "0.0.0.0"  # Allow external connections in dev
    CORS_ORIGINS: List[str] = ["*"]  # Allow all origins in development

    # Development-specific ML settings

    # Relaxed security for development (no CSRF for easier API testing)
    SESSION_COOKIE_SECURE: bool = False

    def init_dev_features(self) -> None:
        """Initialize development-specific features."""
        self.configure_pandas()
        self.setup_logging()
        print("Running in DEVELOPMENT mode with enhanced ML development features")


class ProductionSettings(Settings):
    """Production environment settings with enhanced security."""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DB_ECHO: bool = False

    # More restrictive CORS in production
    CORS_ORIGINS: List[str] = [
        "https://www.skyulf.com",
        "https://app.yourdomain.com"
    ]

    # Security headers and restrictions
    ALLOWED_HOSTS: List[str] = ["skyulf.com", "app.yourdomain.com"]

    # Production security settings
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"

    # Production ML platform settings
    ML_MODEL_CACHE_SIZE: int = 5000  # Larger cache for production
    DATA_SAMPLE_SIZE: int = 50000  # Larger samples for production analysis

    # Security headers for modern web security
    SECURITY_HEADERS: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' fonts.googleapis.com cdnjs.cloudflare.com; "
            "font-src 'self' fonts.gstatic.com;"
        ),
    }

    def init_production_features(self) -> None:
        """Initialize production-specific features."""
        self.configure_pandas()
        self.setup_logging()
        print("Running in PRODUCTION mode with enhanced security and ML capabilities")


class TestingSettings(Settings):
    """Testing environment settings."""
    TESTING: bool = True
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite+aiosqlite:///./test_mlops.db"
    LOG_LEVEL: str = "DEBUG"

    # Reasonable token expiration for development
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours instead of 5 minutes

    # Smaller data processing limits for testing
    DATA_SAMPLE_SIZE: int = 100
    ML_MODEL_CACHE_SIZE: int = 10

    def init_test_features(self) -> None:
        """Initialize testing-specific features."""
        self.setup_logging()
        print("[TEST] Running in TESTING mode")


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings based on environment.
    Uses lru_cache to avoid recreating settings on every call.
    """
    env = os.getenv("FASTAPI_ENV", "development").lower()

    if env == "production":
        production_settings = ProductionSettings()
        production_settings.init_production_features()
        settings: Settings = production_settings
    elif env == "testing":
        testing_settings = TestingSettings()
        testing_settings.init_test_features()
        settings = testing_settings
    else:
        development_settings = DevelopmentSettings()
        development_settings.init_dev_features()
        settings = development_settings

    # Create necessary directories
    settings.create_directories()

    return settings


# Global settings instance accessor
def get_app_settings() -> Settings:
    """Convenience function to get application settings."""
    return get_settings()
