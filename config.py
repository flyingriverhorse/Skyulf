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
) -> None:
    """
    Universal logging setup for FastAPI applications.
    Enhanced for async applications and modern Python practices.

    Args:
        log_file: Path to log file (creates directory if needed)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
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
        logging.WARNING
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
    API_LICENSE_NAME: str | None = "Proprietary"
    API_LICENSE_URL: str | None = None
    API_LOGO_URL: str | None = None

    # Session configuration
    PERMANENT_SESSION_LIFETIME: int = 28800  # 8 hours
    SESSION_COOKIE_SECURE: bool = False  # Will be overridden in production
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"

    # === PAGE-LEVEL SECURITY CONFIGURATION ===
    # Global setting: if True, ALL pages require authentication by default
    # Individual pages can be marked as public in PUBLIC_PAGES list
    REQUIRE_LOGIN_BY_DEFAULT: bool = True

    # Pages that are accessible without authentication (when REQUIRE_LOGIN_BY_DEFAULT=True)
    # or pages that require authentication (when REQUIRE_LOGIN_BY_DEFAULT=False)
    # Can be set as comma-separated string in environment variables
    PUBLIC_PAGES: str = "/login,/"

    # Pages that ALWAYS require admin privileges (overrides everything)
    # Can be set as comma-separated string in environment variables
    ADMIN_REQUIRED_PAGES: str = "/admin,/admin/*,/admin/dashboard,/admin/users,/health"

    # Redirect URL for unauthenticated users
    LOGIN_REDIRECT_URL: str = "/login"

    # Redirect URL after successful login
    LOGIN_SUCCESS_REDIRECT_URL: str = "/"

    # === USER REGISTRATION SETTINGS ===
    # Allow new user registration via the frontend
    ALLOW_USER_REGISTRATION: bool = True

    # Require email verification for new registrations
    REQUIRE_EMAIL_VERIFICATION: bool = False

    # === DATABASE CONFIGURATION ===
    DATABASE_URL: str = "sqlite+aiosqlite:///./mlops_database.db"
    DB_ECHO: bool = False  # Set to True for SQL query logging
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

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
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
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

    # === COMPREHENSIVE CLEANUP SYSTEM ===
    # Global cleanup enablement
    SYSTEM_CLEANUP_ENABLED: bool = True

    # Log file cleanup
    LOG_CLEANUP_ENABLED: bool = False
    LOG_CLEANUP_MAX_FILES: int = 10
    LOG_CLEANUP_MAX_AGE_DAYS: int = 30
    LOG_FILES_PATTERN: str = "*.log*"

    # Temporary files cleanup
    TEMP_CLEANUP_ENABLED: bool = False
    TEMP_CLEANUP_MAX_FILES: int = 5
    TEMP_CLEANUP_MAX_AGE_DAYS: int = 1
    TEMP_FILES_PATTERN: str = "*.*"

    # Export files cleanup
    EXPORT_CLEANUP_ENABLED: bool = False
    EXPORT_CLEANUP_MAX_FILES: int = 20
    EXPORT_CLEANUP_MAX_AGE_DAYS: int = 14
    EXPORT_FILES_PATTERN: str = "*.*"

    # Model cache cleanup
    MODEL_CLEANUP_ENABLED: bool = False
    MODEL_CLEANUP_MAX_FILES: int = 15
    MODEL_CLEANUP_MAX_AGE_DAYS: int = 7
    MODEL_FILES_PATTERN: str = "*.*"

    # Upload folder comprehensive cleanup
    UPLOAD_CLEANUP_ENABLED: bool = False
    UPLOAD_CLEANUP_MAX_FILES: int = 20
    UPLOAD_CLEANUP_MAX_AGE_DAYS: int = 7

    # Data folder cleanup (uploads/data and related data sources)
    DATA_CLEANUP_ENABLED: bool = True
    DATA_CLEANUP_MAX_FILES: int = 1
    DATA_CLEANUP_MAX_AGE_DAYS: int = 0
    DATA_CLEANUP_REMOVE_DATA_SOURCES: bool = True  # Remove related data sources from database
    DATA_FILES_PATTERN: str = "*.*"

    # Cleanup scheduling
    CLEANUP_ON_STARTUP: bool = False  # Run cleanup when application starts
    CLEANUP_SCHEDULER_ENABLED: bool = False  # Enable automatic scheduled cleanup
    CLEANUP_SCHEDULE_HOURS: int = 24  # How often to run cleanup (in hours)

    # Data source deletion settings
    DELETE_FILES_ON_SOURCE_REMOVAL: bool = True  # Delete physical files when data source is deleted
    ALLOW_DIRECTORY_DELETION: bool = False  # Allow deletion of directories (extra safety measure)
    FILES_ONLY_DELETION: bool = True  # Only delete files, never directories

    # User limits
    USER_MAX_DATA_SOURCES: int = 1

    # Data processing directories
    TEMP_DIR: str = "temp/processing"
    EXPORT_DIR: str = "exports/data"
    MODELS_DIR: str = "uploads/models"
    APP_DATA_DIR: str = "data"
    APP_DATA_DIR_CACHE: str = "data/cache"
    APP_DATA_DIR_EDA_CACHE: str = "data/eda_cache"
    APP_DATA_DIR_EDA_SESSIONS: str = "data/eda_sessions"
    APP_DATA_DIR_USER_EDA_HISTORY: str = "data/user_eda_history"

    # === PERFORMANCE SETTINGS ===
    ENABLE_DATABASE: bool = True
    DB_QUERY_TIMEOUT: int = 30  # Increased for complex ML queries
    MAX_SEARCH_RESULTS: int = 10000  # Increased for ML datasets
    SEARCH_TIMEOUT: int = 120  # Longer timeout for data processing

    # Controls how the AutoSyncManager is run
    SYNC_RUNNER_MODE: str = "disabled"  # 'disabled', 'inprocess', 'separate'

    # === PANDAS AND DATA PROCESSING (Enhanced for ML) ===
    PANDAS_OPTIONS: Dict[str, Any] = {
        "display.max_columns": 50,
        "display.max_rows": 100,
        "display.precision": 3,
        "mode.chained_assignment": "warn",
        "compute.use_numba": True,
    }

    # Data processing settings
    DATA_SAMPLE_SIZE: int = 10000  # Default sample size for previews
    DATA_CHUNK_SIZE: int = 50000  # Process data in chunks for memory efficiency
    DATA_LOADER_MAX_ROWS: int = 5000000  # Maximum rows to hold in memory

    # === ML PLATFORM CONFIGURATION ===
    # Used in training workflows for cache sizing (core/ml services)
    ML_MODEL_CACHE_SIZE: int = 1000  # Number of models to keep in memory
    # Used to guard long-running fits (core/ml services)
    ML_MAX_TRAINING_TIME: int = 3600  # 1 hour max training time
    # Used when splitting datasets in training utilities (core/ml services)
    ML_DEFAULT_TEST_SIZE: float = 0.2  # 20% test split by default
    # Used anywhere we seed random operations (core/ml services)
    ML_RANDOM_STATE: int = 42  # Reproducible results

    HYPERPARAMETER_TUNING_STRATEGIES: List[Dict[str, Any]] = [
        {
            "value": "random",
            "label": "Random search",
            "description": "Sample candidate hyperparameters uniformly at random.",
            "impl": "random",
            "aliases": ["random_search"],
        },
        {
            "value": "grid",
            "label": "Grid search",
            "description": "Evaluate every combination in the search space.",
            "impl": "grid",
            "aliases": ["grid_search"],
        },
        {
            "value": "halving",
            "label": "Successive halving (grid)",
            "description": "Successively allocate resources to the best grid candidates.",
            "impl": "halving",
            "aliases": ["successive_halving", "halving_grid"],
        },
        {
            "value": "halving_random",
            "label": "Successive halving (random)",
            "description": "Random sampling with successive halving to prune weak candidates.",
            "impl": "halving_random",
            "aliases": ["successive_halving_random", "halving_search"],
        },
        {
            "value": "optuna",
            "label": "Optuna (TPE)",
            "description": "Bayesian optimisation with pruning via Optuna.",
            "impl": "optuna",
            "aliases": ["bayesian", "optuna_tpe"],
        },
    ]

    # === ADVANCED EDA / GRANULAR RUNTIME SETTINGS ===
    # None indicates no imposed limit on row counts.
    # Analysis runtime: max rows before sampling (granular runtime module)
    GRANULAR_RUNTIME_MAX_ROWS: Optional[int] = None
    # Analysis runtime: sampling strategy used in the advanced EDA UI
    GRANULAR_RUNTIME_SAMPLING_STRATEGY: str = "random"
    # Analysis runtime: seed for deterministic sampling
    GRANULAR_RUNTIME_RANDOM_STATE: Optional[int] = 42
    # Analysis runtime: toggle for cached results in the analysis UI
    GRANULAR_RUNTIME_CACHE_ENABLED: bool = True
    # Analysis runtime: cache TTL for result reuse
    GRANULAR_RUNTIME_CACHE_TTL_SECONDS: int = 300
    # Analysis runtime: max cached entries for repeat runs
    GRANULAR_RUNTIME_CACHE_MAX_ITEMS: int = 128
    # Column insights: cache size backing async EDA service
    GRANULAR_CACHE_MAX_SIZE: int = 256
    # Column insights UI: TTL for cached column insights payload
    GRANULAR_COLUMN_INSIGHTS_CACHE_TTL: int = 300
    # Multi-analysis/code generation: TTL for cached analysis bundles (core/eda/advanced_eda/services.py)
    GRANULAR_ANALYSIS_CACHE_TTL: int = 180
    # Text analysis tiles: max rows fed into NLP routines (core/eda/advanced_eda/granular_runtime/text.py)
    GRANULAR_TEXT_NLP_SAMPLE_LIMIT: Optional[int] = None
    # Text analysis tiles: cap on stats (core/eda/advanced_eda/granular_runtime/text.py)
    GRANULAR_TEXT_FEATURE_SAMPLE_LIMIT: Optional[int] = None
    # Domain detection & dataset preview: row cutoff before returning full data (core/eda/advanced_eda/data_manager.py)
    EDA_PREVIEW_FULL_DATA_THRESHOLD: Optional[int] = None
    # Unified sample limit for EDA workflows (applied when explicitly configured)
    EDA_GLOBAL_SAMPLE_LIMIT: Optional[int] = None
    # Column insights sampling cap (core/eda/advanced_eda/services.py)
    EDA_COLUMN_INSIGHTS_SAMPLE_LIMIT: Optional[int] = 1000
    # Custom code execution rate limiting (core/eda/security/rate_limiter.py)
    EDA_CUSTOM_EXECUTIONS_PER_MINUTE: Optional[int] = 20
    EDA_CUSTOM_MAX_CONCURRENT_EXECUTIONS: Optional[int] = 1
    EDA_CUSTOM_RATE_CLEANUP_INTERVAL_SECONDS: Optional[int] = 300

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

    # === LLM CONFIGURATION (Comprehensive) ===
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_ORG_ID: Optional[str] = None
    OPENAI_DEFAULT_MODEL: str = "gpt-3.5-turbo"

    # DeepSeek Configuration
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_API_URL: str = "https://api.deepseek.com"
    DEEPSEEK_DEFAULT_MODEL: str = "deepseek-chat"
    DEEPSEEK_CODE_MODEL: str = "deepseek-coder"  # For code-specific tasks
    DEEPSEEK_MATH_MODEL: str = "deepseek-math"   # For mathematical analysis
    DEEPSEEK_TIMEOUT_SECONDS: int = 90
    DEEPSEEK_MAX_RETRIES: int = 2
    DEEPSEEK_RETRY_BACKOFF_SECONDS: float = 1.5

    # Anthropic Claude Configuration
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_DEFAULT_MODEL: str = "claude-3-haiku-20240307"

    # Local LLM Configuration (Ollama, LM Studio, etc.)
    LOCAL_LLM_URL: str = "http://localhost:11434"
    LOCAL_LLM_MODEL: str = "llama2"
    LOCAL_LLM_TYPE: str = "ollama"  # ollama, lmstudio, textgen

    # Default LLM settings
    DEFAULT_LLM_PROVIDER: str = "openai"
    DEFAULT_LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_MAX_HISTORY_MESSAGES: int = 12
    LLM_MAX_HISTORY_CHAR_LENGTH: int = 12000
    LLM_SYSTEM_PROMPT_CHAR_LIMIT: int = 6000
    LLM_USER_HISTORY_MESSAGES: int = 12
    LLM_USER_HISTORY_CHAR_LENGTH: int = 12000
    LLM_CELL_HISTORY_MESSAGES: int = 8
    LLM_CELL_HISTORY_CHAR_LENGTH: int = 8000

    # === DATA INGESTION FEATURE TOGGLES ===
    ENABLE_LINEAGE: bool = True
    ENABLE_SCHEMA_DRIFT: bool = True
    ENABLE_RETENTION: bool = True

    # === LOGGING ===
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/fastapi_app.log"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s [%(filename)s:%(lineno)d in %(funcName)s()]"
    LOG_MAX_SIZE: int = 50 * 1024 * 1024  # 50MB
    LOG_BACKUP_COUNT: int = 10
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

    # Granular Runtime Configuration
    # The validator now treats None and empty strings as "no limit" and
    # also accepts string aliases such as "none" or "unlimited".
    @field_validator("GRANULAR_RUNTIME_SAMPLING_STRATEGY")
    @classmethod
    def validate_granular_sampling_strategy(cls, v: str) -> str:
        if not v:
            return "random"
        normalized = v.lower()
        if normalized not in {"random", "head", "stratified"}:
            raise ValueError("GRANULAR_RUNTIME_SAMPLING_STRATEGY must be one of: random, head, stratified")
        return normalized

    # Granular Runtime Configuration
    @staticmethod
    def _parse_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in {"", "none", "null", "nan", "unlimited", "infinite", "inf"}:
                return None
            try:
                return int(stripped)
            except ValueError:
                raise ValueError(f"Could not parse integer from value '{value}'")
        return value

    @field_validator(
        "GRANULAR_RUNTIME_MAX_ROWS",
        "GRANULAR_RUNTIME_RANDOM_STATE",
        "GRANULAR_TEXT_NLP_SAMPLE_LIMIT",
        "GRANULAR_TEXT_FEATURE_SAMPLE_LIMIT",
        "EDA_PREVIEW_FULL_DATA_THRESHOLD",
        "EDA_GLOBAL_SAMPLE_LIMIT",
        "EDA_COLUMN_INSIGHTS_SAMPLE_LIMIT",
        "EDA_CUSTOM_EXECUTIONS_PER_MINUTE",
        "EDA_CUSTOM_MAX_CONCURRENT_EXECUTIONS",
        "EDA_CUSTOM_RATE_CLEANUP_INTERVAL_SECONDS",
        mode="before",
    )
    @classmethod
    def parse_optional_granular_ints(cls, v):
        return cls._parse_optional_int(v)

    @field_validator("PUBLIC_PAGES", mode="before")
    @classmethod
    def parse_public_pages(cls, v):
        """Parse PUBLIC_PAGES from environment variable string if needed."""
        if isinstance(v, str) and v:
            # Keep as string - we'll parse it in a property
            return v.strip()
        elif isinstance(v, list):
            # Convert list back to comma-separated string for consistency
            return ",".join(str(item).strip() for item in v if item)
        return v or "/login,/health,/"

    @field_validator("ADMIN_REQUIRED_PAGES", mode="before")
    @classmethod
    def parse_admin_pages(cls, v):
        """Parse ADMIN_REQUIRED_PAGES from environment variable string if needed."""
        if isinstance(v, str) and v:
            # Keep as string - we'll parse it in a property
            return v.strip()
        elif isinstance(v, list):
            # Convert list back to comma-separated string for consistency
            return ",".join(str(item).strip() for item in v if item)
        return v or "/admin,/admin/*,/admin/dashboard,/admin/users"

    @field_validator("ML_DEFAULT_TEST_SIZE")
    @classmethod
    def validate_test_size(cls, v):
        if not 0.0 < v < 1.0:
            raise ValueError("ML_DEFAULT_TEST_SIZE must be between 0 and 1")
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
            self.APP_DATA_DIR,
            self.APP_DATA_DIR_CACHE,
            self.APP_DATA_DIR_EDA_CACHE,
            self.APP_DATA_DIR_EDA_SESSIONS,
            self.APP_DATA_DIR_USER_EDA_HISTORY,
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

            # Apply all pandas options
            for option, value in self.PANDAS_OPTIONS.items():
                pd.set_option(option, value)

            # Enable copy-on-write mode for better memory efficiency (Pandas 2.x feature)
            pd.options.mode.copy_on_write = True

            print("✓ Pandas configured with optimized settings for ML workflows")

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

    @property
    def public_pages_list(self) -> List[str]:
        """Get PUBLIC_PAGES as a list of strings."""
        if isinstance(self.PUBLIC_PAGES, str):
            return [page.strip() for page in self.PUBLIC_PAGES.split(",") if page.strip()]
        return []

    @property
    def admin_required_pages_list(self) -> List[str]:
        """Get ADMIN_REQUIRED_PAGES as a list of strings."""
        if isinstance(self.ADMIN_REQUIRED_PAGES, str):
            return [page.strip() for page in self.ADMIN_REQUIRED_PAGES.split(",") if page.strip()]
        return []


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
        print("🧪 Running in TESTING mode")


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


# === ML PLATFORM UTILITIES (Migrated from Flask) ===

def get_optimal_chunk_size(file_size_mb: float) -> int:
    """
    Calculate optimal chunk size for processing large datasets.
    This helps with memory management in the ML pipeline.

    Args:
        file_size_mb: File size in megabytes

    Returns:
        Optimal chunk size for processing
    """
    if file_size_mb < 10:
        return 10000  # Small files: process all at once
    elif file_size_mb < 100:
        return 50000  # Medium files: moderate chunks
    elif file_size_mb < 1000:
        return 100000  # Large files: bigger chunks
    else:
        return 250000  # Very large files: large chunks for efficiency


def validate_ml_environment() -> Dict[str, Any]:
    """
    Validate that the ML environment is properly configured.
    This helps with troubleshooting when users report issues.

    Returns:
        Dictionary with validation results
    """
    results: Dict[str, Any] = {
        "pandas_version": None,
        "numpy_version": None,
        "sklearn_available": False,
        "pandas_performance_features": False,
        "memory_available_gb": None,
        "disk_space_gb": None,
        "fastapi_version": None,
        "pydantic_version": None,
    }

    try:
        import pandas as pd
        results["pandas_version"] = pd.__version__
        # Check if we have Pandas 2.x performance features
        results["pandas_performance_features"] = hasattr(pd.options.mode, "copy_on_write")
    except ImportError:
        pass

    try:
        import numpy as np
        results["numpy_version"] = np.__version__
    except ImportError:
        pass

    try:
        import fastapi
        results["fastapi_version"] = fastapi.__version__
    except ImportError:
        pass

    try:
        import pydantic
        results["pydantic_version"] = pydantic.VERSION
    except ImportError:
        pass

    import importlib.util
    results["sklearn_available"] = bool(importlib.util.find_spec("sklearn"))

    # Check system resources
    try:
        import psutil
        results["memory_available_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        results["disk_space_gb"] = round(psutil.disk_usage(".").free / (1024**3), 2)
    except (ImportError, OSError):
        pass

    return results


# Global settings instance accessor
def get_app_settings() -> Settings:
    """Convenience function to get application settings."""
    return get_settings()
