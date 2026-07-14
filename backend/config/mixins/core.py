"""Core application metadata and server settings."""

from importlib.metadata import PackageNotFoundError, version

try:
    _APP_VERSION = version("skyulf")
except PackageNotFoundError:
    # Fallback for dev environments where the package is not installed
    _APP_VERSION = "0.0.0-dev"


class CoreMixin:
    """App identity and server configuration."""

    APP_NAME: str = "Skyulf"
    # Single source of truth: pyproject.toml [project] version
    APP_VERSION: str = _APP_VERSION
    APP_SUMMARY: str = "Skyulf MLops service surface for data, experimentation, and automation."
    APP_DESCRIPTION: str = (
        "Programmatic interface for Skyulf's MLops platform covering data ingestion, model lifecycle, "
        "feature engineering, and analysis workflows."
    )
    DEBUG: bool = False
    TESTING: bool = False

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 1

    # Observability — opt-in; no-op when not set
    SENTRY_DSN: str | None = None
