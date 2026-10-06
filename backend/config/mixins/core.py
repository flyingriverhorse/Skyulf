"""Core application metadata and server settings."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal


def _application_version() -> str:
    """Prefer installed release metadata, then the version of this source checkout."""
    try:
        installed_version = version("skyulf")
    except PackageNotFoundError:
        installed_version = None
    if isinstance(installed_version, str) and installed_version.strip():
        return installed_version

    try:
        manifest = Path(__file__).resolve().parents[3] / "pyproject.toml"
        with manifest.open("rb") as source:
            source_version = tomllib.load(source)["project"]["version"]
    except (OSError, ValueError, KeyError, TypeError):
        return "0.0.0-dev"
    return (
        source_version
        if isinstance(source_version, str) and source_version.strip()
        else "0.0.0-dev"
    )


_APP_VERSION = _application_version()


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

    # DataFrame engine used end-to-end by dataset reads and pipeline execution
    # (backend Polars migration, Phase 1). Polars is the platform default;
    # pandas remains a first-class, explicitly selectable option.
    SKYULF_ENGINE: Literal["polars", "pandas"] = "polars"

    # Environment name reported by health checks / monitoring (e.g. "development",
    # "staging", "production"). Read from the ENVIRONMENT env var. When unset, falls
    # back to deriving from DEBUG (see Settings.environment_name) to preserve prior
    # behavior for deployments that don't set it explicitly.
    ENVIRONMENT: str | None = None

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 1

    # Observability — opt-in; no-op when not set
    SENTRY_DSN: str | None = None

    # ── Pagination defaults ──────────────────────────────────────────────
    # Fallback page size used by list endpoints that don't declare a more
    # specific default of their own. Centralized so "what's a reasonable
    # default page size" is a single tunable instead of ad-hoc literals
    # (previously 5/20/50/100/200 scattered across routers/services).
    DEFAULT_PAGE_SIZE: int = 50
    # Upper bound list endpoints should clamp a caller-supplied `limit` to,
    # regardless of what the endpoint's own default is.
    MAX_PAGE_SIZE: int = 500
    # Default number of rows returned by lightweight "preview"/"sample"
    # endpoints (e.g. dataset sample previews).
    DEFAULT_SAMPLE_ROWS: int = 5
