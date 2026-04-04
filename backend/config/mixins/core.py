"""Core application metadata and server settings."""


class CoreMixin:
    """App identity and server configuration."""

    APP_NAME: str = "Skyulf"
    APP_VERSION: str = "0.4.1"
    APP_SUMMARY: str = (
        "Skyulf MLops service surface for data, experimentation, and automation."
    )
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
