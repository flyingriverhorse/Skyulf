"""Logging configuration settings."""

from typing import Optional

from backend.utils.logging_utils import setup_universal_logging


class LoggingMixin:
    """Log file, level, rotation, and format settings."""

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/fastapi_app.log"
    LOG_FORMAT: str = (
        "%(asctime)s [%(levelname)8s] %(name)s: %(message)s "
        "[%(filename)s:%(lineno)d in %(funcName)s()]"
    )
    LOG_MAX_SIZE: int = 50 * 1024 * 1024  # 50 MB
    LOG_BACKUP_COUNT: int = 5
    LOG_ROTATION_TYPE: str = "size"
    LOG_ROTATION_WHEN: Optional[str] = "midnight"
    LOG_ROTATION_INTERVAL: int = 1

    def setup_logging(self) -> None:
        """Initialize application logging."""
        setup_universal_logging(self.LOG_FILE, self.LOG_LEVEL)  # type: ignore[attr-defined]
