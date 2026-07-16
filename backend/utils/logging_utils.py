"""
Simple logging utility for data actions
Replaces the Flask log_data_action function
"""

import logging
import os
from logging import Handler
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path

# Get logger for data actions
data_logger = logging.getLogger("data_actions")


def log_data_action(action: str, success: bool = True, details: str | None = None):
    """
    Log data-related actions for monitoring and debugging

    Args:
        action: The action being performed
        success: Whether the action succeeded
        details: Additional details about the action
    """
    level = logging.INFO if success else logging.ERROR
    message = f"Action: {action}"

    if details:
        message += f" | Details: {details}"

    if not success:
        message += " | Status: FAILED"
    else:
        message += " | Status: SUCCESS"

    data_logger.log(level, message)


def _build_file_handler(
    log_file: str,
    log_level: str,
    rotation_type: str,
    rotation_when: str | None,
    rotation_interval: int,
    max_bytes: int,
    backup_count: int,
) -> Handler | None:
    """Build the rotating (size or time based) file handler, or None if it could not be created."""
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
            if os.name == "nt":
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
        return file_handler
    except OSError as e:
        # Fallback if file logging fails — use stderr since logging may not be ready
        import sys

        print(f"Warning: Could not setup file logging to {log_file}: {e}", file=sys.stderr)
        return None


def _build_console_handler(console_log_level: str) -> Handler:
    """Build the console handler, preferring Rich's handler and falling back to a plain one."""
    try:
        from rich.logging import RichHandler  # ty: ignore[unresolved-import]

        console_handler: Handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
        # Rich handler has its own formatter, no need to set one
    except ImportError:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)

    console_handler.setLevel(getattr(logging, console_log_level.upper(), logging.WARNING))
    return console_handler


def _silence_noisy_loggers() -> None:
    """Set known noisy third-party/framework loggers to WARNING to reduce log spam."""
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)


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
    log_dir = Path(log_file).parent
    if str(log_dir) != ".":  # Only create if there's actually a directory path
        log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove all existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Choose a file handler based on rotation_type (size or time)
    file_handler = _build_file_handler(
        log_file,
        log_level,
        rotation_type,
        rotation_when,
        rotation_interval,
        max_bytes,
        backup_count,
    )
    if file_handler is not None:
        root_logger.addHandler(file_handler)

    # Console handler with cleaner output for development
    console_handler = _build_console_handler(console_log_level)
    root_logger.addHandler(console_handler)

    # === NOISE REDUCTION ===
    # Set noisy libraries to WARNING or ERROR
    _silence_noisy_loggers()
