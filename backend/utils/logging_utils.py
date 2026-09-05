"""
Simple logging utility for data actions
Replaces the Flask log_data_action function
"""

import logging
import os
import re
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


# C0 control block (includes CR/LF) plus DEL, mapped to a visible escape.
_CONTROL_ESCAPES = {**{c: f"\\x{c:02x}" for c in range(0x20)}, 0x7F: "\\x7f"}


def sanitize_for_log(value: object) -> str:
    """Render a possibly user-controlled value so it cannot forge log lines.

    Values reaching the backend from HTTP path parameters or uploaded filenames
    can carry CR/LF, letting a caller split one log record into several and
    fabricate entries that never happened (CWE-117). Control characters are
    escaped rather than deleted so an injection attempt stays visible.

    Args:
        value: The value to render.

    Returns:
        A single-line string safe to interpolate into a log record.
    """
    return str(value).translate(_CONTROL_ESCAPES)


# AWS access key IDs are 20 characters: a 4-character prefix naming the
# credential type, then 16 uppercase alphanumerics.
_AWS_KEY_ID_RE = re.compile(
    r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}\b"
)

# Option names whose *value* is a credential. `key` is included because both S3
# modules use it as the access-key-id option name (`key` <-> `aws_access_key_id`).
_SECRET_NAMES = (
    "x-amz-credential",
    "x-amz-security-token",
    "x-amz-signature",
    "aws_secret_access_key",
    "aws_access_key_id",
    "awsaccesskeyid",
    "secret_access_key",
    "session_token",
    "security_token",
    "signature",
    "secret",
    "password",
    "key",
)

# `name=value`, `name: value` and the quoted/dict-repr forms of both. The value
# charset stops at quotes, separators and brackets so a value already replaced
# by an earlier pass is not matched again, and so the scrub cannot run past the
# credential into the surrounding prose.
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?P<name>" + "|".join(_SECRET_NAMES) + r")\b"
    r"(?P<sep>['\"]?\s*[:=]\s*['\"]?)"
    r"(?P<value>[^\s,&'\"<>{}\[\]()]+)",
    re.IGNORECASE,
)

# S3's own 403 body reports the signature and string-to-sign as XML elements,
# where the separator is `>` rather than `=` or `:`.
_SECRET_XML_RE = re.compile(
    r"<(?P<tag>SignatureProvided|StringToSign|StringToSignBytes"
    r"|AWSSecretAccessKey|SecretAccessKey|SessionToken)>"
    r"(?P<value>[^<]*)</(?P=tag)>",
    re.IGNORECASE,
)


def redact_credentials(value: object) -> str:
    """Scrub AWS credentials and signatures out of text destined for a log.

    Redaction keys off the *shape of a credential*, not the name of a setting:
    S3 surfaces secrets in forms containing no option name at all — its own 403
    XML body, and any presigned URL. A presigned URL is a bearer credential, so
    whoever can read the log can replay it against the object until it expires
    with no AWS account required.

    Each secret is replaced in place rather than discarding the whole message,
    so the surrounding diagnostic survives. Matching on key names instead
    produced both failure modes at once: real credentials passed through
    unredacted, while a benign object-key message (`key=reports/q3.csv`) lost
    its entire text.

    Args:
        value: The value to render.

    Returns:
        The string form of ``value`` with credentials replaced by ``[REDACTED]``.
    """
    text = str(value)
    text = _AWS_KEY_ID_RE.sub("[REDACTED]", text)
    text = _SECRET_ASSIGNMENT_RE.sub(
        lambda m: f"{m.group('name')}{m.group('sep')}[REDACTED]",
        text,
    )
    return _SECRET_XML_RE.sub(
        lambda m: f"<{m.group('tag')}>[REDACTED]</{m.group('tag')}>",
        text,
    )


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
