"""Environment-specific settings.

© 2025 Murat Unsal — Skyulf Project

Subclasses of Settings that override defaults for development, production, and testing.
Uses model_post_init to set overrides — avoids Pydantic "shadows parent" warnings.
"""

import logging
from typing import Any

from backend.config.base import Settings

# ── Helpers ──────────────────────────────────────────────────────────────────
_DEV_DEFAULTS: dict[str, Any] = {
    "DEBUG": True,
    "LOG_LEVEL": "DEBUG",
    "DB_ECHO": False,
    "HOST": "0.0.0.0",  # nosec B104 - dev-only default; intended to be reachable from containers/LAN
    "CORS_ORIGINS": ["*"],
}

_PROD_DEFAULTS: dict[str, Any] = {
    "DEBUG": False,
    "LOG_LEVEL": "INFO",
    "DB_ECHO": False,
    "CORS_ORIGINS": ["https://www.skyulf.com", "https://app.yourdomain.com"],
    "ALLOWED_HOSTS": ["skyulf.com", "app.yourdomain.com"],
}

_TEST_DEFAULTS: dict[str, Any] = {
    "TESTING": True,
    "DEBUG": True,
    "DATABASE_URL": "sqlite+aiosqlite:///./test_mlops.db",
    "LOG_LEVEL": "DEBUG",
    "ACCESS_TOKEN_EXPIRE_MINUTES": 480,
}

_PROD_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    # Canvas exports/maps and opt-in Swagger/ReDoc need these asset sources.
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' fonts.googleapis.com cdnjs.cloudflare.com "
        "https://cdn.jsdelivr.net; "
        "font-src 'self' fonts.gstatic.com; "
        "img-src 'self' data: blob: https://*.tile.openstreetmap.org "
        "https://fastapi.tiangolo.com https://cdn.redoc.ly/redoc/logo-mini.svg; "
        "worker-src 'self' blob:;"
    ),
}


def _apply_defaults(instance: Settings, defaults: dict[str, Any]) -> None:
    """Apply default values only for fields not explicitly set via env / .env."""
    for field, value in defaults.items():
        if not instance.is_field_set(field):
            object.__setattr__(instance, field, value)


# ── Environment Sub-classes ──────────────────────────────────────────────────
class DevelopmentSettings(Settings):
    """Development environment settings."""

    def model_post_init(self, __context: Any) -> None:
        """Apply the development overrides, then configure pandas and logging.

        Only fields the operator left unset are touched (see ``_apply_defaults``).
        This is the permissive profile — ``DEBUG=True``, ``CORS_ORIGINS=["*"]``
        and ``HOST`` bound to ``0.0.0.0`` — which is why ``resolve_environment()``
        fails closed on an unrecognized ``FASTAPI_ENV`` instead of landing here.
        """
        _apply_defaults(self, _DEV_DEFAULTS)
        super().model_post_init(__context)
        self.configure_pandas()
        self.setup_logging()
        logging.getLogger(__name__).info(
            "Running in DEVELOPMENT mode with enhanced ML development features"
        )


class ProductionSettings(Settings):
    """Production environment settings with enhanced security."""

    SECURITY_HEADERS: dict[str, str] = _PROD_SECURITY_HEADERS
    ML_MODEL_CACHE_SIZE: int = 5000
    DATA_SAMPLE_SIZE: int = 50000

    def model_post_init(self, __context: Any) -> None:
        """Apply the production overrides, then configure pandas and logging.

        ``DEBUG`` defaults to ``False`` and ``CORS_ORIGINS``/``ALLOWED_HOSTS`` to
        the production hostnames. Middleware applies ``SECURITY_HEADERS`` to
        HTTP responses. The two ML sizing fields exist only on this subclass
        and ``TestingSettings`` — the base ``Settings`` defines none of the three,
        so they are absent entirely under ``DevelopmentSettings``.
        """
        _apply_defaults(self, _PROD_DEFAULTS)
        super().model_post_init(__context)
        self.configure_pandas()
        self.setup_logging()
        logging.getLogger(__name__).info(
            "Running in PRODUCTION mode with enhanced security and ML capabilities"
        )


class TestingSettings(Settings):
    """Testing environment settings."""

    DATA_SAMPLE_SIZE: int = 100
    ML_MODEL_CACHE_SIZE: int = 10

    def model_post_init(self, __context: Any) -> None:
        """Apply the testing overrides, then configure logging.

        Redirects ``DATABASE_URL`` to a throwaway SQLite file and lengthens
        ``ACCESS_TOKEN_EXPIRE_MINUTES`` so a long test session does not expire
        mid-run. Note the asymmetry with the other two profiles: this one never
        calls ``configure_pandas()``, so pandas copy-on-write is not enabled
        under ``TestingSettings``.
        """
        _apply_defaults(self, _TEST_DEFAULTS)
        super().model_post_init(__context)
        self.setup_logging()
        logging.getLogger(__name__).info("Running in TESTING mode")
