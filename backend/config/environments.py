"""
Environment-Specific Settings

© 2025 Murat Unsal — Skyulf Project

Subclasses of Settings that override defaults for development, production, and testing.
Uses model_post_init to set overrides — avoids Pydantic "shadows parent" warnings.
"""

from typing import Any, Dict

from backend.config.base import Settings


# ── Helpers ──────────────────────────────────────────────────────────────────
_DEV_DEFAULTS: Dict[str, Any] = {
    "DEBUG": True,
    "LOG_LEVEL": "DEBUG",
    "DB_ECHO": False,
    "HOST": "0.0.0.0",
    "CORS_ORIGINS": ["*"],
    "SESSION_COOKIE_SECURE": False,
}

_PROD_DEFAULTS: Dict[str, Any] = {
    "DEBUG": False,
    "LOG_LEVEL": "INFO",
    "DB_ECHO": False,
    "CORS_ORIGINS": ["https://www.skyulf.com", "https://app.yourdomain.com"],
    "ALLOWED_HOSTS": ["skyulf.com", "app.yourdomain.com"],
    "SESSION_COOKIE_SECURE": True,
    "SESSION_COOKIE_HTTPONLY": True,
    "SESSION_COOKIE_SAMESITE": "Lax",
}

_TEST_DEFAULTS: Dict[str, Any] = {
    "TESTING": True,
    "DEBUG": True,
    "DATABASE_URL": "sqlite+aiosqlite:///./test_mlops.db",
    "LOG_LEVEL": "DEBUG",
    "ACCESS_TOKEN_EXPIRE_MINUTES": 480,
}

_PROD_SECURITY_HEADERS: Dict[str, str] = {
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


def _apply_defaults(instance: Settings, defaults: Dict[str, Any]) -> None:
    """Apply default values only for fields not explicitly set via env / .env."""
    for field, value in defaults.items():
        if not instance.is_field_set(field):
            object.__setattr__(instance, field, value)


# ── Environment Sub-classes ──────────────────────────────────────────────────
class DevelopmentSettings(Settings):
    """Development environment settings."""

    def model_post_init(self, __context: Any) -> None:
        _apply_defaults(self, _DEV_DEFAULTS)
        super().model_post_init(__context)
        self.configure_pandas()
        self.setup_logging()
        print("Running in DEVELOPMENT mode with enhanced ML development features")


class ProductionSettings(Settings):
    """Production environment settings with enhanced security."""

    SECURITY_HEADERS: Dict[str, str] = _PROD_SECURITY_HEADERS
    ML_MODEL_CACHE_SIZE: int = 5000
    DATA_SAMPLE_SIZE: int = 50000

    def model_post_init(self, __context: Any) -> None:
        _apply_defaults(self, _PROD_DEFAULTS)
        super().model_post_init(__context)
        self.configure_pandas()
        self.setup_logging()
        print("Running in PRODUCTION mode with enhanced security and ML capabilities")


class TestingSettings(Settings):
    """Testing environment settings."""

    DATA_SAMPLE_SIZE: int = 100
    ML_MODEL_CACHE_SIZE: int = 10

    def model_post_init(self, __context: Any) -> None:
        _apply_defaults(self, _TEST_DEFAULTS)
        super().model_post_init(__context)
        self.setup_logging()
        print("[TEST] Running in TESTING mode")
