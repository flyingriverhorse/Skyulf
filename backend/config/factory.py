"""
Settings Factory

© 2025 Murat Unsal — Skyulf Project

Creates the correct Settings subclass based on the FASTAPI_ENV environment variable.
"""

from functools import lru_cache

from backend.config.base import Settings, resolve_environment
from backend.config.environments import (
    DevelopmentSettings,
    ProductionSettings,
    TestingSettings,
)

# Must cover exactly ``KNOWN_ENVIRONMENTS``; ``resolve_environment()`` guarantees
# the lookup key is one of them, so a gap here fails loudly rather than silently
# selecting the most permissive profile.
_ENV_SETTINGS_MAP: dict[str, type[Settings]] = {
    "development": DevelopmentSettings,
    "production": ProductionSettings,
    "testing": TestingSettings,
}


@lru_cache
def get_settings() -> Settings:
    """
    Get application settings based on ``FASTAPI_ENV``.

    Returns a cached singleton so the config is built once per process.

    Raises:
        ValueError: If ``FASTAPI_ENV`` is set to an unknown value. Selection
            fails closed rather than defaulting to the most permissive profile.
    """
    settings: Settings = _ENV_SETTINGS_MAP[resolve_environment()]()
    settings.create_directories()
    return settings


def get_app_settings() -> Settings:
    """Convenience alias for ``get_settings()``."""
    return get_settings()
