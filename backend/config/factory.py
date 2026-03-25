"""
Settings Factory

© 2025 Murat Unsal — Skyulf Project

Creates the correct Settings subclass based on the FASTAPI_ENV environment variable.
"""

import os
from functools import lru_cache

from backend.config.base import Settings
from backend.config.environments import (
    DevelopmentSettings,
    ProductionSettings,
    TestingSettings,
)


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings based on ``FASTAPI_ENV``.

    Returns a cached singleton so the config is built once per process.
    """
    env = os.getenv("FASTAPI_ENV", "development").lower()

    env_map = {
        "production": ProductionSettings,
        "testing": TestingSettings,
    }
    settings: Settings = env_map.get(env, DevelopmentSettings)()
    settings.create_directories()
    return settings


def get_app_settings() -> Settings:
    """Convenience alias for ``get_settings()``."""
    return get_settings()
