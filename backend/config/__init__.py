"""
Configuration Package

© 2025 Murat Unsal — Skyulf Project

Re-exports the public API so existing imports continue to work:
    from backend.config import Settings, get_settings
"""

from backend.config.base import Settings  # noqa: F401
from backend.config.environments import (  # noqa: F401
    DevelopmentSettings,
    ProductionSettings,
    TestingSettings,
)
from backend.config.factory import get_app_settings, get_settings  # noqa: F401

__all__ = [
    "Settings",
    "DevelopmentSettings",
    "ProductionSettings",
    "TestingSettings",
    "get_settings",
    "get_app_settings",
]
