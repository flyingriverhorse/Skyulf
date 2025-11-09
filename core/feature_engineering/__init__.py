"""Feature engineering module exports with lazy loading to avoid heavy side effects."""

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = ["routes"]


def __getattr__(name: str) -> Any:
    if name == "routes":
        module: ModuleType = import_module("core.feature_engineering.routes")
        return module
    raise AttributeError(f"module 'core.feature_engineering' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
