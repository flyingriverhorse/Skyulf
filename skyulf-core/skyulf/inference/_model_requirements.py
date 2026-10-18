"""Discover optional native distributions retained in fitted inference objects."""

import inspect
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import Any

import numpy as np

_NATIVE_PACKAGES = frozenset({"xgboost", "lightgbm"})


def native_model_requirements(*roots: Any) -> tuple[str, ...]:
    """Pin retained native model dependencies without importing optional libraries.

    Class ancestry covers Core subclasses and wrappers containing fitted native
    estimators. Walk saved state, not configuration names or installed packages;
    training-only estimators that were discarded need no serving dependency.
    """
    packages: set[str] = set()
    pending = list(roots)
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        cls = value if isinstance(value, type) else type(value)
        packages.update(
            base.__module__.split(".")[0]
            for base in cls.__mro__
            if base.__module__.split(".")[0] in _NATIVE_PACKAGES
        )
        pending.extend(_state_children(value))
    return tuple(_installed_pin(package) for package in sorted(packages))


def _state_children(value: Any) -> Iterable[Any]:
    """Visit retained containers and instance state without walking imported code."""
    if isinstance(value, (type, ModuleType)) or inspect.isroutine(value):
        return ()
    if isinstance(value, dict):
        return (*value.keys(), *value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return value
    if isinstance(value, np.ndarray):
        return value.flat if value.dtype == object else ()
    return vars(value).values() if hasattr(value, "__dict__") else ()


def _installed_pin(package: str) -> str:
    """Report missing distribution metadata explicitly before creating an artifact."""
    try:
        return f"{package}=={version(package)}"
    except PackageNotFoundError as exc:
        raise ValueError(f"Fitted model dependency {package} is not installed.") from exc
