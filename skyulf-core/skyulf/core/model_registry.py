"""Explicit, in-process model registration with versioning.

Instantiate :class:`InMemoryModelRegistry` and register models directly to use
this API. Each instance keeps object references keyed by ``(name, version)``;
pipeline training does not register models here automatically. This registry
has no connection to the backend's persisted job and model registry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

__all__ = [
    "InMemoryModelRegistry",
    "ModelRegistry",
    "ModelVersion",
]


@dataclass
class ModelVersion:
    """An in-memory model reference with its local name, version and metadata.

    ``skyulf.core.model_registry.ModelVersion`` is this dataclass.
    ``backend.ml_pipeline.model_registry.schemas.ModelVersion`` is a separate
    Pydantic response schema describing backend jobs and model artifacts.
    """

    name: str
    version: int
    model: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry(ABC):
    """Abstract versioned model registry."""

    @abstractmethod
    def register(
        self, name: str, model: Any, metadata: dict[str, Any] | None = None
    ) -> ModelVersion:
        """Register ``model`` under ``name`` and return the new version."""
        raise NotImplementedError

    @abstractmethod
    def get(self, name: str, version: int | None = None) -> ModelVersion:
        """Return a version (latest when ``version`` is ``None``)."""
        raise NotImplementedError

    @abstractmethod
    def versions(self, name: str) -> list[ModelVersion]:
        """Return all versions registered under ``name`` (oldest first)."""
        raise NotImplementedError


class InMemoryModelRegistry(ModelRegistry):
    """Registry instantiated by callers; per-name versions auto-increment from 1."""

    def __init__(self) -> None:
        """Create an empty registry plus the lock that serialises concurrent version assignment."""
        self._store: dict[str, list[ModelVersion]] = {}
        # Guards read-modify-write of `_store[name]` so concurrent `register()`
        # calls for the same model name can't both read the same
        # `len(versions)` and assign the same next version number.
        self._lock = Lock()

    def register(
        self, name: str, model: Any, metadata: dict[str, Any] | None = None
    ) -> ModelVersion:
        """Register ``model`` under ``name``, assigning the next auto-incremented version."""
        with self._lock:
            versions = self._store.setdefault(name, [])
            entry = ModelVersion(name, len(versions) + 1, model, dict(metadata or {}))
            versions.append(entry)
        return entry

    def get(self, name: str, version: int | None = None) -> ModelVersion:
        """Return the requested version of ``name`` (latest when ``version`` is ``None``)."""
        versions = self._store.get(name)
        if not versions:
            raise KeyError(f"No model registered under '{name}'.")
        if version is None:
            return versions[-1]
        for entry in versions:
            if entry.version == version:
                return entry
        raise KeyError(f"Version {version} not found for model '{name}'.")

    def versions(self, name: str) -> list[ModelVersion]:
        """Return a copy of all versions under ``name`` (oldest first); empty if unregistered."""
        return list(self._store.get(name, []))
