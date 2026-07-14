"""Model registry seam with versioning.

Additive, non-breaking seam ahead of the Databricks/MLflow phases. The default
:class:`InMemoryModelRegistry` keeps fitted models in-process keyed by
``(name, version)``. An artifact-store backed registry (MLflow / Unity Catalog)
can later subclass :class:`ModelRegistry` without changing call sites.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

__all__ = [
    "ModelVersion",
    "ModelRegistry",
    "InMemoryModelRegistry",
]


@dataclass
class ModelVersion:
    """A single registered model version."""

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
    """Default in-process registry. Versions auto-increment from 1."""

    def __init__(self) -> None:
        self._store: dict[str, list[ModelVersion]] = {}
        # Guards read-modify-write of `_store[name]` so concurrent `register()`
        # calls for the same model name can't both read the same
        # `len(versions)` and assign the same next version number.
        self._lock = Lock()

    def register(
        self, name: str, model: Any, metadata: dict[str, Any] | None = None
    ) -> ModelVersion:
        with self._lock:
            versions = self._store.setdefault(name, [])
            entry = ModelVersion(name, len(versions) + 1, model, dict(metadata or {}))
            versions.append(entry)
        return entry

    def get(self, name: str, version: int | None = None) -> ModelVersion:
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
        return list(self._store.get(name, []))
