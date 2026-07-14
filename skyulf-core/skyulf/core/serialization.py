"""Model serialization seam.

Additive, non-breaking seam ahead of the Databricks/MLflow phases. The default
:class:`JoblibModelSerializer` preserves today's joblib behaviour; an MLflow or
cloud-object serializer can later implement the same interface without changing
call sites.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

__all__ = [
    "ModelSerializer",
    "JoblibModelSerializer",
    "get_model_serializer",
    "set_model_serializer",
]

PathLike = str | Path


class ModelSerializer(ABC):
    """Abstract model (de)serializer."""

    format: str = "abstract"

    @abstractmethod
    def dump(self, model: Any, path: PathLike) -> None:
        """Persist ``model`` to ``path``."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: PathLike) -> Any:
        """Load and return a model previously written to ``path``."""
        raise NotImplementedError


class JoblibModelSerializer(ModelSerializer):
    """Default joblib-backed serializer (matches current backend behaviour)."""

    format = "joblib"

    def dump(self, model: Any, path: PathLike) -> None:
        import joblib

        joblib.dump(model, path)

    def load(self, path: PathLike) -> Any:
        import joblib

        return joblib.load(path)


_DEFAULT_SERIALIZER: ModelSerializer = JoblibModelSerializer()


def get_model_serializer() -> ModelSerializer:
    """Return the active model serializer (joblib by default)."""
    return _DEFAULT_SERIALIZER


def set_model_serializer(serializer: ModelSerializer) -> None:
    """Install a model serializer process-wide."""
    global _DEFAULT_SERIALIZER
    _DEFAULT_SERIALIZER = serializer
