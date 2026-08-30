"""Model serialization seam.

Additive, non-breaking seam ahead of the Databricks/MLflow phases. The default
:class:`JoblibModelSerializer` preserves today's joblib behaviour; an MLflow or
cloud-object serializer can later implement the same interface without changing
call sites.

The active serializer is held in a :class:`contextvars.ContextVar`: a
change is visible to the current thread/asyncio task and anything spawned
from it afterwards, but concurrent pipelines in other contexts cannot
reconfigure each other mid-run. Use :func:`model_serializer` to scope an
override to a single block.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import joblib

__all__ = [
    "JoblibModelSerializer",
    "ModelSerializer",
    "get_model_serializer",
    "model_serializer",
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
        joblib.dump(model, path)

    def load(self, path: PathLike) -> Any:
        return joblib.load(path)


_DEFAULT_SERIALIZER: ContextVar[ModelSerializer] = ContextVar(
    "skyulf_default_model_serializer",
    default=JoblibModelSerializer(),  # noqa: B039 - stateless shared singleton, never mutated
)


def get_model_serializer() -> ModelSerializer:
    """Return the active model serializer for the current context (joblib by default)."""
    return _DEFAULT_SERIALIZER.get()


def set_model_serializer(serializer: ModelSerializer) -> None:
    """Install a model serializer for the current context.

    Prefer the :func:`model_serializer` context manager for overrides that
    should end with the enclosing block; this setter keeps the serializer
    for the lifetime of the current context.
    """
    _DEFAULT_SERIALIZER.set(serializer)


@contextmanager
def model_serializer(serializer: ModelSerializer) -> Iterator[ModelSerializer]:
    """Scope ``serializer`` to the enclosed block, restoring the prior selection on exit."""
    token = _DEFAULT_SERIALIZER.set(serializer)
    try:
        yield serializer
    finally:
        _DEFAULT_SERIALIZER.reset(token)
