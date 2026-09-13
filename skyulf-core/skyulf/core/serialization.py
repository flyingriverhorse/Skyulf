"""Explicit model serialization utilities with a context-local provider.

Call :func:`get_model_serializer` or instantiate a serializer to use this API.
Changing its provider affects only callers that resolve that provider;
``SkyulfPipeline.save/load`` use pickle directly, and the backend artifact
stores use joblib directly. Neither automatically consumes this selection.

The provider is held in a :class:`contextvars.ContextVar`. New asyncio tasks
inherit the current selection; independent threads start with the joblib
default unless a context is explicitly copied. Use :func:`model_serializer`
to restore the previous selection when leaving an override block.
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
    """Joblib implementation returned by the provider unless explicitly overridden."""

    format = "joblib"

    def dump(self, model: Any, path: PathLike) -> None:
        """Persist ``model`` to ``path`` with ``joblib.dump``."""
        joblib.dump(model, path)

    def load(self, path: PathLike) -> Any:
        """Load and return the model written to ``path`` via ``joblib.load``."""
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
    for the lifetime of the current context. Only explicit callers of
    :func:`get_model_serializer` observe this selection; pipeline persistence
    and backend artifact stores do not use it.
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
