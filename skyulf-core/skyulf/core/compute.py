"""Compute backend seam — execution-engine abstraction.

Additive, non-breaking seam introduced ahead of the Databricks integration.
Local execution keeps running through :class:`LocalComputeBackend`; a
distributed backend (e.g. Spark) can later implement the same interface
without touching node logic. Nothing in the library wires this in by default,
so behaviour is unchanged until a backend is explicitly installed.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

__all__ = [
    "ComputeBackend",
    "LocalComputeBackend",
    "get_compute_backend",
    "set_compute_backend",
]


class ComputeBackend(ABC):
    """Abstract execution engine: runs a unit of work and returns its result."""

    name: str = "abstract"

    @abstractmethod
    def execute[R](self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """Run ``func(*args, **kwargs)`` on this backend and return the result."""
        raise NotImplementedError


class LocalComputeBackend(ComputeBackend):
    """Default in-process backend. Calls the function directly."""

    name = "local"

    def execute[R](self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        return func(*args, **kwargs)


_DEFAULT_BACKEND: ComputeBackend = LocalComputeBackend()


def get_compute_backend() -> ComputeBackend:
    """Return the active compute backend (local by default)."""
    return _DEFAULT_BACKEND


def set_compute_backend(backend: ComputeBackend) -> None:
    """Install a compute backend process-wide. Pass a fresh instance to swap engines."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = backend
