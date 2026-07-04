"""Tests for skyulf.core.compute (ComputeBackend seam)."""

import typing

import pytest

from skyulf.core.compute import (
    ComputeBackend,
    LocalComputeBackend,
    get_compute_backend,
    set_compute_backend,
)


def test_local_backend_executes_function_directly():
    """LocalComputeBackend.execute should call func(*args, **kwargs) and return its result."""
    backend = LocalComputeBackend()
    result = backend.execute(lambda a, b: a + b, 2, 3)
    assert result == 5


def test_local_backend_passes_kwargs():
    """LocalComputeBackend.execute should forward keyword arguments correctly."""
    backend = LocalComputeBackend()
    result = backend.execute(lambda a, b=0: a - b, 10, b=4)
    assert result == 6


def test_get_compute_backend_defaults_to_local():
    """The default compute backend should be a LocalComputeBackend instance named 'local'."""
    backend = get_compute_backend()
    assert isinstance(backend, LocalComputeBackend)
    assert backend.name == "local"


def test_set_compute_backend_installs_new_backend():
    """set_compute_backend should replace the process-wide active backend."""
    original = get_compute_backend()
    try:
        custom = LocalComputeBackend()
        set_compute_backend(custom)
        assert get_compute_backend() is custom
    finally:
        set_compute_backend(original)


def test_compute_backend_execute_is_abstract():
    """ComputeBackend.execute is abstract and must raise NotImplementedError if bypassed."""
    with pytest.raises(NotImplementedError):
        ComputeBackend.execute(typing.cast(ComputeBackend, object()), lambda: None)
