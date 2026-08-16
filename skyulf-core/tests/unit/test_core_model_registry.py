"""Tests for skyulf.core.model_registry (InMemoryModelRegistry with versioning)."""

import typing

import pytest

from skyulf.core.model_registry import InMemoryModelRegistry, ModelRegistry, ModelVersion


def test_register_creates_version_one_for_new_name():
    """The first registration under a name should be assigned version 1."""
    registry = InMemoryModelRegistry()
    entry = registry.register("my_model", object())
    assert entry.version == 1
    assert entry.name == "my_model"


def test_register_increments_version_for_same_name():
    """Repeated registrations under the same name should auto-increment the version."""
    registry = InMemoryModelRegistry()
    registry.register("my_model", object())
    second = registry.register("my_model", object())
    assert second.version == 2


def test_register_stores_metadata():
    """Metadata passed to register() should be stored on the ModelVersion."""
    registry = InMemoryModelRegistry()
    entry = registry.register("my_model", object(), metadata={"accuracy": 0.9})
    assert entry.metadata == {"accuracy": 0.9}


def test_register_is_thread_safe_no_duplicate_or_skipped_versions():
    """Concurrent register() calls for the same name must each get a unique,
    contiguous version number — regression guard against the
    read-len-then-append race in the old unguarded implementation."""
    import threading

    registry = InMemoryModelRegistry()
    n_threads = 20
    barrier = threading.Barrier(n_threads)

    def _register():
        barrier.wait()  # maximize the chance of interleaving
        registry.register("concurrent_model", object())

    threads = [threading.Thread(target=_register) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    versions = sorted(v.version for v in registry.versions("concurrent_model"))
    assert versions == list(range(1, n_threads + 1))


def test_register_defaults_metadata_to_empty_dict():
    """Omitting metadata should default to an empty dict, not None."""
    registry = InMemoryModelRegistry()
    entry = registry.register("my_model", object())
    assert entry.metadata == {}


def test_get_latest_version_when_version_not_specified():
    """get(name) without a version should return the most recently registered entry."""
    registry = InMemoryModelRegistry()
    registry.register("my_model", "model_v1")
    registry.register("my_model", "model_v2")
    latest = registry.get("my_model")
    assert latest.model == "model_v2"
    assert latest.version == 2


def test_get_specific_version():
    """get(name, version) should return the exact matching version."""
    registry = InMemoryModelRegistry()
    registry.register("my_model", "model_v1")
    registry.register("my_model", "model_v2")
    entry = registry.get("my_model", version=1)
    assert entry.model == "model_v1"


def test_get_unregistered_name_raises_key_error():
    """Requesting a name that was never registered should raise KeyError."""
    registry = InMemoryModelRegistry()
    with pytest.raises(KeyError, match="No model registered"):
        registry.get("unknown")


def test_get_unknown_version_raises_key_error():
    """Requesting a version number that doesn't exist should raise KeyError."""
    registry = InMemoryModelRegistry()
    registry.register("my_model", "model_v1")
    with pytest.raises(KeyError, match="Version 99"):
        registry.get("my_model", version=99)


def test_versions_returns_all_registered_entries_oldest_first():
    """versions() should return every registered version in registration order."""
    registry = InMemoryModelRegistry()
    registry.register("my_model", "v1")
    registry.register("my_model", "v2")
    versions = registry.versions("my_model")
    assert [v.model for v in versions] == ["v1", "v2"]


def test_versions_returns_empty_list_for_unknown_name():
    """versions() for a name never registered should return an empty list, not raise."""
    registry = InMemoryModelRegistry()
    assert registry.versions("unknown") == []


def test_model_version_is_a_plain_dataclass():
    """ModelVersion should be constructible directly with the documented fields."""
    version = ModelVersion(name="n", version=1, model="m")
    assert version.metadata == {}


def test_model_registry_abstract_methods_raise_not_implemented():
    """ModelRegistry's abstract register/get/versions must raise if bypassed."""
    with pytest.raises(NotImplementedError):
        ModelRegistry.register(typing.cast(ModelRegistry, object()), "name", object())
    with pytest.raises(NotImplementedError):
        ModelRegistry.get(typing.cast(ModelRegistry, object()), "name")
    with pytest.raises(NotImplementedError):
        ModelRegistry.versions(typing.cast(ModelRegistry, object()), "name")
