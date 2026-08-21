"""Tests for the pre-Databricks core seams: compute, serialization, registry."""

import pytest

from skyulf.core import (
    ComputeBackend,
    InMemoryModelRegistry,
    JoblibModelSerializer,
    LocalComputeBackend,
    ModelVersion,
    get_compute_backend,
    get_model_serializer,
    set_compute_backend,
    set_model_serializer,
)


def test_local_compute_backend_executes_directly():
    backend = LocalComputeBackend()
    assert backend.name == "local"
    assert backend.execute(lambda a, b: a + b, 2, 3) == 5


def test_default_compute_backend_is_local():
    assert isinstance(get_compute_backend(), LocalComputeBackend)


def test_set_compute_backend_swaps_and_restores():
    class Doubling(ComputeBackend):
        name = "doubling"

        def execute(self, func, *args, **kwargs):
            return func(*args, **kwargs) * 2

    original = get_compute_backend()
    try:
        set_compute_backend(Doubling())
        assert get_compute_backend().execute(lambda: 4) == 8
    finally:
        set_compute_backend(original)
    assert isinstance(get_compute_backend(), LocalComputeBackend)


def test_joblib_serializer_round_trip(tmp_path):
    serializer = JoblibModelSerializer()
    assert serializer.format == "joblib"
    path = tmp_path / "model.joblib"
    serializer.dump({"weights": [1, 2, 3]}, path)
    assert serializer.load(path) == {"weights": [1, 2, 3]}


def test_default_serializer_is_joblib():
    assert isinstance(get_model_serializer(), JoblibModelSerializer)
    set_model_serializer(JoblibModelSerializer())
    assert isinstance(get_model_serializer(), JoblibModelSerializer)


def test_in_memory_registry_versions_auto_increment():
    registry = InMemoryModelRegistry()
    v1 = registry.register("clf", object(), {"acc": 0.9})
    v2 = registry.register("clf", object())
    assert (v1.version, v2.version) == (1, 2)
    assert isinstance(registry.get("clf"), ModelVersion)
    assert registry.get("clf").version == 2
    assert registry.get("clf", version=1).metadata == {"acc": 0.9}
    assert len(registry.versions("clf")) == 2


def test_in_memory_registry_missing_raises():
    registry = InMemoryModelRegistry()
    with pytest.raises(KeyError):
        registry.get("missing")
    registry.register("clf", object())
    with pytest.raises(KeyError):
        registry.get("clf", version=99)
    assert registry.versions("nope") == []
