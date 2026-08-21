"""Tests for skyulf.core.serialization (ModelSerializer seam / JoblibModelSerializer)."""

import typing

import pytest
from sklearn.linear_model import LinearRegression

from skyulf.core.serialization import (
    JoblibModelSerializer,
    ModelSerializer,
    get_model_serializer,
    set_model_serializer,
)


def test_joblib_serializer_dump_and_load_round_trip(tmp_path):
    """A model dumped via joblib should load back with identical predictions."""
    model = LinearRegression().fit([[1], [2], [3]], [2, 4, 6])
    path = tmp_path / "model.joblib"
    serializer = JoblibModelSerializer()

    serializer.dump(model, path)
    loaded = serializer.load(path)

    assert loaded.predict([[4]]) == pytest.approx(model.predict([[4]]))


def test_joblib_serializer_format_attribute():
    """JoblibModelSerializer should expose 'joblib' as its format label."""
    assert JoblibModelSerializer().format == "joblib"


def test_get_model_serializer_defaults_to_joblib():
    """The default active serializer should be a JoblibModelSerializer."""
    assert isinstance(get_model_serializer(), JoblibModelSerializer)


def test_set_model_serializer_installs_custom_serializer():
    """set_model_serializer should replace the process-wide active serializer."""
    original = get_model_serializer()

    class _NoOpSerializer(ModelSerializer):
        format = "noop"

        def dump(self, model, path):
            pass

        def load(self, path):
            return None

    try:
        custom = _NoOpSerializer()
        set_model_serializer(custom)
        assert get_model_serializer() is custom
    finally:
        set_model_serializer(original)


def test_model_serializer_dump_and_load_are_abstract():
    """ModelSerializer.dump/.load are abstract and must raise if bypassed."""
    with pytest.raises(NotImplementedError):
        ModelSerializer.dump(typing.cast(ModelSerializer, object()), None, "path")
    with pytest.raises(NotImplementedError):
        ModelSerializer.load(typing.cast(ModelSerializer, object()), "path")
