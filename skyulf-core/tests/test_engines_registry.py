"""Tests for skyulf.engines.registry (EngineRegistry auto-detection dispatch)."""

import logging

import pandas as pd
import polars as pl
import pytest

from skyulf.engines.pandas_engine import PandasEngine, SkyulfPandasWrapper
from skyulf.engines.polars_engine import PolarsEngine, SkyulfPolarsWrapper
from skyulf.engines.registry import BaseEngine, EngineName, EngineRegistry, get_engine


def test_engine_name_enum_values():
    """EngineName should expose the three expected string values."""
    assert EngineName.PANDAS == "pandas"
    assert EngineName.POLARS == "polars"
    assert EngineName.BASE == "base"


def test_engine_name_str_returns_value():
    """EngineName is a StrEnum: str()/f-string must yield the bare value,
    not `ClassName.MEMBER` (regression guard against reverting to `(str, Enum)`)."""
    assert str(EngineName.PANDAS) == "pandas"
    assert f"{EngineName.POLARS}" == "polars"


def test_base_engine_is_compatible_raises_not_implemented():
    """BaseEngine.is_compatible is abstract and must raise if not overridden."""
    with pytest.raises(NotImplementedError):
        BaseEngine.is_compatible(None)


def test_base_engine_from_pandas_raises_not_implemented():
    """BaseEngine.from_pandas is abstract and must raise if not overridden."""
    with pytest.raises(NotImplementedError):
        BaseEngine.from_pandas(None)


def test_base_engine_to_numpy_raises_not_implemented():
    """BaseEngine.to_numpy is abstract and must raise if not overridden."""
    with pytest.raises(NotImplementedError):
        BaseEngine.to_numpy(None)


def test_base_engine_wrap_raises_not_implemented():
    """BaseEngine.wrap is abstract and must raise if not overridden."""
    with pytest.raises(NotImplementedError):
        BaseEngine.wrap(None)


def test_base_engine_create_dataframe_raises_not_implemented():
    """BaseEngine.create_dataframe is abstract and must raise if not overridden."""
    with pytest.raises(NotImplementedError):
        BaseEngine.create_dataframe(None)


def test_registry_get_unknown_engine_raises_value_error():
    """Requesting an unregistered engine name should raise ValueError with available names."""
    with pytest.raises(ValueError, match="not found"):
        EngineRegistry.get("nonexistent_engine")


def test_registry_register_and_get_round_trip():
    """A custom engine registered via .register() should be retrievable via .get()."""

    class DummyEngine(BaseEngine):
        name = EngineName.BASE

    EngineRegistry.register("dummy_for_test", DummyEngine)
    assert EngineRegistry.get("dummy_for_test") is DummyEngine


def test_resolve_with_none_returns_active_default_engine():
    """resolve(None) should return the currently configured default engine."""
    resolved = EngineRegistry.resolve(None)
    assert resolved is EngineRegistry.get(EngineRegistry._active_engine)


def test_set_active_engine_changes_default(monkeypatch):
    """set_active_engine() should change which engine resolve(None) returns
    (regression guard for r5 EngineRegistry missing setter finding)."""
    original = EngineRegistry._active_engine
    try:
        EngineRegistry.set_active_engine("polars")
        assert EngineRegistry._active_engine == "polars"
        assert EngineRegistry.resolve(None) is PolarsEngine
    finally:
        EngineRegistry.set_active_engine(original)


def test_set_active_engine_raises_for_unknown_engine():
    """set_active_engine() should validate the name against registered engines."""
    with pytest.raises(ValueError, match="not found"):
        EngineRegistry.set_active_engine("nonexistent_engine")


def test_resolve_detects_pandas_dataframe():
    """resolve() should identify a pandas.DataFrame's module and return PandasEngine."""
    df = pd.DataFrame({"a": [1, 2]})
    assert EngineRegistry.resolve(df) is PandasEngine


def test_resolve_detects_polars_dataframe():
    """resolve() should identify a polars.DataFrame's module and return PolarsEngine."""
    df = pl.DataFrame({"a": [1, 2]})
    assert EngineRegistry.resolve(df) is PolarsEngine


def test_resolve_unknown_type_falls_back_to_default_and_warns(caplog):
    """An unrecognised data type should fall back to the default engine with a warning."""

    class _Unknown:
        pass

    with caplog.at_level(logging.WARNING, logger="skyulf.engines.registry"):
        resolved = EngineRegistry.resolve(_Unknown())
    assert resolved is EngineRegistry.get(EngineRegistry._active_engine)
    assert any("Unknown data type" in record.message for record in caplog.records)


def test_resolve_plain_list_falls_back_without_warning(caplog):
    """Regression test: a plain Python list (e.g. a raw y target) is a common,
    expected input shape, not a genuinely unknown type - resolve() must fall
    back to the default engine silently, without emitting the 'Unknown data
    type' warning."""
    with caplog.at_level(logging.WARNING, logger="skyulf.engines.registry"):
        resolved = EngineRegistry.resolve([1, 2, 3])
    assert resolved is EngineRegistry.get(EngineRegistry._active_engine)
    assert not any("Unknown data type" in record.message for record in caplog.records)


def test_resolve_plain_tuple_falls_back_without_warning(caplog):
    """Same as above but for a plain tuple."""
    with caplog.at_level(logging.WARNING, logger="skyulf.engines.registry"):
        resolved = EngineRegistry.resolve((1, 2, 3))
    assert resolved is EngineRegistry.get(EngineRegistry._active_engine)
    assert not any("Unknown data type" in record.message for record in caplog.records)


def test_resolve_uses_top_level_package_not_substring_match():
    """A module whose name merely *contains* 'pandas'/'polars' as a substring
    (e.g. a third-party 'fake_polars_stub' or 'my_pandas_wrapper' module) must
    NOT be misdetected as the real pandas/polars engine — regression guard
    against the old `"pandas" in module` substring check."""

    class _FakePolarsLookalike:
        pass

    _FakePolarsLookalike.__module__ = "fake_polars_stub.frame"
    resolved = EngineRegistry.resolve(_FakePolarsLookalike())
    assert resolved is EngineRegistry.get(EngineRegistry._active_engine)

    class _FakePandasLookalike:
        pass

    _FakePandasLookalike.__module__ = "my_pandas_wrapper.core"
    resolved = EngineRegistry.resolve(_FakePandasLookalike())
    assert resolved is EngineRegistry.get(EngineRegistry._active_engine)


def test_resolve_matches_submodules_of_real_pandas_polars():
    """A dotted submodule of the real library (e.g. `pandas.core.frame`) must
    still match via the top-level package check."""

    class _RealPandasSubmoduleType:
        pass

    _RealPandasSubmoduleType.__module__ = "pandas.core.frame"
    assert EngineRegistry.resolve(_RealPandasSubmoduleType()) is EngineRegistry.get("pandas")

    class _RealPolarsSubmoduleType:
        pass

    _RealPolarsSubmoduleType.__module__ = "polars.dataframe.frame"
    assert EngineRegistry.resolve(_RealPolarsSubmoduleType()) is EngineRegistry.get("polars")


def test_wrap_dispatches_to_correct_engine_for_pandas():
    """EngineRegistry.wrap should auto-detect pandas input and produce a pandas wrapper."""
    df = pd.DataFrame({"a": [1, 2]})
    wrapped = EngineRegistry.wrap(df)
    assert isinstance(wrapped, SkyulfPandasWrapper)


def test_wrap_dispatches_to_correct_engine_for_polars():
    """EngineRegistry.wrap should auto-detect polars input and produce a polars wrapper."""
    df = pl.DataFrame({"a": [1, 2]})
    wrapped = EngineRegistry.wrap(df)
    assert isinstance(wrapped, SkyulfPolarsWrapper)


def test_get_engine_helper_matches_resolve():
    """The module-level get_engine() helper should behave identically to EngineRegistry.resolve()."""
    df = pd.DataFrame({"a": [1]})
    assert get_engine(df) is EngineRegistry.resolve(df)


def test_resolve_pyspark_like_module_falls_back_without_spark_engine():
    """A pyspark-namespaced type with no registered 'spark' engine should fall back to default."""

    class _FakePysparkFrame:
        pass

    _FakePysparkFrame.__module__ = "pyspark.sql.dataframe"
    resolved = EngineRegistry.resolve(_FakePysparkFrame())
    assert resolved is EngineRegistry.get(EngineRegistry._active_engine)


def test_resolve_dask_like_module_falls_back_without_dask_engine():
    """A dask-namespaced type with no registered 'dask' engine should fall back to default."""

    class _FakeDaskFrame:
        pass

    _FakeDaskFrame.__module__ = "dask.dataframe.core"
    resolved = EngineRegistry.resolve(_FakeDaskFrame())
    assert resolved is EngineRegistry.get(EngineRegistry._active_engine)


def test_resolve_pyspark_like_module_uses_registered_spark_engine():
    """When a 'spark' engine is registered, a pyspark-namespaced type should resolve to it."""

    class DummySparkEngine(BaseEngine):
        name = EngineName.BASE

    EngineRegistry.register("spark", DummySparkEngine)
    try:

        class _FakePysparkFrame:
            pass

        _FakePysparkFrame.__module__ = "pyspark.sql.dataframe"
        resolved = EngineRegistry.resolve(_FakePysparkFrame())
        assert resolved is DummySparkEngine
    finally:
        del EngineRegistry._engines["spark"]


def test_resolve_dask_like_module_uses_registered_dask_engine():
    """When a 'dask' engine is registered, a dask-namespaced type should resolve to it."""

    class DummyDaskEngine(BaseEngine):
        name = EngineName.BASE

    EngineRegistry.register("dask", DummyDaskEngine)
    try:

        class _FakeDaskFrame:
            pass

        _FakeDaskFrame.__module__ = "dask.dataframe.core"
        resolved = EngineRegistry.resolve(_FakeDaskFrame())
        assert resolved is DummyDaskEngine
    finally:
        del EngineRegistry._engines["dask"]


def test_register_logs_debug_message(caplog):
    """register() should emit a debug log entry naming the registered engine."""

    class AnotherDummyEngine(BaseEngine):
        name = EngineName.BASE

    with caplog.at_level(logging.DEBUG, logger="skyulf.engines.registry"):
        EngineRegistry.register("another_dummy", AnotherDummyEngine)
    assert any("another_dummy" in record.message for record in caplog.records)
