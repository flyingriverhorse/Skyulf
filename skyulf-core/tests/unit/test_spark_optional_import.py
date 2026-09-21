"""Verify optional Spark boundaries without requiring a JVM or PySpark install."""

import subprocess
import sys

import pandas as pd
import pytest

from skyulf.engines import EngineRegistry, SparkEngine


def test_core_import_does_not_load_optional_runtimes():
    """Base installs must import and run local engines even if extras are unavailable."""
    code = """
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyspark', 'mlflow'}:
            raise ImportError('optional runtime deliberately unavailable')
sys.meta_path.insert(0, BlockExtras())
import skyulf
import pandas as pd
from skyulf.engines import EngineRegistry, SparkEngine, get_engine
assert get_engine().name == 'polars'
assert get_engine(pd.DataFrame({'x': [1]})).name == 'pandas'
assert not SparkEngine.is_compatible(pd.DataFrame())
assert not any(name.split('.')[0] in {'pyspark', 'mlflow'} for name in sys.modules)
for operation in (lambda: EngineRegistry.get('spark'),
                  lambda: EngineRegistry.set_active_engine('spark')):
    try:
        operation()
    except ImportError as exc:
        assert 'skyulf-core[spark]' in str(exc)
    else:
        raise AssertionError('missing optional dependency accepted')
assert get_engine().name == 'polars'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_spark_wrapper_rejects_local_frames():
    """Wrapping local data must not create an implicit Spark session."""
    with pytest.raises(TypeError, match="pyspark.sql.DataFrame"):
        SparkEngine.wrap(pd.DataFrame({"x": [1]}))


def test_namespace_lookalike_requires_real_dataframe(monkeypatch):
    """A matching module path must not authorize a non-dataframe Spark object."""
    monkeypatch.setattr(SparkEngine, "ensure_available", classmethod(lambda cls: None))
    monkeypatch.setattr(SparkEngine, "is_compatible", classmethod(lambda cls, data: False))
    lookalike = type("Lookalike", (), {"__module__": "pyspark.sql.dataframe"})()
    with pytest.raises(TypeError, match="pyspark.sql.DataFrame"):
        EngineRegistry.resolve(lookalike)
