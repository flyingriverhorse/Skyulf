"""Regression tests for two Polars/Pandas-conversion bugs found in this audit round.

- ``backend/ml_pipeline/_internal/_advisor.py::DataProfiler.generate_profile``:
  a fully-null pandas nullable-dtype column (e.g. ``Int64``) makes ``.min()``/
  ``.max()``/``.mean()``/``.std()``/``.skew()`` return the ``pd.NA`` sentinel,
  and ``float(pd.NA)`` raises ``TypeError`` — aborting the whole profile
  instead of degrading gracefully for just that column.
- ``backend/data_ingestion/engine/profiler.py::DataProfiler.profile``: the
  numeric-dtype allow-list was missing several integer dtype variants
  (``Int8``/``Int16``/``UInt8``/``UInt16``/``UInt32``/``UInt64``), so e.g. a
  ``pl.UInt8`` column silently fell through both the numeric and string
  branches and got zero statistics.
"""

import pandas as pd
import polars as pl

from backend.data_ingestion.engine.profiler import DataProfiler as IngestionProfiler
from backend.ml_pipeline._internal._advisor import DataProfiler as AdvisorProfiler


def test_advisor_generate_profile_all_null_nullable_int_column_does_not_crash():
    """An all-null pandas ``Int64`` column must degrade to ``None`` stats, not raise."""
    df = pd.DataFrame({"a": pd.array([None, None, None], dtype="Int64"), "b": [1, 2, 3]})
    profile = AdvisorProfiler.generate_profile(df)

    col = profile.columns["a"]
    assert col["column_type"] == "numeric"
    assert col["missing_count"] == 3
    assert col["min_value"] is None
    assert col["max_value"] is None
    assert col["mean_value"] is None
    assert col["std_value"] is None
    assert col["skewness"] is None

    # Sibling non-null column is unaffected.
    assert profile.columns["b"]["mean_value"] == 2.0


def test_advisor_generate_profile_all_null_nullable_boolean_column_does_not_crash():
    """Same regression as above, for pandas ``boolean`` nullable dtype."""
    df = pd.DataFrame({"flag": pd.array([None, None], dtype="boolean")})
    profile = AdvisorProfiler.generate_profile(df)
    col = profile.columns["flag"]
    assert col["min_value"] is None
    assert col["max_value"] is None


def test_advisor_generate_profile_partially_null_nullable_int_column_computes_stats():
    """A partially-null nullable-int column should still compute real stats
    for its non-null values (this must keep working after the fix)."""
    df = pd.DataFrame({"a": pd.array([1, 2, None], dtype="Int64")})
    profile = AdvisorProfiler.generate_profile(df)
    col = profile.columns["a"]
    assert col["min_value"] == 1.0
    assert col["max_value"] == 2.0
    assert col["mean_value"] == 1.5


def test_ingestion_profiler_uint8_column_computes_statistics():
    """A ``pl.UInt8`` column was previously missing from the numeric
    dtype allow-list, silently producing zero statistics."""
    df = pl.DataFrame({"age": [1, 2, 3]}, schema={"age": pl.UInt8})
    result = IngestionProfiler.profile(df)
    stats = result["columns"]["age"]
    assert stats["mean"] == 2.0
    assert stats["min"] == 1
    assert stats["max"] == 3
    assert stats["median"] == 2.0


def test_ingestion_profiler_all_integer_dtype_variants_compute_statistics():
    """Every integer dtype variant in the allow-list must produce real stats,
    not just the originally-covered Int32/Int64/Float32/Float64."""
    for dtype in [pl.Int8, pl.Int16, pl.UInt16, pl.UInt32, pl.UInt64]:
        df = pl.DataFrame({"v": [1, 2, 3]}, schema={"v": dtype})
        stats = IngestionProfiler.profile(df)["columns"]["v"]
        assert stats["mean"] == 2.0, f"failed for dtype {dtype}"
