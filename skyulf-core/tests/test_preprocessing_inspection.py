"""Tests for skyulf.preprocessing.inspection (DatasetProfile, DataSnapshot nodes)."""

import typing

import pandas as pd
import polars as pl
import pytest
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.inspection import (
    DatasetProfileApplier,
    DatasetProfileCalculator,
    DataSnapshotApplier,
    DataSnapshotCalculator,
    _extract_polars_numeric_stats,
)

_engine_cases = TestCaseLoader("preprocessing/inspection_engine").load()


@pytest.fixture
def pandas_df():
    """A small deterministic pandas DataFrame with numeric and missing values."""
    return pd.DataFrame({"num": [1.0, 2.0, None, 4.0], "cat": ["a", "b", "c", "d"]})


@pytest.fixture
def polars_df():
    """A small deterministic polars DataFrame with numeric and null values."""
    return pl.DataFrame({"num": [1.0, 2.0, None, 4.0], "cat": ["a", "b", "c", "d"]})


# ---------------------------------------------------------------------------
# DatasetProfile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_engine_cases)
def test_dataset_profile_fit_reports_shape(engine, pandas_df, polars_df) -> None:
    """Fitting on either engine's frame should report correct row/column counts."""
    df = pandas_df if engine == "pandas" else polars_df
    artifact = DatasetProfileCalculator().fit(df, {})
    profile = artifact["profile"]
    assert profile["rows"] == 4
    assert profile["columns"] == 2


@pytest.mark.parametrize(*_engine_cases)
def test_dataset_profile_fit_reports_missing_counts(engine, pandas_df, polars_df) -> None:
    """Missing-value counts per column should be captured for either engine's path."""
    df = pandas_df if engine == "pandas" else polars_df
    artifact = DatasetProfileCalculator().fit(df, {})
    assert artifact["profile"]["missing"]["num"] == 1
    assert artifact["profile"]["missing"]["cat"] == 0


@pytest.mark.parametrize(*_engine_cases)
def test_dataset_profile_fit_numeric_stats_present(engine, pandas_df, polars_df) -> None:
    """describe()-based numeric stats should be attached for numeric columns, either engine."""
    df = pandas_df if engine == "pandas" else polars_df
    artifact = DatasetProfileCalculator().fit(df, {})
    assert "numeric_stats" in artifact["profile"]
    assert "num" in artifact["profile"]["numeric_stats"]


def test_extract_polars_numeric_stats_empty_columns_returns_empty_dict(polars_df):
    """No numeric columns selected should short-circuit to an empty stats dict."""
    assert _extract_polars_numeric_stats(polars_df, []) == {}


def test_extract_polars_numeric_stats_contains_mean(polars_df):
    """The extracted stats dict should include a 'mean' (or 'null_count' etc) metric key."""
    stats = _extract_polars_numeric_stats(polars_df, ["num"])
    assert "num" in stats
    assert len(stats["num"]) > 0


def test_extract_polars_numeric_stats_skips_row_without_metric_key():
    """A describe() row lacking both 'describe' and 'statistic' keys is skipped (line 36)."""

    class _FakeDescribeDf:
        def to_dicts(self):
            # Simulates a malformed/unexpected describe() row shape.
            return [{"num": 1.0}, {"describe": "mean", "num": 2.5}]

    class _FakeSelected:
        def describe(self):
            return _FakeDescribeDf()

    class _FakeX:
        def select(self, cols):
            return _FakeSelected()

    stats = _extract_polars_numeric_stats(_FakeX(), ["num"])
    assert stats == {"num": {"mean": 2.5}}


def test_dataset_profile_applier_is_passthrough(pandas_df):
    """DatasetProfileApplier.apply should return the input frame unchanged."""
    result = DatasetProfileApplier().apply(pandas_df, {})
    assert result is pandas_df


def test_dataset_profile_infer_output_schema_is_identity():
    """infer_output_schema should return the input schema unchanged (read-only node)."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["a", "b"])
    result = DatasetProfileCalculator().infer_output_schema(schema, {})
    assert result is schema


# ---------------------------------------------------------------------------
# DataSnapshot
# ---------------------------------------------------------------------------


def test_data_snapshot_fit_pandas_default_n_rows(pandas_df):
    """Default n_rows (5) should cap the returned snapshot at the frame length."""
    artifact = DataSnapshotCalculator().fit(pandas_df, {})
    assert len(artifact["snapshot"]) == 4  # frame only has 4 rows
    assert artifact["type"] == "data_snapshot"


@pytest.mark.parametrize(*_engine_cases)
def test_data_snapshot_fit_respects_n_rows(engine, pandas_df, polars_df) -> None:
    """A configured n_rows should limit the number of snapshot records returned, either engine."""
    df = pandas_df if engine == "pandas" else polars_df
    artifact = DataSnapshotCalculator().fit(df, {"n_rows": 2})
    snapshot = typing.cast(list, artifact["snapshot"])
    assert len(snapshot) == 2
    assert snapshot[0]["cat"] == "a"


def test_data_snapshot_applier_is_passthrough(pandas_df):
    """DataSnapshotApplier.apply should return the input frame unchanged."""
    result = DataSnapshotApplier().apply(pandas_df, {})
    assert result is pandas_df
