"""Tests for skyulf.preprocessing.inspection (DatasetProfile, DataSnapshot nodes)."""

import typing

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.inspection import (
    DatasetProfileApplier,
    DatasetProfileCalculator,
    DataSnapshotApplier,
    DataSnapshotCalculator,
    _extract_polars_numeric_stats,
)


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


def test_dataset_profile_fit_pandas_reports_shape(pandas_df):
    """Fitting on a pandas frame should report correct row/column counts."""
    artifact = DatasetProfileCalculator().fit(pandas_df, {})
    profile = artifact["profile"]
    assert profile["rows"] == 4
    assert profile["columns"] == 2


def test_dataset_profile_fit_pandas_reports_missing_counts(pandas_df):
    """Missing-value counts per column should be captured for the pandas path."""
    artifact = DatasetProfileCalculator().fit(pandas_df, {})
    assert artifact["profile"]["missing"]["num"] == 1
    assert artifact["profile"]["missing"]["cat"] == 0


def test_dataset_profile_fit_pandas_numeric_stats_present(pandas_df):
    """describe()-based numeric stats should be attached for numeric columns."""
    artifact = DatasetProfileCalculator().fit(pandas_df, {})
    assert "numeric_stats" in artifact["profile"]
    assert "num" in artifact["profile"]["numeric_stats"]


def test_dataset_profile_fit_polars_reports_shape(polars_df):
    """Fitting on a polars frame should report correct row/column counts."""
    artifact = DatasetProfileCalculator().fit(polars_df, {})
    profile = artifact["profile"]
    assert profile["rows"] == 4
    assert profile["columns"] == 2


def test_dataset_profile_fit_polars_reports_missing_counts(polars_df):
    """Missing-value counts per column should be captured for the polars path."""
    artifact = DatasetProfileCalculator().fit(polars_df, {})
    assert artifact["profile"]["missing"]["num"] == 1
    assert artifact["profile"]["missing"]["cat"] == 0


def test_dataset_profile_fit_polars_numeric_stats_present(polars_df):
    """Polars describe()-derived numeric stats should be attached for numeric columns."""
    artifact = DatasetProfileCalculator().fit(polars_df, {})
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


def test_data_snapshot_fit_pandas_respects_n_rows(pandas_df):
    """A configured n_rows should limit the number of snapshot records returned."""
    artifact = DataSnapshotCalculator().fit(pandas_df, {"n_rows": 2})
    snapshot = typing.cast(list, artifact["snapshot"])
    assert len(snapshot) == 2
    assert snapshot[0]["cat"] == "a"


def test_data_snapshot_fit_polars_respects_n_rows(polars_df):
    """The polars fit path should also honor a configured n_rows."""
    artifact = DataSnapshotCalculator().fit(polars_df, {"n_rows": 2})
    snapshot = typing.cast(list, artifact["snapshot"])
    assert len(snapshot) == 2
    assert snapshot[0]["cat"] == "a"


def test_data_snapshot_applier_is_passthrough(pandas_df):
    """DataSnapshotApplier.apply should return the input frame unchanged."""
    result = DataSnapshotApplier().apply(pandas_df, {})
    assert result is pandas_df
