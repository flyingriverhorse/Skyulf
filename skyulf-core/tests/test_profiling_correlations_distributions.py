"""Tests for skyulf.profiling.correlations.calculate_correlations and
skyulf.profiling.distributions.calculate_histogram.
"""

import numpy as np
import polars as pl
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.profiling.correlations import calculate_correlations
from skyulf.profiling.distributions import calculate_histogram


def test_calculate_correlations_returns_matrix_for_correlated_columns() -> None:
    """Two collinear numeric columns should produce a near +/-1 correlation matrix."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = a * 2 + rng.normal(0, 0.01, 200)
    df = pl.DataFrame({"a": a, "b": b}).lazy()

    matrix = calculate_correlations(df, ["a", "b"])

    assert matrix is not None
    assert matrix.columns == ["a", "b"]
    assert abs(matrix.values[0][1]) > 0.99


def test_calculate_correlations_none_for_single_column() -> None:
    """A single numeric column can't form a correlation matrix; returns None."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]}).lazy()
    assert calculate_correlations(df, ["a"]) is None


def test_calculate_correlations_drops_constant_columns() -> None:
    """Constant columns (std == 0) should be excluded from the resulting matrix."""
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 5.0, 5.0, 5.0],
            "c": [4.0, 3.0, 2.0, 1.0],
        }
    ).lazy()

    matrix = calculate_correlations(df, ["a", "b", "c"])

    assert matrix is not None
    assert "b" not in matrix.columns
    assert set(matrix.columns) == {"a", "c"}


def test_calculate_correlations_none_when_all_columns_constant() -> None:
    """If fewer than 2 columns survive the constant-column filter, return None."""
    df = pl.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]}).lazy()
    assert calculate_correlations(df, ["a", "b"]) is None


def test_calculate_correlations_caps_at_twenty_columns() -> None:
    """More than 20 numeric columns should be truncated to the first 20."""
    data = {f"c{i}": list(np.linspace(0, 1, 10) + i) for i in range(25)}
    df = pl.DataFrame(data).lazy()
    cols = list(data.keys())

    matrix = calculate_correlations(df, cols)

    assert matrix is not None
    assert len(matrix.columns) <= 20


def test_calculate_correlations_returns_none_on_unexpected_error() -> None:
    """An internal error (e.g. a column that doesn't exist) should be caught and return None."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]}).lazy()
    assert calculate_correlations(df, ["a", "does_not_exist"]) is None


def test_calculate_histogram_basic_bins() -> None:
    """Histogram bins should cover the min/max range and sum counts to row count."""
    df = pl.DataFrame({"x": list(range(100))}).lazy()
    hist = calculate_histogram(df, "x", bins=10)

    assert hist is not None
    assert len(hist) == 10
    assert hist[0].start == 0.0
    assert hist[-1].end == 99.0
    assert sum(b.count for b in hist) == 100


def test_calculate_histogram_none_for_constant_column() -> None:
    """A constant column has no range, so no histogram can be built."""
    df = pl.DataFrame({"x": [5.0, 5.0, 5.0]}).lazy()
    assert calculate_histogram(df, "x") is None


def test_calculate_histogram_none_for_all_null_column() -> None:
    """An all-null column also has no min/max, so histogram should be None."""
    df = pl.DataFrame({"x": pl.Series([None, None], dtype=pl.Float64)}).lazy()
    assert calculate_histogram(df, "x") is None


def test_calculate_histogram_returns_none_on_unexpected_error() -> None:
    """An internal error (e.g. a column that doesn't exist) should be caught and return None."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]}).lazy()
    assert calculate_histogram(df, "does_not_exist") is None


def test_calculate_histogram_skips_unparseable_null_bin_group() -> None:
    """Null values form a 'None' bin group that can't be parsed as an int; it should be skipped."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, None, 5.0]}).lazy()
    hist = calculate_histogram(df, "x", bins=5)

    assert hist is not None
    # The single null value is excluded from every bin's count.
    assert sum(b.count for b in hist) == 4


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing age/income values — closer to production data than the
    small synthetic frames used elsewhere in this file.
    """

    def test_correlations_and_histogram_on_customers_income(self) -> None:
        df_eager = load_sample_dataset("customers", engine="polars")
        df = df_eager.lazy()

        matrix = calculate_correlations(df, ["age", "income"])
        assert matrix is not None
        assert set(matrix.columns) == {"age", "income"}

        hist = calculate_histogram(df, "income", bins=5)
        assert hist is not None
        # 3 of the 15 rows have a missing income; those rows are excluded from
        # every bin's count.
        non_null_income = df_eager["income"].drop_nulls().len()
        assert sum(b.count for b in hist) == non_null_income
