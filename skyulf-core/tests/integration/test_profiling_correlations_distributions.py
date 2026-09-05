"""Tests for skyulf.profiling.correlations.calculate_correlations and
skyulf.profiling.distributions.calculate_histogram.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.profiling.correlations import (
    _clean_cell,
    _pairwise_correlation_matrix,
    calculate_correlations,
)
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


def test_calculate_correlations_truncation_logs_warning_with_dropped_columns(
    caplog,
) -> None:
    """Regression test: truncating to the first 20 columns (by order, not
    variance/relevance) previously gave the caller/UI no signal that data was
    dropped. Must now log a warning naming the dropped columns."""
    import logging

    data = {f"c{i}": list(np.linspace(0, 1, 10) + i) for i in range(25)}
    df = pl.DataFrame(data).lazy()
    cols = list(data.keys())

    with caplog.at_level(logging.WARNING, logger="skyulf.profiling.correlations"):
        matrix = calculate_correlations(df, cols)

    assert matrix is not None
    assert any("truncating to the first 20" in rec.message for rec in caplog.records)
    # The dropped columns (c20..c24) must be named in the warning.
    assert any("c20" in rec.message for rec in caplog.records)


def test_calculate_correlations_no_warning_when_under_cap(caplog) -> None:
    """20 or fewer numeric columns must not trigger the truncation warning."""
    import logging

    data = {f"c{i}": list(np.linspace(0, 1, 10) + i) for i in range(10)}
    df = pl.DataFrame(data).lazy()
    cols = list(data.keys())

    with caplog.at_level(logging.WARNING, logger="skyulf.profiling.correlations"):
        matrix = calculate_correlations(df, cols)

    assert matrix is not None
    assert not any("truncating" in rec.message for rec in caplog.records)


def test_calculate_correlations_returns_none_on_unexpected_error() -> None:
    """An internal error (e.g. a column that doesn't exist) should be caught and return None."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]}).lazy()
    assert calculate_correlations(df, ["a", "does_not_exist"]) is None


def test_calculate_correlations_keeps_nan_bearing_column() -> None:
    """OC-43: a NaN in one column must not discard the entire matrix.

    ``std()`` of a NaN-bearing polars column is NaN, and the constant-column
    filter tested ``std > 1e-9`` — False for NaN — so the column was dropped as
    if it had no variance. With two columns that left one survivor and
    ``calculate_correlations`` returned ``None`` for the whole matrix, while
    pandas reports a perfectly good 1.0.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, float("nan"), 4.0], "y": [1.0, 2.0, 3.0, 4.0]}).lazy()

    matrix = calculate_correlations(df, ["x", "y"])

    assert matrix is not None
    assert matrix.columns == ["x", "y"]
    assert matrix.values[0][1] == pytest.approx(1.0)


def test_calculate_correlations_uses_pairwise_not_listwise_deletion() -> None:
    """OC-43: each coefficient must be scored on its own pair's observed rows.

    The old path called ``drop_nulls()`` first, discarding every row missing
    *any* column, then ``DataFrame.corr()`` (which is itself listwise and
    returns NaN for every cell if a single null survives). On the frame below
    the complete-row subset is rows 2-5, where ``corr(a, b)`` is 0.6; pairwise
    over all six rows it is 0.8286, which is what pandas reports.
    """
    data = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        "c": [None, None, 3.0, 4.0, 5.0, 6.0],
    }
    expected = pd.DataFrame(data).corr()

    matrix = calculate_correlations(pl.DataFrame(data).lazy(), ["a", "b", "c"])

    assert matrix is not None
    assert matrix.columns == ["a", "b", "c"]
    for i, left in enumerate(matrix.columns):
        for j, right in enumerate(matrix.columns):
            assert matrix.values[i][j] == pytest.approx(expected.loc[left, right]), (left, right)
    # Discriminating: the listwise answer for this pair is materially different.
    assert matrix.values[0][1] == pytest.approx(0.8285714285714286)
    complete_rows = pd.DataFrame(data).dropna()
    assert complete_rows[["a", "b"]].corr().iloc[0, 1] == pytest.approx(0.6)


def test_calculate_correlations_zeroes_and_warns_on_degenerate_pair(caplog) -> None:
    """A pair sharing fewer than ``MIN_PAIRWISE_OVERLAP`` rows is not reported
    as a coefficient: at n=2 Pearson r is always exactly +/-1.0, so the cell
    would assert a perfect relationship the data cannot support.
    """
    data = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "c": [None, None, None, None, 5.0, -50.0],
    }

    with caplog.at_level("WARNING"):
        matrix = calculate_correlations(pl.DataFrame(data).lazy(), ["a", "b", "c"])

    assert matrix is not None
    # a~b is fully observed and perfectly collinear.
    assert matrix.values[0][1] == pytest.approx(1.0)
    # Every pair involving c shares only 2 rows.
    c_index = matrix.columns.index("c")
    for other in range(len(matrix.columns)):
        if other != c_index:
            assert matrix.values[c_index][other] == 0.0
    assert any("fewer than" in rec.message for rec in caplog.records)


def test_calculate_correlations_still_drops_all_missing_column() -> None:
    """A column with no observed values at all has no std and stays excluded —
    pandas reports NaN for it, and NaN is not expressible in the matrix.
    """
    data = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 4.0, 6.0, 8.0],
        "empty": [float("nan"), float("nan"), float("nan"), float("nan")],
    }

    matrix = calculate_correlations(pl.DataFrame(data).lazy(), ["a", "b", "empty"])

    assert matrix is not None
    assert matrix.columns == ["a", "b"]
    # The surviving pair is fully observed, so it is still scored normally.
    assert matrix.values[0][1] == pytest.approx(1.0)


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


class TestPairwiseDeletionEdgeCases:
    """Branches pairwise deletion opens up that the happy path never reaches."""

    def test_pair_with_no_variance_in_its_overlap_is_reported_as_zero(self) -> None:
        """``b`` varies overall, so it survives the constant-column filter, but
        it is constant across the four rows where *both* columns are observed.
        ``pl.corr`` then divides by a zero standard deviation and yields NaN
        rather than a coefficient, and ``CorrelationMatrix`` has no "unknown"
        cell — so the pair must land on 0.0 instead of leaking a NaN into the
        matrix."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, None], "b": [5.0, 5.0, 5.0, 5.0, 99.0]}).lazy()

        matrix = calculate_correlations(df, ["a", "b"])

        assert matrix is not None
        assert matrix.columns == ["a", "b"]
        assert matrix.values == [[1.0, 0.0], [0.0, 1.0]]

    def test_clean_cell_coerces_missing_and_non_finite_to_zero(self) -> None:
        """Direct call on purpose for the ``None`` case. ``calculate_correlations``
        cannot produce one: a pair sharing fewer than ``MIN_PAIRWISE_OVERLAP``
        observations is replaced by a literal ``0.0`` before ``_clean_cell`` runs,
        and every other pair gets a float back from polars (NaN when the overlap
        has no variance, as the test above pins). ``None`` is defended against
        because ``_clean_cell`` is typed ``object`` and polars' null-vs-NaN choice
        for a degenerate coefficient is not something to bet the matrix on."""
        assert _clean_cell(None) == 0.0
        assert _clean_cell(float("nan")) == 0.0
        assert _clean_cell(float("inf")) == 0.0
        assert _clean_cell(-0.87) == -0.87

    def test_frame_with_fewer_than_two_rows_yields_none(self) -> None:
        """One row (or none) cannot produce a coefficient, however many usable
        numeric columns are declared."""
        single = pl.DataFrame({"a": [1.0], "b": [2.0]}).lazy()
        assert calculate_correlations(single, ["a", "b"]) is None

        empty = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Float64, "b": pl.Float64}).lazy()
        assert calculate_correlations(empty, ["a", "b"]) is None

    def test_matrix_is_the_identity_when_there_are_no_pairs(self) -> None:
        """Direct call on purpose: ``calculate_correlations`` returns None
        whenever fewer than two columns survive filtering, so the no-pairs
        early exit is only reachable by a caller handing the helper a 0- or
        1-column list."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})

        assert _pairwise_correlation_matrix(df, ["a"]) == [[1.0]]
        assert _pairwise_correlation_matrix(df, []) == []
