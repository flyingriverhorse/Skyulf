"""Unit tests for the DropMissingRows cleaning node.

Covers: Calculator.fit artifact shape, Applier.apply for pandas + polars
(how='any'/'all', subset filtering, threshold), y-synchronization for tuple
input, and edge cases (empty DataFrame, all-NaN column, no missing values).
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import polars as pl
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.preprocessing.drop_and_missing.drop_rows import (
    DropMissingRowsApplier,
    DropMissingRowsCalculator,
)

# ---------------------------------------------------------------------------
# Calculator.fit
# ---------------------------------------------------------------------------


def test_fit_default_how_is_any() -> None:
    """fit must default `how` to 'any' when not specified."""
    df = pd.DataFrame({"a": [1, np.nan]})
    params = DropMissingRowsCalculator().fit(df, {})
    assert params["how"] == "any"
    assert params["type"] == "drop_missing_rows"


def test_fit_preserves_subset_and_threshold() -> None:
    """fit must pass through the configured subset and threshold values."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    params = DropMissingRowsCalculator().fit(df, {"subset": ["a"], "threshold": 1})
    assert params["subset"] == ["a"]
    assert params["threshold"] == 1


def test_fit_infer_output_schema_passes_through() -> None:
    """infer_output_schema must return the input schema (row-only drop)."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["a"], {"a": "float64"})
    result = DropMissingRowsCalculator().infer_output_schema(schema, {})
    assert result is schema


# ---------------------------------------------------------------------------
# Applier.apply — pandas, how='any' (default)
# ---------------------------------------------------------------------------


def test_apply_pandas_how_any_drops_rows_with_any_nan() -> None:
    """how='any' must drop a row if any column in subset has a NaN."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, np.nan]})
    params: Dict[str, Any] = {"subset": None, "how": "any", "threshold": None}
    result = DropMissingRowsApplier().apply(df, params)
    # Rows 0 kept (no NaN); rows 1 and 2 dropped (each has a NaN).
    assert result["a"].tolist() == [1.0]
    assert result["b"].tolist() == [1.0]


def test_apply_pandas_how_all_drops_only_fully_nan_rows() -> None:
    """how='all' must drop a row only when every subset column is NaN."""
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [1.0, 2.0, np.nan]})
    params: Dict[str, Any] = {"subset": None, "how": "all", "threshold": None}
    result = DropMissingRowsApplier().apply(df, params)
    # Row 1 (a=NaN, b=2.0) kept because not all NaN; row 2 (both NaN) dropped.
    assert result.shape[0] == 2
    assert result["b"].tolist() == [1.0, 2.0]
    assert np.isnan(result["a"].iloc[1])


def test_apply_pandas_subset_limits_check_columns() -> None:
    """Only columns in `subset` should trigger a drop; other NaNs must be ignored."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, np.nan]})
    params: Dict[str, Any] = {"subset": ["a"], "how": "any", "threshold": None}
    result = DropMissingRowsApplier().apply(df, params)
    # Only row index 1 (a is NaN) should be dropped; b's NaNs are irrelevant.
    assert result["a"].tolist() == [1.0, 3.0]
    assert result.shape[0] == 2


def test_apply_pandas_threshold_keeps_rows_with_enough_non_na() -> None:
    """threshold=2 must keep rows with at least 2 non-null values."""
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [1.0, 2.0, np.nan],
            "c": [1.0, np.nan, 1.0],
        }
    )
    params: Dict[str, Any] = {"subset": None, "how": "any", "threshold": 2}
    result = DropMissingRowsApplier().apply(df, params)
    # Row 0: 3 non-null (kept). Row 1: 1 non-null (dropped). Row 2: 2 non-null (kept).
    assert result["a"].tolist() == [1.0, 3.0]


def test_apply_pandas_no_missing_values_keeps_all_rows() -> None:
    """A DataFrame with no missing values must have all rows retained."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    params: Dict[str, Any] = {"subset": None, "how": "any", "threshold": None}
    result = DropMissingRowsApplier().apply(df, params)
    assert result["a"].tolist() == [1.0, 2.0, 3.0]


def test_apply_pandas_all_nan_column_drops_all_rows() -> None:
    """An all-NaN column with how='any' must drop every row."""
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
    params: Dict[str, Any] = {"subset": None, "how": "any", "threshold": None}
    result = DropMissingRowsApplier().apply(df, params)
    assert result.shape[0] == 0


def test_apply_pandas_empty_dataframe() -> None:
    """Applying to an already-empty DataFrame must not raise."""
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    params: Dict[str, Any] = {"subset": None, "how": "any", "threshold": None}
    result = DropMissingRowsApplier().apply(df, params)
    assert result.shape == (0, 1)


def test_apply_pandas_tuple_xy_syncs_y_after_drop() -> None:
    """Dropping rows from X (as an (X, y) tuple) must drop the same rows from y."""
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    y = pd.Series([10, 20, 30])
    params: Dict[str, Any] = {"subset": None, "how": "any", "threshold": None}
    X_out, y_out = DropMissingRowsApplier().apply((X, y), params)
    assert X_out["a"].tolist() == [1.0, 3.0]
    assert y_out.tolist() == [10, 30]


# ---------------------------------------------------------------------------
# Applier.apply — polars
# ---------------------------------------------------------------------------


def test_apply_polars_how_any_drops_rows_with_any_null() -> None:
    """Polars path: how='any' must drop rows containing any null in subset."""
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, None]})
    params: Dict[str, Any] = {"subset": None, "how": "any", "threshold": None}
    result = DropMissingRowsApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["a"].tolist() == [1.0]


def test_apply_polars_how_all_drops_only_fully_null_rows() -> None:
    """Polars path: how='all' drops a row only when every subset column is null."""
    df = pl.DataFrame({"a": [1.0, None, None], "b": [1.0, 2.0, None]})
    params: Dict[str, Any] = {"subset": None, "how": "all", "threshold": None}
    result = DropMissingRowsApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result.shape[0] == 2


def test_apply_polars_subset_limits_check_columns() -> None:
    """Polars path: only subset columns must trigger a row drop."""
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [None, None, None]})
    params: Dict[str, Any] = {"subset": ["a"], "how": "any", "threshold": None}
    result = DropMissingRowsApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["a"].tolist() == [1.0, 3.0]


def test_apply_polars_threshold_keeps_rows_with_enough_non_null() -> None:
    """Polars path: threshold must keep rows meeting the minimum non-null count."""
    df = pl.DataFrame(
        {
            "a": [1.0, None, 3.0],
            "b": [1.0, 2.0, None],
            "c": [1.0, None, 1.0],
        }
    )
    params: Dict[str, Any] = {"subset": None, "how": "any", "threshold": 2}
    result = DropMissingRowsApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["a"].tolist() == [1.0, 3.0]


def test_apply_polars_tuple_xy_syncs_y_after_drop() -> None:
    """Polars path: dropping rows from X must drop matching rows from a polars y series."""
    X = pl.DataFrame({"a": [1.0, None, 3.0]})
    y = pl.Series("target", [10, 20, 30])
    params: Dict[str, Any] = {"subset": None, "how": "any", "threshold": None}
    X_out, y_out = DropMissingRowsApplier().apply((X, y), params)
    if hasattr(X_out, "to_pandas"):
        X_out = X_out.to_pandas()
    if hasattr(y_out, "to_list"):
        y_out = y_out.to_list()
    assert X_out["a"].tolist() == [1.0, 3.0]
    assert list(y_out) == [10, 30]


# ---------------------------------------------------------------------------
# fit -> apply round trip
# ---------------------------------------------------------------------------


def test_fit_then_apply_round_trip_subset() -> None:
    """fit()+apply() must drop only rows missing values in the fitted subset."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, np.nan]})
    calc = DropMissingRowsCalculator()
    applier = DropMissingRowsApplier()
    params = calc.fit(df, {"subset": ["a"]})
    result = applier.apply(df, params)
    assert result["a"].tolist() == [1.0, 3.0]


def test_fit_then_apply_round_trip_default_how_any() -> None:
    """fit()+apply() with default config drops any row containing a NaN."""
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]})
    calc = DropMissingRowsCalculator()
    applier = DropMissingRowsApplier()
    params = calc.fit(df, {})
    result = applier.apply(df, params)
    assert result.shape[0] == 1
    assert result["a"].tolist() == [1.0]


# ---------------------------------------------------------------------------
# Real-shaped dataset integration
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has NaN values across multiple columns (``age``, ``income``, ``city``,
    ``lat``, ``lon``) — closer to production data than the small synthetic
    frames used elsewhere in this file.
    """

    def test_drop_any_nan_rows_reduces_row_count(self) -> None:
        """Dropping rows where any column has a NaN must reduce the 15-row
        customers dataset since multiple rows have missing values.
        """
        df = load_sample_dataset("customers")
        assert df.shape[0] == 15

        calc = DropMissingRowsCalculator()
        applier = DropMissingRowsApplier()
        params = calc.fit(df, {"how": "any"})
        result = applier.apply(df, params)

        assert result.shape[0] < 15
        assert result.isna().sum().sum() == 0

    def test_drop_subset_age_removes_only_missing_age_rows(self) -> None:
        """Dropping on subset=['age'] must remove only the rows where age is NaN,
        leaving income/city NaN rows intact.
        """
        df = load_sample_dataset("customers")
        n_age_nan = df["age"].isna().sum()
        assert n_age_nan > 0

        calc = DropMissingRowsCalculator()
        applier = DropMissingRowsApplier()
        params = calc.fit(df, {"how": "any", "subset": ["age"]})
        result = applier.apply(df, params)

        assert result.shape[0] == 15 - n_age_nan
        assert result["age"].isna().sum() == 0
