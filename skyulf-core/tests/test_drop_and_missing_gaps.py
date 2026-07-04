"""Coverage-gap tests for MissingIndicator, Deduplicate, DropMissingColumns.

Targets the pandas+polars fit/apply branches, threshold parsing, and
``infer_output_schema`` predictions that were not yet exercised.
"""

import pandas as pd
import polars as pl

from skyulf.preprocessing._schema import SkyulfSchema
from skyulf.preprocessing.drop_and_missing.deduplicate import (
    DeduplicateApplier,
    DeduplicateCalculator,
)
from skyulf.preprocessing.drop_and_missing.drop_columns import (
    DropMissingColumnsApplier,
    DropMissingColumnsCalculator,
)
from skyulf.preprocessing.drop_and_missing.missing_indicator import (
    MissingIndicatorApplier,
    MissingIndicatorCalculator,
)


def _missing_df() -> pd.DataFrame:
    """Build a small frame with two columns that have missing values."""
    return pd.DataFrame(
        {
            "a": [1.0, None, 3.0, None],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": [None, None, None, None],
        }
    )


# ---------------------------------------------------------------------------
# MissingIndicator
# ---------------------------------------------------------------------------


def test_missing_indicator_auto_detects_columns_with_nulls_pandas() -> None:
    """Without explicit columns, pandas fit auto-detects columns containing NaNs."""
    df = _missing_df()
    art = MissingIndicatorCalculator().fit(df, {})
    assert set(art["columns"]) == {"a", "c"}


def test_missing_indicator_auto_detects_columns_with_nulls_polars() -> None:
    """Polars fit path must auto-detect the same missing columns as pandas."""
    df = pl.from_pandas(_missing_df())
    art = MissingIndicatorCalculator().fit(df, {})
    assert set(art["columns"]) == {"a", "c"}


def test_missing_indicator_explicit_columns_filtered_to_existing() -> None:
    """Explicitly configured columns not present in the frame are dropped."""
    df = _missing_df()
    art = MissingIndicatorCalculator().fit(df, {"columns": ["a", "nonexistent"]})
    assert art["columns"] == ["a"]


def test_missing_indicator_apply_creates_binary_flags_pandas() -> None:
    """Applying the artifact adds a ``<col>_missing`` binary indicator column."""
    df = _missing_df()
    art = MissingIndicatorCalculator().fit(df, {"columns": ["a"]})
    out = MissingIndicatorApplier().apply(df, art)
    assert out["a_missing"].tolist() == [0, 1, 0, 1]


def test_missing_indicator_apply_creates_binary_flags_polars() -> None:
    """Polars apply path must match the pandas binary-flag semantics."""
    df = pl.from_pandas(_missing_df())
    art = MissingIndicatorCalculator().fit(df, {"columns": ["a"]})
    out = MissingIndicatorApplier().apply(df, art)
    assert out["a_missing"].to_list() == [0, 1, 0, 1]


def test_missing_indicator_apply_noop_without_columns() -> None:
    """An empty ``columns`` list leaves the frame unchanged."""
    df = _missing_df()
    out = MissingIndicatorApplier().apply(df, {"columns": []})
    assert "a_missing" not in out.columns


def test_missing_indicator_infer_output_schema_explicit_columns() -> None:
    """Explicit columns predict one new bool column per indicator."""
    schema = SkyulfSchema.from_columns(["a", "b"], {"a": "float64", "b": "float64"})
    out = MissingIndicatorCalculator().infer_output_schema(schema, {"columns": ["a"]})
    assert out is not None
    assert "a_missing" in out
    assert out.dtypes["a_missing"] == "bool"


def test_missing_indicator_infer_output_schema_none_without_explicit_columns() -> None:
    """Without an explicit column list the output schema is data-dependent (None)."""
    schema = SkyulfSchema.from_columns(["a"], {"a": "float64"})
    out = MissingIndicatorCalculator().infer_output_schema(schema, {})
    assert out is None


# ---------------------------------------------------------------------------
# Deduplicate
# ---------------------------------------------------------------------------


def test_deduplicate_pandas_keep_first_default() -> None:
    """Default ``keep="first"`` drops later duplicate rows."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    art = DeduplicateCalculator().fit(df, {})
    out = DeduplicateApplier().apply(df, art)
    assert len(out) == 2
    assert out["a"].tolist() == [1, 2]


def test_deduplicate_pandas_keep_none_drops_all_dupes() -> None:
    """``keep="none"`` removes every row that has any duplicate, including the first."""
    df = pd.DataFrame({"a": [1, 1, 2]})
    art = DeduplicateCalculator().fit(df, {"keep": "none"})
    out = DeduplicateApplier().apply(df, art)
    assert out["a"].tolist() == [2]


def test_deduplicate_polars_keep_none_drops_all_dupes() -> None:
    """Polars path must apply the same "none" semantics as pandas."""
    df = pl.DataFrame({"a": [1, 1, 2]})
    art = DeduplicateCalculator().fit(df.to_pandas(), {"keep": "none"})
    out = DeduplicateApplier().apply(df, art)
    assert out["a"].to_list() == [2]


def test_deduplicate_with_y_tuple_syncs_rows_pandas() -> None:
    """Dropping duplicate rows in X must drop the paired rows in y too."""
    X = pd.DataFrame({"a": [1, 1, 2]})
    y = pd.Series([10, 20, 30])
    art = DeduplicateCalculator().fit(X, {})
    X_out, y_out = DeduplicateApplier().apply((X, y), art)
    assert len(X_out) == len(y_out) == 2
    assert y_out.tolist() == [10, 30]


def test_deduplicate_with_y_tuple_syncs_rows_polars() -> None:
    """Polars (X, y) tuple path must drop matching rows from y."""
    X = pl.DataFrame({"a": [1, 1, 2]})
    y = pl.Series("y", [10, 20, 30])
    art = DeduplicateCalculator().fit(X.to_pandas(), {})
    X_out, y_out = DeduplicateApplier().apply((X, y), art)
    assert X_out.height == 2
    assert y_out.to_list() == [10, 30]


def test_deduplicate_subset_filters_to_existing_columns() -> None:
    """A ``subset`` referencing a missing column falls back gracefully."""
    df = pd.DataFrame({"a": [1, 1], "b": [1, 2]})
    art = DeduplicateCalculator().fit(df, {"subset": ["a", "nonexistent"]})
    out = DeduplicateApplier().apply(df, art)
    # subset resolves to just "a", both rows share a==1 -> dedup keeps 1st
    assert len(out) == 1


def test_deduplicate_infer_output_schema_preserves_columns() -> None:
    """Deduplication only removes rows, so the schema is unchanged."""
    schema = SkyulfSchema.from_columns(["a", "b"], {"a": "int64", "b": "int64"})
    out = DeduplicateCalculator().infer_output_schema(schema, {})
    assert out is schema


# ---------------------------------------------------------------------------
# DropMissingColumns
# ---------------------------------------------------------------------------


def test_drop_missing_columns_by_explicit_list_pandas() -> None:
    """Explicit ``columns`` are queued for drop regardless of missing-%."""
    df = _missing_df()
    art = DropMissingColumnsCalculator().fit(df, {"columns": ["b"]})
    out = DropMissingColumnsApplier().apply(df, art)
    assert "b" not in out.columns


def test_drop_missing_columns_by_threshold_pandas() -> None:
    """Columns whose missing-% meets/exceeds the threshold are dropped."""
    df = _missing_df()
    art = DropMissingColumnsCalculator().fit(df, {"missing_threshold": 50})
    out = DropMissingColumnsApplier().apply(df, art)
    # "a" is 50% missing, "c" is 100% missing -> both dropped; "b" kept
    assert "a" not in out.columns
    assert "c" not in out.columns
    assert "b" in out.columns


def test_drop_missing_columns_by_threshold_polars() -> None:
    """Polars threshold computation must match the pandas percentage math."""
    df = pl.from_pandas(_missing_df())
    art = DropMissingColumnsCalculator().fit(df, {"missing_threshold": 50})
    out = DropMissingColumnsApplier().apply(df, art)
    assert "a" not in out.columns
    assert "c" not in out.columns
    assert "b" in out.columns


def test_drop_missing_columns_threshold_non_numeric_ignored() -> None:
    """A non-numeric threshold is treated as absent (resolves to None)."""
    df = _missing_df()
    art = DropMissingColumnsCalculator().fit(df, {"missing_threshold": "not-a-number"})
    assert art["columns_to_drop"] == []


def test_drop_missing_columns_threshold_zero_or_negative_ignored() -> None:
    """A threshold of 0 or negative is treated as "no threshold configured"."""
    df = _missing_df()
    art = DropMissingColumnsCalculator().fit(df, {"missing_threshold": 0})
    assert art["columns_to_drop"] == []
    art_neg = DropMissingColumnsCalculator().fit(df, {"missing_threshold": -5})
    assert art_neg["columns_to_drop"] == []


def test_drop_missing_columns_apply_noop_without_matching_columns() -> None:
    """Applying with an already-absent column list leaves the frame unchanged."""
    df = _missing_df()
    out = DropMissingColumnsApplier().apply(df, {"columns_to_drop": ["nonexistent"]})
    assert list(out.columns) == list(df.columns)


def test_drop_missing_columns_infer_output_schema_explicit_only() -> None:
    """Schema inference drops the explicit column list when no threshold given."""
    schema = SkyulfSchema.from_columns(["a", "b"], {"a": "float64", "b": "float64"})
    out = DropMissingColumnsCalculator().infer_output_schema(schema, {"columns": ["a"]})
    assert out is not None
    assert "a" not in out
    assert "b" in out


def test_drop_missing_columns_infer_output_schema_none_with_threshold() -> None:
    """A positive threshold makes the output schema data-dependent (None)."""
    schema = SkyulfSchema.from_columns(["a"], {"a": "float64"})
    out = DropMissingColumnsCalculator().infer_output_schema(schema, {"missing_threshold": 30})
    assert out is None
