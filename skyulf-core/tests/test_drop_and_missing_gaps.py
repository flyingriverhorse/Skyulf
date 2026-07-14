"""Coverage-gap tests for MissingIndicator, Deduplicate, DropMissingColumns.

Targets the pandas+polars fit/apply branches, threshold parsing, and
``infer_output_schema`` predictions that were not yet exercised.
"""

import pandas as pd
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing._schema import SkyulfSchema
from skyulf.preprocessing.drop_and_missing._common import (
    _normalize_subset,
    _polars_filter_y_by_kept_indices,
)
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

_threshold_ignored_cases = TestCaseLoader(
    "preprocessing/drop_and_missing_gaps", group="threshold_ignored"
).load()
_normalize_subset_cases = TestCaseLoader(
    "preprocessing/drop_and_missing_gaps", group="normalize_subset"
).load()


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


def test_missing_indicator_apply_noop_without_columns_polars() -> None:
    """Polars apply with an empty ``columns`` list must leave the frame unchanged."""
    df = pl.from_pandas(_missing_df())
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


def test_missing_indicator_custom_flag_suffix_pandas() -> None:
    """Regression test: a custom flag_suffix must actually be honored, not
    silently ignored in favor of a hardcoded "_missing" - previously
    node_meta declared unused "features"/"sparse" params while the
    frontend's flag_suffix field (forwarded via pipelineConverter.ts) was
    accepted into params but never read by the apply functions."""
    df = _missing_df()
    art = MissingIndicatorCalculator().fit(df, {"columns": ["a"], "flag_suffix": "_was_missing"})
    assert art["flag_suffix"] == "_was_missing"
    out = MissingIndicatorApplier().apply(df, art)
    assert "a_was_missing" in out.columns
    assert "a_missing" not in out.columns
    assert out["a_was_missing"].tolist() == [0, 1, 0, 1]


def test_missing_indicator_custom_flag_suffix_polars() -> None:
    """Polars apply path must also honor a custom flag_suffix."""
    df = pl.from_pandas(_missing_df())
    art = MissingIndicatorCalculator().fit(df, {"columns": ["a"], "flag_suffix": "_was_missing"})
    out = MissingIndicatorApplier().apply(df, art)
    assert "a_was_missing" in out.columns
    assert "a_missing" not in out.columns
    assert out["a_was_missing"].to_list() == [0, 1, 0, 1]


def test_missing_indicator_default_flag_suffix_when_not_configured() -> None:
    """Omitting flag_suffix must still default to '_missing' for backward compat."""
    df = _missing_df()
    art = MissingIndicatorCalculator().fit(df, {"columns": ["a"]})
    assert art["flag_suffix"] == "_missing"
    out = MissingIndicatorApplier().apply(df, art)
    assert "a_missing" in out.columns


def test_missing_indicator_infer_output_schema_honors_custom_flag_suffix() -> None:
    """infer_output_schema must predict the correctly-suffixed column name
    when a custom flag_suffix is configured."""
    schema = SkyulfSchema.from_columns(["a", "b"], {"a": "float64", "b": "float64"})
    out = MissingIndicatorCalculator().infer_output_schema(
        schema, {"columns": ["a"], "flag_suffix": "_was_missing"}
    )
    assert out is not None
    assert "a_was_missing" in out
    assert "a_missing" not in out


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


class TestDropMissingColumnsThresholdIgnored:
    """A non-numeric or non-positive threshold is treated as "no threshold
    configured" — scenarios loaded from
    ``tests/test_cases/preprocessing/drop_and_missing_gaps.json`` (group ``threshold_ignored``).
    """

    @pytest.mark.parametrize(*_threshold_ignored_cases)
    def test_drop_missing_columns_threshold_ignored(self, missing_threshold: object) -> None:
        df = _missing_df()
        art = DropMissingColumnsCalculator().fit(df, {"missing_threshold": missing_threshold})
        assert art["columns_to_drop"] == []


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


# ---------------------------------------------------------------------------
# _common._polars_filter_y_by_kept_indices — direct unit tests
# ---------------------------------------------------------------------------


def test_polars_filter_y_by_kept_indices_none_y_passthrough() -> None:
    """A None ``y`` must pass through unchanged."""
    kept = pl.Series([0, 2])
    assert _polars_filter_y_by_kept_indices(None, kept) is None


def test_polars_filter_y_by_kept_indices_dataframe_y_filters_rows() -> None:
    """A Polars DataFrame ``y`` must be filtered to the rows in ``kept_indices``."""
    y = pl.DataFrame({"label": [10, 20, 30, 40]})
    kept = pl.Series([0, 2])
    out = _polars_filter_y_by_kept_indices(y, kept)
    assert isinstance(out, pl.DataFrame)
    assert out["label"].to_list() == [10, 30]
    assert "__idx__" not in out.columns


def test_polars_filter_y_by_kept_indices_series_y_gathers_rows() -> None:
    """A Polars Series ``y`` must be gathered at the kept row positions."""
    y = pl.Series("label", [10, 20, 30, 40])
    kept = pl.Series([1, 3])
    out = _polars_filter_y_by_kept_indices(y, kept)
    assert isinstance(out, pl.Series)
    assert out.to_list() == [20, 40]


def test_polars_filter_y_by_kept_indices_non_polars_y_passthrough() -> None:
    """A non-Polars, non-None ``y`` (e.g. a pandas Series) must pass through unchanged."""
    y = pd.Series([10, 20, 30])
    kept = pl.Series([0, 1])
    out = _polars_filter_y_by_kept_indices(y, kept)
    assert out is y


# ---------------------------------------------------------------------------
# _common._normalize_subset — direct unit tests
# ---------------------------------------------------------------------------


class TestNormalizeSubset:
    """``_normalize_subset`` column-filtering behavior — scenarios loaded from
    ``tests/test_cases/preprocessing/drop_and_missing_gaps.json`` (group ``normalize_subset``).
    """

    @pytest.mark.parametrize(*_normalize_subset_cases)
    def test_normalize_subset(
        self, subset: list[str] | None, existing_cols: list[str], expected: list[str] | None
    ) -> None:
        assert _normalize_subset(subset, existing_cols) == expected


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing values in ``age``/``income``/``city``/``lat``/``lon`` —
    closer to production data than the small synthetic frame used elsewhere
    in this file.
    """

    def test_missing_indicator_flags_columns_with_real_gaps(self) -> None:
        df = load_sample_dataset("customers")
        params = MissingIndicatorCalculator().fit(df, {})
        assert set(params["columns"]) == {"age", "income", "city", "lat", "lon"}

        out = MissingIndicatorApplier().apply(df, params)
        for col in params["columns"]:
            expected = df[col].isna().astype(int)
            pd.testing.assert_series_equal(out[f"{col}_missing"], expected, check_names=False)
