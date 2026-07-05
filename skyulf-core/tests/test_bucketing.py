"""Unit tests for the binning / discretization nodes in `bucketing.py`.

Covers: fit-time bin-edge computation for each strategy (equal_width,
equal_frequency, kmeans, custom, kbins), apply-time bin assignment (ordinal,
range, custom labels), missing-value strategies, out-of-range values at
apply time, single-unique-value (constant) columns, drop_original /
output_suffix options, and the polars apply path.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.preprocessing.bucketing import (
    CustomBinningApplier,
    CustomBinningCalculator,
    GeneralBinningApplier,
    GeneralBinningCalculator,
    KBinsDiscretizerCalculator,
    _fit_equal_frequency,
    _fit_one_column_into_maps,
    _resolve_kbins_strategy,
)


def _series_0_to_9() -> pd.DataFrame:
    """A simple 10-row numeric DataFrame: x = 0..9."""
    return pd.DataFrame({"x": list(range(10))})


# ---------------------------------------------------------------------------
# Fit — equal_width
# ---------------------------------------------------------------------------


def test_fit_equal_width_bin_edges_exact() -> None:
    """equal_width with n_bins=5 on 0..9 must produce the known pandas.cut edges."""
    df = _series_0_to_9()
    params = GeneralBinningCalculator().fit(
        df, {"columns": ["x"], "strategy": "equal_width", "n_bins": 5}
    )
    edges = params["bin_edges"]["x"]
    assert edges == pytest.approx([0.0, 1.8, 3.6, 5.4, 7.2, 9.0])


def test_fit_equal_width_edge_zero_clamped_to_series_min() -> None:
    """The first edge must be clamped to the exact series minimum (never below it)."""
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
    params = GeneralBinningCalculator().fit(
        df, {"columns": ["x"], "strategy": "equal_width", "n_bins": 2}
    )
    assert params["bin_edges"]["x"][0] == 10.0


# ---------------------------------------------------------------------------
# Fit — equal_frequency (quantile)
# ---------------------------------------------------------------------------


def test_fit_equal_frequency_bin_edges_exact() -> None:
    """equal_frequency (qcut) with n_bins=4 on 0..9 must produce quartile edges."""
    df = _series_0_to_9()
    params = GeneralBinningCalculator().fit(
        df, {"columns": ["x"], "strategy": "equal_frequency", "n_bins": 4}
    )
    edges = params["bin_edges"]["x"]
    assert edges == pytest.approx([0.0, 2.25, 4.5, 6.75, 9.0])


def test_fit_equal_frequency_each_bin_has_similar_population() -> None:
    """Applying the qcut edges must place a near-equal number of rows per bin."""
    df = _series_0_to_9()
    calc = GeneralBinningCalculator()
    applier = GeneralBinningApplier()
    params = calc.fit(df, {"columns": ["x"], "strategy": "equal_frequency", "n_bins": 4})
    result = applier.apply(df, params)
    counts = result["x_binned"].value_counts().sort_index()
    # 10 rows into 4 bins -> most bins have 2-3 rows.
    assert set(counts.tolist()) <= {2, 3}


# ---------------------------------------------------------------------------
# Fit — kmeans / kbins
# ---------------------------------------------------------------------------


def test_fit_kmeans_bin_edges_exact() -> None:
    """kmeans strategy on 0..9 with n_bins=3 must match sklearn's KBinsDiscretizer directly."""
    df = _series_0_to_9()
    params = GeneralBinningCalculator().fit(
        df, {"columns": ["x"], "strategy": "kmeans", "n_bins": 3}
    )
    edges = params["bin_edges"]["x"]
    assert edges == pytest.approx([0.0, 3.25, 6.5, 9.0])


def test_fit_kbins_via_wrapper_uses_uniform_edges() -> None:
    """KBinsDiscretizerCalculator with strategy='uniform' must match sklearn uniform edges."""
    df = _series_0_to_9()
    params = KBinsDiscretizerCalculator().fit(
        df, {"columns": ["x"], "n_bins": 3, "strategy": "uniform"}
    )
    edges = params["bin_edges"]["x"]
    assert edges == pytest.approx([0.0, 3.0, 6.0, 9.0])


def test_resolve_kbins_strategy_aliases() -> None:
    """UI-friendly strategy aliases must map to sklearn's strategy names."""
    assert _resolve_kbins_strategy("equal_width") == "uniform"
    assert _resolve_kbins_strategy("equal_frequency") == "quantile"
    assert _resolve_kbins_strategy("kmeans") == "kmeans"


# ---------------------------------------------------------------------------
# Fit — custom edges
# ---------------------------------------------------------------------------


def test_fit_custom_binning_sorts_user_edges() -> None:
    """CustomBinningCalculator must sort the user-supplied bin edges."""
    df = _series_0_to_9()
    params = CustomBinningCalculator().fit(df, {"columns": ["x"], "bins": [9, 0, 5]})
    assert params["bin_edges"]["x"] == [0, 5, 9]


def test_fit_custom_binning_empty_bins_yields_empty_map() -> None:
    """An empty `bins` list must produce an empty bin_edges map (no-op)."""
    df = _series_0_to_9()
    params = CustomBinningCalculator().fit(df, {"columns": ["x"], "bins": []})
    assert params["bin_edges"] == {}


# ---------------------------------------------------------------------------
# Fit — user picked no columns
# ---------------------------------------------------------------------------


def test_fit_general_binning_no_columns_returns_empty_artifact() -> None:
    """An explicit empty `columns` list must short-circuit to an empty artifact."""
    df = _series_0_to_9()
    params = GeneralBinningCalculator().fit(df, {"columns": []})
    assert params == {}


# ---------------------------------------------------------------------------
# Apply — bin assignment values
# ---------------------------------------------------------------------------


def test_apply_ordinal_label_format_assigns_expected_bin_indices() -> None:
    """Ordinal (default) label_format must assign integer bin indices matching pd.cut."""
    df = _series_0_to_9()
    calc = GeneralBinningCalculator()
    applier = GeneralBinningApplier()
    params = calc.fit(df, {"columns": ["x"], "strategy": "equal_width", "n_bins": 5})
    result = applier.apply(df, params)
    expected = pd.cut(
        df["x"], bins=sorted(set(params["bin_edges"]["x"])), labels=False, include_lowest=True
    )
    assert result["x_binned"].tolist() == expected.tolist()


def test_apply_range_label_format_produces_interval_strings() -> None:
    """label_format='range' must produce '[a, b]'/'(a, b]' style string labels."""
    df = _series_0_to_9()
    config = {
        "columns": ["x"],
        "strategy": "equal_width",
        "n_bins": 5,
        "label_format": "range",
    }
    params = GeneralBinningCalculator().fit(df, config)
    result = GeneralBinningApplier().apply(df, params)
    first_label = result["x_binned"].iloc[0]
    assert first_label.startswith("[")
    assert "," in first_label


def test_apply_custom_labels_assigns_named_bins() -> None:
    """Custom bin edges with matching custom_labels must assign the given names."""
    df = _series_0_to_9()
    params: Dict[str, Any] = {
        "bin_edges": {"x": [0, 5, 9]},
        "custom_labels": {"x": ["low", "high"]},
        "output_suffix": "_binned",
        "drop_original": False,
        "label_format": "ordinal",
        "missing_strategy": "keep",
        "missing_label": "Missing",
        "include_lowest": True,
        "precision": 3,
    }
    result = GeneralBinningApplier().apply(df, params)
    # Values 0-4 -> "low", values 5-9 -> "high" (right-closed bins).
    assert result["x_binned"].iloc[0] == "low"
    assert result["x_binned"].iloc[9] == "high"


def test_apply_drop_original_removes_source_column() -> None:
    """drop_original=True must remove the source column from the output."""
    df = _series_0_to_9()
    config = {"columns": ["x"], "strategy": "equal_width", "n_bins": 5, "drop_original": True}
    params = GeneralBinningCalculator().fit(df, config)
    result = GeneralBinningApplier().apply(df, params)
    assert "x" not in result.columns
    assert "x_binned" in result.columns


def test_apply_output_suffix_is_configurable() -> None:
    """A custom output_suffix must be honoured for the new binned column name."""
    df = _series_0_to_9()
    config = {
        "columns": ["x"],
        "strategy": "equal_width",
        "n_bins": 5,
        "output_suffix": "_bucket",
    }
    params = GeneralBinningCalculator().fit(df, config)
    result = GeneralBinningApplier().apply(df, params)
    assert "x_bucket" in result.columns


# ---------------------------------------------------------------------------
# Apply — edge cases
# ---------------------------------------------------------------------------


def test_apply_out_of_range_values_become_nan() -> None:
    """Values outside the fitted edges must become NaN at apply time (missing_strategy='keep')."""
    df = pd.DataFrame({"x": [-5.0, 2.0, 7.0, 15.0]})
    params: Dict[str, Any] = {
        "bin_edges": {"x": [0.0, 5.0, 10.0]},
        "output_suffix": "_binned",
        "drop_original": False,
        "label_format": "ordinal",
        "missing_strategy": "keep",
        "missing_label": "Missing",
        "include_lowest": True,
        "precision": 3,
    }
    result = GeneralBinningApplier().apply(df, params)
    assert np.isnan(result["x_binned"].iloc[0])  # -5 below min
    assert np.isnan(result["x_binned"].iloc[3])  # 15 above max
    assert result["x_binned"].iloc[1] == 0.0
    assert result["x_binned"].iloc[2] == 1.0


def test_apply_missing_strategy_label_tags_out_of_range() -> None:
    """missing_strategy='label' must tag out-of-range/NaN values with missing_label."""
    df = pd.DataFrame({"x": [-5.0, 2.0]})
    params: Dict[str, Any] = {
        "bin_edges": {"x": [0.0, 5.0, 10.0]},
        "output_suffix": "_binned",
        "drop_original": False,
        "label_format": "ordinal",
        "missing_strategy": "label",
        "missing_label": "OOR",
        "include_lowest": True,
        "precision": 3,
    }
    result = GeneralBinningApplier().apply(df, params)
    assert result["x_binned"].iloc[0] == "OOR"


def test_apply_single_unique_value_column_assigns_one_bin() -> None:
    """A constant column must still bin consistently (all rows to the same bin)."""
    df = pd.DataFrame({"x": [5.0] * 10})
    calc = GeneralBinningCalculator()
    applier = GeneralBinningApplier()
    params = calc.fit(df, {"columns": ["x"], "strategy": "equal_width", "n_bins": 5})
    result = applier.apply(df, params)
    assert result["x_binned"].nunique() == 1


def test_apply_empty_bin_edges_is_noop() -> None:
    """Empty bin_edges (e.g. from no_columns fit) must leave the frame unchanged."""
    df = _series_0_to_9()
    result = GeneralBinningApplier().apply(df, {})
    pd.testing.assert_frame_equal(result, df)


def test_apply_empty_dataframe() -> None:
    """Applying to an empty DataFrame must not raise."""
    df = pd.DataFrame({"x": pd.Series([], dtype=float)})
    params: Dict[str, Any] = {
        "bin_edges": {"x": [0.0, 5.0, 10.0]},
        "output_suffix": "_binned",
        "drop_original": False,
        "label_format": "ordinal",
        "missing_strategy": "keep",
        "missing_label": "Missing",
        "include_lowest": True,
        "precision": 3,
    }
    result = GeneralBinningApplier().apply(df, params)
    assert result.shape[0] == 0
    assert "x_binned" in result.columns


def test_apply_column_missing_from_frame_is_skipped() -> None:
    """A bin_edges entry for a column absent from the frame must be silently skipped."""
    df = pd.DataFrame({"y": [1, 2, 3]})
    params: Dict[str, Any] = {
        "bin_edges": {"x": [0.0, 5.0, 10.0]},
        "output_suffix": "_binned",
        "drop_original": False,
        "label_format": "ordinal",
        "missing_strategy": "keep",
        "missing_label": "Missing",
        "include_lowest": True,
        "precision": 3,
    }
    result = GeneralBinningApplier().apply(df, params)
    assert list(result.columns) == ["y"]


# ---------------------------------------------------------------------------
# Apply — polars engine parity
# ---------------------------------------------------------------------------


def test_apply_polars_ordinal_matches_pandas_bin_assignment() -> None:
    """The polars apply path must assign the same ordinal bin indices as pandas."""
    df_pd = _series_0_to_9()
    df_pl = pl.from_pandas(df_pd)
    calc = GeneralBinningCalculator()
    applier = GeneralBinningApplier()
    params = calc.fit(df_pd, {"columns": ["x"], "strategy": "equal_width", "n_bins": 5})
    result_pd = applier.apply(df_pd, params)
    result_pl = applier.apply(df_pl, params)
    if hasattr(result_pl, "to_pandas"):
        result_pl = result_pl.to_pandas()
    assert result_pl["x_binned"].tolist() == result_pd["x_binned"].tolist()


def test_apply_polars_drop_original_removes_column() -> None:
    """The polars apply path must honour drop_original like the pandas path."""
    df_pl = pl.DataFrame({"x": list(range(10))})
    config = {"columns": ["x"], "strategy": "equal_width", "n_bins": 5, "drop_original": True}
    params = GeneralBinningCalculator().fit(pd.DataFrame({"x": list(range(10))}), config)
    result = GeneralBinningApplier().apply(df_pl, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert "x" not in result.columns


def test_apply_polars_empty_bin_edges_is_noop() -> None:
    """Empty bin_edges on the polars path must leave the frame unchanged."""
    df_pl = pl.DataFrame({"x": [1, 2, 3]})
    result = GeneralBinningApplier().apply(df_pl, {})
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["x"].to_list() == [1, 2, 3]


# ---------------------------------------------------------------------------
# fit -> apply round trip
# ---------------------------------------------------------------------------


def test_fit_then_apply_round_trip_custom_binning() -> None:
    """CustomBinningCalculator.fit + CustomBinningApplier.apply round trip end-to-end."""
    df = _series_0_to_9()
    calc = CustomBinningCalculator()
    applier = CustomBinningApplier()
    params = calc.fit(df, {"columns": ["x"], "bins": [0, 3, 6, 9]})
    result = applier.apply(df, params)
    expected = pd.cut(df["x"], bins=[0, 3, 6, 9], labels=False, include_lowest=True)
    assert result["x_binned"].tolist() == expected.tolist()


def test_fit_then_apply_round_trip_kbins_discretizer() -> None:
    """KBinsDiscretizerCalculator.fit + GeneralBinningApplier.apply round trip end-to-end."""
    df = _series_0_to_9()
    calc = KBinsDiscretizerCalculator()
    applier = GeneralBinningApplier()
    params = calc.fit(df, {"columns": ["x"], "n_bins": 3, "strategy": "uniform"})
    result = applier.apply(df, params)
    assert result["x_binned"].nunique() == 3


# ---------------------------------------------------------------------------
# Polars — degenerate edges / missing columns / non-ordinal label format
# ---------------------------------------------------------------------------


def test_apply_polars_degenerate_edges_produce_no_expr() -> None:
    """A column whose sorted-unique edges collapse to <2 must be skipped (no cut expr)."""
    df_pl = pl.DataFrame({"x": [1, 2, 3]})
    params: Dict[str, Any] = {
        "bin_edges": {"x": [5.0, 5.0]},
        "output_suffix": "_binned",
        "drop_original": False,
        "label_format": "ordinal",
        "custom_labels": {},
    }
    result = GeneralBinningApplier().apply(df_pl, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    # No cut expr was built, so no new column should appear.
    assert "x_binned" not in result.columns


def test_apply_polars_missing_column_is_skipped() -> None:
    """A bin_edges entry for a column absent from the polars frame must be skipped."""
    df_pl = pl.DataFrame({"y": [1, 2, 3]})
    params: Dict[str, Any] = {
        "bin_edges": {"x": [0.0, 5.0, 10.0]},
        "output_suffix": "_binned",
        "drop_original": False,
        "label_format": "ordinal",
        "custom_labels": {},
    }
    result = GeneralBinningApplier().apply(df_pl, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert list(result.columns) == ["y"]


def test_apply_polars_range_label_format_keeps_categorical_alias() -> None:
    """label_format='range' on the polars path must not cast to UInt32 (keeps labels)."""
    df_pl = pl.DataFrame({"x": list(range(10))})
    config = {"columns": ["x"], "strategy": "equal_width", "n_bins": 5, "label_format": "range"}
    params = GeneralBinningCalculator().fit(pd.DataFrame({"x": list(range(10))}), config)
    result = GeneralBinningApplier().apply(df_pl, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert "x_binned" in result.columns
    # Should be categorical/string-like labels, not raw integer codes.
    assert not pd.api.types.is_integer_dtype(result["x_binned"])


# ---------------------------------------------------------------------------
# Pandas — missing-label formatting on categorical (range) output
# ---------------------------------------------------------------------------


def test_apply_range_format_with_missing_label_tags_out_of_range_category() -> None:
    """range format + missing_strategy='label' must format the added string category as-is."""
    df = pd.DataFrame({"x": [-5.0, 2.0, 7.0]})
    params: Dict[str, Any] = {
        "bin_edges": {"x": [0.0, 5.0, 10.0]},
        "output_suffix": "_binned",
        "drop_original": False,
        "label_format": "range",
        "missing_strategy": "label",
        "missing_label": "OOR",
        "include_lowest": True,
        "precision": 3,
    }
    result = GeneralBinningApplier().apply(df, params)
    assert result["x_binned"].iloc[0] == "OOR"
    assert result["x_binned"].iloc[1].startswith("[")


# ---------------------------------------------------------------------------
# Pandas — degenerate edges raise + are swallowed by the apply loop
# ---------------------------------------------------------------------------


def test_apply_pandas_degenerate_edges_column_is_skipped() -> None:
    """A column with <2 unique edges must raise internally and be skipped, not crash apply."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    params: Dict[str, Any] = {
        "bin_edges": {"x": [5.0, 5.0]},
        "output_suffix": "_binned",
        "drop_original": False,
        "label_format": "ordinal",
        "missing_strategy": "keep",
        "missing_label": "Missing",
        "include_lowest": True,
        "precision": 3,
        "custom_labels": {},
    }
    result = GeneralBinningApplier().apply(df, params)
    assert "x_binned" not in result.columns
    assert "x" in result.columns


# ---------------------------------------------------------------------------
# Fit — equal_frequency edge clamp (defensive: forces qcut to underflow the min)
# ---------------------------------------------------------------------------


def test_fit_equal_frequency_edge_clamped_to_series_min(monkeypatch) -> None:
    """If qcut ever returns an edge below the true series min, it must be clamped."""
    import skyulf.preprocessing.bucketing as bucketing_mod

    original_qcut = bucketing_mod.pd.qcut

    def fake_qcut(series, q, labels=None, retbins=True, duplicates="drop"):
        binned, edges = original_qcut(
            series, q=q, labels=labels, retbins=retbins, duplicates=duplicates
        )
        edges = edges.copy()
        edges[0] = edges[0] - 1.0
        return binned, edges

    monkeypatch.setattr(bucketing_mod.pd, "qcut", fake_qcut)
    series = pd.Series(range(10), dtype=float)
    edges = _fit_equal_frequency(series, n_bins=4, duplicates="drop")
    assert edges[0] == series.min()


# ---------------------------------------------------------------------------
# Fit — kbins quantile strategy (quantile_method kwarg)
# ---------------------------------------------------------------------------


def test_fit_kbins_default_quantile_strategy_uses_quantile_method_kwarg() -> None:
    """KBinsDiscretizerCalculator's default 'quantile' strategy must fit without error."""
    df = _series_0_to_9()
    params = KBinsDiscretizerCalculator().fit(df, {"columns": ["x"], "n_bins": 3})
    assert len(params["bin_edges"]["x"]) == 4


# ---------------------------------------------------------------------------
# Fit — custom strategy via column_strategies / global strategy
# ---------------------------------------------------------------------------


def test_fit_general_binning_custom_strategy_uses_custom_bins_and_labels() -> None:
    """strategy='custom' must resolve edges/labels from custom_bins/custom_labels config."""
    df = _series_0_to_9()
    config = {
        "columns": ["x"],
        "strategy": "custom",
        "custom_bins": {"x": [0, 5, 9]},
        "custom_labels": {"x": ["low", "high"]},
    }
    params = GeneralBinningCalculator().fit(df, config)
    assert params["bin_edges"]["x"] == [0, 5, 9]
    assert params["custom_labels"]["x"] == ["low", "high"]


def test_fit_general_binning_column_strategy_override_custom() -> None:
    """Per-column strategy override to 'custom' must take precedence over the global strategy."""
    df = _series_0_to_9()
    config = {
        "columns": ["x"],
        "strategy": "equal_width",
        "column_strategies": {
            "x": {"strategy": "custom", "custom_bins": [0, 4, 9]},
        },
    }
    params = GeneralBinningCalculator().fit(df, config)
    assert params["bin_edges"]["x"] == [0, 4, 9]


# ---------------------------------------------------------------------------
# Fit — all-NaN column is skipped
# ---------------------------------------------------------------------------


def test_fit_general_binning_all_nan_column_is_skipped() -> None:
    """A column that is entirely NaN must be skipped (empty series after dropna)."""
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    params = GeneralBinningCalculator().fit(
        df, {"columns": ["x"], "strategy": "equal_width", "n_bins": 5}
    )
    assert "x" not in params["bin_edges"]


# ---------------------------------------------------------------------------
# Fit — errors inside a per-column fit are swallowed
# ---------------------------------------------------------------------------


def test_fit_one_column_into_maps_swallows_sklearn_errors() -> None:
    """An invalid n_bins that makes sklearn raise must be caught, leaving the maps untouched."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    bin_edges_map: Dict[str, Any] = {}
    custom_labels_map: Dict[str, Any] = {}
    _fit_one_column_into_maps(
        df,
        "x",
        {"strategy": "kmeans"},
        {"default_n_bins": 1, "n_bins": 1, "q_bins": 1, "duplicates": "drop"},
        bin_edges_map,
        custom_labels_map,
    )
    assert "x" not in bin_edges_map


# ---------------------------------------------------------------------------
# Fit — CustomBinningCalculator no-columns short circuit
# ---------------------------------------------------------------------------


def test_fit_custom_binning_no_columns_returns_empty_artifact() -> None:
    """An explicit empty `columns` list must short-circuit CustomBinningCalculator too."""
    df = _series_0_to_9()
    params = CustomBinningCalculator().fit(df, {"columns": []})
    assert params == {}


def test_fit_general_binning_unknown_strategy_yields_no_edges() -> None:
    """An unrecognised strategy name must produce no edges/labels for that column."""
    df = _series_0_to_9()
    params = GeneralBinningCalculator().fit(df, {"columns": ["x"], "strategy": "not_a_strategy"})
    assert "x" not in params["bin_edges"]
    assert "x" not in params["custom_labels"]


# ---------------------------------------------------------------------------
# Real-shaped dataset integration
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing values in ``age``/``income`` — closer to production data
    than the small synthetic frames used elsewhere in this file.
    """

    def test_bucketing_age_with_nan_preserves_missing_rows(self) -> None:
        """Bucketing a numeric column containing NaN must produce a binned output
        column where NaN rows remain NaN (missing_strategy='keep' default).
        """
        df = load_sample_dataset("customers")
        calc = GeneralBinningCalculator()
        applier = GeneralBinningApplier()
        params = calc.fit(df, {"columns": ["age"], "strategy": "equal_width", "n_bins": 3})
        result = applier.apply(df, params)

        assert "age_binned" in result.columns
        # Rows where age was NaN must still be NaN in the binned column.
        nan_mask = df["age"].isna()
        assert result.loc[nan_mask, "age_binned"].isna().all()
        # Non-missing rows must have a valid bin index assigned.
        assert result.loc[~nan_mask, "age_binned"].notna().all()
