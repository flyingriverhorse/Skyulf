"""Tests for ManualBoundsCalculator and ManualBoundsApplier.

Covers:
- fit returns the passed bounds verbatim
- pandas apply: rows outside [lower, upper] are removed
- polars apply: same filtering, including boundary equality
- NaN values are treated as inliers (never removed)
- No bounds config → identity passthrough
- Columns absent from the dataset are silently ignored
- Tuple (X, y) inputs: y is filtered in sync with X
"""

import typing

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.outliers.manual_bounds import (
    ManualBoundsApplier,
    ManualBoundsCalculator,
    _manual_bounds_col_mask_pandas,
    _manual_bounds_col_mask_polars,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def df_pandas() -> pd.DataFrame:
    """Deterministic DataFrame with values spanning a wide range."""
    return pd.DataFrame(
        {
            "val": [-10.0, 0.0, 5.0, 10.0, 20.0, 100.0],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


@pytest.fixture()
def df_polars(df_pandas: pd.DataFrame) -> pl.DataFrame:
    """Polars version of the same frame."""
    return pl.from_pandas(df_pandas)


# ---------------------------------------------------------------------------
# Calculator (fit)
# ---------------------------------------------------------------------------


class TestManualBoundsCalculator:
    def test_fit_stores_bounds(self) -> None:
        """fit must echo the bounds config into the artifact."""
        config = {"bounds": {"val": {"lower": 0.0, "upper": 50.0}}}
        params = ManualBoundsCalculator().fit(pd.DataFrame({"val": [1, 2, 3]}), config)
        assert params["type"] == "manual_bounds"
        assert params["bounds"]["val"]["lower"] == 0.0
        assert params["bounds"]["val"]["upper"] == 50.0

    def test_fit_empty_bounds_returns_empty_dict(self) -> None:
        """No bounds config → empty bounds artifact."""
        params = ManualBoundsCalculator().fit(pd.DataFrame({"val": [1, 2]}), {})
        assert params["bounds"] == {}

    def test_fit_ignores_data_values(self) -> None:
        """fit only passes bounds through; it never inspects the data."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        config = {"bounds": {"x": {"lower": -999.0}}}
        params = ManualBoundsCalculator().fit(df, config)
        # Regardless of actual data, lower bound is exactly what was configured.
        assert params["bounds"]["x"]["lower"] == -999.0


# ---------------------------------------------------------------------------
# Low-level mask helpers
# ---------------------------------------------------------------------------


class TestManualBoundsColMaskPandas:
    def test_lower_bound_only(self) -> None:
        """Values below lower must be False; values at or above are True."""
        s = pd.Series([-1.0, 0.0, 1.0, 5.0])
        mask = _manual_bounds_col_mask_pandas(s, {"lower": 0.0})
        assert mask.tolist() == [False, True, True, True]

    def test_upper_bound_only(self) -> None:
        """Values above upper must be False; boundary value is True."""
        s = pd.Series([0.0, 5.0, 10.0, 11.0])
        mask = _manual_bounds_col_mask_pandas(s, {"upper": 10.0})
        assert mask.tolist() == [True, True, True, False]

    def test_lower_and_upper(self) -> None:
        """Values outside [lower, upper] are False; boundary values are True."""
        s = pd.Series([-1.0, 0.0, 5.0, 10.0, 11.0])
        mask = _manual_bounds_col_mask_pandas(s, {"lower": 0.0, "upper": 10.0})
        assert mask.tolist() == [False, True, True, True, False]

    def test_nan_always_inlier(self) -> None:
        """NaN values must always pass the mask — we never remove missings."""
        s = pd.Series([float("nan"), 5.0])
        mask = _manual_bounds_col_mask_pandas(s, {"lower": 0.0, "upper": 1.0})
        assert mask.iloc[0] is True or bool(mask.iloc[0]) is True
        assert bool(mask.iloc[1]) is False

    def test_no_bounds_all_true(self) -> None:
        """Empty bounds dict means no restriction — all rows pass."""
        s = pd.Series([-1000.0, 0.0, 1000.0])
        mask = _manual_bounds_col_mask_pandas(s, {})
        assert mask.all()


class TestManualBoundsColMaskPolars:
    def test_lower_bound_polars(self) -> None:
        """Values below lower must be excluded by the Polars mask."""
        df = pl.DataFrame({"v": [-1.0, 0.0, 1.0]})
        mask_expr = _manual_bounds_col_mask_polars("v", {"lower": 0.0})
        result = df.select(mask_expr.alias("m"))["m"].to_list()
        assert result == [False, True, True]

    def test_boundary_value_included_polars(self) -> None:
        """The value exactly equal to upper should be kept (<=, not <)."""
        df = pl.DataFrame({"v": [9.9, 10.0, 10.1]})
        mask_expr = _manual_bounds_col_mask_polars("v", {"upper": 10.0})
        result = df.select(mask_expr.alias("m"))["m"].to_list()
        assert result == [True, True, False]

    def test_null_always_inlier_polars(self) -> None:
        """Nulls must pass through; they are not outliers by definition."""
        df = pl.DataFrame({"v": [None, 0.0, 999.0]})
        mask_expr = _manual_bounds_col_mask_polars("v", {"lower": 0.0, "upper": 10.0})
        result = df.select(mask_expr.alias("m"))["m"].to_list()
        assert result[0] is True  # null row kept
        assert result[1] is True  # 0.0 at boundary kept
        assert result[2] is False  # 999 > 10 removed


# ---------------------------------------------------------------------------
# ManualBoundsApplier — pandas engine
# ---------------------------------------------------------------------------


class TestManualBoundsApplierPandas:
    _APPLIER = ManualBoundsApplier()
    _CALC = ManualBoundsCalculator()

    def _run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Fit + apply helper."""
        params = self._CALC.fit(df, config)
        return self._APPLIER.apply(df, params)

    def test_rows_outside_range_removed(self, df_pandas: pd.DataFrame) -> None:
        """Rows with val outside [0, 50] must be filtered out."""
        out = self._run(df_pandas, {"bounds": {"val": {"lower": 0.0, "upper": 50.0}}})
        assert out["val"].min() >= 0.0
        assert out["val"].max() <= 50.0

    def test_boundary_values_kept(self, df_pandas: pd.DataFrame) -> None:
        """Boundary values 0.0 and 20.0 must survive the filter."""
        out = self._run(df_pandas, {"bounds": {"val": {"lower": 0.0, "upper": 20.0}}})
        assert 0.0 in out["val"].values
        assert 20.0 in out["val"].values

    def test_no_bounds_passthrough(self, df_pandas: pd.DataFrame) -> None:
        """Empty bounds → all rows preserved."""
        out = self._run(df_pandas, {})
        assert len(out) == len(df_pandas)

    def test_unknown_column_ignored(self, df_pandas: pd.DataFrame) -> None:
        """Bounds referencing a column that doesn't exist must not raise."""
        out = self._run(df_pandas, {"bounds": {"nonexistent": {"lower": 0.0}}})
        assert len(out) == len(df_pandas)

    def test_nan_rows_retained(self) -> None:
        """NaN values must be treated as inliers and never removed."""
        df = pd.DataFrame({"val": [float("nan"), 5.0, 200.0]})
        out = self._run(df, {"bounds": {"val": {"lower": 0.0, "upper": 100.0}}})
        # NaN row stays; 200.0 is removed.
        assert len(out) == 2
        assert pd.isna(out["val"].iloc[0])

    def test_tuple_xy_apply_returns_tuple(self) -> None:
        """apply on a (X, y) tuple input must return a (X_out, y_out) tuple."""
        X = pd.DataFrame({"val": [1.0, 50.0, 200.0]})
        y = pd.Series([0, 1, 2])
        config = {"bounds": {"val": {"upper": 100.0}}}
        params = self._CALC.fit(X, config)
        result = self._APPLIER.apply((X, y), params)
        assert isinstance(result, tuple)
        X_out, _ = result
        # Rows outside the bound must be removed from X.
        assert 200.0 not in X_out["val"].values

    def test_tuple_xy_apply_keeps_y_in_sync_with_filtered_x(self) -> None:
        """Regression: apply's public entry point must not drop y before it
        reaches apply_dual_engine's own unpacking — X and y must stay the
        same length and the surviving y values must match the surviving rows.
        """
        X = pd.DataFrame({"val": [1.0, 50.0, 200.0]})
        y = pd.Series([10, 20, 30])
        config = {"bounds": {"val": {"upper": 100.0}}}
        params = self._CALC.fit(X, config)
        X_out, y_out = self._APPLIER.apply((X, y), params)
        assert len(X_out) == len(y_out)
        assert 30 not in y_out.values
        assert list(y_out.values) == [10, 20]

    def test_apply_pandas_direct_y_filtered_in_sync(self) -> None:
        """_apply_pandas directly must synchronise y when called with a non-null y."""
        X = pd.DataFrame({"val": [1.0, 50.0, 200.0]})
        y = pd.Series([0, 1, 2])
        params = {"bounds": {"val": {"upper": 100.0}}}
        from skyulf.preprocessing.outliers.manual_bounds import ManualBoundsApplier

        X_out, y_out = ManualBoundsApplier._apply_pandas(X, y, params)
        assert len(X_out) == len(y_out)
        assert 2 not in y_out.values

    def test_multiple_column_bounds_intersection(self) -> None:
        """Rows must survive all column bounds simultaneously (AND logic)."""
        df = pd.DataFrame({"a": [1.0, 5.0, 10.0], "b": [100.0, 5.0, 3.0]})
        config = {"bounds": {"a": {"upper": 7.0}, "b": {"upper": 10.0}}}
        out = self._run(df, config)
        # Row 0: a=1 OK, b=100 > 10 → removed
        # Row 1: a=5 OK, b=5 OK → kept
        # Row 2: a=10 > 7 → removed
        assert len(out) == 1
        assert out["a"].iloc[0] == 5.0


# ---------------------------------------------------------------------------
# ManualBoundsApplier — polars engine
# ---------------------------------------------------------------------------


class TestManualBoundsApplierPolars:
    _APPLIER = ManualBoundsApplier()
    _CALC = ManualBoundsCalculator()

    def _run(self, df: pl.DataFrame, config: dict) -> pl.DataFrame:
        """Fit on pandas equivalent, apply on polars frame."""
        params = self._CALC.fit(df.to_pandas(), config)
        return self._APPLIER.apply(df, params)

    def test_rows_outside_range_removed(self, df_polars: pl.DataFrame) -> None:
        """Polars engine must filter rows outside the configured range."""
        out = self._run(df_polars, {"bounds": {"val": {"lower": 0.0, "upper": 50.0}}})
        assert typing.cast(float, out["val"].min()) >= 0.0
        assert typing.cast(float, out["val"].max()) <= 50.0

    def test_boundary_equality_lower(self, df_polars: pl.DataFrame) -> None:
        """Value equal to lower must be kept (>= semantics)."""
        out = self._run(df_polars, {"bounds": {"val": {"lower": 0.0}}})
        assert 0.0 in out["val"].to_list()

    def test_boundary_equality_upper(self, df_polars: pl.DataFrame) -> None:
        """Value equal to upper must be kept (<= semantics)."""
        out = self._run(df_polars, {"bounds": {"val": {"upper": 100.0}}})
        assert 100.0 in out["val"].to_list()

    def test_no_bounds_passthrough(self, df_polars: pl.DataFrame) -> None:
        """Empty bounds → no rows removed by polars engine."""
        out = self._run(df_polars, {})
        assert out.shape[0] == df_polars.shape[0]

    def test_polars_parity_with_pandas(self, df_pandas: pd.DataFrame) -> None:
        """Polars and pandas engines must keep the same rows for identical bounds."""
        config = {"bounds": {"val": {"lower": 0.0, "upper": 50.0}}}
        params = ManualBoundsCalculator().fit(df_pandas, config)
        pd_out = ManualBoundsApplier().apply(df_pandas, params)
        pl_out = ManualBoundsApplier().apply(pl.from_pandas(df_pandas), params)
        assert list(pd_out["val"]) == pl_out["val"].to_list()

    def test_unknown_column_ignored_polars(self, df_polars: pl.DataFrame) -> None:
        """Bounds referencing a column absent from the Polars frame must not raise
        (covers the ``continue`` branch in ``_apply_polars``)."""
        out = self._run(df_polars, {"bounds": {"nonexistent": {"lower": 0.0}}})
        assert out.shape[0] == df_polars.shape[0]

    def test_polars_tuple_xy_with_polars_series_y(self, df_polars: pl.DataFrame) -> None:
        """A Polars Series ``y`` paired with a Polars X must be filtered in sync."""
        y = pl.Series("y", list(range(df_polars.shape[0])))
        config = {"bounds": {"val": {"lower": 0.0, "upper": 50.0}}}
        params = ManualBoundsCalculator().fit(df_polars.to_pandas(), config)
        X_out, y_out = self._APPLIER.apply((df_polars, y), params)
        assert isinstance(y_out, pl.Series)
        assert X_out.height == len(y_out)

    def test_polars_tuple_xy_with_polars_dataframe_y(self, df_polars: pl.DataFrame) -> None:
        """A Polars DataFrame ``y`` paired with a Polars X must be filtered in sync."""
        y = pl.DataFrame({"label": list(range(df_polars.shape[0]))})
        config = {"bounds": {"val": {"lower": 0.0, "upper": 50.0}}}
        params = ManualBoundsCalculator().fit(df_polars.to_pandas(), config)
        X_out, y_out = self._APPLIER.apply((df_polars, y), params)
        assert isinstance(y_out, pl.DataFrame)
        assert X_out.height == y_out.height

    def test_polars_tuple_xy_with_non_polars_y_passthrough(self, df_polars: pl.DataFrame) -> None:
        """A non-Polars, non-None ``y`` alongside a Polars X must be returned
        unchanged (the ``_filter_y_polars`` fallback branch)."""
        y = [0, 1, 2, 3, 4, 5]
        config = {"bounds": {"val": {"lower": 0.0, "upper": 50.0}}}
        params = ManualBoundsCalculator().fit(df_polars.to_pandas(), config)
        _, y_out = self._APPLIER.apply((df_polars, y), params)
        assert y_out is y
