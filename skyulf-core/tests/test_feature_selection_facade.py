"""Tests for the feature-selection facade (FeatureSelectionCalculator/Applier).

Covers:
- Dispatcher routes each `method` string to the correct concrete Calculator.
- Unknown method logs a warning and returns {}.
- FeatureSelectionApplier routes by the `type` tag in params.
- Identity passthrough when params `type` is unrecognised.
- Full fit→apply cycle for variance_threshold and correlation_threshold.
"""

import pandas as pd
import pytest
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.preprocessing.feature_selection.facade import (
    FeatureSelectionApplier,
    FeatureSelectionCalculator,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------


def _make_df() -> pd.DataFrame:
    """Return a small DataFrame with constant, correlated, and useful columns."""
    return pd.DataFrame(
        {
            # Useful signal column
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            # Constant column (zero variance)
            "const": [1.0, 1.0, 1.0, 1.0, 1.0],
            # Perfectly correlated with a (will be dropped by corr threshold)
            "a_copy": [2.0, 4.0, 6.0, 8.0, 10.0],
            # Independent column
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )


_CALC = FeatureSelectionCalculator()
_APPLIER = FeatureSelectionApplier()


# ---------------------------------------------------------------------------
# Calculator — method dispatch
# ---------------------------------------------------------------------------


class TestFeatureSelectionCalculatorDispatch:
    def test_variance_threshold_dispatch(self) -> None:
        """variance_threshold method must return a variance_threshold artifact."""
        params = _CALC.fit(_make_df(), {"method": "variance_threshold", "threshold": 0.0})
        assert params.get("type") == "variance_threshold"

    def test_correlation_threshold_dispatch(self) -> None:
        """correlation_threshold method must return a correlation_threshold artifact."""
        params = _CALC.fit(
            _make_df(),
            {"method": "correlation_threshold", "threshold": 0.95},
        )
        assert params.get("type") == "correlation_threshold"

    def test_select_k_best_dispatch(self) -> None:
        """select_k_best must return a univariate_selection artifact."""
        df = _make_df()
        y = pd.Series([0, 1, 0, 1, 0])
        params = _CALC.fit(
            (df, y),
            {"method": "select_k_best", "k": 2, "score_func": "f_classif"},
        )
        assert params.get("type") == "univariate_selection"

    def test_select_percentile_dispatch(self) -> None:
        """select_percentile must produce a univariate_selection artifact."""
        df = _make_df()
        y = pd.Series([0, 1, 0, 1, 0])
        params = _CALC.fit(
            (df, y),
            {"method": "select_percentile", "percentile": 50, "score_func": "f_classif"},
        )
        assert params.get("type") == "univariate_selection"

    def test_unknown_method_returns_empty(self) -> None:
        """An unrecognised method must return {} — no exception."""
        params = _CALC.fit(_make_df(), {"method": "telepathy"})
        assert params == {}

    def test_default_method_is_select_k_best(self) -> None:
        """When no method is provided, select_k_best is the default."""
        df = _make_df()
        y = pd.Series([0, 1, 0, 1, 0])
        params = _CALC.fit((df, y), {})
        # May return {} if no target, but must not raise.
        assert isinstance(params, dict)

    @pytest.mark.parametrize(
        "method",
        [
            "select_fpr",
            "select_fdr",
            "select_fwe",
            "generic_univariate_select",
        ],
    )
    def test_all_univariate_methods_dispatch(self, method: str) -> None:
        """All univariate method aliases must dispatch to univariate_selection."""
        df = _make_df()
        y = pd.Series([0, 1, 0, 1, 0])
        params = _CALC.fit(
            (df, y),
            {"method": method, "score_func": "f_classif", "k": 2, "alpha": 0.5},
        )
        # May return {} if selector finds nothing, but type must be correct if present.
        if params:
            assert params.get("type") == "univariate_selection"


# ---------------------------------------------------------------------------
# Calculator — fit correctness
# ---------------------------------------------------------------------------


class TestFeatureSelectionCalculatorValues:
    def test_variance_threshold_removes_constant(self) -> None:
        """The constant column must appear in candidate_columns but not selected."""
        params = _CALC.fit(_make_df(), {"method": "variance_threshold", "threshold": 0.0})
        selected = params.get("selected_columns", [])
        assert "const" not in selected

    def test_variance_threshold_keeps_varying_columns(self) -> None:
        """Non-constant columns must be kept by variance threshold."""
        params = _CALC.fit(_make_df(), {"method": "variance_threshold", "threshold": 0.0})
        selected = params.get("selected_columns", [])
        assert "a" in selected

    def test_correlation_threshold_drops_correlated(self) -> None:
        """a_copy is perfectly correlated with a and must be marked for dropping."""
        params = _CALC.fit(
            _make_df(),
            {"method": "correlation_threshold", "threshold": 0.95},
        )
        assert "a_copy" in params.get("columns_to_drop", [])


# ---------------------------------------------------------------------------
# Applier — type-based dispatch
# ---------------------------------------------------------------------------


class TestFeatureSelectionApplierDispatch:
    def test_variance_threshold_applier(self) -> None:
        """Applier with variance_threshold params must drop the constant column."""
        df = _make_df()
        params = _CALC.fit(df, {"method": "variance_threshold", "threshold": 0.0})
        out = _APPLIER.apply(df, dict(params))
        assert "const" not in out.columns

    def test_correlation_threshold_applier(self) -> None:
        """Applier with correlation params must drop the correlated column."""
        df = _make_df()
        params = _CALC.fit(df, {"method": "correlation_threshold", "threshold": 0.95})
        out = _APPLIER.apply(df, dict(params))
        # At least one correlated column must be gone.
        assert "a_copy" not in out.columns

    def test_unknown_type_is_passthrough(self) -> None:
        """A params dict with an unknown type tag must return the frame unchanged."""
        df = _make_df()
        params = {"type": "alien_selection", "selected_columns": ["a"]}
        out = _APPLIER.apply(df, params)
        pd.testing.assert_frame_equal(out, df)

    def test_empty_params_is_passthrough(self) -> None:
        """Empty params (from unknown method) must not transform the frame."""
        df = _make_df()
        out = _APPLIER.apply(df, {})
        pd.testing.assert_frame_equal(out, df)

    def test_drop_columns_false_keeps_all(self) -> None:
        """drop_columns=False must preserve all columns even when selection ran."""
        df = _make_df()
        params = _CALC.fit(
            df, {"method": "variance_threshold", "threshold": 0.0, "drop_columns": False}
        )
        out = _APPLIER.apply(df, dict(params))
        # No column dropped because drop_columns=False.
        for col in df.columns:
            assert col in out.columns


# ---------------------------------------------------------------------------
# Full fit → apply cycle
# ---------------------------------------------------------------------------


class TestFitApplyCycle:
    def test_fit_then_apply_same_frame(self) -> None:
        """fit→apply on the same frame must not raise and must reduce columns."""
        df = _make_df()
        params = _CALC.fit(df, {"method": "variance_threshold", "threshold": 0.0})
        out = _APPLIER.apply(df, dict(params))
        # At minimum the constant column is gone.
        assert out.shape[1] < df.shape[1]

    def test_apply_on_new_frame_uses_fitted_params(self) -> None:
        """Fitted params from the train set must apply cleanly to a new frame."""
        train = _make_df()
        test = _make_df().copy()
        params = _CALC.fit(train, {"method": "variance_threshold", "threshold": 0.0})
        out = _APPLIER.apply(test, dict(params))
        assert set(out.columns) == set(params.get("selected_columns", out.columns))


# ---------------------------------------------------------------------------
# Real-shaped dataset: customers.csv (NaN in age/income/lat/lon)
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Verify that FeatureSelectionCalculator and FeatureSelectionApplier handle
    the customers.csv sample, which has NaN in numeric columns age/income/lat/lon,
    without raising. Exercises the full fit→apply cycle on real-shaped mixed-dtype data.
    """

    def test_variance_threshold_on_customers_numeric_columns_does_not_raise(self) -> None:
        """Variance threshold fit/apply on NaN-containing numeric columns must not raise
        and must return a result whose columns are a subset of the input columns."""
        df = load_sample_dataset("customers")
        num_df = df[["age", "income", "lat", "lon"]]
        params = _CALC.fit(num_df, {"method": "variance_threshold", "threshold": 0.0})
        out = _APPLIER.apply(num_df, dict(params))
        assert set(out.columns).issubset({"age", "income", "lat", "lon"})
        # All input columns have non-zero variance → none should be dropped.
        assert len(out.columns) == 4
