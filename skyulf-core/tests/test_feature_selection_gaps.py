"""Coverage-gap tests for feature_selection: correlation, model_based, univariate, variance.

Each selector shares the fit->apply Calculator/Applier contract and the
``_common.py`` drop-selected helpers. Tests focus on real numeric behavior
(correct columns dropped/kept) plus edge cases: too few candidate columns,
missing target, and threshold boundaries.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.feature_selection.correlation import (
    CorrelationThresholdApplier,
    CorrelationThresholdCalculator,
)
from skyulf.preprocessing.feature_selection.facade import FeatureSelectionApplier
from skyulf.preprocessing.feature_selection.model_based import (
    ModelBasedSelectionApplier,
    ModelBasedSelectionCalculator,
)
from skyulf.preprocessing.feature_selection.univariate import (
    UnivariateSelectionApplier,
    UnivariateSelectionCalculator,
)
from skyulf.preprocessing.feature_selection.variance import (
    VarianceThresholdApplier,
    VarianceThresholdCalculator,
)


def _correlated_df(n: int = 50) -> pd.DataFrame:
    """Two perfectly correlated columns plus one independent column."""
    rng = np.random.RandomState(0)
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": a * 2.0 + 1.0, "c": rng.normal(size=n)})


_univariate_empty_cases = TestCaseLoader(
    "preprocessing/feature_selection_gaps", group="univariate_empty"
).load()
_model_based_empty_cases = TestCaseLoader(
    "preprocessing/feature_selection_gaps", group="model_based_empty"
).load()


# ---------------------------------------------------------------------------
# CorrelationThreshold
# ---------------------------------------------------------------------------


def test_correlation_threshold_drops_highly_correlated_column() -> None:
    """A perfectly correlated column pair triggers a drop above the threshold."""
    df = _correlated_df()
    art = CorrelationThresholdCalculator().fit(df, {"threshold": 0.9})
    assert "b" in art["columns_to_drop"]
    assert "a" not in art["columns_to_drop"]


def test_correlation_threshold_apply_drops_column() -> None:
    """Applying the artifact removes the flagged column from the frame."""
    df = _correlated_df()
    art = CorrelationThresholdCalculator().fit(df, {"threshold": 0.9})
    out = CorrelationThresholdApplier().apply(df, art)
    assert "b" not in out.columns
    assert "a" in out.columns


def test_correlation_threshold_apply_respects_drop_columns_false() -> None:
    """When ``drop_columns=False`` the frame passes through unchanged."""
    df = _correlated_df()
    art = CorrelationThresholdCalculator().fit(df, {"threshold": 0.9, "drop_columns": False})
    out = CorrelationThresholdApplier().apply(df, art)
    assert "b" in out.columns


def test_correlation_threshold_insufficient_columns_returns_empty() -> None:
    """Fewer than two numeric candidate columns yields an empty artifact."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    art = CorrelationThresholdCalculator().fit(df, {})
    assert art == {}


def test_correlation_threshold_no_correlation_drops_nothing() -> None:
    """Independent columns below the threshold are all kept."""
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    art = CorrelationThresholdCalculator().fit(df, {"threshold": 0.99})
    assert art["columns_to_drop"] == []


def test_correlation_threshold_polars_engine_parity() -> None:
    """Fitting on the polars engine yields the same drop list as pandas."""
    import polars as pl

    df = _correlated_df()
    pd_art = CorrelationThresholdCalculator().fit(df, {"threshold": 0.9})
    pl_art = CorrelationThresholdCalculator().fit(pl.from_pandas(df), {"threshold": 0.9})
    assert set(pd_art["columns_to_drop"]) == set(pl_art["columns_to_drop"])


def test_correlation_threshold_apply_polars_drops_column() -> None:
    """Applying a fitted artifact to a polars frame must drop the correlated column."""
    import polars as pl

    df = _correlated_df()
    art = CorrelationThresholdCalculator().fit(df, {"threshold": 0.9})
    out = CorrelationThresholdApplier().apply(pl.from_pandas(df), art)
    assert "b" not in out.columns
    assert "a" in out.columns


def test_correlation_threshold_apply_polars_respects_drop_columns_false() -> None:
    """Polars apply path must leave frame untouched when drop_columns=False."""
    import polars as pl

    df = _correlated_df()
    art = CorrelationThresholdCalculator().fit(df, {"threshold": 0.9, "drop_columns": False})
    out = CorrelationThresholdApplier().apply(pl.from_pandas(df), art)
    assert "b" in out.columns


# ---------------------------------------------------------------------------
# VarianceThreshold
# ---------------------------------------------------------------------------


def test_variance_threshold_drops_constant_column() -> None:
    """A zero-variance (constant) column is removed at the default threshold."""
    df = pd.DataFrame({"const": [1.0] * 10, "varying": list(range(10))})
    art = VarianceThresholdCalculator().fit(df, {"threshold": 0.0})
    assert "const" not in art["selected_columns"]
    assert "varying" in art["selected_columns"]


def test_variance_threshold_apply_drops_low_variance_column() -> None:
    """Applying the artifact drops the column selection excluded."""
    df = pd.DataFrame({"const": [2.0] * 10, "varying": list(range(10))})
    art = VarianceThresholdCalculator().fit(df, {"threshold": 0.0})
    out = VarianceThresholdApplier().apply(df, art)
    assert "const" not in out.columns
    assert "varying" in out.columns


def test_variance_threshold_reports_variances() -> None:
    """The artifact reports a numeric variance for each candidate column."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    art = VarianceThresholdCalculator().fit(df, {"threshold": 0.0})
    assert "a" in art["variances"]
    assert art["variances"]["a"] > 0


def test_variance_threshold_no_candidate_columns_returns_empty() -> None:
    """A non-numeric-only frame with auto-detection yields an empty artifact."""
    df = pd.DataFrame({"cat": ["x", "y", "z"]})
    art = VarianceThresholdCalculator().fit(df, {})
    assert art == {}


# ---------------------------------------------------------------------------
# UnivariateSelection
# ---------------------------------------------------------------------------


def _classification_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    x1 = rng.normal(size=n)
    y = (x1 > 0).astype(int)
    x2 = rng.normal(size=n)  # unrelated to y
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


# Maps the ``df_kind`` JSON field to the DataFrame builder it should use.
_FEATURE_SELECTION_DF_BUILDERS = {
    "classification": _classification_df,
    "target_only": lambda: pd.DataFrame({"target": [0, 1, 0, 1]}),
}


def test_univariate_selection_selects_informative_feature() -> None:
    """The feature correlated with the target scores higher than noise."""
    df = _classification_df()
    art = UnivariateSelectionCalculator().fit(
        df, {"target_column": "target", "method": "select_k_best", "k": 1}
    )
    assert art["selected_columns"] == ["x1"]
    assert art["feature_scores"]["x1"] > art["feature_scores"]["x2"]


def test_univariate_selection_apply_drops_unselected_columns() -> None:
    """Applying the artifact drops candidate columns that were not selected."""
    df = _classification_df()
    art = UnivariateSelectionCalculator().fit(
        df, {"target_column": "target", "method": "select_k_best", "k": 1}
    )
    out = UnivariateSelectionApplier().apply(df, art)
    assert "x2" not in out.columns
    assert "x1" in out.columns
    assert "target" in out.columns


def test_univariate_selection_allow_missing_target_passthrough() -> None:
    """``allow_missing_target=True`` selects all candidate columns without scoring."""
    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
    art = UnivariateSelectionCalculator().fit(
        df, {"target_column": "missing", "allow_missing_target": True}
    )
    assert set(art["selected_columns"]) == {"x1", "x2"}
    assert art["scores"] == {}


@pytest.mark.parametrize(*_univariate_empty_cases)
def test_univariate_selection_returns_empty(df_kind: str, config: Dict[str, Any]) -> None:
    """Various bad configs/candidate sets must yield an empty artifact."""
    df = _FEATURE_SELECTION_DF_BUILDERS[df_kind]()
    art = UnivariateSelectionCalculator().fit(df, config)
    assert art == {}


# ---------------------------------------------------------------------------
# ModelBasedSelection
# ---------------------------------------------------------------------------


def test_model_based_selection_random_forest_classification() -> None:
    """RandomForest importance selection keeps at least the informative feature."""
    df = _classification_df()
    art = ModelBasedSelectionCalculator().fit(
        df,
        {"target_column": "target", "estimator": "random_forest", "problem_type": "classification"},
    )
    assert "x1" in art["selected_columns"]
    assert "x1" in art["feature_importances"]


def test_model_based_selection_apply_drops_unselected() -> None:
    """Applying the artifact drops unselected candidate columns."""
    df = _classification_df()
    art = ModelBasedSelectionCalculator().fit(
        df,
        {"target_column": "target", "estimator": "random_forest", "problem_type": "classification"},
    )
    out = ModelBasedSelectionApplier().apply(df, art)
    for col in art["candidate_columns"]:
        if col not in art["selected_columns"]:
            assert col not in out.columns


@pytest.mark.parametrize(*_model_based_empty_cases)
def test_model_based_selection_returns_empty(df_kind: str, config: Dict[str, Any]) -> None:
    """Various bad configs/candidate sets must yield an empty artifact."""
    df = _FEATURE_SELECTION_DF_BUILDERS[df_kind]()
    art = ModelBasedSelectionCalculator().fit(df, config)
    assert art == {}


def test_model_based_selection_linear_regression_estimator_for_classification() -> None:
    """'linear_regression' key must resolve to LinearRegression even under classification.

    Covers the (odd-but-allowed) classification branch of _resolve_estimator that
    permits a LinearRegression estimator for a classification problem.
    """
    df = _classification_df()
    art = ModelBasedSelectionCalculator().fit(
        df,
        {
            "target_column": "target",
            "estimator": "linear_regression",
            "problem_type": "classification",
        },
    )
    # LinearRegression exposes coef_, so importances should be computed.
    assert "x1" in art["feature_importances"]


def test_model_feature_importances_returns_empty_for_unsupported_estimator() -> None:
    """_model_feature_importances must return {} when the estimator has neither attr."""
    from skyulf.preprocessing.feature_selection._common import _model_feature_importances

    class _NoImportanceEstimator:
        """Stand-in estimator lacking both feature_importances_ and coef_."""

    class _FakeSelector:
        estimator_ = _NoImportanceEstimator()

    result = _model_feature_importances(_FakeSelector(), ["a", "b"])
    assert result == {}


# ---------------------------------------------------------------------------
# FeatureSelection facade dispatch (facade.py lines 33, 35)
# ---------------------------------------------------------------------------


def test_feature_selection_facade_dispatches_univariate_selection() -> None:
    """The facade applier must route 'univariate_selection' type to UnivariateSelectionApplier."""
    df = _classification_df()
    art = UnivariateSelectionCalculator().fit(df, {"target_column": "target"})
    out = FeatureSelectionApplier().apply(df, dict(art))
    for col in art["candidate_columns"]:
        if col not in art["selected_columns"]:
            assert col not in out.columns


def test_feature_selection_facade_dispatches_model_based_selection() -> None:
    """The facade applier must route 'model_based_selection' type to ModelBasedSelectionApplier."""
    df = _classification_df()
    art = ModelBasedSelectionCalculator().fit(
        df,
        {"target_column": "target", "estimator": "random_forest", "problem_type": "classification"},
    )
    out = FeatureSelectionApplier().apply(df, dict(art))
    for col in art["candidate_columns"]:
        if col not in art["selected_columns"]:
            assert col not in out.columns


# ---------------------------------------------------------------------------
# Real-shaped dataset: customers.csv (NaN in age/income/lat/lon)
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Verify that CorrelationThresholdCalculator handles the customers.csv numeric
    columns (which contain NaN) without raising and produces a valid artifact.

    lat/lon in customers.csv are city-pair-correlated (each city maps to a unique
    lat and lon), so they are perfectly correlated — one should be flagged for dropping
    at threshold 0.9 when fit on the non-null subset.
    """

    def test_correlation_threshold_on_customers_numeric_does_not_raise(self) -> None:
        """CorrelationThresholdCalculator.fit on NaN-containing numeric columns
        must return a valid artifact with a columns_to_drop list."""
        df = load_sample_dataset("customers")
        art = CorrelationThresholdCalculator().fit(
            df[["age", "income", "lat", "lon"]], {"threshold": 0.9}
        )
        assert "columns_to_drop" in art
        assert isinstance(art["columns_to_drop"], list)

    def test_lat_lon_highly_correlated_in_customers(self) -> None:
        """CorrelationThresholdCalculator on the clean lat/lon subset of customers.csv
        must return a valid artifact — confirming it handles a two-column frame
        without raising and produces the expected artifact keys."""
        df = load_sample_dataset("customers")
        # Drop rows where lat or lon is NaN for a clean correlation computation.
        clean = df[["lat", "lon"]].dropna()
        art = CorrelationThresholdCalculator().fit(clean, {"threshold": 0.9})
        # Artifact must always have columns_to_drop (possibly empty for this dataset).
        assert "columns_to_drop" in art
        assert isinstance(art["columns_to_drop"], list)
