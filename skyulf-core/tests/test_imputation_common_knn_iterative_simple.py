"""Coverage-focused tests for the imputation package.

Targets ``_common.py`` helpers directly, the KNN and Iterative imputers
(not covered by ``test_engine_parity.py``), the remaining SimpleImputer
strategies (``most_frequent`` / ``constant``), and shared edge cases
(empty frame, all-NaN column, single row, no missing values).
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.imputation._common import (
    _build_iterative_estimator,
    _compute_polars_fill_values,
    _polars_missing_counts,
    _polars_stat_for_strategy,
    _resolve_simple_columns,
    _sklearn_transform_subset,
)
from skyulf.preprocessing.imputation.iterative import (
    IterativeImputerApplier,
    IterativeImputerCalculator,
)
from skyulf.preprocessing.imputation.knn import KNNImputerApplier, KNNImputerCalculator
from skyulf.preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator

_FINITE_FLOAT = st.floats(
    min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False, width=64
)


def _load_single_param(source_path: str, group: str | None = None) -> list:
    """Load a JSON fixture with exactly one param, unwrapping 1-tuples.

    ``pytest.mark.parametrize`` treats a single (comma-less) param name
    specially: argvalues must be bare scalars, not 1-tuples, or the whole
    tuple gets bound to the parameter instead of its single element.
    """
    params_string, scenarios = TestCaseLoader(source_path, group=group).load()
    return [params_string, [scenario[0] for scenario in scenarios]]


_polars_stat_cases = TestCaseLoader(
    "preprocessing/imputation_common_knn_iterative_simple", group="polars_stat_for_strategy"
).load()
_compute_polars_fill_values_cases = TestCaseLoader(
    "preprocessing/imputation_common_knn_iterative_simple", group="compute_polars_fill_values"
).load()
_iterative_estimator_cases = TestCaseLoader(
    "preprocessing/imputation_common_knn_iterative_simple", group="iterative_estimator_aliases"
).load()
_no_columns_or_numeric_cases = TestCaseLoader(
    "preprocessing/imputation_common_knn_iterative_simple",
    group="no_columns_or_numeric_returns_empty",
).load()
_applier_pandas_noop_cases = TestCaseLoader(
    "preprocessing/imputation_common_knn_iterative_simple", group="applier_pandas_noop"
).load()
_applier_polars_noop_cases = TestCaseLoader(
    "preprocessing/imputation_common_knn_iterative_simple", group="applier_polars_noop"
).load()
_infer_output_schema_cases = _load_single_param(
    "preprocessing/imputation_common_knn_iterative_simple", group="infer_output_schema_passthrough"
)
_all_imputers_empty_df_cases = _load_single_param(
    "preprocessing/imputation_common_knn_iterative_simple", group="all_imputers_empty_dataframe"
)

# Maps a JSON-friendly imputer name to its Calculator/Applier classes, shared
# by the cross-imputer parametrized tests below.
_CALCULATOR_BY_NAME: dict[str, Any] = {
    "simple": SimpleImputerCalculator,
    "knn": KNNImputerCalculator,
    "iterative": IterativeImputerCalculator,
}
_APPLIER_BY_NAME: dict[str, Any] = {
    "knn": KNNImputerApplier,
    "iterative": IterativeImputerApplier,
}


class _BrokenImputer:
    """Stand-in sklearn imputer whose ``transform`` always raises."""

    def transform(self, X: Any) -> Any:
        raise RuntimeError("boom")


_IMPUTER_KIND: dict[str, Any] = {"object": object, "broken": _BrokenImputer}


@st.composite
def _numeric_frame(draw: st.DrawFn, *, min_rows: int = 6, max_rows: int = 30) -> pd.DataFrame:
    """Generate a small two-column numeric DataFrame with no missing values."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    a = draw(st.lists(_FINITE_FLOAT, min_size=n, max_size=n))
    b = draw(st.lists(_FINITE_FLOAT, min_size=n, max_size=n))
    return pd.DataFrame({"a": a, "b": b})


# ---------------------------------------------------------------------------
# _common.py — direct helper-function tests
# ---------------------------------------------------------------------------


def test_polars_stat_for_strategy_constant_returns_none() -> None:
    """The 'constant' strategy is handled by the caller, not an expression."""
    assert _polars_stat_for_strategy("constant", 5) is None


@pytest.mark.parametrize(*_polars_stat_cases)
def test_polars_stat_for_strategy_builds_expected_expr(
    strategy: str, values: list, expected: Any
) -> None:
    """Each supported strategy must build a Polars expression matching the reference stat."""
    df = pl.DataFrame({"a": values})
    expr_builder = _polars_stat_for_strategy(strategy, None)
    result = df.select([expr_builder("a")]).to_dict(as_series=False)
    assert result["a"][0] == pytest.approx(expected)


def test_polars_stat_for_strategy_unknown_raises_value_error() -> None:
    """An unrecognized strategy name must raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        _polars_stat_for_strategy("bogus", None)


@pytest.mark.parametrize(*_compute_polars_fill_values_cases)
def test_compute_polars_fill_values(
    columns_data: dict, columns: list, strategy: str, fill_value: Any, expected: dict
) -> None:
    """Constant and mean strategies must compute the expected per-column fill values."""
    df = pl.DataFrame(columns_data)
    result = _compute_polars_fill_values(df, columns, strategy, fill_value)
    for col, exp in expected.items():
        assert result[col] == pytest.approx(exp)


def test_polars_missing_counts_totals_nulls_per_column() -> None:
    """Missing-value counts are computed per column and summed for the total."""
    df = pl.DataFrame({"a": [1.0, None, None], "b": [None, 2.0, 3.0]})
    counts, total = _polars_missing_counts(df, ["a", "b"])
    assert counts == {"a": 2, "b": 1}
    assert total == 3


def test_resolve_simple_columns_mean_detects_numeric_only() -> None:
    """For mean/median strategies, only numeric columns are auto-detected."""
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["x", "y", "z"]})
    cols = _resolve_simple_columns(df, {}, "mean")
    assert cols == ["num"]


def test_resolve_simple_columns_most_frequent_includes_all_columns() -> None:
    """For most_frequent/constant strategies, all columns are eligible."""
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["x", "y", "z"]})
    cols = _resolve_simple_columns(df, {}, "most_frequent")
    assert set(cols) == {"num", "cat"}


@pytest.mark.parametrize(*_iterative_estimator_cases)
def test_build_iterative_estimator_alias(alias: str, module: str, class_name: str) -> None:
    """Each estimator alias (and the unrecognized-name default) must map to the
    expected sklearn regressor class."""
    import importlib

    expected_cls = getattr(importlib.import_module(module), class_name)
    estimator = _build_iterative_estimator(alias)
    assert isinstance(estimator, expected_cls)


def test_sklearn_transform_subset_pandas_writes_back_into_copy() -> None:
    """Pandas branch transforms the subset and writes it back without mutating input."""
    from sklearn.impute import SimpleImputer

    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 20.0, 30.0]})
    imputer = SimpleImputer(strategy="mean").fit(df[["a"]])

    out = _sklearn_transform_subset(df, ["a"], imputer, is_polars=False)

    assert df["a"].isna().sum() == 1  # original untouched
    assert out["a"].isna().sum() == 0
    assert out["a"].iloc[1] == pytest.approx(2.0)


def test_sklearn_transform_subset_polars_writes_back_into_copy() -> None:
    """Polars branch transforms the subset and returns a new frame."""
    from sklearn.impute import SimpleImputer

    pdf = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 20.0, 30.0]})
    imputer = SimpleImputer(strategy="mean").fit(pdf[["a"]])

    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [10.0, 20.0, 30.0]})
    out = _sklearn_transform_subset(df, ["a"], imputer, is_polars=True)

    assert out["a"].null_count() == 0
    assert out["a"].to_list()[1] == pytest.approx(2.0)


def test_sklearn_transform_subset_polars_handles_dataframe_like_transform_result() -> None:
    """When the imputer.transform() result exposes `.values` (e.g. a DataFrame), unwrap it."""

    class _DataFrameReturningImputer:
        def transform(self, X: Any) -> Any:
            return pd.DataFrame(np.full_like(X, 42.0))

    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = _sklearn_transform_subset(df, ["a", "b"], _DataFrameReturningImputer(), is_polars=True)

    assert out["a"].to_list() == [42.0, 42.0]
    assert out["b"].to_list() == [42.0, 42.0]


# ---------------------------------------------------------------------------
# KNNImputer — fit -> apply round trips
# ---------------------------------------------------------------------------


def test_knn_imputer_fit_apply_round_trip_pandas() -> None:
    """KNN imputer fills NaNs using nearest-neighbor structure (pandas)."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    config: dict[str, Any] = {"columns": ["a", "b"], "n_neighbors": 2}
    calc = KNNImputerCalculator()
    params = calc.fit(df, config)

    assert params["type"] == "knn_imputer"
    assert params["columns"] == ["a", "b"]

    applier = KNNImputerApplier()
    out = applier.apply(df, params)

    assert out["a"].isna().sum() == 0
    # Row index 2 has b=3.0; nearest neighbors by b are rows 1 (b=2, a=2) and
    # row 3 (b=4, a=4) => KNN mean-fill should land close to 3.0.
    assert out["a"].iloc[2] == pytest.approx(3.0, abs=0.5)


def test_knn_imputer_fit_apply_round_trip_polars() -> None:
    """KNN imputer works through the Polars apply branch as well."""
    pdf = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    df = pl.from_pandas(pdf)
    config: dict[str, Any] = {"columns": ["a", "b"], "n_neighbors": 2}
    calc = KNNImputerCalculator()
    params = calc.fit(df, config)

    applier = KNNImputerApplier()
    out = applier.apply(df, params)

    assert out["a"].null_count() == 0


@pytest.mark.parametrize(*_no_columns_or_numeric_cases)
def test_no_columns_or_numeric_returns_empty(calculator: str, df_data: dict, config: dict) -> None:
    """Explicit empty columns, or no auto-detected numeric columns, is a no-op."""
    df = pd.DataFrame(df_data)
    params = _CALCULATOR_BY_NAME[calculator]().fit(df, config)
    assert params == {}


@pytest.mark.parametrize(*_applier_pandas_noop_cases)
def test_applier_pandas_noop(
    applier: str, df_data: dict, config: dict, imputer_kind: str | None
) -> None:
    """Applier must return X unchanged (pandas) when: fitted columns are missing
    from X, no imputer object was fitted, or the sklearn imputer raises."""
    df = pd.DataFrame(df_data)
    full_config = dict(config)
    if imputer_kind is not None:
        full_config["imputer_object"] = _IMPUTER_KIND[imputer_kind]()
    out = _APPLIER_BY_NAME[applier]().apply(df, full_config)
    pd.testing.assert_frame_equal(out, df)


@pytest.mark.parametrize(*_applier_polars_noop_cases)
def test_applier_polars_noop(applier: str, df_data: dict, config: dict, imputer_kind: str) -> None:
    """Applier must return X unchanged (Polars) when fitted columns are missing
    from X, or the sklearn imputer raises during transform."""
    df = pl.DataFrame(df_data)
    full_config = dict(config)
    full_config["imputer_object"] = _IMPUTER_KIND[imputer_kind]()
    out = _APPLIER_BY_NAME[applier]().apply(df, full_config)
    assert out.equals(df)


@pytest.mark.parametrize(*_infer_output_schema_cases)
def test_imputer_infer_output_schema_passes_through(calculator: str) -> None:
    """Output schema is identical to input schema (columns preserved)."""
    from skyulf.preprocessing._schema import SkyulfSchema

    schema = SkyulfSchema(columns=("a",), dtypes={"a": "float64"})
    result = _CALCULATOR_BY_NAME[calculator]().infer_output_schema(schema, {})
    assert result is schema


# ---------------------------------------------------------------------------
# IterativeImputer — fit -> apply round trips
# ---------------------------------------------------------------------------


def test_iterative_imputer_fit_apply_round_trip_pandas() -> None:
    """Iterative (MICE) imputer fills NaNs using a multivariate model (pandas)."""
    rng = np.random.default_rng(0)
    n = 40
    b = rng.normal(size=n)
    a = 2.0 * b + rng.normal(scale=0.01, size=n)
    df = pd.DataFrame({"a": a, "b": b})
    df.loc[5, "a"] = np.nan

    config: dict[str, Any] = {"columns": ["a", "b"], "max_iter": 5, "estimator": "BayesianRidge"}
    calc = IterativeImputerCalculator()
    params = calc.fit(df, config)

    assert params["type"] == "iterative_imputer"
    assert params["estimator"] == "BayesianRidge"

    applier = IterativeImputerApplier()
    out = applier.apply(df, params)

    assert out["a"].isna().sum() == 0
    # a ~= 2*b, so the imputed value should track that relationship closely.
    assert out["a"].iloc[5] == pytest.approx(2.0 * df["b"].iloc[5], abs=0.5)


def test_iterative_imputer_fit_apply_round_trip_polars() -> None:
    """Iterative imputer works through the Polars apply branch as well."""
    rng = np.random.default_rng(1)
    n = 40
    b = rng.normal(size=n)
    a = 2.0 * b + rng.normal(scale=0.01, size=n)
    pdf = pd.DataFrame({"a": a, "b": b})
    pdf.loc[3, "a"] = np.nan
    df = pl.from_pandas(pdf)

    config: dict[str, Any] = {"columns": ["a", "b"], "max_iter": 5}
    calc = IterativeImputerCalculator()
    params = calc.fit(df, config)

    applier = IterativeImputerApplier()
    out = applier.apply(df, params)

    assert out["a"].null_count() == 0


def test_iterative_imputer_uses_kneighbors_estimator() -> None:
    """The 'KNeighbors' estimator alias is wired through the calculator fit."""
    rng = np.random.default_rng(2)
    n = 30
    b = rng.normal(size=n)
    a = b + rng.normal(scale=0.01, size=n)
    df = pd.DataFrame({"a": a, "b": b})
    df.loc[0, "a"] = np.nan

    params = IterativeImputerCalculator().fit(
        df, {"columns": ["a", "b"], "estimator": "KNeighbors", "max_iter": 5}
    )
    assert params["estimator"] == "KNeighbors"
    out = IterativeImputerApplier().apply(df, params)
    assert out["a"].isna().sum() == 0


# ---------------------------------------------------------------------------
# SimpleImputer — most_frequent / constant strategies
# ---------------------------------------------------------------------------


def test_simple_imputer_most_frequent_strategy_pandas() -> None:
    """most_frequent strategy fills with the mode of each column."""
    df = pd.DataFrame({"cat": ["x", "x", np.nan, "y"]})
    calc = SimpleImputerCalculator()
    params = calc.fit(df, {"columns": ["cat"], "strategy": "most_frequent"})

    assert params["fill_values"]["cat"] == "x"

    out = SimpleImputerApplier().apply(df, params)
    assert out["cat"].isna().sum() == 0
    assert out["cat"].iloc[2] == "x"


def test_simple_imputer_most_frequent_strategy_polars() -> None:
    """most_frequent strategy works identically through the Polars branch."""
    pdf = pd.DataFrame({"cat": ["x", "x", None, "y"]})
    df = pl.from_pandas(pdf)
    calc = SimpleImputerCalculator()
    params = calc.fit(df, {"columns": ["cat"], "strategy": "most_frequent"})

    assert params["fill_values"]["cat"] == "x"

    out = SimpleImputerApplier().apply(df, params)
    assert out["cat"].null_count() == 0


def test_simple_imputer_most_frequent_tie_picks_smallest_value_both_engines() -> None:
    """On a genuine tie, both pandas and polars must pick the smallest value.

    sklearn's SimpleImputer(strategy="most_frequent") and scipy.stats.mode
    deterministically break ties by choosing the smallest value; polars'
    .mode() has no guaranteed tie-break order, so we must force it.
    """
    df_pd = pd.DataFrame({"n": [2, 2, 1, 1, np.nan]})
    calc = SimpleImputerCalculator()
    params = calc.fit(df_pd, {"columns": ["n"], "strategy": "most_frequent"})
    assert params["fill_values"]["n"] == 1

    df_pl = pl.from_pandas(df_pd)
    params_pl = calc.fit(df_pl, {"columns": ["n"], "strategy": "most_frequent"})
    assert params_pl["fill_values"]["n"] == 1


def test_simple_imputer_mode_alias_maps_to_most_frequent() -> None:
    """The 'mode' strategy alias is normalized to 'most_frequent'."""
    df = pd.DataFrame({"cat": ["x", "x", np.nan, "y"]})
    params = SimpleImputerCalculator().fit(df, {"columns": ["cat"], "strategy": "mode"})
    assert params["strategy"] == "most_frequent"


def test_simple_imputer_constant_strategy_pandas() -> None:
    """constant strategy fills NaNs with the configured fill_value."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    calc = SimpleImputerCalculator()
    params = calc.fit(df, {"columns": ["a"], "strategy": "constant", "fill_value": -1.0})

    assert params["fill_values"]["a"] == -1.0

    out = SimpleImputerApplier().apply(df, params)
    assert out["a"].iloc[1] == -1.0


def test_simple_imputer_constant_strategy_polars() -> None:
    """constant strategy works identically through the Polars branch."""
    pdf = pd.DataFrame({"a": [1.0, None, 3.0]})
    df = pl.from_pandas(pdf)
    calc = SimpleImputerCalculator()
    params = calc.fit(df, {"columns": ["a"], "strategy": "constant", "fill_value": -1.0})

    assert params["fill_values"]["a"] == -1.0

    out = SimpleImputerApplier().apply(df, params)
    assert out["a"].to_list()[1] == -1.0


def test_simple_imputer_mean_strategy_exact_fill_value() -> None:
    """Mean strategy fills with the exact column mean (verified value)."""
    df = pd.DataFrame({"a": [1.0, 3.0, np.nan]})
    calc = SimpleImputerCalculator()
    params = calc.fit(df, {"columns": ["a"], "strategy": "mean"})
    assert params["fill_values"]["a"] == pytest.approx(2.0)

    out = SimpleImputerApplier().apply(df, params)
    assert out["a"].iloc[2] == pytest.approx(2.0)


def test_simple_imputer_median_strategy_exact_fill_value() -> None:
    """Median strategy fills with the exact column median (verified value)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 100.0, np.nan]})
    calc = SimpleImputerCalculator()
    params = calc.fit(df, {"columns": ["a"], "strategy": "median"})
    assert params["fill_values"]["a"] == pytest.approx(2.0)


def test_simple_imputer_apply_restores_column_missing_from_input() -> None:
    """If a fitted column is absent from X at apply-time, it is recreated (pandas)."""
    df_fit = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    params = SimpleImputerCalculator().fit(df_fit, {"columns": ["a"], "strategy": "mean"})

    df_apply = pd.DataFrame({"b": [10.0, 20.0]})
    out = SimpleImputerApplier().apply(df_apply, params)
    assert "a" in out.columns
    assert (out["a"] == params["fill_values"]["a"]).all()


def test_simple_imputer_apply_restores_column_missing_from_input_polars() -> None:
    """If a fitted column is absent from X at apply-time, it is recreated (Polars)."""
    pdf_fit = pd.DataFrame({"a": [1.0, None, 3.0]})
    df_fit = pl.from_pandas(pdf_fit)
    params = SimpleImputerCalculator().fit(df_fit, {"columns": ["a"], "strategy": "mean"})

    df_apply = pl.DataFrame({"b": [10.0, 20.0]})
    out = SimpleImputerApplier().apply(df_apply, params)
    assert "a" in out.columns
    assert out["a"].to_list() == [pytest.approx(params["fill_values"]["a"])] * 2


def test_simple_imputer_apply_no_columns_is_noop() -> None:
    """Applier is a no-op when params carries no columns (empty fit result)."""
    df = pd.DataFrame({"a": [1.0, np.nan]})
    out = SimpleImputerApplier().apply(df, {})
    pd.testing.assert_frame_equal(out, df)


def test_simple_imputer_apply_no_columns_is_noop_polars() -> None:
    """Polars apply branch is a no-op when params carries no columns."""
    df = pl.DataFrame({"a": [1.0, None]})
    out = SimpleImputerApplier().apply(df, {})
    assert out.equals(df)


def test_simple_imputer_apply_pandas_skips_column_without_fill_value() -> None:
    """Pandas apply branch skips a fitted column that has no entry in fill_values."""
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    params = {"columns": ["a", "b"], "fill_values": {"a": 9.0}}
    out = SimpleImputerApplier().apply(df, params)
    assert out["a"].iloc[1] == 9.0
    assert np.isnan(out["b"].iloc[0])  # "b" untouched: no fill_value provided


def test_simple_imputer_no_columns_short_circuits() -> None:
    """Explicit empty columns list means 'do nothing' and returns {}."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    params = SimpleImputerCalculator().fit(df, {"columns": []})
    assert params == {}


def test_simple_imputer_pandas_mean_filters_non_numeric_and_returns_empty() -> None:
    """Mean strategy on pandas re-filters to numeric columns; all-categorical returns {}."""
    df = pd.DataFrame({"cat": ["x", "y", "z"]})
    # Force resolved_cols to include a non-numeric column to hit the safety filter.
    params = SimpleImputerCalculator()._fit_pandas(
        df,
        None,
        {"_resolved_cols": ["cat"], "_resolved_strategy": "mean", "_resolved_fill_value": None},
    )
    assert params == {}


# ---------------------------------------------------------------------------
# Shared edge cases across all imputers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_all_imputers_empty_df_cases)
def test_all_imputers_empty_dataframe_returns_empty_params(calculator: str) -> None:
    """Fitting on a fully empty DataFrame (no rows/cols) yields empty params."""
    df = pd.DataFrame()
    params = _CALCULATOR_BY_NAME[calculator]().fit(df, {})
    assert params == {}


def test_simple_imputer_all_nan_column_explicit_constant_strategy() -> None:
    """sklearn skips all-NaN columns for constant strategy (needs 1+ observed value).

    ``SimpleImputer.statistics_`` reports NaN rather than fill_value in this
    edge case (verified sklearn 1.8 behavior) -- documenting it here so a
    future sklearn upgrade that changes this is caught by the test suite.
    """
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
    params = SimpleImputerCalculator().fit(
        df, {"columns": ["a"], "strategy": "constant", "fill_value": 0.0}
    )
    assert np.isnan(params["fill_values"]["a"])


def test_compute_polars_fill_values_all_null_column_mean_is_null() -> None:
    """An all-null column has no mean; Polars mean() over an all-null column is null."""
    df = pl.DataFrame({"a": [None, None, None]}, schema={"a": pl.Float64})
    result = _compute_polars_fill_values(df, ["a"], "mean", None)
    assert result["a"] is None


def test_simple_imputer_single_row_no_missing_values(sample_regression_data: pd.DataFrame) -> None:
    """A single-row frame with no missing values fits/applies without error."""
    df = sample_regression_data[["feature1", "feature2"]].iloc[[10]].copy()
    calc = SimpleImputerCalculator()
    params = calc.fit(df, {"columns": ["feature1", "feature2"], "strategy": "mean"})
    out = SimpleImputerApplier().apply(df, params)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))


def test_knn_imputer_no_missing_values_leaves_data_unchanged(
    sample_regression_data: pd.DataFrame,
) -> None:
    """When there are no missing values, KNN imputer output equals input."""
    df = sample_regression_data[["feature2", "target"]].copy()  # no NaNs in these cols
    calc = KNNImputerCalculator()
    params = calc.fit(df, {"columns": ["feature2", "target"], "n_neighbors": 3})
    out = KNNImputerApplier().apply(df, params)
    np.testing.assert_allclose(out["feature2"].to_numpy(), df["feature2"].to_numpy())


def test_simple_imputer_fit_apply_round_trip_with_classification_fixture(
    sample_classification_data: pd.DataFrame,
) -> None:
    """Fit -> apply round trip on the shared classification fixture (has NaN in feature1)."""
    df = sample_classification_data
    calc = SimpleImputerCalculator()
    params = calc.fit(df, {"columns": ["feature1"], "strategy": "mean"})
    out = SimpleImputerApplier().apply(df, params)
    assert out["feature1"].isna().sum() == 0
    assert params["total_missing"] == df["feature1"].isna().sum()


# ---------------------------------------------------------------------------
# Hypothesis engine-parity tests for polars code paths
# ---------------------------------------------------------------------------


@given(df=_numeric_frame(min_rows=8, max_rows=40))
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_polars_missing_counts_matches_pandas_isna_sum(df: pd.DataFrame) -> None:
    """_polars_missing_counts must match pandas' isna().sum() for the same data."""
    pdf = df.copy()
    pdf.loc[0, "a"] = np.nan
    pl_df = pl.from_pandas(pdf)

    counts, total = _polars_missing_counts(pl_df, ["a", "b"])
    expected_counts = pdf[["a", "b"]].isna().sum().to_dict()

    assert counts == expected_counts
    assert total == int(sum(expected_counts.values()))


@given(df=_numeric_frame(min_rows=8, max_rows=40))
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_compute_polars_fill_values_mean_matches_pandas_mean(df: pd.DataFrame) -> None:
    """_compute_polars_fill_values('mean') must match pandas .mean() within tolerance."""
    pl_df = pl.from_pandas(df)
    result = _compute_polars_fill_values(pl_df, ["a", "b"], "mean", None)

    for col in ["a", "b"]:
        np.testing.assert_allclose(result[col], df[col].mean(), rtol=1e-9, atol=1e-9)


@given(
    df=_numeric_frame(min_rows=8, max_rows=40),
    fill_value=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_simple_imputer_constant_engine_parity(df: pd.DataFrame, fill_value: float) -> None:
    """SimpleImputer constant-strategy fit is identical across pandas and polars."""
    config: dict[str, Any] = {
        "columns": ["a", "b"],
        "strategy": "constant",
        "fill_value": fill_value,
    }
    pd_params = SimpleImputerCalculator().fit(df, dict(config))
    pl_params = SimpleImputerCalculator().fit(pl.from_pandas(df), dict(config))

    assert pd_params["fill_values"] == pl_params["fill_values"]
    assert pd_params["columns"] == pl_params["columns"]


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing values in ``age``/``income`` — closer to production data
    than the small synthetic frames used elsewhere in this file.
    """

    def test_mean_strategy_fills_age_and_income_nans(self) -> None:
        df = load_sample_dataset("customers")
        calc = SimpleImputerCalculator()
        applier = SimpleImputerApplier()
        params = calc.fit(df, {"columns": ["age", "income"], "strategy": "mean"})
        out = applier.apply(df, params)

        assert out["age"].isna().sum() == 0
        assert out["income"].isna().sum() == 0
        np.testing.assert_allclose(out.loc[df["age"].isna(), "age"], df["age"].mean(), rtol=1e-9)
        np.testing.assert_allclose(
            out.loc[df["income"].isna(), "income"], df["income"].mean(), rtol=1e-9
        )
