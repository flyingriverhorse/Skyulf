"""Unit tests for the WOEEncoder Calculator/Applier (fit + apply, dual-engine)."""

import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.encoding.woe import WOEEncoderApplier, WOEEncoderCalculator

_empty_params_cases = TestCaseLoader("preprocessing/encoding_woe_empty_params").load_with_ids()
_no_resolvable_columns_cases = TestCaseLoader(
    "preprocessing/encoding_woe_no_resolvable_columns"
).load_with_ids()


def _fit_apply(
    X: pd.DataFrame, y: pd.Series, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run WOEEncoderCalculator.fit then WOEEncoderApplier.apply on ``(X, y)``."""
    params = WOEEncoderCalculator().fit((X, y), config)
    result = WOEEncoderApplier().apply((X, y), dict(params))
    X_out, _ = result
    return dict(params), X_out


def _expected_woe(pos: int, neg: int, total_pos: int, total_neg: int, reg: float) -> float:
    """Hand-compute the WOE formula used by ``_column_woe`` for verification."""
    dist_pos = (pos + reg) / (total_pos + reg)
    dist_neg = (neg + reg) / (total_neg + reg)
    return math.log(dist_neg / dist_pos)


def test_fit_computes_correct_woe_values() -> None:
    """WOE values match the hand-computed log-odds formula for a simple 2-category case."""
    X = pd.DataFrame({"city": ["a", "a", "a", "b", "b", "b"]})
    y = pd.Series([1, 1, 0, 0, 0, 1], name="target")
    # city=a: pos=2, neg=1 ; city=b: pos=1, neg=2 ; total_pos=3, total_neg=3
    params = WOEEncoderCalculator().fit((X, y), {"columns": ["city"], "regularization": 0.5})

    expected_a = _expected_woe(pos=2, neg=1, total_pos=3, total_neg=3, reg=0.5)
    expected_b = _expected_woe(pos=1, neg=2, total_pos=3, total_neg=3, reg=0.5)
    np.testing.assert_allclose(params["mappings"]["city"]["a"], expected_a, rtol=1e-9)
    np.testing.assert_allclose(params["mappings"]["city"]["b"], expected_b, rtol=1e-9)
    assert params["information_value"]["city"] > 0


def test_fit_apply_round_trip_replaces_values_in_place() -> None:
    """apply() replaces each category with its WOE value, keeping the column name."""
    X = pd.DataFrame({"city": ["a", "a", "b", "b"]})
    y = pd.Series([1, 0, 0, 1], name="target")
    params, out = _fit_apply(X, y, {"columns": ["city"]})

    assert list(out.columns) == ["city"]
    assert out.loc[0, "city"] == params["mappings"]["city"]["a"]
    assert out.loc[2, "city"] == params["mappings"]["city"]["b"]
    assert out["city"].dtype == float


def test_unseen_category_at_apply_time_falls_back_to_default() -> None:
    """A category unseen during fit maps to the configured default (0.0) at apply time."""
    X_train = pd.DataFrame({"city": ["a", "a", "b", "b"]})
    y_train = pd.Series([1, 0, 0, 1], name="target")
    params = WOEEncoderCalculator().fit((X_train, y_train), {"columns": ["city"]})

    X_test = pd.DataFrame({"city": ["a", "c"]})  # "c" never seen at fit time
    y_test = pd.Series([1, 0], name="target")
    out, _ = WOEEncoderApplier().apply((X_test, y_test), dict(params))

    assert out.loc[0, "city"] == params["mappings"]["city"]["a"]
    assert out.loc[1, "city"] == params.get("default", 0.0)


def test_mixed_string_target_with_none_uses_object_null_mask() -> None:
    """A non-numeric target containing None exercises the object-array null-mask fallback."""
    X = pd.DataFrame({"city": ["a", "a", "b", "b", "b"]})
    y = pd.Series(["yes", "no", "no", "yes", None], name="target")
    params = WOEEncoderCalculator().fit((X, y), {"columns": ["city"]})
    assert params != {}
    assert set(params["mappings"]["city"].keys()) == {"a", "b"}


def test_fit_polars_resolves_target_column_from_within_x() -> None:
    """Polars fit path also supports resolving y from a target_column name inside X."""
    X_pl = pl.DataFrame({"city": ["a", "a", "b", "b"], "target": [1, 0, 0, 1]})
    params = WOEEncoderCalculator().fit(X_pl, {"columns": ["city"], "target_column": "target"})
    assert params != {}
    assert "target" not in params["columns"]
    assert set(params["mappings"]["city"].keys()) == {"a", "b"}


class TestBinaryTargetInvariantReturnsEmptyParams:
    """Scenarios (non-binary target, single-row, empty frame) that all fail the
    binary-target check and cause ``fit()`` to short-circuit to ``{}``. Loaded
    from ``tests/test_cases/preprocessing/encoding_woe_empty_params.json``.
    """

    @pytest.mark.parametrize(
        _empty_params_cases[0], _empty_params_cases[1], ids=_empty_params_cases[2]
    )
    def test_returns_empty_params(
        self,
        city: list[str],
        y: list[int],
        city_dtype: str | None,
        y_dtype: str | None,
        columns: list[str],
    ) -> None:
        X = pd.DataFrame({"city": pd.Series(city, dtype=city_dtype)})
        y_series = pd.Series(y, dtype=y_dtype, name="target")
        params = WOEEncoderCalculator().fit((X, y_series), {"columns": columns})
        assert params == {}


def test_missing_target_returns_empty_params() -> None:
    """Fitting with no target at all (no y, no target_column) returns {}."""
    X = pd.DataFrame({"city": ["a", "b"]})
    params = WOEEncoderCalculator().fit(X, {"columns": ["city"]})
    assert params == {}


def test_target_column_resolved_from_config_key() -> None:
    """When y is None but target_column names a column in X, that column is used as y."""
    X = pd.DataFrame({"city": ["a", "a", "b", "b"], "target": [1, 0, 0, 1]})
    params = WOEEncoderCalculator().fit(X, {"columns": ["city"], "target_column": "target"})
    assert params != {}
    assert "target" not in params["columns"]
    assert set(params["mappings"]["city"].keys()) == {"a", "b"}


def test_no_columns_selected_returns_input_unchanged() -> None:
    """Explicitly picking zero columns is a no-op (user_picked_no_columns short-circuit)."""
    X = pd.DataFrame({"city": ["a", "b"]})
    y = pd.Series([0, 1], name="target")
    params = WOEEncoderCalculator().fit((X, y), {"columns": []})
    assert params == {}
    out, _ = WOEEncoderApplier().apply((X, y), dict(params))
    pd.testing.assert_frame_equal(out, X)


def test_regularization_changes_woe_for_zero_count_category() -> None:
    """Higher regularization pulls a zero-negative category's WOE further from ±inf-adjacent."""
    X = pd.DataFrame({"city": ["a", "a", "a", "b"]})
    y = pd.Series([1, 1, 1, 0], name="target")  # city=a has 0 negatives
    small_reg = WOEEncoderCalculator().fit((X, y), {"columns": ["city"], "regularization": 0.01})
    large_reg = WOEEncoderCalculator().fit((X, y), {"columns": ["city"], "regularization": 5.0})

    woe_small = small_reg["mappings"]["city"]["a"]
    woe_large = large_reg["mappings"]["city"]["a"]
    assert woe_small != woe_large


def test_polars_apply_path_matches_pandas_values() -> None:
    """Polars apply path (replace_strict) yields identical WOE values to the pandas path."""
    X_pd = pd.DataFrame({"city": ["a", "a", "b", "b"]})
    y_pd = pd.Series([1, 0, 0, 1], name="target")
    params = WOEEncoderCalculator().fit((X_pd, y_pd), {"columns": ["city"]})

    out_pd, _ = WOEEncoderApplier().apply((X_pd, y_pd), dict(params))

    X_pl = pl.from_pandas(X_pd)
    y_pl = pl.Series("target", y_pd)
    out_pl, _ = WOEEncoderApplier().apply((X_pl, y_pl), dict(params))
    out_pl_pd = out_pl.to_pandas()

    np.testing.assert_allclose(out_pd["city"].to_numpy(), out_pl_pd["city"].to_numpy())


@st.composite
def _categorical_frame(draw: st.DrawFn, *, min_rows: int = 20, max_rows: int = 60) -> pd.DataFrame:
    """Generate a frame with one categorical feature and a binary target."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    cats = draw(st.lists(st.sampled_from(["x", "y", "z"]), min_size=n, max_size=n))
    target = draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n))
    assume(len(set(target)) == 2 and len(set(cats)) > 1)
    return pd.DataFrame({"city": cats, "target": target})


@given(df=_categorical_frame())
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_apply_engine_parity_pandas_vs_polars(df: pd.DataFrame) -> None:
    """apply() on pandas vs polars must produce numerically identical WOE-mapped values."""
    X = df[["city"]]
    y = df["target"]
    params = WOEEncoderCalculator().fit((X, y), {"columns": ["city"]})

    out_pd, _ = WOEEncoderApplier().apply((X, y), dict(params))

    X_pl = pl.from_pandas(X)
    y_pl = pl.Series("target", y)
    out_pl, _ = WOEEncoderApplier().apply((X_pl, y_pl), dict(params))
    out_pl_pd = out_pl.to_pandas()

    np.testing.assert_allclose(
        out_pd["city"].to_numpy(), out_pl_pd["city"].to_numpy(), rtol=1e-9, atol=1e-9
    )


def test_polars_apply_no_valid_columns_is_noop() -> None:
    """Polars apply returns X, y unchanged when configured columns/mappings aren't available."""
    X = pd.DataFrame({"city": ["a", "b"]})
    y = pd.Series([0, 1], name="target")
    params = dict(WOEEncoderCalculator().fit((X, y), {"columns": ["city"]}))

    X_pl = pl.DataFrame({"other": [1, 2]})
    y_pl = pl.Series("target", [0, 1])
    out_pl, y_out = WOEEncoderApplier().apply((X_pl, y_pl), dict(params))

    assert out_pl.equals(X_pl)
    assert list(y_out) == list(y_pl)


def test_extract_y_returns_none_when_target_column_missing_from_x() -> None:
    """_extract_y's final fallback: y stays None if target_col isn't found in X either."""
    X = pd.DataFrame({"city": ["a", "b"]})
    params = WOEEncoderCalculator().fit(X, {"columns": ["city"], "target_column": "nonexistent"})
    assert params == {}


def test_woe_fit_polars_no_target_returns_empty_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Polars fit without a resolvable target logs a warning and returns {}."""
    X_pl = pl.DataFrame({"city": ["a", "b"]})
    with caplog.at_level("WARNING"):
        params = WOEEncoderCalculator().fit(X_pl, {"columns": ["city"]})

    assert params == {}
    assert any("requires a target variable" in rec.message for rec in caplog.records)


class TestFitNoResolvableColumnsReturnsEmpty:
    """A purely-numeric frame yields no encodable columns, so fit() returns {}.
    Scenarios (pandas/polars) loaded from
    ``tests/test_cases/preprocessing/encoding_woe_no_resolvable_columns.json``.
    """

    @pytest.mark.parametrize(
        _no_resolvable_columns_cases[0],
        _no_resolvable_columns_cases[1],
        ids=_no_resolvable_columns_cases[2],
    )
    def test_fit_no_resolvable_columns_returns_empty(self, engine: str) -> None:
        if engine == "polars":
            X = pl.DataFrame({"amount": [1, 2, 3, 4]})
            y = pl.Series("target", [0, 1, 0, 1])
        else:
            X = pd.DataFrame({"amount": [1, 2, 3, 4]})
            y = pd.Series([0, 1, 0, 1], name="target")
        params = WOEEncoderCalculator().fit((X, y), {})
        assert params == {}


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample.
    ``plan_type`` (no NaN, 3 categories) + binary ``churned`` target — exercises
    multi-category WOE computation on production-like data, including a category
    with zero positive-class observations (enterprise: all churned=0).
    """

    def test_plan_type_woe_encoding_with_binary_churn_target(self) -> None:
        """WOE encoding of ``plan_type`` produces a finite WOE value for every category.

        Regularization prevents ±inf WOE for the enterprise group (zero churned),
        and every category must receive a distinct finite float mapping.
        """
        df = load_sample_dataset("customers")
        X = df[["plan_type"]].copy()
        y = df["churned"]
        params = WOEEncoderCalculator().fit((X, y), {"columns": ["plan_type"]})

        assert params != {}
        assert set(params["mappings"]["plan_type"].keys()) == set(df["plan_type"].unique())
        for woe_val in params["mappings"]["plan_type"].values():
            assert isinstance(woe_val, float)
            assert math.isfinite(woe_val)
