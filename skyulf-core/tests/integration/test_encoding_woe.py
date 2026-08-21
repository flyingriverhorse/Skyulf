"""Unit tests for the WOEEncoder Calculator/Applier (fit + apply, dual-engine)."""

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.encoding.woe import WOEEncoderApplier, WOEEncoderCalculator

_empty_params_cases = TestCaseLoader(
    "preprocessing/encoding_woe", group="empty_params"
).load_with_ids()
_no_resolvable_columns_cases = TestCaseLoader(
    "preprocessing/encoding_woe", group="no_resolvable_columns"
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
    from ``tests/test_cases/preprocessing/encoding_woe.json`` (group ``empty_params``).
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


def test_polars_fit_integer_column_with_nulls_matches_apply_representation() -> None:
    """Fit-time string keys for a Polars integer column must match apply-time rendering.

    Regression test: pandas' ``.to_pandas()`` upcasts an int column containing
    nulls to float, so a naive ``frame[col].astype(str)`` fit-time key would be
    "1.0" while the Polars apply path (``.cast(pl.Utf8)``) renders "1" -- every
    known category would then silently miss the mapping and fall back to
    ``default`` at apply time.
    """
    X_fit = pl.DataFrame({"cat": [1, 2, 1, 2, 1, None, 2, 1]})
    y_fit = pl.Series("y", [1, 0, 1, 0, 1, 0, 0, 1])
    params = WOEEncoderCalculator().fit((X_fit, y_fit), {"columns": ["cat"]})

    mappings = params["mappings"]["cat"]
    assert set(mappings.keys()) == {"1", "2", "nan"}

    X_apply = pl.DataFrame({"cat": [1, 2, 1, None]})
    y_apply = pl.Series("y", [1, 0, 1, 0])
    out, _ = WOEEncoderApplier().apply((X_apply, y_apply), dict(params))

    expected = [mappings["1"], mappings["2"], mappings["1"], mappings["nan"]]
    np.testing.assert_allclose(out["cat"].to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_polars_fit_integer_column_without_nulls_uses_bare_string_keys() -> None:
    """A Polars integer column with no nulls should key the WOE map as plain ints-as-strings."""
    X_fit = pl.DataFrame({"cat": [1, 2, 1, 2, 1, 2, 2, 1]})
    y_fit = pl.Series("y", [1, 0, 1, 0, 1, 0, 0, 1])
    params = WOEEncoderCalculator().fit((X_fit, y_fit), {"columns": ["cat"]})

    assert set(params["mappings"]["cat"].keys()) == {"1", "2"}


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


def test_extract_target_returns_none_when_target_column_missing_from_x() -> None:
    """Shared target extraction should leave y as None when target_col is absent."""
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
    ``tests/test_cases/preprocessing/encoding_woe.json`` (group ``no_resolvable_columns``).
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


# ---------------------------------------------------------------------------
# fit_transform_train — leakage-safe cross-fitting of training rows (F-14)
# ---------------------------------------------------------------------------


def _expected_out_of_fold_woe(
    cats: list[str], y_bin: np.ndarray, reg: float, n_folds: int, default: float = 0.0
) -> np.ndarray:
    """Independently recompute out-of-fold WOE.

    Each row is encoded with the mapping fitted on the complement of its own
    fold; a category unseen in that complement falls back to ``default``,
    mirroring the apply path's unseen-category behaviour.
    """
    from sklearn.model_selection import KFold

    values = np.asarray(cats)
    encoded = np.zeros(len(cats))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, hold_idx in kf.split(np.arange(len(cats))):
        train_vals = values[train_idx]
        y_sub = y_bin[train_idx]
        total_pos = float(y_sub.sum())
        total_neg = float(len(train_idx) - total_pos)
        for i in hold_idx:
            mask = train_vals == values[i]
            if not mask.any():
                encoded[i] = default
                continue
            pos = float(y_sub[mask].sum())
            neg = float(mask.sum() - pos)
            dist_pos = (pos + reg) / (total_pos + reg)
            dist_neg = (neg + reg) / (total_neg + reg)
            encoded[i] = math.log(dist_neg / dist_pos)
    return encoded


def test_fit_transform_train_cross_fits_training_rows() -> None:
    """Training rows must be encoded out-of-fold; the artifact keeps the full-data fit."""
    X = pd.DataFrame({"city": ["a", "b", "c", "a", "b", "c"]})
    y = pd.Series([1, 0, 0, 1, 0, 1], name="target")
    config: dict[str, Any] = {"columns": ["city"], "regularization": 0.5}

    artifact, transformed = WOEEncoderCalculator().fit_transform_train((X, y), config)
    X_out, y_out = transformed

    full_artifact = WOEEncoderCalculator().fit((X, y), config)
    assert artifact["mappings"] == full_artifact["mappings"]

    y_bin = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    expected = _expected_out_of_fold_woe(list(X["city"]), y_bin, reg=0.5, n_folds=3)
    np.testing.assert_allclose(X_out["city"].to_numpy(), expected)

    leaky = X["city"].map(full_artifact["mappings"]["city"]).to_numpy()
    assert not np.allclose(X_out["city"].to_numpy(), leaky)
    assert list(y_out) == list(y)


def test_fit_transform_train_cross_fit_matches_across_engines() -> None:
    """Pandas and Polars fit_transform_train must produce identical encoded values."""
    X_pd = pd.DataFrame({"city": ["a", "b", "c", "a", "b", "c"]})
    y_pd = pd.Series([1, 0, 0, 1, 0, 1], name="target")
    config: dict[str, Any] = {"columns": ["city"], "regularization": 0.5}

    _, (X_out_pd, _) = WOEEncoderCalculator().fit_transform_train((X_pd, y_pd), config)

    X_pl = pl.from_pandas(X_pd)
    y_pl = pl.Series("target", y_pd)
    _, (X_out_pl, _) = WOEEncoderCalculator().fit_transform_train((X_pl, y_pl), config)

    np.testing.assert_allclose(
        X_out_pd["city"].to_numpy(), X_out_pl["city"].to_numpy(), rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fit_transform_train_noise_target_auc_stays_near_chance(engine: str) -> None:
    """On a pure-noise target, cross-fitted train rows must not leak the label.

    With many categories and few rows per category, the leaky full-fit encoding
    memorises each row's own label into its category's WOE, becoming strongly
    predictive of the noise target. WOE uses a log(neg/pos) convention, so the
    leak shows up as an AUC far from 0.5 in *either* direction — measure
    discriminative power as ``max(auc, 1 - auc)``.
    """
    from sklearn.metrics import roc_auc_score

    def discriminative_power(y_true: Any, values: np.ndarray) -> float:
        auc = roc_auc_score(y_true, values)
        return max(auc, 1.0 - auc)

    rng = np.random.default_rng(42)
    n, n_categories = 400, 200
    X_pd = pd.DataFrame({"city": [f"c{v}" for v in rng.integers(0, n_categories, size=n)]})
    y_pd = pd.Series(rng.integers(0, 2, size=n), name="target")
    config: dict[str, Any] = {"columns": ["city"], "regularization": 0.5}

    leaky_params = WOEEncoderCalculator().fit((X_pd, y_pd), config)
    leaky_out, _ = WOEEncoderApplier().apply((X_pd, y_pd), dict(leaky_params))
    disc_leaky = discriminative_power(y_pd, leaky_out["city"].to_numpy())

    if engine == "pandas":
        _, (X_out, _) = WOEEncoderCalculator().fit_transform_train((X_pd, y_pd), config)
    else:
        X_pl = pl.from_pandas(X_pd)
        y_pl = pl.Series("target", y_pd)
        _, (X_out_pl, _) = WOEEncoderCalculator().fit_transform_train((X_pl, y_pl), config)
        X_out = X_out_pl.to_pandas()
    disc_cross = discriminative_power(y_pd, X_out["city"].to_numpy())

    assert disc_leaky > 0.75, f"expected the leaky encoding to memorise labels, got {disc_leaky}"
    assert disc_cross < 0.60, f"cross-fitted encoding should sit near chance, got {disc_cross}"


def test_fit_transform_train_no_columns_picked_is_noop() -> None:
    """Explicitly picking zero columns returns an empty artifact and the input."""
    X = pd.DataFrame({"city": ["a", "b"]})
    y = pd.Series([0, 1], name="target")
    artifact, transformed = WOEEncoderCalculator().fit_transform_train((X, y), {"columns": []})
    assert artifact == {}
    X_out, y_out = transformed
    pd.testing.assert_frame_equal(X_out, X)
    assert list(y_out) == list(y)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_feature_engineer_pipeline_cross_fits_woe_training_rows(engine: str) -> None:
    """Pipeline training rows get out-of-fold WOE; held-out splits get the full artifact."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.preprocessing.pipeline import FeatureEngineer

    train_x_pd = pd.DataFrame({"city": ["a", "b", "c"] * 10})
    train_y_pd = pd.Series([1, 0, 0, 1, 0, 1] * 5, name="target")
    test_x_pd = pd.DataFrame({"city": ["a", "b", "c"]})
    test_y_pd = pd.Series([1, 0, 1], name="target")

    if engine == "polars":
        dataset = SplitDataset(
            train=cast(Any, (pl.from_pandas(train_x_pd), pl.Series("target", train_y_pd))),
            test=cast(Any, (pl.from_pandas(test_x_pd), pl.Series("target", test_y_pd))),
        )
    else:
        dataset = SplitDataset(train=(train_x_pd, train_y_pd), test=(test_x_pd, test_y_pd))

    engineer = FeatureEngineer(
        [{"name": "woe_city", "transformer": "WOEEncoder", "params": {"columns": ["city"]}}]
    )
    result, _ = engineer.fit_transform(dataset)
    assert isinstance(result, SplitDataset)

    config: dict[str, Any] = {"columns": ["city"]}
    full_artifact = WOEEncoderCalculator().fit((train_x_pd, train_y_pd), config)
    _, (expected_train, _) = WOEEncoderCalculator().fit_transform_train(
        (train_x_pd, train_y_pd), config
    )

    train_out, train_y_out = result.train
    test_out, _ = result.test
    train_values = (
        train_out.get_column("city").to_numpy()
        if engine == "polars"
        else train_out["city"].to_numpy()
    )
    test_values = (
        test_out.get_column("city").to_numpy()
        if engine == "polars"
        else test_out["city"].to_numpy()
    )

    np.testing.assert_allclose(train_values, expected_train["city"].to_numpy())
    leaky = train_x_pd["city"].map(full_artifact["mappings"]["city"]).to_numpy()
    assert not np.allclose(train_values, leaky)

    expected_test = test_x_pd["city"].map(full_artifact["mappings"]["city"]).to_numpy()
    np.testing.assert_allclose(test_values, expected_test)
    assert list(train_y_out) == list(train_y_pd)
