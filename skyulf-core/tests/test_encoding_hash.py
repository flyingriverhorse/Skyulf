"""Unit tests for the HashEncoder Calculator/Applier (fit + apply, dual-engine)."""

from typing import Any

import pandas as pd
import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.preprocessing.encoding.hash import (
    HashEncoderApplier,
    HashEncoderCalculator,
)


def _fit_apply(df: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run HashEncoderCalculator.fit then HashEncoderApplier.apply on ``df``."""
    params = HashEncoderCalculator().fit(df, config)
    result = HashEncoderApplier().apply(df, dict(params))
    return dict(params), result


def test_fit_records_columns_and_n_features() -> None:
    """fit() stores the resolved columns and configured (or default) n_features."""
    df = pd.DataFrame({"city": ["ny", "la", "sf"]})
    params = HashEncoderCalculator().fit(df, {"columns": ["city"], "n_features": 4})
    assert params["columns"] == ["city"]
    assert params["n_features"] == 4
    assert params["type"] == "hash_encoder"


def test_fit_default_n_features_is_ten() -> None:
    """When n_features is omitted from config, fit() falls back to 10."""
    df = pd.DataFrame({"city": ["ny", "la"]})
    params = HashEncoderCalculator().fit(df, {"columns": ["city"]})
    assert params["n_features"] == 10


def test_apply_buckets_are_within_valid_range() -> None:
    """Every hashed value falls in [0, n_features) and column name is preserved in place."""
    df = pd.DataFrame({"city": ["ny", "la", "sf", "ny", "chicago"]})
    n_features = 5
    params, out = _fit_apply(df, {"columns": ["city"], "n_features": n_features})
    assert list(out.columns) == ["city"]
    assert out["city"].between(0, n_features - 1).all()


def test_same_category_hashes_to_same_bucket_consistently() -> None:
    """Repeated occurrences of the same category value hash to the same bucket."""
    df = pd.DataFrame({"city": ["ny", "la", "ny", "la", "ny"]})
    _, out = _fit_apply(df, {"columns": ["city"], "n_features": 7})
    assert out.loc[0, "city"] == out.loc[2, "city"] == out.loc[4, "city"]
    assert out.loc[1, "city"] == out.loc[3, "city"]


def test_high_cardinality_forces_collisions_with_small_n_features() -> None:
    """With more distinct categories than buckets, at least one collision must occur."""
    categories = [f"category_{i}" for i in range(50)]
    df = pd.DataFrame({"city": categories})
    n_features = 3
    _, out = _fit_apply(df, {"columns": ["city"], "n_features": n_features})
    buckets_used = out["city"].nunique()
    # Pigeonhole principle: 50 distinct values into 3 buckets guarantees collisions.
    assert buckets_used <= n_features
    assert len(df) > n_features


def test_unseen_category_at_apply_time_still_hashes_deterministically() -> None:
    """HashEncoder has no learned mapping, so unseen categories still hash validly."""
    train = pd.DataFrame({"city": ["ny", "la"]})
    test = pd.DataFrame({"city": ["ny", "tokyo"]})  # "tokyo" never seen at fit time

    params = HashEncoderCalculator().fit(train, {"columns": ["city"], "n_features": 6})
    out = HashEncoderApplier().apply(test, dict(params))

    assert out["city"].between(0, 5).all()
    # Re-hashing "ny" via apply again yields the same bucket (pure function of value).
    out2 = HashEncoderApplier().apply(test, dict(params))
    assert out.loc[0, "city"] == out2.loc[0, "city"]


def test_empty_dataframe_returns_empty_output() -> None:
    """Fitting and applying on a zero-row DataFrame does not raise and returns 0 rows."""
    df = pd.DataFrame({"city": pd.Series([], dtype="object")})
    params, out = _fit_apply(df, {"columns": ["city"]})
    assert params["columns"] == ["city"]
    assert len(out) == 0


def test_single_row_dataframe() -> None:
    """A single-row frame is hashed to a single valid bucket value."""
    df = pd.DataFrame({"city": ["ny"]})
    _, out = _fit_apply(df, {"columns": ["city"], "n_features": 4})
    assert out.loc[0, "city"] in range(4)


def test_all_nan_column_hashes_the_string_representation() -> None:
    """An all-NaN column is coerced to the string 'nan' and hashes to a single bucket."""
    df = pd.DataFrame({"city": [None, None, None]})
    _, out = _fit_apply(df, {"columns": ["city"], "n_features": 4})
    assert out["city"].nunique() == 1


def test_no_columns_selected_returns_input_unchanged() -> None:
    """Explicitly picking zero columns is a no-op (user_picked_no_columns short-circuit)."""
    df = pd.DataFrame({"city": ["ny", "la"]})
    params = HashEncoderCalculator().fit(df, {"columns": []})
    assert params == {}
    out = HashEncoderApplier().apply(df, dict(params))
    pd.testing.assert_frame_equal(out, df)


def test_target_column_excluded_from_encoding() -> None:
    """The target column is excluded from encoding since HashEncoder destroys columns."""
    X = pd.DataFrame({"city": ["ny", "la"]})
    y = pd.Series(["ny", "la"], name="target")
    params = HashEncoderCalculator().fit((X, y), {"columns": ["city", "target"]})
    assert "target" not in params["columns"]


def test_polars_apply_path_produces_valid_bucket_range() -> None:
    """Polars apply path (native pl.hash) also stays within [0, n_features)."""
    df_pd = pd.DataFrame({"city": ["ny", "la", "sf"]})
    df_pl = pl.from_pandas(df_pd)
    n_features = 4

    params = HashEncoderCalculator().fit(df_pd, {"columns": ["city"], "n_features": n_features})
    out_pl = HashEncoderApplier().apply(df_pl, dict(params))
    out_pl_pd = out_pl.to_pandas()

    assert out_pl_pd["city"].between(0, n_features - 1).all()
    # Same-value rows must still hash to the same polars bucket.
    df_pl_dup = pl.from_pandas(pd.DataFrame({"city": ["ny", "ny"]}))
    out_dup = HashEncoderApplier().apply(df_pl_dup, dict(params)).to_pandas()
    assert out_dup.loc[0, "city"] == out_dup.loc[1, "city"]


def test_polars_apply_no_valid_columns_is_noop() -> None:
    """Polars apply returns X, y unchanged when configured columns aren't present in X."""
    df_pl = pl.DataFrame({"other": [1, 2]})
    params = {"columns": ["city"], "n_features": 5}
    out = HashEncoderApplier().apply(df_pl, dict(params))
    assert out.equals(df_pl)


def test_fit_no_resolvable_columns_returns_empty() -> None:
    """Fitting with only numeric columns and no explicit selection returns {}."""
    df = pd.DataFrame({"amount": [1, 2, 3]})
    params = HashEncoderCalculator().fit(df, {})
    assert params == {}


@given(
    cats=st.lists(st.sampled_from(["a", "b", "c"]), min_size=5, max_size=40),
)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fit_engine_parity_pandas_vs_polars(cats: list[str]) -> None:
    """fit() does not branch by engine, so pandas/polars inputs must yield identical artifacts."""
    df_pd = pd.DataFrame({"city": cats})
    df_pl = pl.from_pandas(df_pd)
    config = {"columns": ["city"], "n_features": 5}

    pd_params = HashEncoderCalculator().fit(df_pd, dict(config))
    pl_params = HashEncoderCalculator().fit(df_pl, dict(config))

    assert pd_params == pl_params


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has categorical columns ``city``/``plan_type`` — closer to production
    data than the small synthetic frames used elsewhere in this file.
    """

    def test_plan_type_hash_encoding_stays_within_bucket_range(self) -> None:
        """HashEncoder on ``plan_type`` (no NaN) maps every row to a valid bucket in [0, n_features).

        Verifies the deterministic-hashing guarantee on a multi-category real column:
        same plan_type value must always hash to the same bucket.
        """
        df = load_sample_dataset("customers")
        n_features = 6
        params, out = _fit_apply(df, {"columns": ["plan_type"], "n_features": n_features})

        assert out["plan_type"].between(0, n_features - 1).all()
        # Same plan_type value must hash to exactly one bucket every time.
        for val in df["plan_type"].unique():
            mask = df["plan_type"] == val
            assert out.loc[mask, "plan_type"].nunique() == 1
