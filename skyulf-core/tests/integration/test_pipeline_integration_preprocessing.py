"""Integration tests chaining multiple preprocessing nodes on real-shaped data.

Unlike the per-node unit tests elsewhere in this suite, these tests thread the
output of one Calculator/Applier pair directly into the next node's input,
using the 300-row ``pipeline_dataset.csv`` fixture (see
``tests/utils/dataset_loader.py``). Each test verifies that intermediate
outputs are valid inputs for the following node, and that the final output of
the chain is sane (no unexpected NaNs, correct dtypes/columns) — the same
"fit once, apply many" contract the real ``SkyulfPipeline`` relies on.
"""

from typing import Any

import numpy as np
import pandas as pd
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.preprocessing.encoding.one_hot import (
    OneHotEncoderApplier,
    OneHotEncoderCalculator,
)
from skyulf.preprocessing.feature_generation.interaction import (
    FeatureInteractionApplier,
    FeatureInteractionCalculator,
)
from skyulf.preprocessing.feature_selection.variance import (
    VarianceThresholdApplier,
    VarianceThresholdCalculator,
)
from skyulf.preprocessing.geo.distance import GeoDistanceApplier, GeoDistanceCalculator
from skyulf.preprocessing.imputation.simple import (
    SimpleImputerApplier,
    SimpleImputerCalculator,
)
from skyulf.preprocessing.scaling.standard import (
    StandardScalerApplier,
    StandardScalerCalculator,
)


def _fit_apply(
    calculator: Any, applier: Any, df: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Run ``calculator.fit`` then ``applier.apply`` and return the transformed frame."""
    params = calculator.fit(df, config)
    return applier.apply(df, dict(params))


def _load_dataset() -> pd.DataFrame:
    """Load the 300-row pipeline-scale fixture used across these chain tests."""
    return load_sample_dataset("pipeline_dataset")


def test_impute_then_scale_numeric_columns() -> None:
    """Imputing age/income then standard-scaling them yields a NaN-free, ~N(0,1) result.

    Verifies the Imputer -> Scaler chain: the Scaler must be able to consume
    the Imputer's output directly (no leftover NaNs breaking sklearn's
    ``StandardScaler.fit``), and the final values must match the manual
    z-score formula computed on the *imputed* data (not the raw data).
    """
    df = _load_dataset()
    assert df["age"].isna().any()
    assert df["income"].isna().any()

    imputed = _fit_apply(
        SimpleImputerCalculator(),
        SimpleImputerApplier(),
        df,
        {"strategy": "mean", "columns": ["age", "income"]},
    )
    assert imputed["age"].isna().sum() == 0
    assert imputed["income"].isna().sum() == 0

    scaled = _fit_apply(
        StandardScalerCalculator(),
        StandardScalerApplier(),
        imputed,
        {"columns": ["age", "income"]},
    )

    assert scaled[["age", "income"]].isna().sum().sum() == 0
    for col in ("age", "income"):
        expected_mean = imputed[col].to_numpy().mean()
        expected_std = imputed[col].to_numpy().std(ddof=0)
        manual = (imputed[col].to_numpy() - expected_mean) / expected_std
        np.testing.assert_allclose(scaled[col].to_numpy(), manual, rtol=1e-10)
        assert abs(scaled[col].mean()) < 1e-8
        assert abs(scaled[col].std(ddof=0) - 1.0) < 1e-8


def test_impute_then_onehot_encode_categoricals() -> None:
    """Imputing the ``city`` column then one-hot encoding categoricals produces
    a fully-numeric, NaN-free block with exactly one dummy per observed category.
    """
    df = _load_dataset()
    n_city_categories = df["city"].nunique(dropna=True)
    n_plan_categories = df["plan_type"].nunique(dropna=True)
    assert df["city"].isna().any()
    assert not df["plan_type"].isna().any()

    imputed = _fit_apply(
        SimpleImputerCalculator(),
        SimpleImputerApplier(),
        df,
        {"strategy": "most_frequent", "columns": ["city"]},
    )
    assert imputed["city"].isna().sum() == 0

    encoded = _fit_apply(
        OneHotEncoderCalculator(),
        OneHotEncoderApplier(),
        imputed,
        {"columns": ["city", "plan_type"]},
    )

    # Original categorical columns are consumed (drop_original defaults to True).
    assert "city" not in encoded.columns
    assert "plan_type" not in encoded.columns

    city_dummy_cols = [c for c in encoded.columns if c.startswith("city_")]
    plan_dummy_cols = [c for c in encoded.columns if c.startswith("plan_type_")]
    assert len(city_dummy_cols) == n_city_categories
    assert len(plan_dummy_cols) == n_plan_categories

    dummy_cols = city_dummy_cols + plan_dummy_cols
    assert encoded[dummy_cols].isna().sum().sum() == 0
    # One-hot columns are 0/1 indicators.
    assert set(np.unique(encoded[dummy_cols].to_numpy())).issubset({0, 1})
    # Every row has exactly one active dummy per original categorical column.
    assert (encoded[city_dummy_cols].sum(axis=1) == 1).all()
    assert (encoded[plan_dummy_cols].sum(axis=1) == 1).all()


def test_four_step_chain_impute_scale_encode_interact() -> None:
    """A 4-step Imputer -> Scaler -> Encoder -> FeatureInteraction chain produces
    a fully-numeric, NaN-free DataFrame ready to feed into a model.
    """
    df = _load_dataset()

    imputed = _fit_apply(
        SimpleImputerCalculator(),
        SimpleImputerApplier(),
        df,
        {"strategy": "mean", "columns": ["age", "income"]},
    )
    imputed = _fit_apply(
        SimpleImputerCalculator(),
        SimpleImputerApplier(),
        imputed,
        {"strategy": "most_frequent", "columns": ["city"]},
    )

    scaled = _fit_apply(
        StandardScalerCalculator(),
        StandardScalerApplier(),
        imputed,
        {"columns": ["age", "income"]},
    )

    encoded = _fit_apply(
        OneHotEncoderCalculator(),
        OneHotEncoderApplier(),
        scaled,
        {"columns": ["city", "plan_type"]},
    )

    interacted = _fit_apply(
        FeatureInteractionCalculator(),
        FeatureInteractionApplier(),
        encoded,
        {"columns": ["age", "income"], "degree": 2, "interaction_only": True},
    )

    assert "age_x_income" in interacted.columns
    # The interaction is the product of the already-scaled columns.
    np.testing.assert_allclose(
        interacted["age_x_income"].to_numpy(),
        (interacted["age"] * interacted["income"]).to_numpy(),
    )

    # Drop non-numeric leftovers (signup_date, customer_id are not part of this
    # chain's concern) before asserting the chain's own outputs are clean.
    chain_related_cols = [
        c
        for c in interacted.columns
        if c in ("age", "income", "age_x_income")
        or c.startswith("city_")
        or c.startswith("plan_type_")
    ]
    assert interacted[chain_related_cols].isna().sum().sum() == 0
    for col in chain_related_cols:
        assert pd.api.types.is_numeric_dtype(interacted[col])


def test_geo_distance_chain_after_dropping_missing_coordinates() -> None:
    """Dropping rows with missing lat/lon, then computing GeoDistance to a fixed
    reference point, yields finite non-negative distances for every remaining row.
    """
    df = _load_dataset()
    assert df["lat"].isna().sum() > 0
    assert df["lon"].isna().sum() > 0

    complete = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    assert complete["lat"].isna().sum() == 0
    assert complete["lon"].isna().sum() == 0

    # Use the dataset's own centroid as the fixed reference point for every row.
    ref_lat = complete["lat"].mean()
    ref_lon = complete["lon"].mean()
    with_ref = complete.copy()
    with_ref["ref_lat"] = ref_lat
    with_ref["ref_lon"] = ref_lon

    params = GeoDistanceCalculator().fit(
        with_ref,
        {
            "lat1_col": "lat",
            "lon1_col": "lon",
            "lat2_col": "ref_lat",
            "lon2_col": "ref_lon",
            "method": "haversine",
            "unit": "km",
            "output_column": "distance_to_centroid_km",
        },
    )
    out = GeoDistanceApplier().apply(with_ref, dict(params))

    assert "distance_to_centroid_km" in out.columns
    distances = out["distance_to_centroid_km"].to_numpy()
    assert np.isfinite(distances).all()
    assert (distances >= 0).all()
    # The centroid itself is inside the data's own bounding box, so distances
    # should be well below "opposite side of the earth" (~20000km).
    assert distances.max() < 20000.0


def test_feature_selection_after_generation_keeps_targets() -> None:
    """VarianceThreshold applied after generating interaction features drops only
    the low-variance generated columns, leaving target columns untouched.

    The selector's ``columns`` config is scoped to generated features only, so
    ``monthly_spend``/``churned`` are never candidates for removal — this
    guards against a chain accidentally dropping the modeling target.
    """
    df = _load_dataset()
    imputed = _fit_apply(
        SimpleImputerCalculator(),
        SimpleImputerApplier(),
        df,
        {"strategy": "mean", "columns": ["age", "income"]},
    )

    generated = _fit_apply(
        FeatureInteractionCalculator(),
        FeatureInteractionApplier(),
        imputed,
        {
            "columns": ["age", "income", "tenure_months"],
            "degree": 2,
            "interaction_only": True,
        },
    )
    generated_feature_names = [
        "age_x_income",
        "age_x_tenure_months",
        "income_x_tenure_months",
    ]
    assert all(c in generated.columns for c in generated_feature_names)

    # Add one artificially constant column to prove the threshold actually filters.
    generated = generated.copy()
    generated["constant_feature"] = 1.0
    candidate_cols = generated_feature_names + ["constant_feature"]

    params = VarianceThresholdCalculator().fit(
        generated, {"columns": candidate_cols, "threshold": 0.0}
    )
    selected = VarianceThresholdApplier().apply(generated, dict(params))

    assert "constant_feature" not in selected.columns
    # Non-constant generated features survive a threshold of 0.0 (removes only
    # exactly-zero-variance columns).
    for name in generated_feature_names:
        assert name in selected.columns
    # Targets and other untouched columns are never candidates, so they remain.
    assert "monthly_spend" in selected.columns
    assert "churned" in selected.columns
    assert len(selected.columns) == len(generated.columns) - 1


def test_fit_on_train_slice_then_apply_only_on_test_slice_no_leakage() -> None:
    """Params fitted on one slice of data are reusable via apply()-only on a
    disjoint slice, without re-fitting — the core "fit once, apply many" contract.

    Splits the 300-row fixture into a train slice (first 200 rows) and a test
    slice (last 100 rows). The Imputer + Scaler chain is fit ONLY on train;
    applying the resulting params to test must use the *train* statistics, not
    statistics recomputed from test — proven by comparing against what the
    test slice's own (different) statistics would have produced.
    """
    df = _load_dataset()
    train_df = df.iloc[:200].reset_index(drop=True)
    test_df = df.iloc[200:].reset_index(drop=True)

    impute_config = {"strategy": "mean", "columns": ["age", "income"]}
    impute_params = SimpleImputerCalculator().fit(train_df, impute_config)
    train_imputed = SimpleImputerApplier().apply(train_df, dict(impute_params))
    test_imputed = SimpleImputerApplier().apply(test_df, dict(impute_params))

    # The fill value applied to test must be the TRAIN mean, not the test mean.
    train_mean_age = train_df["age"].mean()
    test_mean_age = test_df["age"].mean()
    assert not np.isclose(train_mean_age, test_mean_age, rtol=1e-6)
    filled_test_age_rows = test_df["age"].isna()
    assert filled_test_age_rows.any()
    np.testing.assert_allclose(
        test_imputed.loc[filled_test_age_rows, "age"].to_numpy(),
        train_mean_age,
        rtol=1e-8,
    )

    scale_config = {"columns": ["age", "income"]}
    scale_params = StandardScalerCalculator().fit(train_imputed, scale_config)
    test_scaled = StandardScalerApplier().apply(test_imputed, dict(scale_params))

    # Re-derive what test's own (leaked) mean/std would have produced, and
    # confirm the actual output does NOT match that leaked computation.
    leaked_mean = test_imputed["age"].to_numpy().mean()
    leaked_std = test_imputed["age"].to_numpy().std(ddof=0)
    leaked_scaled_age = (test_imputed["age"].to_numpy() - leaked_mean) / leaked_std

    train_mean = train_imputed["age"].to_numpy().mean()
    train_std = train_imputed["age"].to_numpy().std(ddof=0)
    expected_scaled_age = (test_imputed["age"].to_numpy() - train_mean) / train_std

    np.testing.assert_allclose(test_scaled["age"].to_numpy(), expected_scaled_age, rtol=1e-10)
    assert not np.allclose(test_scaled["age"].to_numpy(), leaked_scaled_age, rtol=1e-6)
    assert test_scaled[["age", "income"]].isna().sum().sum() == 0
