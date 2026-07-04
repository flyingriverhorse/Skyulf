"""Value-correctness and edge-case tests for the scaling nodes.

Complements ``test_engine_parity.py`` (which only checks pandas/polars fit
parity). These tests hand-verify the actual scaled numbers against the
scaler's documented formula and cover edge cases not exercised there: empty
frames, single-row frames, all-NaN columns, constant (zero-variance/zero-
range) columns, out-of-range values seen only at apply time, and the full
fit -> apply round trip via the public Calculator/Applier API.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from skyulf.preprocessing.scaling.maxabs import (
    MaxAbsScalerApplier,
    MaxAbsScalerCalculator,
)
from skyulf.preprocessing.scaling.minmax import (
    MinMaxScalerApplier,
    MinMaxScalerCalculator,
)
from skyulf.preprocessing.scaling.robust import (
    RobustScalerApplier,
    RobustScalerCalculator,
)
from skyulf.preprocessing.scaling.standard import (
    StandardScalerApplier,
    StandardScalerCalculator,
)


def _fit_apply(
    calculator: Any, applier: Any, df: pd.DataFrame, config: Dict[str, Any]
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Run fit then apply and return (transformed_df, params)."""
    params = calculator.fit(df, config)
    out = applier.apply(df, params)
    return out, params


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------


def test_standard_scaler_matches_manual_zscore() -> None:
    """Standard scaler output equals (x - mean) / population std."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out, params = _fit_apply(
        StandardScalerCalculator(), StandardScalerApplier(), df, {"columns": ["a"]}
    )

    mean = df["a"].to_numpy().mean()
    std = df["a"].to_numpy().std(ddof=0)
    expected = (df["a"].to_numpy() - mean) / std

    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)
    assert params["mean"] == pytest.approx([mean])
    assert params["scale"] == pytest.approx([std])


def test_standard_scaler_with_mean_false_only_scales() -> None:
    """with_mean=False skips centering and divides by std only."""
    df = pd.DataFrame({"a": [2.0, 4.0, 6.0, 8.0]})
    config = {"columns": ["a"], "with_mean": False}
    out, params = _fit_apply(StandardScalerCalculator(), StandardScalerApplier(), df, config)

    std = df["a"].to_numpy().std(ddof=0)
    expected = df["a"].to_numpy() / std
    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_standard_scaler_with_std_false_still_centers() -> None:
    """with_std=False makes fit's scale=None (sklearn behavior), but centering
    (with_mean, default True) must still be applied at apply time — the
    early-return guard only bails on a missing artifact the active flag
    actually needs, not on `scale is None` unconditionally.
    """
    df = pd.DataFrame({"a": [2.0, 4.0, 6.0, 8.0]})
    config = {"columns": ["a"], "with_std": False}
    out, params = _fit_apply(StandardScalerCalculator(), StandardScalerApplier(), df, config)

    assert params["scale"] is None
    mean = df["a"].to_numpy().mean()
    expected = df["a"].to_numpy() - mean
    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_standard_scaler_constant_column_has_unit_scale() -> None:
    """A zero-variance column gets scale=1 (sklearn's zero-guard) so output is 0."""
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    out, params = _fit_apply(
        StandardScalerCalculator(), StandardScalerApplier(), df, {"columns": ["a"]}
    )

    assert params["scale"] == pytest.approx([1.0])
    assert params["var"] == pytest.approx([0.0])
    np.testing.assert_allclose(out["a"].to_numpy(), [0.0, 0.0, 0.0])


def test_standard_scaler_single_row() -> None:
    """A single-row fit yields var=0/scale=1 and centers the lone value to 0."""
    df = pd.DataFrame({"a": [3.0], "b": [7.0]})
    out, params = _fit_apply(
        StandardScalerCalculator(), StandardScalerApplier(), df, {"columns": ["a", "b"]}
    )

    assert params["mean"] == pytest.approx([3.0, 7.0])
    assert params["scale"] == pytest.approx([1.0, 1.0])
    np.testing.assert_allclose(out[["a", "b"]].to_numpy(), [[0.0, 0.0]])


def test_standard_scaler_all_nan_column_propagates_nan() -> None:
    """An all-NaN column fits to NaN stats and stays NaN after apply."""
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
    out, params = _fit_apply(
        StandardScalerCalculator(), StandardScalerApplier(), df, {"columns": ["a"]}
    )

    assert np.isnan(params["mean"][0])
    assert np.isnan(out["a"].to_numpy()).all()


def test_standard_scaler_empty_dataframe_raises() -> None:
    """Fitting on zero rows raises, matching sklearn's own minimum-sample check."""
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    with pytest.raises(ValueError):
        StandardScalerCalculator().fit(df, {"columns": ["a"]})


def test_standard_scaler_out_of_range_values_at_apply_time() -> None:
    """Values outside the fit-time range are transformed with the fitted formula, not clipped."""
    fit_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    params = StandardScalerCalculator().fit(fit_df, {"columns": ["a"]})

    apply_df = pd.DataFrame({"a": [-100.0, 100.0]})
    out = StandardScalerApplier().apply(apply_df, params)

    mean = fit_df["a"].to_numpy().mean()
    std = fit_df["a"].to_numpy().std(ddof=0)
    expected = (apply_df["a"].to_numpy() - mean) / std
    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_standard_scaler_no_columns_selected_is_noop() -> None:
    """An explicit empty `columns` list short-circuits fit to {} and apply is a no-op."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    params = StandardScalerCalculator().fit(df, {"columns": []})
    assert params == {}

    out = StandardScalerApplier().apply(df, params)
    pd.testing.assert_frame_equal(out, df)


def test_standard_scaler_fit_transform_round_trip() -> None:
    """StatefulTransformer.fit_transform matches calling fit then apply directly."""
    from skyulf.preprocessing.base import StatefulTransformer

    df = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0]})
    config = {"columns": ["a"]}

    direct_params = StandardScalerCalculator().fit(df, config)
    direct_out = StandardScalerApplier().apply(df, direct_params)

    transformer = StatefulTransformer(
        StandardScalerCalculator(), StandardScalerApplier(), node_id="StandardScaler"
    )
    piped_out = transformer.fit_transform(df, config)

    pd.testing.assert_frame_equal(piped_out, direct_out)
    assert transformer.params == direct_params


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------


def test_minmax_scaler_matches_manual_formula_default_range() -> None:
    """Default feature_range=[0, 1] matches (x - min) / (max - min)."""
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0]})
    out, params = _fit_apply(
        MinMaxScalerCalculator(), MinMaxScalerApplier(), df, {"columns": ["a"]}
    )

    data_min, data_max = df["a"].min(), df["a"].max()
    expected = (df["a"].to_numpy() - data_min) / (data_max - data_min)
    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)
    assert params["data_min"] == pytest.approx([data_min])
    assert params["data_max"] == pytest.approx([data_max])


def test_minmax_scaler_custom_feature_range() -> None:
    """feature_range=[-1, 1] rescales into [-1, 1] using the same min/max stats."""
    df = pd.DataFrame({"a": [0.0, 5.0, 10.0]})
    config = {"columns": ["a"], "feature_range": [-1, 1]}
    out, params = _fit_apply(MinMaxScalerCalculator(), MinMaxScalerApplier(), df, config)

    data_min, data_max = df["a"].min(), df["a"].max()
    scale = (1 - (-1)) / (data_max - data_min)
    expected = (df["a"].to_numpy() - data_min) * scale - 1
    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_minmax_scaler_constant_column() -> None:
    """A constant column has data_min==data_max and maps every value to feature_range min."""
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    out, params = _fit_apply(
        MinMaxScalerCalculator(), MinMaxScalerApplier(), df, {"columns": ["a"]}
    )

    assert params["data_min"] == pytest.approx([5.0])
    assert params["data_max"] == pytest.approx([5.0])
    np.testing.assert_allclose(out["a"].to_numpy(), [0.0, 0.0, 0.0])


def test_minmax_scaler_out_of_range_values_at_apply_time() -> None:
    """Values beyond the fitted [min, max] range extrapolate outside [0, 1], no clipping."""
    fit_df = pd.DataFrame({"a": [0.0, 10.0]})
    params = MinMaxScalerCalculator().fit(fit_df, {"columns": ["a"]})

    apply_df = pd.DataFrame({"a": [-10.0, 20.0]})
    out = MinMaxScalerApplier().apply(apply_df, params)

    expected = (apply_df["a"].to_numpy() - 0.0) / (10.0 - 0.0)
    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)
    assert out["a"].iloc[0] < 0
    assert out["a"].iloc[1] > 1


def test_minmax_scaler_single_row() -> None:
    """A single-row fit treats min==max and maps the lone value to the range floor."""
    df = pd.DataFrame({"a": [42.0]})
    out, params = _fit_apply(
        MinMaxScalerCalculator(), MinMaxScalerApplier(), df, {"columns": ["a"]}
    )
    np.testing.assert_allclose(out["a"].to_numpy(), [0.0])


def test_minmax_scaler_empty_dataframe_raises() -> None:
    """Fitting on zero rows raises, matching sklearn's own minimum-sample check."""
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    with pytest.raises(ValueError):
        MinMaxScalerCalculator().fit(df, {"columns": ["a"]})


def test_minmax_scaler_no_columns_selected_is_noop() -> None:
    """An explicit empty `columns` list short-circuits fit to {} and apply is a no-op."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    params = MinMaxScalerCalculator().fit(df, {"columns": []})
    assert params == {}
    out = MinMaxScalerApplier().apply(df, params)
    pd.testing.assert_frame_equal(out, df)


def test_minmax_scaler_fit_transform_round_trip() -> None:
    """StatefulTransformer.fit_transform matches calling fit then apply directly."""
    from skyulf.preprocessing.base import StatefulTransformer

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0]})
    config = {"columns": ["a"]}

    direct_params = MinMaxScalerCalculator().fit(df, config)
    direct_out = MinMaxScalerApplier().apply(df, direct_params)

    transformer = StatefulTransformer(
        MinMaxScalerCalculator(), MinMaxScalerApplier(), node_id="MinMaxScaler"
    )
    piped_out = transformer.fit_transform(df, config)

    pd.testing.assert_frame_equal(piped_out, direct_out)


# ---------------------------------------------------------------------------
# MaxAbsScaler
# ---------------------------------------------------------------------------


def test_maxabs_scaler_matches_manual_formula() -> None:
    """MaxAbsScaler output equals x / max(abs(x))."""
    df = pd.DataFrame({"a": [-4.0, 2.0, -8.0, 6.0]})
    out, params = _fit_apply(
        MaxAbsScalerCalculator(), MaxAbsScalerApplier(), df, {"columns": ["a"]}
    )

    max_abs = np.abs(df["a"].to_numpy()).max()
    expected = df["a"].to_numpy() / max_abs
    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)
    assert params["max_abs"] == pytest.approx([max_abs])
    assert params["scale"] == pytest.approx([max_abs])


def test_maxabs_scaler_all_zero_column_scale_guarded_to_one() -> None:
    """An all-zero column has max_abs=0; scale is guarded to 1 so output stays 0."""
    df = pd.DataFrame({"a": [0.0, 0.0, 0.0]})
    out, params = _fit_apply(
        MaxAbsScalerCalculator(), MaxAbsScalerApplier(), df, {"columns": ["a"]}
    )

    assert params["max_abs"] == pytest.approx([0.0])
    assert params["scale"] == pytest.approx([1.0])
    np.testing.assert_allclose(out["a"].to_numpy(), [0.0, 0.0, 0.0])


def test_maxabs_scaler_out_of_range_values_at_apply_time() -> None:
    """Apply-time values larger than the fit-time max abs scale beyond [-1, 1]."""
    fit_df = pd.DataFrame({"a": [-5.0, 5.0]})
    params = MaxAbsScalerCalculator().fit(fit_df, {"columns": ["a"]})

    apply_df = pd.DataFrame({"a": [-10.0, 10.0]})
    out = MaxAbsScalerApplier().apply(apply_df, params)

    np.testing.assert_allclose(out["a"].to_numpy(), [-2.0, 2.0])


def test_maxabs_scaler_single_row() -> None:
    """A single-row fit divides the lone value by its own absolute value (unit magnitude)."""
    df = pd.DataFrame({"a": [-3.0]})
    out, params = _fit_apply(
        MaxAbsScalerCalculator(), MaxAbsScalerApplier(), df, {"columns": ["a"]}
    )
    np.testing.assert_allclose(out["a"].to_numpy(), [-1.0])


def test_maxabs_scaler_no_columns_selected_is_noop() -> None:
    """An explicit empty `columns` list short-circuits fit to {} and apply is a no-op."""
    df = pd.DataFrame({"a": [1.0, -2.0, 3.0]})
    params = MaxAbsScalerCalculator().fit(df, {"columns": []})
    assert params == {}
    out = MaxAbsScalerApplier().apply(df, params)
    pd.testing.assert_frame_equal(out, df)


def test_maxabs_scaler_fit_transform_round_trip() -> None:
    """StatefulTransformer.fit_transform matches calling fit then apply directly."""
    from skyulf.preprocessing.base import StatefulTransformer

    df = pd.DataFrame({"a": [-9.0, 3.0, 6.0, -1.0]})
    config = {"columns": ["a"]}

    direct_params = MaxAbsScalerCalculator().fit(df, config)
    direct_out = MaxAbsScalerApplier().apply(df, direct_params)

    transformer = StatefulTransformer(
        MaxAbsScalerCalculator(), MaxAbsScalerApplier(), node_id="MaxAbsScaler"
    )
    piped_out = transformer.fit_transform(df, config)

    pd.testing.assert_frame_equal(piped_out, direct_out)


# ---------------------------------------------------------------------------
# RobustScaler
# ---------------------------------------------------------------------------


def test_robust_scaler_matches_manual_median_iqr() -> None:
    """RobustScaler output equals (x - median) / IQR(25, 75) by default."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out, params = _fit_apply(
        RobustScalerCalculator(), RobustScalerApplier(), df, {"columns": ["a"]}
    )

    arr = df["a"].to_numpy()
    median = np.median(arr)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    expected = (arr - median) / iqr

    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)
    assert params["center"] == pytest.approx([median])
    assert params["scale"] == pytest.approx([iqr])


def test_robust_scaler_custom_quantile_range() -> None:
    """A custom quantile_range changes the IQR used for scaling."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    config = {"columns": ["a"], "quantile_range": [10.0, 90.0]}
    out, params = _fit_apply(RobustScalerCalculator(), RobustScalerApplier(), df, config)

    arr = df["a"].to_numpy()
    median = np.median(arr)
    q10, q90 = np.percentile(arr, [10, 90])
    expected = (arr - median) / (q90 - q10)

    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)
    assert params["quantile_range"] == (10.0, 90.0)


def test_robust_scaler_with_centering_false() -> None:
    """with_centering=False skips subtracting the median."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    config = {"columns": ["a"], "with_centering": False}
    out, params = _fit_apply(RobustScalerCalculator(), RobustScalerApplier(), df, config)

    arr = df["a"].to_numpy()
    q1, q3 = np.percentile(arr, [25, 75])
    expected = arr / (q3 - q1)
    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_robust_scaler_with_scaling_false() -> None:
    """with_scaling=False skips dividing by the IQR."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    config = {"columns": ["a"], "with_scaling": False}
    out, params = _fit_apply(RobustScalerCalculator(), RobustScalerApplier(), df, config)

    arr = df["a"].to_numpy()
    median = np.median(arr)
    expected = arr - median
    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_robust_scaler_constant_column_scale_guarded_to_one() -> None:
    """A constant column has IQR=0; scale is guarded to 1 so output stays 0."""
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    out, params = _fit_apply(
        RobustScalerCalculator(), RobustScalerApplier(), df, {"columns": ["a"]}
    )

    assert params["center"] == pytest.approx([5.0])
    assert params["scale"] == pytest.approx([1.0])
    np.testing.assert_allclose(out["a"].to_numpy(), [0.0, 0.0, 0.0])


def test_robust_scaler_single_row() -> None:
    """A single-row fit centers to its own value with a guarded scale of 1."""
    df = pd.DataFrame({"a": [3.0], "b": [7.0]})
    out, params = _fit_apply(
        RobustScalerCalculator(), RobustScalerApplier(), df, {"columns": ["a", "b"]}
    )
    assert params["center"] == pytest.approx([3.0, 7.0])
    assert params["scale"] == pytest.approx([1.0, 1.0])
    np.testing.assert_allclose(out[["a", "b"]].to_numpy(), [[0.0, 0.0]])


def test_robust_scaler_out_of_range_values_at_apply_time() -> None:
    """Apply-time outliers beyond the fit-time IQR bounds are not clipped."""
    fit_df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    params = RobustScalerCalculator().fit(fit_df, {"columns": ["a"]})

    apply_df = pd.DataFrame({"a": [-100.0, 100.0]})
    out = RobustScalerApplier().apply(apply_df, params)

    arr = fit_df["a"].to_numpy()
    median = np.median(arr)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    expected = (apply_df["a"].to_numpy() - median) / iqr
    np.testing.assert_allclose(out["a"].to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_robust_scaler_empty_dataframe_raises() -> None:
    """Fitting on zero rows raises, matching sklearn's own minimum-sample check."""
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    with pytest.raises(ValueError):
        RobustScalerCalculator().fit(df, {"columns": ["a"]})


def test_robust_scaler_all_nan_column_propagates_nan() -> None:
    """An all-NaN column fits to NaN stats and stays NaN after apply."""
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan, np.nan]})
    out, params = _fit_apply(
        RobustScalerCalculator(), RobustScalerApplier(), df, {"columns": ["a"]}
    )
    assert np.isnan(params["center"][0])
    assert np.isnan(out["a"].to_numpy()).all()


def test_robust_scaler_no_columns_selected_is_noop() -> None:
    """An explicit empty `columns` list short-circuits fit to {} and apply is a no-op."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    params = RobustScalerCalculator().fit(df, {"columns": []})
    assert params == {}
    out = RobustScalerApplier().apply(df, params)
    pd.testing.assert_frame_equal(out, df)


def test_robust_scaler_missing_fit_column_at_apply_is_noop() -> None:
    """Applying params whose fitted column is absent from the apply-time frame is a no-op."""
    fit_df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    params = RobustScalerCalculator().fit(fit_df, {"columns": ["a"]})

    other_df = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
    out = RobustScalerApplier().apply(other_df, params)
    pd.testing.assert_frame_equal(out, other_df)


def test_robust_scaler_fit_transform_round_trip() -> None:
    """StatefulTransformer.fit_transform matches calling fit then apply directly."""
    from skyulf.preprocessing.base import StatefulTransformer

    df = pd.DataFrame({"a": [1.0, 5.0, 3.0, 9.0, 2.0, 7.0]})
    config = {"columns": ["a"]}

    direct_params = RobustScalerCalculator().fit(df, config)
    direct_out = RobustScalerApplier().apply(df, direct_params)

    transformer = StatefulTransformer(
        RobustScalerCalculator(), RobustScalerApplier(), node_id="RobustScaler"
    )
    piped_out = transformer.fit_transform(df, config)

    pd.testing.assert_frame_equal(piped_out, direct_out)
    assert transformer.params == direct_params


# ---------------------------------------------------------------------------
# Multi-column real-world-ish fixture (reuses conftest fixture, hand-verified)
# ---------------------------------------------------------------------------


def test_standard_scaler_on_sample_regression_data_matches_manual(
    sample_regression_data: pd.DataFrame,
) -> None:
    """Standard scaler on the shared regression fixture matches manual z-score per column."""
    cols = ["feature1", "feature2"]
    df = sample_regression_data
    out, params = _fit_apply(
        StandardScalerCalculator(), StandardScalerApplier(), df, {"columns": cols}
    )

    for i, col in enumerate(cols):
        mean = df[col].to_numpy().mean()
        std = df[col].to_numpy().std(ddof=0)
        expected = (df[col].to_numpy() - mean) / std
        np.testing.assert_allclose(out[col].to_numpy(), expected, rtol=1e-9, atol=1e-9)
        assert params["mean"][i] == pytest.approx(mean)
        assert params["scale"][i] == pytest.approx(std)
