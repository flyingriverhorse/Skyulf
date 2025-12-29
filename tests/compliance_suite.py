import pytest
import pandas as pd
import numpy as np
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from skyulf.preprocessing.scaling import (
    StandardScalerCalculator, StandardScalerApplier,
    MinMaxScalerCalculator, MinMaxScalerApplier,
    RobustScalerCalculator, RobustScalerApplier,
    MaxAbsScalerCalculator, MaxAbsScalerApplier
)
from skyulf.preprocessing.imputation import (
    SimpleImputerCalculator, SimpleImputerApplier,
    KNNImputerCalculator, KNNImputerApplier,
    IterativeImputerCalculator, IterativeImputerApplier
)
from skyulf.preprocessing.encoding import (
    OneHotEncoderCalculator, OneHotEncoderApplier,
    OrdinalEncoderCalculator, OrdinalEncoderApplier,
    TargetEncoderCalculator, TargetEncoderApplier,
    HashEncoderCalculator, HashEncoderApplier,
    DummyEncoderCalculator, DummyEncoderApplier
)
from skyulf.preprocessing.bucketing import (
    GeneralBinningCalculator, GeneralBinningApplier,
    KBinsDiscretizerCalculator, KBinsDiscretizerApplier,
    CustomBinningCalculator, CustomBinningApplier
)
from skyulf.preprocessing.outliers import (
    IQRCalculator, IQRApplier,
    ZScoreCalculator, ZScoreApplier,
    WinsorizeCalculator, WinsorizeApplier,
    ManualBoundsCalculator, ManualBoundsApplier,
    EllipticEnvelopeCalculator, EllipticEnvelopeApplier
)
from skyulf.engines import get_engine

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_standard_scaler_parity():
    print("Starting StandardScaler Parity Test...")
    # Data
    data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["a", "b"], "with_mean": True, "with_std": True}
    
    # Fit Pandas
    calc = StandardScalerCalculator()
    params_pd = calc.fit(df_pd, config)
    
    # Fit Polars
    params_pl = calc.fit(df_pl, config)
    
    # Assert Params Equality
    assert np.allclose(params_pd["mean"], params_pl["mean"])
    assert np.allclose(params_pd["scale"], params_pl["scale"])
    
    # Apply Pandas
    applier = StandardScalerApplier()
    res_pd = applier.apply(df_pd, params_pd)
    
    # Apply Polars
    res_pl = applier.apply(df_pl, params_pl)
    
    # Convert Polars result to Pandas for comparison
    res_pl_pd = res_pl.to_pandas()
    
    # Assert Result Equality
    pd.testing.assert_frame_equal(res_pd, res_pl_pd)
    
    print("StandardScaler Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_minmax_scaler_parity():
    print("Starting MinMaxScaler Parity Test...")
    data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["a", "b"], "feature_range": (0, 1)}
    
    calc = MinMaxScalerCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    assert np.allclose(params_pd["min"], params_pl["min"])
    assert np.allclose(params_pd["scale"], params_pl["scale"])
    
    applier = MinMaxScalerApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas())
    print("MinMaxScaler Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_robust_scaler_parity():
    print("Starting RobustScaler Parity Test...")
    data = {"a": [1.0, 2.0, 3.0, 100.0, 5.0], "b": [10.0, 20.0, 30.0, 400.0, 50.0]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["a", "b"], "quantile_range": (25.0, 75.0)}
    
    calc = RobustScalerCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    assert np.allclose(params_pd["center"], params_pl["center"])
    assert np.allclose(params_pd["scale"], params_pl["scale"])
    
    applier = RobustScalerApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas())
    print("RobustScaler Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_maxabs_scaler_parity():
    print("Starting MaxAbsScaler Parity Test...")
    data = {"a": [1.0, -2.0, 3.0, -4.0, 5.0], "b": [10.0, -20.0, 30.0, -40.0, 50.0]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["a", "b"]}
    
    calc = MaxAbsScalerCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    assert np.allclose(params_pd["scale"], params_pl["scale"])
    
    applier = MaxAbsScalerApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas())
    print("MaxAbsScaler Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_simple_imputer_parity():
    print("Starting SimpleImputer Parity Test...")
    data = {
        "a": [1.0, 2.0, None, 4.0, 5.0], 
        "b": [10.0, None, 30.0, 40.0, 50.0],
        "c": ["x", "y", "x", None, "z"]
    }
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    # Test Mean
    config_mean = {"columns": ["a", "b"], "strategy": "mean"}
    calc = SimpleImputerCalculator()
    params_pd = calc.fit(df_pd, config_mean)
    params_pl = calc.fit(df_pl, config_mean)
    
    # Check fill values
    assert np.allclose(params_pd["fill_values"]["a"], params_pl["fill_values"]["a"])
    
    applier = SimpleImputerApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    pd.testing.assert_frame_equal(res_pd[["a", "b"]], res_pl.to_pandas()[["a", "b"]])
    
    # Test Most Frequent (Mode)
    config_mode = {"columns": ["c"], "strategy": "most_frequent"}
    params_pd_mode = calc.fit(df_pd, config_mode)
    params_pl_mode = calc.fit(df_pl, config_mode)
    
    assert params_pd_mode["fill_values"]["c"] == params_pl_mode["fill_values"]["c"]
    
    res_pd_mode = applier.apply(df_pd, params_pd_mode)
    res_pl_mode = applier.apply(df_pl, params_pl_mode)
    
    pd.testing.assert_frame_equal(res_pd_mode[["c"]], res_pl_mode.to_pandas()[["c"]])

    print("SimpleImputer Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_onehot_encoder_parity():
    print("Starting OneHotEncoder Parity Test...")
    data = {
        "cat": ["a", "b", "a", "c", "b"],
        "num": [1, 2, 3, 4, 5]
    }
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["cat"], "drop_first": False, "drop_original": True}
    
    calc = OneHotEncoderCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    # Check feature names
    assert params_pd["feature_names"] == params_pl["feature_names"]
    
    applier = OneHotEncoderApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    # Sort columns to ensure order match
    res_pl_pd = res_pl.to_pandas()
    
    # Ensure columns match
    assert set(res_pd.columns) == set(res_pl_pd.columns)
    
    # Check values
    pd.testing.assert_frame_equal(res_pd.sort_index(axis=1), res_pl_pd.sort_index(axis=1))
    print("OneHotEncoder Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_ordinal_encoder_parity():
    print("Starting OrdinalEncoder Parity Test...")
    data = {
        "cat": ["a", "b", "a", "c", "b"],
        "num": [1, 2, 3, 4, 5]
    }
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["cat"]}
    
    calc = OrdinalEncoderCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    applier = OrdinalEncoderApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas())
    print("OrdinalEncoder Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_target_encoder_parity():
    print("Starting TargetEncoder Parity Test...")
    data = {
        "cat": ["a", "b", "a", "c", "b"],
        "num": [1, 2, 3, 4, 5]
    }
    y = [10.0, 20.0, 10.0, 30.0, 20.0]
    
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    y_pd = pd.Series(y, name="target")
    y_pl = pl.Series("target", y)
    
    config = {"columns": ["cat"], "smooth": 1.0, "target_type": "continuous"}
    
    calc = TargetEncoderCalculator()
    params_pd = calc.fit((df_pd, y_pd), config)
    params_pl = calc.fit((df_pl, y_pl), config)
    
    applier = TargetEncoderApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas())
    print("TargetEncoder Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_hash_encoder_parity():
    print("Starting HashEncoder Parity Test...")
    data = {
        "cat": ["a", "b", "a", "c", "b"],
        "num": [1, 2, 3, 4, 5]
    }
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["cat"], "n_features": 5}
    
    calc = HashEncoderCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    applier = HashEncoderApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    # Note: Polars hash() and Python hash() might differ.
    # We check if the structure is correct and values are integers.
    res_pl_pd = res_pl.to_pandas()
    assert res_pd.shape == res_pl_pd.shape
    assert pd.api.types.is_integer_dtype(res_pl_pd["cat"])
    
    print("HashEncoder Parity Test Passed (Structure Only)!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_dummy_encoder_parity():
    print("Starting DummyEncoder Parity Test...")
    data = {
        "cat": ["a", "b", "a", "c", "b"],
        "num": [1, 2, 3, 4, 5]
    }
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["cat"], "drop_first": False}
    
    calc = DummyEncoderCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    assert sorted(params_pd["categories"]["cat"]) == sorted(params_pl["categories"]["cat"])
    
    applier = DummyEncoderApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    # Sort columns to ensure order match
    res_pl_pd = res_pl.to_pandas()
    
    # Ensure columns match
    assert set(res_pd.columns) == set(res_pl_pd.columns)
    
    # Check values
    # Cast to int because Polars might return Int8/Int32/Int64
    pd.testing.assert_frame_equal(
        res_pd.sort_index(axis=1).astype(int), 
        res_pl_pd.sort_index(axis=1).astype(int)
    )
    print("DummyEncoder Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_knn_imputer_parity():
    print("Starting KNNImputer Parity Test...")
    data = {
        "a": [1.0, 2.0, None, 4.0, 5.0], 
        "b": [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["a", "b"], "n_neighbors": 2}
    
    calc = KNNImputerCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    applier = KNNImputerApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas())
    print("KNNImputer Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_iterative_imputer_parity():
    print("Starting IterativeImputer Parity Test...")
    data = {
        "a": [1.0, 2.0, None, 4.0, 5.0], 
        "b": [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["a", "b"], "max_iter": 5}
    
    calc = IterativeImputerCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    applier = IterativeImputerApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas())
    print("IterativeImputer Parity Test Passed!")



@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_bucketing_parity():
    print("Starting Bucketing Parity Test...")
    data = {"a": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    # KBinsDiscretizer
    calc = KBinsDiscretizerCalculator()
    config = {"columns": ["a"], "n_bins": 3, "strategy": "uniform", "encode": "ordinal"}
    params = calc.fit(df_pd, config)
    
    applier = KBinsDiscretizerApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    # Convert Polars to Pandas for comparison
    res_pl_pd = res_pl.to_pandas()
    
    # Check columns
    assert "a_binned" in res_pd.columns
    assert "a_binned" in res_pl_pd.columns
    
    # Check values (might be categorical or int depending on implementation)
    # Our implementation returns UInt32 for ordinal in Polars, and float/int in Pandas?
    # Let's check dtypes.
    # Pandas pd.cut with labels=False returns integers (int8/16/32/64).
    # Polars cast(pl.UInt32) returns UInt32.
    
    # We compare values.
    pd.testing.assert_series_equal(
        res_pd["a_binned"].astype(float), 
        res_pl_pd["a_binned"].astype(float), 
        check_names=False, check_dtype=False
    )
    print("Bucketing Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_iqr_parity():
    print("Starting IQR Parity Test...")
    data = {"a": [1, 2, 3, 4, 5, 100]} # 100 is outlier
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    calc = IQRCalculator()
    config = {"columns": ["a"], "multiplier": 1.5}
    params = calc.fit(df_pd, config)
    
    applier = IQRApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    res_pl_pd = res_pl.to_pandas()
    
    # Should remove 100
    assert len(res_pd) == 5
    assert len(res_pl_pd) == 5
    assert 100 not in res_pd["a"].values
    
    pd.testing.assert_frame_equal(res_pd.reset_index(drop=True), res_pl_pd.reset_index(drop=True), check_dtype=False)
    print("IQR Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_zscore_parity():
    print("Starting ZScore Parity Test...")
    data = {"a": [0, 0, 0, 0, 100]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    calc = ZScoreCalculator()
    config = {"columns": ["a"], "threshold": 1.5}
    params = calc.fit(df_pd, config)
    
    applier = ZScoreApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    res_pl_pd = res_pl.to_pandas()
    
    assert len(res_pd) == 4
    pd.testing.assert_frame_equal(res_pd.reset_index(drop=True), res_pl_pd.reset_index(drop=True), check_dtype=False)
    print("ZScore Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_winsorize_parity():
    print("Starting Winsorize Parity Test...")
    data = {"a": range(100)}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    calc = WinsorizeCalculator()
    config = {"columns": ["a"], "lower_percentile": 5, "upper_percentile": 95}
    params = calc.fit(df_pd, config)
    
    applier = WinsorizeApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    res_pl_pd = res_pl.to_pandas()
    
    pd.testing.assert_frame_equal(res_pd, res_pl_pd, check_dtype=False)
    print("Winsorize Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_manual_bounds_parity():
    print("Starting ManualBounds Parity Test...")
    data = {"a": [1, 5, 10, 15]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    calc = ManualBoundsCalculator()
    config = {"bounds": {"a": {"lower": 4, "upper": 12}}}
    params = calc.fit(df_pd, config)
    
    applier = ManualBoundsApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    res_pl_pd = res_pl.to_pandas()
    
    assert len(res_pd) == 2 # 5, 10
    pd.testing.assert_frame_equal(res_pd.reset_index(drop=True), res_pl_pd.reset_index(drop=True), check_dtype=False)
    print("ManualBounds Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_general_binning_parity():
    print("Starting GeneralBinning Parity Test...")
    data = {"a": np.linspace(0, 100, 20)}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    calc = GeneralBinningCalculator()
    
    # Test Equal Width
    config_width = {"columns": ["a"], "strategy": "equal_width", "n_bins": 4}
    params_width = calc.fit(df_pd, config_width)
    
    applier = GeneralBinningApplier()
    res_pd = applier.apply(df_pd, params_width)
    res_pl = applier.apply(df_pl, params_width)
    
    pd.testing.assert_series_equal(
        res_pd["a_binned"].astype(float), 
        res_pl.to_pandas()["a_binned"].astype(float), 
        check_names=False, check_dtype=False
    )
    
    # Test Equal Frequency
    config_freq = {"columns": ["a"], "strategy": "equal_frequency", "n_bins": 4}
    params_freq = calc.fit(df_pd, config_freq)
    
    res_pd_freq = applier.apply(df_pd, params_freq)
    res_pl_freq = applier.apply(df_pl, params_freq)
    
    pd.testing.assert_series_equal(
        res_pd_freq["a_binned"].astype(float), 
        res_pl_freq.to_pandas()["a_binned"].astype(float), 
        check_names=False, check_dtype=False
    )
    print("GeneralBinning Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_custom_binning_parity():
    print("Starting CustomBinning Parity Test...")
    data = {"a": [1, 5, 15, 25, 35]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    calc = CustomBinningCalculator()
    # Bins: 0-10, 10-20, 20-30, 30-40
    config = {"columns": ["a"], "bins": [0, 10, 20, 30, 40]}
    params = calc.fit(df_pd, config)
    
    applier = CustomBinningApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    pd.testing.assert_series_equal(
        res_pd["a_binned"].astype(float), 
        res_pl.to_pandas()["a_binned"].astype(float), 
        check_names=False, check_dtype=False
    )
    print("CustomBinning Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_elliptic_envelope_parity():
    print("Starting EllipticEnvelope Parity Test...")
    # Generate data with a clear outlier
    rng = np.random.RandomState(42)
    X = 0.3 * rng.randn(20, 2)
    X_outliers = rng.uniform(low=-4, high=4, size=(5, 2))
    X = np.r_[X + 2, X - 2, X_outliers]
    
    data = {"a": X[:, 0], "b": X[:, 1]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    calc = EllipticEnvelopeCalculator()
    config = {"columns": ["a", "b"], "contamination": 0.1}
    params = calc.fit(df_pd, config)
    
    applier = EllipticEnvelopeApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    # Since EllipticEnvelope is deterministic with fixed random_state (but we didn't set it in the node),
    # we might have slight variations if the fit wasn't controlled. 
    # However, we fit ONCE on pandas (in the test setup above, we fit params on df_pd).
    # Wait, we fit params on df_pd for both.
    # The applier uses the fitted model.
    # The model.predict() should be deterministic for the same input.
    
    pd.testing.assert_frame_equal(res_pd.reset_index(drop=True), res_pl.to_pandas().reset_index(drop=True), check_dtype=False)
    print("EllipticEnvelope Parity Test Passed!")

if __name__ == "__main__":
    print(f"Has Polars: {HAS_POLARS}")
    if HAS_POLARS:
        test_standard_scaler_parity()
        test_minmax_scaler_parity()
        test_robust_scaler_parity()
        test_maxabs_scaler_parity()
        test_simple_imputer_parity()
        test_onehot_encoder_parity()
        test_ordinal_encoder_parity()
        test_target_encoder_parity()
        test_hash_encoder_parity()
        test_dummy_encoder_parity()
        test_knn_imputer_parity()
        test_iterative_imputer_parity()
        test_bucketing_parity()
        test_iqr_parity()
        test_zscore_parity()
        test_winsorize_parity()
        test_manual_bounds_parity()
        test_general_binning_parity()
        test_custom_binning_parity()
        test_elliptic_envelope_parity()
    else:
        print("Skipping test because Polars is not installed.")
