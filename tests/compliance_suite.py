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
from skyulf.preprocessing.casting import (
    CastingCalculator, CastingApplier
)
from skyulf.preprocessing.cleaning import (
    TextCleaningApplier,
    ValueReplacementApplier,
    AliasReplacementApplier,
    InvalidValueReplacementApplier
)
from skyulf.preprocessing.imputation import (
    SimpleImputerCalculator, SimpleImputerApplier,
    KNNImputerCalculator, KNNImputerApplier,
    IterativeImputerCalculator, IterativeImputerApplier
)
from skyulf.preprocessing.drop_and_missing import (
    DeduplicateCalculator, DeduplicateApplier,
    DropMissingColumnsCalculator, DropMissingColumnsApplier,
    DropMissingRowsApplier,
    MissingIndicatorCalculator, MissingIndicatorApplier
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

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_casting_parity():
    print("Starting Casting Parity Test...")
    data = {"a": ["1", "2", "3"], "b": [1, 2, 3], "c": ["x", "y", "z"]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["a", "b", "c"], "target_type": "float"} # This config is for Calculator fit, but CastingCalculator fit just returns config
    
    # CastingCalculator fit
    calc = CastingCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    assert params_pd == params_pl
    
    # CastingApplier apply
    # Let's test specific casting
    # a -> float, b -> string, c -> category
    params = {
        "columns": ["a", "b", "c"],
        "target_type": "float", 
    }
    
    # Let's construct params manually for Applier to test multiple types
    params_mixed = {
        "type_map": {
            "a": "float",
            "b": "string",
            "c": "category"
        }
    }
    
    applier = CastingApplier()
    res_pd = applier.apply(df_pd, params_mixed)
    res_pl = applier.apply(df_pl, params_mixed)
    
    # Check types
    assert pd.api.types.is_float_dtype(res_pd["a"])
    assert pd.api.types.is_string_dtype(res_pd["b"]) or pd.api.types.is_object_dtype(res_pd["b"])
    assert isinstance(res_pd["c"].dtype, pd.CategoricalDtype)
    
    assert res_pl["a"].dtype == pl.Float64
    assert res_pl["b"].dtype == pl.String
    assert res_pl["c"].dtype == pl.Categorical
    
    # Check values
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas(), check_dtype=False)
    print("Casting Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_text_cleaning_parity():
    print("Starting Text Cleaning Parity Test...")
    data = {
        "text": ["  Hello World  ", "Foo Bar", "123 Go!"],
        "mixed": ["UPPER", "lower", "MiXeD"]
    }
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    params = {
        "columns": ["text", "mixed"],
        "operations": [
            {"op": "trim", "mode": "both"},
            {"op": "case", "mode": "lower"},
            {"op": "remove_special", "mode": "keep_alphanumeric_space"}
        ]
    }
    
    applier = TextCleaningApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    # Expected:
    # "  Hello World  " -> "hello world"
    # "Foo Bar" -> "foo bar"
    # "123 Go!" -> "123 go" (exclamation removed)
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas(), check_dtype=False)
    print("Text Cleaning Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_value_replacement_parity():
    print("Starting ValueReplacement Parity Test...")
    data = {"a": [1, 2, 3, 4], "b": ["x", "y", "z", "x"]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    # Test 1: Global mapping
    params_map = {"columns": ["a"], "mapping": {1: 10, 2: 20}}
    applier = ValueReplacementApplier()
    res_pd = applier.apply(df_pd, params_map)
    res_pl = applier.apply(df_pl, params_map)
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas(), check_dtype=False)
    
    # Test 2: Column specific mapping
    params_nested = {"columns": ["a", "b"], "mapping": {"a": {3: 30}, "b": {"x": "X"}}}
    res_pd_n = applier.apply(df_pd, params_nested)
    res_pl_n = applier.apply(df_pl, params_nested)
    pd.testing.assert_frame_equal(res_pd_n, res_pl_n.to_pandas(), check_dtype=False)
    
    # Test 3: to_replace / value
    params_val = {"columns": ["a"], "to_replace": 4, "value": 40}
    res_pd_v = applier.apply(df_pd, params_val)
    res_pl_v = applier.apply(df_pl, params_val)
    pd.testing.assert_frame_equal(res_pd_v, res_pl_v.to_pandas(), check_dtype=False)
    
    print("ValueReplacement Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_alias_replacement_parity():
    print("Starting AliasReplacement Parity Test...")
    data = {"country": ["U.S.A.", "united states", "UK", "Great Britain", "Other"]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    params = {"columns": ["country"], "alias_type": "country"}
    
    applier = AliasReplacementApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    # Expected: "USA", "USA", "United Kingdom", "United Kingdom", "Other"
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas(), check_dtype=False)
    print("AliasReplacement Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_invalid_value_replacement_parity():
    print("Starting InvalidValueReplacement Parity Test...")
    data = {"a": [1.0, np.inf, -np.inf, 4.0]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    # Test 1: Replace both infs with NaN (default)
    params = {"columns": ["a"], "replace_inf": True, "replace_neg_inf": True, "value": None}
    applier = InvalidValueReplacementApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    # Should be [1.0, NaN, NaN, 4.0]
    assert np.isnan(res_pd["a"][1])
    assert np.isnan(res_pd["a"][2])
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas(), check_dtype=False)
    
    # Test 2: Replace only +inf with 999
    params_pos = {"columns": ["a"], "replace_inf": True, "replace_neg_inf": False, "value": 999.0}
    res_pd_p = applier.apply(df_pd, params_pos)
    res_pl_p = applier.apply(df_pl, params_pos)
    
    # Should be [1.0, 999.0, -inf, 4.0]
    assert res_pd_p["a"][1] == 999.0
    assert res_pd_p["a"][2] == -np.inf
    
    pd.testing.assert_frame_equal(res_pd_p, res_pl_p.to_pandas(), check_dtype=False)
    print("InvalidValueReplacement Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_missing_indicator_parity():
    print("Starting MissingIndicator Parity Test...")
    data = {"a": [1, None, 3], "b": [None, 2, 3]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {} # Auto detect
    calc = MissingIndicatorCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    assert set(params_pd["columns"]) == {"a", "b"}
    assert set(params_pl["columns"]) == {"a", "b"}
    
    applier = MissingIndicatorApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    # Should have a_missing, b_missing
    assert "a_missing" in res_pd.columns
    assert "b_missing" in res_pd.columns
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas(), check_dtype=False)
    print("MissingIndicator Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_deduplicate_parity():
    print("Starting Deduplicate Parity Test...")
    data = {"id": [1, 1, 2, 3, 3], "val": ["a", "b", "c", "d", "e"]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    params = {"subset": ["id"], "keep": "first"}
    
    applier = DeduplicateApplier()
    res_pd = applier.apply(df_pd, params)
    res_pl = applier.apply(df_pl, params)
    
    # Should keep 1-a, 2-c, 3-d
    assert len(res_pd) == 3
    assert len(res_pl) == 3
    
    pd.testing.assert_frame_equal(res_pd.reset_index(drop=True), res_pl.to_pandas().reset_index(drop=True), check_dtype=False)
    print("Deduplicate Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_drop_missing_columns_parity():
    print("Starting DropMissingColumns Parity Test...")
    data = {
        "a": [1, 2, 3, 4],
        "b": [1, None, None, 4], # 50% missing
        "c": [None, None, None, None] # 100% missing
    }
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"missing_threshold": 40.0} # Drop if >= 40% missing
    
    calc = DropMissingColumnsCalculator()
    params_pd = calc.fit(df_pd, config)
    params_pl = calc.fit(df_pl, config)
    
    # Should drop b and c
    assert "b" in params_pd["columns_to_drop"]
    assert "c" in params_pd["columns_to_drop"]
    assert set(params_pd["columns_to_drop"]) == set(params_pl["columns_to_drop"])
    
    applier = DropMissingColumnsApplier()
    res_pd = applier.apply(df_pd, params_pd)
    res_pl = applier.apply(df_pl, params_pl)
    
    assert "b" not in res_pd.columns
    assert "c" not in res_pd.columns
    assert "a" in res_pd.columns
    
    pd.testing.assert_frame_equal(res_pd, res_pl.to_pandas(), check_dtype=False)
    print("DropMissingColumns Parity Test Passed!")

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_drop_missing_rows_parity():
    print("Starting DropMissingRows Parity Test...")
    data = {
        "a": [1, 2, None, 4],
        "b": [1, None, None, 4]
    }
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    # Test 'any'
    params_any = {"subset": ["a", "b"], "how": "any"}
    applier = DropMissingRowsApplier()
    res_pd = applier.apply(df_pd, params_any)
    res_pl = applier.apply(df_pl, params_any)
    
    # Should drop row 1 (b is None) and row 2 (a and b are None)
    # Keep row 0 and 3
    assert len(res_pd) == 2
    pd.testing.assert_frame_equal(res_pd.reset_index(drop=True), res_pl.to_pandas().reset_index(drop=True), check_dtype=False)
    
    # Test 'threshold'
    params_thresh = {"subset": ["a", "b"], "threshold": 1} # Keep if at least 1 non-null
    res_pd_t = applier.apply(df_pd, params_thresh)
    res_pl_t = applier.apply(df_pl, params_thresh)
    
    # Should keep row 0 (2 non-null), row 1 (1 non-null), row 3 (2 non-null)
    # Drop row 2 (0 non-null)
    assert len(res_pd_t) == 3
    pd.testing.assert_frame_equal(res_pd_t.reset_index(drop=True), res_pl_t.to_pandas().reset_index(drop=True), check_dtype=False)
    
    print("DropMissingRows Parity Test Passed!")

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
        test_general_binning_parity()
        test_custom_binning_parity()
        test_iqr_parity()
        test_zscore_parity()
        test_winsorize_parity()
        test_manual_bounds_parity()
        test_elliptic_envelope_parity()
        test_casting_parity()
        test_text_cleaning_parity()
        test_value_replacement_parity()
        test_alias_replacement_parity()
        test_invalid_value_replacement_parity()
        test_missing_indicator_parity()
        test_deduplicate_parity()
        test_drop_missing_columns_parity()
        test_drop_missing_rows_parity()
    else:
        print("Skipping test because Polars is not installed.")
