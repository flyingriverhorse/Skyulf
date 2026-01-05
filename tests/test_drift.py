import pytest
import polars as pl
import numpy as np
from skyulf.profiling.drift import DriftCalculator

def test_drift_calculation_no_drift():
    # Generate two identical distributions
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(0, 1, 1000)
    
    df_ref = pl.DataFrame({"feature_a": data1})
    df_curr = pl.DataFrame({"feature_a": data2})
    
    calculator = DriftCalculator(df_ref, df_curr)
    report = calculator.calculate_drift()
    
    assert report.drifted_columns_count == 0
    assert not report.column_drifts["feature_a"].drift_detected
    
    # Check metrics are low
    metrics = {m.metric: m.value for m in report.column_drifts["feature_a"].metrics}
    assert metrics["psi"] < 0.1
    assert metrics["wasserstein_distance"] < 0.1

def test_drift_calculation_with_drift():
    # Generate two different distributions
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1, 1000) # Shifted mean
    
    df_ref = pl.DataFrame({"feature_a": data1})
    df_curr = pl.DataFrame({"feature_a": data2})
    
    calculator = DriftCalculator(df_ref, df_curr)
    report = calculator.calculate_drift()
    
    assert report.drifted_columns_count == 1
    assert report.column_drifts["feature_a"].drift_detected
    
    # Check metrics are high
    metrics = {m.metric: m.value for m in report.column_drifts["feature_a"].metrics}
    assert metrics["psi"] > 0.2
    assert metrics["wasserstein_distance"] > 0.1

def test_drift_calculation_mixed():
    np.random.seed(42)
    # Feature A: No drift
    a1 = np.random.normal(0, 1, 1000)
    a2 = np.random.normal(0, 1, 1000)
    
    # Feature B: Drift
    b1 = np.random.normal(0, 1, 1000)
    b2 = np.random.normal(0, 5, 1000) # Changed variance
    
    df_ref = pl.DataFrame({"feature_a": a1, "feature_b": b1})
    df_curr = pl.DataFrame({"feature_a": a2, "feature_b": b2})
    
    calculator = DriftCalculator(df_ref, df_curr)
    report = calculator.calculate_drift()
    
    assert report.drifted_columns_count == 1
    assert not report.column_drifts["feature_a"].drift_detected
    assert report.column_drifts["feature_b"].drift_detected
