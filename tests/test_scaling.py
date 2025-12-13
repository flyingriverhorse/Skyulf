import pytest
import pandas as pd
import numpy as np
from skyulf.preprocessing.scaling import (
    StandardScalerCalculator, StandardScalerApplier,
    MinMaxScalerCalculator, MinMaxScalerApplier,
    RobustScalerCalculator, RobustScalerApplier,
    MaxAbsScalerCalculator, MaxAbsScalerApplier
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0, 5.0],
        'B': [10.0, 20.0, 30.0, 40.0, 50.0],
        'C': ['x', 'y', 'z', 'x', 'y'] # Categorical, should be ignored
    })

def test_standard_scaler(sample_df):
    # 1. Fit
    calc = StandardScalerCalculator()
    config = {'columns': ['A', 'B']}
    params = calc.fit(sample_df, config)
    
    assert params['type'] == 'standard_scaler'
    assert params['columns'] == ['A', 'B']
    assert np.isclose(params['mean'][0], 3.0) # Mean of 1..5 is 3
    assert np.isclose(params['mean'][1], 30.0) # Mean of 10..50 is 30
    
    # 2. Apply
    applier = StandardScalerApplier()
    transformed_df = applier.apply(sample_df, params)
    
    # Check values
    # (3 - 3) / std = 0
    assert np.isclose(transformed_df['A'].iloc[2], 0.0)
    # (1 - 3) / std -> negative
    assert transformed_df['A'].iloc[0] < 0
    
    # Check categorical untouched
    assert transformed_df['C'].equals(sample_df['C'])

def test_minmax_scaler(sample_df):
    # 1. Fit
    calc = MinMaxScalerCalculator()
    config = {'columns': ['A']}
    params = calc.fit(sample_df, config)
    
    assert params['type'] == 'minmax_scaler'
    
    # 2. Apply
    applier = MinMaxScalerApplier()
    transformed_df = applier.apply(sample_df, params)
    
    # Min (1) -> 0, Max (5) -> 1
    assert np.isclose(transformed_df['A'].min(), 0.0)
    assert np.isclose(transformed_df['A'].max(), 1.0)
    assert np.isclose(transformed_df['A'].iloc[2], 0.5) # 3 is mid

def test_robust_scaler(sample_df):
    # Add outlier
    df = sample_df.copy()
    df.loc[5] = [1000.0, 1000.0, 'z']
    
    # 1. Fit
    calc = RobustScalerCalculator()
    config = {'columns': ['A']}
    params = calc.fit(df, config)
    
    assert params['type'] == 'robust_scaler'
    
    # 2. Apply
    applier = RobustScalerApplier()
    transformed_df = applier.apply(df, params)
    
    # Median of 1,2,3,4,5,1000 is (3+4)/2 = 3.5
    # Center should be median
    assert np.isclose(params['center'][0], 3.5)
    
    # Transformed values should be centered around 0 (roughly)
    # The outlier should still be large, but scaled by IQR
    assert transformed_df['A'].iloc[5] > 10 

def test_maxabs_scaler(sample_df):
    # Add negative values to test symmetry
    df = sample_df.copy()
    df.loc[0, 'A'] = -10.0 # Max abs is now 10
    
    # 1. Fit
    calc = MaxAbsScalerCalculator()
    config = {'columns': ['A']}
    params = calc.fit(df, config)
    
    assert params['type'] == 'maxabs_scaler'
    assert np.isclose(params['max_abs'][0], 10.0)
    
    # 2. Apply
    applier = MaxAbsScalerApplier()
    transformed_df = applier.apply(df, params)
    
    # -10 / 10 = -1
    assert np.isclose(transformed_df['A'].iloc[0], -1.0)
    # 5 / 10 = 0.5
    assert np.isclose(transformed_df['A'].iloc[4], 0.5)

def test_scaler_auto_column_selection(sample_df):
    calc = StandardScalerCalculator()
    # No columns specified -> should pick numeric A and B
    params = calc.fit(sample_df, {})
    
    assert 'A' in params['columns']
    assert 'B' in params['columns']
    assert 'C' not in params['columns']
