import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.bucketing import (
    KBinsDiscretizerCalculator, KBinsDiscretizerApplier,
    CustomBinningCalculator, CustomBinningApplier
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'age': [10, 20, 30, 40, 50, 60, 70, 80, 90],
        'income': [100, 200, 300, 400, 500, 600, 700, 800, 900]
    })

def test_kbins_discretizer(sample_df):
    # 1. Fit (Equal Width / Uniform)
    calc = KBinsDiscretizerCalculator()
    config = {'columns': ['age'], 'n_bins': 3, 'strategy': 'uniform', 'encode': 'ordinal'}
    params = calc.fit(sample_df, config)
    
    assert params['type'] == 'kbins'
    assert len(params['bin_edges']) == 1
    # Edges should be approx [10, 36.6, 63.3, 90]
    
    # 2. Apply
    applier = KBinsDiscretizerApplier()
    transformed = applier.apply(sample_df, params)
    
    # Should have 0, 1, 2
    assert transformed['age'].nunique() == 3
    assert transformed['age'].min() == 0
    assert transformed['age'].max() == 2

def test_custom_binning(sample_df):
    # 1. Fit
    calc = CustomBinningCalculator()
    config = {
        'columns': ['age'], 
        'bins': [0, 18, 65, 100], 
        'labels': ['child', 'adult', 'senior']
    }
    params = calc.fit(sample_df, config)
    
    assert params['type'] == 'custom_binning'
    
    # 2. Apply
    applier = CustomBinningApplier()
    transformed = applier.apply(sample_df, params)
    
    assert transformed['age'].iloc[0] == 'child' # 10
    assert transformed['age'].iloc[4] == 'adult' # 50
    assert transformed['age'].iloc[8] == 'senior' # 90
