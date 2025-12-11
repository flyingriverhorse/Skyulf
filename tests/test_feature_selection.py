import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.feature_selection import (
    VarianceThresholdCalculator, VarianceThresholdApplier,
    UnivariateSelectionCalculator, UnivariateSelectionApplier
)

def test_variance_threshold():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1, 1, 1, 1, 1], # Constant
        'C': [1, 1, 1, 1, 0]  # Low variance
    })
    
    calc = VarianceThresholdCalculator()
    # Threshold 0 drops B
    params = calc.fit(df, {'threshold': 0.0})
    
    applier = VarianceThresholdApplier()
    res = applier.apply(df, params)
    
    assert 'A' in res.columns
    assert 'B' not in res.columns
    assert 'C' in res.columns

def test_select_k_best():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5], # Correlated with target
        'B': [5, 4, 3, 2, 1], # Correlated with target
        'C': [1, 0, 1, 0, 1], # Randomish
        'target': [10, 20, 30, 40, 50]
    })
    
    calc = UnivariateSelectionCalculator()
    config = {
        'method': 'select_k_best',
        'k': 2, 
        'score_func': 'f_regression', 
        'target_column': 'target',
        'columns': ['A', 'B', 'C']
    }
    params = calc.fit(df, config)
    
    applier = UnivariateSelectionApplier()
    res = applier.apply(df, params)
    
    assert 'A' in res.columns
    assert 'B' in res.columns
    assert 'C' not in res.columns
