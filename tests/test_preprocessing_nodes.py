import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.feature_selection import CorrelationThresholdCalculator, CorrelationThresholdApplier
from core.ml_pipeline.preprocessing.feature_generation import DatePartsCalculator, DatePartsApplier

def test_correlation_threshold():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, 2, 3, 4, 5], # Perfectly correlated with a
        'c': [5, 4, 3, 2, 1], # Perfectly negatively correlated with a
        'd': [1, 0, 1, 0, 1]  # Uncorrelated
    })
    
    calc = CorrelationThresholdCalculator()
    applier = CorrelationThresholdApplier()
    
    # Threshold 0.99 should drop 'b' and 'c' (abs correlation)
    config = {'threshold': 0.99}
    artifacts = calc.fit(df, config)
    
    assert 'b' in artifacts['columns_to_drop'] or 'a' in artifacts['columns_to_drop']
    # Note: which one is dropped depends on column order in correlation matrix iteration
    
    result = applier.apply(df, artifacts)
    assert result.shape[1] < 4

def test_date_parts():
    df = pd.DataFrame({'date': pd.to_datetime(['2021-01-01 10:00:00'])})
    
    calc = DatePartsCalculator()
    applier = DatePartsApplier()
    
    config = {'columns': ['date'], 'parts': ['year', 'month', 'hour']}
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)
    
    assert 'date_year' in result.columns
    assert result['date_year'].iloc[0] == 2021
    assert result['date_hour'].iloc[0] == 10
