import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.cleaning import DateStandardizerCalculator, DateStandardizerApplier
from core.ml_pipeline.preprocessing.feature_selection import CorrelationThresholdCalculator, CorrelationThresholdApplier
from core.ml_pipeline.preprocessing.outliers import IsolationForestOutlierCalculator, IsolationForestOutlierApplier
from core.ml_pipeline.preprocessing.feature_generation import DatePartsCalculator, DatePartsApplier
from core.ml_pipeline.preprocessing.transformations import QuantileTransformerCalculator, QuantileTransformerApplier

def test_date_standardizer():
    # Use consistent format or ensure pandas can parse it
    df = pd.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-03-01']})
    calc = DateStandardizerCalculator()
    applier = DateStandardizerApplier()
    
    config = {'columns': ['date'], 'target_format': '%Y/%m/%d'}
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)
    
    assert result['date'].iloc[0] == '2021/01/01'
    assert result['date'].iloc[1] == '2021/01/02'
    assert result['date'].iloc[2] == '2021/03/01'

def test_date_standardizer_mixed():
    # Test with dayfirst=True on consistent format
    df = pd.DataFrame({'date': ['01/02/2021', '05/02/2021']}) # 1st Feb, 5th Feb
    calc = DateStandardizerCalculator()
    applier = DateStandardizerApplier()
    
    config = {'columns': ['date'], 'target_format': '%Y-%m-%d', 'dayfirst': True}
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)
    
    # 01/02/2021 -> Feb 1st -> 2021-02-01
    assert result['date'].iloc[0] == '2021-02-01'
    # 05/02/2021 -> Feb 5th -> 2021-02-05
    assert result['date'].iloc[1] == '2021-02-05'

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

def test_isolation_forest():
    # Generate data with outliers
    np.random.seed(42)
    X = 0.3 * np.random.randn(100, 2)
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X + 2, X - 2, X_outliers]
    df = pd.DataFrame(X, columns=['f1', 'f2'])
    
    calc = IsolationForestOutlierCalculator()
    applier = IsolationForestOutlierApplier()
    
    config = {'contamination': 0.1, 'action': 'flag'}
    artifacts = calc.fit(df, config)
    
    assert 'model' in artifacts
    
    result = applier.apply(df, artifacts)
    assert 'is_outlier' in result.columns
    assert result['is_outlier'].sum() > 0

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

def test_quantile_transformer():
    df = pd.DataFrame({'a': np.random.rand(100)})
    
    calc = QuantileTransformerCalculator()
    applier = QuantileTransformerApplier()
    
    config = {'output_distribution': 'normal'}
    artifacts = calc.fit(df, config)
    
    assert 'transformer_object' in artifacts
    
    result = applier.apply(df, artifacts)
    # Check if it looks roughly normal (mean ~ 0, std ~ 1)
    assert abs(result['a'].mean()) < 0.5 # Loose check
