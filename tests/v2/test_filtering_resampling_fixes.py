import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.filtering import MissingIndicatorCalculator, MissingIndicatorApplier
from core.ml_pipeline.preprocessing.resampling import ResamplingCalculator, ResamplingApplier

def test_missing_indicator():
    df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, 6]})
    calc = MissingIndicatorCalculator()
    applier = MissingIndicatorApplier()
    
    config = {'columns': ['a']}
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)
    
    assert 'a_missing' in result.columns
    assert result['a_missing'].iloc[0] == 0
    assert result['a_missing'].iloc[1] == 1
    assert result['a_missing'].iloc[2] == 0

def test_under_sampling():
    # Create imbalanced dataset
    # Class 0: 90 samples, Class 1: 10 samples
    X = np.random.rand(100, 2)
    y = np.array([0]*90 + [1]*10)
    df = pd.DataFrame(X, columns=['f1', 'f2'])
    df['target'] = y
    
    calc = ResamplingCalculator()
    applier = ResamplingApplier()
    
    # Random Under Sampling
    config = {
        'method': 'random_under',
        'target_column': 'target',
        'sampling_strategy': 'auto' # Resample to 50-50 (downsample majority)
    }
    
    artifacts = calc.fit(df, config)
    df_res = applier.apply(df, artifacts)
    
    counts = df_res['target'].value_counts()
    assert counts[0] == 10 # Should match minority
    assert counts[1] == 10
    assert len(df_res) == 20
