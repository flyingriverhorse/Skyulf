import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.resampling import ResamplingCalculator, ResamplingApplier

def test_smote_resampling():
    # Create imbalanced dataset
    # Class 0: 90 samples, Class 1: 10 samples
    X = np.random.rand(100, 2)
    y = np.array([0]*90 + [1]*10)
    df = pd.DataFrame(X, columns=['f1', 'f2'])
    df['target'] = y
    
    calc = ResamplingCalculator()
    applier = ResamplingApplier()
    
    config = {
        'method': 'smote',
        'target_column': 'target',
        'sampling_strategy': 'auto', # Resample to 50-50
        'k_neighbors': 2
    }
    
    artifacts = calc.fit(df, config)
    
    # Check artifacts
    assert artifacts['type'] == 'resampling'
    assert artifacts['method'] == 'smote'
    
    # Apply
    df_res = applier.apply(df, artifacts)
    
    # Check counts
    counts = df_res['target'].value_counts()
    assert counts[0] == 90
    assert counts[1] == 90 # Should be balanced
    
    # Check shape
    assert len(df_res) == 180

def test_adasyn_resampling():
    # Create imbalanced dataset
    X = np.random.rand(100, 2)
    y = np.array([0]*90 + [1]*10)
    df = pd.DataFrame(X, columns=['f1', 'f2'])
    df['target'] = y
    
    calc = ResamplingCalculator()
    applier = ResamplingApplier()
    
    config = {
        'method': 'adasyn',
        'target_column': 'target',
        'k_neighbors': 2
    }
    
    artifacts = calc.fit(df, config)
    df_res = applier.apply(df, artifacts)
    
    counts = df_res['target'].value_counts()
    # ADASYN might not be exactly balanced due to density
    assert counts[1] > 10 
    assert len(df_res) > 100

def test_resampling_missing_target():
    df = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})
    calc = ResamplingCalculator()
    applier = ResamplingApplier()
    
    config = {'method': 'smote', 'target_column': 'target'}
    artifacts = calc.fit(df, config)
    
    # Should return empty artifacts or handle gracefully
    assert artifacts == {}
    
    # Apply with empty artifacts (or if target missing in df)
    res = applier.apply(df, {'target_column': 'target'})
    assert len(res) == 2 # Unchanged
