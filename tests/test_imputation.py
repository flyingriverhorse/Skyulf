import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.imputation import (
    SimpleImputerCalculator, SimpleImputerApplier,
    KNNImputerCalculator, KNNImputerApplier,
    IterativeImputerCalculator, IterativeImputerApplier
)

@pytest.fixture
def sample_df_missing():
    return pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0, 5.0],
        'B': [10.0, np.nan, 30.0, 40.0, 50.0],
        'C': ['x', 'y', np.nan, 'x', 'y']
    })

def test_simple_imputer_mean(sample_df_missing):
    # 1. Fit (Mean)
    calc = SimpleImputerCalculator()
    config = {'strategy': 'mean', 'columns': ['A', 'B']}
    params = calc.fit(sample_df_missing, config)
    
    assert params['type'] == 'simple_imputer'
    # Mean of A (1,2,4,5) = 3.0
    assert params['fill_values']['A'] == 3.0
    # Mean of B (10,30,40,50) = 32.5
    assert params['fill_values']['B'] == 32.5
    
    # 2. Apply
    applier = SimpleImputerApplier()
    transformed_df = applier.apply(sample_df_missing, params)
    
    assert transformed_df['A'].isna().sum() == 0
    assert transformed_df['A'].iloc[2] == 3.0
    assert transformed_df['B'].iloc[1] == 32.5

def test_simple_imputer_mode(sample_df_missing):
    # 1. Fit (Most Frequent)
    calc = SimpleImputerCalculator()
    config = {'strategy': 'most_frequent', 'columns': ['C']}
    params = calc.fit(sample_df_missing, config)
    
    # Mode of C (x, y, x, y) -> x (sorted first usually) or y
    # Sklearn most_frequent returns the smallest if there is a tie. 'x' < 'y'.
    assert params['fill_values']['C'] == 'x'
    
    # 2. Apply
    applier = SimpleImputerApplier()
    transformed_df = applier.apply(sample_df_missing, params)
    
    assert transformed_df['C'].iloc[2] == 'x'

def test_knn_imputer(sample_df_missing):
    # KNN only works on numeric
    df = sample_df_missing[['A', 'B']].copy()
    
    # 1. Fit
    calc = KNNImputerCalculator()
    config = {'columns': ['A', 'B'], 'k_neighbors': 2}
    params = calc.fit(df, config)
    
    assert params['type'] == 'knn_imputer'
    assert params['imputer_object'] is not None
    
    # 2. Apply
    applier = KNNImputerApplier()
    transformed_df = applier.apply(df, params)
    
    assert transformed_df.isna().sum().sum() == 0
    # Value should be imputed based on neighbors
    assert not np.isnan(transformed_df['A'].iloc[2])

def test_iterative_imputer(sample_df_missing):
    # Iterative only works on numeric
    df = sample_df_missing[['A', 'B']].copy()
    
    # 1. Fit
    calc = IterativeImputerCalculator()
    config = {'columns': ['A', 'B'], 'max_iter': 5}
    params = calc.fit(df, config)
    
    assert params['type'] == 'iterative_imputer'
    assert params['imputer_object'] is not None
    
    # 2. Apply
    applier = IterativeImputerApplier()
    transformed_df = applier.apply(df, params)
    
    assert transformed_df.isna().sum().sum() == 0
