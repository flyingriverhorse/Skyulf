import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.feature_generation import FeatureMathCalculator, FeatureMathApplier

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [10, 20, 30, 40, 50],
        'B': [2, 4, 5, 8, 10],
        'C': [1, 1, 1, 1, 1],
        'D': [0, 0, 0, 0, 0] # For division by zero check
    })

def test_arithmetic_operations(sample_df):
    calc = FeatureMathCalculator()
    applier = FeatureMathApplier()
    
    config = {
        'operations': [
            {
                'operation_type': 'arithmetic',
                'method': 'add',
                'input_columns': ['A', 'B'],
                'output_column': 'A_plus_B'
            },
            {
                'operation_type': 'arithmetic',
                'method': 'subtract',
                'input_columns': ['A', 'B'],
                'output_column': 'A_minus_B'
            },
            {
                'operation_type': 'arithmetic',
                'method': 'multiply',
                'input_columns': ['A', 'B'],
                'output_column': 'A_times_B'
            },
            {
                'operation_type': 'arithmetic',
                'method': 'divide',
                'input_columns': ['A', 'B'],
                'output_column': 'A_div_B'
            },
            {
                'operation_type': 'arithmetic',
                'method': 'add',
                'input_columns': ['A'],
                'constants': [100],
                'output_column': 'A_plus_100'
            }
        ]
    }
    
    params = calc.fit(sample_df, config)
    result = applier.apply(sample_df, params)
    
    assert result['A_plus_B'].iloc[0] == 12
    assert result['A_minus_B'].iloc[0] == 8
    assert result['A_times_B'].iloc[0] == 20
    assert result['A_div_B'].iloc[0] == 5.0
    assert result['A_plus_100'].iloc[0] == 110

def test_division_by_zero(sample_df):
    calc = FeatureMathCalculator()
    applier = FeatureMathApplier()
    
    config = {
        'operations': [
            {
                'operation_type': 'arithmetic',
                'method': 'divide',
                'input_columns': ['A', 'D'],
                'output_column': 'A_div_D'
            }
        ],
        'epsilon': 1e-9
    }
    
    params = calc.fit(sample_df, config)
    result = applier.apply(sample_df, params)
    
    # Should not be infinity, but a large number (A / epsilon)
    assert not np.isinf(result['A_div_D'].iloc[0])
    assert result['A_div_D'].iloc[0] > 1000000

def test_ratio_operation(sample_df):
    calc = FeatureMathCalculator()
    applier = FeatureMathApplier()
    
    # (A + B) / (C)
    # (10+2)/1 = 12
    config = {
        'operations': [
            {
                'operation_type': 'ratio',
                'input_columns': ['A', 'B'], # Numerator
                'secondary_columns': ['C'],  # Denominator
                'output_column': 'ratio_res'
            }
        ]
    }
    
    params = calc.fit(sample_df, config)
    result = applier.apply(sample_df, params)
    
    assert result['ratio_res'].iloc[0] == 12.0

def test_stat_operations(sample_df):
    calc = FeatureMathCalculator()
    applier = FeatureMathApplier()
    
    # Mean of A and B -> (10+2)/2 = 6
    config = {
        'operations': [
            {
                'operation_type': 'stat',
                'method': 'mean',
                'input_columns': ['A', 'B'],
                'output_column': 'mean_AB'
            },
            {
                'operation_type': 'stat',
                'method': 'max',
                'input_columns': ['A', 'B'],
                'output_column': 'max_AB'
            }
        ]
    }
    
    params = calc.fit(sample_df, config)
    result = applier.apply(sample_df, params)
    
    assert result['mean_AB'].iloc[0] == 6.0
    assert result['max_AB'].iloc[0] == 10.0

def test_overwrite_protection(sample_df):
    calc = FeatureMathCalculator()
    applier = FeatureMathApplier()
    
    # Try to overwrite 'A' without permission
    config = {
        'operations': [
            {
                'operation_type': 'arithmetic',
                'method': 'add',
                'input_columns': ['A'],
                'constants': [1],
                'output_column': 'A'
            }
        ],
        'allow_overwrite': False
    }
    
    params = calc.fit(sample_df, config)
    result = applier.apply(sample_df, params)
    
    # Should create A_1 instead of overwriting A
    assert 'A_1' in result.columns
    assert result['A'].iloc[0] == 10 # Original
    assert result['A_1'].iloc[0] == 11 # New

