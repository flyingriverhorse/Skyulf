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
        'D': [0, 0, 0, 0, 0], # For division by zero check
        'Str1': ['apple', 'banana', 'cherry', 'date', 'elderberry'],
        'Str2': ['apple', 'bananas', 'berry', 'data', 'elder']
    })

def test_similarity_operation(sample_df):
    calc = FeatureMathCalculator()
    applier = FeatureMathApplier()
    
    config = {
        'operations': [
            {
                'operation_type': 'similarity',
                'method': 'ratio', # Default SequenceMatcher
                'input_columns': ['Str1'],
                'secondary_columns': ['Str2'],
                'output_column': 'sim_score'
            }
        ]
    }
    
    params = calc.fit(sample_df, config)
    result = applier.apply(sample_df, params)
    
    # 'apple' vs 'apple' -> 100.0
    assert result['sim_score'].iloc[0] == 100.0
    # 'banana' vs 'bananas' -> high similarity
    assert result['sim_score'].iloc[1] > 80.0
    # 'cherry' vs 'berry' -> some similarity
    assert result['sim_score'].iloc[2] > 0.0

def test_ratio_complex(sample_df):
    calc = FeatureMathCalculator()
    applier = FeatureMathApplier()
    
    # (A + B) / (C + B)
    # Row 0: (10 + 2) / (1 + 2) = 12 / 3 = 4
    config = {
        'operations': [
            {
                'operation_type': 'ratio',
                'input_columns': ['A', 'B'],
                'secondary_columns': ['C', 'B'],
                'output_column': 'complex_ratio'
            }
        ]
    }
    
    params = calc.fit(sample_df, config)
    result = applier.apply(sample_df, params)
    
    assert result['complex_ratio'].iloc[0] == 4.0

def test_polynomial_features_integration():
    # Testing the PolynomialFeaturesCalculator specifically
    from core.ml_pipeline.preprocessing.feature_generation import PolynomialFeaturesCalculator, PolynomialFeaturesApplier
    
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    calc = PolynomialFeaturesCalculator()
    applier = PolynomialFeaturesApplier()
    
    config = {
        'columns': ['A', 'B'],
        'degree': 2,
        'interaction_only': False,
        'include_bias': False,
        'output_prefix': 'poly'
    }
    
    # Fit
    params = calc.fit(df, config)
    assert params['type'] == 'polynomial_features'
    assert 'A' in params['columns']
    
    # Apply
    result = applier.apply(df, params)
    
    # Expected columns: A, B (original), poly_A, poly_B, poly_A^2, poly_A_B, poly_B^2 (renamed)
    # Sklearn names: A, B, A^2, A B, B^2
    # Our renames: poly_A, poly_B, poly_A_pow_2, poly_A_B, poly_B_pow_2
    
    assert 'poly_A_B' in result.columns or 'poly_A_B' in [c.replace(' ', '_') for c in result.columns]
    # Check value for A*B row 0: 1*4 = 4
    # Find the interaction column
    interaction_col = [c for c in result.columns if 'A_B' in c][0]
    assert result[interaction_col].iloc[0] == 4.0
