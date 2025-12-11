import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.feature_generation import (
    PolynomialFeaturesCalculator, PolynomialFeaturesApplier,
    FeatureGenerationCalculator, FeatureGenerationApplier
)

def test_polynomial_features():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    calc = PolynomialFeaturesCalculator()
    params = calc.fit(df, {'degree': 2, 'interaction_only': False, 'columns': ['A', 'B']})
    
    applier = PolynomialFeaturesApplier()
    res = applier.apply(df, params)
    
    # Should have A, B, A^2, A*B, B^2
    # Sklearn names: A, B, A^2, A B, B^2
    # Our implementation prefixes them with "poly_"
    cols = res.columns
    assert any('poly_' in c for c in cols)
    assert res.shape[1] > 2

def test_math_expression():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    calc = FeatureGenerationCalculator()
    # Using 'arithmetic' operation type to simulate A + B
    config = {
        'operations': [{
            'operation_type': 'arithmetic',
            'method': 'add',
            'input_columns': ['A', 'B'],
            'output_column': 'C'
        }]
    }
    params = calc.fit(df, config)
    
    applier = FeatureGenerationApplier()
    res = applier.apply(df, params)
    
    assert 'C' in res.columns
    assert res['C'].iloc[0] == 4
