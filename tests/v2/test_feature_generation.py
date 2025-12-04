import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.feature_generation import (
    PolynomialFeaturesCalculator, PolynomialFeaturesApplier,
    MathExpressionCalculator, MathExpressionApplier
)

def test_polynomial_features():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    calc = PolynomialFeaturesCalculator()
    params = calc.fit(df, {'degree': 2, 'interaction_only': False})
    
    applier = PolynomialFeaturesApplier()
    res = applier.apply(df, params)
    
    # Should have A, B, A^2, A*B, B^2
    # Sklearn names: A, B, A^2, A B, B^2
    assert 'A^2' in res.columns or 'A 2' in res.columns or any('^2' in c for c in res.columns)
    assert res.shape[1] > 2

def test_math_expression():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    calc = MathExpressionCalculator()
    params = calc.fit(df, {'expression': 'A + B', 'new_column': 'C'})
    
    applier = MathExpressionApplier()
    res = applier.apply(df, params)
    
    assert 'C' in res.columns
    assert res['C'].iloc[0] == 4
