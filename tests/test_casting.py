import pytest
import pandas as pd
import numpy as np
from skyulf.preprocessing.casting import CastingCalculator, CastingApplier

def test_casting():
    df = pd.DataFrame({
        'A': ['1', '2', '3'],
        'B': [1, 2, 3],
        'C': ['2021-01-01', '2021-01-02', 'not_a_date']
    })
    
    # 1. Fit
    calc = CastingCalculator()
    config = {
        'column_types': {
            'A': 'int',
            'B': 'float',
            'C': 'datetime'
        }
    }
    params = calc.fit(df, config)
    
    assert params['type_map']['A'] == 'int64'
    assert params['type_map']['B'] == 'float64'
    
    # 2. Apply
    applier = CastingApplier()
    transformed = applier.apply(df, params)
    
    assert pd.api.types.is_integer_dtype(transformed['A'])
    assert pd.api.types.is_float_dtype(transformed['B'])
    assert pd.api.types.is_datetime64_any_dtype(transformed['C'])
    
    # Check values
    assert transformed['A'].iloc[0] == 1
    assert transformed['B'].iloc[0] == 1.0
    assert pd.isna(transformed['C'].iloc[2]) # Coerced error
