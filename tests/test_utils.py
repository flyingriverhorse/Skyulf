import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.utils import detect_numeric_columns

def test_detect_numeric_columns_basic():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [1.1, 2.2, 3.3],
        'C': ['x', 'y', 'z']
    })
    cols = detect_numeric_columns(df)
    assert 'A' in cols
    assert 'B' in cols
    assert 'C' not in cols

def test_detect_numeric_columns_string_numbers():
    df = pd.DataFrame({
        'A': ['1.5', '2.5', '3.5'], # String numbers
        'B': ['1', '2', '3'],       # String integers
        'C': ['1', 'x', '3']        # Mixed (should be excluded if coercion fails for too many?)
                                    # Logic: coerce -> dropna -> check empty. 
                                    # 'x' becomes NaN. 1 and 3 remain. Not empty.
    })
    cols = detect_numeric_columns(df)
    assert 'A' in cols
    assert 'B' in cols
    # C has valid numbers 1 and 3, so it IS detected as numeric-like by this logic
    # unless we want stricter validation. V1 logic says: if valid is not empty.
    assert 'C' in cols 

def test_detect_numeric_columns_exclusions():
    df = pd.DataFrame({
        'Binary': [0, 1, 0, 1],
        'Constant': [5, 5, 5, 5],
        'Bool': [True, False, True, False],
        'Valid': [10, 20, 30, 40]
    })
    cols = detect_numeric_columns(df)
    
    assert 'Binary' not in cols
    assert 'Constant' not in cols
    assert 'Bool' not in cols
    assert 'Valid' in cols
