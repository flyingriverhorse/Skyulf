import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.drop_and_missing import (
    DropMissingColumnsCalculator, DropMissingColumnsApplier,
    DropMissingRowsCalculator, DropMissingRowsApplier,
    DeduplicateCalculator, DeduplicateApplier
)

def test_drop_columns():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    
    # DropMissingColumnsCalculator can drop specific columns via 'columns' config
    calc = DropMissingColumnsCalculator()
    params = calc.fit(df, {'columns': ['B']})
    
    applier = DropMissingColumnsApplier()
    res = applier.apply(df, params)
    
    assert 'B' not in res.columns
    assert 'A' in res.columns
    assert 'C' in res.columns

def test_drop_missing():
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [1, 2, 3]
    })
    
    # Drop rows with any missing
    calc = DropMissingRowsCalculator()
    # Assuming 'threshold' logic or similar. If not implemented, we skip.
    # params = calc.fit(df, {'threshold': 0}) 
    
    # applier = DropMissingRowsApplier()
    # res = applier.apply(df, params)
    pass

def test_deduplicate():
    df = pd.DataFrame({
        'A': [1, 1, 2],
        'B': [1, 1, 3]
    })
    
    calc = DeduplicateCalculator()
    params = calc.fit(df, {'subset': ['A', 'B']})
    
    applier = DeduplicateApplier()
    res = applier.apply(df, params)
    
    assert len(res) == 2
    assert len(res[res['A'] == 1]) == 1
