import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.feature_generation import RowStatisticsCalculator, RowStatisticsApplier

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })

def test_row_statistics(sample_df):
    calc = RowStatisticsCalculator()
    app = RowStatisticsApplier()
    
    config = {
        'columns': ['A', 'B', 'C'],
        'operations': ['mean', 'sum', 'min', 'max'],
        'new_column_prefix': 'row'
    }
    
    params = calc.fit(sample_df, config)
    res = app.apply(sample_df, params)
    
    assert 'row_mean' in res.columns
    assert 'row_sum' in res.columns
    assert 'row_min' in res.columns
    assert 'row_max' in res.columns
    
    assert res['row_sum'].iloc[0] == 1 + 4 + 7
    assert res['row_mean'].iloc[0] == (1 + 4 + 7) / 3
    assert res['row_min'].iloc[0] == 1
    assert res['row_max'].iloc[0] == 7
