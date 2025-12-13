import pytest
import pandas as pd
import numpy as np
from skyulf.preprocessing.outliers import (
    IQRCalculator, IQRApplier,
    ZScoreCalculator, ZScoreApplier
)

@pytest.fixture
def outlier_df():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100], # 100 is outlier
        'B': [10, 10, 10, 10, 10, 10]
    })

def test_iqr_clip(outlier_df):
    # Note: Current IQRCalculator implementation in outliers.py only calculates bounds.
    # It does NOT support 'action': 'clip' or 'drop' directly in the calculator config.
    # The Applier uses the bounds to filter (drop) or mask.
    # Looking at outliers.py: IQRApplier filters (drops) rows outside bounds.
    # It does NOT seem to support clipping in the current implementation shown in context.
    # It returns X_filtered = X[mask].
    
    calc = IQRCalculator()
    params = calc.fit(outlier_df, {'multiplier': 1.5, 'columns': ['A']})
    
    applier = IQRApplier()
    res = applier.apply(outlier_df, params)
    
    # Should drop the outlier
    assert len(res) == 5
    assert 100 not in res['A'].values

def test_zscore_drop(outlier_df):
    # Similarly, ZScore likely drops.
    calc = ZScoreCalculator()
    params = calc.fit(outlier_df, {'threshold': 2.0, 'columns': ['A']})
    
    applier = ZScoreApplier()
    res = applier.apply(outlier_df, params)
    
    assert len(res) == 5
    assert 100 not in res['A'].values
