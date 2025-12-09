import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.outliers import (
    IQROutlierCalculator, IQROutlierApplier,
    ZScoreOutlierCalculator, ZScoreOutlierApplier
)

@pytest.fixture
def outlier_df():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100], # 100 is outlier
        'B': [10, 10, 10, 10, 10, 10]
    })

def test_iqr_clip(outlier_df):
    calc = IQROutlierCalculator()
    params = calc.fit(outlier_df, {'action': 'clip', 'factor': 1.5})
    
    applier = IQROutlierApplier()
    res = applier.apply(outlier_df, params)
    
    # Q1=2.25, Q3=4.75, IQR=2.5. Upper = 4.75 + 1.5*2.5 = 8.5
    assert res['A'].iloc[5] < 100
    assert res['A'].iloc[5] <= 8.5
    assert len(res) == 6

def test_iqr_drop(outlier_df):
    calc = IQROutlierCalculator()
    params = calc.fit(outlier_df, {'action': 'drop', 'factor': 1.5})
    
    applier = IQROutlierApplier()
    res = applier.apply(outlier_df, params)
    
    assert len(res) == 5
    assert 100 not in res['A'].values

def test_zscore_clip(outlier_df):
    calc = ZScoreOutlierCalculator()
    params = calc.fit(outlier_df, {'action': 'clip', 'threshold': 2.0})
    
    applier = ZScoreOutlierApplier()
    res = applier.apply(outlier_df, params)
    
    assert res['A'].iloc[5] < 100
    assert len(res) == 6
