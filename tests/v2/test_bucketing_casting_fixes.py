import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.bucketing import CustomBinningCalculator, CustomBinningApplier
from core.ml_pipeline.preprocessing.casting import CastingCalculator, CastingApplier

def test_custom_binning():
    df = pd.DataFrame({'age': [5, 15, 25, 35, 45]})
    calc = CustomBinningCalculator()
    applier = CustomBinningApplier()
    
    # Bins: 0-10, 10-20, 20-30, 30-100
    config = {
        'bins': {'age': [0, 10, 20, 30, 100]},
        'label_format': 'bin_index'
    }
    
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)
    
    # 5 -> 0 (0-10)
    # 15 -> 1 (10-20)
    # 25 -> 2 (20-30)
    # 35 -> 3 (30-100)
    # 45 -> 3 (30-100)
    
    assert 'age_binned' in result.columns
    assert result['age_binned'].iloc[0] == 0
    assert result['age_binned'].iloc[1] == 1
    assert result['age_binned'].iloc[3] == 3

def test_casting_int_coerce():
    df = pd.DataFrame({'id': ['1', '2', 'invalid', '4']})
    calc = CastingCalculator()
    applier = CastingApplier()
    
    config = {'columns': ['id'], 'target_type': 'int', 'coerce_on_error': True}
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)
    
    # Should be nullable Int64 because of 'invalid' -> NaN
    assert pd.api.types.is_integer_dtype(result['id'])
    assert result['id'].iloc[2] is pd.NA or np.isnan(result['id'].iloc[2])
    assert result['id'].iloc[0] == 1
