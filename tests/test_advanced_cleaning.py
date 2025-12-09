import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.cleaning import BooleanNormalizerCalculator, BooleanNormalizerApplier, CountryStandardizerCalculator, CountryStandardizerApplier

def test_boolean_normalizer():
    df = pd.DataFrame({
        'bool_col': ['yes', 'No', 'TRUE', '0', '1', 'invalid']
    })
    
    calc = BooleanNormalizerCalculator()
    applier = BooleanNormalizerApplier()
    
    config = {'columns': ['bool_col']}
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)
    
    expected = [True, False, True, False, True, pd.NA]
    # Check values. Note: pd.NA comparison needs care.
    res_list = result['bool_col'].tolist()
    
    assert res_list[0] is True
    assert res_list[1] is False
    assert res_list[2] is True
    assert res_list[3] is False
    assert res_list[4] is True
    assert pd.isna(res_list[5])

def test_country_standardizer():
    df = pd.DataFrame({
        'country': ['United States', 'UK', 'Turkey', 'Unknown']
    })
    
    calc = CountryStandardizerCalculator()
    applier = CountryStandardizerApplier()
    
    config = {'columns': ['country']}
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)
    
    assert result['country'].iloc[0] == 'USA'
    assert result['country'].iloc[1] == 'GBR'
    assert result['country'].iloc[2] == 'TUR'
    assert result['country'].iloc[3] == 'Unknown' # Should keep original if not mapped
