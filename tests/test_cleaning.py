import pytest
import pandas as pd
import numpy as np
from skyulf.preprocessing.cleaning import (
    TextCleaningCalculator, TextCleaningApplier,
    ValueReplacementCalculator, ValueReplacementApplier
)

def test_text_cleaning():
    df = pd.DataFrame({
        'text': ['  Hello  ', 'WORLD', '123-456']
    })
    
    calc = TextCleaningCalculator()
    config = {
        'columns': ['text'],
        'operations': [
            {'op': 'trim'},
            {'op': 'case', 'mode': 'lower'},
            {'op': 'regex', 'pattern': '-', 'repl': ''}
        ]
    }
    params = calc.fit(df, config)
    
    applier = TextCleaningApplier()
    res = applier.apply(df, params)
    
    assert res['text'].iloc[0] == 'hello'
    assert res['text'].iloc[1] == 'world'
    assert res['text'].iloc[2] == '123456'

def test_value_replacement():
    df = pd.DataFrame({
        'A': [1, 2, 999]
    })
    
    calc = ValueReplacementCalculator()
    config = {
        'columns': ['A'],
        'mapping': {999: np.nan}
    }
    params = calc.fit(df, config)
    
    applier = ValueReplacementApplier()
    res = applier.apply(df, params)
    
    assert pd.isna(res['A'].iloc[2])
    assert res['A'].iloc[0] == 1
