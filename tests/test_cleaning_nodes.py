import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.cleaning import (
    TextCleaningCalculator, TextCleaningApplier,
    ValueReplacementCalculator, ValueReplacementApplier
)

# --- Text Cleaning Tests ---

def test_text_cleaning_trim():
    df = pd.DataFrame({
        'A': ['  foo  ', 'bar  ', '  baz'],
        'B': [1, 2, 3]
    })
    
    config = {
        'columns': ['A'],
        'operations': [
            {'op': 'trim', 'mode': 'both'}
        ]
    }
    
    calc = TextCleaningCalculator()
    params = calc.fit(df, config)
    
    applier = TextCleaningApplier()
    res = applier.apply(df, params)
    
    assert res['A'].tolist() == ['foo', 'bar', 'baz']

def test_text_cleaning_case():
    df = pd.DataFrame({
        'A': ['Foo', 'BAR', 'baz'],
    })
    
    config = {
        'columns': ['A'],
        'operations': [
            {'op': 'case', 'mode': 'upper'}
        ]
    }
    
    calc = TextCleaningCalculator()
    params = calc.fit(df, config)
    
    applier = TextCleaningApplier()
    res = applier.apply(df, params)
    
    assert res['A'].tolist() == ['FOO', 'BAR', 'BAZ']

def test_text_cleaning_remove_special():
    df = pd.DataFrame({
        'A': ['foo@bar.com', '123#456', 'hello world!'],
    })
    
    config = {
        'columns': ['A'],
        'operations': [
            {'op': 'remove_special', 'mode': 'keep_alphanumeric'}
        ]
    }
    
    calc = TextCleaningCalculator()
    params = calc.fit(df, config)
    
    applier = TextCleaningApplier()
    res = applier.apply(df, params)
    
    # keep_alphanumeric removes everything except a-z, A-Z, 0-9
    assert res['A'].tolist() == ['foobarcom', '123456', 'helloworld']

def test_text_cleaning_regex():
    df = pd.DataFrame({
        'A': ['foo  bar', 'baz   qux'],
    })
    
    config = {
        'columns': ['A'],
        'operations': [
            {'op': 'regex', 'mode': 'collapse_whitespace'}
        ]
    }
    
    calc = TextCleaningCalculator()
    params = calc.fit(df, config)
    
    applier = TextCleaningApplier()
    res = applier.apply(df, params)
    
    assert res['A'].tolist() == ['foo bar', 'baz qux']

def test_text_cleaning_multiple_ops():
    df = pd.DataFrame({
        'A': ['  Foo@Bar  ', '  BAZ#QUX  '],
    })
    
    config = {
        'columns': ['A'],
        'operations': [
            {'op': 'trim', 'mode': 'both'},
            {'op': 'case', 'mode': 'lower'},
            {'op': 'remove_special', 'mode': 'keep_alphanumeric'}
        ]
    }
    
    calc = TextCleaningCalculator()
    params = calc.fit(df, config)
    
    applier = TextCleaningApplier()
    res = applier.apply(df, params)
    
    assert res['A'].tolist() == ['foobar', 'bazqux']

# --- Value Replacement Tests ---

def test_value_replacement_simple():
    df = pd.DataFrame({
        'A': ['Yes', 'No', 'Maybe'],
        'B': [1, 2, 3]
    })
    
    config = {
        'columns': ['A'],
        'mapping': {'Yes': 1, 'No': 0}
    }
    
    calc = ValueReplacementCalculator()
    params = calc.fit(df, config)
    
    applier = ValueReplacementApplier()
    res = applier.apply(df, params)
    
    assert res['A'].tolist() == [1, 0, 'Maybe']

def test_value_replacement_numeric():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
    })
    
    config = {
        'columns': ['A'],
        'mapping': {1: 10, 2: 20}
    }
    
    calc = ValueReplacementCalculator()
    params = calc.fit(df, config)
    
    applier = ValueReplacementApplier()
    res = applier.apply(df, params)
    
    assert res['A'].tolist() == [10, 20, 3, 4]


def test_value_replacement_replacements_list():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
    })
    
    config = {
        'columns': ['A'],
        'replacements': [
            {'old': 1, 'new': 100},
            {'old': 2, 'new': 200}
        ]
    }
    
    calc = ValueReplacementCalculator()
    params = calc.fit(df, config)
    
    applier = ValueReplacementApplier()
    res = applier.apply(df, params)
    
    assert res['A'].tolist() == [100, 200, 3, 4]


# --- Alias Replacement Tests ---

from core.ml_pipeline.preprocessing.cleaning import AliasReplacementCalculator, AliasReplacementApplier

def test_alias_replacement_custom():
    df = pd.DataFrame({
        'A': ['NY', 'CA', 'TX'],
    })
    
    config = {
        'columns': ['A'],
        'mode': 'custom',
        'custom_pairs': {'NY': 'New York', 'CA': 'California'}
    }
    
    calc = AliasReplacementCalculator()
    params = calc.fit(df, config)
    
    applier = AliasReplacementApplier()
    res = applier.apply(df, params)
    
    assert res['A'].tolist() == ['New York', 'California', 'TX']


# --- Invalid Value Replacement Tests ---

from core.ml_pipeline.preprocessing.cleaning import InvalidValueReplacementCalculator, InvalidValueReplacementApplier

def test_invalid_value_replacement_negative():
    df = pd.DataFrame({
        'A': [10, -5, 20, -1],
    })
    
    config = {
        'columns': ['A'],
        'mode': 'negative_to_nan'
    }
    
    calc = InvalidValueReplacementCalculator()
    params = calc.fit(df, config)
    
    applier = InvalidValueReplacementApplier()
    res = applier.apply(df, params)
    
    assert pd.isna(res['A'][1])
    assert pd.isna(res['A'][3])
    assert res['A'][0] == 10

def test_invalid_value_replacement_range():
    df = pd.DataFrame({
        'A': [10, 150, 50, -10],
    })
    
    config = {
        'columns': ['A'],
        'mode': 'custom_range',
        'min_value': 0,
        'max_value': 100
    }
    
    calc = InvalidValueReplacementCalculator()
    params = calc.fit(df, config)
    
    applier = InvalidValueReplacementApplier()
    res = applier.apply(df, params)
    
    # 150 > 100 -> NaN
    # -10 < 0 -> NaN
    assert pd.isna(res['A'][1])
    assert pd.isna(res['A'][3])
    assert res['A'][0] == 10
    assert res['A'][2] == 50
