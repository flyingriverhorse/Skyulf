import pytest
import pandas as pd
import numpy as np
from skyulf.preprocessing.bucketing import GeneralBinningCalculator, GeneralBinningApplier

@pytest.fixture
def sample_data():
    # Create a dataframe with clear patterns for testing
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # Uniform
        'B': [1, 1, 1, 2, 2, 5, 8, 8, 9, 10], # Skewed
        'C': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    })
    return df

def test_equal_width_binning(sample_data):
    config = {
        'columns': ['A'],
        'strategy': 'equal_width',
        'n_bins': 2,
        'output_suffix': '_binned',
        'label_format': 'ordinal'
    }
    
    calc = GeneralBinningCalculator()
    params = calc.fit(sample_data, config)
    
    applier = GeneralBinningApplier()
    result = applier.apply(sample_data, params)
    
    assert 'A_binned' in result.columns
    # 1-5.5 should be bin 0, 5.5-10 should be bin 1
    assert result['A_binned'].iloc[0] == 0.0
    assert result['A_binned'].iloc[-1] == 1.0

def test_equal_frequency_binning(sample_data):
    config = {
        'columns': ['B'],
        'strategy': 'equal_frequency',
        'n_bins': 2,
        'output_suffix': '_q',
        'label_format': 'ordinal'
    }
    
    calc = GeneralBinningCalculator()
    params = calc.fit(sample_data, config)
    
    applier = GeneralBinningApplier()
    result = applier.apply(sample_data, params)
    
    assert 'B_q' in result.columns
    # Should have roughly equal counts
    counts = result['B_q'].value_counts()
    assert abs(counts.iloc[0] - counts.iloc[1]) <= 1

def test_custom_binning(sample_data):
    config = {
        'columns': ['C'],
        'strategy': 'custom',
        'custom_bins': {'C': [0, 50, 100]},
        'output_suffix': '_custom',
        'label_format': 'ordinal'
    }
    
    calc = GeneralBinningCalculator()
    params = calc.fit(sample_data, config)
    
    applier = GeneralBinningApplier()
    result = applier.apply(sample_data, params)
    
    assert 'C_custom' in result.columns
    # 10-50 -> bin 0, 60-100 -> bin 1
    assert result['C_custom'].iloc[0] == 0.0 # 10
    assert result['C_custom'].iloc[4] == 0.0 # 50 (include_lowest=True usually handles left edge, but pd.cut default right=True)
    # pd.cut(..., right=True) -> (0, 50], (50, 100]
    assert result['C_custom'].iloc[5] == 1.0 # 60

def test_kmeans_binning(sample_data):
    # Kmeans requires sklearn
    config = {
        'columns': ['A'],
        'strategy': 'kbins',
        'kbins_strategy': 'kmeans',
        'n_bins': 3,
        'output_suffix': '_kmeans'
    }
    
    calc = GeneralBinningCalculator()
    params = calc.fit(sample_data, config)
    
    applier = GeneralBinningApplier()
    result = applier.apply(sample_data, params)
    
    assert 'A_kmeans' in result.columns
    assert result['A_kmeans'].nunique() == 3

def test_label_format_range(sample_data):
    config = {
        'columns': ['A'],
        'strategy': 'equal_width',
        'n_bins': 2,
        'label_format': 'range',
        'output_suffix': '_range'
    }
    
    calc = GeneralBinningCalculator()
    params = calc.fit(sample_data, config)
    
    applier = GeneralBinningApplier()
    result = applier.apply(sample_data, params)
    
    assert 'A_range' in result.columns
    val = result['A_range'].iloc[0]
    assert isinstance(val, str)
    assert '(' in val or '[' in val
    assert ']' in val

def test_drop_original(sample_data):
    config = {
        'columns': ['A'],
        'strategy': 'equal_width',
        'n_bins': 2,
        'drop_original': True
    }
    
    calc = GeneralBinningCalculator()
    params = calc.fit(sample_data, config)
    
    applier = GeneralBinningApplier()
    result = applier.apply(sample_data, params)
    
    assert 'A_binned' in result.columns
    assert 'A' not in result.columns

def test_multiple_columns_mixed_strategies(sample_data):
    # This tests the "General" calculator's ability to handle overrides if implemented, 
    # or just applying the global strategy to multiple columns
    config = {
        'columns': ['A', 'B'],
        'strategy': 'equal_width',
        'n_bins': 3
    }
    
    calc = GeneralBinningCalculator()
    params = calc.fit(sample_data, config)
    
    applier = GeneralBinningApplier()
    result = applier.apply(sample_data, params)
    
    assert 'A_binned' in result.columns
    assert 'B_binned' in result.columns
