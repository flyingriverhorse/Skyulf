import pytest
import pandas as pd
import numpy as np
from skyulf.preprocessing.split import DataSplitter

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'feature1': range(100),
        'feature2': range(100, 200),
        'target': ['A'] * 50 + ['B'] * 50
    })

def test_data_splitter_basic(sample_df):
    splitter = DataSplitter(test_size=0.2, random_state=42)
    ds = splitter.split(sample_df)
    
    assert len(ds.train) == 80
    assert len(ds.test) == 20
    assert ds.validation is None

def test_data_splitter_validation(sample_df):
    splitter = DataSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    ds = splitter.split(sample_df)
    
    # Total 100. Test=20. Val=10. Train=70.
    assert len(ds.test) == 20
    assert len(ds.validation) == 10
    assert len(ds.train) == 70

def test_data_splitter_stratify(sample_df):
    splitter = DataSplitter(test_size=0.2, stratify_col='target', random_state=42)
    ds = splitter.split(sample_df)
    
    # Check stratification in test set (should be 50/50 split of A/B roughly)
    # 20 samples total -> 10 A, 10 B
    assert ds.test['target'].value_counts()['A'] == 10
    assert ds.test['target'].value_counts()['B'] == 10


