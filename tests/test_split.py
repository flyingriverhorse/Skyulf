import numpy as np
import pandas as pd
import pytest
from skyulf.preprocessing.split import DataSplitter


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": ["A"] * 50 + ["B"] * 50,
        }
    )


def test_data_splitter_basic(sample_df):
    splitter = DataSplitter(test_size=0.2, random_state=42)
    ds = splitter.split(sample_df)

    # split() returns (DataFrame, None) tuples in SplitDataset
    train_df, train_y = ds.train
    test_df, test_y = ds.test
    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_y is None
    assert test_y is None
    assert ds.validation is None


def test_data_splitter_validation(sample_df):
    splitter = DataSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    ds = splitter.split(sample_df)

    # Total 100. Test=20. Val=10. Train=70.
    test_df, _ = ds.test
    val_df, _ = ds.validation
    train_df, _ = ds.train
    assert len(test_df) == 20
    assert len(val_df) == 10
    assert len(train_df) == 70


def test_data_splitter_stratify(sample_df):
    splitter = DataSplitter(test_size=0.2, stratify_col="target", random_state=42)
    ds = splitter.split(sample_df)

    # Check stratification in test set (should be 50/50 split of A/B roughly)
    # 20 samples total -> 10 A, 10 B
    test_df, _ = ds.test
    assert test_df["target"].value_counts()["A"] == 10
    assert test_df["target"].value_counts()["B"] == 10
