import numpy as np
import pandas as pd
import pytest
from skyulf.preprocessing.encoding import (
    DummyEncoderApplier,
    DummyEncoderCalculator,
    HashEncoderApplier,
    HashEncoderCalculator,
    LabelEncoderApplier,
    LabelEncoderCalculator,
    OneHotEncoderApplier,
    OneHotEncoderCalculator,
    OrdinalEncoderApplier,
    OrdinalEncoderCalculator,
    TargetEncoderApplier,
    TargetEncoderCalculator,
)


@pytest.fixture
def sample_df_cat():
    return pd.DataFrame(
        {
            "Color": ["Red", "Blue", "Green", "Red", "Blue"],
            "Size": ["S", "M", "L", "S", "M"],
            "Target": [1, 0, 1, 1, 0],
        }
    )


def test_onehot_encoder(sample_df_cat):
    # 1. Fit
    calc = OneHotEncoderCalculator()
    config = {"columns": ["Color"], "drop_first": False}
    params = calc.fit(sample_df_cat, config)

    assert params["type"] == "onehot"
    assert len(params["feature_names"]) == 3  # Red, Blue, Green

    # 2. Apply
    applier = OneHotEncoderApplier()
    transformed_df = applier.apply(sample_df_cat, params)

    # Original column removed
    assert "Color" not in transformed_df.columns
    # New columns added
    assert "Color_Red" in transformed_df.columns
    assert "Color_Blue" in transformed_df.columns
    assert transformed_df["Color_Red"].iloc[0] == 1


def test_onehot_encoder_unknown(sample_df_cat):
    # Fit on sample
    calc = OneHotEncoderCalculator()
    params = calc.fit(sample_df_cat, {"columns": ["Color"]})

    # Apply on new data with unknown category
    new_df = pd.DataFrame({"Color": ["Yellow"], "Size": ["S"], "Target": [0]})
    applier = OneHotEncoderApplier()
    transformed_df = applier.apply(new_df, params)

    # Should be all zeros for colors (handle_unknown='ignore')
    assert transformed_df["Color_Red"].iloc[0] == 0
    assert transformed_df["Color_Blue"].iloc[0] == 0
    assert transformed_df["Color_Green"].iloc[0] == 0


def test_ordinal_encoder(sample_df_cat):
    # 1. Fit
    calc = OrdinalEncoderCalculator()
    config = {"columns": ["Size"]}
    params = calc.fit(sample_df_cat, config)

    assert params["type"] == "ordinal"

    # 2. Apply
    applier = OrdinalEncoderApplier()
    transformed_df = applier.apply(sample_df_cat, params)

    # Should be numeric now
    assert pd.api.types.is_numeric_dtype(transformed_df["Size"])
    # S, M, L -> 0, 1, 2 (alphabetical usually, or appearance)
    # L=0, M=1, S=2
    assert transformed_df["Size"].iloc[0] == 2.0  # S


def test_target_encoder(sample_df_cat):
    # 1. Fit
    calc = TargetEncoderCalculator()
    config = {"columns": ["Color"], "target_column": "Target"}

    # Must pass (X, y) tuple for TargetEncoder
    X = sample_df_cat.drop(columns=["Target"])
    y = sample_df_cat["Target"]

    params = calc.fit((X, y), config)

    assert params["type"] == "target_encoder"

    # 2. Apply
    applier = TargetEncoderApplier()
    transformed_df, _ = applier.apply((X, y), params)

    # Should be numeric (replaced by mean target)
    assert pd.api.types.is_numeric_dtype(transformed_df["Color"])
    # Red appears twice, Target 1 and 1. Mean = 1.0.
    # With smoothing, it might be slightly different, but close to 1.
    assert transformed_df["Color"].iloc[0] > 0.5


def test_hash_encoder(sample_df_cat):
    # 1. Fit
    calc = HashEncoderCalculator()
    config = {"columns": ["Color"], "n_features": 4}
    params = calc.fit(sample_df_cat, config)

    assert params["type"] == "hash_encoder"

    # 2. Apply
    applier = HashEncoderApplier()
    transformed_df = applier.apply(sample_df_cat, params)

    # Original replaced by hash (numeric)
    assert "Color" in transformed_df.columns
    assert pd.api.types.is_numeric_dtype(transformed_df["Color"])

    # Deterministic check: Red should always hash to same values
    red_row_1 = transformed_df.iloc[0]
    red_row_2 = transformed_df.iloc[3]
    assert red_row_1["Color"] == red_row_2["Color"]


def test_dummy_encoder(sample_df_cat):
    # 1. Fit
    calc = DummyEncoderCalculator()
    config = {"columns": ["Color"], "drop_first": True}
    params = calc.fit(sample_df_cat, config)

    assert params["type"] == "dummy_encoder"
    # Categories: Blue, Green, Red. Drop Blue (first). Keep Green, Red.
    # assert 'Blue' not in params['categories']['Color'] # params stores all categories

    # 2. Apply
    applier = DummyEncoderApplier()
    transformed_df = applier.apply(sample_df_cat, params)

    # Check columns
    assert "Color_Red" in transformed_df.columns
    assert "Color_Green" in transformed_df.columns
    assert "Color_Blue" not in transformed_df.columns

    # Check values
    assert transformed_df["Color_Red"].iloc[0] == 1  # Red
    assert transformed_df["Color_Red"].iloc[1] == 0  # Blue


def test_label_encoder(sample_df_cat):
    # 1. Fit
    calc = LabelEncoderCalculator()
    config = {"columns": ["Size"]}
    params = calc.fit(sample_df_cat, config)

    assert params["type"] == "label_encoder"

    # 2. Apply
    applier = LabelEncoderApplier()
    transformed_df = applier.apply(sample_df_cat, params)

    # Should be numeric now
    assert pd.api.types.is_numeric_dtype(transformed_df["Size"])
    # L=0, M=1, S=2 (alphabetical)
    assert transformed_df["Size"].iloc[0] == 2  # S

    # Test unknown
    new_df = pd.DataFrame({"Size": ["XL"], "Color": ["Red"], "Target": [0]})
    transformed_new = applier.apply(new_df, params)
    assert transformed_new["Size"].iloc[0] == -1  # Unknown mapped to -1
