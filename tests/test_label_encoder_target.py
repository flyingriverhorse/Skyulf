import pandas as pd
import pytest
from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.encoding import LabelEncoderApplier, LabelEncoderCalculator
from skyulf.preprocessing.split import SplitApplier, SplitCalculator

from backend.ml_pipeline.artifacts.local import LocalArtifactStore


def test_label_encoder_on_target_after_split():
    # 1. Setup Data
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )

    # 2. Feature Target Split (Simulated)
    X = df[["feature1"]]
    y = df["target"]
    dataset_tuple = (X, y)

    # 3. Train Test Split
    split_calc = SplitCalculator()
    split_applier = SplitApplier()

    split_config = {"test_size": 0.2, "random_state": 42}
    # fit returns empty config
    split_params = split_calc.fit(dataset_tuple, split_config)
    # apply returns SplitDataset
    split_dataset = split_applier.apply(dataset_tuple, split_params)

    assert isinstance(split_dataset, SplitDataset)
    assert isinstance(split_dataset.train, tuple)
    assert len(split_dataset.train) == 2

    X_train, y_train = split_dataset.train
    assert y_train.dtype == "object"  # Should be strings 'A', 'B'

    # 4. Label Encoder on Target
    le_calc = LabelEncoderCalculator()
    le_applier = LabelEncoderApplier()

    le_config = {"columns": [], "drop_original": True}  # Empty to target y

    # Fit on Train
    le_params = le_calc.fit(split_dataset.train, le_config)

    # Check if target was detected
    assert "__target__" in le_params.get("encoders", {})
    # assert 'target' in le_params['encoders'] # Removed as we use __target__

    # Apply on Train
    train_transformed = le_applier.apply(split_dataset.train, le_params)
    X_train_new, y_train_new = train_transformed

    # Verify y_train is encoded (should be integers)
    assert pd.api.types.is_integer_dtype(y_train_new) or pd.api.types.is_float_dtype(
        y_train_new
    )
    assert set(y_train_new.unique()).issubset({0, 1})

    # Get the mapping from the encoder
    encoder = le_params["encoders"]["__target__"]
    classes = encoder.classes_
    # e.g. classes_ might be ['A', 'B'], so A->0, B->1

    # Check consistency on Train
    # We know y_train had 'A's and 'B's.
    # Let's pick an index where y_train was 'A' and check if y_train_new is the encoded value of 'A'
    original_y_train = split_dataset.train[1]

    # Find an index for 'A'
    idx_A = original_y_train[original_y_train == "A"].index[0]
    encoded_A = y_train_new.loc[idx_A]

    # Find an index for 'B'
    idx_B = original_y_train[original_y_train == "B"].index[0]
    encoded_B = y_train_new.loc[idx_B]

    assert encoded_A != encoded_B

    # Apply on Test
    test_transformed = le_applier.apply(split_dataset.test, le_params)
    X_test_new, y_test_new = test_transformed

    # Verify y_test is encoded
    assert pd.api.types.is_integer_dtype(y_test_new) or pd.api.types.is_float_dtype(
        y_test_new
    )
    assert set(y_test_new.unique()).issubset({0, 1})

    # Check consistency on Test
    original_y_test = split_dataset.test[1]

    # If Test has 'A', it must be encoded to same value as in Train
    if "A" in original_y_test.values:
        idx_test_A = original_y_test[original_y_test == "A"].index[0]
        assert y_test_new.loc[idx_test_A] == encoded_A

    # If Test has 'B', it must be encoded to same value as in Train
    if "B" in original_y_test.values:
        idx_test_B = original_y_test[original_y_test == "B"].index[0]
        assert y_test_new.loc[idx_test_B] == encoded_B

    print(f"Mapping Verified: A -> {encoded_A}, B -> {encoded_B}")
    print("Test Passed Successfully!")


if __name__ == "__main__":
    test_label_encoder_on_target_after_split()
