import pandas as pd
import pytest
from skyulf.preprocessing.encoding import LabelEncoderCalculator, LabelEncoderApplier
from skyulf.preprocessing.split import FeatureTargetSplitApplier, SplitApplier
from skyulf.data.dataset import SplitDataset

def create_sample_data():
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'species': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C']
    })

def test_scenario_1_xy_then_split_then_encode():
    """
    Scenario 1: Feature-Target Split -> Train-Test Split -> Encoding
    """
    print("\n--- Scenario 1: Feature-Target Split -> Train-Test Split -> Encoding ---")
    df = create_sample_data()
    
    # 1. Feature Target Split
    ft_splitter = FeatureTargetSplitApplier()
    xy_data = ft_splitter.apply(df, {'target_column': 'species'})
    
    assert isinstance(xy_data, tuple)
    X, y = xy_data
    print(f"Step 1 (XY Split): X shape={X.shape}, y shape={y.shape}")
    
    # 2. Train Test Split
    tt_splitter = SplitApplier()
    split_data = tt_splitter.apply(xy_data, {'test_size': 0.2, 'random_state': 42})
    
    assert isinstance(split_data, SplitDataset)
    X_train, y_train = split_data.train
    X_test, y_test = split_data.test
    print(f"Step 2 (Train/Test Split): Train size={len(X_train)}, Test size={len(X_test)}")
    
    # 3. Encoding
    # Config: User selects 'species' (target) to encode
    config = {'method': 'label', 'columns': ['species']}
    
    calculator = LabelEncoderCalculator()
    # Fit on TRAIN data
    fit_result = calculator.fit(split_data.train, config)
    
    print(f"Step 3 (Fit): Encoders found: {list(fit_result.get('encoders', {}).keys())}")
    assert '__target__' in fit_result['encoders'], "Target encoder should be present"
    
    applier = LabelEncoderApplier()
    
    # Apply on TRAIN
    train_transformed = applier.apply(split_data.train, fit_result)
    _, y_train_enc = train_transformed
    print(f"Step 3 (Apply Train): y_train sample: {y_train_enc.head(3).tolist()}")
    assert pd.api.types.is_numeric_dtype(y_train_enc), "y_train should be numeric"
    
    # Apply on TEST
    test_transformed = applier.apply(split_data.test, fit_result)
    _, y_test_enc = test_transformed
    print(f"Step 3 (Apply Test): y_test sample: {y_test_enc.head(3).tolist()}")
    assert pd.api.types.is_numeric_dtype(y_test_enc), "y_test should be numeric"


def test_scenario_2_split_then_xy_then_encode():
    """
    Scenario 2: Train-Test Split -> Feature-Target Split -> Encoding
    """
    print("\n--- Scenario 2: Train-Test Split -> Feature-Target Split -> Encoding ---")
    df = create_sample_data()
    
    # 1. Train Test Split (on DataFrame)
    tt_splitter = SplitApplier()
    split_df = tt_splitter.apply(df, {'test_size': 0.2, 'random_state': 42})
    
    assert isinstance(split_df, SplitDataset)
    # split_df.train is (DataFrame, None) tuple — unpack for shape check
    train_df, _ = split_df.train
    print(f"Step 1 (Train/Test Split): Train shape={train_df.shape}")
    
    # 2. Feature Target Split (on SplitDataset)
    ft_splitter = FeatureTargetSplitApplier()
    split_xy = ft_splitter.apply(split_df, {'target_column': 'species'})
    
    assert isinstance(split_xy, SplitDataset)
    X_train, y_train = split_xy.train
    print(f"Step 2 (XY Split): X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    
    # 3. Encoding
    config = {'method': 'label', 'columns': ['species']}
    
    calculator = LabelEncoderCalculator()
    # Fit on TRAIN data
    fit_result = calculator.fit(split_xy.train, config)
    
    print(f"Step 3 (Fit): Encoders found: {list(fit_result.get('encoders', {}).keys())}")
    assert '__target__' in fit_result['encoders'], "Target encoder should be present"
    
    applier = LabelEncoderApplier()
    
    # Apply on TRAIN
    train_transformed = applier.apply(split_xy.train, fit_result)
    _, y_train_enc = train_transformed
    print(f"Step 3 (Apply Train): y_train sample: {y_train_enc.head(3).tolist()}")
    assert pd.api.types.is_numeric_dtype(y_train_enc), "y_train should be numeric"
    
    # Apply on TEST
    test_transformed = applier.apply(split_xy.test, fit_result)
    _, y_test_enc = test_transformed
    print(f"Step 3 (Apply Test): y_test sample: {y_test_enc.head(3).tolist()}")
    assert pd.api.types.is_numeric_dtype(y_test_enc), "y_test should be numeric"

def test_feature_encoding():
    """
    Scenario 3: Encoding a feature column (e.g. Label Encoding 'feature1' if it were categorical)
    """
    print("\n--- Scenario 3: Feature Encoding ---")
    df = pd.DataFrame({
        'cat_feature': ['low', 'high', 'medium', 'low', 'high'],
        'target': [0, 1, 0, 0, 1]
    })
    
    # Config: Encode 'cat_feature'
    config = {'method': 'label', 'columns': ['cat_feature']}
    
    calculator = LabelEncoderCalculator()
    fit_result = calculator.fit(df, config)
    
    print(f"Step 1 (Fit): Encoders found: {list(fit_result.get('encoders', {}).keys())}")
    assert 'cat_feature' in fit_result['encoders'], "Feature encoder should be present"
    assert '__target__' not in fit_result['encoders'], "Target encoder should NOT be present"
    
    applier = LabelEncoderApplier()
    df_transformed = applier.apply(df, fit_result)
    
    # Check if 'cat_feature' is numeric
    print(f"Step 2 (Apply): cat_feature sample: {df_transformed['cat_feature'].tolist()}")
    assert pd.api.types.is_numeric_dtype(df_transformed['cat_feature']), "cat_feature should be numeric"

def test_serialization():
    """
    Scenario 4: Serialization (Pickling fitted params and reloading)
    """
    import pickle
    print("\n--- Scenario 4: Serialization ---")
    
    df = create_sample_data()
    # Split X, y
    X = df[['feature1']]
    y = df['species']
    y.name = 'species'
    
    config = {'method': 'label', 'columns': ['species']}
    
    # Fit
    calculator = LabelEncoderCalculator()
    fit_result = calculator.fit((X, y), config)
    
    # Serialize
    serialized = pickle.dumps(fit_result)
    print(f"Step 1: Serialized size = {len(serialized)} bytes")
    
    # Deserialize
    loaded_fit_result = pickle.loads(serialized)
    print(f"Step 2: Deserialized keys = {loaded_fit_result.keys()}")
    
    # Apply using loaded params
    applier = LabelEncoderApplier()
    result = applier.apply((X, y), loaded_fit_result)
    _, y_out = result
    
    print(f"Step 3 (Apply with Loaded Params): y sample: {y_out.head(3).tolist()}")
    assert pd.api.types.is_numeric_dtype(y_out), "y should be numeric after loading"

if __name__ == "__main__":
    test_scenario_1_xy_then_split_then_encode()
    test_scenario_2_split_then_xy_then_encode()
    test_feature_encoding()
    test_serialization()
