import pytest
import pandas as pd
import os
import shutil
from skyulf.data.dataset import SplitDataset
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.preprocessing.base import BaseCalculator, BaseApplier, StatefulTransformer

# Dummy Scaler Implementation for Testing
class DummyMeanScalerCalculator(BaseCalculator):
    def fit(self, df, config):
        # Calculate mean of numeric columns
        means = df.mean(numeric_only=True).to_dict()
        return {"means": means}

class DummyMeanScalerApplier(BaseApplier):
    def apply(self, df, params):
        df_copy = df.copy()
        means = params["means"]
        for col, mean_val in means.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col] - mean_val
        return df_copy

@pytest.fixture
def sample_data():
    train = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["x", "y", "z"]})
    test = pd.DataFrame({"A": [4, 5], "B": [7, 8], "C": ["a", "b"]})
    return SplitDataset(train=train, test=test)

@pytest.fixture
def artifact_store():
    path = "temp_test_artifacts"
    if os.path.exists(path):
        shutil.rmtree(path)
    store = LocalArtifactStore(path)
    yield store
    if os.path.exists(path):
        shutil.rmtree(path)

def test_stateful_transformer_flow(sample_data, artifact_store):
    # Setup
    calculator = DummyMeanScalerCalculator()
    applier = DummyMeanScalerApplier()
    # Removed artifact_store from constructor
    transformer = StatefulTransformer(calculator, applier, "node_1")
    
    # Execute Fit & Transform
    result_dataset = transformer.fit_transform(sample_data, {})
    
    # Verify Params Stored in Memory
    params = transformer.params
    assert params["means"]["A"] == 2.0 # (1+2+3)/3
    assert params["means"]["B"] == 5.0 # (4+5+6)/3
    
    # Verify Transformation (Train)
    # 1-2 = -1, 2-2 = 0, 3-2 = 1
    assert result_dataset.train["A"].tolist() == [-1.0, 0.0, 1.0]
    
    # Verify Transformation (Test) - Uses Train Mean (2.0)
    # 4-2 = 2, 5-2 = 3
    assert result_dataset.test["A"].tolist() == [2.0, 3.0]
    
def test_transformer_load_and_transform(sample_data, artifact_store):
    # Setup
    calculator = DummyMeanScalerCalculator()
    applier = DummyMeanScalerApplier()
    # Removed artifact_store from constructor
    transformer = StatefulTransformer(calculator, applier, "node_1")
    
    # Manually set params (simulating load)
    transformer.params = {"means": {"A": 10.0, "B": 10.0}}
    
    # Execute Transform only
    result_dataset = transformer.transform(sample_data)
    
    # Verify Transformation (Train) - Uses Loaded Mean (10.0)
    # 1-10 = -9
    assert result_dataset.train["A"].iloc[0] == -9.0
