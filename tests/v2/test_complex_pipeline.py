import pytest
import pandas as pd
import numpy as np
import os
import shutil
from core.ml_pipeline.data.container import SplitDataset
from core.ml_pipeline.artifacts.local import LocalArtifactStore
from core.ml_pipeline.preprocessing.base import StatefulTransformer

# Import nodes
from core.ml_pipeline.preprocessing.cleaning import DateStandardizerCalculator, DateStandardizerApplier
from core.ml_pipeline.preprocessing.casting import CastingCalculator, CastingApplier
from core.ml_pipeline.preprocessing.imputation import SimpleImputerCalculator, SimpleImputerApplier
from core.ml_pipeline.preprocessing.encoding import OneHotEncoderCalculator, OneHotEncoderApplier
from core.ml_pipeline.preprocessing.scaling import StandardScalerCalculator, StandardScalerApplier
from core.ml_pipeline.preprocessing.feature_selection import VarianceThresholdCalculator, VarianceThresholdApplier

@pytest.fixture
def complex_data():
    # Create a complex dataset
    df = pd.DataFrame({
        'id': ['1', '2', '3', '4', '5'],
        'date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'],
        'category': ['A', 'B', 'A', 'C', np.nan],
        'value': [10.0, 20.0, np.nan, 40.0, 50.0],
        'constant': [1, 1, 1, 1, 1]
    })
    return SplitDataset(train=df, test=df.copy())

@pytest.fixture
def artifact_store():
    path = "temp_complex_pipeline_artifacts"
    if os.path.exists(path):
        shutil.rmtree(path)
    store = LocalArtifactStore(path)
    yield store
    if os.path.exists(path):
        shutil.rmtree(path)

def test_complex_pipeline_flow(complex_data, artifact_store):
    # 1. Cast ID to int
    cast_calc = CastingCalculator()
    cast_app = CastingApplier()
    cast_node = StatefulTransformer(cast_calc, cast_app, artifact_store, "node_1_casting")
    
    # 2. Standardize Date
    date_calc = DateStandardizerCalculator()
    date_app = DateStandardizerApplier()
    date_node = StatefulTransformer(date_calc, date_app, artifact_store, "node_2_date")
    
    # 3. Impute Missing Values (Category -> Mode, Value -> Mean)
    impute_calc = SimpleImputerCalculator()
    impute_app = SimpleImputerApplier()
    impute_node_cat = StatefulTransformer(impute_calc, impute_app, artifact_store, "node_3_impute_cat")
    impute_node_val = StatefulTransformer(impute_calc, impute_app, artifact_store, "node_4_impute_val")
    
    # 4. Encode Category
    enc_calc = OneHotEncoderCalculator()
    enc_app = OneHotEncoderApplier()
    enc_node = StatefulTransformer(enc_calc, enc_app, artifact_store, "node_5_encoding")
    
    # 5. Drop Constant Features
    sel_calc = VarianceThresholdCalculator()
    sel_app = VarianceThresholdApplier()
    sel_node = StatefulTransformer(sel_calc, sel_app, artifact_store, "node_6_selection")
    
    # 6. Scale Value
    scale_calc = StandardScalerCalculator()
    scale_app = StandardScalerApplier()
    scale_node = StatefulTransformer(scale_calc, scale_app, artifact_store, "node_7_scaling")
    
    # --- Execution ---
    
    # Step 1: Casting
    ds = complex_data
    ds = cast_node.fit_transform(ds, {'columns': ['id'], 'target_type': 'int'})
    assert pd.api.types.is_integer_dtype(ds.train['id'])
    
    # Step 2: Date
    ds = date_node.fit_transform(ds, {'columns': ['date'], 'target_format': '%Y/%m/%d'})
    assert ds.train['date'].iloc[0] == '2021/01/01'
    
    # Step 3: Impute
    ds = impute_node_cat.fit_transform(ds, {'columns': ['category'], 'strategy': 'most_frequent'})
    assert not ds.train['category'].isna().any()
    
    ds = impute_node_val.fit_transform(ds, {'columns': ['value'], 'strategy': 'mean'})
    assert not ds.train['value'].isna().any()
    
    # Step 4: Encode
    ds = enc_node.fit_transform(ds, {'columns': ['category']})
    assert 'category_A' in ds.train.columns
    
    # Step 5: Selection (Should drop 'constant')
    ds = sel_node.fit_transform(ds, {'threshold': 0.0})
    assert 'constant' not in ds.train.columns
    
    # Step 6: Scaling
    ds = scale_node.fit_transform(ds, {'columns': ['value']})
    assert abs(ds.train['value'].mean()) < 1e-6 # Standardized mean ~ 0
    
    print("Pipeline execution successful!")
