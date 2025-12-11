import pytest
import pandas as pd
import numpy as np
from core.ml_pipeline.preprocessing.imputation import SimpleImputerCalculator, SimpleImputerApplier
from core.ml_pipeline.preprocessing.encoding import OneHotEncoderCalculator, OneHotEncoderApplier
from core.ml_pipeline.preprocessing.scaling import StandardScalerCalculator, StandardScalerApplier

def test_pipeline_chain():
    # 1. Create Data
    df_train = pd.DataFrame({
        'age': [25, np.nan, 30, 35],
        'city': ['NY', 'LA', 'NY', 'SF'],
        'salary': [50000, 60000, 70000, 80000]
    })
    
    df_test = pd.DataFrame({
        'age': [20, 40],
        'city': ['LA', 'NY'], # SF is missing in test, that's fine
        'salary': [45000, 90000]
    })

    # 2. Imputation (Fit on Train)
    imputer_calc = SimpleImputerCalculator()
    imputer_config = {'columns': ['age'], 'strategy': 'mean'}
    imputer_artifact = imputer_calc.fit(df_train, imputer_config)
    
    # 3. Imputation (Apply to Train & Test)
    imputer_applier = SimpleImputerApplier()
    df_train_imputed = imputer_applier.apply(df_train, imputer_artifact)
    df_test_imputed = imputer_applier.apply(df_test, imputer_artifact)
    
    assert df_train_imputed['age'].isna().sum() == 0
    assert df_test_imputed['age'].isna().sum() == 0
    
    # 4. Encoding (Fit on Train Imputed)
    encoder_calc = OneHotEncoderCalculator()
    encoder_config = {'columns': ['city']}
    encoder_artifact = encoder_calc.fit(df_train_imputed, encoder_config)
    
    # 5. Encoding (Apply to Train & Test Imputed)
    encoder_applier = OneHotEncoderApplier()
    df_train_encoded = encoder_applier.apply(df_train_imputed, encoder_artifact)
    df_test_encoded = encoder_applier.apply(df_test_imputed, encoder_artifact)
    
    assert 'city_NY' in df_train_encoded.columns
    assert 'city_NY' in df_test_encoded.columns
    
    # 6. Scaling (Fit on Train Encoded)
    scaler_calc = StandardScalerCalculator()
    scaler_config = {'columns': ['age', 'salary']}
    scaler_artifact = scaler_calc.fit(df_train_encoded, scaler_config)
    
    # 7. Scaling (Apply to Train & Test Encoded)
    scaler_applier = StandardScalerApplier()
    df_train_scaled = scaler_applier.apply(df_train_encoded, scaler_artifact)
    df_test_scaled = scaler_applier.apply(df_test_encoded, scaler_artifact)
    
    # Check results
    assert df_train_scaled['age'].mean() < 1e-10 # Should be approx 0 (standardized)
    assert df_test_scaled.shape[1] == df_train_scaled.shape[1] # Should have same columns
