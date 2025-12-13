import pytest
import pandas as pd
import numpy as np
from skyulf.preprocessing.feature_selection import (
    FeatureSelectionCalculator, FeatureSelectionApplier
)

# --- Fixtures ---

@pytest.fixture
def regression_data():
    # A: Strong linear relationship
    # B: Weak linear relationship
    # C: Random noise (corr ~ 0.0)
    # D: Low variance noise
    np.random.seed(42)
    n = 100
    A = np.linspace(0, 10, n)
    B = np.random.uniform(0, 10, n) # Independent of A
    C = np.random.normal(0, 10, n) # High variance noise
    D = np.random.normal(0, 1, n) # Standard variance noise (was 0.01, causing RFE scale issues)
    
    # Target depends strongly on A, moderately on B
    target = 2 * A + 1.5 * B + np.random.normal(0, 0.1, n)
    
    df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'target': target})
    return df

@pytest.fixture
def classification_data():
    # A: Separates classes well
    # B: Random noise
    # C: Constant
    np.random.seed(42)
    n = 100
    A = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)])
    B = np.random.rand(n)
    C = np.ones(n)
    target = np.concatenate([np.zeros(50), np.ones(50)]) # 0 and 1 classes
    
    df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'target': target})
    return df

# --- Tests ---

def test_variance_threshold(regression_data):
    calc = FeatureSelectionCalculator()
    applier = FeatureSelectionApplier()
    
    # Add a constant column explicitly for this test
    df = regression_data.copy()
    df['constant'] = 1
    
    # Threshold 0 drops constant
    config = {"method": "variance_threshold", "threshold": 0.0}
    params = calc.fit(df, config)
    res = applier.apply(df, params)
    
    assert "constant" not in res.columns
    assert "A" in res.columns
    assert "target" in res.columns

def test_univariate_k_best_regression(regression_data):
    calc = FeatureSelectionCalculator()
    applier = FeatureSelectionApplier()
    
    # Select top 2 features (A and B should be best)
    config = {
        "method": "select_k_best",
        "k": 2,
        "score_func": "f_regression",
        "target_column": "target",
        "problem_type": "regression"
    }
    params = calc.fit(regression_data, config)
    res = applier.apply(regression_data, params)
    
    assert "A" in res.columns
    assert "B" in res.columns
    assert "C" not in res.columns
    assert "D" not in res.columns

def test_univariate_percentile_classification(classification_data):
    calc = FeatureSelectionCalculator()
    applier = FeatureSelectionApplier()
    
    # Select top 50% features. A is good, B is noise, C is constant.
    # 3 features total. 50% is 1.5 -> 1 feature? or 2?
    config = {
        "method": "select_percentile",
        "percentile": 40, # Should pick just A (1/3 = 33%)
        "score_func": "f_classif",
        "target_column": "target",
        "problem_type": "classification"
    }
    params = calc.fit(classification_data, config)
    res = applier.apply(classification_data, params)
    
    assert "A" in res.columns
    assert "B" not in res.columns

def test_univariate_fpr(regression_data):
    calc = FeatureSelectionCalculator()
    applier = FeatureSelectionApplier()
    
    config = {
        "method": "select_fpr",
        "alpha": 0.05,
        "score_func": "f_regression",
        "target_column": "target"
    }
    params = calc.fit(regression_data, config)
    res = applier.apply(regression_data, params)
    
    # A and B should be significant
    assert "A" in res.columns
    assert "B" in res.columns
    # C (random) should likely be dropped
    assert "C" not in res.columns

def test_generic_univariate(regression_data):
    calc = FeatureSelectionCalculator()
    applier = FeatureSelectionApplier()
    
    config = {
        "method": "generic_univariate_select",
        "mode": "k_best",
        "param": 1, # Select top 1
        "score_func": "f_regression",
        "target_column": "target"
    }
    params = calc.fit(regression_data, config)
    res = applier.apply(regression_data, params)
    
    # Should pick A (highest correlation)
    assert "A" in res.columns
    assert "B" not in res.columns

def test_select_from_model_rf(regression_data):
    calc = FeatureSelectionCalculator()
    applier = FeatureSelectionApplier()
    
    config = {
        "method": "select_from_model",
        "estimator": "RandomForest",
        "threshold": "median",
        "target_column": "target",
        "problem_type": "regression"
    }
    params = calc.fit(regression_data, config)
    res = applier.apply(regression_data, params)
    
    # RF should find A and B important
    assert "A" in res.columns
    assert "B" in res.columns
    assert "C" not in res.columns

def test_rfe_linear(regression_data):
    calc = FeatureSelectionCalculator()
    applier = FeatureSelectionApplier()
    
    config = {
        "method": "rfe",
        "estimator": "LinearRegression",
        "k": 2, # n_features_to_select
        "step": 1,
        "target_column": "target",
        "problem_type": "regression"
    }
    params = calc.fit(regression_data, config)
    res = applier.apply(regression_data, params)
    
    assert "A" in res.columns
    assert "B" in res.columns
    assert "C" not in res.columns
