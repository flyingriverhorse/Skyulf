import pytest
import numpy as np
import pandas as pd
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from skyulf.preprocessing.feature_selection import (
    FeatureSelectionApplier,
    FeatureSelectionCalculator,
)

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_variance_threshold_polars():
    calc = FeatureSelectionCalculator()
    applier = FeatureSelectionApplier()

    # Create Polars DataFrame
    df = pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [1, 1, 1, 1, 1], # Constant
        "target": [10, 20, 30, 40, 50]
    })

    # Threshold 0 drops constant
    config = {"method": "variance_threshold", "threshold": 0.0}
    params = calc.fit(df, config)
    res = applier.apply(df, params)

    assert "B" not in res.columns
    assert "A" in res.columns
    assert "target" in res.columns
    assert isinstance(res, pl.DataFrame)

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_univariate_k_best_regression_polars():
    calc = FeatureSelectionCalculator()
    applier = FeatureSelectionApplier()

    # Create Polars DataFrame
    # A: Strong linear relationship
    # B: Random noise
    np.random.seed(42)
    n = 100
    A = np.linspace(0, 10, n)
    B = np.random.uniform(0, 10, n)
    target = 2 * A + np.random.normal(0, 0.1, n)

    df = pl.DataFrame({"A": A, "B": B, "target": target})

    # Select top 1 feature (A should be best)
    config = {
        "method": "select_k_best",
        "k": 1,
        "score_func": "f_regression",
        "target_column": "target",
        "problem_type": "regression",
    }
    params = calc.fit(df, config)
    res = applier.apply(df, params)

    assert "A" in res.columns
    assert "B" not in res.columns
    assert isinstance(res, pl.DataFrame)
