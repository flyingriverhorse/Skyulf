import pytest
import polars as pl
import pandas as pd
from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import LogisticRegressionCalculator, LogisticRegressionApplier
from skyulf.modeling.cross_validation import perform_cross_validation

def test_cross_validation_polars():
    # Create Polars Data
    df = pl.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "f2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    })
    
    X = df.drop("target")
    y = df["target"]
    
    calculator = LogisticRegressionCalculator()
    applier = LogisticRegressionApplier()
    
    # Run CV
    results = perform_cross_validation(
        calculator=calculator,
        applier=applier,
        X=X,
        y=y,
        config={},
        n_folds=2,
        cv_type="k_fold"
    )
    
    assert "aggregated_metrics" in results
    assert "accuracy" in results["aggregated_metrics"]
    assert results["aggregated_metrics"]["accuracy"]["mean"] > 0.0
