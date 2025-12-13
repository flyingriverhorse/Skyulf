"""Pytest fixtures for SDK tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_data():
    """Create a simple classification dataset."""
    np.random.seed(42)
    n_samples = 100
    data = pd.DataFrame({
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.normal(2, 1, n_samples),
        "category": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.choice([0, 1], n_samples)
    })
    # Introduce some missing values
    data.loc[0:5, "feature1"] = np.nan
    return data


@pytest.fixture
def sample_regression_data():
    """Create a simple regression dataset."""
    np.random.seed(42)
    n_samples = 100
    data = pd.DataFrame({
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.normal(2, 1, n_samples),
        "category": np.random.choice(["X", "Y"], n_samples),
        "target": np.random.normal(10, 2, n_samples)
    })
    return data
