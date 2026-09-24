"""Core model metadata must be discoverable without backend setup."""

from skyulf.modeling.classification import (
    LogisticRegressionCalculator,
    RandomForestClassifierCalculator,
)
from skyulf.registry import NodeRegistry


def test_registry():
    """Missing dynamic models or parameters must fail instead of only printing a warning."""
    assert hasattr(LogisticRegressionCalculator, "__node_meta__")
    assert hasattr(RandomForestClassifierCalculator, "__node_meta__")
    metadata = NodeRegistry.get_all_metadata()
    assert "logistic_regression" in metadata
    assert "random_forest_classifier" in metadata
    assert "max_iter" in (metadata["logistic_regression"].get("params") or {})
