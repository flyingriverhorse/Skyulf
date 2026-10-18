"""Scoring rules for the online fare model: none.

This example deliberately saves no scoring rules: config/pre_split.yml selects
training rows only, and the caller validates request values. Spark and endpoints
can also replay saved scoring rules and reused pre-split filters when training
has linted and probed them. See the taxi_fare_robust example for that alternative.
"""


def build_model_rules():
    """Save no scoring rules with the model."""
    return None


def build_combined_rules():
    """A single model has no cross-model results."""
    return []
