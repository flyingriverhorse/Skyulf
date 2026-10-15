"""Scoring rules for the online fare model: none.

Serving endpoints (like Spark scoring) reject any scoring rules saved with the
model, including a reused pre-split recipe. The filters in config/pre_split.yml
therefore select training rows only; validate request values in the caller.
"""


def build_model_rules():
    """Save no scoring rules with the model."""
    return None


build_scoring = build_model_rules


def build_combined_rules():
    """A single model has no cross-model results."""
    return []
