"""Opt-in Unity Catalog feature lookup helpers with lazy SDK dependencies.

These adapters do not replace the existing Skyulf model lifecycle or certify
partition-safe inference. Native Databricks validation remains separate.
"""

from .config import FeatureLookupSpec, FeatureTrainingSpec
from .runtime import create_feature_training_set, log_feature_model, score_feature_model

__all__ = [
    "FeatureLookupSpec",
    "FeatureTrainingSpec",
    "create_feature_training_set",
    "log_feature_model",
    "score_feature_model",
]
