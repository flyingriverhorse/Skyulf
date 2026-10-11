"""Opt-in Unity Catalog feature lookup helpers with lazy SDK dependencies.

These adapters do not replace the existing Skyulf model lifecycle or certify
partition-safe inference. Native Databricks validation remains separate.
"""

from .config import FeatureLookupSpec, FeatureTrainingSpec
from .online_policy import OnlineFeaturePolicy, validate_online_features
from .online_publication import (
    OnlinePublicationSpec,
    online_publication_status,
    publication_receipt,
    publish_online_features,
    restart_online_publication,
    stop_online_publication,
)
from .runtime import create_feature_training_set, log_feature_model, score_feature_model

__all__ = [
    "FeatureLookupSpec",
    "FeatureTrainingSpec",
    "OnlineFeaturePolicy",
    "OnlinePublicationSpec",
    "create_feature_training_set",
    "log_feature_model",
    "online_publication_status",
    "publication_receipt",
    "publish_online_features",
    "restart_online_publication",
    "score_feature_model",
    "stop_online_publication",
    "validate_online_features",
]
