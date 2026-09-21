"""Standalone inference bundles; importing this package needs no Spark or MLflow."""

from .bundle import InferenceBundle, build_bundle, load_bundle, predict_local, save_bundle
from .spark import predict_spark

__all__ = [
    "InferenceBundle",
    "build_bundle",
    "load_bundle",
    "predict_local",
    "predict_spark",
    "save_bundle",
]
