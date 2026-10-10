"""Fitted pipelines and portable inference bundles; no Spark or MLflow import required."""

from .bundle import InferenceBundle, build_bundle, load_bundle, predict_local, save_bundle
from .fitted_pipeline import (
    FittedPipelineArtifact,
    FittedPipelineManifest,
    load_pipeline,
    predict_pipeline,
    save_pipeline,
)
from .pipeline_evaluation import evaluate_holdout
from .pipeline_scoring import PipelinePrediction, score_pipeline, score_pipeline_with_history
from .spark import predict_spark

__all__ = [
    "FittedPipelineArtifact",
    "FittedPipelineManifest",
    "InferenceBundle",
    "PipelinePrediction",
    "build_bundle",
    "evaluate_holdout",
    "load_bundle",
    "load_pipeline",
    "predict_local",
    "predict_pipeline",
    "predict_spark",
    "save_bundle",
    "save_pipeline",
    "score_pipeline",
    "score_pipeline_with_history",
]
