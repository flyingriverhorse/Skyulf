"""Capture and restore only the supported standalone sklearn prediction contract."""

import io
import pickle  # nosec B403 - explicit trusted model persistence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.utils.validation import check_is_fitted

from ..modeling._tuning.engine import TuningApplier
from ..modeling.sklearn_wrapper import SklearnApplier
from ..pipeline import SkyulfPipeline
from ..pipeline.seal import artifact_digest
from ._manifest import BundleManifest, ThresholdProvenance


class _BoundedModelBuffer(io.BytesIO):
    """Stop serialization when its wire output exceeds the configured estimator budget."""

    def __init__(self, limit: int):
        """Store the caller's validated byte limit."""
        super().__init__()
        self.limit = limit

    def write(self, data: Any) -> int:
        """Reject a pickle chunk before growing the output beyond the allowed budget."""
        if self.tell() + memoryview(data).nbytes > self.limit:
            raise ValueError("Estimator payload exceeds model_max_bytes.")
        return super().write(data)


def serialize_model(model: Any, limit: int) -> bytes:
    """Freeze a fitted estimator independently of its mutable training pipeline."""
    output = _BoundedModelBuffer(limit)
    pickle.dump(model, output, protocol=5)
    return output.getvalue()


def validate_model(model: Any) -> None:
    """Fail closed on custom, composite, multi-output or unsupported model semantics."""
    if not isinstance(model, BaseEstimator) or not type(model).__module__.startswith("sklearn."):
        raise ValueError("Only standalone sklearn estimators are supported by this bundle version.")
    if not (is_regressor(model) or is_classifier(model)):
        raise ValueError("Only regression and classification estimators are supported.")
    check_is_fitted(model)
    if getattr(model, "n_outputs_", 1) != 1:
        raise ValueError("Multi-output estimators are not supported.")
    if is_classifier(model) and not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Classification bundles require predict_proba.")


def pipeline_model(pipeline: SkyulfPipeline) -> Any:
    """Reject custom appliers whose prediction behavior would be lost by extracting a model."""
    estimator = pipeline.model_estimator
    if estimator is None or estimator.model is None or pipeline._fit_metrics is None:
        raise ValueError("A successfully fitted standalone model pipeline is required.")
    applier = estimator.applier
    if type(applier) is TuningApplier:
        applier = applier.base_applier
    if (
        type(applier).predict is not SklearnApplier.predict
        or type(applier).predict_proba is not SklearnApplier.predict_proba
    ):
        raise ValueError("Custom model appliers require an explicit inference adapter.")
    model = estimator._unwrap_tuned_model()
    validate_model(model)
    return model


def threshold_provenance(
    pipeline: SkyulfPipeline, classes: tuple, enabled: bool
) -> ThresholdProvenance:
    """Preserve tuning defaults and the opt-in pipeline threshold override separately."""
    estimator = pipeline.model_estimator
    assert estimator is not None
    artifact = estimator.model
    tuning = artifact[1] if isinstance(artifact, tuple) else None
    tuned = getattr(tuning, "decision_thresholds", None)
    saved = pipeline._tuned_thresholds
    if enabled and saved is None:
        raise ValueError("use_tuned_thresholds requires saved pipeline thresholds.")

    def values(mapping: dict | None) -> tuple[float, ...]:
        """Align threshold labels explicitly to the model's probability column order."""
        if mapping is None:
            return ()
        if not classes or set(mapping) != set(classes):
            raise ValueError("Threshold keys must match model classes.")
        return tuple(float(mapping[label]) for label in classes)

    tuning_values, pipeline_values = values(tuned), values(saved)
    source = "pipeline_override" if enabled else "tuning" if tuned is not None else "estimator"
    active = pipeline_values if enabled else tuning_values
    return ThresholdProvenance(
        source=source,
        values=active,
        tuning_values=tuning_values,
        pipeline_values=pipeline_values,
        metric=getattr(tuning, "decision_threshold_metric", None),
    )


def load_model(payload: bytes, manifest: BundleManifest) -> Any:
    """Load a trusted, checksum-validated pickle and cross-check its fitted metadata."""
    model = pickle.loads(payload)  # nosec B301 - caller must trust the producer; checksum is not authentication
    validate_model(model)
    if f"{type(model).__module__}.{type(model).__qualname__}" != manifest.model_class:
        raise ValueError("Model class disagrees with manifest.")
    if artifact_digest(model).hex() != manifest.model_state_digest:
        raise ValueError("Model semantic digest mismatch.")
    if int(model.n_features_in_) != len(manifest.feature_order):
        raise ValueError("Model feature width disagrees with manifest.")
    classes = tuple(np.asarray(model.classes_).tolist()) if is_classifier(model) else ()
    if classes != manifest.classes or is_classifier(model) != (manifest.task == "classification"):
        raise ValueError("Model classes/task disagree with manifest.")
    return model
