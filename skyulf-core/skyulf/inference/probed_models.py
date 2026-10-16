"""Bounded exact model probes sealed against semantic fitted estimator state."""

import pickle
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..modeling._evaluation.thresholds import apply_thresholds
from ..modeling.base import StatefulEstimator
from ..pipeline.seal import artifact_digest
from ._partition_models import CONTRACTS, class_name, model_contract
from ._probe_frames import Frame, ProbeFailure, assert_same, validate_frame
from ._probe_strategies import run_check
from .fitted_pipeline import FittedPipelineArtifact, predict_pipeline
from .pipeline_scoring import prediction_output_schema
from .probed_steps import CHUNK_SIZES, MIN_ROWS, PROBE_KEY, REQUIRED_CHECKS


def has_probed_model(artifact: FittedPipelineArtifact) -> bool:
    """Identify eligible nodes without treating a modified registration as safe."""
    node = _configured_node(artifact)
    return node in {contract[0] for contract in CONTRACTS.values()}


def _configured_node(artifact: FittedPipelineArtifact) -> str | None:
    """Resolve the built-in tuner's base node without changing its metadata contract."""
    config = artifact.pipeline.config.get("modeling", {})
    if config.get("type") == "hyperparameter_tuner":
        config = config.get("base_model", {})
    return config.get("type")


def _identity(artifact: FittedPipelineArtifact) -> dict[str, Any]:
    """Bind model/applier identity, nested classes, tuning policy and fitted state."""
    from .partition_safety import _model_parts  # noqa: PLC0415

    estimator = _estimator(artifact)
    model, applier, _ = _model_parts(estimator)
    node, members = model_contract(model, applier)
    if node != _configured_node(artifact):
        raise ValueError("Configured model disagrees with fitted model class.")
    return {
        "node_type": node,
        "class": class_name(model),
        "applier": class_name(applier),
        "member_classes": members,
        "state_sha256": artifact_digest(
            (
                estimator.model,
                artifact.pipeline._tuned_thresholds,
                artifact.manifest.use_tuned_thresholds,
            )
        ).hex(),
        "probabilities": artifact.manifest.classification_probabilities,
    }


def _estimator(artifact: FittedPipelineArtifact) -> StatefulEstimator:
    """Require the same exact estimator facade during training and worker admission."""
    estimator = artifact.pipeline.model_estimator
    if type(estimator) is not StatefulEstimator:
        raise ValueError("Model probe requires an exact StatefulEstimator.")
    return estimator


def _output(artifact: FittedPipelineArtifact, frame: pd.DataFrame) -> pd.DataFrame:
    """Exercise the same Core applier and threshold paths used by saved-model serving."""
    pipeline = artifact.pipeline
    estimator = _estimator(artifact)
    result = pd.DataFrame(index=frame.index)
    result["prediction"] = np.asarray(estimator.applier.predict(frame, estimator.model))
    if (
        artifact.manifest.task == "classification"
        and artifact.manifest.classification_probabilities
    ):
        probabilities = np.asarray(pipeline._predict_proba_transformed(frame))
        for index in range(probabilities.shape[1]):
            result[f"probability_{index}"] = probabilities[:, index]
        if artifact.manifest.use_tuned_thresholds:
            if pipeline._tuned_thresholds is None:
                raise ValueError("Missing fitted thresholds.")
            result["threshold_prediction"] = apply_thresholds(
                probabilities,
                pipeline._tuned_thresholds,
                classes=estimator._unwrap_tuned_model().classes_,
                positive_class=pipeline._decision_positive_class(),
            )
    return result


def _apply(artifact: FittedPipelineArtifact, frame: Frame) -> pd.DataFrame:
    """Reject input and fitted-state mutation while probing detached estimator state."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Model probes require pandas model inputs.")
    before = _identity(artifact)
    copied = frame.copy(deep=True)
    result = _output(artifact, copied)
    assert_same(frame, copied)
    if before != _identity(artifact):
        raise ProbeFailure("state_mutation")
    validate_frame(result, 256, 8 * 1024 * 1024)
    return result


def probe_model(artifact: FittedPipelineArtifact, holdout: pd.DataFrame | pl.DataFrame) -> dict:
    """Probe transformed rows and replay persisted state; empty uses public serving schema."""
    identity = _identity(artifact)
    raw = holdout.head(256)
    raw = raw.to_pandas() if isinstance(raw, pl.DataFrame) else raw
    raw = raw.loc[:, list(artifact.manifest.input_columns)]
    transformed = artifact.pipeline.feature_engineer.transform(raw, preserve_rows=True)
    if not MIN_ROWS <= len(transformed) <= 256:
        raise ValueError("Model partition probe requires 16..256 transformed holdout rows.")
    validate_frame(transformed, 256, 8 * 1024 * 1024)
    # Serialization detaches fitted third-party state; reloading project modules cannot do this.
    payload = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    detached = pickle.loads(payload)
    checks = []
    expected = None
    for name in REQUIRED_CHECKS:
        current = pickle.loads(payload) if name == "fresh_module" else detached
        if name == "empty":
            check, output = _empty_check(current, raw)
        else:
            check, output = run_check(name, partial(_apply, current), transformed, expected)
        if name == "full":
            expected = output
        checks.append(check)
        _require_check(check, name)
    return {**identity, "rows": len(transformed), "checks": checks}


def _empty_check(artifact: FittedPipelineArtifact, raw: pd.DataFrame) -> tuple[dict, Frame | None]:
    """Exercise real empty serving against declared dtypes, independent of estimator quirks."""
    index = raw.head(0).index
    expected = pd.DataFrame(
        {
            spec.name: pd.Series(index=index, dtype=spec.dtype)
            for spec in prediction_output_schema(artifact)
        },
        index=index,
    )
    return run_check("empty", lambda frame: predict_pipeline(frame, artifact), raw, expected)


def _require_check(check: Any, name: str) -> None:
    """Require every strategy, including the real serving empty boundary, to pass."""
    if type(check) is not dict or check.get("name") != name:
        raise ValueError("Model partition probe check ordering is invalid.")
    if check == {"name": name, "status": "passed"}:
        return
    raise ValueError(
        f"Model partition probe failed {name}: {check.get('reason', 'missing evidence')}. "
        "Exact partition parity is required; retrain with a compatible configuration."
    )


def require_probed_model(artifact: FittedPipelineArtifact) -> dict:
    """Validate sealed model evidence without calling any prediction method."""
    evidence = _require_header(artifact.pipeline.config.get(PROBE_KEY))
    model = evidence.get("model")
    identity = _identity(artifact)
    if type(model) is not dict or set(model) != {*identity, "rows", "checks"}:
        raise ValueError("Malformed model partition probe evidence.")
    if any(model[key] != value for key, value in identity.items()):
        raise ValueError("Model partition probe evidence disagrees with fitted state or class.")
    _require_checks(model)
    if evidence.get("rows") != model["rows"]:
        raise ValueError("Model sample count disagrees with the enclosing probe evidence.")
    return model


def _require_header(evidence: Any) -> dict:
    """Do not accept booleans or floats as version, row-count or chunk-size integers."""
    if (
        type(evidence) is not dict
        or type(evidence.get("version")) is not int
        or evidence.get("version") != 1
        or type(evidence.get("rows")) is not int
        or artifact_digest(evidence.get("chunk_sizes")) != artifact_digest(list(CHUNK_SIZES))
    ):
        raise ValueError("Model requires sealed training partition probe evidence.")
    return evidence


def _require_checks(model: dict) -> None:
    """Reject incomplete or malformed check lists and out-of-bound sample counts."""
    if type(model["rows"]) is not int or not MIN_ROWS <= model["rows"] <= 256:
        raise ValueError("Model partition probe needs 16..256 holdout rows.")
    checks = model["checks"]
    if type(checks) is not list or len(checks) != len(REQUIRED_CHECKS):
        raise ValueError("Incomplete model partition probe checks.")
    for check, name in zip(checks, REQUIRED_CHECKS, strict=True):
        _require_check(check, name)
