"""Bounded model probes bound to fitted state, with exact labels and fixed float tolerance."""

import pickle
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..modeling._evaluation.thresholds import apply_thresholds
from ..modeling.base import StatefulEstimator
from ..pipeline.seal import artifact_digest
from ._manifest import ColumnSpec
from ._partition_models import CONTRACTS, class_name, model_contract
from ._probe_frames import Frame, ProbeFailure, assert_same, copy_frame, validate_frame
from ._probe_strategies import run_check
from .fitted_pipeline import FittedPipelineArtifact, has_exclusion_steps, predict_pipeline
from .pipeline_scoring import prediction_output_schema, score_pipeline
from .probed_steps import (
    CHUNK_SIZES,
    MIN_ROWS,
    PIPELINE_CHECKS,
    PROBE_KEY,
    REQUIRED_CHECKS,
    require_comparison_policy,
    require_pipeline_probe,
    require_scored_rows,
    require_step_rows,
)
from .project_scoring import _typed_column


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
    assert_same(frame, copied, "input_mutation", exact=True)
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
    engineer = artifact.pipeline.feature_engineer
    transformed = (
        engineer.transform_with_exclusions(raw)[0]
        if has_exclusion_steps(artifact)
        else engineer.transform(raw, preserve_rows=True)
    )
    if not MIN_ROWS <= len(transformed) <= 256:
        raise ValueError("Model partition probe requires 16..256 transformed holdout rows.")
    validate_frame(transformed, 256, 8 * 1024 * 1024)
    # Serialization detaches fitted third-party state; reloading project modules cannot do this.
    payload = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    detached = pickle.loads(payload)
    checks = []
    expected = None
    labels = (
        ("prediction", "threshold_prediction") if artifact.manifest.task == "classification" else ()
    )
    for name in REQUIRED_CHECKS:
        current = pickle.loads(payload) if name == "fresh_module" else detached
        if name == "empty":
            check, output = _empty_check(current, raw)
        else:
            check, output = run_check(
                name, partial(_apply, current), transformed, expected, exact_columns=labels
            )
        if name == "full":
            expected = output
        checks.append(check)
        _require_check(check, name)
    return {**identity, "input_rows": len(raw), "rows": len(transformed), "checks": checks}


def _pipeline_state(artifact: FittedPipelineArtifact) -> bytes:
    """Bind the retained semantic state of the complete saved serving path."""
    return artifact_digest(
        (
            artifact.pipeline.config,
            artifact.pipeline.feature_engineer.fitted_steps,
            _estimator(artifact).model,
            artifact.pipeline._tuned_thresholds,
            artifact.manifest.model_dump(),
        )
    )


def _apply_pipeline(artifact: FittedPipelineArtifact, frame: Frame) -> pd.DataFrame:
    """Run real serving on detached inputs and reject exact input or state mutations."""
    argument = copy_frame(frame)
    before = _pipeline_state(artifact)
    try:
        return score_pipeline(argument, artifact)
    finally:
        assert_same(frame, argument, "input_mutation", exact=True)
        if before != _pipeline_state(artifact):
            raise ProbeFailure("state_mutation")


def _pipeline_scored_rows(output: Frame | None, maximum: int) -> int:
    """Count real predicted outcomes rather than successful exclusion-only responses."""
    if not isinstance(output, pd.DataFrame):
        raise ValueError("Raw pipeline partition probe did not return a prediction frame.")
    count = (
        int(output["scoring_status"].eq("predicted").sum())
        if "scoring_status" in output
        else len(output)
    )
    return require_scored_rows(count, maximum)


def probe_pipeline(artifact: FittedPipelineArtifact, holdout: pd.DataFrame | pl.DataFrame) -> dict:
    """Test raw serving end to end so tolerated feature noise cannot hide changed labels."""
    raw = holdout.head(256)
    raw = raw.to_pandas() if isinstance(raw, pl.DataFrame) else raw
    raw = raw.loc[:, list(artifact.manifest.input_columns)]
    require_step_rows(len(raw), 256, "Pipeline partition probe")
    detached = pickle.loads(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    apply = partial(_apply_pipeline, detached)
    labels = (
        ("prediction", "threshold_prediction") if artifact.manifest.task == "classification" else ()
    )
    expected = None
    checks = []
    scored_rows = 0
    for name in PIPELINE_CHECKS:
        check, output = run_check(name, apply, raw, expected, exact_columns=labels)
        _require_check(check, name, "Pipeline")
        if name == "full":
            expected = output
            scored_rows = _pipeline_scored_rows(output, len(raw))
        checks.append(check)
    return {"rows": len(raw), "scored_rows": scored_rows, "checks": checks}


def _empty_check(artifact: FittedPipelineArtifact, raw: pd.DataFrame) -> tuple[dict, Frame | None]:
    """Exercise real empty serving against declared dtypes, independent of estimator quirks."""
    index = raw.head(0).index
    outliers = has_exclusion_steps(artifact)
    schema = prediction_output_schema(artifact)
    if outliers:
        schema += (
            ColumnSpec(name="scoring_status", dtype="string"),
            ColumnSpec(name="exclusion_reason", dtype="string"),
        )
    expected = pd.DataFrame(
        {
            spec.name: (
                _typed_column(pd.Series(index=index), spec.dtype)
                if outliers
                else pd.Series(index=index, dtype=spec.dtype)
            )
            for spec in schema
        },
        index=index,
    )
    return run_check("empty", lambda frame: predict_pipeline(frame, artifact), raw, expected)


def _require_check(check: Any, name: str, label: str = "Model") -> None:
    """Require every strategy, including the real serving empty boundary, to pass."""
    if type(check) is not dict or check.get("name") != name:
        raise ValueError("Model partition probe check ordering is invalid.")
    if check == {"name": name, "status": "passed"}:
        return
    raise ValueError(
        f"{label} partition probe failed {name}: {check.get('reason', 'missing evidence')}. "
        "Partition parity requires exact labels and the fixed floating tolerance; "
        "retrain with a compatible configuration."
    )


def require_probed_model(artifact: FittedPipelineArtifact) -> dict:
    """Validate sealed model evidence without calling any prediction method."""
    evidence = _require_header(artifact.pipeline.config.get(PROBE_KEY))
    model = evidence.get("model")
    identity = _identity(artifact)
    if type(model) is not dict or set(model) != {*identity, "input_rows", "rows", "checks"}:
        raise ValueError("Malformed model partition probe evidence.")
    if any(model[key] != value for key, value in identity.items()):
        raise ValueError("Model partition probe evidence disagrees with fitted state or class.")
    _require_checks(model)
    if evidence.get("rows") != model["input_rows"]:
        raise ValueError("Model sample count disagrees with the enclosing probe evidence.")
    if not has_exclusion_steps(artifact) and model["input_rows"] != model["rows"]:
        raise ValueError("Model sample count changed without fitted row exclusions.")
    return model


def _require_header(evidence: Any) -> dict:
    """Do not accept booleans or floats as version, row-count or chunk-size integers."""
    if (
        type(evidence) is not dict
        or type(evidence.get("rows")) is not int
        or artifact_digest(evidence.get("chunk_sizes")) != artifact_digest(list(CHUNK_SIZES))
    ):
        raise ValueError("Model requires sealed training partition probe evidence.")
    require_comparison_policy(evidence)
    require_pipeline_probe(evidence)
    return evidence


def _require_checks(model: dict) -> None:
    """Reject incomplete or malformed check lists and out-of-bound sample counts."""
    if type(model["rows"]) is not int or not MIN_ROWS <= model["rows"] <= 256:
        raise ValueError("Model partition probe needs 16..256 holdout rows.")
    if type(model["input_rows"]) is not int or not model["rows"] <= model["input_rows"] <= 256:
        raise ValueError("Model probe input rows must bound its surviving rows within 256.")
    checks = model["checks"]
    if type(checks) is not list or len(checks) != len(REQUIRED_CHECKS):
        raise ValueError("Incomplete model partition probe checks.")
    for check, name in zip(checks, REQUIRED_CHECKS, strict=True):
        _require_check(check, name)
