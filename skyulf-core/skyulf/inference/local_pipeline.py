"""Versioned artifacts for fitted pandas/Polars pipelines.

The payload is pickle: only load artifacts from a trusted producer. The
checksum detects damaged bytes but does not authenticate their origin.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import is_classifier

from ..core.portable_state import _bad_constant, _unique_object
from ..core.schema import SkyulfSchema
from ..pipeline import SkyulfPipeline
from ._manifest import checksum, runtime_requirements
from .project_code import MAX_PROJECT_SOURCE_BYTES, load_project_module, project_source_digest

_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_PIPELINE_BYTES = 256 * 1024 * 1024


class LocalPipelineManifest(BaseModel):
    """Record the fit engine, raw schema, runtime and serialized pipeline identity."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    format_version: Literal[1] = 1
    fitted_engine: Literal["pandas", "polars"]
    input_columns: tuple[str, ...] = Field(min_length=1)
    input_dtypes: tuple[str, ...]
    feature_columns: tuple[str, ...] = Field(min_length=1)
    feature_dtypes: tuple[str, ...]
    requirements: tuple[tuple[str, str], ...]
    pipeline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_class: str = Field(min_length=1)
    execution_scope: Literal["whole_frame_local"] = "whole_frame_local"
    task: Literal["regression", "classification"]
    classes: tuple[str | int | float | bool, ...] = ()
    use_tuned_thresholds: bool = False
    project_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LocalPipelineArtifact:
    """A validated manifest and fitted local pipeline loaded from one directory."""

    manifest: LocalPipelineManifest
    pipeline: SkyulfPipeline


def _recorded_schemas(pipeline: SkyulfPipeline) -> tuple[SkyulfSchema, SkyulfSchema]:
    """Require the raw and model feature schemas from a successful model fit."""
    schemas = getattr(pipeline, "_inference_schemas", None)
    if schemas is None or pipeline.fitted_engine not in {"pandas", "polars"}:
        raise ValueError("Missing fitted local engine or schemas; refit this pipeline.")
    if pipeline.model_estimator is None or pipeline.model_estimator.model is None:
        raise ValueError("Local artifact requires a fitted model.")
    return schemas


def _manifest(
    pipeline: SkyulfPipeline, payload: bytes, use_tuned_thresholds: bool
) -> LocalPipelineManifest:
    """Derive metadata from the successful fit rather than caller-supplied labels."""
    raw, features = _recorded_schemas(pipeline)
    estimator = pipeline.model_estimator
    if estimator is None:
        raise ValueError("Local artifact requires a fitted model.")
    model = estimator._unwrap_tuned_model()
    fitted_engine = pipeline.fitted_engine
    if fitted_engine not in ("pandas", "polars"):
        raise ValueError("Local artifact requires a recorded pandas or Polars fit engine.")
    classification = is_classifier(model)
    classes = tuple(np.asarray(model.classes_).tolist()) if classification else ()
    if classification and not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Local classification artifact requires predict_proba().")
    if use_tuned_thresholds and (not classification or pipeline._tuned_thresholds is None):
        raise ValueError("Tuned thresholds require fitted classification thresholds.")
    return LocalPipelineManifest(
        fitted_engine=fitted_engine,
        input_columns=raw.columns,
        input_dtypes=tuple(raw.dtypes.get(name, "unknown") for name in raw.columns),
        feature_columns=features.columns,
        feature_dtypes=tuple(features.dtypes.get(name, "unknown") for name in features.columns),
        requirements=runtime_requirements(),
        pipeline_sha256=checksum(payload),
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        task="classification" if classification else "regression",
        classes=classes,
        use_tuned_thresholds=use_tuned_thresholds,
        project_source_sha256=(
            project_source_digest(pipeline.config["project_python_source"])
            if "project_python_source" in pipeline.config
            else None
        ),
    )


def _check_runtime(manifest: LocalPipelineManifest) -> None:
    """Reject incompatible Python or dependency versions before loading pickle."""
    current = dict(runtime_requirements())
    required = dict(manifest.requirements)
    if set(required) != set(current):
        raise ValueError("Local artifact runtime requirements are incomplete.")
    for name in ("python", "skyulf-core", "scikit-learn", "numpy", "scipy", "pandas", "polars"):
        actual, expected = current[name], required[name]
        matches = (
            actual.split(".")[:2] == expected.split(".")[:2]
            if name == "python"
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"Local artifact runtime mismatch for {name}: requires {expected}, found {actual}."
            )


def save_local_pipeline(
    pipeline: SkyulfPipeline, path: str | Path, *, use_tuned_thresholds: bool = False
) -> None:
    """Write a fitted pipeline and its manifest to a new directory."""
    if type(pipeline) is not SkyulfPipeline:
        raise TypeError("Expected a fitted SkyulfPipeline.")
    _recorded_schemas(pipeline)
    payload = pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_PIPELINE_BYTES:
        raise ValueError("Local pipeline payload exceeds the size limit.")
    manifest = _manifest(pipeline, payload, use_tuned_thresholds)
    metadata = manifest.model_dump_json().encode("utf-8")
    if len(metadata) > _MAX_MANIFEST_BYTES:
        raise ValueError("Local pipeline manifest exceeds the size limit.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "manifest.json").write_bytes(metadata)
    (destination / "pipeline.pkl").write_bytes(payload)
    if manifest.project_source_sha256 is not None:
        (destination / "preprocessing.py").write_text(
            pipeline.config["project_python_source"], encoding="utf-8", newline="\n"
        )


def _read_bounded(path: Path, limit: int) -> bytes:
    """Limit artifact bytes before JSON or pickle decoding."""
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Local artifact {path.name} exceeds the size limit.")
    return data


def load_local_pipeline(path: str | Path) -> LocalPipelineArtifact:
    """Validate and load a trusted producer's local pipeline artifact."""
    source = Path(path)
    metadata = _read_bounded(source / "manifest.json", _MAX_MANIFEST_BYTES)
    document = json.loads(
        metadata.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_bad_constant
    )
    if type(document) is not dict or type(document.get("format_version")) is not int:
        raise ValueError("Invalid local artifact format_version.")
    manifest = LocalPipelineManifest.model_validate_json(metadata)
    if len(manifest.input_columns) != len(manifest.input_dtypes) or len(
        manifest.feature_columns
    ) != len(manifest.feature_dtypes):
        raise ValueError("Local artifact schema columns and dtypes disagree.")
    if any(
        len(set(columns)) != len(columns)
        for columns in (manifest.input_columns, manifest.feature_columns)
    ):
        raise ValueError("Local artifact schema columns must be unique.")
    _check_runtime(manifest)
    payload = _read_bounded(source / "pipeline.pkl", _MAX_PIPELINE_BYTES)
    if checksum(payload) != manifest.pipeline_sha256:
        raise ValueError("Local pipeline payload checksum mismatch.")
    if manifest.project_source_sha256 is not None:
        code = _read_bounded(source / "preprocessing.py", MAX_PROJECT_SOURCE_BYTES).decode("utf-8")
        if project_source_digest(code) != manifest.project_source_sha256:
            raise ValueError("Project preprocessing source checksum mismatch.")
        load_project_module(code)
    pipeline = pickle.loads(payload)  # nosec B301 -- trusted producer only, after size/runtime/checksum checks
    if type(pipeline) is not SkyulfPipeline:
        raise ValueError("Local artifact payload is not a SkyulfPipeline.")
    if _manifest(pipeline, payload, manifest.use_tuned_thresholds) != manifest:
        raise ValueError("Local artifact manifest disagrees with its fitted pipeline.")
    return LocalPipelineArtifact(manifest, pipeline)


def require_local_pipeline_scope(
    artifact: LocalPipelineArtifact,
    scope: Literal["whole_frame_local", "row_local_http", "spark_worker", "spark_native"],
) -> None:
    """Reject execution modes that the local artifact has not been proven to support."""
    if not isinstance(artifact, LocalPipelineArtifact):
        raise TypeError("Expected a LocalPipelineArtifact.")
    if scope != artifact.manifest.execution_scope:
        raise ValueError(f"Local pipeline artifact is not eligible for {scope}.")


def predict_local_pipeline(
    frame: pd.DataFrame | pl.DataFrame, artifact: LocalPipelineArtifact
) -> pd.DataFrame:
    """Apply the fitted local FE and model with the recorded fit engine."""
    if not isinstance(artifact, LocalPipelineArtifact):
        raise TypeError("Expected a LocalPipelineArtifact.")
    if not isinstance(frame, pd.DataFrame | pl.DataFrame):
        raise TypeError("Local prediction requires a pandas or Polars DataFrame.")
    if artifact.manifest.fitted_engine == "polars":
        native = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    else:
        native = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame
    expected = SkyulfSchema(
        artifact.manifest.input_columns,
        dict(zip(artifact.manifest.input_columns, artifact.manifest.input_dtypes, strict=True)),
    )
    expected.assert_compatible(
        SkyulfSchema.from_dataframe(native),
        check_dtypes=True,
        check_order=True,
        where="local pipeline input",
    )
    prediction = np.asarray(
        artifact.pipeline.predict(
            native, use_tuned_thresholds=artifact.manifest.use_tuned_thresholds
        )
    )
    if prediction.shape != (len(frame),):
        raise ValueError("Local pipeline prediction shape disagrees with input rows.")
    index = frame.index if isinstance(frame, pd.DataFrame) else pd.RangeIndex(len(frame))
    result = pd.DataFrame({"prediction": pd.Series(prediction, index=index)})
    if artifact.manifest.task == "classification":
        transformed = artifact.pipeline.feature_engineer.transform(native, preserve_rows=True)
        probabilities = np.asarray(artifact.pipeline._predict_proba_transformed(transformed))
        if probabilities.shape != (len(frame), len(artifact.manifest.classes)):
            raise ValueError("Local pipeline probability shape disagrees with recorded classes.")
        for position in range(len(artifact.manifest.classes)):
            result[f"probability_{position}"] = probabilities[:, position]
    return result
