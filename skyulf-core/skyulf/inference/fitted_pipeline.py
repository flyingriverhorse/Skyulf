"""Versioned artifacts for fitted pandas/Polars pipelines.

The payload is pickle: only load artifacts from a trusted producer. The
checksum detects damaged bytes but does not authenticate their origin.
"""

import json
import pickle
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from sklearn.base import is_classifier

from ..core.portable_state import _bad_constant, _unique_object
from ..core.schema import SkyulfSchema
from ..pipeline import SkyulfPipeline
from ..preprocessing._exclusion import is_exclusion_step
from ..preprocessing._target_labels import original_labels
from ._manifest import checksum, label_dtype, runtime_requirements
from ._model_requirements import native_model_requirements
from .project_code import MAX_PROJECT_SOURCE_BYTES, load_project_module, project_source_digest
from .project_dependencies import (
    parse_project_requirements,
    source_project_requirements,
    verify_project_requirements,
)
from .project_scoring import _typed_column, validate_scoring_config

_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_PIPELINE_BYTES = 256 * 1024 * 1024


class SourceColumnConversion(BaseModel):
    """Record one observed Spark Decimal input conversion without retaining values."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    column: str = Field(min_length=1)
    source_dtype: str = Field(
        pattern=r"^decimal\(([1-9]|[12][0-9]|3[0-8]),([0-9]|[12][0-9]|3[0-8])\)$"
    )
    target_dtype: Literal["float64"]

    @model_validator(mode="after")
    def validate_scale(self) -> "SourceColumnConversion":
        """Reject a Decimal scale larger than its declared precision."""
        precision, scale = map(int, self.source_dtype[8:-1].split(","))
        if scale > precision:
            raise ValueError("Decimal conversion scale cannot exceed its precision.")
        return self


class FittedPipelineManifest(BaseModel):
    """Record the fit engine, raw schema, runtime and serialized pipeline identity."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    format_version: Literal[1, 2] = 2
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
    classification_probabilities: bool = True
    use_tuned_thresholds: bool = False
    project_requirements: tuple[str, ...] = ()
    project_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_column_conversions: tuple[SourceColumnConversion, ...] | None = None

    @model_validator(mode="after")
    def validate_source_conversions(self) -> "FittedPipelineManifest":
        """Bind observed conversions to unique fitted float64 model inputs."""
        conversions = self.source_column_conversions or ()
        if not conversions:
            return self
        names = [item.column for item in conversions]
        if len(names) != len(set(names)):
            raise ValueError("Source conversions must identify unique model inputs.")
        dtypes = dict(zip(self.input_columns, self.input_dtypes, strict=True))
        if any(dtypes.get(name, "").lower() != "float64" for name in names):
            raise ValueError("Source conversions must identify fitted float64 model inputs.")
        return self

    @model_serializer(mode="wrap")
    def serialize_manifest(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Preserve legacy JSON and semantic proof hashes when evidence was not recorded."""
        document = handler(self)
        if self.source_column_conversions is None:
            document.pop("source_column_conversions", None)
        return document


@dataclass(frozen=True)
class FittedPipelineArtifact:
    """A validated manifest and fitted pandas/Polars pipeline loaded from one directory."""

    manifest: FittedPipelineManifest
    pipeline: SkyulfPipeline
    feature_lookup_json: str | None = None


def _recorded_schemas(pipeline: SkyulfPipeline) -> tuple[SkyulfSchema, SkyulfSchema]:
    """Require the raw and model feature schemas from a successful model fit."""
    schemas = getattr(pipeline, "_inference_schemas", None)
    if schemas is None or pipeline.fitted_engine not in {"pandas", "polars"}:
        raise ValueError("Missing fitted local engine or schemas; refit this pipeline.")
    if pipeline.model_estimator is None or pipeline.model_estimator.model is None:
        raise ValueError("Local artifact requires a fitted model.")
    return schemas


def _manifest(
    pipeline: SkyulfPipeline,
    payload: bytes,
    use_tuned_thresholds: bool,
    *,
    format_version: Literal[1, 2] = 2,
) -> FittedPipelineManifest:
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
    classes = tuple(original_labels(pipeline, model.classes_).tolist()) if classification else ()
    classification_probabilities = not classification or callable(
        getattr(model, "predict_proba", None)
    )
    _validate_tuned_thresholds(
        pipeline, use_tuned_thresholds, classification, classification_probabilities
    )
    return FittedPipelineManifest(
        format_version=format_version,
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
        classification_probabilities=classification_probabilities,
        use_tuned_thresholds=use_tuned_thresholds,
        project_requirements=_project_contract(pipeline),
        project_source_sha256=(
            project_source_digest(pipeline.config["project_python_source"])
            if "project_python_source" in pipeline.config
            else None
        ),
        source_column_conversions=_source_conversions(pipeline),
    )


def _source_conversions(pipeline: SkyulfPipeline) -> tuple[SourceColumnConversion, ...] | None:
    """Read conversion evidence from the pipeline covered by the payload checksum."""
    value = pipeline.config.get("source_column_conversions")
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise ValueError("Source column conversions must be an ordered sequence.")
    return tuple(SourceColumnConversion.model_validate(item) for item in value)


def _validate_tuned_thresholds(
    pipeline: SkyulfPipeline, enabled: bool, classification: bool, probabilities: bool
) -> None:
    """Require probability support and fitted thresholds before recording their use."""
    if enabled and not probabilities:
        raise ValueError("Tuned thresholds require classification probabilities.")
    if enabled and (not classification or pipeline._tuned_thresholds is None):
        raise ValueError("Tuned thresholds require fitted classification thresholds.")


class ArtifactRuntimeMismatchError(ValueError):
    """A saved artifact was produced by different pinned runtime versions."""


def _check_runtime(manifest: FittedPipelineManifest) -> None:
    """Reject incompatible Python or dependency versions before loading pickle."""
    verify_project_requirements(manifest.project_requirements)
    current = dict(runtime_requirements())
    required = dict(manifest.requirements)
    if tuple(name for name, _ in manifest.requirements) != tuple(current):
        raise ValueError("Local artifact runtime requirements are incomplete.")
    for name in ("python", "skyulf-core", "scikit-learn", "numpy", "scipy", "pandas", "polars"):
        actual, expected = current[name], required[name]
        matches = (
            actual.split(".")[:2] == expected.split(".")[:2]
            if name == "python"
            else actual == expected
        )
        if not matches:
            raise ArtifactRuntimeMismatchError(
                f"Local artifact runtime mismatch for {name}: requires {expected}, found {actual}."
            )


def save_pipeline(
    pipeline: SkyulfPipeline, path: str | Path, *, use_tuned_thresholds: bool = False
) -> None:
    """Write version 2 with the decision policy inside the checksummed pickle payload."""
    if type(pipeline) is not SkyulfPipeline:
        raise TypeError("Expected a fitted SkyulfPipeline.")
    _recorded_schemas(pipeline)
    prediction_pipeline = copy(pipeline)
    prediction_pipeline.feature_engineer = copy(pipeline.feature_engineer)
    prediction_pipeline.feature_engineer.train_sample_weight_ = None
    payload = pickle.dumps(
        {"pipeline": prediction_pipeline, "use_tuned_thresholds": use_tuned_thresholds},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
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


def read_bounded_artifact(path: Path, limit: int) -> bytes:
    """Limit artifact bytes before JSON or pickle decoding."""
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Local artifact {path.name} exceeds the size limit.")
    return data


def load_pipeline(path: str | Path) -> FittedPipelineArtifact:
    """Validate and load a trusted producer's fitted pipeline artifact.

    Version 1 remains readable only without fitted thresholds, because its
    payload cannot establish the saved decision policy. Re-export those older
    threshold-bearing pipelines with an explicit ``use_tuned_thresholds`` choice.
    Checksums establish integrity, not producer authenticity.
    """
    source = Path(path)
    metadata = read_bounded_artifact(source / "manifest.json", _MAX_MANIFEST_BYTES)
    document = json.loads(
        metadata.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_bad_constant
    )
    if type(document) is not dict or type(document.get("format_version")) is not int:
        raise ValueError("Invalid local artifact format_version.")
    manifest = FittedPipelineManifest.model_validate_json(metadata)
    _validate_manifest_schema(manifest)
    _check_runtime(manifest)
    payload = read_bounded_artifact(source / "pipeline.pkl", _MAX_PIPELINE_BYTES)
    if checksum(payload) != manifest.pipeline_sha256:
        raise ValueError("Local pipeline payload checksum mismatch.")
    if manifest.project_source_sha256 is not None:
        code = read_bounded_artifact(source / "preprocessing.py", MAX_PROJECT_SOURCE_BYTES).decode(
            "utf-8"
        )
        if project_source_digest(code) != manifest.project_source_sha256:
            raise ValueError("Project preprocessing source checksum mismatch.")
        load_project_module(code)
    pipeline = _load_pipeline_payload(payload, manifest)
    if (
        _manifest(
            pipeline,
            payload,
            manifest.use_tuned_thresholds,
            format_version=manifest.format_version,
        )
        # _check_runtime already validated these pins, allowing Python patch differences.
        .model_copy(update={"requirements": manifest.requirements})
        != manifest
    ):
        raise ValueError("Local artifact manifest disagrees with its fitted pipeline.")
    return FittedPipelineArtifact(manifest, pipeline)


def _load_pipeline_payload(payload: bytes, manifest: FittedPipelineManifest) -> SkyulfPipeline:
    """Restore the versioned payload only after transport and runtime validation."""
    saved = pickle.loads(payload)  # nosec B301 -- trusted producer only, after size/runtime/checksum checks
    if manifest.format_version == 2:
        if type(saved) is not dict or set(saved) != {"pipeline", "use_tuned_thresholds"}:
            raise ValueError("Invalid local artifact payload envelope.")
        policy = saved["use_tuned_thresholds"]
        if type(policy) is not bool:
            raise ValueError("Local artifact payload threshold policy must be a boolean.")
        if policy != manifest.use_tuned_thresholds:
            raise ValueError("Local artifact manifest threshold policy disagrees with its payload.")
        pipeline = saved["pipeline"]
    else:
        pipeline = saved
    if type(pipeline) is not SkyulfPipeline:
        raise ValueError("Local artifact payload is not a SkyulfPipeline.")
    if manifest.format_version == 1 and getattr(pipeline, "_tuned_thresholds", None) is not None:
        raise ValueError(
            "Local artifact version 1 cannot bind fitted threshold policy; re-export the "
            "trusted fitted pipeline with an explicit use_tuned_thresholds choice."
        )
    return pipeline


def _validate_manifest_schema(manifest: FittedPipelineManifest) -> None:
    """Require matching dtype counts and unique names for input and model schemas."""
    if len(manifest.input_columns) != len(manifest.input_dtypes) or len(
        manifest.feature_columns
    ) != len(manifest.feature_dtypes):
        raise ValueError("Local artifact schema columns and dtypes disagree.")
    if any(
        len(set(columns)) != len(columns)
        for columns in (manifest.input_columns, manifest.feature_columns)
    ):
        raise ValueError("Local artifact schema columns must be unique.")


def require_pipeline_scope(
    artifact: FittedPipelineArtifact,
    scope: Literal["whole_frame_local", "row_local_http", "spark_worker", "spark_native"],
) -> None:
    """Reject execution modes that the fitted artifact has not been proven to support."""
    if not isinstance(artifact, FittedPipelineArtifact):
        raise TypeError("Expected a FittedPipelineArtifact.")
    if scope != artifact.manifest.execution_scope:
        raise ValueError(f"Local pipeline artifact is not eligible for {scope}.")


def predict_pipeline(
    frame: pd.DataFrame | pl.DataFrame, artifact: FittedPipelineArtifact
) -> pd.DataFrame:
    """Apply saved preprocessing and publish null outcomes for fitted row exclusions."""
    if not isinstance(artifact, FittedPipelineArtifact):
        raise TypeError("Expected a FittedPipelineArtifact.")
    if not isinstance(frame, pd.DataFrame | pl.DataFrame):
        raise TypeError("Local prediction requires a pandas or Polars DataFrame.")
    native = validate_pipeline_input(frame, artifact)
    if has_exclusion_steps(artifact):
        return _predict_with_exclusions(frame, native, artifact)
    if not len(native):
        return _empty_prediction(frame, artifact.manifest)
    prediction = np.asarray(
        artifact.pipeline.predict(
            native, use_tuned_thresholds=artifact.manifest.use_tuned_thresholds
        )
    )
    if prediction.shape != (len(frame),):
        raise ValueError("Local pipeline prediction shape disagrees with input rows.")
    index = frame.index if isinstance(frame, pd.DataFrame) else pd.RangeIndex(len(frame))
    result = pd.DataFrame({"prediction": pd.Series(prediction, index=index)})
    _add_probabilities(result, native, artifact)
    return result


def _add_probabilities(
    result: pd.DataFrame,
    native: pd.DataFrame | pl.DataFrame,
    artifact: FittedPipelineArtifact,
) -> None:
    """Append recorded classifier probabilities while preserving the ordinary prediction path."""
    if (
        artifact.manifest.task == "classification"
        and artifact.manifest.classification_probabilities
    ):
        transformed = artifact.pipeline.feature_engineer.transform(native, preserve_rows=True)
        probabilities = np.asarray(artifact.pipeline._predict_proba_transformed(transformed))
        if probabilities.shape != (len(result), len(artifact.manifest.classes)):
            raise ValueError("Local pipeline probability shape disagrees with recorded classes.")
        for position in range(len(artifact.manifest.classes)):
            result[f"probability_{position}"] = probabilities[:, position]


def has_exclusion_steps(artifact: FittedPipelineArtifact) -> bool:
    """Identify saved filters whose prediction outputs require explicit coverage columns."""
    return any(is_exclusion_step(step) for step in artifact.pipeline.feature_engineer.fitted_steps)


def _predict_with_exclusions(
    frame: pd.DataFrame | pl.DataFrame,
    native: pd.DataFrame | pl.DataFrame,
    artifact: FittedPipelineArtifact,
) -> pd.DataFrame:
    """Scatter eligible predictions into typed, full-length nullable output columns."""
    pipeline, manifest = artifact.pipeline, artifact.manifest
    transformed, positions, reasons = pipeline.feature_engineer.transform_with_exclusions(native)
    index = frame.index if isinstance(frame, pd.DataFrame) else pd.RangeIndex(len(frame))
    empty = _empty_prediction(frame.head(0), manifest)
    result = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    label = "float64" if manifest.task == "regression" else label_dtype(manifest.classes)
    for column in empty:
        dtype = label if column == "prediction" else "float64"
        result[column] = _typed_column(pd.Series(pd.NA, index=result.index), dtype)
    if len(positions):
        result.loc[positions, "prediction"] = np.asarray(
            pipeline._predict_transformed(
                transformed, use_tuned_thresholds=manifest.use_tuned_thresholds
            )
        )
        if manifest.task == "classification" and manifest.classification_probabilities:
            probabilities = np.asarray(pipeline._predict_proba_transformed(transformed))
            if probabilities.shape != (len(positions), len(manifest.classes)):
                raise ValueError(
                    "Local pipeline probability shape disagrees with recorded classes."
                )
            result.loc[positions, list(empty.columns[1:])] = probabilities
    result["scoring_status"] = pd.Series(
        np.where(reasons.isna(), "predicted", "excluded"), index=result.index, dtype="string"
    )
    result["exclusion_reason"] = reasons
    result.index = index
    return result


def _empty_prediction(
    frame: pd.DataFrame | pl.DataFrame, manifest: FittedPipelineManifest
) -> pd.DataFrame:
    """Return the declared empty output without invoking preprocessing or estimators."""
    index = frame.index if isinstance(frame, pd.DataFrame) else pd.RangeIndex(0)
    dtype = "float64" if manifest.task == "regression" else label_dtype(manifest.classes)
    result = pd.DataFrame({"prediction": pd.Series(index=index, dtype=dtype)}, index=index)
    if manifest.task == "classification" and manifest.classification_probabilities:
        for position in range(len(manifest.classes)):
            result[f"probability_{position}"] = pd.Series(index=index, dtype="float64")
    return result


def _prediction_frame(
    frame: pd.DataFrame | pl.DataFrame, manifest: FittedPipelineManifest
) -> pd.DataFrame | pl.DataFrame:
    """Convert inputs to the engine recorded during the successful model fit."""
    if manifest.fitted_engine == "polars":
        return pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


def _project_contract(pipeline: SkyulfPipeline) -> tuple[str, ...]:
    """Validate saved scoring policies and dependency pins as part of model identity."""
    source = pipeline.config.get("project_python_source")
    config = pipeline.config.get("project_scoring")
    if config is not None:
        if not source:
            raise ValueError("Project scoring requires saved project Python source.")
        validate_scoring_config(config, source)
    pins = set(source_project_requirements(source) if source else ())
    pins.update(
        native_model_requirements(
            getattr(pipeline, "model_estimator", None),
            [record["artifact"] for record in pipeline.feature_engineer.fitted_steps],
        )
    )
    for record in pipeline.feature_engineer.fitted_steps:
        if record["type"] == "sentence_embedder":
            pins.update(_embedding_requirements(record["artifact"]))
    requirements = parse_project_requirements("\n".join(sorted(pins)))
    verify_project_requirements(requirements)
    return requirements


def _embedding_requirements(artifact: dict) -> tuple[str, ...]:
    """Read exact optional pins without importing or restoring the fitted encoder."""
    requirements = artifact.get("model_requirements", ())
    if not isinstance(requirements, tuple) or any(not isinstance(pin, str) for pin in requirements):
        raise ValueError("Saved embedding requirements must be a tuple of exact pins.")
    return parse_project_requirements("\n".join(requirements))


def validate_pipeline_input(
    frame: pd.DataFrame | pl.DataFrame, artifact: FittedPipelineArtifact
) -> pd.DataFrame | pl.DataFrame:
    """Validate raw columns and dtypes even when scoring eligibility excludes all rows."""
    native = _prediction_frame(frame, artifact.manifest)
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
    return native
