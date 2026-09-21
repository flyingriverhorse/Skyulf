"""Explicit standalone model bundles with raw/features inference contracts.

Estimator payloads use pickle and must come from a trusted producer. Checksums
detect corruption; they do not authenticate a producer or make pickle safe.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import is_classifier

from ..core.execution import ExecutionOptions
from ..core.schema import SkyulfSchema
from ..engines.pandas_engine import SkyulfPandasWrapper
from ..engines.polars_engine import SkyulfPolarsWrapper
from ..engines.sklearn_bridge import SklearnBridge
from ..engines.spark_engine import is_spark_input
from ..modeling._evaluation.thresholds import apply_thresholds
from ..pipeline import SkyulfPipeline
from ..pipeline.seal import artifact_digest
from ..preprocessing._feature_state import export_feature_state
from ..preprocessing.pipeline import FeatureEngineer
from ._manifest import (
    BundleManifest,
    ColumnSpec,
    check_runtime,
    checksum,
    label_dtype,
    manifest_bytes,
    runtime_requirements,
    schema_columns,
    semantic_digest,
)
from ._model import load_model, pipeline_model, serialize_model, threshold_provenance
from ._storage import read_payloads, write_payloads


@dataclass(frozen=True)
class InferenceBundle:
    """Frozen metadata and payload bytes, without a live estimator, session or connection."""

    manifest: BundleManifest
    feature_state: bytes
    model_payload: bytes

    @property
    def input_stage(self) -> str:
        """Expose whether the consumer supplies raw or already transformed features."""
        return self.manifest.input_stage

    @property
    def feature_order(self) -> tuple[str, ...]:
        """Expose the exact feature positions used during model training."""
        return self.manifest.feature_order

    @property
    def probability_columns(self) -> tuple[str, ...]:
        """Expose probability names in the same order as model classes."""
        return self.manifest.probability_columns

    @property
    def classes(self) -> tuple:
        """Expose original scalar class labels without converting them to column names."""
        return self.manifest.classes

    @property
    def positive_label(self) -> Any:
        """Expose the binary probability convention, or None for other tasks."""
        return self.manifest.positive_label

    @property
    def semantic_digest(self) -> str:
        """Expose content identity independently of estimator pickle protocol."""
        return self.manifest.semantic_digest


def build_bundle(
    pipeline: SkyulfPipeline,
    *,
    input_stage: str,
    feature_order: tuple[str, ...],
    options: ExecutionOptions | None = None,
    use_tuned_thresholds: bool = False,
) -> InferenceBundle:
    """Freeze a supported fitted standalone pipeline without refitting or retaining data.

    The requested order must match recorded model training columns. Raw input
    applies FE once; features input bypasses it. Tuning thresholds keep their
    existing default behavior; pipeline thresholds require explicit opt-in.
    Legacy artifacts without recorded schemas require a new successful fit.
    """
    if type(pipeline) is not SkyulfPipeline:
        raise TypeError("build_bundle requires a fitted standalone SkyulfPipeline.")
    if input_stage not in ("raw", "features") or type(use_tuned_thresholds) is not bool:
        raise ValueError("Invalid input_stage or use_tuned_thresholds.")
    budget = _options(options)
    model = pipeline_model(pipeline)
    schemas = getattr(pipeline, "_inference_schemas", None)
    if schemas is None:
        raise ValueError("Missing fitted input schemas; refit this legacy standalone pipeline.")
    raw_schema, feature_schema = (schema_columns(schema) for schema in schemas)
    if type(feature_order) is not tuple or feature_order != tuple(
        col.name for col in feature_schema
    ):
        raise ValueError("feature_order must match the actual model training order.")
    if len(feature_order) != int(model.n_features_in_):
        raise ValueError("Recorded model feature width is inconsistent.")
    feature_state = export_feature_state(pipeline.feature_engineer, options=budget)
    if len(feature_state) > budget.state_max_bytes:
        raise ValueError("Feature state exceeds state_max_bytes.")
    payload = serialize_model(model, budget.model_max_bytes)
    classification = is_classifier(model)
    classes = tuple(np.asarray(model.classes_).tolist()) if classification else ()
    probabilities = tuple(f"probability_{i}" for i in range(len(classes)))
    output = (
        ColumnSpec(name="prediction", dtype=label_dtype(classes) if classification else "float64"),
        *(ColumnSpec(name=name, dtype="float64") for name in probabilities),
    )
    manifest = BundleManifest(
        input_stage=input_stage,
        feature_order=feature_order,
        input_schema=raw_schema if input_stage == "raw" else feature_schema,
        feature_schema=feature_schema,
        output_schema=output,
        task="classification" if classification else "regression",
        classes=classes,
        positive_label=classes[1] if len(classes) == 2 else None,
        probability_columns=probabilities,
        thresholds=threshold_provenance(pipeline, classes, use_tuned_thresholds),
        requirements=runtime_requirements(),
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        model_sha256=checksum(payload),
        model_state_digest=artifact_digest(model).hex(),
        fe_sha256=checksum(feature_state),
        fe_semantic_digest=json.loads(feature_state)["semantic_digest"],
    )
    manifest = manifest.model_copy(update={"semantic_digest": semantic_digest(manifest)})
    bundle = InferenceBundle(manifest, feature_state, payload)
    _validate_bundle(bundle, budget)
    return bundle


def _options(options: ExecutionOptions | None) -> ExecutionOptions:
    """Use existing immutable byte budgets without implicitly selecting a runtime."""
    if options is not None and not isinstance(options, ExecutionOptions):
        raise TypeError("options must be ExecutionOptions.")
    return options if options is not None else ExecutionOptions("pandas")


def _validate_bundle(bundle: InferenceBundle, options: ExecutionOptions) -> None:
    """Check all metadata and byte identities before any estimator deserialization."""
    if not isinstance(bundle, InferenceBundle):
        raise TypeError("Expected InferenceBundle.")
    manifest = BundleManifest.model_validate_json(manifest_bytes(bundle.manifest))
    if len(manifest_bytes(manifest)) + len(bundle.feature_state) > options.state_max_bytes:
        raise ValueError("Bundle metadata and FE exceed state_max_bytes.")
    if len(bundle.model_payload) > options.model_max_bytes:
        raise ValueError("Estimator payload exceeds model_max_bytes.")
    if (
        checksum(bundle.model_payload) != manifest.model_sha256
        or checksum(bundle.feature_state) != manifest.fe_sha256
    ):
        raise ValueError("Bundle payload checksum mismatch.")
    if semantic_digest(manifest) != manifest.semantic_digest:
        raise ValueError("Bundle semantic digest mismatch.")
    FeatureEngineer.from_state(
        bundle.feature_state,
        execution_options=ExecutionOptions("pandas", state_max_bytes=options.state_max_bytes),
    )
    if json.loads(bundle.feature_state)["semantic_digest"] != manifest.fe_semantic_digest:
        raise ValueError("Feature state semantic digest mismatch.")
    check_runtime(manifest)


def _validate_frame(frame: Any, expected: tuple[ColumnSpec, ...], where: str) -> None:
    """Require names, positions and normalized dtypes; never silently reorder or cast."""
    actual = schema_columns(SkyulfSchema.from_dataframe(frame))
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("Duplicate input columns are not supported.")
    wanted = SkyulfSchema(
        tuple(col.name for col in expected), {col.name: col.dtype for col in expected}
    )
    observed = SkyulfSchema(
        tuple(col.name for col in actual), {col.name: col.dtype for col in actual}
    )
    wanted.assert_compatible(observed, check_dtypes=True, check_order=True, where=where)


def predict_local(
    frame: Any, bundle: InferenceBundle, *, options: ExecutionOptions | None = None
) -> pd.DataFrame:
    """Predict from a local pandas/Polars frame using exactly the declared input stage.

    Supply feature columns only, in the recorded order and dtypes. Extra IDs or
    targets are rejected; the output preserves a pandas input index. Spark
    frames require the separate distributed runner and are never collected.
    """
    if is_spark_input(frame):
        raise TypeError("predict_local rejects Spark input; use the distributed inference runner.")
    if isinstance(frame, SkyulfPandasWrapper | SkyulfPolarsWrapper):
        frame = frame.to_native()
    if not isinstance(frame, pd.DataFrame | pl.DataFrame):
        raise TypeError("predict_local requires a pandas or Polars DataFrame.")
    budget = _options(options)
    _validate_bundle(bundle, budget)
    manifest = bundle.manifest
    _validate_frame(frame, manifest.input_schema, "bundle input")
    features = frame
    if bundle.input_stage == "raw":
        local_options = replace(
            budget, engine="polars" if isinstance(frame, pl.DataFrame) else "pandas"
        )
        features = FeatureEngineer.from_state(
            bundle.feature_state, execution_options=local_options
        ).transform(frame, preserve_rows=True)
    _validate_frame(features, manifest.feature_schema, "model features")
    index = frame.index if isinstance(frame, pd.DataFrame) else pd.RangeIndex(len(frame))
    if len(frame) == 0:
        return pd.DataFrame(
            {col.name: pd.Series(index=index, dtype=col.dtype) for col in manifest.output_schema}
        )
    model = load_model(bundle.model_payload, manifest)
    return _predict_features(features, model, manifest, index)


def _predict_features(
    features: Any, model: Any, manifest: BundleManifest, index: pd.Index
) -> pd.DataFrame:
    """Apply an already loaded estimator without preprocessing or retaining batch state."""
    values, _ = SklearnBridge.to_sklearn(features, validate_features=True)
    probabilities = None
    if manifest.task == "classification":
        probabilities = np.asarray(model.predict_proba(values))
        if probabilities.shape != (len(features), len(manifest.classes)):
            raise ValueError("Model probability shape violates the bundle contract.")
    if manifest.thresholds.values:
        thresholds = dict(zip(manifest.classes, manifest.thresholds.values, strict=True))
        predictions = apply_thresholds(probabilities, thresholds, classes=manifest.classes)
    else:
        predictions = np.asarray(model.predict(values))
    if predictions.shape != (len(features),):
        raise ValueError("Model prediction shape violates the bundle contract.")
    output = pd.DataFrame(
        {"prediction": pd.Series(predictions, index=index, dtype=manifest.output_schema[0].dtype)}
    )
    if probabilities is not None:
        for position, name in enumerate(manifest.probability_columns):
            output[name] = probabilities[:, position]
    return output


def save_bundle(
    bundle: InferenceBundle, path: str | Path, *, options: ExecutionOptions | None = None
) -> None:
    """Write a new bundle directory; refuse to overwrite an existing path."""
    _validate_bundle(bundle, _options(options))
    write_payloads(
        path, manifest_bytes(bundle.manifest), bundle.feature_state, bundle.model_payload
    )


def load_bundle(path: str | Path, *, options: ExecutionOptions | None = None) -> InferenceBundle:
    """Load a trusted bundle directory after size, schema, checksum and runtime checks.

    This operation deserializes pickle. Only load bundles from trusted producers;
    checksum verification is not an authentication mechanism.
    """
    budget = _options(options)
    manifest, features, model = read_payloads(path, budget)
    bundle = InferenceBundle(manifest, features, model)
    _validate_bundle(bundle, budget)
    load_model(model, manifest)
    return bundle
