"""Native Spark preprocessing followed by bounded worker-side Python prediction."""

import importlib
from collections.abc import Callable, Iterator
from typing import Any

import pandas as pd

from ..core.capabilities import UnsupportedExecutionError
from ..core.execution import ExecutionOptions, FrameSpec
from ..preprocessing._spark import _column, _native, _resolved_names
from ..preprocessing.pipeline import FeatureEngineer
from ._manifest import BundleManifest, check_runtime, checksum
from ._model import load_model
from .bundle import InferenceBundle, _predict_features, _validate_bundle, _validate_frame


def predict_spark(
    frame: Any,
    bundle: InferenceBundle,
    *,
    frame_spec: FrameSpec,
    options: ExecutionOptions,
    mode: str,
) -> Any:
    """Apply frozen native FE and distribute a regression estimator over Spark workers.

    Currently accepts raw regression bundles with native row-preserving FE.
    Supply unique non-null integer, string or boolean keys; output contains
    those keys and prediction, with no row ordering guarantee. Extra unused
    input columns are projected away. Declared features must retain their
    training order and dtypes; keys never become model features.
    Integral and boolean model features require a nonnullable Spark schema,
    because Arrow otherwise can widen their values in pandas batches. Native
    FE output dtype changes are rejected against the fitted bundle schema.

    Schema, stage, capability and bundle checks precede distributed validation.
    Existing native FE performs bounded key and row-preservation checks on the
    driver. Prediction stays distributed, loading the model once per iterator.
    ``python_batch_rows`` bounds estimator calls, not Arrow transport batches
    or bytes. Configure ``spark.sql.execution.arrow.maxRecordsPerBatch`` on
    the caller's session separately; this function never changes that setting.
    Estimator pickle payloads must come from a trusted producer.
    """
    if not isinstance(frame_spec, FrameSpec):
        raise TypeError("frame_spec must be FrameSpec.")
    if not isinstance(options, ExecutionOptions) or options.engine != "spark":
        raise ValueError("predict_spark requires ExecutionOptions(engine='spark').")
    if frame_spec.target is not None:
        raise ValueError("Inference frame_spec must not declare a target.")
    if mode != "native_features":
        raise UnsupportedExecutionError(
            "inference", "predict", "spark", "Only mode='native_features' is implemented."
        )
    _validate_bundle(bundle, options)
    manifest = bundle.manifest
    if manifest.input_stage != "raw":
        raise ValueError("native_features currently requires a raw input_stage bundle.")
    if manifest.task != "regression":
        raise UnsupportedExecutionError(
            "inference", "predict", "spark", "Only regression bundles are currently supported."
        )
    native = _native(frame)
    if native.isStreaming:
        raise UnsupportedExecutionError(
            "inference", "predict", "spark", "Streaming inference is not supported."
        )
    _validate_names_and_keys(native, manifest, frame_spec)
    raw_names = {column.name for column in manifest.input_schema}
    raw = native.select(*[_column(native, name) for name in native.columns if name in raw_names])
    _validate_frame(raw, manifest.input_schema, "bundle input")
    selected = native.select(
        *[_column(native, name) for name in (*frame_spec.row_keys, *raw.columns)]
    )
    engineer = FeatureEngineer.from_state(
        bundle.feature_state, execution_options=options, frame_spec=frame_spec
    )
    # The portable allowlist contains native appliers that only construct Spark
    # expressions. Inspect their output types before FE's key-validation actions.
    preview = selected
    for step in engineer.fitted_steps:
        preview = _native(step["applier"].apply(preview, step["artifact"]))
    _model_features(preview, manifest)
    transformed = _native(engineer.transform(selected, preserve_rows=True))
    features = _model_features(transformed, manifest)
    worker_frame = transformed.select(
        *[_column(transformed, name) for name in (*frame_spec.row_keys, *features.columns)]
    )
    types = importlib.import_module("pyspark.sql.types")
    schema = types.StructType(
        [native.schema[key] for key in frame_spec.row_keys]
        + [
            types.StructField(column.name, types.DoubleType(), True)
            for column in manifest.output_schema
        ]
    )
    worker = _prediction_iterator(
        bundle.model_payload, manifest, frame_spec.row_keys, options.python_batch_rows
    )
    return worker_frame.mapInPandas(worker, schema=schema)


def _model_features(frame: Any, manifest: BundleManifest) -> Any:
    """Check native output positions and dtypes using only the lazy Spark schema."""
    features = frame.select(
        *[_column(frame, name) for name in frame.columns if name in manifest.feature_order]
    )
    _validate_frame(features, manifest.feature_schema, "model features")
    integral = {"byte", "short", "integer", "long", "boolean"}
    for field in features.schema.fields:
        if field.nullable and field.dataType.typeName() in integral:
            raise UnsupportedExecutionError(
                "inference",
                "predict",
                "spark",
                f"Nullable model feature {field.name!r} risks Arrow dtype conversion. "
                "Use a nonnullable Spark schema, or normalize to floats before training.",
            )
    return features


def _validate_names_and_keys(frame: Any, manifest: BundleManifest, spec: FrameSpec) -> None:
    """Reject ambiguous names, protected-column collisions and unsafe Arrow key types."""
    resolved = _resolved_names(frame)
    if len(resolved) != len(set(resolved)):
        raise ValueError("Duplicate Spark column names are unsupported.")
    sensitive = frame.sparkSession.conf.get("spark.sql.caseSensitive") == "true"

    def names(columns: tuple[str, ...]) -> set[str]:
        """Apply the same identifier resolution used by Spark for collision checks."""
        return set(columns if sensitive else (name.lower() for name in columns))

    keys = names(spec.row_keys)
    raw = names(tuple(col.name for col in manifest.input_schema))
    features = names(manifest.feature_order)
    output = names(tuple(col.name for col in manifest.output_schema))
    if len(keys) != len(spec.row_keys) or keys & (raw | features | output):
        raise ValueError("row_keys collide with feature or prediction columns.")
    if output & (raw | features | set(resolved)):
        raise ValueError("Input and prediction output columns collide.")
    if len(raw) != len(manifest.input_schema) or len(features) != len(manifest.feature_order):
        raise ValueError("Duplicate Spark feature names in bundle schema.")
    missing = set(spec.row_keys).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Spark row_keys: {sorted(missing)}")
    supported = {"byte", "short", "integer", "long", "string", "boolean"}
    for key in spec.row_keys:
        if frame.schema[key].dataType.typeName() not in supported:
            raise TypeError(
                f"Unsupported inference key dtype for {key}; use integer/string/boolean."
            )


def _prediction_iterator(
    payload: bytes, manifest: BundleManifest, row_keys: tuple[str, ...], batch_rows: int
) -> Callable[[Iterator[pd.DataFrame]], Iterator[pd.DataFrame]]:
    """Capture only frozen metadata and bytes, never a Spark session or dataframe."""

    def predict_batches(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Validate the worker runtime once and reuse one estimator across bounded chunks."""
        check_runtime(manifest)
        if checksum(payload) != manifest.model_sha256:
            raise ValueError("Bundle payload checksum mismatch on worker.")
        model = load_model(payload, manifest)
        for batch in batches:
            for offset in range(0, len(batch), batch_rows):
                chunk = batch.iloc[offset : offset + batch_rows]
                features = chunk.loc[:, list(manifest.feature_order)]
                _validate_frame(features, manifest.feature_schema, "worker model features")
                predictions = _predict_features(features, model, manifest, chunk.index)
                yield pd.concat([chunk.loc[:, list(row_keys)], predictions], axis=1)

    return predict_batches
