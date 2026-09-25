"""Spark inference with native or worker-local portable Python preprocessing."""

import importlib
from collections.abc import Callable, Iterator
from typing import Any

import pandas as pd

from ..core.capabilities import UnsupportedExecutionError
from ..core.execution import ExecutionOptions, FrameSpec
from ..preprocessing._spark import (
    _case_sensitive,
    _column,
    _native,
    _resolved_names,
    _validate_keys,
)
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
    """Apply frozen FE and distribute a regression or classification estimator.

    Raw bundles support native Spark FE (``native_features``) and
    portable pandas FE in each worker iterator (``python_pipeline``).
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
    if mode not in ("native_features", "python_pipeline"):
        raise UnsupportedExecutionError(
            "inference",
            "predict",
            "spark",
            "Only mode='native_features' or mode='python_pipeline' is implemented.",
        )
    _validate_bundle(bundle, options)
    manifest = bundle.manifest
    if manifest.input_stage != "raw":
        raise ValueError(f"{mode} currently requires a raw input_stage bundle.")
    native = _native(frame)
    if native.isStreaming:
        raise UnsupportedExecutionError(
            "inference", "predict", "spark", "Streaming inference is not supported."
        )
    _validate_names_and_keys(native, manifest, frame_spec)
    input_names = {column.name for column in manifest.input_schema}
    raw = native.select(*[_column(native, name) for name in native.columns if name in input_names])
    _validate_frame(raw, manifest.input_schema, "bundle input")
    selected_names = (
        tuple(frame_spec.record_key_columns)
        + tuple(column.name for column in manifest.input_schema)
        if mode == "python_pipeline"
        else tuple(frame_spec.record_key_columns) + tuple(raw.columns)
    )
    selected = native.select(*[_column(native, name) for name in selected_names])
    if mode == "python_pipeline":
        _validate_python_input_transport(native, manifest)
        engineer = FeatureEngineer.from_state(
            bundle.feature_state,
            execution_options=ExecutionOptions("pandas", state_max_bytes=options.state_max_bytes),
        )
        _validate_python_pipeline(engineer)
        _validate_keys(native, frame_spec)
        worker = _python_pipeline_prediction_iterator(
            bundle.feature_state,
            bundle.model_payload,
            manifest,
            frame_spec.record_key_columns,
            options.python_batch_rows,
            options.state_max_bytes,
        )
        return selected.mapInPandas(worker, schema=_prediction_schema(native, manifest, frame_spec))
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
        *[
            _column(transformed, name)
            for name in (*frame_spec.record_key_columns, *features.columns)
        ]
    )
    schema = _prediction_schema(native, manifest, frame_spec)
    worker = _prediction_iterator(
        bundle.model_payload, manifest, frame_spec.record_key_columns, options.python_batch_rows
    )
    return worker_frame.mapInPandas(worker, schema=schema)


def _prediction_schema(native: Any, manifest: BundleManifest, spec: FrameSpec) -> Any:
    """Build the stable Spark output schema from source key fields and the manifest."""
    types = importlib.import_module("pyspark.sql.types")
    return types.StructType(
        [native.schema[key] for key in spec.record_key_columns]
        + [
            types.StructField(column.name, _spark_output_type(types, column.dtype), True)
            for column in manifest.output_schema
        ]
    )


def _spark_output_type(types: Any, dtype: str) -> Any:
    """Map the manifest's primitive prediction dtype to an explicit Spark type."""
    mapping = {
        "bool": types.BooleanType,
        "float64": types.DoubleType,
        "int64": types.LongType,
        "string": types.StringType,
    }
    try:
        return mapping[dtype]()
    except KeyError as error:
        raise ValueError(f"Unsupported Spark prediction dtype {dtype!r}.") from error


def _validate_python_pipeline(engineer: FeatureEngineer) -> None:
    """Allow only the frozen, row-independent portable FE nodes in worker mode."""
    supported = {"SimpleImputer", "StandardScaler"}
    for step in engineer.fitted_steps:
        node_type = step["type"]
        params = step.get("params", {})
        if node_type == "SimpleImputer" and params.get("strategy", "mean") not in (
            "mean",
            "constant",
        ):
            raise UnsupportedExecutionError(
                node_type,
                "apply",
                "spark",
                "python_pipeline supports only mean or constant SimpleImputer state.",
            )
        if node_type not in supported:
            raise UnsupportedExecutionError(
                node_type,
                "apply",
                "spark",
                "python_pipeline requires portable row-independent, row-preserving FE.",
            )


def _validate_python_input_transport(native: Any, manifest: BundleManifest) -> None:
    """Reject nullable integral inputs before Arrow can widen them in Python workers."""
    integral = {"byte", "short", "integer", "long", "boolean"}
    fields = {field.name: field for field in native.schema.fields}
    for column in manifest.input_schema:
        field = fields[column.name]
        if field.nullable and field.dataType.typeName() in integral:
            raise UnsupportedExecutionError(
                "inference",
                "predict",
                "spark",
                f"Nullable model input {field.name!r} risks Arrow dtype conversion. "
                "Use a nonnullable Spark schema, or normalize to floats before training.",
            )


def _python_pipeline_prediction_iterator(
    feature_state: bytes,
    payload: bytes,
    manifest: BundleManifest,
    record_key_columns: tuple[str, ...],
    batch_rows: int,
    state_max_bytes: int,
) -> Callable[[Iterator[pd.DataFrame]], Iterator[pd.DataFrame]]:
    """Restore portable pandas FE once, then apply it to each Arrow batch before prediction."""

    def predict_batches(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Run frozen local FE and one loaded estimator across worker batches."""
        check_runtime(manifest)
        if checksum(feature_state) != manifest.fe_sha256:
            raise ValueError("Bundle feature state checksum mismatch on worker.")
        if checksum(payload) != manifest.model_sha256:
            raise ValueError("Bundle payload checksum mismatch on worker.")
        engineer = FeatureEngineer.from_state(
            feature_state,
            execution_options=ExecutionOptions("pandas", state_max_bytes=state_max_bytes),
        )
        _validate_python_pipeline(engineer)
        model = load_model(payload, manifest)
        raw_columns = [column.name for column in manifest.input_schema]
        for batch in batches:
            if batch.empty:
                continue
            source = batch.loc[:, [*record_key_columns, *raw_columns]]
            transformed = engineer.transform(source, preserve_rows=True)
            features = transformed.loc[:, list(manifest.feature_order)]
            _validate_frame(features, manifest.feature_schema, "worker model features")
            for offset in range(0, len(features), batch_rows):
                chunk = features.iloc[offset : offset + batch_rows]
                keys = source.iloc[offset : offset + batch_rows].loc[:, list(record_key_columns)]
                predictions = _predict_features(chunk, model, manifest, chunk.index)
                yield pd.concat(
                    [keys.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1
                )

    return predict_batches


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
    sensitive = _case_sensitive(frame)

    def names(columns: tuple[str, ...]) -> set[str]:
        """Apply the same identifier resolution used by Spark for collision checks."""
        return set(columns if sensitive else (name.lower() for name in columns))

    keys = names(spec.record_key_columns)
    raw = names(tuple(col.name for col in manifest.input_schema))
    features = names(manifest.feature_order)
    output = names(tuple(col.name for col in manifest.output_schema))
    if len(keys) != len(spec.record_key_columns) or keys & (raw | features | output):
        raise ValueError("record_key_columns collide with feature or prediction columns.")
    if output & (raw | features | set(resolved)):
        raise ValueError("Input and prediction output columns collide.")
    if len(raw) != len(manifest.input_schema) or len(features) != len(manifest.feature_order):
        raise ValueError("Duplicate Spark feature names in bundle schema.")
    missing = set(spec.record_key_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Spark record_key_columns: {sorted(missing)}")
    supported = {"byte", "short", "integer", "long", "string", "boolean"}
    for key in spec.record_key_columns:
        if frame.schema[key].dataType.typeName() not in supported:
            raise TypeError(
                f"Unsupported inference key dtype for {key}; use integer/string/boolean."
            )


def _prediction_iterator(
    payload: bytes, manifest: BundleManifest, record_key_columns: tuple[str, ...], batch_rows: int
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
                yield pd.concat([chunk.loc[:, list(record_key_columns)], predictions], axis=1)

    return predict_batches
