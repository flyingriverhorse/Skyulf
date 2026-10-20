"""Named-input Spark pyfunc inference with pinned, inspected local artifacts."""

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ....inference.fitted_pipeline import FittedPipelineArtifact
from ....inference.model_set import ModelSetArtifact
from ....inference.model_set_scoring import model_set_output_schema
from ....inference.pipeline_scoring import scoring_output_schema
from ...databricks.feature_store.scoring import (
    feature_binding,
    feature_source_columns,
    native_feature_score,
    validate_feature_snapshot,
    validate_feature_source,
)
from ..registration.registry import download_registered_package, validate_registry_options
from ..shared._client import make_registry_client
from ..shared._model_metadata import mlflow_dtype, normalized_dtype
from ..shared._nullable_transport import TRANSPORT_KEY, transport_spec
from ._spark_environment import package_source_root, source_digest
from ._spark_output import (
    DEFAULT_PREDICTION_BATCH_ROWS,
    SPARK_NULL_TEXT,
    nullable_outputs,
    spark_output_params,
    spark_transport_dtype,
    spark_udf_params,
    validate_prediction_batch_rows,
)

SAFETY_KEY = "skyulf_partition_safety"
SOURCE_KEY = "skyulf_runtime_source_sha256"


def partition_safety_certificate(
    artifact: FittedPipelineArtifact | ModelSetArtifact,
) -> dict[str, Any]:
    """Inspect the loaded payload and detach JSON-compatible certificate evidence."""
    from ....inference.model_set_partition_safety import (  # noqa: PLC0415
        require_partition_safe_model_set,
    )
    from ....inference.partition_safety import require_partition_safe_pipeline  # noqa: PLC0415

    if isinstance(artifact, FittedPipelineArtifact):
        evidence = asdict(require_partition_safe_pipeline(artifact))
    elif isinstance(artifact, ModelSetArtifact):
        evidence = require_partition_safe_model_set(artifact)
    else:
        raise TypeError("Spark pyfunc requires a loaded local pipeline or model set artifact.")
    return json.loads(json.dumps(evidence, allow_nan=False))


def optional_partition_certificate(
    artifact: FittedPipelineArtifact | ModelSetArtifact,
) -> dict[str, Any] | None:
    """Add inspected evidence to new packages without narrowing existing local logging."""
    try:
        return partition_safety_certificate(artifact)
    except ValueError:
        return None


def runtime_source_digest() -> str:
    """Bind worker imports to the exact Python source shipped with this package."""
    return source_digest(package_source_root())


def validate_worker_certificate(
    artifact: FittedPipelineArtifact | ModelSetArtifact,
    certificate: dict[str, Any] | None,
    source_sha256: str | None,
) -> None:
    """Reinspect worker-loaded fitted assets and reject stale or substituted evidence."""
    if certificate is None:
        return
    if partition_safety_certificate(artifact) != certificate:
        raise ValueError("Spark partition safety certificate differs from the loaded artifact.")
    if source_sha256 != runtime_source_digest():
        raise ValueError("Spark worker runtime source differs from the packaged source.")


def _contract(artifact: FittedPipelineArtifact | ModelSetArtifact) -> tuple[list, tuple]:
    """Return ordered artifact inputs and every declared prediction output."""
    if isinstance(artifact, FittedPipelineArtifact):
        inputs = list(
            zip(artifact.manifest.input_columns, artifact.manifest.input_dtypes, strict=True)
        )
        return inputs, scoring_output_schema(artifact)
    return (
        [(column.name, column.dtype) for column in artifact.manifest.input_schema],
        model_set_output_schema(artifact),
    )


def _validate_request(model_uri: str, env_manager: str, keys: tuple[str, ...]) -> None:
    """Reject mutable model selectors and ambiguous keys before worker preparation."""
    if not isinstance(model_uri, str) or not re.fullmatch(
        r"models:/[^/@?#]+/[1-9][0-9]*", model_uri
    ):
        raise ValueError("Spark pyfunc requires a concrete models:/name/version model URI.")
    if env_manager not in {"local", "virtualenv"}:
        raise ValueError("Spark pyfunc env_manager must explicitly be local or virtualenv.")
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("Spark pyfunc requires named record_key_columns.")
    if len({key.casefold() for key in keys}) != len(keys):
        raise ValueError("Spark pyfunc record keys must be unique.")


def _validate_columns(frame: Any, inputs: list, keys: tuple[str, ...], outputs: tuple) -> None:
    """Reject missing or ambiguous source fields and output/key collisions eagerly."""
    columns = frame.columns
    if len({name.casefold() for name in columns}) != len(columns):
        raise ValueError("Spark input columns must be unique ignoring case.")
    missing = ({name for name, _ in inputs} | set(keys)) - set(columns)
    if missing:
        raise ValueError(f"Spark pyfunc input is missing required columns: {sorted(missing)}.")
    output_names = {column.name.casefold() for column in outputs}
    if len(output_names) != len(outputs):
        raise ValueError("Spark pyfunc output columns must be unique ignoring case.")


def _validate_model_keys(
    artifact: FittedPipelineArtifact | ModelSetArtifact,
    keys: tuple[str, ...],
    inputs: list,
    outputs: tuple,
) -> None:
    """Separate raw single-model identities and enforce the model-set keyed contract."""
    if isinstance(artifact, ModelSetArtifact):
        if keys != artifact.manifest.record_key_columns:
            raise ValueError("Spark record keys differ from the model-set contract.")
        return
    names = {key.casefold() for key in keys}
    if names & {name.casefold() for name, _ in inputs}:
        raise ValueError("Spark single-pipeline record keys cannot be model inputs.")
    if names & {column.name.casefold() for column in outputs}:
        raise ValueError("Spark prediction output collides with a record key.")


def _download_package(
    model_uri: str,
    tracking_uri: str | None,
    registry_uri: str | None,
) -> tuple[str, Any]:
    """Download one concrete registry package using explicit, unchanged client stores."""
    import mlflow  # noqa: PLC0415  # ty: ignore[unresolved-import]

    name, version = model_uri.removeprefix("models:/").rsplit("/", 1)
    validate_registry_options(name, tracking_uri, registry_uri)
    client = make_registry_client(mlflow, tracking_uri, registry_uri)
    local = download_registered_package(mlflow, client, name, version, tracking_uri)
    return local, mlflow.models.Model.load(Path(local))


def _validate_package(info: Any, certificate: dict, inputs: list, outputs: tuple) -> None:
    """Bind the pinned MLflow package to the driver-inspected artifact and transport."""
    metadata = info.metadata or {}
    if metadata.get(SAFETY_KEY) != certificate:
        raise ValueError(
            "Pinned MLflow package lacks the matching Spark partition safety certificate."
        )
    _validate_package_identity(metadata, certificate)
    if metadata.get(SOURCE_KEY) != runtime_source_digest():
        raise ValueError("Pinned MLflow package runtime source differs from the scoring runtime.")
    if metadata.get(TRANSPORT_KEY) != transport_spec(inputs):
        raise ValueError("Pinned MLflow package nullable transport differs from its artifact.")
    _validate_signature(info.signature, inputs, outputs)
    if (
        info.signature.params is None
        or info.signature.params.to_dict() != spark_output_params().to_dict()
    ):
        raise ValueError(
            "Pinned MLflow package lacks the certified Spark output transport contract."
        )


def _scoring_package(artifact: Any, path: str, package: Any) -> Any:
    """Validate the native feature envelope before applying unchanged raw pyfunc gates."""
    binding = feature_binding(artifact)
    packaged = (package.metadata or {}).get("skyulf_feature_store")
    if packaged != binding:
        raise ValueError("Pinned package feature lookup differs from the loaded artifact binding.")
    if binding is None:
        return package
    from ..models.feature_model import feature_package_models  # noqa: PLC0415

    outer, raw, _ = feature_package_models(path)
    if outer.metadata.get("skyulf_feature_store") != binding:
        raise ValueError("Pinned package feature lookup differs from the loaded artifact binding.")
    return raw


def _validate_package_identity(metadata: dict, certificate: dict) -> None:
    """Bind additive safety evidence to the original artifact kind, scope and digest."""
    if "pipeline_sha256" in certificate:
        kind, field, digest = (
            "local_pipeline",
            "local_pipeline_digest",
            certificate["pipeline_sha256"],
        )
    else:
        kind, field, digest = "model_set", "model_set_digest", certificate["model_set_sha256"]
    expected = {
        "skyulf_artifact_kind": kind,
        "skyulf_execution_scope": "whole_frame_local",
        field: digest,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Pinned MLflow package identity differs from its inspected artifact.")


def _validate_signature(signature: Any, inputs: list, outputs: tuple) -> None:
    """Reject changed field names or scalar coercions before creating the UDF."""
    expected_inputs = _expected_inputs(inputs)
    expected_outputs = [(column.name, mlflow_dtype(column.dtype)) for column in outputs]
    if signature is None or signature.inputs is None or signature.outputs is None:
        raise ValueError("Spark pyfunc requires an explicit complete model signature.")
    actual_inputs = [(column.name, column.type) for column in signature.inputs.inputs]
    actual_outputs = [(column.name, column.type) for column in signature.outputs.inputs]
    if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
        raise ValueError("Pinned MLflow package signature differs from its artifact contract.")


def _expected_inputs(inputs: list) -> list:
    """Resolve exact saved transport and scalar aliases for signature admission."""
    spec = transport_spec(inputs)
    encoded = spec["columns"] if spec else {}
    return [
        (name, mlflow_dtype("string" if name in encoded else _normalized_dtype(dtype)))
        for name, dtype in inputs
    ]


def _normalized_dtype(dtype: str) -> str:
    """Translate pandas scalar aliases into the package's MLflow type vocabulary."""
    return normalized_dtype(dtype)


def _spark_type(dtype: str) -> Any:
    """Map each supported exact scalar dtype without a scalar-double approximation."""
    from pyspark.sql import types  # noqa: PLC0415  # ty: ignore[unresolved-import]

    mapping = {
        "int32": types.IntegerType,
        "int64": types.LongType,
        "float32": types.FloatType,
        "float64": types.DoubleType,
        "bool": types.BooleanType,
        "string": types.StringType,
    }
    try:
        return mapping[dtype]()
    except KeyError as exc:
        raise ValueError(f"Unsupported Spark pyfunc output dtype: {dtype}.") from exc


def _quoted(name: str) -> str:
    """Quote literal Spark column names including dots and embedded backticks."""
    return "`" + name.replace("`", "``") + "`"


def _named_inputs(frame: Any, inputs: list) -> Any:
    """Cast only declared extension transport columns before Arrow sees nullable ints."""
    from pyspark.sql import functions as functions  # noqa: PLC0415  # ty: ignore[unresolved-import]

    spec = transport_spec(inputs)
    encoded = spec["columns"] if spec else {}
    selected = []
    for name, dtype in inputs:
        column = functions.col(_quoted(name))
        if name in encoded:
            expected = {
                "Int32": "int",
                "Int64": "bigint",
                "boolean": "boolean",
                "Boolean": "boolean",
            }
            if frame.schema[name].dataType.simpleString() != expected[dtype]:
                raise ValueError(
                    f"Nullable Spark input {name!r} requires exact native {dtype} storage."
                )
            column = column.cast("string")
        selected.append(column.alias(name))
    return functions.struct(*selected)


def _normalize_model_values(frame: Any, inputs: list, keys: tuple, binding: dict | None) -> Any:
    """Normalize numeric model inputs without rounding record or native lookup identities."""
    from skyulf.engines.spark_engine import normalize_spark_decimals  # noqa: PLC0415

    columns = {name for name, _ in inputs}.difference(keys)
    if binding is not None:
        from ...databricks.feature_store.lifecycle_config import (  # noqa: PLC0415
            deserialize_feature_spec,
        )

        lookup = deserialize_feature_spec(binding["lookup_spec"])
        controls = {
            name for item in lookup.lookups for name in (*item.lookup_key, *item.timestamp_columns)
        }
        columns.difference_update(controls)
    normalized, _ = normalize_spark_decimals(frame, columns=columns)
    return normalized


def predict_spark_pyfunc(
    spark: Any,
    frame: Any,
    *,
    model_uri: str,
    artifact: FittedPipelineArtifact | ModelSetArtifact,
    record_key_columns: tuple[str, ...],
    env_manager: str,
    prediction_batch_rows: int = DEFAULT_PREDICTION_BATCH_ROWS,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> Any:
    """Return keyed distributed predictions through an inspected named-input pyfunc.

    ``local`` requires executor dependencies matching the fitted runtime; it does
    not inherit notebook installations by promise. ``virtualenv`` recreates the
    saved exact dependency pins. MLflow validates compute-specific restrictions.
    ``prediction_batch_rows`` bounds each fitted scorer call inside a worker;
    it does not change the platform-managed Arrow input/output allocation.
    Callers validate global key uniqueness before publication.
    """
    keys = tuple(record_key_columns)
    _validate_request(model_uri, env_manager, keys)
    validate_prediction_batch_rows(prediction_batch_rows)
    certificate = partition_safety_certificate(artifact)
    inputs, outputs = _contract(artifact)
    binding = feature_binding(artifact)
    source_inputs = [(name, "") for name in feature_source_columns(artifact)] if binding else inputs
    _validate_columns(frame, source_inputs, keys, outputs)
    validate_feature_source(frame, artifact)
    _validate_model_keys(artifact, keys, inputs, outputs)
    frame = _normalize_model_values(frame, inputs, keys, binding)
    package_path, package = _download_package(model_uri, tracking_uri, registry_uri)
    scoring = _scoring_package(artifact, package_path, package)
    _validate_package(scoring, certificate, inputs, outputs)
    udf_params = spark_udf_params(prediction_batch_rows)
    if binding is not None:
        return _predict_feature_frame(
            spark,
            frame,
            artifact,
            model_uri,
            keys,
            outputs,
            env_manager,
            udf_params,
            tracking_uri,
            registry_uri,
        )

    import mlflow  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from pyspark.sql import types  # noqa: PLC0415  # ty: ignore[unresolved-import]

    result_type = _result_type(types, outputs, _nullable(artifact), keys)
    named = _named_inputs(frame, inputs)
    udf = mlflow.pyfunc.spark_udf(
        spark,
        package_path,
        result_type=result_type,
        env_manager=env_manager,
        params=udf_params,
    )
    return _keyed_result(frame, keys, outputs, udf(named), nullable=_nullable(artifact))


def _predict_feature_frame(
    spark: Any,
    frame: Any,
    artifact: Any,
    model_uri: str,
    keys: tuple[str, ...],
    outputs: tuple,
    env_manager: str,
    udf_params: dict[str, Any],
    tracking_uri: str | None,
    registry_uri: str | None,
) -> Any:
    """Expand the native SDK's prediction struct and verify exact source membership."""
    from pyspark.sql import functions, types  # noqa: PLC0415  # ty: ignore[unresolved-import]

    validate_feature_snapshot(spark, artifact)
    result_type = _result_type(types, outputs, _nullable(artifact), keys)
    scored = native_feature_score(
        frame,
        model_uri=model_uri,
        result_type=result_type,
        env_manager=env_manager,
        params=udf_params,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    if scored.schema["prediction"].dataType != result_type:
        raise ValueError("Native feature prediction schema differs from the artifact contract.")
    result = _keyed_result(
        scored, keys, outputs, functions.col("prediction"), nullable=_nullable(artifact)
    )
    _validate_feature_keys(frame, result, keys)
    validate_feature_snapshot(spark, artifact)
    return result


def _validate_feature_keys(source: Any, output: Any, keys: tuple[str, ...]) -> None:
    """Require one output for every original key after potentially expanding SDK joins."""
    from ...databricks.scoring.batch.spark_scoring import read_distributed_rows  # noqa: PLC0415

    source_rows = read_distributed_rows(source, keys, keys)
    output_rows = read_distributed_rows(output, keys, keys)
    if len(source_rows) != len(output_rows):
        raise ValueError("Native feature lookup changed source row cardinality.")
    unexpected = output_rows.frame.join(source_rows.frame, on=list(keys), how="left_anti")
    if unexpected.limit(1).count():
        raise ValueError("Native feature lookup changed source record keys.")


def _nullable(artifact: Any) -> bool:
    """Saved scoring rules leave predictions and rule outputs null for excluded rows."""
    return nullable_outputs(artifact)


def _result_type(types: Any, outputs: tuple, nullable: bool, keys: tuple[str, ...] = ()) -> Any:
    """Declare each field with the dtype it travels in from the workers."""
    return types.StructType(
        [
            types.StructField(
                column.name,
                _spark_type(
                    spark_transport_dtype(column.dtype, nullable and column.name not in keys)
                ),
                True,
            )
            for column in outputs
        ]
    )


def _decoded(field: Any, dtype: str, nullable: bool) -> Any:
    """Turn the workers' null markers back into SQL nulls and restore the declared type."""
    from pyspark.sql import functions  # noqa: PLC0415  # ty: ignore[unresolved-import]

    if dtype == "string":
        return functions.when(field == SPARK_NULL_TEXT, functions.lit(None)).otherwise(field)
    if not nullable or dtype not in ("float64", "int64", "bool"):
        return field
    value = functions.when(functions.isnan(field), functions.lit(None)).otherwise(field)
    return value if dtype == "float64" else value.cast(_spark_type(dtype))


def _keyed_result(
    frame: Any, keys: tuple[str, ...], outputs: tuple, result: Any, *, nullable: bool = False
) -> Any:
    """Keep raw record identities separate from expanded model output fields."""
    from pyspark.sql import functions  # noqa: PLC0415  # ty: ignore[unresolved-import]

    # A separate projection avoids collisions with user columns or existing predictions.
    prediction = "__skyulf_pyfunc_result"
    while prediction.casefold() in {key.casefold() for key in keys}:
        prediction += "_"
    scored = frame.select(*_record_keys(frame, keys, outputs), result.alias(prediction))
    return scored.select(
        *(functions.col(_quoted(key)) for key in keys),
        *(
            _decoded(
                functions.col(_quoted(prediction)).getField(column.name), column.dtype, nullable
            ).alias(column.name)
            for column in outputs
            if column.name not in keys
        ),
    )


def _record_keys(frame: Any, keys: tuple[str, ...], outputs: tuple) -> list[Any]:
    """Widen native integer set identities to their declared BIGINT publication contract."""
    from pyspark.sql import functions  # noqa: PLC0415  # ty: ignore[unresolved-import]

    declared = {column.name: column.dtype for column in outputs}
    result = []
    for key in keys:
        value = functions.col(_quoted(key))
        if declared.get(key) == "int64":
            if frame.schema[key].dataType.simpleString() not in {
                "tinyint",
                "smallint",
                "int",
                "bigint",
            }:
                raise ValueError(
                    f"Model-set integer record key {key!r} requires native integer storage."
                )
            value = value.cast("bigint").alias(key)
        result.append(value)
    return result
