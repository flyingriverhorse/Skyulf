"""Named-input Spark pyfunc inference with pinned, inspected local artifacts."""

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...inference.local_pipeline import LocalPipelineArtifact
from ...inference.local_scoring import scoring_output_schema
from ...inference.model_set import ModelSetArtifact
from ...inference.model_set_scoring import model_set_output_schema
from ._client import make_registry_client
from ._model_metadata import mlflow_dtype
from ._nullable_transport import TRANSPORT_KEY, transport_spec
from ._spark_environment import source_digest
from ._spark_output import (
    DEFAULT_PREDICTION_BATCH_ROWS,
    SPARK_BATCH_ROWS_PARAM,
    SPARK_OUTPUT_PARAM,
    spark_output_params,
    validate_prediction_batch_rows,
)
from .registry import download_registered_package, validate_registry_options

SAFETY_KEY = "skyulf_partition_safety"
SOURCE_KEY = "skyulf_runtime_source_sha256"


def partition_safety_certificate(
    artifact: LocalPipelineArtifact | ModelSetArtifact,
) -> dict[str, Any]:
    """Inspect the loaded payload and detach JSON-compatible certificate evidence."""
    from ...inference.model_set_partition_safety import (  # noqa: PLC0415
        require_partition_safe_model_set,
    )
    from ...inference.partition_safety import require_partition_safe_pipeline  # noqa: PLC0415

    if isinstance(artifact, LocalPipelineArtifact):
        evidence = asdict(require_partition_safe_pipeline(artifact))
    elif isinstance(artifact, ModelSetArtifact):
        evidence = require_partition_safe_model_set(artifact)
    else:
        raise TypeError("Spark pyfunc requires a loaded local pipeline or model set artifact.")
    return json.loads(json.dumps(evidence, allow_nan=False))


def optional_partition_certificate(
    artifact: LocalPipelineArtifact | ModelSetArtifact,
) -> dict[str, Any] | None:
    """Add inspected evidence to new packages without narrowing existing local logging."""
    try:
        return partition_safety_certificate(artifact)
    except ValueError:
        return None


def runtime_source_digest() -> str:
    """Bind worker imports to the exact Python source shipped with this package."""
    root = Path(__file__).resolve().parents[2]
    return source_digest(root)


def validate_worker_certificate(
    artifact: LocalPipelineArtifact | ModelSetArtifact,
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


def _contract(artifact: LocalPipelineArtifact | ModelSetArtifact) -> tuple[list, tuple]:
    """Return ordered artifact inputs and every declared prediction output."""
    if isinstance(artifact, LocalPipelineArtifact):
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
    artifact: LocalPipelineArtifact | ModelSetArtifact,
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
    aliases = {"object": "string", "str": "string", "utf8": "string", "boolean": "bool"}
    return aliases.get(dtype.lower(), dtype.lower())


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


def predict_spark_pyfunc(
    spark: Any,
    frame: Any,
    *,
    model_uri: str,
    artifact: LocalPipelineArtifact | ModelSetArtifact,
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
    _validate_columns(frame, inputs, keys, outputs)
    _validate_model_keys(artifact, keys, inputs, outputs)
    package_path, package = _download_package(model_uri, tracking_uri, registry_uri)
    _validate_package(package, certificate, inputs, outputs)

    import mlflow  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from pyspark.sql import types  # noqa: PLC0415  # ty: ignore[unresolved-import]

    result_type = types.StructType(
        [types.StructField(column.name, _spark_type(column.dtype), True) for column in outputs]
    )
    named = _named_inputs(frame, inputs)
    udf = mlflow.pyfunc.spark_udf(
        spark,
        package_path,
        result_type=result_type,
        env_manager=env_manager,
        params={SPARK_OUTPUT_PARAM: True, SPARK_BATCH_ROWS_PARAM: prediction_batch_rows},
    )
    return _keyed_result(frame, keys, outputs, udf(named))


def _keyed_result(frame: Any, keys: tuple[str, ...], outputs: tuple, result: Any) -> Any:
    """Keep raw record identities separate from expanded model output fields."""
    from pyspark.sql import functions  # noqa: PLC0415  # ty: ignore[unresolved-import]

    # A separate projection avoids collisions with user columns or existing predictions.
    prediction = "__skyulf_pyfunc_result"
    while prediction.casefold() in {key.casefold() for key in keys}:
        prediction += "_"
    scored = frame.select(*(functions.col(_quoted(key)) for key in keys), result.alias(prediction))
    return scored.select(
        *(functions.col(_quoted(key)) for key in keys),
        *(
            functions.col(_quoted(prediction)).getField(column.name).alias(column.name)
            for column in outputs
            if column.name not in keys
        ),
    )
