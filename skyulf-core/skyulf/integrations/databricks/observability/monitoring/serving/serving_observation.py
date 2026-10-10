"""Read actual serving events with pinned table evidence and saved model schemas."""

from datetime import datetime
from typing import Any

from skyulf.integrations.mlflow.shared._model_metadata import normalized_dtype

from ..monitoring_config import MonitorConfig
from ..monitoring_sources import observation_window
from .serving_payloads import REQUEST_KEY, ROW_KEY, parse_serving_payloads
from .serving_sources import read_serving_payload_snapshot, verify_serving_payload_snapshot


def observation_keys(config: MonitorConfig, spec: Any) -> tuple[str, ...]:
    """Keep repeat requests separate while preserving the original training key contract."""
    return (REQUEST_KEY, ROW_KEY) if config.serving_endpoint else spec.record_key_columns


def serving_identity(config: MonitorConfig) -> dict:
    """Bind metric semantics to an endpoint and its actual parent release when present."""
    if not config.serving_endpoint:
        return {}
    return {
        "serving_endpoint": config.serving_endpoint,
        "served_model": config.model_set_name or config.model_name,
        "served_version": config.model_set_version or config.model_version,
        "served_branch": config.model_set_branch,
    }


def _spark_dtype(dtype: str) -> str:
    """Map declared fitted scalar types without inferring types from live request values."""
    normalized = normalized_dtype(dtype)
    types = {
        "float32": "float",
        "float64": "double",
        "bool": "boolean",
        "string": "string",
        "int8": "byte",
        "int16": "short",
        "int32": "int",
        "int64": "long",
        "uint8": "short",
        "uint16": "int",
        "uint32": "long",
    }
    if normalized not in types:
        raise ValueError(f"Unsupported serving monitoring scalar dtype: {dtype}.")
    return types[normalized]


def _payload_contract(artifact: Any, config: MonitorConfig) -> tuple[tuple, tuple, str]:
    """Project component inputs and standard outputs without mixing sibling targets."""
    from skyulf.inference.pipeline_scoring import prediction_output_schema  # noqa: PLC0415

    manifest = artifact.manifest
    inputs = tuple(
        (name, _spark_dtype(dtype))
        for name, dtype in zip(manifest.input_columns, manifest.input_dtypes, strict=True)
    )
    prefix = f"{config.model_set_branch}__" if config.model_set_branch else ""
    outputs = tuple(
        (prefix + c.name, _spark_dtype(c.dtype)) for c in prediction_output_schema(artifact)
    )
    if config.model_set_branch or artifact.pipeline.config.get("project_scoring") is not None:
        outputs += ((prefix + "scoring_status", "string"),)
    return inputs, outputs, prefix


def read_serving_observation(
    spark: Any,
    config: MonitorConfig,
    version: str,
    keys: tuple[str, ...],
    features: tuple[str, ...],
    *,
    probabilities: int,
    as_of: datetime,
    start: datetime,
    end: datetime,
    artifact: Any,
) -> tuple[Any, Any, dict, datetime | None]:
    """Use request-time windows on the latest pinned logging snapshot at the cutoff."""
    observation_window(as_of, start, end)
    if (
        not config.serving_endpoint
        or version != config.model_version
        or keys != (REQUEST_KEY, ROW_KEY)
    ):
        raise ValueError("Serving observation identity differs from the configured model.")
    manifest = artifact.manifest
    expected_probabilities = len(manifest.classes) if manifest.classification_probabilities else 0
    if features != manifest.input_columns or probabilities != expected_probabilities:
        raise ValueError("Serving observation schema differs from the saved model.")
    payloads, capture_evidence = read_serving_payload_snapshot(spark, config.source_table, as_of)
    entities = spark.table("system.serving.served_entities")
    inputs, outputs, prefix = _payload_contract(artifact, config)
    current, predictions, summary, observed = parse_serving_payloads(
        payloads,
        entities,
        endpoint_name=config.serving_endpoint,
        model_name=config.model_set_name or config.model_name,
        model_version=config.model_set_version or version,
        input_columns=inputs,
        output_columns=outputs,
        output_prefix=prefix,
        start=start,
        end=end,
    )
    verify_serving_payload_snapshot(spark, config.source_table, capture_evidence)
    evidence = {
        **capture_evidence,
        "window_basis": "serving_request_timestamp",
        "execution_engine": "spark",
        "serving": serving_identity(config) | summary,
    }
    return current, predictions, evidence, observed
