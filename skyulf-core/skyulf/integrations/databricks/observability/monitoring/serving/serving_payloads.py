"""Parse custom-model inference-table payloads without collecting request rows."""

import importlib
import json
import math
import re
from datetime import UTC, datetime
from typing import Any

REQUEST_KEY = "databricks_request_id"
ROW_KEY = "request_row_index"
_PAYLOAD_COLUMNS = (
    REQUEST_KEY,
    "request_time",
    "status_code",
    "sampling_fraction",
    "execution_duration_ms",
    "request",
    "response",
    "served_entity_id",
    "logging_error_codes",
)
_ENTITY_COLUMNS = ("served_entity_id", "endpoint_name", "entity_name", "entity_version")
_SCALAR_TYPE = re.compile(
    r"(?:string|boolean|bool|byte|tinyint|short|smallint|int|integer|long|bigint|"
    r"float|double|date|timestamp|decimal\([1-9][0-9]?,[0-9]{1,2}\))",
    re.IGNORECASE,
)


def _spark_functions() -> Any:
    """Load optional Spark only when the serving parser is called."""
    return importlib.import_module("pyspark.sql.functions")


def _require_columns(frame: Any, names: tuple[str, ...], source: str) -> None:
    """Reject inference tables whose documented identity fields are unavailable."""
    missing = set(names).difference(frame.columns)
    if missing:
        raise ValueError(f"{source} missing required columns: {', '.join(sorted(missing))}")


def _exists(frame: Any) -> bool:
    """Check a distributed predicate using at most one Spark row."""
    return bool(frame.limit(1).count())


def _identity(payloads: Any, entities: Any, endpoint: str, model: str, version: str) -> Any:
    """Join actual served entities while rejecting ambiguous dimension history."""
    fn = _spark_functions()
    ids = payloads.select("served_entity_id").distinct()
    if _exists(ids.filter(fn.col("served_entity_id").isNull())):
        raise ValueError("served entity identity is missing")
    relevant = entities.select(*_ENTITY_COLUMNS).join(ids, "served_entity_id", "inner")
    if _exists(
        relevant.filter(
            fn.col("endpoint_name").isNull()
            | fn.col("entity_name").isNull()
            | fn.col("entity_version").isNull()
        )
    ):
        raise ValueError("served entity mapping is incomplete")
    if _exists(
        ids.join(relevant.select("served_entity_id").distinct(), "served_entity_id", "left_anti")
    ):
        raise ValueError("served entity mapping is missing")
    conflicts = relevant.groupBy("served_entity_id").agg(
        fn.countDistinct(fn.struct("endpoint_name", "entity_name", "entity_version")).alias(
            "variants"
        )
    )
    if _exists(conflicts.filter(fn.col("variants") > 1)):
        raise ValueError("conflicting served entity mapping")
    selected = (
        relevant.filter(
            (fn.col("endpoint_name") == endpoint)
            & (fn.col("entity_name") == model)
            & (fn.col("entity_version") == str(version))
        )
        .select("served_entity_id")
        .distinct()
    )
    return payloads.join(selected, "served_entity_id", "inner")


def _deduplicate(payloads: Any) -> Any:
    """Keep exact delivery retries once and fail on differing request copies."""
    fn = _spark_functions()
    fingerprint = fn.sha2(fn.to_json(fn.struct(*_PAYLOAD_COLUMNS)), 256)
    copies = payloads.withColumn("_payload_fingerprint", fingerprint)
    variants = copies.groupBy(REQUEST_KEY).agg(
        fn.countDistinct("_payload_fingerprint").alias("variants")
    )
    if _exists(variants.filter(fn.col("variants") > 1)):
        raise ValueError("conflicting copies of Databricks request ID")
    return copies.drop("_payload_fingerprint").dropDuplicates([REQUEST_KEY])


def _check_capture(payloads: Any) -> None:
    """Reject missing IDs, partial capture, and telemetry sampling."""
    fn = _spark_functions()
    if _exists(payloads.filter(fn.col(REQUEST_KEY).isNull() | fn.col("served_entity_id").isNull())):
        raise ValueError("request or served entity identity is missing")
    if _exists(
        payloads.filter(fn.col("sampling_fraction").isNull() | (fn.col("sampling_fraction") != 1.0))
    ):
        raise ValueError("sampling prevents complete serving observation")
    if _exists(
        payloads.filter(
            fn.col("logging_error_codes").isNotNull() & (fn.size("logging_error_codes") > 0)
        )
    ):
        raise ValueError("logging errors prevent complete serving observation")
    if _exists(payloads.filter(fn.col("request_time").isNull())):
        raise ValueError("serving request time is missing")
    if _exists(payloads.filter(fn.col("status_code").isNull())):
        raise ValueError("serving request metadata is incomplete")


def _scalar_record(record: Any) -> bool:
    """Retain JSON primitives without accepting nested values or nonfinite numbers."""
    if not isinstance(record, dict):
        return False
    return all(
        not isinstance(value, (dict, list))
        and (not isinstance(value, float) or math.isfinite(value))
        for value in record.values()
    )


def _scalar_envelope(raw: str | None, field: str) -> bool:
    """Inspect one JSON document on its executor before Spark coerces map values."""
    try:
        document = json.loads(raw) if raw is not None else None
    except (TypeError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    records = document.get(field)
    return isinstance(records, list) and all(_scalar_record(record) for record in records)


def _check_scalar_envelopes(successes: Any) -> None:
    """Keep strict shape checks distributed; only the existence probe reaches the driver."""
    fn = _spark_functions()
    scalar = fn.udf(_scalar_envelope, "boolean")
    for source, field in (("request", "dataframe_records"), ("response", "predictions")):
        if _exists(successes.filter(~scalar(fn.col(source), fn.lit(field)))):
            raise ValueError(f"{source} {field} must contain finite scalar records")


def _parse_envelopes(successes: Any) -> Any:
    """Accept only named dataframe records and record-shaped predictions."""
    fn = _spark_functions()
    request_schema = "STRUCT<dataframe_records: ARRAY<MAP<STRING, STRING>>>"
    response_schema = "STRUCT<predictions: ARRAY<MAP<STRING, STRING>>>"
    parsed = successes.withColumn(
        "_inputs", fn.from_json("request", request_schema).getField("dataframe_records")
    )
    parsed = parsed.withColumn(
        "_outputs", fn.from_json("response", response_schema).getField("predictions")
    )
    if _exists(parsed.filter(fn.col("_inputs").isNull() | (fn.size("_inputs") == 0))):
        raise ValueError("request must contain nonempty dataframe_records")
    if _exists(parsed.filter(fn.col("_outputs").isNull())):
        raise ValueError("response predictions must be record-oriented")
    if _exists(parsed.filter(fn.size("_inputs") != fn.size("_outputs"))):
        raise ValueError("request and prediction row count mismatch")
    if _exists(
        parsed.filter(
            fn.expr("exists(_inputs, x -> x is null) or exists(_outputs, x -> x is null)")
        )
    ):
        raise ValueError("request and predictions must contain records")
    _check_scalar_envelopes(successes)
    return parsed


def _expand(parsed: Any, field: str) -> Any:
    """Explode rows with one stable ordinal per Databricks request."""
    fn = _spark_functions()
    return parsed.select(REQUEST_KEY, fn.posexplode(fn.col(field)).alias(ROW_KEY, "_record"))


def _typed_records(frame: Any, columns: tuple[tuple[str, str], ...], prefix: str = "") -> Any:
    """Cast scalar transport values and detect missing or invalid values."""
    fn = _spark_functions()
    selected = frame
    for name, dtype in columns:
        if not name.startswith(prefix):
            raise ValueError(f"output column {name!r} lacks prefix {prefix!r}")
        output = name.removeprefix(prefix)
        if not output or output in (REQUEST_KEY, ROW_KEY):
            raise ValueError(f"invalid output column {name!r}")
        if not _SCALAR_TYPE.fullmatch(dtype):
            raise ValueError(f"unsupported scalar Spark type {dtype!r}")
        source = fn.col("_record").getItem(name)
        cast = source.try_cast(dtype)
        if dtype.lower() in ("boolean", "bool"):
            cast = fn.when(fn.lower(source).isin("true", "false"), cast)
        selected = selected.withColumn(output, cast)
        invalid = (~fn.array_contains(fn.map_keys("_record"), name)) | (
            source.isNotNull() & fn.col(output).isNull()
        )
        if _exists(selected.filter(invalid)):
            raise ValueError(f"invalid or missing payload column {name!r}")
    return selected.select(
        REQUEST_KEY, ROW_KEY, *(name.removeprefix(prefix) for name, _ in columns)
    )


def _summary(
    payloads: Any, predictions: Any
) -> tuple[dict[str, int | float | None], datetime | None]:
    """Collect only bounded request aggregates and the latest event time."""
    fn = _spark_functions()
    total = payloads.agg(
        fn.count("*").alias("requests"),
        fn.sum(fn.when(fn.col("status_code").between(200, 299), 1).otherwise(0)).alias("successes"),
        fn.avg("execution_duration_ms").alias("latency_mean_ms"),
        fn.max("execution_duration_ms").alias("latency_max_ms"),
        fn.max("request_time").alias("observed_at"),
    ).first()
    observed = total.observed_at
    if observed is not None:
        observed = (
            observed.replace(tzinfo=UTC) if observed.tzinfo is None else observed.astimezone(UTC)
        )
    success_count = int(total.successes or 0)
    request_count = int(total.requests)
    return {
        "request_count": request_count,
        "success_count": success_count,
        "error_count": request_count - success_count,
        "latency_mean_ms": float(total.latency_mean_ms)
        if total.latency_mean_ms is not None
        else None,
        "latency_max_ms": int(total.latency_max_ms) if total.latency_max_ms is not None else None,
        "prediction_row_count": predictions.count(),
    }, observed


def parse_serving_payloads(
    payloads: Any,
    entities: Any,
    *,
    endpoint_name: str,
    model_name: str,
    model_version: str,
    input_columns: tuple[tuple[str, str], ...],
    output_columns: tuple[tuple[str, str], ...],
    output_prefix: str,
    start: datetime,
    end: datetime,
) -> tuple[Any, Any, dict[str, int | float | None], datetime | None]:
    """Read one served version into distributed features and saved predictions.

    Only scalar aggregates leave Spark. HTTP failures remain in operational
    evidence, while successful named rows supply model monitoring populations.
    """
    fn = _spark_functions()
    _require_columns(payloads, _PAYLOAD_COLUMNS, "inference payloads")
    _require_columns(entities, _ENTITY_COLUMNS, "served entities")
    if start >= end:
        raise ValueError("serving observation window must have start before end")
    window = payloads.filter(
        fn.col("request_time").isNull()
        | ((fn.col("request_time") >= start) & (fn.col("request_time") < end))
    )
    selected = _identity(window, entities, endpoint_name, model_name, model_version)
    _check_capture(selected)
    selected = _deduplicate(selected)
    successes = selected.filter(fn.col("status_code").between(200, 299))
    parsed = _parse_envelopes(successes)
    current = _typed_records(_expand(parsed, "_inputs"), input_columns)
    predictions = _typed_records(_expand(parsed, "_outputs"), output_columns, output_prefix)
    summary, observed_at = _summary(selected, predictions)
    return current, predictions, summary, observed_at
