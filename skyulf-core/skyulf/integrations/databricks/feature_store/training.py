"""Retain native lookup lineage across bounded training and separate job tasks."""

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from .config import FeatureTrainingSpec
from .lifecycle_config import (
    binding_json,
    deserialize_feature_spec,
    parse_feature_binding,
    serialize_feature_spec,
)
from .runtime import _client, _validate_input, create_feature_training_set


def lookup_controls(spec: Any) -> tuple[str, ...]:
    """Include only explicitly declared lookup identifiers in the source projection."""
    if spec.feature_lookup_json is None:
        return ()
    lookup = deserialize_feature_spec(json.loads(spec.feature_lookup_json))
    return tuple(
        dict.fromkeys(
            name for item in lookup.lookups for name in (*item.lookup_key, *item.timestamp_columns)
        )
    )


def _declared_lookup(value: Any) -> FeatureTrainingSpec:
    """Parse bounded training declarations before deriving logging exclusions."""
    if not isinstance(value, str) or len(value) > 64 * 1024:
        raise ValueError("feature_lookup_json must be a bounded JSON string.")
    try:
        return deserialize_feature_spec(json.loads(value))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid feature_lookup_json.") from exc


def training_lookup(spec: Any) -> FeatureTrainingSpec | None:
    """Derive logging exclusions from the same source and raw model column contract."""
    if spec.feature_lookup_json is None:
        if spec.feature_binding_json is not None:
            raise ValueError("feature_binding_json requires feature_lookup_json.")
        return None
    lookup = _declared_lookup(spec.feature_lookup_json)
    if lookup.label != spec.target_column or not set(lookup.feature_names).issubset(
        spec.input_columns
    ):
        raise ValueError("Feature lookup label or inputs differ from the training contract.")
    exclude = tuple(
        name
        for name in spec.source_columns
        if name not in {*spec.input_columns, spec.target_column}
    )
    result = replace(lookup, exclude_columns=exclude)
    if spec.feature_binding_json is not None:
        binding = parse_feature_binding(spec.feature_binding_json)
        if deserialize_feature_spec(binding["lookup_spec"]) != result:
            raise ValueError("Pinned feature binding differs from the training lookup.")
    return result


def pin_training_lookup(spark: Any, spec: Any) -> Any:
    """Pin feature Delta identities once so later notebook tasks reject changed history."""
    lookup = training_lookup(spec)
    if lookup is None:
        return spec
    from .snapshots import snapshot_tables, validate_snapshots  # noqa: PLC0415

    if spec.feature_binding_json is not None:
        validate_snapshots(
            spark,
            parse_feature_binding(spec.feature_binding_json)["lookup_evidence"]["feature_tables"],
        )
        return spec
    binding = {
        "version": 1,
        "lookup_spec": serialize_feature_spec(lookup),
        "lookup_evidence": {
            "policy": "training_snapshot",
            "feature_tables": snapshot_tables(spark, lookup),
        },
    }
    return replace(spec, feature_binding_json=binding_json(binding))


def pin_fit_lookup(spark: Any, spec: Any, engine: str) -> Any:
    """Enforce admitted pandas fitting even when callers bypass workflow configuration."""
    if spec.feature_lookup_json is not None and engine != "pandas":
        raise ValueError("Native feature lookup training requires engine=pandas.")
    return pin_training_lookup(spark, spec)


def create_checked_training_set(
    source: Any,
    lookup: FeatureTrainingSpec,
    binding: dict[str, Any],
    *,
    spark: Any,
    client: Any = None,
    lookup_factory: Callable[..., Any] | None = None,
) -> Any:
    """Check feature keys and history before constructing a native SDK training set.

    Feature tables must have an exclusive writer during this operation and its
    downstream action. The SDK does not expose Delta versionAsOf for lookups;
    before/after guards detect changes but cannot lock out concurrent writes.
    """
    from .snapshots import validate_snapshots  # noqa: PLC0415
    from .table_checks import validate_feature_tables  # noqa: PLC0415

    binding = parse_feature_binding(binding)
    expected = deserialize_feature_spec(binding["lookup_spec"])
    if lookup.lookups != expected.lookups:
        raise ValueError("Native training lookup differs from its pinned feature binding.")
    _validate_input(source, lookup, training=True, allow_feature_overrides=False)
    records = binding["lookup_evidence"]["feature_tables"]
    validate_snapshots(spark, records)
    client = _client(client)
    validate_feature_tables(spark, source, lookup, client, records)
    result = create_feature_training_set(
        source, lookup, client=client, lookup_factory=lookup_factory
    )
    validate_snapshots(spark, records)
    return result


def _training_source(source: Any, spec: Any, lookup: FeatureTrainingSpec) -> Any:
    """Reject overrides on the full source before selecting the raw lookup columns."""
    features = {name.casefold() for name in lookup.feature_names}
    if any(name.casefold() in features for name in source.columns):
        raise ValueError("Training source columns would override feature table lookups.")
    return source.select(
        *[name for name in spec.source_columns if name not in lookup.feature_names]
    )


def enrich_training_source(spark: Any, source: Any, spec: Any) -> Any:
    """Retain split metadata while the SDK adds historical model inputs in Spark."""
    lookup = training_lookup(spec)
    if lookup is None:
        return source
    binding = parse_feature_binding(spec.feature_binding_json)
    source = _training_source(source, spec, lookup)
    return create_checked_training_set(
        source, replace(lookup, exclude_columns=()), binding, spark=spark
    ).load_df()


def logging_training_set(spark: Any, spec: Any) -> Any:
    """Reconstruct excluded native lineage from the same pinned base and feature tables."""
    lookup = training_lookup(spec)
    if lookup is None:
        raise ValueError("Native logging requires a feature lookup.")
    binding = parse_feature_binding(spec.feature_binding_json)
    source = spark.read.format("delta").option("versionAsOf", spec.version).table(spec.table)
    source = _training_source(source, spec, lookup)
    return create_checked_training_set(source, lookup, binding, spark=spark)


def validate_training_frame(frame: Any, spec: Any) -> None:
    """Bind saved materialized rows to the exact feature evidence frozen at prepare."""
    actual = frame.attrs.get("feature_lookup")
    expected = spec.feature_binding_json
    if expected is None and actual is None:
        return
    if actual is None or expected is None or binding_json(actual) != expected:
        raise ValueError("Prepared training frame differs from the pinned feature binding.")


def retain_training_binding(spark: Any, frame: Any, spec: Any) -> None:
    """Check feature tables after materialization and attach durable provenance to the rows."""
    if spec.feature_binding_json is None:
        return
    from .snapshots import validate_snapshots  # noqa: PLC0415

    binding = parse_feature_binding(spec.feature_binding_json)
    validate_snapshots(spark, binding["lookup_evidence"]["feature_tables"])
    frame.attrs["feature_lookup"] = binding


def feature_log_options(spark: Any, spec: Any) -> dict[str, Any]:
    """Keep ordinary model logging unchanged when native lookup is not enabled."""
    return {} if spec.feature_lookup_json is None else {"spark": spark, "spec": spec}


def log_training_feature_model(
    path: Any, *, spark: Any, spec: Any, run_id: str, tracking_uri: str
) -> str:
    """Package fitted bytes with the frozen native lookup used to read training rows."""
    from ...mlflow.models.feature_model import (  # noqa: PLC0415 - optional MLflow boundary
        log_feature_pipeline_model as log_local_feature_model,
    )
    from .snapshots import validate_snapshots  # noqa: PLC0415

    binding = parse_feature_binding(spec.feature_binding_json)
    lookup = training_lookup(spec)
    native = logging_training_set(spark, spec)
    assert lookup is not None
    uri = log_local_feature_model(
        path,
        training_set=native,
        lookup_spec=lookup,
        lookup_binding=binding,
        run_id=run_id,
        artifact_path="model",
        tracking_uri=tracking_uri,
    )
    validate_snapshots(spark, binding["lookup_evidence"]["feature_tables"])
    return uri
