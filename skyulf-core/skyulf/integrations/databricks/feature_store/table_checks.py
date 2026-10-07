"""Check Unity Catalog key declarations and actual feature rows before native lookup."""

from importlib import import_module
from typing import Any

from ..shared._contracts import column_name
from .config import FeatureLookupSpec, FeatureTrainingSpec


def _table_keys(
    metadata: Any, lookup: FeatureLookupSpec
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Require real primary and TIMESERIES keys instead of inferring them from data."""
    primary = tuple(metadata.primary_keys)
    temporal = tuple(metadata.timestamp_keys or ())
    keys = tuple(key for key in primary if key not in temporal)
    if not keys or len(keys) != len(lookup.lookup_key):
        raise ValueError(f"Feature table {lookup.table_name} primary keys differ from lookup keys.")
    expected = 1 if lookup.timestamp_lookup_key is not None else 0
    if len(temporal) != expected or not set(temporal).issubset(primary):
        raise ValueError(f"Feature table {lookup.table_name} TIMESERIES keys differ from lookup.")
    return keys, temporal


def _check_types(
    source: Any,
    features: Any,
    lookup: FeatureLookupSpec,
    keys: tuple[str, ...],
    temporal: tuple[str, ...],
) -> None:
    """Preserve source/table key types and named features without implicit casts."""
    source_types, feature_types = dict(source.dtypes), dict(features.dtypes)
    if any(name not in feature_types for name in (*keys, *temporal, *lookup.feature_names)):
        raise ValueError(f"Missing declared columns in feature table {lookup.table_name}.")
    for source_key, table_key in zip(lookup.lookup_key, keys, strict=True):
        if source_types.get(source_key) != feature_types[table_key]:
            raise ValueError(f"Feature lookup key type differs for {source_key!r}.")
    if temporal and feature_types[temporal[0]] != lookup.timestamp_type:
        raise ValueError("Feature table TIMESERIES type differs from its lookup timestamp type.")


def _check_rows(features: Any, names: tuple[str, ...], table: str) -> None:
    """Reject unenforced null or duplicate primary keys before a lookup can multiply rows."""
    functions = import_module("pyspark.sql.functions")
    nulls = " OR ".join(f"{column_name(name)} IS NULL" for name in names)
    if features.where(nulls).limit(1).count():
        raise ValueError(f"Feature table {table} contains null primary keys.")
    count_name = "__skyulf_feature_key_count"
    while count_name in names:
        count_name += "_"
    groups = features.groupBy(*names).agg(functions.count("*").alias(count_name))
    if groups.where(functions.col(count_name) > 1).limit(1).count():
        raise ValueError(f"Feature table {table} contains duplicate primary keys.")


def validate_feature_tables(
    spark: Any, source: Any, spec: FeatureTrainingSpec, client: Any, records: list[dict[str, Any]]
) -> None:
    """Inspect UC metadata and full pinned feature keys without driver collection."""
    versions = {record["table_name"]: record["version"] for record in records}
    for lookup in spec.lookups:
        metadata = client.get_table(name=lookup.table_name)
        keys, temporal = _table_keys(metadata, lookup)
        features = (
            spark.read.format("delta")
            .option("versionAsOf", versions[lookup.table_name])
            .table(lookup.table_name)
        )
        _check_types(source, features, lookup, keys, temporal)
        _check_rows(features, (*keys, *temporal), lookup.table_name)
