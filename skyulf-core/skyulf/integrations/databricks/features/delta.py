"""Persist keyed feature corrections without erasing earlier observation history."""

import uuid
from typing import Any

from ..shared._contracts import column_name, table_name
from .joins import validate_feature_frame


def current_version(spark: Any, table: str) -> int:
    """Read only the latest Delta commit number, never feature records."""
    row = spark.sql(f"DESCRIBE HISTORY {table_name(table)} LIMIT 1").select("version").first()
    if row is None:
        raise ValueError(f"Feature table has no Delta history: {table}.")
    return int(row[0])


def read_version(spark: Any, table: str, version: int) -> Any:
    """Require a concrete Delta version for every cross-task data dependency."""
    table_name(table)
    if type(version) is not int or version < 0:
        raise ValueError("Feature source version must be a nonnegative integer.")
    return spark.read.format("delta").option("versionAsOf", version).table(table)


def _merge_rows(spark: Any, frame: Any, table: str, grain: tuple[str, ...]) -> None:
    """Update changed values only; retries preserve unaffected rows and history."""
    view = "skyulf_feature_" + uuid.uuid4().hex
    frame.createOrReplaceTempView(view)
    names = [column_name(name) for name in frame.columns]
    condition = " AND ".join(f"target.{column_name(k)} = source.{column_name(k)}" for k in grain)
    changes = " OR ".join(f"NOT(target.{name} <=> source.{name})" for name in names)
    try:
        spark.sql(
            f"MERGE INTO {table_name(table)} AS target USING `{view}` AS source "
            f"ON {condition} WHEN MATCHED AND ({changes}) THEN UPDATE SET * "
            "WHEN NOT MATCHED THEN INSERT *"
        )
    finally:
        spark.catalog.dropTempView(view)


def publish_feature_table(
    spark: Any,
    frame: Any,
    table: str,
    keys: tuple[str, ...],
    timestamp: str,
) -> dict[str, Any]:
    """Create a CDF-enabled Delta table or upsert a checked compatible feature batch.

    A feature job serializes its own runs. Tables must have one owning producer;
    this API does not implement a distributed lock against unrelated writers.
    Output schemas are explicit and cannot evolve implicitly during a repair.
    """
    table_name(table)
    rows = validate_feature_frame(frame, keys, timestamp)
    for name in frame.columns:
        column_name(name)
    if not spark.catalog.tableExists(table):
        frame.write.format("delta").mode("error").option(
            "delta.enableChangeDataFeed", "true"
        ).saveAsTable(table)
    else:
        before = spark.table(table)
        validate_feature_frame(before, keys, timestamp)
        expected = {f.name: f.dataType for f in before.schema.fields}
        if {f.name: f.dataType for f in frame.schema.fields} != expected:
            raise ValueError(f"Feature table schema differs; use an explicit migration: {table}.")
        properties = spark.sql(f"DESCRIBE DETAIL {table_name(table)}").first()["properties"]
        if properties.get("delta.enableChangeDataFeed") != "true":
            raise ValueError(f"Enable Delta change data feed explicitly before using {table}.")
        _merge_rows(spark, frame, table, (*keys, timestamp))
    return {"table": table, "version": current_version(spark, table), "rows": rows}
