# Databricks notebook source
"""Prepare one real taxi source and verify the minimal generic Bundle rehearsal."""

import json
from typing import Any

from pyspark.sql import Window  # ty: ignore[unresolved-import]
from pyspark.sql import functions as F  # ty: ignore[unresolved-import]

SCHEMA = "workspace.skyulf_bundle_first_20260924"
SOURCE = f"{SCHEMA}.skyulf_reset_dev_source"
TARGET = f"{SCHEMA}.skyulf_reset_dev_predictions"
FEATURES = ("trip_distance", "trip_duration_minutes", "pickup_hour", "pickup_weekday")
SAMPLE = "samples.nyctaxi.trips"


def _rows(spark: Any) -> Any:
    """Choose reproducible real trips with unique keys and completed labels."""
    raw = spark.table(SAMPLE)
    duration = (
        F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")
    ) / F.lit(60.0)
    clean = (
        raw.withColumn("trip_duration_minutes", duration)
        .where(F.col("fare_amount").between(2.5, 100.0))
        .where(F.col("trip_distance").between(0.1, 30.0))
        .where(F.col("trip_duration_minutes").between(1.0, 120.0))
        .dropna(
            subset=["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount"]
        )
        .withColumn(
            "entity_id",
            F.sha2(
                F.to_json(
                    F.struct(
                        "tpep_pickup_datetime",
                        "tpep_dropoff_datetime",
                        "trip_distance",
                        "fare_amount",
                        "pickup_zip",
                        "dropoff_zip",
                    )
                ),
                256,
            ),
        )
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime").cast("double"))
        .withColumn("pickup_weekday", F.dayofweek("tpep_pickup_datetime").cast("double"))
        .select(
            "entity_id",
            F.col("tpep_pickup_datetime").alias("event_time"),
            F.col("tpep_dropoff_datetime").alias("label_at"),
            F.col("trip_distance").cast("double").alias("trip_distance"),
            F.col("trip_duration_minutes").cast("double").alias("trip_duration_minutes"),
            "pickup_hour",
            "pickup_weekday",
            F.col("fare_amount").cast("double").alias("target"),
        )
        .orderBy("entity_id")
        .limit(650)
        .withColumn("split_index", F.row_number().over(Window.orderBy("entity_id")))
    )
    return clean


def prepare(spark: Any) -> dict[str, Any]:
    """Create only the one CDF-enabled source table; Bundle setup owns outputs."""
    if spark.catalog.tableExists(SOURCE):
        raise ValueError(f"Source already exists: {SOURCE}")
    first = _rows(spark).where(F.col("split_index") <= 600).drop("split_index")
    first.write.format("delta").saveAsTable(SOURCE)
    spark.sql(f"ALTER TABLE {SOURCE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    source = spark.table(SOURCE)
    rows = source.count()
    fit = source.where(F.col("event_time") < F.lit("2016-02-01")).count()
    heldout = source.where(F.col("event_time") >= F.lit("2016-02-01")).count()
    if (
        rows != 600
        or fit < 50
        or heldout < 50
        or source.select("entity_id").distinct().count() != rows
    ):
        raise AssertionError(
            "The first source must have 600 unique trips and both temporal splits."
        )
    version = spark.sql(f"DESCRIBE HISTORY {SOURCE}").first().version
    return {
        "source": SOURCE,
        "rows": rows,
        "fit_rows": fit,
        "heldout_rows": heldout,
        "source_version": version,
    }


def append(spark: Any) -> dict[str, Any]:
    """Add 50 new keys only after the first 600 predictions are committed."""
    if spark.table(SOURCE).count() != 600 or spark.table(TARGET).count() != 600:
        raise AssertionError("Initial scoring must commit 600 rows before the append.")
    later = _rows(spark).where(F.col("split_index") > 600).drop("split_index")
    if later.count() != 50:
        raise AssertionError("Expected 50 deterministic new trips.")
    later.write.format("delta").mode("append").saveAsTable(SOURCE)
    return {"source": SOURCE, "source_rows": spark.table(SOURCE).count()}


def verify(spark: Any) -> dict[str, Any]:
    """Confirm one target contains 600 original and 50 later pinned predictions."""
    source = spark.table(SOURCE)
    target = spark.table(TARGET)
    if source.count() != 650 or target.count() != 650:
        raise AssertionError("Expected 650 source and prediction rows after the second score.")
    if target.select("entity_id").distinct().count() != 650:
        raise AssertionError("Predicted row keys must be unique.")
    identities = target.select("model_name", "model_version").distinct().collect()
    if len(identities) != 1 or identities[0].model_version != "1":
        raise AssertionError("All predictions must use the first pinned UC model version.")
    return {"source_rows": 650, "prediction_rows": 650, "model_version": "1"}


if __name__ == "__main__":
    phase = globals()["dbutils"].widgets.get("phase")
    result = {"prepare": prepare, "append": append, "verify": verify}[phase](globals()["spark"])
    globals()["dbutils"].notebook.exit(json.dumps(result))
