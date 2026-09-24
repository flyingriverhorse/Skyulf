# Databricks notebook source
"""Provision and verify the isolated two-run SM-20a Bundle rehearsal."""

import json
from typing import Any

from pyspark.sql import functions as F  # ty: ignore[unresolved-import]

from skyulf.integrations.databricks.delta import table_identity
from skyulf.integrations.mlflow.promotion import alias_resource_id

SCHEMA = "workspace.skyulf_sm24a_20260923"
RECORDS = "workspace.skyulf_nyctaxi_e2e_20260923.taxi_records"
SOURCE = f"{SCHEMA}.skyulf_sm20a_r1_source"
TARGET = f"{SCHEMA}.skyulf_sm20a_r1_predictions"
CONTROL = f"{SCHEMA}.skyulf_sm20a_r1_score_admission"
ALIAS_CONTROL = f"{SCHEMA}.skyulf_sm20a_r1_alias_admission"
MODEL = f"{SCHEMA}.skyulf_sm20a_r1_model"
FEATURES = ("trip_distance", "trip_duration_minutes", "pickup_hour", "pickup_weekday")


def _rows(spark: Any, first: int, last: int):
    """Select disjoint real taxi keys without changing their raw features."""
    return (
        spark.table(RECORDS)
        .where(F.col("split_index").between(first, last))
        .select("entity_id", *FEATURES)
    )


def setup(spark: Any) -> None:
    """Create new append-only scoring tables and one shared admission row."""
    for table in (SOURCE, TARGET, CONTROL):
        if spark.catalog.tableExists(table):
            raise ValueError(f"Test table already exists: {table}")
    initial = _rows(spark, 3701, 3702)
    if initial.count() != 2:
        raise AssertionError("Expected two real taxi rows for bootstrap.")
    initial.write.format("delta").saveAsTable(SOURCE)
    spark.sql(f"ALTER TABLE {SOURCE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    spark.sql(
        f"CREATE TABLE {TARGET} (entity_id STRING, prediction DOUBLE, run_id STRING, "
        "model_name STRING, model_version STRING) USING DELTA"
    )
    spark.sql(f"CREATE TABLE {CONTROL} (target_id STRING, owner STRING) USING DELTA")
    spark.sql(
        f"INSERT INTO {CONTROL} SELECT :target_id, CAST(NULL AS STRING)",
        args={"target_id": table_identity(spark, TARGET)},
    ).collect()
    print(json.dumps({"source": SOURCE, "initial_rows": 2, "target": TARGET}))


def append(spark: Any) -> None:
    """Add exactly two unseen keys after the first Bundle scoring job."""
    if spark.table(SOURCE).count() != 2 or spark.table(TARGET).count() != 2:
        raise AssertionError("First scoring run must commit exactly two predictions.")
    incoming = _rows(spark, 3703, 3704)
    if incoming.count() != 2:
        raise AssertionError("Expected two new real taxi rows.")
    incoming.write.format("delta").mode("append").saveAsTable(SOURCE)
    print(json.dumps({"source_rows": spark.table(SOURCE).count()}))


def bootstrap_alias(spark: Any) -> None:
    """Explicitly establish the first champion and its separate shared admission."""
    import mlflow  # noqa: PLC0415 - test-only Databricks MLflow boundary

    if spark.catalog.tableExists(ALIAS_CONTROL):
        raise ValueError("Test alias admission table already exists.")
    spark.sql(f"CREATE TABLE {ALIAS_CONTROL} (target_id STRING, owner STRING) USING DELTA")
    spark.sql(
        f"INSERT INTO {ALIAS_CONTROL} SELECT :target_id, CAST(NULL AS STRING)",
        args={"target_id": alias_resource_id(MODEL)},
    ).collect()
    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    client.set_registered_model_alias(MODEL, "champion", "1")
    if str(client.get_model_version_by_alias(MODEL, "champion").version) != "1":
        raise AssertionError("Initial champion alias must point to version 1.")
    print(json.dumps({"model": MODEL, "champion_version": "1"}))


def verify(spark: Any) -> None:
    """Confirm two commits produced four unique, pinned-model predictions."""
    source = spark.table(SOURCE)
    target = spark.table(TARGET)
    if source.count() != 4 or target.count() != 4:
        raise AssertionError("Second run must append only two new predictions.")
    if target.select("entity_id").distinct().count() != 4:
        raise AssertionError("Prediction keys must remain unique.")
    if target.select("model_name", "model_version").distinct().count() != 1:
        raise AssertionError("Both runs must pin the same model identity.")
    identity = target.select("model_name", "model_version").first()
    if identity.model_name != MODEL or identity.model_version != "1":
        raise AssertionError("Predictions must use the configured concrete UC model version.")
    if target.select("run_id").distinct().count() != 2:
        raise AssertionError("The two source versions need distinct commit receipts.")
    history = spark.sql(f"DESCRIBE HISTORY {TARGET}").select("version", "userMetadata").collect()
    receipts = [
        (row.version, json.loads(row.userMetadata))
        for row in history
        if row.userMetadata and '"skyulf_mode": "incremental_append"' in row.userMetadata
    ]
    if len(receipts) != 2 or sorted(receipt["input_count"] for _, receipt in receipts) != [2, 2]:
        raise AssertionError("Expected two committed 2-row incremental receipts.")
    if max(row.version for row in history) != max(version for version, _ in receipts):
        raise AssertionError("The no-input replay must not create another target commit.")
    print(json.dumps({"source_rows": 4, "prediction_rows": 4, "receipts": receipts}))


def verify_alias(spark: Any) -> None:
    """Confirm promotion changed aliases without rewriting historical predictions."""
    import mlflow  # noqa: PLC0415 - test-only Databricks MLflow boundary

    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    aliases = {
        str(alias): str(version)
        for alias, version in client.get_registered_model(MODEL).aliases.items()
    }
    if aliases.get("champion") != "2" or aliases.get("previous_champion") != "1":
        raise AssertionError("Promotion aliases did not reflect versions 2 and 1.")
    if "challenger" in aliases:
        raise AssertionError("Promoted candidate must no longer hold @challenger.")
    prediction_versions = [
        row.model_version
        for row in spark.table(TARGET).select("model_version").distinct().collect()
    ]
    if prediction_versions != ["1"]:
        raise AssertionError("Existing predictions must retain their pinned model version.")
    print(json.dumps({"aliases": aliases, "prediction_versions": prediction_versions}))


if __name__ == "__main__":
    phase = globals()["dbutils"].widgets.get("phase")
    {
        "setup": setup,
        "append": append,
        "bootstrap_alias": bootstrap_alias,
        "verify": verify,
        "verify_alias": verify_alias,
    }[phase](globals()["spark"])
