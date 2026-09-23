# Databricks notebook source
"""Isolated two-run, date-independent local incremental Delta validation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from skyulf.integrations.databricks import (
    InputSource,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    prepare_local_workflow,
    run_incremental_local_batch,
)
from skyulf.integrations.databricks.delta import table_identity
from skyulf.integrations.databricks.delta_admission import DeltaTableAdmission

SCHEMA = "workspace.skyulf_sm24a_20260923"
BASE_SOURCE = f"{SCHEMA}.skyulf_sm24a_score_source"
SOURCE = f"{SCHEMA}.skyulf_sm15i_source_r1"
TARGET = f"{SCHEMA}.skyulf_sm15i_predictions_r1"
CONTROL = f"{SCHEMA}.skyulf_sm15i_admission_r1"
MODEL = f"{SCHEMA}.skyulf_sm24a_metrics_r1"


def _prepare() -> Any:
    """Load the same pinned fitted pandas model for both scheduled score runs."""
    return prepare_local_workflow(
        LocalWorkflowConfig(
            runtime="databricks",
            engine="pandas",
            source=InputSource(
                kind="uc_table",
                table=SOURCE,
                read_mode="incremental",
                max_rows=200,
                max_bytes=4_000_000,
            ),
            model=ModelSelection(
                kind="local_pipeline",
                name=MODEL,
                version="1",
                tracking_uri="databricks",
                registry_uri="databricks-uc",
            ),
            sink=OutputSink(kind="uc_delta", table=TARGET),
        )
    )


def _verify(spark: Any, prepared: Any, expected_count: int) -> None:
    """Check every persisted value against direct local prediction by stable key."""
    source = spark.table(SOURCE).select("entity_id", "x", "z").orderBy("entity_id").toPandas()
    expected = prepared.predict(source.loc[:, ["x", "z"]])
    actual = spark.table(TARGET).select("entity_id", "prediction").orderBy("entity_id").toPandas()
    if len(source) != expected_count or len(actual) != expected_count:
        raise AssertionError("Source and target counts differ from the expected stage.")
    if actual["entity_id"].tolist() != source["entity_id"].tolist():
        raise AssertionError("Prediction keys differ from source keys.")
    np.testing.assert_allclose(
        actual["prediction"].to_numpy(),
        expected["prediction"].to_numpy(),
        rtol=0,
        atol=1e-9,
    )


def bootstrap(spark: Any) -> dict[str, Any]:
    """Create isolated tables and score the initial snapshot without a date filter."""
    for table in (SOURCE, TARGET, CONTROL):
        if spark.catalog.tableExists(table):
            raise ValueError(f"Test table already exists: {table}")
    base = spark.table(BASE_SOURCE).select("entity_id", "x", "z")
    base.orderBy("entity_id").limit(80).write.format("delta").saveAsTable(SOURCE)
    spark.sql(f"ALTER TABLE {SOURCE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    spark.sql(
        f"CREATE TABLE {TARGET} (entity_id STRING, prediction DOUBLE, "
        "run_id STRING, model_name STRING, "
        "model_version STRING) USING DELTA"
    )
    spark.sql(f"CREATE TABLE {CONTROL} (target_id STRING, owner STRING) USING DELTA")
    spark.sql(
        f"INSERT INTO {CONTROL} SELECT :target_id, CAST(NULL AS STRING)",
        args={"target_id": table_identity(spark, TARGET)},
    ).collect()
    prepared = _prepare()
    result = run_incremental_local_batch(
        spark,
        prepared,
        row_keys=("entity_id",),
        admission=DeltaTableAdmission(spark, CONTROL),
    )
    _verify(spark, prepared, 80)
    if result.input_count != 80 or result.noop:
        raise AssertionError("Initial snapshot was not scored once.")
    return {
        "stage": "bootstrap",
        "rows": result.input_count,
        "source_end_version": result.source_end_version,
        "target_version": result.commit_version,
        "model_digest": result.manifest["model_digest"] if result.manifest else None,
    }


def append(spark: Any) -> dict[str, Any]:
    """Append source rows, score only those inserts, then prove a no-op retry."""
    if not all(spark.catalog.tableExists(table) for table in (SOURCE, TARGET, CONTROL)):
        raise ValueError("Bootstrap tables are missing.")
    before = spark.table(TARGET).orderBy("entity_id").collect()
    if len(before) != 80:
        raise ValueError("Bootstrap must publish exactly 80 predictions.")
    remaining = (
        spark.table(BASE_SOURCE)
        .select("entity_id", "x", "z")
        .join(spark.table(SOURCE).select("entity_id"), on="entity_id", how="left_anti")
    )
    if remaining.count() != 80:
        raise ValueError("Expected exactly 80 new fixture records.")
    remaining.write.format("delta").mode("append").saveAsTable(SOURCE)
    prepared = _prepare()
    admission = DeltaTableAdmission(spark, CONTROL)
    result = run_incremental_local_batch(
        spark, prepared, row_keys=("entity_id",), admission=admission
    )
    replay = run_incremental_local_batch(
        spark, prepared, row_keys=("entity_id",), admission=admission
    )
    _verify(spark, prepared, 160)
    after = spark.table(TARGET).orderBy("entity_id").collect()
    if [row for row in after if row.entity_id in {old.entity_id for old in before}] != before:
        raise AssertionError("The first run's predictions changed.")
    if result.input_count != 80 or replay.input_count != 0 or not replay.noop:
        raise AssertionError("Incremental append or no-op retry changed membership.")
    if result.commit_version != replay.commit_version:
        raise AssertionError("No-op retry wrote another target version.")
    return {
        "stage": "append",
        "new_rows": result.input_count,
        "total_rows": len(after),
        "source_end_version": result.source_end_version,
        "target_version": result.commit_version,
        "noop_version": replay.commit_version,
    }


runtime_dbutils: Any = globals().get("dbutils")
runtime_spark: Any = globals().get("spark")
if runtime_dbutils is not None and runtime_spark is not None:
    runtime_dbutils.widgets.text("stage", "bootstrap")
    stage = runtime_dbutils.widgets.get("stage")
    if stage == "bootstrap":
        result = bootstrap(runtime_spark)
    elif stage == "append":
        result = append(runtime_spark)
    else:
        raise ValueError(f"Unknown SM-15I stage: {stage}")
    runtime_dbutils.notebook.exit(json.dumps(result))
