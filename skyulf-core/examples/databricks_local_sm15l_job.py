# Databricks notebook source
"""Isolated SM-15L UC Delta two-job publication rehearsal."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from typing import Any, Literal

import numpy as np

from skyulf.integrations.databricks import (
    BatchSpec,
    InputSource,
    LocalSourceSpec,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    prepare_local_workflow,
    run_local_batch,
    score_local_source,
)
from skyulf.integrations.databricks.admission import BatchConflictError
from skyulf.integrations.databricks.delta import history, table_identity
from skyulf.integrations.databricks.delta_admission import DeltaTableAdmission

SCHEMA = "workspace.skyulf_sm24a_20260923"
BASE_SOURCE = f"{SCHEMA}.skyulf_sm24a_score_source"
SOURCE = f"{SCHEMA}.skyulf_sm15l_source_r1"
TARGET = f"{SCHEMA}.skyulf_sm15l_predictions_r1"
CONTROL = f"{SCHEMA}.skyulf_sm15l_admission_r1"
MODELS = {
    "pandas": ("workspace.skyulf_sm24a_20260923.skyulf_sm24a_metrics_r1", "1", ("x", "z")),
    "polars": ("workspace.skyulf_sm24a_20260923.skyulf_sm24a_metrics_r2", "1", ("x", "z", "city")),
}
PERIODS = {
    "january": (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)),
    "february": (datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
}


def _source_spec(month: str, source_version: int, columns: tuple[str, ...]) -> LocalSourceSpec:
    """Bind a narrow, finite read to one existing Delta snapshot."""
    start, end = PERIODS[month]
    return LocalSourceSpec(
        table=SOURCE,
        version=source_version,
        period_start=start,
        period_end=end,
        record_key_columns=("entity_id",),
        input_columns=columns,
        max_rows=100,
        max_bytes=4_000_000,
    )


def _prepared(
    month: str, engine: Literal["pandas", "polars"], source_version: int
) -> tuple[Any, LocalSourceSpec]:
    """Resolve a concrete UC model version and its fitted local engine."""
    name, model_version, columns = MODELS[engine]
    source = _source_spec(month, source_version, columns)
    config = LocalWorkflowConfig(
        runtime="databricks",
        engine=engine,
        source=InputSource(
            kind="uc_table",
            table=SOURCE,
            version=source_version,
            max_rows=source.max_rows,
            max_bytes=source.max_bytes,
        ),
        model=ModelSelection(
            kind="local_pipeline",
            name=name,
            version=model_version,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
        ),
        sink=OutputSink(kind="uc_delta", table=TARGET),
    )
    return prepare_local_workflow(config), source


def _request(
    source: LocalSourceSpec, prepared: Any, *, run_id: str, expected_target_version: int
) -> BatchSpec:
    """Make the source, model and expected target state explicit."""
    digest = prepared.preflight.model_digest
    if not digest:
        raise ValueError("Prepared model has no digest.")
    name = prepared.config.model.name
    model_version = prepared.preflight.model_version
    if not name or not model_version:
        raise ValueError("Prepared model has no concrete registry identity.")
    return BatchSpec(
        period_start=source.period_start,
        period_end=source.period_end,
        as_of=datetime.now(UTC) + timedelta(minutes=2),
        record_key_columns=source.record_key_columns,
        output_table=TARGET,
        model_name=name,
        model_version=model_version,
        source_version=source.version,
        code_version=version("skyulf-core"),
        run_id=run_id,
        model_digest=digest,
        expected_target_version=expected_target_version,
        mode="local_pipeline",
    )


def _verify_month(
    spark: Any, source: LocalSourceSpec, prepared: Any, run_id: str
) -> dict[str, Any]:
    """Compare every persisted prediction to the direct local score by business key."""
    gold = score_local_source(spark, source, prepared).predictions.set_index("entity_id")
    from pyspark.sql import functions as F

    rows = (
        spark.table(TARGET)
        .where(
            (F.col("event_time") >= F.lit(source.period_start))
            & (F.col("event_time") < F.lit(source.period_end))
        )
        .select("entity_id", "prediction", "run_id")
        .orderBy("entity_id")
        .collect()
    )
    actual_keys = [row["entity_id"] for row in rows]
    expected_keys = sorted(gold.index.tolist())
    if actual_keys != expected_keys or any(row["run_id"] != run_id for row in rows):
        raise AssertionError("Persisted keys or run metadata differ from the local gold rows.")
    np.testing.assert_allclose(
        [row["prediction"] for row in rows],
        gold.loc[actual_keys, "prediction"].to_numpy(),
        rtol=0,
        atol=1e-9,
    )
    return {"rows": len(rows), "keys": actual_keys, "run_id": run_id}


def january(spark: Any) -> dict[str, Any]:
    """Create only test-owned tables and publish January with the pandas model."""
    for table in (SOURCE, TARGET, CONTROL):
        if spark.catalog.tableExists(table):
            raise ValueError(f"Test table already exists: {table}")
    from pyspark.sql import functions as F

    base = spark.read.format("delta").option("versionAsOf", 0).table(BASE_SOURCE)
    month_start, month_end = PERIODS["january"]
    base.where(
        (F.col("event_time") >= F.lit(month_start)) & (F.col("event_time") < F.lit(month_end))
    ).write.format("delta").saveAsTable(SOURCE)
    spark.sql(
        f"CREATE TABLE {TARGET} (entity_id STRING, event_time TIMESTAMP, prediction DOUBLE, "
        "run_id STRING, model_name STRING, model_version STRING) USING DELTA"
    )
    spark.sql(f"CREATE TABLE {CONTROL} (target_id STRING, owner STRING) USING DELTA")
    target_id = table_identity(spark, TARGET)
    spark.sql(
        f"INSERT INTO {CONTROL} SELECT :target_id, CAST(NULL AS STRING)",
        args={"target_id": target_id},
    ).collect()
    prepared, source = _prepared("january", "pandas", 0)
    request = _request(source, prepared, run_id="sm15l-january-r1", expected_target_version=0)
    result = run_local_batch(
        spark, source, prepared, request, admission=DeltaTableAdmission(spark, CONTROL)
    )
    verified = _verify_month(spark, source, prepared, request.run_id)
    if result.output_count != 80 or verified["rows"] != 80:
        raise AssertionError("January must publish exactly 80 keyed predictions.")
    return {
        "stage": "january",
        "source_version": 0,
        "target_version": result.commit_version,
        "model_name": request.model_name,
        "model_version": request.model_version,
        "model_digest": request.model_digest,
        "verified_rows": verified["rows"],
        "replayed": result.replayed,
    }


def february(spark: Any) -> dict[str, Any]:
    """Append February source rows and publish only February with the Polars model."""
    from pyspark.sql import functions as F

    latest = int(history(spark, SOURCE).select("version").first()["version"])
    if latest != 0:
        raise ValueError("February stage requires the original January-only source.")
    january_before = (
        spark.table(TARGET)
        .where(
            (F.col("event_time") >= F.lit(PERIODS["january"][0]))
            & (F.col("event_time") < F.lit(PERIODS["january"][1]))
        )
        .orderBy("entity_id")
        .collect()
    )
    if len(january_before) != 80:
        raise ValueError("January must be committed before February starts.")
    base = spark.read.format("delta").option("versionAsOf", 0).table(BASE_SOURCE)
    month_start, month_end = PERIODS["february"]
    base.where(
        (F.col("event_time") >= F.lit(month_start)) & (F.col("event_time") < F.lit(month_end))
    ).write.format("delta").mode("append").saveAsTable(SOURCE)
    prepared, source = _prepared("february", "polars", 1)
    request = _request(source, prepared, run_id="sm15l-february-r1", expected_target_version=1)
    admission = DeltaTableAdmission(spark, CONTROL)
    result = run_local_batch(spark, source, prepared, request, admission=admission)
    verified = _verify_month(spark, source, prepared, request.run_id)
    replay = run_local_batch(spark, source, prepared, request, admission=admission)
    january_after = (
        spark.table(TARGET)
        .where(
            (F.col("event_time") >= F.lit(PERIODS["january"][0]))
            & (F.col("event_time") < F.lit(PERIODS["january"][1]))
        )
        .orderBy("entity_id")
        .collect()
    )
    if january_after != january_before:
        raise AssertionError("February publication changed January predictions or metadata.")
    if not replay.replayed or replay.commit_version != result.commit_version:
        raise AssertionError("Replay must reuse the committed February receipt.")
    try:
        run_local_batch(
            spark,
            source,
            prepared,
            replace(request, run_id="sm15l-stale-r1", expected_target_version=1),
            admission=admission,
        )
    except BatchConflictError:
        stale_rejected = True
    else:
        stale_rejected = False
    if not stale_rejected or spark.table(TARGET).count() != 160:
        raise AssertionError("Stale publication changed the target or was not rejected.")
    return {
        "stage": "february",
        "source_version": 1,
        "target_version": result.commit_version,
        "model_name": request.model_name,
        "model_version": request.model_version,
        "model_digest": request.model_digest,
        "verified_rows": verified["rows"],
        "january_preserved": len(january_after),
        "replay_version": replay.commit_version,
        "stale_rejected": stale_rejected,
    }


runtime_dbutils: Any = globals().get("dbutils")
runtime_spark: Any = globals().get("spark")
if runtime_dbutils is not None and runtime_spark is not None:
    runtime_dbutils.widgets.text("stage", "january")
    stage = runtime_dbutils.widgets.get("stage")
    if stage == "january":
        result = january(runtime_spark)
    elif stage == "february":
        result = february(runtime_spark)
    else:
        raise ValueError(f"Unknown SM-15L stage: {stage}")
    runtime_dbutils.notebook.exit(json.dumps(result))
