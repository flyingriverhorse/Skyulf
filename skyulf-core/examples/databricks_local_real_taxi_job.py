# Databricks notebook source
"""Train and replay a Skyulf local model on real NYC taxi Delta rows."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow  # ty: ignore[unresolved-import]
import numpy as np
from pyspark.sql import Window  # ty: ignore[unresolved-import]
from pyspark.sql import functions as F  # ty: ignore[unresolved-import]

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks import (
    InputSource,
    LocalSourceSpec,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    evaluate_local_holdout,
    fit_local_workflow,
    prepare_local_workflow,
    read_local_source,
    run_incremental_local_batch,
)
from skyulf.integrations.databricks.delta import table_identity
from skyulf.integrations.databricks.delta_admission import DeltaTableAdmission
from skyulf.integrations.mlflow.local_model import log_local_model
from skyulf.integrations.mlflow.registry import register_model

SCHEMA = "workspace.skyulf_nyctaxi_e2e_20260923"
EXPERIMENT = "/Users/edwardwolfe99@gmail.com/skyulf_nyctaxi_e2e_20260923/experiment"
SAMPLE = "samples.nyctaxi.trips"
RECORDS = f"{SCHEMA}.taxi_records"
TRAINING = f"{SCHEMA}.taxi_training"
SOURCE = f"{SCHEMA}.taxi_score_source"
TARGET = f"{SCHEMA}.taxi_predictions"
CONTROL = f"{SCHEMA}.taxi_admission"
MODEL = f"{SCHEMA}.taxi_fare_skyulf"
FEATURES = (
    "trip_distance",
    "trip_duration_minutes",
    "pickup_hour",
    "pickup_weekday",
    "pickup_zip",
    "dropoff_zip",
)
MODEL_CONFIG: dict[str, Any] = {
    "preprocessing": [
        {
            "name": "fill_numeric",
            "transformer": "SimpleImputer",
            "params": {"columns": list(FEATURES[:4]), "strategy": "median"},
        },
        {
            "name": "scale_numeric",
            "transformer": "StandardScaler",
            "params": {"columns": list(FEATURES[:4])},
        },
        {
            "name": "encode_zips",
            "transformer": "OneHotEncoder",
            "params": {"columns": list(FEATURES[4:]), "handle_unknown": "ignore"},
        },
    ],
    "modeling": {
        "type": "random_forest_regressor",
        "params": {"n_estimators": 40, "max_depth": 9, "random_state": 42, "n_jobs": 1},
    },
}


def _new_tables(spark: Any) -> None:
    """Create isolated real-data tables once, with disjoint training and scoring keys."""
    names = (RECORDS, TRAINING, SOURCE, TARGET, CONTROL)
    existing = [name for name in names if spark.catalog.tableExists(name)]
    if existing:
        raise ValueError(f"This test revision already has tables: {existing}")
    raw_columns = (
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
        "pickup_zip",
        "dropoff_zip",
    )
    source = spark.table(SAMPLE)
    duration = (
        F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")
    ) / F.lit(60.0)
    clean = (
        source.withColumn("trip_duration_minutes", duration)
        .where(F.col("fare_amount").between(2.5, 100.0))
        .where(F.col("trip_distance").between(0.1, 30.0))
        .where(F.col("trip_duration_minutes").between(1.0, 120.0))
        .dropna(subset=list(raw_columns))
        .withColumn("entity_id", F.sha2(F.to_json(F.struct(*raw_columns)), 256))
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime").cast("double"))
        .withColumn("pickup_weekday", F.dayofweek("tpep_pickup_datetime").cast("double"))
        .withColumn("pickup_zip", F.col("pickup_zip").cast("string"))
        .withColumn("dropoff_zip", F.col("dropoff_zip").cast("string"))
        .select("entity_id", "tpep_pickup_datetime", *FEATURES, "fare_amount")
        .orderBy("entity_id")
        .limit(3_800)
        .withColumn("split_index", F.row_number().over(Window.orderBy("entity_id")))
    )
    clean.write.format("delta").saveAsTable(RECORDS)
    records = spark.table(RECORDS)
    if records.count() != 3_800 or records.select("entity_id").distinct().count() != 3_800:
        raise AssertionError("The real taxi copy must contain 3,800 unique trips.")
    records.where(F.col("split_index") <= 3_500).drop("split_index").write.format(
        "delta"
    ).saveAsTable(TRAINING)
    records.where(F.col("split_index").between(3_501, 3_700)).select(
        "entity_id", *FEATURES
    ).write.format("delta").saveAsTable(SOURCE)
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


def _training_frame(spark: Any) -> Any:
    """Read an explicit bounded training snapshot through Skyulf's UC reader."""
    source_version = int(spark.sql(f"DESCRIBE HISTORY {TRAINING}").first().version)
    frame = read_local_source(
        spark,
        LocalSourceSpec(
            table=TRAINING,
            version=source_version,
            period_start=datetime(2016, 1, 1, tzinfo=UTC),
            period_end=datetime(2016, 3, 1, tzinfo=UTC),
            period_column="tpep_pickup_datetime",
            row_keys=("entity_id",),
            input_columns=(*FEATURES, "fare_amount"),
            max_rows=3_500,
            max_bytes=12_000_000,
        ),
    )
    if len(frame) != 3_500:
        raise AssertionError("The training read did not return 3,500 real trips.")
    return frame


def train(spark: Any) -> dict[str, Any]:
    """Fit Skyulf FE/model, log held-out metrics and register one UC version."""
    _new_tables(spark)
    frame = _training_frame(spark)
    train_frame = frame.iloc[:3_000].loc[:, [*FEATURES, "fare_amount"]]
    heldout = frame.iloc[3_000:].loc[:, [*FEATURES, "fare_amount"]]
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT)
    with tempfile.TemporaryDirectory(prefix="skyulf-taxi-") as directory:
        artifact_path = Path(directory) / "taxi_model"
        artifact = fit_local_workflow(
            MODEL_CONFIG,
            SplitDataset(train=train_frame, test=heldout),
            target_column="fare_amount",
            artifact_path=artifact_path,
            max_rows=3_500,
            max_bytes=12_000_000,
        )
        metrics = evaluate_local_holdout(artifact, heldout, target_column="fare_amount")
        if not all(np.isfinite(value) for value in metrics.values()):
            raise AssertionError("Held-out model metrics are not finite.")
        with mlflow.start_run(run_name="skyulf-nyctaxi-train") as run:
            mlflow.set_tags({"skyulf_phase": "train", "dataset": SAMPLE, "validation": "held_out"})
            mlflow.log_params(
                {
                    "source_table": TRAINING,
                    "training_rows": 3_000,
                    "heldout_rows": 500,
                    "fit_engine": "pandas",
                    "target_column": "fare_amount",
                    "model_type": "random_forest_regressor",
                    "model_digest": artifact.manifest.pipeline_sha256,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_dict(MODEL_CONFIG, "skyulf_pipeline_config.json")
            mlflow.log_dict(
                {
                    "sample": SAMPLE,
                    "records": RECORDS,
                    "training": TRAINING,
                    "features": FEATURES,
                    "split": [3_000, 500, 200, 100],
                },
                "data_lineage.json",
            )
            model_uri = log_local_model(
                artifact_path,
                run_id=run.info.run_id,
                artifact_path="model",
                tracking_uri="databricks",
            )
        registered = register_model(
            model_uri,
            MODEL,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
        )
    if str(registered.version) != "1":
        raise AssertionError("A fresh schema must create the first UC model version.")
    return {
        "stage": "train",
        "sample": SAMPLE,
        "records": 3_800,
        "training": 3_000,
        "heldout": 500,
        "initial_source": int(spark.table(SOURCE).count()),
        "model": MODEL,
        "version": str(registered.version),
        "run_id": run.info.run_id,
        "digest": artifact.manifest.pipeline_sha256,
        "metrics": metrics,
    }


def _prepared() -> Any:
    """Resolve the immutable first UC model version for both score jobs."""
    return prepare_local_workflow(
        LocalWorkflowConfig(
            runtime="databricks",
            engine="pandas",
            source=InputSource(
                kind="uc_table",
                table=SOURCE,
                read_mode="incremental",
                max_rows=300,
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


def _verify(spark: Any, prepared: Any, expected: int) -> dict[str, float]:
    """Compare every persisted score with direct model output and unseen labels."""
    source = spark.table(SOURCE).select("entity_id", *FEATURES).orderBy("entity_id").toPandas()
    actual = spark.table(TARGET).select("entity_id", "prediction").orderBy("entity_id").toPandas()
    if len(source) != expected or len(actual) != expected:
        raise AssertionError("Source and target counts differ from the expected stage.")
    if source["entity_id"].tolist() != actual["entity_id"].tolist():
        raise AssertionError("Persisted prediction keys do not match source keys.")
    direct = prepared.predict(source.loc[:, list(FEATURES)])
    np.testing.assert_allclose(
        actual["prediction"].to_numpy(),
        direct["prediction"].to_numpy(),
        rtol=0,
        atol=1e-9,
    )
    labels = (
        spark.table(RECORDS)
        .join(spark.table(SOURCE).select("entity_id"), on="entity_id", how="inner")
        .select("entity_id", "fare_amount")
        .toPandas()
    )
    aligned = actual.merge(labels, on="entity_id", how="left", validate="one_to_one")
    if aligned["fare_amount"].isna().any():
        raise AssertionError("An inference row lacks a held-back evaluation label.")
    errors = aligned["prediction"] - aligned["fare_amount"]
    return {
        "observed_mae": float(np.abs(errors).mean()),
        "observed_rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def _score(spark: Any, stage: str, expected_input: int, expected_total: int) -> dict[str, Any]:
    """Record a separate MLflow inference run and validate the committed Delta rows."""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(EXPERIMENT)
    prepared = _prepared()
    admission = DeltaTableAdmission(spark, CONTROL)
    with mlflow.start_run(run_name=f"skyulf-nyctaxi-{stage}") as run:
        mlflow.set_tags({"skyulf_phase": stage, "dataset": SAMPLE})
        result = run_incremental_local_batch(
            spark, prepared, row_keys=("entity_id",), admission=admission
        )
        if result.noop or result.input_count != expected_input:
            raise AssertionError("The incremental scorer selected the wrong number of trips.")
        observed = _verify(spark, prepared, expected_total)
        mlflow.log_params(
            {
                "model_name": MODEL,
                "model_version": "1",
                "source_table": SOURCE,
                "target_table": TARGET,
                "source_end_version": result.source_end_version,
            }
        )
        mlflow.log_metrics(
            {
                "input_rows": result.input_count,
                "output_rows": result.output_count,
                "target_rows": expected_total,
                **observed,
            }
        )
        if result.manifest is not None:
            mlflow.log_dict(result.manifest, "delta_publication_receipt.json")
    return {
        "stage": stage,
        "input_rows": result.input_count,
        "target_rows": expected_total,
        "source_end_version": result.source_end_version,
        "target_version": result.commit_version,
        "mlflow_run_id": run.info.run_id,
        **observed,
    }


def score_initial(spark: Any) -> dict[str, Any]:
    """Score the first 200 records without a date or source-version argument."""
    return _score(spark, "score_initial", 200, 200)


def score_append(spark: Any) -> dict[str, Any]:
    """Append 100 held-back real trips, then score only those new keys."""
    before = spark.table(TARGET).orderBy("entity_id").collect()
    if len(before) != 200 or spark.table(SOURCE).count() != 200:
        raise ValueError("Initial 200-row scoring stage has not completed cleanly.")
    pending = (
        spark.table(RECORDS)
        .where(F.col("split_index").between(3_701, 3_800))
        .select("entity_id", *FEATURES)
    )
    if pending.count() != 100:
        raise AssertionError("Expected exactly 100 held-back real trips.")
    pending.write.format("delta").mode("append").saveAsTable(SOURCE)
    result = _score(spark, "score_append", 100, 300)
    after = spark.table(TARGET).orderBy("entity_id").collect()
    original_keys = {row.entity_id for row in before}
    if [row for row in after if row.entity_id in original_keys] != before:
        raise AssertionError("The first 200 persisted predictions changed after append.")
    replay = run_incremental_local_batch(
        spark, _prepared(), row_keys=("entity_id",), admission=DeltaTableAdmission(spark, CONTROL)
    )
    if (
        not replay.noop
        or replay.input_count != 0
        or replay.commit_version != result["target_version"]
    ):
        raise AssertionError("An immediate replay created another target commit.")
    return {**result, "unchanged_initial_rows": 200, "noop_target_version": replay.commit_version}


runtime_dbutils: Any = globals().get("dbutils")
runtime_spark: Any = globals().get("spark")
if runtime_dbutils is not None and runtime_spark is not None:
    runtime_dbutils.widgets.text("stage", "train")
    stage = runtime_dbutils.widgets.get("stage")
    if stage == "train":
        outcome = train(runtime_spark)
    elif stage == "score_initial":
        outcome = score_initial(runtime_spark)
    elif stage == "score_append":
        outcome = score_append(runtime_spark)
    else:
        raise ValueError(f"Unknown taxi stage: {stage}")
    runtime_dbutils.notebook.exit(json.dumps(outcome))
