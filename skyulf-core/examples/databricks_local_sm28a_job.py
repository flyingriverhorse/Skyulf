# Databricks notebook source
"""One-time SM-28a rehearsal on isolated, real NYC taxi Delta records."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow  # ty: ignore[unresolved-import]
from pyspark.sql import functions as F  # ty: ignore[unresolved-import]

from skyulf.integrations.databricks.local_retraining import (
    LocalTrainingSpec,
    train_local_candidate,
)

SCHEMA = "workspace.skyulf_sm24a_20260923"
ORIGINAL = "workspace.skyulf_nyctaxi_e2e_20260923.taxi_training"
SOURCE = f"{SCHEMA}.skyulf_sm28a_labels_r1"
MODEL = f"{SCHEMA}.skyulf_sm28a_candidate_r1"
EXPERIMENT = "/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/sm28a_r1_experiment"
FEATURES = ("trip_distance", "trip_duration_minutes", "pickup_hour", "pickup_weekday")


def main() -> None:
    """Create a pinned label table and compare two registered candidates."""
    spark = globals().get("spark")
    if spark is None:
        raise RuntimeError("This example requires a Databricks notebook Spark session.")
    if spark.catalog.tableExists(SOURCE):
        raise RuntimeError("SM-28a test table already exists; use a new revision.")
    original = spark.table(ORIGINAL)
    late_id = original.orderBy("entity_id").first().entity_id
    labeled = original.select(
        "entity_id",
        F.col("tpep_pickup_datetime").alias("event_time"),
        *FEATURES,
        "fare_amount",
    ).withColumn(
        "label_at",
        F.when(
            F.col("entity_id") == late_id, F.lit("2016-03-02 00:00:00").cast("timestamp")
        ).otherwise(F.expr("event_time + INTERVAL 1 HOUR")),
    )
    labeled.write.format("delta").saveAsTable(SOURCE)
    source_version = int(spark.sql(f"DESCRIBE HISTORY {SOURCE}").first().version)
    spec = LocalTrainingSpec(
        table=SOURCE,
        version=source_version,
        start=datetime(2016, 1, 1, tzinfo=UTC),
        holdout_start=datetime(2016, 2, 1, tzinfo=UTC),
        cutoff=datetime(2016, 3, 1, tzinfo=UTC),
        event_column="event_time",
        result_available_at_column="label_at",
        record_key_columns=("entity_id",),
        input_columns=FEATURES,
        target_column="fare_amount",
        max_rows=3500,
        max_bytes=18_000_000,
    )
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    with tempfile.TemporaryDirectory(prefix="skyulf-sm28a-") as directory:
        baseline = train_local_candidate(
            spark,
            spec,
            {"preprocessing": [], "modeling": {"type": "linear_regression"}},
            model_name=MODEL,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
            experiment_name=EXPERIMENT,
            run_name="sm28a-baseline",
            artifact_path=Path(directory) / "baseline",
            metric="heldout_rmse",
            min_improvement=0,
        )
        candidate = train_local_candidate(
            spark,
            spec,
            {
                "preprocessing": [
                    {
                        "name": "impute",
                        "transformer": "SimpleImputer",
                        "params": {"columns": list(FEATURES), "strategy": "median"},
                    },
                    {
                        "name": "scale",
                        "transformer": "StandardScaler",
                        "params": {"columns": list(FEATURES)},
                    },
                ],
                "modeling": {
                    "type": "random_forest_regressor",
                    "params": {"n_estimators": 30, "max_depth": 8, "random_state": 42, "n_jobs": 1},
                },
            },
            model_name=MODEL,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
            experiment_name=EXPERIMENT,
            run_name="sm28a-candidate",
            artifact_path=Path(directory) / "candidate",
            metric="heldout_rmse",
            min_improvement=0,
            champion_version=baseline.model_version,
        )
    if baseline.training_rows < 100 or baseline.holdout_rows < 100:
        raise AssertionError("Temporal split did not preserve both large partitions.")
    if baseline.unavailable_labels < 1:
        raise AssertionError("The intentionally late label was not excluded.")
    if candidate.comparison.champion_version != baseline.model_version:
        raise AssertionError("Comparison did not use the pinned baseline version.")
    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    aliases = client.get_registered_model(MODEL).aliases
    if any(alias.alias in ("champion", "challenger") for alias in aliases):
        raise AssertionError("SM-28a must not set promotion aliases.")
    for result in (baseline, candidate):
        artifacts = {item.path for item in client.list_artifacts(result.run_id)}
        if not {"model", "candidate_comparison.json", "skyulf_pipeline_config.json"} <= artifacts:
            raise AssertionError("Candidate run is missing required evidence artifacts.")
    print(
        json.dumps(
            {
                "source": SOURCE,
                "source_version": source_version,
                "baseline_version": baseline.model_version,
                "baseline_run_id": baseline.run_id,
                "candidate_version": candidate.model_version,
                "candidate_run_id": candidate.run_id,
                "training_rows": candidate.training_rows,
                "holdout_rows": candidate.holdout_rows,
                "unavailable_labels": candidate.unavailable_labels,
                "candidate_rmse": candidate.comparison.candidate_metrics["heldout_rmse"],
                "baseline_rmse": candidate.comparison.champion_metrics["heldout_rmse"],
                "eligible": candidate.comparison.eligible,
                "reason": candidate.comparison.reason,
            },
            sort_keys=True,
        )
    )


main()
