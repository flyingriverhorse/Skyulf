# Databricks notebook source
"""Bounded H2 acceptance on new resources in the existing personal test schema.

Prepared locally; run only after explicit approval of the accompanying plan.
This is a test harness, not a generated production Bundle.
"""

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
import polars as pl

from skyulf.inference.local_pipeline import predict_local_pipeline
from skyulf.integrations.databricks.job_runtime import run_bundle_action
from skyulf.integrations.databricks.project import load_project_workflow
from skyulf.integrations.mlflow.registry import load_registered_local_pipeline, resolve_model

SCHEMA = "workspace.skyulf_lifecycle_test"
PREFIX = "sm33h2_20260926_r1"
SOURCE = f"{SCHEMA}.{PREFIX}_source"
REMOTE = "/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/sm33h2/r1"
EXPERIMENT = f"{REMOTE}/experiment"
FRAME_SCHEMA = "record_id long, x double, z double, age double, target double"


def rows(start, stop):
    """Create deterministic inputs including missing labels and invalid training ages."""
    return [
        (
            index,
            None if index % 13 == 0 else float(index % 10),
            float(index % 3),
            -1.0 if index % 19 == 0 else float(index % 80),
            None if index % 17 == 0 else 20.0 + 3 * (index % 10) + (index % 3),
        )
        for index in range(start, stop)
    ]


def config(engine, *, fit_intercept=True):
    """Use the actual operator runtime with explicit local training and score policies."""
    return {
        "engine": engine,
        "training_table": SOURCE,
        "training_version": 0,
        "training_window_mode": "full_snapshot",
        "split_strategy": "random",
        "test_size": 0.25,
        "random_state": 42,
        "stratify": False,
        "filter_unavailable_results": False,
        "record_key_columns": ["record_id"],
        "input_columns": ["x", "z"],
        "target_column": "target",
        "max_rows": 1000,
        "max_input_mb": 8,
        "model_name": f"{SCHEMA}.{PREFIX}_{engine}_model",
        "model_version": "1",
        "score_source_table": SOURCE,
        "prediction_table": f"{SCHEMA}.{PREFIX}_{engine}_predictions",
        "score_model_selection": "champion",
        "model_change_mode": "incremental_append",
        "promotion_policy": "manual_approval",
        "score_handoff": "disabled",
        "tracking_uri": "databricks",
        "registry_uri": "databricks-uc",
        "metric": "heldout_rmse",
        "min_improvement": 0.01,
        "quality_threshold": 1000.0,
        "cv_enabled": True,
        "cv_folds": 3,
        "cv_type": "k_fold",
        "cv_shuffle": True,
        "cv_random_state": 42,
        "pipeline": {
            "preprocessing": [],
            "modeling": {"type": "linear_regression", "params": {"fit_intercept": fit_intercept}},
        },
    }


def recipe(lower):
    """Save genuine Python recipes through the same loader used by generated projects."""
    filters = [
        {
            "name": "known_result",
            "transformer": "DropMissingRows",
            "params": {"subset": ["target"]},
        },
        {
            "name": "valid_age",
            "transformer": "ManualBounds",
            "params": {"bounds": {"age": {"lower": lower, "upper": 120}}},
        },
    ]
    steps = [
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"columns": ["x", "z"], "strategy": "mean"},
        },
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x", "z"]}},
    ]
    return (
        '"""Saved H2 training recipe."""\n'
        f"def build_pre_split_steps():\n    return {filters!r}\n\n"
        f"def build_preprocessing():\n    return {steps!r}\n"
    )


def operator(spark, settings, action, **parameters):
    """Use the public job-action decoder and concrete lifecycle controls."""
    return run_bundle_action(
        spark,
        settings,
        {"lifecycle_action": action, **parameters},
        task_role="lifecycle",
    ).result


def train_engine(spark, engine):
    """Train, approve, promote, reject and roll back with real UC/MLflow evidence."""
    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    results = []
    with TemporaryDirectory(prefix="skyulf-h2-live-") as directory:
        source_path = Path(directory) / "preprocessing.py"
        for index, (intercept, lower) in enumerate(((False, 20), (True, 0), (False, 0)), 1):
            settings = config(engine, fit_intercept=intercept)
            source_path.write_text(recipe(lower), encoding="utf-8")
            training = load_project_workflow(settings, source_path)
            candidate = run_bundle_action(
                spark,
                training,
                {"lifecycle_action": "train"},
                task_role="lifecycle",
                experiment_name=EXPERIMENT,
                artifact_path=Path(directory) / f"model-{index}",
            ).result
            assert candidate.model_version == str(index)
            # Today's project file is deliberately unusable; approval must use saved evidence.
            source_path.write_text(
                "raise RuntimeError('Current recipe must not run')\n", encoding="utf-8"
            )
            metrics = client.get_run(candidate.run_id).data.metrics
            assert "heldout_rmse" in metrics and any(key.startswith("cv_") for key in metrics)
            evidence_path = client.download_artifacts(
                candidate.run_id, "training_filter_evidence.json", directory
            )
            evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
            if index == 3:
                change = operator(
                    spark,
                    settings,
                    "reject",
                    candidate_version="3",
                    expected_champion_version="2",
                    rejection_reason="H2 explicit rejection test",
                )
            else:
                change = operator(
                    spark,
                    settings,
                    "approve",
                    candidate_version=str(index),
                    expected_champion_version="none" if index == 1 else "1",
                )
            results.append(
                {
                    "version": str(index),
                    "run_id": candidate.run_id,
                    "metrics": metrics,
                    "evidence": evidence,
                    "transition": asdict(change),
                }
            )
            if index == 2:
                promotion = change
        rollback = operator(
            spark,
            config(engine),
            "rollback",
            expected_champion_version="2",
            promotion_receipt_json=json.dumps(asdict(promotion)),
        )
        assert rollback.new_version == "1"
    model = client.get_registered_model(config(engine)["model_name"])
    assert str(model.aliases["champion"]) == "1"
    return {"engine": engine, "candidates": results, "rollback": asdict(rollback)}


def score_engine(spark, engine, expected_rows):
    """Score real Delta inserts through the public score runtime without training targets."""
    settings = config(engine)
    result = run_bundle_action(spark, settings, {}, task_role="score").result
    assert result.input_count == result.output_count == expected_rows
    resolved = resolve_model(
        settings["model_name"], version="1", tracking_uri="databricks", registry_uri="databricks-uc"
    )
    artifact = load_registered_local_pipeline(
        resolved, tracking_uri="databricks", registry_uri="databricks-uc"
    )
    raw = pd.DataFrame({"x": [None, 2.0, 7.0], "z": [1.0, 2.0, 0.0]})
    local = predict_local_pipeline(pl.from_pandas(raw) if engine == "polars" else raw, artifact)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    wrapped = mlflow.pyfunc.load_model(resolved.model_uri).predict(raw)
    assert np.allclose(np.asarray(local).reshape(-1), np.asarray(wrapped).reshape(-1))
    assert spark.table(settings["prediction_table"]).count() == (
        240 if expected_rows == 240 else 243
    )
    return asdict(result)


def prediction_snapshot(spark, engine):
    """Capture bounded published identities, predictions and model versions in key order."""
    return (
        spark.table(config(engine)["prediction_table"])
        .select("record_id", "prediction", "model_version")
        .orderBy("record_id")
        .collect()
    )


def main(spark, phase):
    """Keep fixture creation, each training engine and fresh-process scoring separate."""
    if phase == "pandas_train":
        assert not spark.catalog.tableExists(SOURCE), "Do not overwrite an existing rehearsal."
        client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
        for engine in ("pandas", "polars"):
            settings = config(engine)
            assert not spark.catalog.tableExists(settings["prediction_table"])
            try:
                client.get_registered_model(settings["model_name"])
            except mlflow.exceptions.MlflowException as error:
                if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
            else:
                raise ValueError("Rehearsal model already exists; inspect before retrying.")
        spark.createDataFrame(rows(0, 240), FRAME_SCHEMA).write.format("delta").option(
            "delta.enableChangeDataFeed", "true"
        ).mode("error").saveAsTable(SOURCE)
        return train_engine(spark, "pandas")
    if phase == "polars_train":
        assert spark.table(SOURCE).count() == 240
        return train_engine(spark, "polars")
    if phase != "score":
        raise ValueError(f"Unknown phase: {phase}")
    initial = {engine: score_engine(spark, engine, 240) for engine in ("pandas", "polars")}
    original_outputs = {}
    for engine in ("pandas", "polars"):
        published = prediction_snapshot(spark, engine)
        assert [row["record_id"] for row in published] == list(range(240))
        assert all(str(row["model_version"]) == "1" for row in published)
        assert all(np.isfinite(row["prediction"]) for row in published)
        original_outputs[engine] = published
    spark.createDataFrame(rows(240, 243), FRAME_SCHEMA).write.format("delta").mode(
        "append"
    ).saveAsTable(SOURCE)
    appended = {engine: score_engine(spark, engine, 3) for engine in ("pandas", "polars")}
    noops = {}
    for engine in ("pandas", "polars"):
        published = prediction_snapshot(spark, engine)
        assert [row["record_id"] for row in published] == list(range(243))
        assert published[:240] == original_outputs[engine]
        assert all(str(row["model_version"]) == "1" for row in published)
        assert all(np.isfinite(row["prediction"]) for row in published)
        result = run_bundle_action(spark, config(engine), {}, task_role="score").result
        assert result.noop and result.input_count == result.output_count == 0
        assert result.commit_version == appended[engine]["commit_version"]
        assert prediction_snapshot(spark, engine) == published
        noops[engine] = asdict(result)
    return {"source": SOURCE, "initial": initial, "appended": appended, "noops": noops}


if __name__ == "__main__":
    dbutils = globals()["dbutils"]
    output = main(globals()["spark"], dbutils.widgets.get("phase"))
    dbutils.notebook.exit(json.dumps(output, default=str))
