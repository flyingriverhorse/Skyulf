# Databricks notebook source
"""One-time SM-24a training and separate monthly scoring validation job.

This script creates only isolated test sources in the supplied schema. It does
not publish predictions or move a registry alias. The two stages are submitted
as separate serverless runs; scoring receives the training receipt explicitly.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import mlflow  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
import polars as pl

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import predict_local_pipeline
from skyulf.integrations.databricks import (
    InputSource,
    LocalSourceSpec,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    PreflightError,
    evaluate_local_holdout,
    fit_local_workflow,
    prepare_local_workflow,
    read_local_source,
    score_local_source,
)
from skyulf.integrations.mlflow.local_model import log_local_model
from skyulf.integrations.mlflow.registry import register_model

SCHEMA = "workspace.skyulf_sm24a_20260923"
EXPERIMENT = "/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/experiment"
TRAIN_SOURCE = f"{SCHEMA}.skyulf_sm24a_train_source"
SCORE_SOURCE = f"{SCHEMA}.skyulf_sm24a_score_source"
STARTS = {
    "2026-01": (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)),
    "2026-02": (datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
}


def _rows(count: int) -> pd.DataFrame:
    """Generate deterministic numeric nulls, outliers, labels and categories."""
    positions = np.arange(count)
    x = positions.astype("float64") / 12.0
    x[positions == 17] = 45.0
    z = np.sin(positions / 7.0) * 3.0 + 4.0
    z[positions % 19 == 0] = np.nan
    clean_z = np.nan_to_num(z, nan=4.0)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "city": np.array(["Vilnius", "Riga", "Tallinn"])[positions % 3],
            "regression_target": 2.0 * x + clean_z + (positions % 3),
            "classification_target": (((x % 3.0) + clean_z / 3.0) > 2.4).astype("int64"),
        }
    )


def _source_spec(
    table: str, source_version: int, month: str, columns: tuple[str, ...], *, rows: int
) -> LocalSourceSpec:
    """Build one fixed snapshot/period request for a bounded local transfer."""
    start, end = STARTS[month]
    return LocalSourceSpec(
        table=table,
        version=source_version,
        period_start=start,
        period_end=end,
        row_keys=("entity_id",),
        input_columns=columns,
        max_rows=rows,
        max_bytes=4_000_000,
    )


def _create_sources(spark) -> dict[str, int]:
    """Create exactly two test-owned managed Delta tables once."""
    train = _rows(1_000)
    train_rows = [
        (
            f"train-{index:04d}",
            datetime(2026, 1, index % 28 + 1, tzinfo=UTC),
            float(cast(float, row.x)),
            None if pd.isna(row.z) else float(cast(float, row.z)),
            str(row.city),
            float(cast(float, row.regression_target)),
            int(cast(int, row.classification_target)),
        )
        for index, row in enumerate(train.itertuples(index=False))
    ]
    spark.createDataFrame(
        train_rows,
        "entity_id string, event_time timestamp, x double, z double, city string, "
        "regression_target double, classification_target long",
    ).write.format("delta").saveAsTable(TRAIN_SOURCE)
    score = _rows(160)
    score_rows = [
        (
            f"score-{index:04d}",
            datetime(2026, 1 if index < 80 else 2, index % 28 + 1, tzinfo=UTC),
            float(cast(float, row.x)),
            None if pd.isna(row.z) else float(cast(float, row.z)),
            "Stockholm" if index in (0, 80) else str(row.city),
        )
        for index, row in enumerate(score.itertuples(index=False))
    ]
    spark.createDataFrame(
        score_rows,
        "entity_id string, event_time timestamp, x double, z double, city string",
    ).write.format("delta").saveAsTable(SCORE_SOURCE)
    return {
        "train": int(spark.sql(f"DESCRIBE HISTORY {TRAIN_SOURCE}").select("version").first()[0]),
        "score": int(spark.sql(f"DESCRIBE HISTORY {SCORE_SOURCE}").select("version").first()[0]),
    }


def train(
    spark,
    cases: list,
    *,
    source_versions: dict[str, int] | None = None,
    registration_prefix: str = "skyulf_sm24a_",
) -> dict:
    """Fit five packages and register the model with its held-out metrics."""
    if source_versions is None:
        source_versions = _create_sources(spark)
    source = read_local_source(
        spark,
        _source_spec(
            TRAIN_SOURCE,
            source_versions["train"],
            "2026-01",
            ("x", "z", "city", "regression_target", "classification_target"),
            rows=1_000,
        ),
    )
    if len(source) != 1_000 or source["entity_id"].nunique() != 1_000:
        raise AssertionError("Training source membership is not the expected 1,000 keys.")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT)
    receipt = {"source_versions": source_versions, "models": {}, "reference": {}}
    with tempfile.TemporaryDirectory(prefix="skyulf-sm24a-") as directory:
        for name, engine, model, target, columns, preprocessing in cases:
            frame = source.loc[:, [*columns, target]]
            native = pl.from_pandas(frame) if engine == "polars" else frame
            training = native[:800]
            heldout = native[800:]
            if len(training) != 800 or len(heldout) != 200:
                raise AssertionError("Expected exactly 800 train and 200 held-out rows.")
            artifact_path = Path(directory) / name
            artifact = fit_local_workflow(
                {"preprocessing": preprocessing, "modeling": {"type": model}},
                SplitDataset(train=training, test=heldout),
                target_column=target,
                artifact_path=artifact_path,
                max_rows=1_000,
                max_bytes=4_000_000,
            )
            heldout_metrics = evaluate_local_holdout(artifact, heldout, target_column=target)
            receipt["reference"][name] = {}
            for month in STARTS:
                query = read_local_source(
                    spark,
                    _source_spec(SCORE_SOURCE, source_versions["score"], month, columns, rows=80),
                )
                predicted = predict_local_pipeline(query.loc[:, list(columns)], artifact)
                receipt["reference"][name][month] = {
                    key: [
                        value.item() if isinstance(value, np.generic) else value
                        for value in predicted.iloc[index]
                    ]
                    for index, key in enumerate(query["entity_id"])
                }
            with mlflow.start_run(run_name=f"sm24a-{registration_prefix}{name}") as run:
                mlflow.log_params(
                    {
                        "fitted_engine": engine,
                        "model_type": model,
                        "target_column": target,
                        "training_rows": 800,
                        "heldout_rows": 200,
                        "source_table": TRAIN_SOURCE,
                        "source_version": source_versions["train"],
                    }
                )
                mlflow.log_metrics(heldout_metrics)
                model_uri = log_local_model(
                    artifact_path,
                    run_id=run.info.run_id,
                    artifact_path="model",
                    tracking_uri="databricks",
                )
            registered_name = f"{SCHEMA}.{registration_prefix}{name.lower()}"
            registered = register_model(
                model_uri,
                registered_name,
                tracking_uri="databricks",
                registry_uri="databricks-uc",
            )
            receipt["models"][name] = {
                "name": registered_name,
                "version": str(registered.version),
                "run_id": run.info.run_id,
                "digest": artifact.manifest.pipeline_sha256,
                "engine": engine,
                "columns": columns,
                "task": artifact.manifest.task,
                "model_class": artifact.manifest.model_class,
                "heldout_rows": 200,
                "heldout_metrics": heldout_metrics,
            }
    receipt["runtime"] = {
        name: version(name)
        for name in ("skyulf-core", "mlflow", "pandas", "polars", "scikit-learn")
    }
    return receipt


def _expect_rejection(action, expected: str) -> str:
    """Require a bounded negative case to fail with an actionable message."""
    try:
        action()
    except (PreflightError, ValueError) as exc:
        if expected not in str(exc):
            raise AssertionError(f"Expected {expected!r} in {type(exc).__name__}: {exc}") from exc
        return type(exc).__name__
    raise AssertionError(f"Expected rejection containing {expected!r}.")


def score(spark, receipt: dict, *, audit_negative: bool = False) -> dict:
    """Read two months from the pinned source and compare five registered models."""
    source_version = int(receipt["source_versions"]["score"])
    report = {"source_version": source_version, "cases": {}}
    replay = None
    for name, model in receipt["models"].items():
        config = LocalWorkflowConfig(
            runtime="databricks",
            engine=model["engine"],
            source=InputSource(
                kind="uc_table",
                table=SCORE_SOURCE,
                version=source_version,
                max_rows=80,
                max_bytes=4_000_000,
            ),
            model=ModelSelection(
                kind="local_pipeline",
                name=model["name"],
                version=model["version"],
                tracking_uri="databricks",
                registry_uri="databricks-uc",
            ),
            sink=OutputSink(kind="return_frame"),
        )
        prepared = prepare_local_workflow(config)
        if prepared.preflight.model_digest != model["digest"]:
            raise AssertionError(f"{name}: registry digest differs from training receipt.")
        report["cases"][name] = {}
        for month in STARTS:
            result = score_local_source(
                spark,
                _source_spec(SCORE_SOURCE, source_version, month, tuple(model["columns"]), rows=80),
                prepared,
            )
            actual = result.predictions
            expected = receipt["reference"][name][month]
            if set(actual["entity_id"]) != set(expected):
                raise AssertionError(f"{name}: {month} membership differs from reference.")
            output_columns = [str(column) for column in actual.columns if column != "entity_id"]
            for _, row in actual.iterrows():
                values = expected[row["entity_id"]]
                if model["task"] == "classification" and row["prediction"] != values[0]:
                    raise AssertionError(f"{name}: class label differs for {row['entity_id']}.")
                for position, column in enumerate(output_columns):
                    if column == "prediction" and model["task"] == "classification":
                        continue
                    if not np.isclose(
                        float(row[column]), float(values[position]), rtol=0, atol=1e-9
                    ):
                        raise AssertionError(f"{name}: {column} differs for {row['entity_id']}.")
            if model["task"] == "classification":
                np.testing.assert_allclose(
                    actual[
                        [
                            column
                            for column in actual.columns
                            if str(column).startswith("probability_")
                        ]
                    ].sum(axis=1),
                    1.0,
                    rtol=0,
                    atol=1e-9,
                )
            report["cases"][name][month] = {
                **result.diagnostics,
                "reference_match": True,
                "row_count": len(actual),
            }
            if name == "R1" and month == "2026-01":
                replay = (prepared, config, result)
    if audit_negative:
        if replay is None:
            raise AssertionError("The R1 January replay fixture was not scored.")
        prepared, config, first = replay
        model = receipt["models"]["R1"]
        spec = _source_spec(
            SCORE_SOURCE, source_version, "2026-01", tuple(model["columns"]), rows=80
        )
        second = score_local_source(spark, spec, prepared)
        pd.testing.assert_frame_equal(first.predictions, second.predictions)
        if first.diagnostics != second.diagnostics:
            raise AssertionError("The same request changed its model/source identity.")
        report["replay"] = {"case": "R1", "month": "2026-01", "identical": True}
        report["negative_cases"] = {
            "reordered_inputs": _expect_rejection(
                lambda: score_local_source(
                    spark, replace(spec, input_columns=("z", "x")), prepared
                ),
                "raw column order",
            ),
            "missing_input": _expect_rejection(
                lambda: score_local_source(spark, replace(spec, input_columns=("x",)), prepared),
                "raw column order",
            ),
            "changed_source_version": _expect_rejection(
                lambda: score_local_source(spark, replace(spec, version=1), prepared),
                "pinned Delta table and version",
            ),
            "over_rows": _expect_rejection(
                lambda: score_local_source(spark, replace(spec, max_rows=1), prepared),
                "max_rows",
            ),
            "over_bytes": _expect_rejection(
                lambda: score_local_source(spark, replace(spec, max_bytes=1), prepared),
                "max_bytes",
            ),
            "wrong_engine": _expect_rejection(
                lambda: prepare_local_workflow(config.model_copy(update={"engine": "polars"})),
                "engine_mismatch",
            ),
            "wrong_artifact_kind": _expect_rejection(
                lambda: prepare_local_workflow(
                    config.model_copy(
                        update={
                            "model": ModelSelection(
                                kind="portable_bundle",
                                name=model["name"],
                                version=model["version"],
                                tracking_uri="databricks",
                                registry_uri="databricks-uc",
                            )
                        }
                    )
                ),
                "registry_or_artifact_invalid",
            ),
            "spark_runtime": _expect_rejection(
                lambda: prepare_local_workflow(config.model_copy(update={"runtime": "spark"})),
                "runtime_unsupported",
            ),
        }
    report["runtime"] = {
        name: version(name)
        for name in ("skyulf-core", "mlflow", "pandas", "polars", "scikit-learn")
    }
    return report


runtime_dbutils: Any = globals().get("dbutils")
runtime_spark: Any = globals().get("spark")
if runtime_dbutils is not None and runtime_spark is not None:
    runtime_dbutils.widgets.text("stage", "train")
    runtime_dbutils.widgets.text("payload", "")
    stage = runtime_dbutils.widgets.get("stage")
    payload = json.loads(runtime_dbutils.widgets.get("payload"))
    if stage == "train":
        result = train(runtime_spark, payload["cases"])
    elif stage == "train_existing":
        result = train(
            runtime_spark,
            payload["cases"],
            source_versions=payload["source_versions"],
            registration_prefix="skyulf_sm24a_metrics_",
        )
    elif stage == "score":
        result = score(
            runtime_spark,
            json.loads(Path(payload["receipt_path"]).read_text(encoding="utf-8")),
            audit_negative=bool(payload.get("audit_negative", False)),
        )
    else:
        raise ValueError("stage must be train or score.")
    runtime_dbutils.notebook.exit(json.dumps(result, default=str))
