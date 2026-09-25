# Databricks notebook source
"""Replay local scaler and outlier pipelines through MLflow on Databricks.

The train stage reads the existing isolated SM-24a source and creates one
additional test-only scoring table. The score stage runs in a separate job,
loads concrete Unity Catalog model versions, and compares keyed predictions.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import predict_local_pipeline
from skyulf.integrations.databricks import (
    InputSource,
    LocalSourceSpec,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    fit_local_workflow,
    prepare_local_workflow,
    read_local_source,
    score_local_source,
)
from skyulf.integrations.mlflow.local_model import log_local_model
from skyulf.integrations.mlflow.registry import register_model
from skyulf.pipeline import SkyulfPipeline

SCHEMA = "workspace.skyulf_sm24a_20260923"
EXPERIMENT = "/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/experiment"
TRAIN_SOURCE = f"{SCHEMA}.skyulf_sm24a_train_source"
SCORE_SOURCE = f"{SCHEMA}.skyulf_sm24a_scaler_outlier_score_source"
PERIODS = {
    "2026-01": (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)),
    "2026-02": (datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
}
NODES = ("MaxAbsScaler", "MinMaxScaler", "RobustScaler", "StandardScaler", "Winsorize")
FILTER_NODES = ("IQR", "ZScore", "ManualBounds", "EllipticEnvelope")


def _source(
    table: str, version: int, month: str, columns: tuple[str, ...], rows: int
) -> LocalSourceSpec:
    """Pin the Delta snapshot, month, raw column order, and driver budget."""
    start, end = PERIODS[month]
    return LocalSourceSpec(
        table=table,
        version=version,
        period_start=start,
        period_end=end,
        record_key_columns=("entity_id",),
        input_columns=columns,
        max_rows=rows,
        max_bytes=4_000_000,
    )


def _create_score_source(spark: Any) -> int:
    """Create a test-only two-month source containing one extreme row per month."""
    rows = [
        (
            f"probe-{month}-{index:02d}",
            datetime(2026, month, index % 20 + 1, tzinfo=UTC),
            1000.0 if index == 0 else float(index),
        )
        for month in (1, 2)
        for index in range(20)
    ]
    spark.createDataFrame(rows, "entity_id string, event_time timestamp, x double").write.format(
        "delta"
    ).saveAsTable(SCORE_SOURCE)
    return int(spark.sql(f"DESCRIBE HISTORY {SCORE_SOURCE}").select("version").first()[0])


def _filter_probes() -> dict[str, dict[str, str]]:
    """Require row-filtering outlier nodes to reject a shortened prediction batch."""
    report: dict[str, dict[str, str]] = {}
    train = pd.DataFrame({"x": np.arange(100, dtype="float64")})
    train["target"] = 2 * train["x"] + 1
    query = pd.DataFrame({"x": [20.0, 1000.0, 40.0]})
    for node in FILTER_NODES:
        report[node] = {}
        params: dict[str, Any] = {"columns": ["x"]}
        if node == "ManualBounds":
            params["bounds"] = {"x": {"lower": 0, "upper": 99}}
        if node == "EllipticEnvelope":
            params["random_state"] = 42
        config = {
            "preprocessing": [{"name": "outliers", "transformer": node, "params": params}],
            "modeling": {"type": "linear_regression"},
        }
        for engine in ("pandas", "polars"):
            native = train if engine == "pandas" else pl.from_pandas(train)
            native_query = query if engine == "pandas" else pl.from_pandas(query)
            pipeline = SkyulfPipeline(config)
            pipeline.fit(native, target_column="target")
            if len(pipeline.feature_engineer.transform(native_query)) != 2:
                raise AssertionError(f"{node}/{engine}: fixture did not filter the outlier.")
            try:
                pipeline.predict(native_query)
            except ValueError as exc:
                if "outliers" not in str(exc) or "row count" not in str(exc):
                    raise AssertionError(f"{node}/{engine}: wrong rejection: {exc}") from exc
                report[node][engine] = "rejected_filtered_batch"
            else:
                raise AssertionError(f"{node}/{engine}: silently scored a shortened batch.")
    return report


def train(spark: Any) -> dict[str, Any]:
    """Train ten small saved pipelines and log heldout metrics beside each model."""
    train_version = int(spark.sql(f"DESCRIBE HISTORY {TRAIN_SOURCE}").select("version").first()[0])
    score_version = _create_score_source(spark)
    source = read_local_source(
        spark, _source(TRAIN_SOURCE, train_version, "2026-01", ("x", "regression_target"), 1000)
    )
    if len(source) != 1000:
        raise AssertionError("Training source was not the expected 1,000 rows.")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT)
    receipt: dict[str, Any] = {
        "train_version": train_version,
        "score_version": score_version,
        "models": {},
        "reference": {},
        "filter_probes": _filter_probes(),
    }
    with tempfile.TemporaryDirectory(prefix="skyulf-sm24a-scaler-") as directory:
        for node in NODES:
            for engine in ("pandas", "polars"):
                name = f"{node.lower()}_{engine}"
                native = source.loc[:, ["x", "regression_target"]]
                if engine == "polars":
                    native = pl.from_pandas(native)
                artifact_path = Path(directory) / name
                artifact = fit_local_workflow(
                    {
                        "preprocessing": [
                            {"name": "feature", "transformer": node, "params": {"columns": ["x"]}}
                        ],
                        "modeling": {"type": "linear_regression"},
                    },
                    SplitDataset(train=native[:800], test=native[800:]),
                    target_column="regression_target",
                    artifact_path=artifact_path,
                    max_rows=1000,
                    max_bytes=4_000_000,
                )
                heldout = source.iloc[800:]
                predicted = predict_local_pipeline(heldout.loc[:, ["x"]], artifact)["prediction"]
                actual = heldout["regression_target"].to_numpy()
                metrics = {
                    "heldout_mae": float(mean_absolute_error(actual, predicted)),
                    "heldout_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                    "heldout_r2": float(r2_score(actual, predicted)),
                }
                receipt["reference"][name] = {}
                for month in PERIODS:
                    query = read_local_source(
                        spark, _source(SCORE_SOURCE, score_version, month, ("x",), 20)
                    )
                    values = predict_local_pipeline(query.loc[:, ["x"]], artifact)["prediction"]
                    receipt["reference"][name][month] = dict(
                        zip(query["entity_id"], values.tolist(), strict=True)
                    )
                if node == "Winsorize":
                    probe = pd.DataFrame({"x": [1000.0]})
                    clipped = artifact.pipeline.feature_engineer.transform(
                        probe if engine == "pandas" else pl.from_pandas(probe)
                    )
                    if (
                        float(clipped["x"].iloc[0] if engine == "pandas" else clipped["x"][0])
                        >= 1000
                    ):
                        raise AssertionError(
                            f"{name}: fitted Winsorize bounds did not clip the outlier."
                        )
                with mlflow.start_run(run_name=f"sm24a-{name}") as run:
                    mlflow.log_params({"engine": engine, "preprocessing": node, "train_rows": 800})
                    mlflow.log_metrics(metrics)
                    model_uri = log_local_model(
                        artifact_path,
                        run_id=run.info.run_id,
                        artifact_path="model",
                        tracking_uri="databricks",
                    )
                registered_name = f"{SCHEMA}.skyulf_sm24a_audit_{name}"
                registered = register_model(
                    model_uri,
                    registered_name,
                    tracking_uri="databricks",
                    registry_uri="databricks-uc",
                )
                receipt["models"][name] = {
                    "node": node,
                    "engine": engine,
                    "name": registered_name,
                    "version": str(registered.version),
                    "run_id": run.info.run_id,
                    "digest": artifact.manifest.pipeline_sha256,
                    "metrics": metrics,
                }
    return receipt


def score(spark: Any, receipt: dict[str, Any]) -> dict[str, Any]:
    """Load registered versions in another job and replay both pinned months."""
    version = int(receipt["score_version"])
    report: dict[str, Any] = {"score_version": version, "cases": {}}
    for name, model in receipt["models"].items():
        config = LocalWorkflowConfig(
            runtime="databricks",
            engine=model["engine"],
            source=InputSource(
                kind="uc_table",
                table=SCORE_SOURCE,
                version=version,
                max_rows=20,
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
            raise AssertionError(f"{name}: loaded model digest differs from training.")
        report["cases"][name] = {}
        for month in PERIODS:
            result = score_local_source(
                spark, _source(SCORE_SOURCE, version, month, ("x",), 20), prepared
            )
            predictions = result.predictions
            expected = receipt["reference"][name][month]
            if set(predictions["entity_id"]) != set(expected):
                raise AssertionError(f"{name}/{month}: source row keys changed.")
            for row in predictions.itertuples(index=False):
                if not np.isclose(
                    cast(float, row.prediction),
                    cast(float, expected[row.entity_id]),
                    rtol=0,
                    atol=1e-9,
                ):
                    raise AssertionError(f"{name}/{month}: prediction changed for {row.entity_id}.")
            report["cases"][name][month] = {
                "rows": len(predictions),
                "reference_match": True,
                "outlier_predicted": any("-00" in key for key in predictions["entity_id"]),
                "model_version": result.diagnostics["model_version"],
            }
    return report


runtime_dbutils: Any = globals().get("dbutils")
runtime_spark: Any = globals().get("spark")
if runtime_dbutils is not None and runtime_spark is not None:
    runtime_dbutils.widgets.text("stage", "train")
    runtime_dbutils.widgets.text("receipt_path", "")
    stage = runtime_dbutils.widgets.get("stage")
    if stage == "train":
        output = train(runtime_spark)
    elif stage == "score":
        output = score(
            runtime_spark,
            json.loads(
                Path(runtime_dbutils.widgets.get("receipt_path")).read_text(encoding="utf-8")
            ),
        )
    else:
        raise ValueError("stage must be train or score.")
    runtime_dbutils.notebook.exit(json.dumps(output, default=str))
