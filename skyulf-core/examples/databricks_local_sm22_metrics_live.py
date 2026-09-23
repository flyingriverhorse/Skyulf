# Databricks notebook source
"""One-time Databricks validation of Core-held-out metrics and UC comparison."""

import json
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
import polars as pl

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_evaluation import evaluate_local_holdout
from skyulf.integrations.databricks.local_batch import fit_local_workflow
from skyulf.integrations.mlflow.local_model import log_local_model
from skyulf.integrations.mlflow.registry import register_model, resolve_model
from skyulf.integrations.mlflow.validation import compare_registered_local_models

EXPERIMENT = "/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/sm22_metrics_r1_experiment"
MODEL_NAME = "workspace.skyulf_sm24a_20260923.skyulf_sm22_metrics_r1"
REGRESSION_METRICS = {
    "heldout_mae",
    "heldout_mse",
    "heldout_rmse",
    "heldout_r2",
    "heldout_mape",
    "heldout_explained_variance",
}
CLASSIFICATION_METRICS = {
    "heldout_accuracy",
    "heldout_balanced_accuracy",
    "heldout_precision_weighted",
    "heldout_recall_weighted",
    "heldout_f1_weighted",
    "heldout_matthews_corrcoef",
    "heldout_precision",
    "heldout_recall",
    "heldout_f1",
    "heldout_log_loss",
    "heldout_roc_auc",
    "heldout_pr_auc",
}


def _native(frame: pd.DataFrame, engine: str) -> pd.DataFrame | pl.DataFrame:
    """Use the requested local engine without changing row order."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _fit(
    path: Path,
    engine: str,
    task: str,
    train: pd.DataFrame,
    heldout: pd.DataFrame,
):
    """Fit and reload a real Skyulf local artifact."""
    model_type = "linear_regression" if task == "regression" else "logistic_regression"
    return fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": model_type}},
        SplitDataset(train=_native(train, engine), test=_native(heldout, engine)),
        target_column="target",
        artifact_path=path,
        max_rows=200,
        max_bytes=200_000,
    )


def _log_metrics(
    name: str, metrics: dict[str, float], *, artifact_path: Path | None = None
) -> tuple[str, str | None]:
    """Log every finite metric and optionally package the fitted model."""
    if not metrics or not all(np.isfinite(value) for value in metrics.values()):
        raise AssertionError(f"{name} produced missing or non-finite metrics")
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_metrics(metrics)
        mlflow.log_params({"validation": "heldout", "case": name})
        model_uri = (
            log_local_model(
                artifact_path,
                run_id=run.info.run_id,
                artifact_path="model",
                tracking_uri="databricks",
            )
            if artifact_path is not None
            else None
        )
        run_id = run.info.run_id
    recorded = mlflow.MlflowClient(tracking_uri="databricks").get_run(run_id).data.metrics
    if not metrics.keys() <= recorded.keys():
        raise AssertionError(f"{name} is missing MLflow metrics")
    for key, value in metrics.items():
        if not np.isclose(recorded[key], value):
            raise AssertionError(f"{name} changed MLflow metric {key}")
    return run_id, model_uri


def run() -> dict[str, Any]:
    """Exercise four local-engine cases and one pinned UC model comparison."""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT)
    report: dict[str, Any] = {"cases": {}, "runtime": {}}
    train_x = np.arange(80, dtype="float64")
    heldout_x = np.arange(80, 110, dtype="float64")
    reg_train = pd.DataFrame({"x": train_x, "target": 3.0 * train_x + 2.0})
    reg_heldout = pd.DataFrame({"x": heldout_x, "target": 3.0 * heldout_x + 2.0})
    class_x = np.r_[np.linspace(-4.0, -0.2, 50), np.linspace(0.2, 4.0, 50)]
    class_train = pd.DataFrame({"x": class_x, "target": (class_x > 0).astype("int64")})
    class_test_x = np.r_[np.linspace(-2.0, -0.3, 10), np.linspace(0.3, 2.0, 10)]
    class_heldout = pd.DataFrame({"x": class_test_x, "target": (class_test_x > 0).astype("int64")})
    with tempfile.TemporaryDirectory(prefix="skyulf-sm22-live-") as directory:
        root = Path(directory)
        candidate_uri = None
        for engine in ("pandas", "polars"):
            for task, train, heldout, required in (
                ("regression", reg_train, reg_heldout, REGRESSION_METRICS),
                ("classification", class_train, class_heldout, CLASSIFICATION_METRICS),
            ):
                case = f"{engine}_{task}"
                path = root / case
                artifact = _fit(path, engine, task, train, heldout)
                metrics = evaluate_local_holdout(
                    artifact, _native(heldout, engine), target_column="target"
                )
                if not required <= metrics.keys():
                    raise AssertionError(f"{case} lacks {sorted(required - metrics.keys())}")
                run_id, model_uri = _log_metrics(
                    f"sm22-metrics-{case}",
                    metrics,
                    artifact_path=path if case == "pandas_regression" else None,
                )
                report["cases"][case] = {"run_id": run_id, "metrics": metrics}
                if case == "pandas_regression":
                    candidate_uri = model_uri
        if candidate_uri is None:
            raise AssertionError("Candidate model was not packaged.")
        champion_train = reg_train.copy()
        champion_train["target"] += 10.0
        champion_path = root / "champion"
        champion_artifact = _fit(champion_path, "pandas", "regression", champion_train, reg_heldout)
        champion_metrics = evaluate_local_holdout(
            champion_artifact, reg_heldout, target_column="target"
        )
        champion_run_id, champion_uri = _log_metrics(
            "sm22-metrics-champion",
            champion_metrics,
            artifact_path=champion_path,
        )
        if champion_uri is None:
            raise AssertionError("Champion model was not packaged.")
        champion_version = register_model(
            champion_uri, MODEL_NAME, tracking_uri="databricks", registry_uri="databricks-uc"
        )
        candidate_version = register_model(
            candidate_uri, MODEL_NAME, tracking_uri="databricks", registry_uri="databricks-uc"
        )
        champion = resolve_model(
            MODEL_NAME,
            version=champion_version.version,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
        )
        candidate = resolve_model(
            MODEL_NAME,
            version=candidate_version.version,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
        )
        comparison = compare_registered_local_models(
            candidate,
            champion,
            reg_heldout,
            target_column="target",
            dataset_id="synthetic-sm22-metrics-r1/holdout",
            metric="heldout_mse",
            min_improvement=1.0,
            quality_threshold=1.0,
            max_rows=100,
            max_bytes=100_000,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
        )
        if not comparison.eligible or comparison.improvement is None:
            raise AssertionError(f"UC comparison rejected candidate: {comparison.reason}")
        report["comparison"] = {
            "model_name": MODEL_NAME,
            "candidate_version": candidate.version,
            "champion_version": champion.version,
            "metric": comparison.metric,
            "candidate_mse": comparison.candidate_metrics["heldout_mse"],
            "champion_mse": comparison.champion_metrics["heldout_mse"]
            if comparison.champion_metrics is not None
            else None,
            "eligible": comparison.eligible,
            "champion_run_id": champion_run_id,
        }
    report["runtime"] = {
        name: version(name)
        for name in ("skyulf-core", "mlflow", "pandas", "polars", "scikit-learn")
    }
    return report


runtime_dbutils: Any = globals().get("dbutils")
if runtime_dbutils is not None:
    runtime_dbutils.notebook.exit(json.dumps(run(), default=str))
