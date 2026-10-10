"""Fit, evaluate and register one complete raw-input pipeline without implicit scoring."""

from pathlib import Path
from tempfile import TemporaryDirectory

from skyulf.data.dataset import SplitDataset
from skyulf.inference.pipeline_scoring import score_pipeline
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.mlflow.models.pipeline_model import log_pipeline_model
from skyulf.integrations.mlflow.registration.registry import register_model
from skyulf.integrations.mlflow.spark.spark_model import partition_safety_certificate

from .data import bounded_pandas, read_cohort

STORES = {"tracking_uri": "databricks", "registry_uri": "databricks-uc"}


def fit_customer_pipeline(frame, config, path):
    """Fit preprocessing only on training membership, leaving the holdout untouched."""
    from sklearn.model_selection import train_test_split

    train, holdout = train_test_split(
        frame,
        test_size=config["holdout_fraction"],
        random_state=config["random_state"],
        stratify=frame[config["target_column"]],
    )
    artifact = fit_workflow(
        config["pipeline"],
        SplitDataset(train=train, test=holdout),
        target_column=config["target_column"],
        artifact_path=path,
        max_rows=config["training_rows"],
        max_bytes=16 * 1024 * 1024,
    )
    partition_safety_certificate(artifact)
    return artifact, holdout


def write_optional_batch_predictions(enabled, spark, artifact, namespace, source_version):
    """Do no scoring or Spark work at all when endpoint-only training is selected."""
    if type(enabled) is not bool:
        raise ValueError("write_batch_predictions must be a boolean")
    if not enabled:
        return {"status": "skipped", "reason": "write_batch_predictions is false"}
    from .scoring import write_predictions

    rows = read_cohort(spark, namespace, source_version, "score").limit(10001).toPandas()
    if len(rows) > 10000:
        raise ValueError("Optional demo batch publication supports at most 10000 rows")
    predictions = score_pipeline(rows[list(artifact.manifest.input_columns)], artifact)
    table = f"{namespace}.batch_predictions"
    write_predictions(spark, rows.customer_id.tolist(), predictions, table, source_version)
    return {"status": "written", "table": table, "rows": len(rows)}


def train(spark, namespace, source_version, config, experiment_path):
    """Register a versioned artifact; the optional sink is a separate final decision."""
    import mlflow
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if type(config["write_batch_predictions"]) is not bool:
        raise ValueError("write_batch_predictions must be a boolean")
    columns = [*config["raw_columns"], config["target_column"]]
    frame = bounded_pandas(
        read_cohort(spark, namespace, source_version, "train").select(*columns),
        config["training_rows"],
    )
    tracking = mlflow.MlflowClient(**STORES)
    experiment = tracking.create_experiment(experiment_path)
    run = tracking.create_run(experiment, run_name="raw customer preprocessing and training")
    run_id = run.info.run_id
    try:
        with TemporaryDirectory(prefix="skyulf-raw-customer-") as directory:
            path = Path(directory) / "pipeline"
            artifact, holdout = fit_customer_pipeline(frame, config, path)
            predictions = score_pipeline(holdout[config["raw_columns"]], artifact)
            labels = holdout[config["target_column"]]
            metrics = {
                "holdout_accuracy": accuracy_score(labels, predictions.prediction),
                "holdout_f1_weighted": f1_score(labels, predictions.prediction, average="weighted"),
                "holdout_roc_auc": roc_auc_score(labels, predictions["probability_1"]),
            }
            for name, value in metrics.items():
                tracking.log_metric(run_id, name, float(value))
            tracking.log_param(run_id, "source_table", f"{namespace}.raw_customers")
            tracking.log_param(run_id, "source_version", source_version)
            tracking.log_param(run_id, "write_batch_predictions", config["write_batch_predictions"])
            tracking.log_dict(run_id, config, "training_config.json")
            tracking.log_dict(
                run_id,
                {
                    "raw_columns": list(artifact.manifest.input_columns),
                    "feature_columns": list(artifact.manifest.feature_columns),
                },
                "feature_contract.json",
            )
            uri = log_pipeline_model(
                path, run_id=run_id, artifact_path="pipeline", tracking_uri="databricks"
            )
            version = register_model(uri, f"{namespace}.customer_churn", **STORES)
            batch_predictions = write_optional_batch_predictions(
                config["write_batch_predictions"], spark, artifact, namespace, source_version
            )
        tracking.set_terminated(run_id)
    except Exception:
        tracking.set_terminated(run_id, status="FAILED")
        raise
    return {
        "model_version": str(version.version),
        "run_id": run_id,
        "experiment_id": experiment,
        "metrics": metrics,
        "training_batch_predictions": batch_predictions,
    }
