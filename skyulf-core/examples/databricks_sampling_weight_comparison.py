# Databricks notebook source
"""Prove single-model SMOTE training with and without sample weights.

Run as a Databricks Python notebook with widgets acceptance_id and experiment.
Install the current Skyulf wheel, MLflow and imbalanced-learn in its environment.
Only unique test resources are created. No model aliases are changed.
"""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from skyulf.inference.local_pipeline import predict_local_pipeline
from skyulf.integrations.databricks.local_cv import LocalCVSpec
from skyulf.integrations.databricks.local_retraining import (
    LocalTrainingSpec,
    read_training_partitions,
    train_local_candidate,
)
from skyulf.integrations.databricks.project import load_project_workflow
from skyulf.integrations.mlflow.registry import load_registered_local_pipeline, resolve_model

FEATURES = ["x", "z"]


def array_digest(values):
    """Hash ordered numeric arrays with shape to make comparisons inspectable."""
    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def make_project(root, source, weighted):
    """Load the same editable single-model layout used by generated Bundles."""
    features = root / "src/features"
    models = root / "src/modeling"
    features.mkdir(parents=True)
    models.mkdir(parents=True)
    sampling = {"method": "smote", "random_state": 42, "k_neighbors": 3}
    if weighted:
        sampling["synthetic_weight"] = "class_mean"
    step = {"name": "balance", "transformer": "Oversampling", "params": sampling}
    (features / "preprocessing.py").write_text(
        f"def build_preprocessing():\n    return {[step]!r}\n", encoding="utf-8"
    )
    (features / "__init__.py").write_text(
        "from .preprocessing import build_preprocessing\n", encoding="utf-8"
    )
    model = {
        "type": "logistic_regression",
        "params": {"class_weight": None, "max_iter": 1000, "tol": 1e-10, "random_state": 42},
    }
    column = "importance" if weighted else None
    (models / "single_model.py").write_text(
        f"WEIGHT_COLUMN = {column!r}\nMODELING = {model!r}\n"
        "def build_modeling():\n    return MODELING\n",
        encoding="utf-8",
    )
    config = load_project_workflow(
        {
            "training_layout": "single_model",
            "training_table": source,
            "record_key_columns": ["id"],
            "input_columns": FEATURES,
            "target_column": "target",
            "pipeline": {"preprocessing": [], "modeling": {}},
        },
        features,
    )
    return config, LocalTrainingSpec(
        table=source,
        version=0,
        record_key_columns=("id",),
        input_columns=tuple(FEATURES),
        target_column="target",
        max_rows=1000,
        max_bytes=10_000_000,
        stratify=True,
        random_state=42,
        weight_column=config["weight_column"],
        reserved_weight_columns=tuple(config["reserved_weight_columns"]),
        weights_python_source=config["weights_python_source"],
        weights_python_sha256=config["weights_python_sha256"],
    )


def independent_sampling(train, weighted):
    """Derive expected rows and synthetic weights outside Skyulf's sampler code."""
    X, y = SMOTE(random_state=42, k_neighbors=3).fit_resample(train[FEATURES], train.target)
    weights = None
    if weighted:
        original = train.importance.to_numpy()
        means = train.groupby("target").importance.mean().to_dict()
        weights = np.concatenate([original, [means[label] for label in y.iloc[len(train) :]]])
    assert len(X) > len(train)
    return np.asarray(X), np.asarray(y), weights


def train_with_fit_capture(spark, root, config, spec, model_name, experiment, case):
    """Observe the real fit arguments and delegate unchanged to sklearn fitting."""
    calls = []
    original_fit = LogisticRegression.fit

    def observe(model, X, y, sample_weight=None):
        """Copy evidence without changing features, labels, weights or fitted state."""
        calls.append(
            (
                np.asarray(X).copy(),
                np.asarray(y).copy(),
                None if sample_weight is None else np.asarray(sample_weight).copy(),
            )
        )
        return original_fit(model, X, y, sample_weight=sample_weight)

    with patch.object(LogisticRegression, "fit", observe):
        result = train_local_candidate(
            spark,
            spec,
            config["pipeline"],
            model_name=model_name,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
            experiment_name=experiment,
            run_name=case,
            artifact_path=root / "artifact",
            metric="heldout_accuracy",
            min_improvement=0.0,
            cv=LocalCVSpec(enabled=False),
            engine="pandas",
        )
    assert len(calls) == 1, f"Expected one direct training fit, received {len(calls)}"
    return result, calls[0]


def verify_fit(actual, expected, weighted):
    """Require exact sampled rows and correct presence or absence of fit weights."""
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[0].shape[1] == len(FEATURES)
    if weighted:
        np.testing.assert_allclose(actual[2], expected[2], rtol=0, atol=1e-12)
    else:
        assert actual[2] is None


def reload_and_compare(result, model_name, expected, holdout):
    """Reload from UC and compare parameters, predictions and probabilities to sklearn."""
    reference = resolve_model(
        model_name,
        version=result.model_version,
        tracking_uri="databricks",
        registry_uri="databricks-uc",
    )
    artifact = load_registered_local_pipeline(
        reference, tracking_uri="databricks", registry_uri="databricks-uc"
    )
    assert artifact.manifest.input_columns == tuple(FEATURES)
    assert artifact.manifest.feature_columns == tuple(FEATURES)
    fitted = artifact.pipeline.model_estimator._unwrap_tuned_model()
    assert fitted.class_weight is None
    baseline = clone(fitted).fit(expected[0], expected[1], sample_weight=expected[2])
    np.testing.assert_allclose(fitted.coef_, baseline.coef_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(fitted.intercept_, baseline.intercept_, rtol=1e-9, atol=1e-9)
    predictions = predict_local_pipeline(holdout[FEATURES], artifact)
    np.testing.assert_array_equal(predictions.prediction, baseline.predict(holdout[FEATURES]))
    probability = fitted.predict_proba(holdout[FEATURES])[:, 1]
    expected_probability = baseline.predict_proba(holdout[FEATURES])[:, 1]
    np.testing.assert_allclose(probability, expected_probability, rtol=1e-9, atol=1e-9)
    scored = holdout.copy().reset_index(drop=True)
    scored["prediction"] = predictions.prediction.to_numpy()
    scored["probability"] = probability
    scored["reference_probability"] = expected_probability
    scored["probability_error"] = abs(probability - expected_probability)
    return artifact, fitted, scored


def save_evidence(client, result, root, config, train, actual, expected, report, scored):
    """Persist raw fit inputs, source settings and numeric proof in the MLflow run."""
    directory = root / "proof"
    directory.mkdir()
    rows = pd.DataFrame(actual[0], columns=FEATURES)
    rows["target"] = actual[1]
    rows["is_synthetic"] = np.arange(len(rows)) >= len(train)
    rows["actual_sample_weight"] = actual[2] if actual[2] is not None else np.nan
    rows["expected_sample_weight"] = expected[2] if expected[2] is not None else np.nan
    rows.to_csv(directory / "actual_fit_inputs.csv", index=False)
    train.to_csv(directory / "original_training_rows.csv", index=False)
    scored.to_csv(directory / "holdout_predictions.csv", index=False)
    (directory / "proof.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "resolved_workflow.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    client.log_artifacts(result.run_id, str(directory), artifact_path="sampling_weight_proof")
    client.log_artifacts(result.run_id, str(root / "src"), artifact_path="sampling_weight_project")
    downloaded = client.download_artifacts(result.run_id, "sampling_weight_proof/proof.json")
    assert json.loads(Path(downloaded).read_text(encoding="utf-8")) == report


def run_case(spark, root, source, model_name, experiment, weighted):
    """Run one complete source-to-registered-model acceptance case."""
    case = "smote_weighted" if weighted else "smote_unweighted"
    config, spec = make_project(root, source, weighted)
    _, train, holdout, _ = read_training_partitions(spark, spec, engine="pandas", temporal_cv=False)
    expected = independent_sampling(train, weighted)
    result, actual = train_with_fit_capture(spark, root, config, spec, model_name, experiment, case)
    verify_fit(actual, expected, weighted)
    artifact, fitted, scored = reload_and_compare(result, model_name, expected, holdout)
    summary = artifact.pipeline.config.get("training_weights")
    if weighted:
        assert summary["weight_column"] == "importance" and summary["count"] == len(train)
    else:
        assert summary is None
    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    assert not client.get_registered_model(model_name).aliases
    report = {
        "case": case,
        "run_id": result.run_id,
        "model_version": result.model_version,
        "source_version": 0,
        "source_train_rows": len(train),
        "fit_rows": len(actual[0]),
        "synthetic_rows": len(actual[0]) - len(train),
        "holdout_rows": len(holdout),
        "sample_weight_present": actual[2] is not None,
        "synthetic_weight": "class_mean" if weighted else None,
        "class_weight": fitted.class_weight,
        "feature_columns": list(artifact.manifest.feature_columns),
        "sampled_features_sha256": array_digest(actual[0]),
        "sampled_labels_sha256": array_digest(actual[1]),
        "sample_weight_sha256": array_digest(actual[2]) if weighted else None,
        "sample_weight_sum": float(actual[2].sum()) if weighted else None,
        "class_mean_weights": {
            str(k): float(v) for k, v in train.groupby("target").importance.mean().items()
        }
        if weighted
        else {},
        "max_reference_probability_error": float(scored.probability_error.max()),
        "coefficients": fitted.coef_.tolist(),
        "intercept": fitted.intercept_.tolist(),
        "train_key_sha256": holdout.attrs["train_key_sha256"],
        "holdout_key_sha256": holdout.attrs["holdout_key_sha256"],
        "actual_fit_matches_independent_smote": True,
        "actual_weights_match_reference": True,
        "registered_reload_matches_reference": True,
        "weight_free_scoring": True,
        "aliases_unchanged": True,
    }
    save_evidence(client, result, root, config, train, actual, expected, report, scored)
    scored.insert(0, "case_name", case)
    return report, scored


def main(spark, dbutils):
    """Compare two reproducible cases and persist verified Delta predictions."""
    acceptance_id = dbutils.widgets.get("acceptance_id")
    experiment = dbutils.widgets.get("experiment")
    assert acceptance_id.replace("_", "").isalnum()
    schema = f"workspace.skyulf_sampling_ab_{acceptance_id}"
    source, model_name = f"{schema}.source", f"{schema}.single_model"
    spark.sql(f"CREATE SCHEMA {schema}")
    spark.sql(
        f"CREATE TABLE {source} USING DELTA AS SELECT id, "
        "CAST((id % 17) / 8.0 AS DOUBLE) AS x, "
        "CAST((id % 11) / 5.0 AS DOUBLE) AS z, "
        "CAST(CASE WHEN id % 5 = 0 OR id % 13 = 0 THEN 1 ELSE 0 END AS BIGINT) AS target, "
        "CAST(1 + pow((id * 13) % 17, 2) AS DOUBLE) AS importance FROM range(240)"
    )
    reports, predictions = [], []
    with TemporaryDirectory(prefix="skyulf-sampling-ab-") as directory:
        for weighted in (True, False):
            root = Path(directory) / ("weighted" if weighted else "unweighted")
            report, scored = run_case(spark, root, source, model_name, experiment, weighted)
            reports.append(report)
            predictions.append(scored)
    for key in (
        "train_key_sha256",
        "holdout_key_sha256",
        "sampled_features_sha256",
        "sampled_labels_sha256",
    ):
        assert reports[0][key] == reports[1][key], key
    probability_delta = float(np.max(abs(predictions[0].probability - predictions[1].probability)))
    assert probability_delta > 1e-4, "Weighting must observably change this nonuniform fixture."
    combined = pd.concat(predictions, ignore_index=True)
    prediction_table = f"{schema}.predictions"
    spark.createDataFrame(combined).write.format("delta").saveAsTable(prediction_table)
    persisted = spark.table(prediction_table).toPandas()
    assert len(persisted) == 96
    columns = [
        "case_name",
        *FEATURES,
        "target",
        "prediction",
        "probability",
        "reference_probability",
        "probability_error",
    ]
    pd.testing.assert_frame_equal(
        persisted[columns].sort_values(["case_name", *FEATURES]).reset_index(drop=True),
        combined[columns].sort_values(["case_name", *FEATURES]).reset_index(drop=True),
        check_dtype=False,
    )
    proof = {
        "status": "passed",
        "source": source,
        "model": model_name,
        "prediction_table": prediction_table,
        "prediction_rows": len(persisted),
        "same_source_split_and_smote_rows": True,
        "max_weighted_vs_unweighted_probability_difference": probability_delta,
        "cases": reports,
    }
    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    for report in reports:
        client.log_dict(report["run_id"], proof, "sampling_weight_proof/comparison.json")
    dbutils.notebook.exit(json.dumps(proof))


if __name__ == "__main__":
    main(globals()["spark"], globals()["dbutils"])
