# Databricks notebook source
"""Enroll actual endpoint telemetry with known outcomes for the 32 real test rows.

Run after serve_and_verify.py and after its request appears in the payload view.
This creates a separate labels table, preserving the source labels. The short
report-only window is for acceptance, not a recommended production policy.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
dbutils.widgets.dropdown("project", "diagnostic_single", ["diagnostic_single", "bike_demand_set"])  # noqa: F821
dbutils.widgets.text("model_version", "1")  # noqa: F821
dbutils.widgets.text("monitoring_job_id", "")  # noqa: F821
project = dbutils.widgets.get("project")  # noqa: F821
version = dbutils.widgets.get("model_version")  # noqa: F821
job_id = int(dbutils.widgets.get("monitoring_job_id"))  # noqa: F821
if project not in {"diagnostic_single", "bike_demand_set"} or not version.isdecimal():
    raise ValueError("Choose a prepared project and its concrete model version.")
schema = f"workspace.{project}"
evidence_path = Path(f"/Volumes/workspace/{project}/artifacts/acceptance-v{version}.json")
evidence = json.loads(evidence_path.read_text())
payload = evidence["logging_table"]
requests = (
    spark.table(payload)
    .where(f"client_request_id = 'task233-{project}-parity-v{version}' AND status_code = 200")
    .select("databricks_request_id")
    .distinct()
    .collect()
)
if not requests:
    raise ValueError("The accepted request is not in telemetry yet; wait for delivery, then rerun.")
keys = (
    spark.table(f"{schema}.observations_score")
    .orderBy("record_id")
    .limit(evidence["real_rows"])
    .select("record_id")
    .collect()
)
labels = {
    row.record_id: row.asDict()
    for row in spark.table(f"{schema}.labels")
    .where("record_id IN (" + ",".join(str(row.record_id) for row in keys) + ")")
    .collect()
}
targets = ["malignant"] if project == "diagnostic_single" else ["casual", "registered"]
now = datetime.now(UTC)
rows = [
    {
        "databricks_request_id": request.databricks_request_id,
        "request_row_index": index,
        "available_at": now,
        **{target: labels[key.record_id][target] for target in targets},
    }
    for request in requests
    for index, key in enumerate(keys)
]
label_table = f"{schema}.serving_labels_v{version}"
spark.createDataFrame(rows).writeTo(label_table).using("delta").create()
enrollments = []
for component in evidence["components"]:
    classification = project == "diagnostic_single"
    config = {
        "environment": "test",
        "project": project,
        "model_name": component["name"],
        "model_version": component["version"],
        "serving_endpoint": evidence["endpoint"],
        "source_table": payload,
        "prediction_table": payload,
        "execution_engine": "spark",
        "max_rows": 20000,
        "reference_namespace": schema,
        "label_table": label_table,
        "result_available_at_column": "available_at",
        "performance_policy": {
            "mode": "report",
            "metric": "f1_weighted" if classification else "rmse",
            "direction": "higher" if classification else "lower",
            "baseline": {"kind": "training_holdout", "model_version": component["version"]},
            "tolerance": 0.05 if classification else 15.0,
            "tolerance_mode": "absolute",
            "window_hours": 1 / 6,
            "label_delay_hours": 0,
            "minimum_labeled_rows": 20,
            "minimum_label_coverage": 0.8,
            "consecutive_windows": 1,
        },
    }
    if not classification:
        config.update(
            model_set_name=evidence["model_uri"].split("/")[1],
            model_set_version=version,
            model_set_branch=component["branch"],
        )
    enrollments.append(config)
request_path = evidence_path.with_name(f"serving-enrollments-v{version}.json")
request_path.write_text(json.dumps(enrollments, indent=2))
run = WorkspaceClient().jobs.run_now(
    job_id=job_id,
    job_parameters={
        "monitoring_action": "enroll_serving",
        "monitoring_serving_enrollments": json.dumps(enrollments),
    },
)
dbutils.notebook.exit(json.dumps({"enrollment_run_id": run.run_id, "label_rows": len(rows)}))  # noqa: F821
