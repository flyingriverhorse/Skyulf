# Databricks notebook source
"""Run independent Spark monitoring or explicitly prepare legacy model references."""

import json

from skyulf.integrations.databricks.jobs.monitoring.spark_monitoring_job import (
    run_project_monitoring_notebook,
)
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import (
    notebook_task,
    result_summary,
)

if __name__ == "__main__":
    with notebook_task("monitor_project", globals()["dbutils"]):
        output = run_project_monitoring_notebook(globals()["spark"], globals()["dbutils"])
        print(result_summary(output))

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
