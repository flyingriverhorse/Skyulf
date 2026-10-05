# Databricks notebook source
"""Run independent Spark monitoring or explicitly prepare legacy model references."""

import json

from skyulf.integrations.databricks.spark_monitoring_job import run_project_monitoring_notebook

if __name__ == "__main__":
    output = run_project_monitoring_notebook(globals()["spark"], globals()["dbutils"])

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
