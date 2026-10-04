# Databricks notebook source
"""Run the visible register_monitor task using the shared monitoring implementation."""

import json

from skyulf.integrations.databricks.monitoring_tasks import run_monitor_enrollment_notebook

if __name__ == "__main__":
    output = run_monitor_enrollment_notebook(globals()["spark"], globals()["dbutils"])

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
