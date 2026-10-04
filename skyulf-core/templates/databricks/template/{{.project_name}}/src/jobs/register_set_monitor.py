# Databricks notebook source
"""Register activated model-set components before their first scoring run."""

import json

from skyulf.integrations.databricks.monitoring_model_set import run_set_monitor_enrollment_notebook

if __name__ == "__main__":
    output = run_set_monitor_enrollment_notebook(globals()["spark"], globals()["dbutils"])

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
