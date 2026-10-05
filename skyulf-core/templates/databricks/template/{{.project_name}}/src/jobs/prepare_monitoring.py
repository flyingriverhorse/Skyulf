# Databricks notebook source
"""Validate the successful scoring receipt before starting the native monitoring child job."""

import json

from skyulf.integrations.databricks.monitoring_tasks import prepare_scoring_monitoring_notebook

if __name__ == "__main__":
    output = prepare_scoring_monitoring_notebook(globals()["dbutils"])

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
