# Databricks notebook source
"""Run the visible monitor_model task using the shared monitoring implementation."""

import json

from skyulf.integrations.databricks.monitoring_tasks import run_scoring_monitor_notebook

if __name__ == "__main__":
    output = run_scoring_monitor_notebook(
        globals()["spark"], globals()["dbutils"], display_html=globals().get("displayHTML")
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
