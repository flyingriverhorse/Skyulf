# Databricks notebook source
"""Display independent drift, observed performance and performance loss evidence."""

import json

from skyulf.integrations.databricks.monitoring_output import run_monitoring_report_notebook

if __name__ == "__main__":
    output = run_monitoring_report_notebook(
        globals()["spark"], globals()["dbutils"], display_html=globals().get("displayHTML")
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
