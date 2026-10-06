# Databricks notebook source
"""Display independent drift, observed performance and performance loss evidence."""

import json

from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task
from skyulf.integrations.databricks.observability.monitoring.monitoring_output import (
    run_monitoring_report_notebook,
)

if __name__ == "__main__":
    with notebook_task("monitoring_report", globals()["dbutils"]):
        output = run_monitoring_report_notebook(
            globals()["spark"], globals()["dbutils"], display_html=globals().get("displayHTML")
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
