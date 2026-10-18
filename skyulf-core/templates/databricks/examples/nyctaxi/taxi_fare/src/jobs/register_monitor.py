# Databricks notebook source
"""Run the visible register_monitor task using the shared monitoring implementation."""

import json

from skyulf.integrations.databricks.jobs.monitoring.monitoring_tasks import (
    run_monitor_enrollment_notebook,
)
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import (
    notebook_task,
    result_summary,
)

if __name__ == "__main__":
    with notebook_task("register_monitor", globals()["dbutils"]):
        output = run_monitor_enrollment_notebook(globals()["spark"], globals()["dbutils"])
        print(result_summary(output))

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
