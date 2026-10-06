# Databricks notebook source
"""Validate the successful scoring receipt before starting the native monitoring child job."""

import json

from skyulf.integrations.databricks.jobs.monitoring.monitoring_tasks import (
    prepare_scoring_monitoring_notebook,
)
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import (
    notebook_task,
    result_summary,
)

if __name__ == "__main__":
    with notebook_task("prepare_monitoring", globals()["dbutils"]):
        output = prepare_scoring_monitoring_notebook(globals()["dbutils"])
        print(result_summary(output))

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
