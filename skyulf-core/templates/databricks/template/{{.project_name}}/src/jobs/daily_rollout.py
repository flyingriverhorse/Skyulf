# Databricks notebook source
"""Observe or reconcile one durable serving stage under the configured writer policy."""

import json

from skyulf.integrations.databricks.jobs.serving_job import run_daily_rollout_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import (
    notebook_task,
    result_summary,
)

if __name__ == "__main__":
    with notebook_task("daily_rollout", globals()["dbutils"]):
        output = run_daily_rollout_notebook(globals()["spark"], globals()["dbutils"])
        print(result_summary(output))

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
