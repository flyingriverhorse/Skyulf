# Databricks notebook source
"""Inspect the saved model and publication bindings before creating an online endpoint."""

import json

from skyulf.integrations.databricks.jobs.serving_job import run_online_serving_setup_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import (
    notebook_task,
    result_summary,
)

if __name__ == "__main__":
    with notebook_task("online_feature_lookup_setup", globals()["dbutils"]):
        output = run_online_serving_setup_notebook(globals()["spark"], globals()["dbutils"])
        print(result_summary(output))

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
