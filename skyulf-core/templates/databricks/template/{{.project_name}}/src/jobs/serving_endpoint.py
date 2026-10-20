# Databricks notebook source
"""Serve one concrete version through the owned SQL, REST and Python endpoint."""

import json

from skyulf.integrations.databricks.jobs.serving_endpoint_job import run_serving_endpoint_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

if __name__ == "__main__":
    with notebook_task("serving_endpoint", globals()["dbutils"]):
        output = run_serving_endpoint_notebook(globals()["spark"], globals()["dbutils"])
        print(json.dumps(output, indent=2, allow_nan=False))

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, allow_nan=False))
