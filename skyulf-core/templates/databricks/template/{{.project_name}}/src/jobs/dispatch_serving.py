# Databricks notebook source
"""Submit the separate serving run without waiting inside the training timeout."""

import json

from skyulf.integrations.databricks.jobs.serving_handoff import run_serving_dispatch_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

if __name__ == "__main__":
    with notebook_task("dispatch_serving", globals()["dbutils"]):
        output = run_serving_dispatch_notebook(globals()["dbutils"])
        print(json.dumps(output))

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output))
