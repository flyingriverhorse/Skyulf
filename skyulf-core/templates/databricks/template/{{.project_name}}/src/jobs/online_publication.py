# Databricks notebook source
"""Trigger publication of explicitly selected latest features into an existing store."""

import json

from skyulf.integrations.databricks.jobs.serving_job import run_online_publication_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import (
    notebook_task,
    result_summary,
)

if __name__ == "__main__":
    with notebook_task("online_publication", globals()["dbutils"]):
        output = run_online_publication_notebook(globals()["spark"], globals()["dbutils"])
        print(result_summary(output))

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
