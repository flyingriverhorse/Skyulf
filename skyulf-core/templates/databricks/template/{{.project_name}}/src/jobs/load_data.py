# Databricks notebook source
"""Load and save the bounded source snapshot."""

from skyulf.integrations.databricks.jobs.shared.job_runtime import run_lifecycle_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

if __name__ == "__main__":
    with notebook_task("load_data", globals()["dbutils"]):
        output = run_lifecycle_notebook(
            globals()["spark"],
            globals()["dbutils"],
            phase="load_data",
            exit_notebook=False,
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
