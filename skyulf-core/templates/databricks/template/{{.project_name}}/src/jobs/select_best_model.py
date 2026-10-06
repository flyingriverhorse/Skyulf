# Databricks notebook source
"""Verify and select the requested candidate; currently one candidate."""

from skyulf.integrations.databricks.jobs.shared.job_runtime import run_lifecycle_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

if __name__ == "__main__":
    with notebook_task("select_best_model", globals()["dbutils"]):
        output = run_lifecycle_notebook(
            globals()["spark"],
            globals()["dbutils"],
            phase="select_best_model",
            exit_notebook=False,
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
