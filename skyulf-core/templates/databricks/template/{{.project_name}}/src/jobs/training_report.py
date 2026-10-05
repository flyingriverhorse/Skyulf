# Databricks notebook source
"""Finalize lifecycle status and report verified results and next actions."""

from skyulf.integrations.databricks.jobs.shared.job_runtime import run_lifecycle_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

if __name__ == "__main__":
    with notebook_task("training_report", globals()["dbutils"]):
        output = run_lifecycle_notebook(
            globals()["spark"],
            globals()["dbutils"],
            phase="complete",
            exit_notebook=False,
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
