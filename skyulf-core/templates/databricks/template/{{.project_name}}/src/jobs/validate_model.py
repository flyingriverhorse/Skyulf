# Databricks notebook source
"""Validate the single fitted model before registration; no candidate competition."""

from skyulf.integrations.databricks.jobs.shared.job_runtime import run_lifecycle_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

if __name__ == "__main__":
    with notebook_task("validate_model", globals()["dbutils"]):
        output = run_lifecycle_notebook(
            globals()["spark"],
            globals()["dbutils"],
            phase="select_best_model",
            exit_notebook=False,
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
