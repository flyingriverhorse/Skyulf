# Databricks notebook source
"""Validate settings and pin one lifecycle invocation."""

from skyulf.integrations.databricks.job_runtime import run_lifecycle_notebook

if __name__ == "__main__":
    output = run_lifecycle_notebook(
        globals()["spark"],
        globals()["dbutils"],
        phase="initialize",
        preprocessing_path="../src/features",
        exit_notebook=False,
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
