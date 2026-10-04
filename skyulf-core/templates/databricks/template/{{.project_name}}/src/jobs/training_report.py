# Databricks notebook source
"""Finalize lifecycle status and report verified results and next actions."""

from skyulf.integrations.databricks.job_runtime import run_lifecycle_notebook

if __name__ == "__main__":
    output = run_lifecycle_notebook(
        globals()["spark"],
        globals()["dbutils"],
        phase="complete",
        exit_notebook=False,
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
