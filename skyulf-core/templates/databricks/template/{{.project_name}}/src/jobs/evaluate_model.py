# Databricks notebook source
"""Evaluate the registered candidate against champion and quality gates."""

from skyulf.integrations.databricks.job_runtime import run_lifecycle_notebook

if __name__ == "__main__":
    output = run_lifecycle_notebook(
        globals()["spark"],
        globals()["dbutils"],
        phase="compare",
        exit_notebook=False,
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
