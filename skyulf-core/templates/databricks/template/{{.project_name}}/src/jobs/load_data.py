# Databricks notebook source
"""Load and save the bounded source snapshot."""

from skyulf.integrations.databricks.job_runtime import run_lifecycle_notebook

if __name__ == "__main__":
    output = run_lifecycle_notebook(
        globals()["spark"],
        globals()["dbutils"],
        phase="load_data",
        exit_notebook=False,
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
