# Databricks notebook source
"""Apply fixed cleanup and save the train/holdout partitions."""

from skyulf.integrations.databricks.job_runtime import run_lifecycle_notebook

if __name__ == "__main__":
    output = run_lifecycle_notebook(
        globals()["spark"],
        globals()["dbutils"],
        phase="prepare_dataset",
        exit_notebook=False,
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
