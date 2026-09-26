# Databricks notebook source
"""Run lifecycle actions through the installed Skyulf job adapter."""

from skyulf.integrations.databricks.job_runtime import run_notebook

if __name__ == "__main__":
    output = run_notebook(
        globals()["spark"],
        globals()["dbutils"],
        task_role="lifecycle",
        preprocessing_path="../src/preprocessing.py",
        display_html=globals().get("displayHTML"),
        exit_notebook=False,
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
