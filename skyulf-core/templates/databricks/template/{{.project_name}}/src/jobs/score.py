# Databricks notebook source
"""Run scoring with a fixed role that cannot dispatch lifecycle mutations."""

from skyulf.integrations.databricks.job_runtime import run_score_notebook

if __name__ == "__main__":
    output = run_score_notebook(
        globals()["spark"],
        globals()["dbutils"],
        display_html=globals().get("displayHTML"),
        exit_notebook=False,
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
