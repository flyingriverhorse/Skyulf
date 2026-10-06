# Databricks notebook source
"""Run scoring with a fixed role that cannot dispatch lifecycle mutations."""

from skyulf.integrations.databricks.jobs.shared.job_runtime import run_score_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

if __name__ == "__main__":
    with notebook_task("score", globals()["dbutils"]):
        output = run_score_notebook(
            globals()["spark"],
            globals()["dbutils"],
            display_html=globals().get("displayHTML"),
            exit_notebook=False,
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
