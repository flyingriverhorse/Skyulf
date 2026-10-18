# Databricks notebook source
"""Apply promotion policy or a saved approve/reject/rollback request."""

from skyulf.integrations.databricks.jobs.shared.job_runtime import run_lifecycle_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

if __name__ == "__main__":
    with notebook_task("model_decision", globals()["dbutils"]):
        output = run_lifecycle_notebook(
            globals()["spark"],
            globals()["dbutils"],
            phase="model_decision",
            exit_notebook=False,
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
