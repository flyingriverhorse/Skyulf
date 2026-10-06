# Databricks notebook source
"""Generate optional MLflow Image grid charts from the completed training invocation."""

from skyulf.integrations.databricks.jobs.evaluation_chart_task import run_evaluation_charts_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

if __name__ == "__main__":
    with notebook_task("generate_charts", globals()["dbutils"]):
        output = run_evaluation_charts_notebook(globals()["dbutils"])

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
