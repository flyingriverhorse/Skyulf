# Databricks notebook source
"""Generate optional MLflow Image grid charts from the completed training invocation."""

from skyulf.integrations.databricks.evaluation_chart_task import run_evaluation_charts_notebook

if __name__ == "__main__":
    output = run_evaluation_charts_notebook(globals()["dbutils"])

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
