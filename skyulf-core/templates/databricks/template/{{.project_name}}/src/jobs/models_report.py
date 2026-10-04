# Databricks notebook source
"""models report using the pinned invocation and saved evidence."""

from skyulf.integrations.databricks.training_node_notebook import run_models_report_notebook

if __name__ == "__main__":
    output = run_models_report_notebook(
        globals()["dbutils"],
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
