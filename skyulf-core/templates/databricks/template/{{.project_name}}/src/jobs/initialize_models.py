# Databricks notebook source
"""initialize models using the pinned invocation and saved evidence."""

from skyulf.integrations.databricks.training_node_notebook import run_initialize_models_notebook

if __name__ == "__main__":
    output = run_initialize_models_notebook(
        globals()["spark"],
        globals()["dbutils"],
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
