# Databricks notebook source
"""train model using the pinned invocation and saved evidence."""

from skyulf.integrations.databricks.training_node_notebook import run_model_training_notebook

if __name__ == "__main__":
    output = run_model_training_notebook(
        globals()["spark"],
        globals()["dbutils"],
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
