# Databricks notebook source
"""evaluate model set with frozen training and quality evidence."""

from skyulf.integrations.databricks.training_node_notebook import run_model_set_stage_notebook

if __name__ == "__main__":
    output = run_model_set_stage_notebook(
        globals()["spark"],
        globals()["dbutils"],
        phase="evaluate_model_set",
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
