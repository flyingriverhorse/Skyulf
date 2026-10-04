# Databricks notebook source
"""model set decision with frozen training and quality evidence."""

from skyulf.integrations.databricks.training_node_notebook import run_model_set_stage_notebook

if __name__ == "__main__":
    output = run_model_set_stage_notebook(
        globals()["spark"],
        globals()["dbutils"],
        phase="model_decision",
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
