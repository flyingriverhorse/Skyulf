# Databricks notebook source
"""train model using the pinned invocation and saved evidence."""

from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task
from skyulf.integrations.databricks.jobs.training.training_node_notebook import (
    run_model_training_notebook,
)

if __name__ == "__main__":
    with notebook_task("train_model", globals()["dbutils"]):
        output = run_model_training_notebook(
            globals()["spark"],
            globals()["dbutils"],
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
