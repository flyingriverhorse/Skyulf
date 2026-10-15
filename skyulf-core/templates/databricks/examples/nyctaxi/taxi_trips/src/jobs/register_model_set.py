# Databricks notebook source
"""register model set with frozen training and quality evidence."""

from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task
from skyulf.integrations.databricks.jobs.training.training_node_notebook import (
    run_model_set_stage_notebook,
)

if __name__ == "__main__":
    with notebook_task("register_model_set", globals()["dbutils"]):
        output = run_model_set_stage_notebook(
            globals()["spark"],
            globals()["dbutils"],
            phase="register_model_set",
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
