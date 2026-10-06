# Databricks notebook source
"""model set decision with frozen training and quality evidence."""

from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task
from skyulf.integrations.databricks.jobs.training.training_node_notebook import (
    run_model_set_stage_notebook,
)

if __name__ == "__main__":
    with notebook_task("model_set_decision", globals()["dbutils"]):
        output = run_model_set_stage_notebook(
            globals()["spark"],
            globals()["dbutils"],
            phase="model_decision",
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
