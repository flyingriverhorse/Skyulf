# Databricks notebook source
"""models report using the pinned invocation and saved evidence."""

from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task
from skyulf.integrations.databricks.jobs.training.training_node_notebook import (
    run_models_report_notebook,
)

if __name__ == "__main__":
    with notebook_task("models_report", globals()["dbutils"]):
        output = run_models_report_notebook(
            globals()["dbutils"],
            display_html=globals().get("displayHTML"),
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
