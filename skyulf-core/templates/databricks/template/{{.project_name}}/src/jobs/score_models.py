# Databricks notebook source
"""Score the pinned coherent model set using its saved component and rule code."""

from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task
from skyulf.integrations.databricks.model_sets.model_set_project import run_model_set_score_notebook

if __name__ == "__main__":
    with notebook_task("score_models", globals()["dbutils"]):
        output = run_model_set_score_notebook(
            globals()["spark"],
            globals()["dbutils"],
            display_html=globals().get("displayHTML"),
            exit_notebook=False,
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
