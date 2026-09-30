# Databricks notebook source
"""Score the pinned coherent model set using its saved component and rule code."""

from skyulf.integrations.databricks.model_set_project import run_model_set_score_notebook

if __name__ == "__main__":
    output = run_model_set_score_notebook(
        globals()["spark"],
        globals()["dbutils"],
        display_html=globals().get("displayHTML"),
        exit_notebook=False,
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
