# Databricks notebook source
"""shap report using the pinned invocation and saved evidence."""

from skyulf.integrations.databricks.training_node_notebook import run_shap_notebook

if __name__ == "__main__":
    output = run_shap_notebook(
        globals()["dbutils"],
        display_html=globals().get("displayHTML"),
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
