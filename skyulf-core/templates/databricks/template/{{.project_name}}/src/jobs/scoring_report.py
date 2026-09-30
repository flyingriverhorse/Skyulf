# Databricks notebook source
"""Display the actual completed scoring or CDF recovery result."""

from skyulf.integrations.databricks.scoring_recovery import run_scoring_report_notebook

if __name__ == "__main__":
    output = run_scoring_report_notebook(
        globals()["spark"],
        globals()["dbutils"],
        display_html=globals().get("displayHTML"),
        exit_notebook=False,
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
