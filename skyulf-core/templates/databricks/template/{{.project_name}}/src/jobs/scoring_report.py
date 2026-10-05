# Databricks notebook source
"""Display the actual completed scoring or CDF recovery result."""

from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task
from skyulf.integrations.databricks.scoring.incremental.scoring_recovery import (
    run_scoring_report_notebook,
)

if __name__ == "__main__":
    with notebook_task("scoring_report", globals()["dbutils"]):
        output = run_scoring_report_notebook(
            globals()["spark"],
            globals()["dbutils"],
            display_html=globals().get("displayHTML"),
            exit_notebook=False,
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
