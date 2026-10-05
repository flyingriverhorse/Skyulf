# Databricks notebook source
"""Rebuild predictions only for the score task's pinned CDF expiry request."""

from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task
from skyulf.integrations.databricks.scoring.incremental.scoring_recovery import (
    run_cdf_recovery_notebook,
)

if __name__ == "__main__":
    with notebook_task("recover_predictions", globals()["dbutils"]):
        output = run_cdf_recovery_notebook(
            globals()["spark"],
            globals()["dbutils"],
            display_html=globals().get("displayHTML"),
            exit_notebook=False,
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
