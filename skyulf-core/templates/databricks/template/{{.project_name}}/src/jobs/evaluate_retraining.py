# Databricks notebook source
"""Evaluate either opted-in trigger and request one training job through shared guards."""

import json

from skyulf.integrations.databricks.jobs.lifecycle.retraining_task import run_retraining_notebook
from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

if __name__ == "__main__":
    with notebook_task("evaluate_retraining", globals()["dbutils"]):
        output = run_retraining_notebook(
            globals()["spark"], globals()["dbutils"], display_html=globals().get("displayHTML")
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
