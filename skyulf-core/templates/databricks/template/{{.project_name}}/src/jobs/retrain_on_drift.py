# Databricks notebook source
"""Request the existing training job when opted-in drift guards allow it."""

import json

from skyulf.integrations.databricks.retraining_task import run_retraining_notebook

if __name__ == "__main__":
    output = run_retraining_notebook(
        globals()["spark"], globals()["dbutils"], display_html=globals().get("displayHTML")
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
