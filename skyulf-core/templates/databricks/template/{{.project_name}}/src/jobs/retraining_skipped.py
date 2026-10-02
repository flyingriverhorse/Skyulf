# Databricks notebook source
"""Explain the false retraining branch using the saved eligibility decision."""

import json

from skyulf.integrations.databricks.job_output import output_table

if __name__ == "__main__":
    output = globals()["dbutils"].jobs.taskValues.get(
        taskKey="check_retraining", key="retraining_check"
    )
    if globals().get("displayHTML") is not None:
        globals()["displayHTML"](
            "<h2>Retraining skipped</h2>" + output_table(("Reason",), [(output["status"],)])
        )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
