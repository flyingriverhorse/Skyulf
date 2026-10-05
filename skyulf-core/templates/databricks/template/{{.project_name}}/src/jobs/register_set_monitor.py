# Databricks notebook source
"""Register activated model-set components before their first scoring run."""

import json

from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import (
    notebook_task,
    result_summary,
)
from skyulf.integrations.databricks.model_sets.monitoring_model_set import (
    run_set_monitor_enrollment_notebook,
)

if __name__ == "__main__":
    with notebook_task("register_set_monitor", globals()["dbutils"]):
        output = run_set_monitor_enrollment_notebook(globals()["spark"], globals()["dbutils"])
        print(result_summary(output))

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(json.dumps(output, sort_keys=True))
