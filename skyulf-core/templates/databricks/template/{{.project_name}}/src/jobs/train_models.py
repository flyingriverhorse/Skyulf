# Databricks notebook source
"""Train independent target candidates; activation and multi-target scoring follow in SM-36C."""

from skyulf.integrations.databricks.branch_notebook import run_branch_training_notebook

if __name__ == "__main__":
    output = run_branch_training_notebook(
        globals()["spark"],
        globals()["dbutils"],
        exit_notebook=False,
    )

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
