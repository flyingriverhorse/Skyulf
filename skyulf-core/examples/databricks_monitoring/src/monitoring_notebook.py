# Databricks notebook source
"""Manual central monitoring job with one serialized Delta writer."""

from skyulf.integrations.databricks.monitoring import run_monitoring_notebook

run_monitoring_notebook(globals()["spark"], globals()["dbutils"])
