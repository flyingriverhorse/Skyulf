# Databricks notebook source
"""Confirm a read-only test principal cannot mutate the isolated UC champion alias."""

import json
from typing import Any

import mlflow  # ty: ignore[unresolved-import]

MODEL = "workspace.skyulf_sm24a_20260923.skyulf_sm22b_promotion_r1"


def run() -> dict[str, str]:
    """Read the allowed champion, then require a real registry write denial."""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    original = str(client.get_model_version_by_alias(MODEL, "champion").version)
    if original != "1":
        raise AssertionError(f"Test alias must be restored to version 1, got {original}.")
    try:
        client.set_registered_model_alias(MODEL, "champion", "2")
    except Exception as exc:  # noqa: BLE001 - Databricks may wrap permission errors
        code = str(getattr(exc, "error_code", "")).upper()
        if "PERMISSION_DENIED" not in code and "PERMISSION_DENIED" not in str(exc).upper():
            raise
    else:
        raise AssertionError("Restricted principal unexpectedly changed the champion alias.")
    after = str(client.get_model_version_by_alias(MODEL, "champion").version)
    if after != original:
        raise AssertionError("Denied alias write changed the champion.")
    return {"read_version": original, "final_version": after, "denial": "PERMISSION_DENIED"}


runtime_dbutils: Any = globals().get("dbutils")
if runtime_dbutils is not None:
    runtime_dbutils.notebook.exit(json.dumps(run()))
