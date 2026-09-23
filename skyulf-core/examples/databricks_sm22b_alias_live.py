# Databricks notebook source
"""Isolated Unity Catalog validation of guarded Skyulf champion promotion."""

import json
from typing import Any

import mlflow  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd

from skyulf.integrations.mlflow.promotion import (
    AliasConflictError,
    DeltaAliasAdmission,
    alias_resource_id,
    promote_candidate,
    rollback_promotion,
)
from skyulf.integrations.mlflow.registry import resolve_model
from skyulf.integrations.mlflow.validation import compare_registered_local_models

MODEL = "workspace.skyulf_sm24a_20260923.skyulf_sm22b_promotion_r1"
CONTROL = "workspace.skyulf_sm24a_20260923.skyulf_sm22b_alias_admission_r1"
REFERENCE_URI = "runs:/3fcb9236b2434eb1bec2eb42af8b1d45/model"
CANDIDATE_URI = "runs:/546df4fed4d443999990707df7255335/model"


def run() -> dict[str, Any]:
    """Reuse two disposable versions, promote, prove conflict, and roll back."""
    spark_session: Any = globals()["spark"]
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    reference = client.get_model_version(MODEL, "1")
    candidate = client.get_model_version(MODEL, "2")
    if reference.source != REFERENCE_URI or candidate.source != CANDIDATE_URI:
        raise AssertionError("Existing isolated model versions have unexpected sources.")
    reference_version = str(reference.version)
    candidate_version = str(candidate.version)
    if str(client.get_model_version_by_alias(MODEL, "champion").version) != reference_version:
        raise AssertionError("Test champion alias differs from the original version.")
    key = alias_resource_id(MODEL, "champion")
    rows = spark_session.table(CONTROL).limit(2).collect()
    if len(rows) != 1 or rows[0]["target_id"] != key or rows[0]["owner"] is not None:
        raise AssertionError("Existing alias admission row has unexpected state.")
    admission = DeltaAliasAdmission(spark_session, CONTROL)
    x = np.arange(80, 110, dtype="float64")
    heldout = pd.DataFrame({"x": x, "target": 3.0 * x + 2.0})
    report = compare_registered_local_models(
        resolve_model(
            MODEL,
            version=candidate_version,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
        ),
        resolve_model(
            MODEL,
            version=reference_version,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
        ),
        heldout,
        target_column="target",
        dataset_id="synthetic-sm22b-r1/heldout",
        metric="heldout_mse",
        min_improvement=1.0,
        quality_threshold=1.0,
        max_rows=100,
        max_bytes=100_000,
        tracking_uri="databricks",
        registry_uri="databricks-uc",
    )
    if not report.eligible:
        raise AssertionError(f"Candidate failed comparison: {report.reason}")
    with admission.hold(key):
        try:
            with admission.hold(key):
                raise AssertionError("Contender entered while owner held admission.")
        except AliasConflictError:
            pass
    receipt = promote_candidate(
        report,
        heldout,
        target_column="target",
        expected_champion_version=reference_version,
        admission=admission,
        max_rows=100,
        max_bytes=100_000,
        tracking_uri="databricks",
        registry_uri="databricks-uc",
    )
    if str(client.get_model_version_by_alias(MODEL, "champion").version) != candidate_version:
        raise AssertionError("Promotion did not select the candidate.")
    reversal = rollback_promotion(
        receipt,
        expected_current_version=candidate_version,
        admission=admission,
        tracking_uri="databricks",
        registry_uri="databricks-uc",
    )
    if str(client.get_model_version_by_alias(MODEL, "champion").version) != reference_version:
        raise AssertionError("Rollback did not restore the reference.")
    return {
        "model": MODEL,
        "control_table": CONTROL,
        "reference_version": reference_version,
        "candidate_version": candidate_version,
        "candidate_mse": report.candidate_metrics["heldout_mse"],
        "reference_mse": report.champion_metrics["heldout_mse"]
        if report.champion_metrics is not None
        else None,
        "promotion_event": receipt.event_id,
        "rollback_event": reversal.event_id,
        "final_alias_version": reference_version,
        "conflict_rejected": True,
    }


runtime_dbutils: Any = globals().get("dbutils")
if runtime_dbutils is not None:
    runtime_dbutils.notebook.exit(json.dumps(run()))
