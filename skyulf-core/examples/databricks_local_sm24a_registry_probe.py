# Databricks notebook source
"""Verify unknown-version rejection and alias pinning on test-owned UC models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow  # ty: ignore[unresolved-import]
import numpy as np

from skyulf.integrations.databricks import (
    InputSource,
    LocalSourceSpec,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    PreflightError,
    prepare_local_workflow,
    score_local_source,
)
from skyulf.integrations.mlflow.registry import register_model

RECEIPT = Path(
    "/Workspace/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/training_receipt.json"
)
SOURCE = "workspace.skyulf_sm24a_20260923.skyulf_sm24a_score_source"
ALIAS = "sm24a_pinning_probe"


def _config(model: dict[str, Any], source_version: int, **selector: str) -> LocalWorkflowConfig:
    """Keep all decisions except the registry selector identical across probes."""
    return LocalWorkflowConfig(
        runtime="databricks",
        engine=model["engine"],
        source=InputSource(
            kind="uc_table",
            table=SOURCE,
            version=source_version,
            max_rows=80,
            max_bytes=4_000_000,
        ),
        model=ModelSelection(
            kind="local_pipeline",
            name=model["name"],
            tracking_uri="databricks",
            registry_uri="databricks-uc",
            **selector,
        ),
        sink=OutputSink(kind="return_frame"),
    )


def probe(spark: Any) -> dict[str, Any]:
    """Change a disposable alias while retaining the prepared concrete version."""
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    model = receipt["models"]["R1"]
    source_version = int(receipt["source_versions"]["score"])
    try:
        prepare_local_workflow(_config(model, source_version, version="99999"))
    except PreflightError as exc:
        unknown_version = tuple(issue.code for issue in exc.result.issues)
        if unknown_version != ("registry_model_missing",):
            raise AssertionError(f"Unexpected unknown-version result: {unknown_version}") from exc
    else:
        raise AssertionError("An unknown UC model version was accepted.")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    new_version = register_model(
        f"runs:/{model['run_id']}/model",
        model["name"],
        tracking_uri="databricks",
        registry_uri="databricks-uc",
    )
    if str(new_version.version) == model["version"]:
        raise AssertionError("The test alias needs two different concrete versions.")

    client.set_registered_model_alias(model["name"], ALIAS, model["version"])
    try:
        prepared = prepare_local_workflow(_config(model, source_version, alias=ALIAS))
        if prepared.preflight.model_version != model["version"]:
            raise AssertionError("Alias did not initially resolve to the first version.")
        client.set_registered_model_alias(model["name"], ALIAS, str(new_version.version))
        moved = client.get_model_version_by_alias(model["name"], ALIAS)
        if str(moved.version) != str(new_version.version):
            raise AssertionError("The disposable test alias did not move.")
        result = score_local_source(
            spark,
            LocalSourceSpec(
                table=SOURCE,
                version=source_version,
                period_start=datetime(2026, 1, 1, tzinfo=UTC),
                period_end=datetime(2026, 2, 1, tzinfo=UTC),
                row_keys=("entity_id",),
                input_columns=tuple(model["columns"]),
                max_rows=80,
                max_bytes=4_000_000,
            ),
            prepared,
        )
        expected = receipt["reference"]["R1"]["2026-01"]
        if len(result.predictions) != len(expected):
            raise AssertionError("Alias probe changed row membership.")
        for row in result.predictions.itertuples(index=False):
            if row.entity_id not in expected or not np.isclose(
                float(cast(Any, row.prediction)),
                float(expected[row.entity_id][0]),
                rtol=0,
                atol=1e-9,
            ):
                raise AssertionError(f"Pinned prediction changed for {row.entity_id}.")
        if result.diagnostics["model_version"] != model["version"]:
            raise AssertionError("Prepared workflow followed the moving alias.")
        return {
            "unknown_version": unknown_version[0],
            "alias_initial_version": model["version"],
            "alias_moved_version": str(new_version.version),
            "prepared_version": result.diagnostics["model_version"],
            "reference_match": True,
            "row_count": len(result.predictions),
            "temporary_alias_deleted": True,
        }
    finally:
        client.delete_registered_model_alias(model["name"], ALIAS)


runtime_dbutils: Any = globals().get("dbutils")
runtime_spark: Any = globals().get("spark")
if runtime_dbutils is not None and runtime_spark is not None:
    runtime_dbutils.notebook.exit(json.dumps(probe(runtime_spark)))
