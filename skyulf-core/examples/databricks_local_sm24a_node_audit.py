# Databricks notebook source
"""Probe each remaining local FE registration with a saved full pipeline.

Every candidate is attempted independently on pandas and Polars. Failures
remain visible; a passing fixture is evidence for that configuration only.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import predict_local_pipeline
from skyulf.integrations.databricks import fit_local_workflow


def _fixture() -> pd.DataFrame:
    """Provide bounded numeric, missing, categorical, geo and date source rows."""
    rng = np.random.default_rng(42)
    count = 80
    a = rng.normal(0, 1, count)
    b = rng.normal(5, 2, count)
    missing = rng.normal(0, 1, count)
    missing[::11] = np.nan
    cat_a = rng.choice(["red", "green", "blue"], count)
    return pd.DataFrame(
        {
            "num_a": a,
            "num_b": b,
            "num_c": rng.exponential(1, count) + 0.1,
            "with_missing": missing,
            "cat_a": cat_a,
            "cat_b": rng.choice(["x", "y", "z"], count),
            "cat_low": cat_a,
            "cat_high": rng.choice([f"h{index}" for index in range(20)], count),
            "lat1": rng.uniform(40, 41, count),
            "lon1": rng.uniform(-74, -73, count),
            "lat2": rng.uniform(34, 35, count),
            "lon2": rng.uniform(-118, -117, count),
            "date_col": pd.date_range("2026-01-01", periods=count, freq="D", tz="UTC"),
            "target": ((a + b) > 5).astype("int64"),
        }
    )


def _raw_columns(node: str, params: dict[str, Any]) -> list[str]:
    """Include the candidate's real input columns and stable numeric controls."""
    needed = {"num_a", "num_b", "num_c"}
    needed.update(params.get("columns", []))
    for key in ("lat1_col", "lon1_col", "lat2_col", "lon2_col"):
        if params.get(key):
            needed.add(params[key])
    for operation in params.get("operations", []):
        needed.update(operation.get("input_columns", []))
    if node == "DateFeatures":
        needed.add("date_col")
    return sorted(needed)


def audit(configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fit, save, reload and predict each bounded candidate on both engines."""
    source = _fixture()
    report: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="skyulf-sm24a-nodes-") as directory:
        for node, params in configs.items():
            report[node] = {}
            columns = _raw_columns(node, params)
            for engine in ("pandas", "polars"):
                try:
                    frame = source.loc[:, [*columns, "target"]]
                    native = pl.from_pandas(frame) if engine == "polars" else frame
                    steps = [
                        {"name": "candidate", "transformer": node, "params": params},
                        {
                            "name": "remaining_categories",
                            "transformer": "OneHotEncoder",
                            "params": {"handle_unknown": "ignore"},
                        },
                    ]
                    artifact = fit_local_workflow(
                        {
                            "preprocessing": steps,
                            "modeling": {
                                "type": "random_forest_classifier",
                                "params": {"n_estimators": 8, "max_depth": 4, "n_jobs": 1},
                            },
                        },
                        SplitDataset(train=native[:64], test=native[64:]),
                        target_column="target",
                        artifact_path=Path(directory) / f"{node}-{engine}",
                        max_rows=80,
                        max_bytes=2_000_000,
                    )
                    fitted = artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]
                    if fitted is None or (isinstance(fitted, dict) and not fitted):
                        raise ValueError("Candidate produced an empty fitted artifact.")
                    query = source.loc[72:79, columns]
                    result = predict_local_pipeline(query, artifact)
                    if len(result) != len(query) or result["prediction"].isna().any():
                        raise ValueError("Prediction did not preserve all requested rows.")
                    report[node][engine] = {
                        "status": "pass",
                        "rows": len(result),
                        "raw_columns": columns,
                        "feature_columns": artifact.manifest.feature_columns,
                    }
                except Exception as exc:  # noqa: BLE001 - the audit records each candidate failure
                    report[node][engine] = {
                        "status": "fail",
                        "error": f"{type(exc).__name__}: {exc}"[:400],
                        "raw_columns": columns,
                    }
    return report


runtime_dbutils: Any = globals().get("dbutils")
if runtime_dbutils is not None:
    runtime_dbutils.widgets.text("config_path", "")
    source_path = Path(runtime_dbutils.widgets.get("config_path"))
    runtime_dbutils.notebook.exit(
        json.dumps(audit(json.loads(source_path.read_text(encoding="utf-8"))))
    )
