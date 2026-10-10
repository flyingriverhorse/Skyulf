"""Optional saved-model diagnostics on bounded held-out inputs."""

from typing import Any

import pandas as pd
import polars as pl

from .....inference.local_pipeline import LocalPipelineArtifact
from .....inference.preprocessing_probe import probe_fitted_preprocessing


def log_preprocessing_probe(
    run: Any,
    artifact: LocalPipelineArtifact,
    holdout: pd.DataFrame | pl.DataFrame,
    *,
    enabled: bool,
) -> None:
    """Log a redacted diagnostic without refitting or changing promotion policy.

    Inspect at most the first 256 holdout rows, with the existing probe's 8 MiB
    input/output limit. Callback failures are diagnostic results; MLflow write
    errors retain the surrounding training workflow's failure policy.
    """
    if not enabled:
        return
    report = _holdout_probe(artifact, holdout)
    run.client.log_dict(run.run_id, report, "preprocessing_probe.json")


def _holdout_probe(
    artifact: LocalPipelineArtifact, holdout: pd.DataFrame | pl.DataFrame
) -> dict[str, Any]:
    """Keep values and exception messages out of a failed or unavailable check."""
    report: dict[str, Any] = {
        "report_version": 1,
        "status": "not_run",
        "admission": "diagnostic_only",
        "sample_rows": min(len(holdout), 256),
        "fitted_engine": artifact.manifest.fitted_engine,
        "pipeline_sha256": artifact.manifest.pipeline_sha256,
        "project_source_sha256": artifact.manifest.project_source_sha256,
        "steps": [],
    }
    if not len(holdout):
        return {**report, "reason": "empty_holdout"}
    try:
        columns = list(artifact.manifest.input_columns)
        sample = holdout.head(256)
        sample = (
            sample.select(columns) if isinstance(sample, pl.DataFrame) else sample.loc[:, columns]
        )
        return probe_fitted_preprocessing(artifact, sample)
    except Exception as exc:  # noqa: BLE001 - optional diagnostic must redact callback failures
        return {**report, "status": "failed", "error_type": type(exc).__name__}
