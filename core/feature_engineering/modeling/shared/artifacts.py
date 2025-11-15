"""Shared helpers for persisting modeling artifacts and metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from config import get_settings
from .common import CrossValidationConfig

_settings = get_settings()
logger = logging.getLogger(__name__)


def _persist_best_estimator(job, estimator) -> Optional[str]:
    if estimator is None:
        return None
    try:
        artifact_dir = Path(_settings.TRAINING_ARTIFACT_DIR) / job.pipeline_id / "tuning"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{job.id}_run{job.run_number}.joblib"
        joblib.dump(estimator, artifact_path, compress=("gzip", 3))
        return str(artifact_path)
    except Exception as exc:  # pragma: no cover - artifact persistence failure shouldn't abort job
        logger.warning("Failed to persist best estimator for tuning job %s: %s", job.id, exc)
        return None


def _persist_training_artifact(
    artifact_root: str,
    pipeline_id: str,
    job_id: str,
    version: int,
    artifact_data: Dict[str, Any],
) -> Path:
    artifact_dir = Path(artifact_root) / pipeline_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{job_id}_v{version}.joblib"
    joblib.dump(artifact_data, artifact_path, compress=("gzip", 3))
    return artifact_path


def _write_transformer_debug_snapshot(
    transformers: List[Dict[str, Any]],
    transformer_plan: List[Dict[str, Any]],
    debug_dir: Optional[Path] = None,
) -> None:
    directory = debug_dir if debug_dir is not None else Path("logs")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        debug_file = directory / "fitted_params_debug_latest.json"
        with debug_file.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "transformer_plan": transformer_plan,
                    "transformers_overview": [
                        {
                            "node_id": (t.get("node_id") if isinstance(t, dict) else None),
                            "transformer_name": (t.get("transformer_name") if isinstance(t, dict) else None),
                            "column_name": (t.get("column_name") if isinstance(t, dict) else None),
                            "transformer_type": (
                                t.get("transformer").__class__.__name__
                                if isinstance(t, dict) and t.get("transformer") is not None
                                else None
                            ),
                            "metadata": (t.get("metadata") if isinstance(t, dict) else {}),
                        }
                        for t in transformers
                    ],
                },
                fh,
                indent=2,
                default=str,
            )
        logger.info("Wrote fitted-params debug file: %s", debug_file)
    except Exception:  # pragma: no cover - defensive logging path
        logger.exception("Failed to write fitted-params debug file")


def _build_metadata_update(
    resolved_problem_type: str,
    target_column: str,
    feature_columns: List[str],
    cv_config: CrossValidationConfig,
    dataset_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metadata_update: Dict[str, Any] = {
        "resolved_problem_type": resolved_problem_type,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "cross_validation": {
            "enabled": cv_config.enabled,
            "strategy": cv_config.strategy,
            "folds": cv_config.folds,
            "shuffle": cv_config.shuffle,
            "random_state": cv_config.random_state,
            "refit_strategy": cv_config.refit_strategy,
        },
    }
    if dataset_meta:
        metadata_update["dataset"] = dataset_meta
    return metadata_update


__all__ = [
    "_build_metadata_update",
    "_persist_best_estimator",
    "_persist_training_artifact",
    "_write_transformer_debug_snapshot",
]
