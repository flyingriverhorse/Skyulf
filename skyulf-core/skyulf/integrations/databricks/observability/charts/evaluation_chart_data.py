"""Optional bounded holdout samples for an independent evaluation-chart task."""

import hashlib
import json
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

_DEFAULTS = {"max_rows": 2000, "max_features": 20, "max_classes": 20}
_LIMITS = {"max_rows": (2, 10000), "max_features": (1, 50), "max_classes": (2, 50)}
_MAX_BYTES = 16 * 1024 * 1024


def chart_settings(value: Any) -> dict[str, Any] | None:
    """Validate explicit opt-in and plotting budgets without importing visualization packages."""
    if value is None:
        return None
    if type(value) is not dict or set(value) - {"enabled", *_DEFAULTS}:
        raise ValueError("evaluation_charts must contain only supported settings.")
    if type(value.get("enabled")) is not bool:
        raise ValueError("evaluation_charts.enabled must be a boolean.")
    settings = {**_DEFAULTS, **value}
    for key, (lower, upper) in _LIMITS.items():
        if type(settings[key]) is not int or not lower <= settings[key] <= upper:
            raise ValueError(f"evaluation_charts.{key} must be an integer from {lower} to {upper}.")
    return settings if settings["enabled"] else None


def chart_recorder(run: Any, artifact: Any, spec: Any, settings: Any) -> Callable | None:
    """Capture exact full-holdout evaluation outputs only for explicitly enabled reporting."""
    limits = chart_settings(settings)
    if limits is None:
        return None
    return partial(save_chart_sample, run=run, artifact=artifact, spec=spec, settings=limits)


def save_chart_sample(
    actual: Any, predictions: pd.DataFrame, *, run: Any, artifact: Any, spec: Any, settings: dict
) -> None:
    """Subsample already evaluated pairs, isolating optional serialization and storage failures."""
    metadata = {
        "version": 2,
        "status": "unavailable",
        "settings": settings,
        "model_digest": artifact.manifest.pipeline_sha256,
        "dataset_id": spec.dataset_id,
        "holdout_key_sha256": spec.holdout_key_sha256,
        "target_column": spec.target_column,
        "holdout_rows": len(actual),
        "sample_rows": min(len(actual), settings["max_rows"]),
        "sampling": "uniform_without_replacement_seed_42",
    }
    try:
        positions = (
            pd.Series(range(len(actual)))
            .sample(n=metadata["sample_rows"], random_state=42)
            .to_numpy()
        )
        sample = predictions.iloc[positions].reset_index(drop=True).copy()
        sample.insert(0, "observed", actual[positions])
        sample.attrs = {}
        metadata.update(status="ready", columns=list(sample.columns))
        if int(sample.memory_usage(deep=True).sum()) > _MAX_BYTES:
            metadata.update(status="unavailable", reason="Chart sample exceeds the 16 MiB budget.")
        else:
            _save_sample_bytes(run, sample, metadata)
    except Exception as error:  # noqa: BLE001 - optional evidence must not fail training
        metadata.update(
            status="unavailable",
            reason=f"Chart sample preparation failed ({type(error).__name__}).",
        )
        logging.getLogger(__name__).warning(metadata["reason"])
    try:
        run.client.log_dict(run.run_id, metadata, "evaluation_charts/sample.json")
    except Exception:  # noqa: BLE001 - the reporting leaf will disclose missing evidence
        logging.getLogger(__name__).warning(
            "Chart sample metadata could not be saved; training continues."
        )


def _save_sample_bytes(run: Any, sample: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Bind the bounded Parquet artifact to a digest before publishing its metadata."""
    with TemporaryDirectory(prefix="skyulf-chart-sample-") as directory:
        path = Path(directory) / "holdout.parquet"
        sample.to_parquet(path, index=False)
        if path.stat().st_size > _MAX_BYTES:
            metadata.update(
                status="unavailable", reason="Chart sample exceeds the 16 MiB file budget."
            )
            return
        metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        run.client.log_artifact(run.run_id, str(path), "evaluation_charts")


def load_chart_sample(
    client: Any, run_id: str, model_digest: str
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Read the exact saved sample, rejecting changed model, bytes, row counts or columns."""
    with TemporaryDirectory(prefix="skyulf-chart-read-") as directory:
        metadata_path = Path(
            client.download_artifacts(run_id, "evaluation_charts/sample.json", directory)
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("version") != 2 or metadata.get("model_digest") != model_digest:
            raise ValueError("Chart sample model identity differs from its training receipt.")
        if chart_settings(metadata["settings"]) is None:
            raise ValueError("Chart sample has no enabled settings.")
        if metadata["status"] == "unavailable":
            return None, metadata
        path = Path(
            client.download_artifacts(run_id, "evaluation_charts/holdout.parquet", directory)
        )
        if path.stat().st_size > _MAX_BYTES:
            raise ValueError("Chart sample exceeds the 16 MiB file budget.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Chart sample digest differs from its saved metadata.")
        frame = pd.read_parquet(path)
    _validate_sample_frame(frame, metadata)
    return frame, metadata


def _validate_sample_frame(frame: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Reject malformed or expanded prediction samples before rendering diagnostics."""
    if (
        len(frame) != metadata["sample_rows"]
        or len(frame) > min(metadata["holdout_rows"], metadata["settings"]["max_rows"])
        or list(frame.columns) != metadata["columns"]
        or int(frame.memory_usage(deep=True).sum()) > _MAX_BYTES
    ):
        raise ValueError("Chart sample schema or size differs from its saved metadata.")
