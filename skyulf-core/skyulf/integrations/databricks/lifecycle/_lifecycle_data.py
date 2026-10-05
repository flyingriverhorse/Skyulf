"""Checked, bounded training datasets shared by separate lifecycle tasks."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from ..training.fitting import local_retraining as training
from ..training.thresholds.decision_thresholds import needs_threshold_time
from ..training.tuning.local_cv import LocalCVSpec
from ._lifecycle_state import PhaseStore


def save_frame(store: PhaseStore, name: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Log Parquet and its digest; keep pandas row-membership attributes in the receipt."""
    with TemporaryDirectory(prefix="skyulf-stage-data-") as directory:
        path = Path(directory) / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        store.client.log_artifact(store.run_id, str(path), "lifecycle/data")
    return {"sha256": digest, "rows": len(frame), "attrs": frame.attrs}


def load_frame(store: PhaseStore, name: str, evidence: dict[str, Any]) -> pd.DataFrame:
    """Check saved bytes before reading a partition, then restore verified metadata."""
    with TemporaryDirectory(prefix="skyulf-stage-read-") as directory:
        path = Path(
            store.client.download_artifacts(
                store.run_id, f"lifecycle/data/{name}.parquet", directory
            )
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]:
            raise ValueError("Prepared dataset digest differs from its saved receipt.")
        frame = pd.read_parquet(path)
    if len(frame) != evidence["rows"]:
        raise ValueError("Prepared dataset row count differs from its saved receipt.")
    frame.attrs = evidence["attrs"]
    return frame


def load_source(spark: Any, store: PhaseStore, spec: training.LocalTrainingSpec) -> dict[str, Any]:
    """Materialize the bounded pinned source before any splitting or fitting."""
    frame = training.read_training_snapshot(spark, spec)
    return {
        "source_table": spec.table,
        "source_version": spec.version,
        "source_rows": len(frame),
        "source": save_frame(store, "source", frame),
    }


def prepare_dataset(store: PhaseStore, spec: training.LocalTrainingSpec) -> dict[str, Any]:
    """Apply fixed eligibility/cleanup and persist raw training/holdout partitions."""
    config = store.request["config"]
    cv = LocalCVSpec.from_workflow(config)
    frame = load_frame(store, "source", store.receipt("load_data")["output"]["source"])
    train, holdout, unavailable = training.split_labeled_snapshot(
        frame,
        spec,
        engine=config["engine"],
        keep_training_event=(cv.enabled and cv.temporal) or needs_threshold_time(config),
    )
    return {
        "split_strategy": spec.split_strategy,
        "training_rows": len(train),
        "holdout_rows": len(holdout),
        "unavailable_labels": unavailable,
        "fixed_filters": holdout.attrs["pre_split_filter_counts"],
        "train": save_frame(store, "train", train),
        "holdout": save_frame(store, "holdout", holdout),
    }


def training_partitions(
    store: PhaseStore,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Load all verified partitions without repeating source reads or split work."""
    prepared = store.receipt("prepare_dataset")["output"]
    source = store.receipt("load_data")["output"]
    return (
        load_frame(store, "source", source["source"]),
        load_frame(store, "train", prepared["train"]),
        load_frame(store, "holdout", prepared["holdout"]),
        prepared["unavailable_labels"],
    )
