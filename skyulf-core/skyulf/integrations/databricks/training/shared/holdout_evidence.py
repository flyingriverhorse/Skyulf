"""Reuse bounded lifecycle Parquet as comparison-bound offline approval evidence."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ....mlflow.lifecycle.validation import _validate_holdout_budget
from ...data.delta_io.delta import table_identity
from ...feature_store.lifecycle_config import parse_feature_binding
from ...feature_store.training import validate_training_frame

_ARTIFACT = "lifecycle/data/holdout.parquet"


def frame_schema(frame: pd.DataFrame) -> list[list[str]]:
    """Preserve column order and pandas dtypes including timezone and nullable types."""
    return [[str(name), str(dtype)] for name, dtype in frame.dtypes.items()]


def save_frame_artifact(client: Any, run_id: str, name: str, frame: pd.DataFrame) -> dict:
    """Write the existing lifecycle format and reject unserializable context explicitly."""
    try:
        json.dumps(frame.attrs, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Prepared dataset context must be finite JSON; history cannot be omitted."
        ) from exc
    with TemporaryDirectory(prefix="skyulf-stage-data-") as directory:
        path = Path(directory) / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        receipt = _frame_receipt(path, frame)
        client.log_artifact(run_id, str(path), "lifecycle/data")
    return receipt


def _frame_receipt(path: Path, frame: pd.DataFrame) -> dict:
    """Bind exact bytes and typed rows to detached split and feature metadata."""
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "attrs": deepcopy(frame.attrs),
        "bytes": path.stat().st_size,
        "memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
        "schema": frame_schema(frame),
        "arrow_schema_sha256": hashlib.sha256(pq.read_schema(path).serialize()).hexdigest(),
    }


def _artifact_size(client: Any, run_id: str) -> int | None:
    """Require one concrete regular file with trustworthy pre-download size metadata."""
    matches = [
        item for item in client.list_artifacts(run_id, "lifecycle/data") if item.path == _ARTIFACT
    ]
    if not matches:
        return None
    if len(matches) != 1 or matches[0].is_dir or type(matches[0].file_size) is not int:
        raise ValueError("Saved holdout artifact has invalid file metadata.")
    return matches[0].file_size


def _read_parquet(path: Path, evidence: dict, spec: Any) -> pd.DataFrame:
    """Verify bytes, footer rows and schema before decoding, then bound decoded memory."""
    _validate_memory_limit(evidence, spec.max_bytes)
    size = path.stat().st_size
    if size != evidence["bytes"] or size > spec.max_bytes:
        raise ValueError("Saved holdout exceeds byte budget or differs from its receipt.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]:
        raise ValueError("Saved holdout digest differs from its comparison.")
    _validate_parquet_footer(path, evidence, spec)
    frame = pd.read_parquet(path)
    _validate_holdout_budget(frame, spec.max_rows, spec.max_bytes)
    if frame_schema(frame) != evidence["schema"] or frame.attrs != evidence["attrs"]:
        raise ValueError("Saved holdout schema or split context differs from its comparison.")
    return frame


def _validate_parquet_footer(path: Path, evidence: dict, spec: Any) -> None:
    """Bound typed row metadata and uncompressed allocations before decoding columns."""
    metadata = pq.read_metadata(path)
    if metadata.num_rows != evidence["rows"] or metadata.num_rows > spec.max_rows:
        raise ValueError("Saved holdout row count differs from its comparison or budget.")
    schema_digest = hashlib.sha256(pq.read_schema(path).serialize()).hexdigest()
    if schema_digest != evidence["arrow_schema_sha256"]:
        raise ValueError("Saved holdout Parquet schema differs from its comparison.")
    # Footer uncompressed sizes also bound highly compressed allocations before decoding.
    uncompressed = sum(
        metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)
    )
    if uncompressed > spec.max_bytes:
        raise ValueError("Saved holdout uncompressed bytes exceed budget.")


def _saved_or_new_frame(client: Any, run_id: str, frame: pd.DataFrame, spec: Any) -> dict:
    """Reuse prepared lifecycle bytes only when they equal the actual comparison rows."""
    _validate_holdout_budget(frame, spec.max_rows, spec.max_bytes)
    size = _artifact_size(client, run_id)
    if size is None:
        receipt = save_frame_artifact(client, run_id, "holdout", frame)
        if receipt["bytes"] > spec.max_bytes:
            raise ValueError("Saved holdout exceeds byte budget.")
        return receipt
    if not 0 < size <= spec.max_bytes:
        raise ValueError("Saved holdout exceeds byte budget.")
    with TemporaryDirectory(prefix="skyulf-holdout-") as directory:
        path = Path(client.download_artifacts(run_id, _ARTIFACT, directory))
        if path.stat().st_size != size:
            raise ValueError("Saved holdout artifact size changed during download.")
        receipt = _frame_receipt(path, frame)
        existing = _read_parquet(path, receipt, spec)
    if not existing.equals(frame):
        raise ValueError("Prepared holdout differs from the actual comparison rows.")
    return receipt


def save_holdout_reference(run: Any, frame: pd.DataFrame, spec: Any, report: Any) -> dict:
    """Bind one trusted candidate run's original enriched rows into its comparison digest."""
    validate_training_frame(frame, spec)
    source_id = frame.attrs.get("source_table_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("Saved holdout requires the original source table identity.")
    receipt = _saved_or_new_frame(run.client, run.run_id, frame, spec)
    return {
        "version": 1,
        "run_id": run.run_id,
        "artifact_path": _ARTIFACT,
        "model_name": report.model_name,
        "model_version": report.candidate_version,
        "model_digest": report.candidate_digest,
        "dataset_id": spec.dataset_id,
        "holdout_key_sha256": spec.holdout_key_sha256,
        "feature_binding_json": spec.feature_binding_json,
        "source_table": spec.table,
        "source_table_id": source_id,
        "frame": receipt,
    }


def _validate_reference(client: Any, report: Any, spec: Any) -> dict:
    """Reject wrong candidate owners, paths and bindings before any artifact download."""
    ref = report.holdout_reference
    if not isinstance(ref, dict) or type(ref.get("version")) is not int or ref["version"] != 1:
        raise ValueError("Saved holdout reference has an unsupported version.")
    model = client.get_model_version(report.model_name, report.candidate_version)
    expected = {
        "run_id": model.run_id,
        "artifact_path": _ARTIFACT,
        "model_name": report.model_name,
        "model_version": report.candidate_version,
        "model_digest": report.candidate_digest,
        "dataset_id": spec.dataset_id,
        "holdout_key_sha256": spec.holdout_key_sha256,
        "feature_binding_json": spec.feature_binding_json,
        "source_table": spec.table,
    }
    if not model.run_id or any(ref.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Saved holdout reference differs from candidate ownership or split binding."
        )
    return ref


def _validate_identities(spark: Any, ref: dict, spec: Any) -> None:
    """Allow version advances, but never silently accept replaced base or feature tables."""
    if table_identity(spark, spec.table) != ref.get("source_table_id"):
        raise ValueError("Saved holdout source table identity changed since training.")
    binding = parse_feature_binding(spec.feature_binding_json)
    for record in binding["lookup_evidence"]["feature_tables"]:
        if table_identity(spark, record["table_name"]) != record["table_id"]:
            raise ValueError("Saved holdout feature table identity changed since training.")


def load_saved_holdout(client: Any, spark: Any, report: Any, spec: Any) -> pd.DataFrame:
    """Load verified historical inputs without looking up current values or online freshness."""
    ref = _validate_reference(client, report, spec)
    evidence = ref.get("frame")
    if not isinstance(evidence, dict):
        raise ValueError("Saved holdout frame receipt is missing.")
    try:
        _validate_frame_limits(evidence, report, spec)
        _validate_identities(spark, ref, spec)
        size = _artifact_size(client, ref["run_id"])
        if size != evidence["bytes"] or size is None:
            raise ValueError("Saved holdout is missing or its byte size changed.")
        with TemporaryDirectory(prefix="skyulf-approval-holdout-") as directory:
            path = Path(client.download_artifacts(ref["run_id"], _ARTIFACT, directory))
            frame = _read_parquet(path, evidence, spec)
        validate_training_frame(frame, spec)
        _validate_identities(spark, ref, spec)
        return frame
    except (KeyError, TypeError) as exc:
        raise ValueError("Saved holdout reference is malformed.") from exc


def _validate_frame_limits(evidence: dict, report: Any, spec: Any) -> None:
    """Enforce pinned row, byte and membership limits before remote reads or allocation."""
    if type(evidence.get("rows")) is not int or not 2 <= evidence["rows"] <= spec.max_rows:
        raise ValueError("Saved holdout exceeds row budget.")
    if type(evidence.get("bytes")) is not int or not 0 < evidence["bytes"] <= spec.max_bytes:
        raise ValueError("Saved holdout exceeds byte budget.")
    if evidence["rows"] != report.row_count:
        raise ValueError("Saved holdout row count differs from comparison.")
    if evidence["attrs"]["holdout_key_sha256"] != spec.holdout_key_sha256:
        raise ValueError("Saved holdout membership differs from training split.")
    _validate_memory_limit(evidence, spec.max_bytes)


def _validate_memory_limit(evidence: dict, max_bytes: int) -> None:
    """Bound producer-measured decoded memory before Parquet dictionary expansion."""
    size = evidence.get("memory_bytes")
    if type(size) is not int or not 0 < size <= max_bytes:
        raise ValueError("Saved holdout decoded memory exceeds byte budget.")
