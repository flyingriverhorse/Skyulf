"""Bind prepared monitoring references to the values actually read during training."""

from typing import Any

from .monitoring_config import json_digest


def source_evidence(frame: Any, columns: tuple[str, ...], dataset_id: str) -> dict:
    """Fingerprint normalized original source values, including keys and weight metadata."""
    from ...data.training.retraining_data import (  # noqa: PLC0415 - avoid training import cycles
        _row_counts,
    )

    counts = _row_counts(frame, list(columns))
    return {
        "format": 1,
        "dataset_id": dataset_id,
        "columns": list(columns),
        "rows": len(frame),
        "content_sha256": json_digest(sorted(counts.items())),
    }


def validate_source_evidence(
    receipt: dict, frame: Any, columns: tuple[str, ...], dataset_id: str
) -> None:
    """Reject a same-key replacement source whose original training values cannot be proven."""
    if receipt != source_evidence(frame, columns, dataset_id):
        raise ValueError("Monitoring reference differs from original training source content.")
