"""Transport validated source weights without adding them to model features."""

import hashlib
import json
from typing import Any

import pandas as pd
import polars as pl

from ...modeling._sample_weights import validate_sample_weight
from .weight_config import validate_weight_roles


def validate_weight_snapshot(settings: dict[str, Any]) -> None:
    """Verify captured source bytes without executing the saved Python hook."""
    validate_weight_roles(settings)
    source = settings.get("weights_python_source")
    digest = settings.get("weights_python_sha256")
    if source is None and digest is None:
        return
    if not isinstance(source, str) or not isinstance(digest, str):
        raise ValueError("weights_python_source and weights_python_sha256 require a pair.")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != digest:
        raise ValueError("weights_python_sha256 does not match captured source.")


def extract_training_weights(frame: Any, column: str | None) -> tuple[Any, Any]:
    """Copy positional weights and remove the training-only column before Core fit."""
    if column is None:
        return frame, None
    weights = validate_sample_weight(frame[column].to_numpy(), len(frame))
    features = (
        frame.drop(column) if isinstance(frame, pl.DataFrame) else frame.drop(columns=[column])
    )
    return features, weights


def training_weight_evidence(spec: Any, keyed_train_frame: pd.DataFrame) -> dict[str, Any] | None:
    """Bind ordered typed source identities to weights using aggregate-only evidence."""
    if spec.weight_column is None:
        return None
    weights = validate_sample_weight(
        keyed_train_frame[spec.weight_column].to_numpy(), len(keyed_train_frame)
    )
    assert weights is not None
    keys = keyed_train_frame.loc[:, list(spec.record_key_columns)].itertuples(
        index=False, name=None
    )
    payload = {
        "columns": list(spec.record_key_columns),
        "rows": [
            {"keys": [(type(value).__name__, str(value)) for value in row], "weight": float(weight)}
            for row, weight in zip(keys, weights, strict=True)
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "weight_column": spec.weight_column,
        "weights_python_sha256": spec.weights_python_sha256,
        "table": spec.table,
        "version": spec.version,
        "count": len(weights),
        "sum": float(weights.sum()),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "zero_count": int((weights == 0).sum()),
        "ordered_record_weight_sha256": digest,
    }
