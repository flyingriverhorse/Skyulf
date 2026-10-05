"""Bind bounded temporal continuation to committed Databricks prediction receipts."""

import hashlib
import json
from contextlib import nullcontext
from typing import Any

from .....inference.local_pipeline import LocalPipelineArtifact
from .....preprocessing.time_series.history import TemporalHistorySession


def prediction_history(
    prepared: Any,
    state: dict[str, Any] | None = None,
    *,
    bootstrap: bool = False,
) -> Any:
    """Use artifact seeds or rebuild context from a complete initial source snapshot."""
    artifact = prepared.artifact
    if not isinstance(artifact, LocalPipelineArtifact):
        return nullcontext(None)
    steps = artifact.pipeline.feature_engineer.fitted_steps
    identities = [
        step["artifact"]["history_id"]
        for step in steps
        if step["artifact"].get("history_mode") == "carry"
    ]
    if not identities:
        if state is not None:
            raise ValueError("Saved temporal history requires a model with carry steps.")
        return nullcontext(None)
    model_id = artifact.manifest.pipeline_sha256
    if bootstrap:
        state = {
            "version": 1,
            "model_id": model_id,
            "steps": {identity: [] for identity in identities},
        }
    return TemporalHistorySession(model_id, state)


def incremental_history(prepared: Any, previous: dict | None) -> Any:
    """Continue the last committed context; never silently reuse another model's tail."""
    state = previous.get("temporal_history") if previous else None
    session = prediction_history(prepared, state, bootstrap=previous is None)
    if isinstance(session, TemporalHistorySession) and previous is not None and state is None:
        raise ValueError(
            "Incremental receipt has no temporal history; use a fresh prediction target."
        )
    return session


def history_receipt(session: TemporalHistorySession | None) -> dict[str, Any]:
    """Cap receipt size before any Delta write and include only successful proposals."""
    if session is None:
        return {}
    if session.state is None:
        raise ValueError("Temporal prediction did not complete successfully.")
    if len(json.dumps(session.state, allow_nan=False).encode()) > 64 * 1024:
        raise ValueError("Temporal history exceeds the 64 KiB prediction receipt budget.")
    return {"temporal_history": session.state}


def bind_period_history(manifest: dict[str, Any], session: Any, previous: Any) -> dict[str, Any]:
    """Make period retries conflict when their supplied context or output tail differs."""
    fields = history_receipt(session)
    if not fields:
        return manifest
    fingerprint = json.dumps(
        {"request": manifest["request_digest"], "previous": previous, **fields},
        sort_keys=True,
        allow_nan=False,
    ).encode()
    return manifest | fields | {"request_digest": hashlib.sha256(fingerprint).hexdigest()}
