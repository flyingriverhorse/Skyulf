"""Build and check bounded, versioned local training filter receipts."""

import hashlib
import json
import re
from dataclasses import replace
from typing import Any

import pandas as pd


def evidence_digest(evidence: dict[str, Any]) -> str:
    """Hash the complete JSON receipt with one stable encoding."""
    return hashlib.sha256(
        json.dumps(evidence, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def build_training_evidence(
    spec: Any, heldout: pd.DataFrame, *, project_source_sha256: str | None
) -> dict[str, Any]:
    """Record all ordered populations before model-only columns discard row keys."""
    attrs = heldout.attrs
    return {
        "version": 1,
        "dataset_id": replace(spec, training_evidence_sha256=None).dataset_id,
        "project_source_sha256": project_source_sha256,
        "pre_split_steps": list(spec.pre_split_steps),
        "recipe_sha256": evidence_digest({"steps": list(spec.pre_split_steps)}),
        "record_key_columns": list(spec.record_key_columns),
        "pre_filter_key_sha256": attrs["pre_filter_key_sha256"],
        "survivor_key_sha256": attrs["survivor_key_sha256"],
        "train_key_sha256": attrs["train_key_sha256"],
        "holdout_key_sha256": attrs["holdout_key_sha256"],
        "sample_key_sha256": attrs["sample_key_sha256"],
        "source_rows": attrs["source_rows"],
        "pre_filter_rows": attrs["pre_filter_rows"],
        "survivor_rows": attrs["survivor_rows"],
        "training_rows": attrs["training_rows"],
        "holdout_rows": len(heldout),
        "filter_counts": attrs["pre_split_filter_counts"],
    }


def validate_training_evidence(
    evidence: dict[str, Any],
    spec: Any,
    *,
    project_source_sha256: str | None,
    heldout: pd.DataFrame | None = None,
) -> None:
    """Reject malformed saved receipts and changed source, recipe or membership."""
    if not isinstance(evidence, dict) or evidence.get("version") != 1:
        raise ValueError("Saved training filter evidence has an unsupported version.")
    if not isinstance(spec.training_evidence_sha256, str) or not re.fullmatch(
        r"[a-f0-9]{64}", spec.training_evidence_sha256
    ):
        raise ValueError("Saved training filter evidence has no valid digest.")
    try:
        actual = evidence_digest(evidence)
        if actual != spec.training_evidence_sha256:
            raise ValueError("Saved training filter evidence digest differs from training spec.")
        if evidence["dataset_id"] != replace(spec, training_evidence_sha256=None).dataset_id:
            raise ValueError("Saved training filter evidence has a different dataset identity.")
        if evidence["project_source_sha256"] != project_source_sha256:
            raise ValueError("Saved training filter evidence has a different project source.")
        if evidence["pre_split_steps"] != list(spec.pre_split_steps) or evidence[
            "recipe_sha256"
        ] != evidence_digest({"steps": list(spec.pre_split_steps)}):
            raise ValueError("Saved training filter evidence has a different recipe.")
        if evidence["record_key_columns"] != list(spec.record_key_columns):
            raise ValueError("Saved training filter evidence has different record keys.")
        if (
            evidence["holdout_key_sha256"] != spec.holdout_key_sha256
            or evidence["survivor_key_sha256"] != spec.survivor_key_sha256
        ):
            raise ValueError("Saved training filter evidence has different membership.")
        if evidence["sample_key_sha256"] != spec.sample_key_sha256:
            raise ValueError("Saved training filter evidence has different sample membership.")
        counts = evidence["filter_counts"]
        if not isinstance(counts, list) or len(counts) != len(spec.pre_split_steps):
            raise ValueError("Saved training filter evidence has invalid step counts.")
        remaining = evidence["pre_filter_rows"]
        for step, count in zip(spec.pre_split_steps, counts, strict=True):
            if (
                count["name"] != step["name"]
                or count["transformer"] != step["transformer"]
                or count["input_rows"] != remaining
                or count["excluded_rows"] < 0
                or count["output_rows"] != remaining - count["excluded_rows"]
            ):
                raise ValueError("Saved training filter evidence has inconsistent step counts.")
            remaining = count["output_rows"]
        if (
            remaining != evidence["survivor_rows"]
            or remaining != evidence["training_rows"] + evidence["holdout_rows"]
            or evidence["source_rows"] < evidence["pre_filter_rows"]
        ):
            raise ValueError("Saved training filter evidence has inconsistent population counts.")
        if heldout is not None and evidence != build_training_evidence(
            spec, heldout, project_source_sha256=project_source_sha256
        ):
            raise ValueError("Replayed training filter evidence differs from the saved population.")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Saved training filter evidence is malformed.") from exc
