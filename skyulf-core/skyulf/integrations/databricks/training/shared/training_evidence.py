"""Build training receipts and verify saved candidate evidence for every lifecycle caller."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pandas as pd

from .....inference.project_code import load_project_module, project_source_digest
from ....mlflow.lifecycle.validation import ModelComparisonReport, comparison_digest
from ....mlflow.registration.registry import (
    load_registered_pipeline,
    resolve_model,
)
from ...shared.json_contracts import finite_json_digest

if TYPE_CHECKING:
    from ..fitting.candidate import TrainingSpec


def evidence_digest(evidence: dict[str, Any]) -> str:
    """Hash the complete JSON receipt with one stable encoding."""
    return finite_json_digest(evidence)


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
        **({"group_split": attrs["group_split"]} if "group_split" in attrs else {}),
        **(
            {"training_weights": deepcopy(attrs["training_weights"])}
            if "training_weights" in attrs
            else {}
        ),
    }


def _validate_evidence_identity(
    evidence: dict[str, Any], spec: Any, source_sha: str | None
) -> None:
    """Check the receipt digest, source, recipe and membership before population counts."""
    actual = evidence_digest(evidence)
    if actual != spec.training_evidence_sha256:
        raise ValueError("Saved training filter evidence digest differs from training spec.")
    if evidence["dataset_id"] != replace(spec, training_evidence_sha256=None).dataset_id:
        raise ValueError("Saved training filter evidence has a different dataset identity.")
    if evidence["project_source_sha256"] != source_sha:
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


def _validate_population_counts(evidence: dict[str, Any], spec: Any) -> None:
    """Verify each filter transition and the resulting train/holdout population."""
    counts = evidence["filter_counts"]
    if not isinstance(counts, list) or len(counts) != len(spec.pre_split_steps):
        raise ValueError("Saved training filter evidence has invalid step counts.")
    remaining = _remaining_filter_population(evidence, spec, counts)
    if (
        remaining != evidence["survivor_rows"]
        or remaining != evidence["training_rows"] + evidence["holdout_rows"]
        or evidence["source_rows"] < evidence["pre_filter_rows"]
    ):
        raise ValueError("Saved training filter evidence has inconsistent population counts.")


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
        _validate_evidence_identity(evidence, spec, project_source_sha256)
        _validate_population_counts(evidence, spec)
        if heldout is not None and evidence != build_training_evidence(
            spec, heldout, project_source_sha256=project_source_sha256
        ):
            raise ValueError("Replayed training filter evidence differs from the saved population.")
    except (KeyError, TypeError) as exc:
        raise ValueError("Saved training filter evidence is malformed.") from exc


def _read_candidate_artifacts(
    client: Any, name: str, version: str
) -> tuple[ModelComparisonReport, dict[str, Any], dict[str, Any] | None]:
    """Read one named run and its optional filter receipt within temporary storage."""
    model = client.get_model_version(name, version)
    if not model.run_id:
        raise ValueError("Candidate has no training run with approval evidence.")
    with TemporaryDirectory(prefix="skyulf-approval-") as directory:
        report_path = client.download_artifacts(
            model.run_id, "candidate_comparison.json", directory
        )
        spec_path = client.download_artifacts(
            model.run_id, "candidate_training_spec.json", directory
        )
        report = ModelComparisonReport(**json.loads(Path(report_path).read_text(encoding="utf-8")))
        saved_spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        saved_filter_evidence = None
        if saved_spec.get("training_evidence_sha256") is not None:
            evidence_path = client.download_artifacts(
                model.run_id, "training_filter_evidence.json", directory
            )
            saved_filter_evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
            if not isinstance(saved_filter_evidence, dict):
                raise ValueError("Saved training filter evidence must be a JSON object.")
    return report, saved_spec, saved_filter_evidence


def _validate_comparison_pin(
    report: ModelComparisonReport, name: str, version: str, digest: str
) -> None:
    """Bind comparison contents to the exact requested digest and model version."""
    actual = comparison_digest(report)
    if actual != digest:
        raise ValueError("Saved comparison digest differs from requested approval evidence.")
    if report.model_name != name or report.candidate_version != version:
        raise ValueError("Saved comparison does not identify the requested candidate.")


def _load_candidate_recipe(
    client: Any, name: str, version: str, engine: str, registry_uri: str | None
) -> tuple[str | None, Any]:
    """Verify the fitted engine and register the saved custom recipe before spec validation."""
    tracking_uri = getattr(client, "tracking_uri", None)
    reference = resolve_model(
        name,
        version=version,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri or tracking_uri,
    )
    artifact = load_registered_pipeline(
        reference, tracking_uri=tracking_uri, registry_uri=registry_uri or tracking_uri
    )
    if engine != artifact.manifest.fitted_engine:
        raise ValueError("Saved approval engine differs from fitted model engine.")
    source_sha = artifact.manifest.project_source_sha256
    if source_sha is None:
        return None, None
    source = artifact.pipeline.config["project_python_source"]
    if project_source_digest(source) != source_sha:
        raise ValueError("Saved project source differs from model manifest.")
    factory = getattr(load_project_module(source), "build_pre_split_steps", None)
    recipe = factory() if factory is not None else []
    return source_sha, recipe


def load_candidate_evidence(
    client: Any, name: str, version: str, digest: str, *, registry_uri: str | None = None
) -> tuple[ModelComparisonReport, TrainingSpec, str, dict[str, Any] | None]:
    """Read only the named version's run artifacts and verify the operator's evidence pin."""
    # Retraining uses the filter evidence builders in this module.
    from ..fitting.candidate import TrainingSpec  # noqa: PLC0415 - avoid import cycle

    report, saved_spec, saved_filter_evidence = _read_candidate_artifacts(client, name, version)
    _validate_comparison_pin(report, name, version, digest)
    engine = saved_spec.pop("engine")
    if engine not in ("pandas", "polars"):
        raise ValueError("Saved approval engine must be pandas or polars.")
    source_sha = None
    recipe = None
    if saved_filter_evidence is not None:
        source_sha, recipe = _load_candidate_recipe(client, name, version, engine, registry_uri)
    spec = TrainingSpec.from_payload(saved_spec)
    if spec.holdout_key_sha256 is None:
        raise ValueError("Saved training evidence requires holdout membership proof.")
    if spec.dataset_id != report.dataset_id:
        raise ValueError("Saved training snapshot differs from comparison evidence.")
    if saved_filter_evidence is not None:
        validate_training_evidence(saved_filter_evidence, spec, project_source_sha256=source_sha)
        if source_sha is not None and recipe != list(spec.pre_split_steps):
            raise ValueError("Saved project source recipe differs from training evidence.")
    return report, spec, engine, saved_filter_evidence


def _remaining_filter_population(evidence: dict[str, Any], spec: Any, counts: list[Any]) -> int:
    """Verify ordered filter transitions and return their surviving population."""
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
    return remaining
