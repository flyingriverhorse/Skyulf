"""Reconstruct a verified model-specific training reference without fitting."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ....mlflow.registration.registry import (
    load_registered_pipeline,
    resolve_model,
)
from ....mlflow.shared._client import make_registry_client, require_mlflow
from ...jobs.lifecycle.lifecycle_tasks import phase_training_spec
from ...training.fitting.candidate import read_training_snapshot, split_labeled_snapshot
from ...training.shared.training_evidence import validate_training_evidence
from .monitoring_config import MonitorConfig
from .monitoring_training import capture_training_summary


def reference_document(client: Any, run_id: str, path: str) -> dict:
    """Read exact saved JSON artifacts through a disposable local directory."""
    with TemporaryDirectory(prefix="skyulf-monitor-reference-") as directory:
        downloaded = client.download_artifacts(run_id, path, directory)
        return json.loads(Path(downloaded).read_text(encoding="utf-8"))


def load_monitoring_reference(
    spark: Any,
    config: MonitorConfig,
    *,
    tracking_uri: str | None,
    registry_uri: str | None,
) -> tuple[Any, Any, Any, dict]:
    """Resolve once, replay the saved split and verify exact training/holdout membership."""
    artifact, spec, filters, evidence = load_monitoring_artifact(
        config, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    bounded = replace(
        spec,
        max_rows=min(spec.max_rows, config.max_rows),
        max_bytes=min(spec.max_bytes, config.max_bytes),
    )
    frame = read_training_snapshot(spark, bounded)
    train, holdout, _ = split_labeled_snapshot(frame, spec, engine=artifact.manifest.fitted_engine)
    validate_training_evidence(
        filters,
        spec,
        project_source_sha256=artifact.manifest.project_source_sha256,
        heldout=holdout,
    )
    if spec.weight_column is not None:
        train = train.drop(columns=[spec.weight_column])
    policy = config.performance_policy
    if policy and policy.get("mode") != "off" and policy["baseline"]["kind"] == "training_holdout":
        from .monitoring_performance import measure_holdout_baseline  # noqa: PLC0415

        evidence["performance_baseline"] = measure_holdout_baseline(
            artifact, spec, holdout, config, evidence["model_version"], evidence["training_run_id"]
        )
    return artifact, spec, train, evidence


def load_monitoring_artifact(
    config: MonitorConfig, *, tracking_uri: str | None, registry_uri: str | None
) -> tuple[Any, Any, dict, dict]:
    """Load verified model metadata without reading or materializing training populations."""
    resolved = resolve_model(
        config.model_name,
        version=config.model_version,
        alias=config.model_alias,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    artifact = load_registered_pipeline(
        resolved,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    parent = _verify_model_set(config, resolved, tracking_uri, registry_uri)
    client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
    model = client.get_model_version(resolved.name, resolved.version)
    if not model.run_id:
        raise ValueError("Monitoring requires a saved training run.")
    saved = reference_document(client, model.run_id, "candidate_training_spec.json")
    filters = reference_document(client, model.run_id, "training_filter_evidence.json")
    if saved.get("engine") != artifact.manifest.fitted_engine:
        raise ValueError("Monitoring reference engine differs from the saved artifact.")
    spec = phase_training_spec(saved, artifact.pipeline.config.get("project_python_source"))
    evidence = {
        "model_name": resolved.name,
        "model_version": resolved.version,
        "model_digest": artifact.manifest.pipeline_sha256,
        "training_run_id": model.run_id,
        "reference_table": spec.table,
        "reference_version": spec.version,
        "dataset_id": spec.dataset_id,
        "training_evidence_sha256": spec.training_evidence_sha256,
        "reference_population": "training_partition_excluding_holdout",
        "model_metadata": _model_metadata(client, model, artifact, saved, filters),
    }
    return artifact, spec, filters, evidence | parent


def _model_metadata(
    client: Any, model: Any, artifact: Any, saved_spec: dict, filters: dict
) -> dict:
    """Capture only operator lineage fields, preserving optional metadata failures."""
    created = getattr(model, "creation_timestamp", None)
    aliases = getattr(model, "aliases", None)
    metadata = {
        "status": "recorded",
        "reason": None,
        "captured_at": datetime.now(UTC).isoformat(),
        "created_at": datetime.fromtimestamp(created / 1000, UTC).isoformat() if created else None,
        "aliases": sorted(aliases) if aliases is not None else None,
        "algorithm": artifact.pipeline.config.get("modeling", {}).get("type"),
        "project_source_sha256": artifact.manifest.project_source_sha256,
        "training_experiment_id": None,
        "training_source_commit": None,
    }
    run = None
    try:
        run = client.get_run(model.run_id)
        metadata["training_experiment_id"] = str(run.info.experiment_id)
        metadata["training_source_commit"] = run.data.tags.get("mlflow.source.git.commit")
    except Exception as exc:  # noqa: BLE001 - optional link metadata cannot invalidate metrics
        metadata["status"] = "unavailable"
        metadata["reason"] = f"training_run_metadata:{type(exc).__name__}"
    metadata["training"] = capture_training_summary(
        client, model, artifact, run, saved_spec, filters
    )
    return metadata


def _verify_model_set(
    config: MonitorConfig, resolved: Any, tracking_uri: str | None, registry_uri: str | None
) -> dict:
    """Require the observed component version and digest to belong to the pinned model set."""
    if config.model_set_name is None:
        return {}
    from ....mlflow.models.model_set import load_registered_model_set  # noqa: PLC0415

    parent = resolve_model(
        config.model_set_name,
        version=config.model_set_version,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    artifact = load_registered_model_set(
        parent, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    components = {
        component.branch: component.reference for component in artifact.manifest.components
    }
    component = components.get(config.model_set_branch)
    if component is None or (component.name, component.version, component.digest) != (
        resolved.name,
        resolved.version,
        resolved.digest,
    ):
        raise ValueError("Monitored component does not belong to the pinned model-set release.")
    return {
        "model_set_name": parent.name,
        "model_set_version": parent.version,
        "model_set_branch": config.model_set_branch,
        "model_set_digest": parent.digest,
    }
