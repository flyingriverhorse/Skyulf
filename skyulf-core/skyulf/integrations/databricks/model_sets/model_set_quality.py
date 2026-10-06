"""Re-evaluate every set component on its saved holdout before coherent activation."""

from dataclasses import replace
from math import isfinite
from typing import Any

import polars as pl

from skyulf.integrations.mlflow.shared._client import make_registry_client, require_mlflow

from ....inference.model_set import ModelSetArtifact
from ...mlflow.lifecycle.validation import (
    ModelComparisonReport,
    compare_registered_local_models,
    comparison_payload,
    quality_gate_results,
    quality_gates_pass,
)
from ...mlflow.registration.registry import resolve_model
from ..jobs.shared.notebook_diagnostics import notebook_task
from ..training.fitting import local_retraining
from ..training.shared.local_training_evidence import (
    load_candidate_evidence,
    validate_training_evidence,
)


def _non_regressing(report: ModelComparisonReport) -> bool:
    """Reject unavailable comparisons and any loss on the direction-aware metric."""
    gain = report.improvement
    return gain is not None and isfinite(gain) and gain >= 0


def component_quality(report: ModelComparisonReport) -> dict[str, Any]:
    """Require absolute gates and no regression; track meaningful gains separately."""
    absolute = report.quality_threshold is not None and quality_gates_pass(report)
    non_regressing = _non_regressing(report)
    passed = absolute and (report.champion_version is None or non_regressing)
    reason = report.reason
    if report.quality_threshold is None:
        reason = "missing_quality_threshold"
    elif not absolute:
        reason = "quality_gate_failed"
    elif report.champion_version is None:
        reason = "initial_quality_passed"
    elif non_regressing and not report.eligible:
        reason = "non_regressing"
    return {
        "passed": passed,
        "improved": passed and report.eligible,
        "reason": reason,
        "gates": quality_gate_results(report),
        "comparison": comparison_payload(report),
    }


def _heldout(spark: Any, spec: Any, engine: str, evidence: Any, limits: dict) -> Any:
    """Replay frozen source membership and preprocessing without fitting any model."""
    bounded = replace(
        spec,
        max_rows=min(spec.max_rows, limits["max_rows"]),
        max_bytes=min(spec.max_bytes, limits["max_bytes"]),
    )
    frame = local_retraining.read_training_snapshot(spark, bounded)
    _, heldout, _ = local_retraining.split_labeled_snapshot(frame, bounded, engine=engine)
    if evidence is not None:
        validate_training_evidence(
            evidence, spec, project_source_sha256=evidence["project_source_sha256"], heldout=heldout
        )
    return pl.from_pandas(heldout) if engine == "polars" else heldout


def _reference(component: Any, endpoints: dict) -> Any:
    """Resolve a saved concrete version and reject replacement of its fitted bytes."""
    if component is None:
        return None
    ref = component.reference
    resolved = resolve_model(ref.name, version=ref.version, **endpoints)
    if resolved.digest != ref.digest:
        raise ValueError("Model set quality component digest differs from saved reference.")
    return resolved


def _evaluate_component(
    spark: Any,
    component: Any,
    digest: str,
    champion: Any,
    *,
    max_rows: int,
    max_bytes: int,
    **endpoints: Any,
) -> dict:
    """Compare one candidate against its counterpart from the pinned champion set."""
    client = make_registry_client(
        require_mlflow(), endpoints["tracking_uri"], endpoints["registry_uri"]
    )
    ref = component.reference
    report, spec, engine, evidence = load_candidate_evidence(
        client, ref.name, ref.version, digest, registry_uri=endpoints["registry_uri"]
    )
    if report.candidate_digest != ref.digest:
        raise ValueError("Model set quality comparison identifies different model bytes.")
    candidate = _reference(component, endpoints)
    previous = _reference(champion, endpoints)
    if report.champion_version != (previous.version if previous else None):
        raise ValueError("Saved comparison differs from the model-set baseline component.")
    if report.champion_digest != (previous.digest if previous else None):
        raise ValueError("Saved comparison differs from the model-set baseline digest.")
    heldout = _heldout(
        spark, spec, engine, evidence, {"max_rows": max_rows, "max_bytes": max_bytes}
    )
    fresh = compare_registered_local_models(
        candidate,
        previous,
        heldout,
        target_column=spec.target_column,
        dataset_id=spec.dataset_id,
        metric=report.metric,
        min_improvement=report.min_improvement,
        quality_threshold=report.quality_threshold,
        quality_gates=report.quality_gates,
        max_rows=min(max_rows, spec.max_rows),
        max_bytes=min(max_bytes, spec.max_bytes),
        **endpoints,
    )
    return component_quality(fresh)


def evaluate_model_set_quality(
    spark: Any,
    artifact: ModelSetArtifact,
    champion: ModelSetArtifact | None,
    *,
    expected_champion_version: str | None,
    max_rows: int,
    max_bytes: int,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> dict:
    """Require complete immutable evidence and collect every branch's quality decision."""
    saved = artifact.manifest.quality_evidence
    if saved is None:
        raise ValueError("Model set has no saved quality evidence; train and package a new set.")
    if saved["expected_champion_version"] != expected_champion_version:
        raise ValueError("Model set quality baseline differs from expected champion.")
    components = artifact.manifest.components
    previous = _counterparts(components, champion)
    results = {}
    for component in components:
        with notebook_task(f"model_set_quality {component.branch}", None):
            results[component.branch] = _evaluate_component(
                spark,
                component,
                saved["comparisons"][component.branch],
                previous.get(component.branch),
                max_rows=max_rows,
                max_bytes=max_bytes,
                tracking_uri=tracking_uri,
                registry_uri=registry_uri,
            )
    return _set_decision(results, expected_champion_version)


def _set_decision(results: dict, expected_champion_version: str | None) -> dict:
    """Allow tied peers only when another branch earns replacement of the whole set."""
    initial = expected_champion_version is None
    failed = [name for name, result in results.items() if not result["passed"]]
    improved = [name for name, result in results.items() if result["improved"]]
    reason = "initial_quality_passed" if initial else "set_improved"
    if failed:
        reason = "component_quality_failed"
    elif not initial and not improved:
        reason = "no_component_improved"
    return {
        "passed": reason in {"initial_quality_passed", "set_improved"},
        "reason": reason,
        "failed_components": failed,
        "improved_components": improved,
        "components": results,
        "expected_champion_version": expected_champion_version,
    }


def _counterparts(components: Any, champion: ModelSetArtifact | None) -> dict:
    """Reject missing or added branches before comparing any model."""
    previous = {c.branch: c for c in champion.manifest.components} if champion else {}
    if champion and set(previous) != {c.branch for c in components}:
        raise ValueError("Model set quality requires the same champion branch names.")
    return previous
