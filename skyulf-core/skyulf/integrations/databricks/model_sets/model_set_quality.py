"""Re-evaluate every set component on its saved holdout before coherent activation."""

from dataclasses import replace
from math import isfinite
from typing import Any

import polars as pl

from skyulf.integrations.mlflow.shared._client import make_registry_client, require_mlflow

from ....inference.model_set import ModelSetArtifact, ModelSetManifest
from ...mlflow.lifecycle.validation import (
    ModelComparisonReport,
    compare_registered_pipeline_models,
    comparison_payload,
    quality_gate_results,
    quality_gates_pass,
)
from ...mlflow.registration.registry import resolve_model
from ..jobs.shared.notebook_diagnostics import notebook_task
from ..training.fitting import candidate
from ..training.shared.training_evidence import (
    load_candidate_evidence,
    validate_training_evidence,
)


def _non_regressing(report: ModelComparisonReport) -> bool:
    """Reject unavailable comparisons and any loss on the direction-aware metric."""
    gain = report.improvement
    return gain is not None and isfinite(gain) and gain >= 0


def component_quality(
    report: ModelComparisonReport, *, allow_unloadable_champion: bool = False
) -> dict[str, Any]:
    """Require absolute gates and no regression; track meaningful gains separately."""
    absolute = report.quality_threshold is not None and quality_gates_pass(report)
    non_regressing = _non_regressing(report)
    unavailable = report.reason == "champion_runtime_unavailable" and bool(
        report.champion_unavailable
    )
    passed = absolute and (
        report.champion_version is None
        or non_regressing
        or (allow_unloadable_champion and unavailable)
    )
    result = {
        "passed": passed,
        "improved": passed and report.eligible,
        "reason": _component_reason(report, absolute, non_regressing),
        "gates": quality_gate_results(report),
        "comparison": comparison_payload(report),
    }
    if unavailable:
        result["unloadable_champion_approval_allowed"] = absolute
    return result


def _component_reason(report: ModelComparisonReport, absolute: bool, non_regressing: bool) -> str:
    """Retain existing per-component explanations for failed and comparable candidates."""
    reason = report.reason
    if report.quality_threshold is None:
        reason = "missing_quality_threshold"
    elif not absolute:
        reason = "quality_gate_failed"
    elif report.champion_version is None:
        reason = "initial_quality_passed"
    elif non_regressing and not report.eligible:
        reason = "non_regressing"
    return reason


def _heldout(spark: Any, spec: Any, engine: str, evidence: Any, limits: dict) -> Any:
    """Replay frozen source membership and preprocessing without fitting any model."""
    bounded = replace(
        spec,
        max_rows=min(spec.max_rows, limits["max_rows"]),
        max_bytes=min(spec.max_bytes, limits["max_bytes"]),
    )
    frame = candidate.read_training_snapshot(spark, bounded)
    _, heldout, _ = candidate.split_labeled_snapshot(frame, bounded, engine=engine)
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
    allow_unloadable_champion: bool = False,
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
    fresh = compare_registered_pipeline_models(
        candidate,
        previous,
        heldout,
        target_column=spec.target_column,
        dataset_id=spec.dataset_id,
        metric=report.metric,
        min_improvement=report.min_improvement,
        quality_threshold=report.quality_threshold,
        quality_gates=report.quality_gates,
        min_scored_fraction=report.min_scored_fraction,
        max_coverage_drop=report.max_coverage_drop,
        max_rows=min(max_rows, spec.max_rows),
        max_bytes=min(max_bytes, spec.max_bytes),
        **endpoints,
    )
    return component_quality(fresh, allow_unloadable_champion=allow_unloadable_champion)


def evaluate_model_set_quality(
    spark: Any,
    artifact: ModelSetArtifact,
    champion: ModelSetArtifact | ModelSetManifest | None,
    *,
    expected_champion_version: str | None,
    max_rows: int,
    max_bytes: int,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    allow_unloadable_champion: bool = False,
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
                allow_unloadable_champion=allow_unloadable_champion,
            )
    return _set_decision(results, expected_champion_version, allow_unloadable_champion)


def _set_decision(
    results: dict, expected_champion_version: str | None, allow_unloadable_champion: bool = False
) -> dict:
    """Allow tied peers only when another branch earns replacement of the whole set."""
    if any(result.get("comparison", {}).get("champion_unavailable") for result in results.values()):
        return _unloadable_set_decision(
            results, expected_champion_version, allow_unloadable_champion
        )
    if allow_unloadable_champion:
        raise ValueError(
            "allow_unloadable_champion applies only when the champion cannot be loaded."
        )
    return _comparable_set_decision(results, expected_champion_version)


def _comparable_set_decision(results: dict, expected_champion_version: str | None) -> dict:
    """Preserve normal set decisions and their serialized evidence without recovery fields."""
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


def _unloadable_set_decision(results: dict, expected: str | None, approved: bool) -> dict:
    """Expose explicit recovery only when every candidate passes its saved absolute gates."""
    allowed = all(
        result["passed"] or result.get("unloadable_champion_approval_allowed", False)
        for result in results.values()
    )
    reason = "unloadable_champion_approved" if approved else "champion_runtime_unavailable"
    return {
        "passed": approved and allowed,
        "reason": reason if allowed else "component_quality_failed",
        "failed_components": [name for name, result in results.items() if not result["passed"]],
        "improved_components": [name for name, result in results.items() if result["improved"]],
        "components": results,
        "expected_champion_version": expected,
        "unloadable_champion_approval_allowed": allowed,
        "allow_unloadable_champion": approved,
    }


def champion_components(champion: ModelSetArtifact | ModelSetManifest | None) -> tuple:
    """Read component pins from a loaded set or its independently verified manifest."""
    if champion is None:
        return ()
    manifest = champion if isinstance(champion, ModelSetManifest) else champion.manifest
    return manifest.components


def _counterparts(components: Any, champion: ModelSetArtifact | ModelSetManifest | None) -> dict:
    """Reject missing or added branches before comparing any model."""
    previous = {c.branch: c for c in champion_components(champion)}
    if champion and set(previous) != {c.branch for c in components}:
        raise ValueError("Model set quality requires the same champion branch names.")
    return previous
