"""Bind observed built-in preprocessing parity to exact saved state and recipes."""

from typing import Any

from ..pipeline.seal import artifact_digest
from ._partition_builtins import PROBED_BUILTINS, builtin_contract
from .fitted_pipeline import FittedPipelineArtifact
from .probed_steps import (
    CHUNK_SIZES,
    MIN_ROWS,
    PROBE_KEY,
    PROBE_VERSION,
    REQUIRED_CHECKS,
    require_comparison_policy,
    require_passed,
    require_pipeline_probe,
    require_step_rows,
)


def has_builtin_steps(artifact: FittedPipelineArtifact) -> bool:
    """Report whether an artifact needs built-in step evidence in addition to reviewed steps."""
    return any(
        record["type"] in PROBED_BUILTINS
        for record in artifact.pipeline.feature_engineer.fitted_steps
    )


def builtin_entry(record: dict, recipe: dict) -> dict[str, Any]:
    """Fingerprint exact fitted state, resolved recipe and implementation identity."""
    state, params = builtin_contract(record, recipe)
    owner = type(record["applier"])
    return {
        "name": record["name"],
        "node_type": record["type"],
        "applier": f"{owner.__module__}.{owner.__qualname__}",
        "state_sha256": artifact_digest(state).hex(),
        "config_sha256": artifact_digest(params).hex(),
        "checks": [{"name": name, "status": "passed"} for name in REQUIRED_CHECKS],
    }


def builtin_evidence(artifact: FittedPipelineArtifact, report: dict) -> list[dict[str, Any]]:
    """Seal successful observations only after the built-in's own contracts accept its state."""
    _require_rows(report.get("sample_rows"))
    require_comparison_policy(report, version_key="report_version")
    recipes = {item["name"]: dict(item) for item in artifact.pipeline.feature_engineer.steps_config}
    details = {(item["name"], item["node_type"]): item for item in report.get("steps", [])}
    entries = []
    for record in artifact.pipeline.feature_engineer.fitted_steps:
        if record["type"] not in PROBED_BUILTINS:
            continue
        entry = builtin_entry(record, recipes[record["name"]])
        detail = details.get((record["name"], record["type"]), {})
        entry["input_rows"] = require_step_rows(
            detail.get("input_rows"), report["sample_rows"], f"Built-in step {record['name']!r}"
        )
        require_passed(f"Built-in step {record['name']!r}", detail, report)
        if any(item["name"] == entry["name"] for item in entries):
            raise ValueError("Built-in probe evidence requires unique step names.")
        entries.append(entry)
    return entries


def require_probed_builtin(
    artifact: FittedPipelineArtifact | None, record: dict, recipe: dict
) -> tuple[Any, dict]:
    """Reject absent or stale built-in evidence without executing any transformation."""
    if artifact is None:
        raise ValueError("Built-in worker admission requires sealed training probe evidence.")
    evidence = _require_envelope(
        artifact.pipeline.config.get(PROBE_KEY), artifact.manifest.project_source_sha256
    )
    expected = builtin_entry(record, recipe)
    entries = evidence.get("builtins")
    matches = (
        [item for item in entries if isinstance(item, dict) and item.get("name") == record["name"]]
        if isinstance(entries, list)
        else []
    )
    if len(matches) == 1:
        expected["input_rows"] = require_step_rows(
            matches[0].get("input_rows"), evidence["rows"], f"Built-in step {record['name']!r}"
        )
    if matches != [expected]:
        raise ValueError(
            "Built-in training probe evidence is missing or changed; retrain the model."
        )
    return builtin_contract(record, recipe)


def _require_rows(rows: Any) -> None:
    """Reject samples too small for independent chunks or outside the probe's budget."""
    if type(rows) is not int or not MIN_ROWS <= rows <= 256:
        raise ValueError("Built-in training probe evidence requires 16 to 256 holdout rows.")


def _require_envelope(evidence: Any, project_sha: str | None) -> dict:
    """Keep source binding when present and reject incomplete bounded-sample evidence."""
    if not isinstance(evidence, dict):
        raise ValueError("No built-in training probe evidence was saved; retrain the model.")
    require_comparison_policy(evidence)
    if (
        type(evidence.get("version")) is not int
        or evidence.get("version") != PROBE_VERSION
        or artifact_digest(evidence.get("chunk_sizes")) != artifact_digest(list(CHUNK_SIZES))
    ):
        raise ValueError("Unsupported built-in training probe evidence version.")
    _require_rows(evidence.get("rows"))
    if evidence.get("project_source_sha256") != project_sha:
        raise ValueError("Built-in training probe evidence belongs to other project source.")
    require_pipeline_probe(evidence)
    return evidence
