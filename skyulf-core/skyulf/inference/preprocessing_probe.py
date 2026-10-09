"""Explicit diagnostics for the real saved preprocessing inference path.

This executes trusted model code on a bounded sample. Passing observations are
not partition admission, a sandbox, or proof for unobserved inputs. Load models
through load_local_pipeline first to verify their packages and payload checksums.
"""

from copy import deepcopy
from typing import Any

import pandas as pd

from ..core.schema import SkyulfSchema
from ..pipeline import SkyulfPipeline
from ..pipeline.seal import artifact_digest
from ..preprocessing._spark import use_spark
from ..preprocessing.inference_context import get_inference_capability
from ..preprocessing.pipeline import FeatureEngineer, _apply_prediction_step
from ..registry import NodeRegistry
from ._fitted_contract import check_fitted_schemas, resolve_fitted_step
from ._probe_frames import (
    Frame,
    ProbeFailure,
    assert_same,
    copy_frame,
    probe_sizes,
    reverse_frame,
    slice_frame,
    validate_frame,
)
from .local_pipeline import LocalPipelineArtifact, validate_local_input


def _state_digest(record: dict) -> str:
    """Include raw applier attributes so instance counters are not hidden by class identity."""
    return artifact_digest(
        (record["artifact"], record.get("params", {}), vars(record["applier"]))
    ).hex()


def _check_record_owner(record: dict, project_sha: str | None) -> None:
    """Bind ownership to current registration or the loader's captured source identity."""
    owner = type(record["applier"])
    try:
        registered = NodeRegistry.get_applier(record["type"])
    except ValueError:
        prefix = f"_skyulf_project_{project_sha}"
        captured = owner.__module__ == prefix or owner.__module__.startswith(prefix + ".")
        identity = record["type"]
        if not (
            project_sha
            and captured
            and identity.startswith(owner.__module__ + ".")
            and identity.endswith("." + owner.__qualname__)
        ):
            raise ValueError("Saved applier is not owned by the captured project.") from None
        return
    if owner is not registered:
        raise ValueError("Saved applier disagrees with the registered implementation.")


def _prepare(
    record: dict, config: dict, engine: str, project_sha: str | None, *, active: bool
) -> dict[str, Any]:
    """Inspect a detached record using its own fitted-state and context contracts."""
    _check_record_owner(record, project_sha)
    before = _state_digest(record)
    if active:
        state, params, validated = resolve_fitted_step(record, config, require_portable=False)
        local_validation = vars(type(record["applier"])).get("validate_inference_state")
        if local_validation is not None:
            type(record["applier"]).validate_inference_state(state)
            validated = True
    else:
        if config["name"] != record["name"] or config["transformer"] != record["type"]:
            raise ValueError("Configured step disagrees with fitted name/type.")
        state, params, validated = record["artifact"], record.get("params", {}), False
    capability = get_inference_capability(
        record["type"], params, state, engine=engine, applier=record["applier"]
    )
    if _state_digest(record) != before:
        raise ProbeFailure("state_mutation")
    return {
        "state_sha256": before,
        "state_validation": "node_owned" if validated else "unavailable",
        "context": capability.context if capability else "unknown",
        "row_effect": capability.row_effect if capability else "unknown",
        "history_mode": state.get("history_mode") if isinstance(state, dict) else None,
    }


def _apply_checked(record: dict, frame: Frame, max_rows: int, max_bytes: int) -> Frame:
    """Call the existing prediction applier and inspect mutations even on exceptions."""
    argument = copy_frame(frame)
    before_frame = copy_frame(argument)
    before_state = _state_digest(record)
    try:
        result = _apply_prediction_step(argument, record)
    finally:
        if _state_digest(record) != before_state:
            raise ProbeFailure("state_mutation")
        assert_same(before_frame, argument, "input_mutation")
    try:
        validate_frame(result, max_rows, max_bytes)
    except (TypeError, ValueError):
        raise ProbeFailure("invalid_output_frame_or_budget") from None
    if type(result) is not type(frame):
        raise ProbeFailure("output_engine_mismatch")
    if isinstance(frame, pd.DataFrame) and not frame.index.equals(result.index):
        raise ProbeFailure("output_row_order_mismatch")
    return result


def _run_check(
    name: str,
    record: dict,
    frame: Frame,
    expected: Frame | None,
    budgets: tuple[int, int],
) -> tuple[dict, Frame | None]:
    """Return redacted evidence for one strategy, preserving mutation failures on empty input."""
    try:
        result = _execute_check(name, deepcopy(record), frame, expected, budgets)
        return {"name": name, "status": "passed"}, result
    except ProbeFailure as exc:
        return {"name": name, "status": "failed", "reason": str(exc)}, None
    except Exception as exc:  # noqa: BLE001 - report arbitrary trusted callback failures
        return {
            "name": name,
            "status": "not_supported" if name == "empty" else "failed",
            "reason": "apply_error",
            "error_type": type(exc).__name__,
        }, None


def _execute_check(
    name: str,
    record: dict,
    frame: Frame,
    expected: Frame | None,
    budgets: tuple[int, int],
) -> Frame | None:
    """Use unchanged apply code for full, repeated, partitioned and reordered requests."""
    if name == "full":
        return _apply_checked(record, frame, *budgets)
    assert expected is not None
    if name.startswith("chunks:"):
        _check_chunks(record, frame, expected, int(name.split(":")[1]), budgets)
    elif name == "reverse":
        actual = _apply_checked(record, reverse_frame(frame), *budgets)
        assert_same(expected, reverse_frame(actual))
    elif name == "empty":
        actual = _apply_checked(record, slice_frame(frame, 0, 0), *budgets)
        assert_same(slice_frame(expected, 0, 0), actual)
    else:
        # Reuse the same detached state twice to expose request-to-request changes.
        for _ in range(2):
            assert_same(expected, _apply_checked(record, frame, *budgets))
    return None


def _check_chunks(
    record: dict, frame: Frame, expected: Frame, size: int, budgets: tuple[int, int]
) -> None:
    """Compare each chunk before concatenation could conceal dtype promotion."""
    for start in range(0, len(frame), size):
        actual = _apply_checked(record, slice_frame(frame, start, size), *budgets)
        assert_same(slice_frame(expected, start, size), actual)


def _probe_step(
    record: dict,
    config: dict,
    frame: Frame,
    engine: str,
    sizes: tuple[int, ...],
    budgets: tuple[int, int],
    *,
    active: bool,
    project_sha: str | None,
) -> tuple[dict, Frame]:
    """Report declared context and empirical behavior without granting execution support."""
    detail: dict[str, Any] = {
        "name": record["name"],
        "node_type": record["type"],
        "action": "apply" if active else "skip_preserve_rows",
        "context": "unknown",
        "checks": [],
    }
    try:
        detached = deepcopy(record)
        detail.update(_prepare(detached, deepcopy(config), engine, project_sha, active=active))
    except Exception as exc:  # noqa: BLE001 - custom node contract failures become evidence
        detail.update(
            status="failed", reason="invalid_step_contract", error_type=type(exc).__name__
        )
        return detail, frame
    if not active:
        detail["status"] = "skipped"
        return detail, frame
    if detail["context"] in {"group", "window", "global"} or detail["history_mode"] == "carry":
        detail["status"] = "requires_context"
        return detail, frame
    return _probe_checks(detached, detail, frame, sizes, budgets)


def _probe_checks(
    record: dict, detail: dict, frame: Frame, sizes: tuple[int, ...], budgets: tuple[int, int]
) -> tuple[dict, Frame]:
    """Keep the first successful full output as the next step's observed input."""
    full, transformed = _run_check("full", record, frame, None, budgets)
    detail["checks"].append(full)
    if transformed is None:
        detail["status"] = "failed"
        return detail, frame
    for name in ("repeat", *(f"chunks:{size}" for size in sizes), "reverse", "empty"):
        check, _ = _run_check(name, record, frame, transformed, budgets)
        detail["checks"].append(check)
    detail["status"] = (
        "failed" if any(c["status"] == "failed" for c in detail["checks"]) else "passed"
    )
    return detail, transformed


def _records(artifact: LocalPipelineArtifact) -> list[tuple[dict, dict, bool]]:
    """Bind fitted ordering to recipes and reuse the engineer's actual prediction skips."""
    engineer = artifact.pipeline.feature_engineer
    if artifact_digest(artifact.pipeline.config.get("preprocessing", [])) != artifact_digest(
        engineer.steps_config
    ):
        raise ValueError("Pipeline recipe disagrees with fitted engineer configuration.")
    unrecorded = {"TrainTestSplitter", "Split", "feature_target_split"}
    configs = [
        dict(step) for step in engineer.steps_config if step["transformer"] not in unrecorded
    ]
    if len(configs) != len(engineer.fitted_steps):
        raise ValueError("Incomplete fitted preprocessing history.")
    active = {id(record) for record in engineer._transform_steps(preserve_rows=True)}
    return [
        (record, config, id(record) in active)
        for record, config in zip(engineer.fitted_steps, configs, strict=True)
    ]


def _validate_artifact(artifact: LocalPipelineArtifact) -> None:
    """Require the standard loaded pipeline and local schema contract."""
    if type(artifact) is not LocalPipelineArtifact:
        raise TypeError("Expected a loaded LocalPipelineArtifact.")
    if (
        type(artifact.pipeline) is not SkyulfPipeline
        or type(artifact.pipeline.feature_engineer) is not FeatureEngineer
    ):
        raise ValueError("Probe requires standard fitted preprocessing orchestration.")
    if artifact.manifest.fitted_engine not in {"pandas", "polars"}:
        raise ValueError("Probe requires a local fitted pipeline.")
    if artifact.manifest.fitted_engine != artifact.pipeline.fitted_engine:
        raise ValueError("Manifest and pipeline fitted engines disagree.")
    for component in (artifact.pipeline, artifact.pipeline.feature_engineer):
        if any(callable(value) for value in vars(component).values()):
            raise ValueError("Overridden inference methods are unsupported.")
    check_fitted_schemas(artifact)


def _finish(report: dict, artifact: LocalPipelineArtifact, frame: Frame) -> dict:
    """Compare the actual final feature schema with the saved model input schema."""
    if report["status"] != "passed":
        report["feature_schema"] = "not_run"
        return report
    schemas = artifact.pipeline._inference_schemas
    assert schemas is not None
    try:
        schemas[1].assert_compatible(
            SkyulfSchema.from_dataframe(frame), check_dtypes=True, check_order=True
        )
    except ValueError:
        report.update(status="failed", feature_schema="mismatch")
    else:
        report["feature_schema"] = "passed"
    return report


def probe_fitted_preprocessing(
    artifact: LocalPipelineArtifact,
    sample: Frame,
    *,
    chunk_sizes: tuple[int, ...] = (1, 2, 7),
    max_rows: int = 256,
    max_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Diagnose saved apply behavior on an explicit bounded local sample.

    Uses no fit/learn methods. Each strategy receives detached fitted records.
    Checks exact schema, values, order and semantic fitted-state mutation. Known
    group/window/global steps require context and stop this independent-partition
    probe. Unknown custom steps may pass samples but remain unverified for remote
    admission. Empty-input exceptions are reported as not_supported.

    Load the artifact with load_local_pipeline first; this diagnostic does not
    repeat disk/package verification. Trusted callbacks can access globals or
    external services: copies and budgets are not a code sandbox or time limit.
    Frames must contain immutable scalar cells. Budgets apply per input/output
    frame, not to arbitrary intermediate allocations inside custom functions.
    The JSON-compatible result contains no sampled values or exception messages.
    """
    sizes = probe_sizes(chunk_sizes, max_rows, max_bytes)
    _validate_artifact(artifact)
    validate_frame(sample, max_rows, max_bytes)
    if not len(sample):
        raise ValueError("Probe needs a nonempty sample; empty behavior is checked separately.")
    current = validate_local_input(copy_frame(sample), artifact)
    engineer = artifact.pipeline.feature_engineer
    use_spark(current, engineer.execution_options, engineer.frame_spec)
    validate_frame(current, max_rows, max_bytes)
    report: dict[str, Any] = {
        "report_version": 1,
        "status": "passed",
        "admission": "diagnostic_only",
        "sample_rows": len(sample),
        "fitted_engine": artifact.manifest.fitted_engine,
        "pipeline_sha256": artifact.manifest.pipeline_sha256,
        "project_source_sha256": artifact.manifest.project_source_sha256,
        "steps": [],
    }
    for record, config, active in _records(artifact):
        if report["status"] != "passed":
            report["steps"].append(
                {
                    "name": record["name"],
                    "node_type": record["type"],
                    "status": "not_run",
                    "checks": [],
                }
            )
            continue
        detail, current = _probe_step(
            record,
            config,
            current,
            artifact.manifest.fitted_engine,
            sizes,
            (max_rows, max_bytes),
            active=active,
            project_sha=artifact.manifest.project_source_sha256,
        )
        report["steps"].append(detail)
        if detail["status"] not in {"passed", "skipped"}:
            report["status"] = detail["status"]
    return _finish(report, artifact, current)
