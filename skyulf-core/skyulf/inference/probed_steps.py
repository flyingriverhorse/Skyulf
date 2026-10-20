"""Spark admission evidence for project row steps built with plain functions.

Core cannot prove that a project ``apply`` function uses only its saved state.
Training therefore lints the step's source (``row_local_lint``) and runs the
preprocessing probe on held-out rows, then seals the passing result in
``pipeline.config``. Spark and online-feature projects do this automatically;
other projects opt in with ``admit_probed_custom_steps``. A ``column_step``
or ``fitted_step`` is admitted when it is undeclared or declares
``inference_context="row"`` and the sealed evidence matches its saved state.
Passing a bounded sample is evidence, not proof.
"""

import importlib
import json
import sys
from collections.abc import Callable, Mapping
from typing import Any

from ..pipeline.seal import artifact_digest
from ..preprocessing.function_steps import COLUMN_STEP, FITTED_STEP, resolve_function
from ._probe_frames import COMPARISON_VERSION, float_tolerance
from .fitted_pipeline import FittedPipelineArtifact
from .row_local_lint import (
    BATCH_SIZE,
    LINT_VERSION,
    MODULE_STATE,
    OTHER_ROWS,
    LintFinding,
    lint_callbacks,
)

FLAG_KEY = "admit_probed_custom_steps"
PROBE_KEY = "partition_probe"
PROBED_NODES = (COLUMN_STEP, FITTED_STEP)
PROBE_VERSION = COMPARISON_VERSION
CHUNK_SIZES = (1, 2, 7)
PIPELINE_CHECKS = ("full", "repeat", "chunks:1", "chunks:7")
REQUIRED_CHECKS = (
    "full",
    "repeat",
    *(f"chunks:{size}" for size in CHUNK_SIZES),
    "reverse",
    "index_shift",
    "empty",
    "perturb",
    "fresh_module",
)
EVIDENCE_CHECKS = ("lint", *REQUIRED_CHECKS)
# More than twice the largest chunk, so every chunk check sees several full chunks.
MIN_ROWS = 16
GUIDE = (
    "https://www.skyulf.com/manual/user_guide/databricks_spark_scoring.html#when-the-probe-fails"
)
_OTHER_ROWS = ("group", "window", "global")
_RETRAIN = "retrain the model so training probes it again"
_ROW_LOCAL_HINT = "Compute batch statistics in learn() and read them from state in apply()."
_HINTS = {
    "index_shift": "Do not read df.index; Spark gives every batch a fresh 0, 1, 2, ... index.",
    "empty": "Return an empty result with the same columns and dtypes for empty input.",
    "fresh_module": (
        "Return learned values from learn() as state; do not keep them in classes, function "
        "attributes, default arguments or module globals."
    ),
}
_LINT_HINTS = {
    BATCH_SIZE: (
        "Use per-row column values instead of the row count (len(df), df.shape[0]); an "
        "empty-batch guard such as 'if len(df) == 0: return ...' is fine."
    ),
    MODULE_STATE: (
        "Keep values in the state returned by learn(), not in module globals, class or "
        "function attributes, mutable default arguments or os.environ."
    ),
    OTHER_ROWS: (
        "Compute it in learn() and read it from state in apply(), or move it to a groups/ "
        "Spark producer that sees all rows."
    ),
}
_LINT_STATISTIC = (
    "Compute it in learn() and read it from state in apply(), or move the calculation to a "
    "groups/ Spark producer; for a per-row calculation write axis=1."
)


def guided(message: str, hint: str) -> str:
    """End a failure with one fix hint and the guide section that explains the options."""
    return f"{message} {hint} See {GUIDE}"


def function_state_sha256(state: Any) -> str:
    """Digest the saved step artifact that every worker passes to ``apply``."""
    return artifact_digest(state).hex()


def without_partition_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a pipeline config without the evidence training adds after fitting."""
    return {key: value for key, value in config.items() if key != PROBE_KEY}


def eligible_probed_steps(artifact: FittedPipelineArtifact) -> list[dict]:
    """Return function steps that are undeclared or declare row-only inference context."""
    return [
        record
        for record in artifact.pipeline.feature_engineer.fitted_steps
        if record["type"] in PROBED_NODES
        and isinstance(record["artifact"], dict)
        and record["artifact"].get("inference_context") in (None, "row")
    ]


def has_function_steps(artifact: FittedPipelineArtifact) -> bool:
    """Report whether a fitted pipeline contains any column or fitted function step."""
    return any(
        record["type"] in PROBED_NODES for record in artifact.pipeline.feature_engineer.fitted_steps
    )


def rebind_reference(reference: str, old: str, new: str) -> str:
    """Point a saved ``module:function`` reference at another copy of the project root.

    The rebound submodule is imported at once: a multi-file project package
    loads submodules such as ``pre_split`` lazily, and filter validation needs
    them present in ``sys.modules``.
    """
    if reference == old or reference.startswith((old + ".", old + ":")):
        rebound = new + reference[len(old) :]
        importlib.import_module(rebound.partition(":")[0])
        return rebound
    return reference


def fresh_rebinder(old: str, new: str) -> Callable[[dict], dict]:
    """Return a function rebinding a step record's references to fresh project code."""

    def rebind(record: dict) -> dict:
        """Rewrite function, learn and apply references in the saved state and params."""
        for state in (record["artifact"], record.get("params", {})):
            if isinstance(state, dict):
                for key in ("function", "learn", "apply"):
                    if isinstance(state.get(key), str):
                        state[key] = rebind_reference(state[key], old, new)
        return record

    return rebind


def require_project_source(artifact: FittedPipelineArtifact) -> str:
    """Return the saved project source; admission needs it so workers run probed code."""
    source = artifact.pipeline.config.get("project_python_source")
    if artifact.manifest.project_source_sha256 is None or not source:
        raise ValueError(
            "Spark admission of custom steps requires saved project source "
            "(preprocessing.py), so Spark workers run the probed code."
        )
    return source


def lint_failure(label: str, findings: tuple[LintFinding, ...]) -> str | None:
    """Describe the first lint finding with its location, one fix hint and the guide."""
    if not findings:
        return None
    first = findings[0]
    more = f" ({len(findings) - 1} more finding(s))" if len(findings) > 1 else ""
    return guided(
        f"{label} {first.problem} ({first.file} line {first.line}: {first.call}){more}.",
        _LINT_HINTS.get(first.problem, _LINT_STATISTIC),
    )


def lint_steps(artifact: FittedPipelineArtifact) -> None:
    """Statically reject batch-level code in eligible steps before running the probe."""
    for record in eligible_probed_steps(artifact):
        state = record["artifact"]
        apply_ref = state.get("function") or state.get("apply")
        learn = [resolve_function(state["learn"])] if state.get("learn") else []
        message = lint_failure(
            f"Custom step {record['name']!r}",
            lint_callbacks([resolve_function(apply_ref)], learn),
        )
        if message is not None:
            raise ValueError(message)


def require_sample_rows(rows: int, where: str = "holdout") -> None:
    """Reject evidence from too few rows for the chunk checks to mean anything."""
    if rows < MIN_ROWS:
        raise ValueError(
            guided(
                f"Spark admission needs at least {MIN_ROWS} {where} rows to probe custom "
                f"steps and scoring rules; the {where} has {rows}.",
                f"Use a larger {where}.",
            )
        )


def require_step_rows(rows: Any, maximum: int, label: str) -> int:
    """Require enough actual step inputs, independently of the original raw sample count."""
    if type(rows) is not int or not MIN_ROWS <= rows <= min(maximum, 256):
        raise ValueError(
            f"{label} requires 16 to 256 actual input rows, bounded by the raw holdout; "
            "use a holdout with at least 16 rows surviving earlier filters and retrain."
        )
    return rows


def partition_probe_evidence(artifact: FittedPipelineArtifact, report: dict) -> dict[str, Any]:
    """Build sealed evidence from a probe report, or fail naming the step and check.

    Every eligible step must pass every required check, including ``empty``:
    Spark can hand a worker an empty batch, so ``not_supported`` is a failure.
    Digests come from the loaded artifact, never from the report.
    """
    project_sha = artifact.manifest.project_source_sha256
    if project_sha is None:
        raise ValueError(
            "Spark admission of custom steps requires saved project source "
            "(preprocessing.py), so Spark workers run the probed code."
        )
    require_sample_rows(report["sample_rows"])
    require_comparison_policy(report, version_key="report_version")
    eligible = eligible_probed_steps(artifact)
    _require_unique(eligible)
    steps = {(step.get("name"), step.get("node_type")): step for step in report.get("steps", [])}
    evidence = []
    for record in eligible:
        detail = steps.get((record["name"], record["type"]), {})
        input_rows = require_step_rows(
            detail.get("input_rows"), report["sample_rows"], f"Custom step {record['name']!r}"
        )
        require_passed(f"Custom step {record['name']!r}", detail, report)
        evidence.append(
            {
                "name": record["name"],
                "node_type": record["type"],
                "state_sha256": function_state_sha256(record["artifact"]),
                "input_rows": input_rows,
                "checks": passed_checks(),
            }
        )
    return {
        "version": PROBE_VERSION,
        "float_tolerance": float_tolerance(),
        "lint_version": LINT_VERSION,
        "rows": report["sample_rows"],
        "chunk_sizes": list(CHUNK_SIZES),
        "project_source_sha256": project_sha,
        "steps": evidence,
    }


def passed_checks() -> list[dict[str, str]]:
    """Return the exact check list sealed for an admitted step or rule."""
    return [{"name": name, "status": "passed"} for name in EVIDENCE_CHECKS]


def _require_unique(records: list[dict]) -> None:
    """Keep evidence unambiguous: one entry per step name and node type."""
    seen = set()
    for record in records:
        key = (record["name"], record["type"])
        if key in seen:
            raise ValueError(
                f"Custom step name {record['name']!r} is used by more than one "
                f"{record['type']} step; give each probed step a unique name."
            )
        seen.add(key)


def module_state(artifact: FittedPipelineArtifact) -> dict[str, dict[str, str]]:
    """Snapshot JSON-serializable module globals used by each eligible step's functions.

    Only plain JSON values are compared, so caches held in other objects are not
    seen. Module-level caches are not allowed in probed steps.
    """
    state: dict[str, dict[str, str]] = {}
    for record in eligible_probed_steps(artifact):
        refs = [record["artifact"].get(key) for key in ("function", "learn", "apply")]
        modules = {ref.partition(":")[0] for ref in refs if isinstance(ref, str)}
        snapshot = state.setdefault(record["name"], {})
        for name in sorted(modules):
            module = sys.modules.get(name)
            snapshot.update(_json_globals(name, vars(module) if module is not None else {}))
    return state


def _json_globals(module: str, namespace: dict[str, Any]) -> dict[str, str]:
    """Encode public JSON values so later changes can be compared cheaply."""
    encoded = {}
    for key, value in namespace.items():
        if key.startswith("__"):
            continue
        try:
            encoded[f"{module}.{key}"] = json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            continue
    return encoded


def require_unchanged_module_state(
    artifact: FittedPipelineArtifact, before: dict[str, dict[str, str]]
) -> None:
    """Fail when probing a step changed its module's globals, e.g. a memoized statistic."""
    for name, after in module_state(artifact).items():
        old = before.get(name, {})
        changed = sorted(key for key in old.keys() | after.keys() if old.get(key) != after.get(key))
        if changed:
            variable = changed[0].rsplit(".", 1)[-1]
            raise ValueError(
                guided(
                    f"Custom step {name!r} changed module-level state ({variable}) while it "
                    "was probed (state_mutation); module-level caches are not allowed.",
                    _LINT_HINTS[MODULE_STATE],
                )
            )


def require_passed(label: str, detail: dict, report: dict) -> None:
    """Raise a user-facing message for the first required check that did not pass."""
    checks = {check["name"]: check for check in detail.get("checks", [])}
    for check_name in REQUIRED_CHECKS:
        check = checks.get(check_name)
        if check is None:
            raise ValueError(_not_probed(label, detail, report))
        if check["status"] != "passed":
            raise ValueError(_failed(label, check_name, check))


def _not_probed(label: str, detail: dict, report: dict) -> str:
    """Explain a step or rule that never reached the checks."""
    status = detail.get("status") or report.get("error_type") or report.get("status")
    reason = detail.get("reason")
    hint = detail.get("hint", "Every earlier preprocessing step must pass the probe first.")
    return guided(f"{label} could not be probed ({status}{': ' + reason if reason else ''}).", hint)


def _failed(label: str, check_name: str, check: dict) -> str:
    """Name the failed check with one specific fix."""
    reason = check.get("reason")
    if check["status"] == "not_supported":
        return guided(
            f"{label} failed on an empty batch (empty: not_supported); Spark can send empty "
            "batches.",
            _HINTS["empty"],
        )
    if check_name == "fresh_module" and reason == "apply_error":
        error = f"{check.get('error_type')}: {check.get('error_message', '')}".rstrip(": ")
        return guided(
            f"{label} raised in a freshly loaded copy of the project code (fresh_module: {error}).",
            "Module-level code and imports must work in a new process; do not rely on "
            "values the training process set up before scoring.",
        )
    if check_name == "perturb" and reason == "apply_error":
        return guided(
            f"{label} raised on a changed input value (perturb: apply_error).",
            "Handle unusual but valid values, such as larger numbers or unseen categories, "
            "without raising.",
        )
    return guided(
        f"{label} {_phrase(check_name)} ({check_name}{': ' + reason if reason else ''}); it "
        "must use only its saved training state.",
        _HINTS.get(check_name, _ROW_LOCAL_HINT),
    )


def _phrase(check: str) -> str:
    """Describe a probe strategy in words a project author can act on."""
    if check.startswith("chunks:"):
        return f"changed results when scored in chunks of {check.split(':')[1]}"
    return {
        "full": "failed on the holdout sample",
        "repeat": "changed results when scored again",
        "reverse": "changed results when rows were reversed",
        "index_shift": "changed results when the batch index was renumbered",
        "empty": "returned a different result (often the dtype) for an empty batch",
        "perturb": "changed other rows' results when one row's inputs changed",
        "fresh_module": "gave different results from a freshly loaded copy of the project code",
    }.get(check, "failed")


def require_flag(config: Mapping[str, Any]) -> None:
    """Require the sealed opt-in that training sets whenever it probes."""
    if config.get(FLAG_KEY) is not True:
        raise ValueError(
            "Custom steps and scoring rules run on Spark or endpoints only with training "
            "probe evidence. Spark and online-feature projects create it automatically; "
            "other projects set allow_tested_custom_steps: true in config/training.yml."
        )


def require_probed_step(
    config: Mapping[str, Any], project_sha: str | None, name: str, node: str, state: Any
) -> None:
    """Admit one function step only with sealed, matching, all-passed evidence."""
    require_flag(config)
    context = state.get("inference_context") if isinstance(state, dict) else "unknown"
    if context not in (None, "row"):
        raise ValueError(
            f"This step declares inference_context={context!r}, so it needs other rows and "
            "cannot run on independent Spark batches; keep the project on "
            "batch_compute: single_node."
        )
    entry = evidence_entry(
        config.get(PROBE_KEY), project_sha, "steps", {"name": name, "node_type": node}
    )
    require_step_rows(entry.get("input_rows"), config[PROBE_KEY]["rows"], f"Custom step {name!r}")
    if entry.get("state_sha256") != function_state_sha256(state):
        raise ValueError(
            f"Training probe evidence does not match this step's saved state; {_RETRAIN}."
        )
    if entry.get("checks") != passed_checks():
        raise ValueError(f"Training probe evidence does not show every check passed; {_RETRAIN}.")


def evidence_entry(
    probe: Any,
    project_sha: str | None,
    section: str,
    identity: dict[str, Any],
    *,
    source_key: str = "project_source_sha256",
) -> dict:
    """Return the single sealed entry matching ``identity`` after checking the envelope.

    ``source_key`` names the source digest the evidence is bound to (model-set
    composition evidence binds ``composition_source_sha256``).
    """
    _check_envelope(probe, project_sha, source_key)
    entries = probe.get(section)
    matches = [
        entry
        for entry in (entries if isinstance(entries, list) else [])
        if isinstance(entry, dict) and all(entry.get(k) == v for k, v in identity.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Training probe evidence has no entry for this {section[:-1]}; {_RETRAIN}."
        )
    return matches[0]


def require_scored_rows(rows: Any, maximum: int) -> int:
    """Reject a final serving probe with too few actually executed prediction outcomes."""
    if type(rows) is not int or not MIN_ROWS <= rows <= maximum:
        raise ValueError(
            "Raw pipeline partition probe needs at least 16 actually predicted outcomes "
            "within its raw sample; widen or reorder a representative holdout and retrain."
        )
    return rows


def require_pipeline_probe(evidence: dict) -> None:
    """Require complete raw-input observations, in addition to individual component probes."""
    detail = evidence.get("pipeline")
    if (
        type(detail) is not dict
        or set(detail) != {"rows", "scored_rows", "checks"}
        or detail.get("checks") != [{"name": name, "status": "passed"} for name in PIPELINE_CHECKS]
    ):
        raise ValueError(
            "Missing or invalid raw pipeline partition probe evidence; retrain the model."
        )
    rows = detail.get("rows")
    if type(rows) is not int or not MIN_ROWS <= rows <= 256 or rows != evidence.get("rows"):
        raise ValueError("Invalid raw pipeline partition probe row count; retrain the model.")
    require_scored_rows(detail.get("scored_rows"), rows)


def require_comparison_policy(probe: dict, *, version_key: str = "version") -> None:
    """Reject old evidence and policy changes rather than trusting caller-supplied tolerances."""
    policy = probe.get("float_tolerance")
    expected = float_tolerance()
    if (
        type(probe.get(version_key)) is not int
        or probe[version_key] != PROBE_VERSION
        or type(policy) is not dict
        or set(policy) != set(expected)
        or any(
            type(policy[key]) is not float or policy[key] != value
            for key, value in expected.items()
        )
    ):
        raise ValueError(f"Unsupported training probe comparison policy; {_RETRAIN}.")


def _check_envelope(probe: Any, project_sha: str | None, source_key: str) -> None:
    """Require current comparison evidence with fixed checks bound to the saved project source."""
    if not isinstance(probe, dict):
        raise ValueError(f"No training probe evidence was saved; {_RETRAIN}.")
    require_comparison_policy(probe)
    if (
        probe.get("version") != PROBE_VERSION
        or probe.get("lint_version") != LINT_VERSION
        or probe.get("chunk_sizes") != list(CHUNK_SIZES)
    ):
        raise ValueError(f"Unsupported training probe evidence version; {_RETRAIN}.")
    if project_sha is None or probe.get(source_key) != project_sha:
        raise ValueError(f"Training probe evidence belongs to other project source; {_RETRAIN}.")
    rows = probe.get("rows")
    if type(rows) is not int or rows < MIN_ROWS:
        raise ValueError(
            f"Training probe evidence covers fewer than {MIN_ROWS} holdout rows; {_RETRAIN}."
        )
    if source_key == "project_source_sha256":
        require_pipeline_probe(probe)
