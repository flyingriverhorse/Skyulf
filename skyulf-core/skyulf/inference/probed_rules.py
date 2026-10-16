"""Spark and endpoint admission evidence for saved project scoring rules.

Eligibility and output rules, and a reused pre-split recipe, run inside the
model's pyfunc on every worker batch. Training lints each callback and probes
it with the same checks as custom steps, on the holdout rows its real inputs
reach: the reused pre-split exclusion and eligibility rules on the raw sample,
output rules on up to 256 eligible holdout rows plus the model predictions and
earlier rule outputs. Each callback receives a fresh 0..n index, as in
production, and runs once more from a freshly loaded copy of the project code.
Passing evidence is sealed per rule. A reused pre-split recipe is treated as
one eligibility rule bound to its saved steps; ``Deduplicate`` and custom
filter classes in it stay rejected.
"""

import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pandas as pd
import polars as pl

from ..core.portable_state import _pack
from ..pipeline.seal import artifact_digest
from ..preprocessing.function_steps import FILTER_STEP, resolve_function
from ._probe_strategies import run_check
from .fitted_pipeline import FittedPipelineArtifact, predict_pipeline
from .pipeline_scoring import prediction_output_schema
from .probed_steps import (
    MIN_ROWS,
    PROBE_KEY,
    PROBED_NODES,
    REQUIRED_CHECKS,
    evidence_entry,
    lint_failure,
    passed_checks,
    rebind_reference,
    require_flag,
    require_passed,
)
from .project_code import load_project_module
from .project_scoring import (
    _call_rule,
    _check_rows,
    _is_null,
    _resolve,
    _scoring_eligibility,
    _typed_column,
)
from .row_local_lint import lint_callbacks

RULE_KINDS = ("eligibility", "outputs")
PRE_SPLIT = "pre_split"
SCORING_RULE_NODE = "scoring_rule"
_SAMPLE_ROWS = 256
# Pre-split row filters that decide each row from its own values and saved parameters.
_ROW_LOCAL_FILTERS = frozenset({"DropMissingRows", "ManualBounds", FILTER_STEP})

Apply = Callable[[Any], Any]


def _config(artifact: FittedPipelineArtifact) -> dict[str, Any] | None:
    """Return the saved scoring policy, if the model has one."""
    return artifact.pipeline.config.get("project_scoring")


def _pre_split_rule(saved: dict[str, Any]) -> dict[str, Any]:
    """Describe a reused pre-split recipe as one eligibility rule bound to its saved steps."""
    return {"name": PRE_SPLIT, "version": "1", "function": PRE_SPLIT, "params": saved}


def scoring_rules(config: dict[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
    """List the reused pre-split recipe, eligibility and output rules in execution order."""
    if not config:
        return []
    reused = [(PRE_SPLIT, _pre_split_rule(config[PRE_SPLIT]))] if PRE_SPLIT in config else []
    return reused + [(kind, rule) for kind in RULE_KINDS for rule in config.get(kind, [])]


def has_scoring_rules(artifact: FittedPipelineArtifact) -> bool:
    """Report whether the model saved any rule or a reused pre-split recipe."""
    return bool(scoring_rules(_config(artifact)))


def _label(kind: str, rule: dict[str, Any]) -> str:
    """Name a rule the way every message does."""
    return "Reused pre-split eligibility" if kind == PRE_SPLIT else f"Scoring rule {rule['name']!r}"


def _module(artifact: FittedPipelineArtifact) -> Any:
    """Load the saved project source that defines the rule callbacks."""
    return load_project_module(artifact.pipeline.config["project_python_source"])


def _rule_callbacks(module: Any, kind: str, rule: dict[str, Any]) -> list[tuple[str, Any, int]]:
    """Return ``(label, callback, frame_args)`` for every callback a rule runs."""
    if kind != PRE_SPLIT:
        frames = 1 if kind == "eligibility" else 2
        return [(_label(kind, rule), _resolve(module, rule["function"]), frames)]
    return [
        (f"Pre-split filter {step['name']!r}", resolve_function(step["params"]["function"]), 1)
        for step in rule["params"]["steps"]
        if step["transformer"] == FILTER_STEP
    ]


def lint_rules(artifact: FittedPipelineArtifact) -> None:
    """Statically reject batch-level code in rule callbacks before probing them."""
    rules = scoring_rules(_config(artifact))
    if not rules:
        return
    module = _module(artifact)
    for kind, rule in rules:
        for label, callback, frames in _rule_callbacks(module, kind, rule):
            message = lint_failure(label, lint_callbacks([callback], frames=frames))
            if message is not None:
                raise ValueError(message)


def preview_admission_lint(pipeline: dict[str, Any]) -> list[str]:
    """Lint a resolved recipe's custom steps and rules offline, with training's messages.

    Used by ``preview.py --action train`` so the failure appears before a job.
    """
    lines = [
        _preview_step(step)
        for step in pipeline.get("preprocessing", [])
        if step.get("transformer") in PROBED_NODES
    ]
    config = pipeline.get("project_scoring")
    if config is not None and PRE_SPLIT in config:
        try:
            require_row_local_pre_split(config[PRE_SPLIT]["steps"])
        except ValueError as exc:
            lines.append(str(exc))
    if scoring_rules(config):
        module = load_project_module(pipeline["project_python_source"])
        for kind, rule in scoring_rules(config):
            lines.extend(_preview_rule(module, kind, rule))
    return lines


def _preview_step(step: dict[str, Any]) -> str:
    """Lint one function step from its saved recipe."""
    params = step.get("params", {})
    label = f"Custom step {step['name']!r}"
    context = params.get("inference_context")
    if context not in (None, "row"):
        return f"{label} declares inference_context={context!r}: rejected on Spark."
    learn = [resolve_function(params["learn"])] if params.get("learn") else []
    apply = resolve_function(params.get("function") or params["apply"])
    return lint_failure(label, lint_callbacks([apply], learn)) or f"{label}: lint ok"


def _preview_rule(module: Any, kind: str, rule: dict[str, Any]) -> list[str]:
    """Lint every callback one saved rule runs."""
    return [
        lint_failure(label, lint_callbacks([callback], frames=frames)) or f"{label}: lint ok"
        for label, callback, frames in _rule_callbacks(module, kind, rule)
    ]


def _eligibility_apply(module: Any, rule: dict[str, Any]) -> Apply:
    """Call one eligibility rule as production does, returning its reasons as a frame."""

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        """Give the callback a fresh index and a private copy, then restore labels."""
        fresh = frame.reset_index(drop=True)
        values = _call_rule(module, rule, fresh, fresh.copy(deep=True), deepcopy(rule["params"]))
        _check_rows(values, fresh, pd.Series)
        if any(not _is_null(v) and (not isinstance(v, str) or not v.strip()) for v in values):
            raise ValueError("Eligibility reason must be a nonempty string or null.")
        result = values.astype("string").to_frame("reason")
        result.index = frame.index
        return result

    return apply


def _pre_split_apply(saved: dict[str, Any], old: str | None, new: str | None) -> Apply:
    """Run the saved pre-split recipe as scoring does, optionally from fresh project code."""
    from ..integrations.databricks.scoring.shared.scoring_pre_split import (  # noqa: PLC0415
        pre_split_exclusion_reasons,
    )

    config = deepcopy(saved)
    if old and new:
        for step in config["steps"]:
            if step["transformer"] == FILTER_STEP:
                step["params"]["function"] = rebind_reference(step["params"]["function"], old, new)

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        """Return the first excluding step per row on a fresh-index copy."""
        fresh = frame.reset_index(drop=True)
        result = pre_split_exclusion_reasons(fresh.copy(deep=True), config).to_frame("reason")
        result.index = frame.index
        return result

    return apply


def _output_apply(module: Any, rule: dict[str, Any], raw: list[str], results: list[str]) -> Apply:
    """Call one output rule on input rows plus predictions and earlier rule outputs."""

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        """Give the callback fresh-index copies, then type and relabel its columns."""
        fresh = frame.reset_index(drop=True)
        values = _call_rule(
            module,
            rule,
            fresh,
            fresh.loc[:, raw].copy(deep=True),
            fresh.loc[:, results].copy(deep=True),
            deepcopy(rule["params"]),
        )
        _check_rows(values, fresh, pd.DataFrame)
        expected = [column["name"] for column in rule["columns"]]
        if list(values.columns) != expected:
            raise ValueError("Output callback columns must exactly match declared columns.")
        typed = pd.DataFrame(
            {c["name"]: _typed_column(values[c["name"]], c["dtype"]) for c in rule["columns"]}
        )
        typed.index = frame.index
        return typed

    return apply


def probe_rule_detail(
    kind: str,
    rule: dict[str, Any],
    frame: pd.DataFrame,
    apply: Apply,
    fresh: Apply,
    where: str = "holdout",
) -> dict[str, Any]:
    """Run every required strategy on the rows this rule really receives.

    ``fresh`` is the same rule bound to a freshly loaded copy of the project code.
    """
    detail: dict[str, Any] = {"name": rule["name"], "kind": kind, "checks": []}
    if len(frame) < MIN_ROWS:
        detail.update(
            status="not_run",
            reason=f"only {len(frame)} rows reach this rule",
            hint=(
                f"Only {len(frame)} {where} rows reach this rule; widen the {where} or relax "
                "eligibility."
            ),
        )
        return detail
    full, expected = run_check("full", apply, frame, None)
    detail["checks"].append(full)
    if expected is not None:
        detail["checks"].extend(
            run_check(name, fresh if name == "fresh_module" else apply, frame, expected)[0]
            for name in REQUIRED_CHECKS[1:]
        )
    failed = expected is None or any(c["status"] != "passed" for c in detail["checks"])
    detail["status"] = "failed" if failed else "passed"
    return detail


def _predictions(
    artifact: FittedPipelineArtifact, rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return surviving raw rows and typed predictions in the same positional order."""
    native = pl.from_pandas(rows) if artifact.manifest.fitted_engine == "polars" else rows
    predicted = predict_pipeline(native, artifact).reset_index(drop=True)
    if "scoring_status" in predicted:
        keep = predicted["scoring_status"].eq("predicted")
        rows = rows.loc[keep.to_numpy()].reset_index(drop=True)
        predicted = predicted.loc[keep].reset_index(drop=True)
    return rows, pd.DataFrame(
        {
            column.name: _typed_column(predicted[column.name], column.dtype)
            for column in prediction_output_schema(artifact)
        }
    )


def probe_rules(
    artifact: FittedPipelineArtifact, holdout: pd.DataFrame | pl.DataFrame, fresh: Any
) -> list[dict[str, Any]]:
    """Probe every saved rule on the holdout rows its real inputs reach.

    ``holdout`` holds the model's raw input columns; ``fresh`` is a freshly loaded
    copy of the project module.
    """
    config = _config(artifact)
    if not scoring_rules(config) or config is None:
        return []
    module = _module(artifact)
    full = holdout.to_pandas() if isinstance(holdout, pl.DataFrame) else holdout.copy(deep=True)
    full = full.reset_index(drop=True)
    sample = full.head(_SAMPLE_ROWS)
    details = []
    remaining = sample
    if PRE_SPLIT in config:
        saved = config[PRE_SPLIT]
        apply = _pre_split_apply(saved, None, None)
        fresh_apply = _pre_split_apply(saved, module.__name__, fresh.__name__)
        rule = _pre_split_rule(saved)
        details.append(probe_rule_detail(PRE_SPLIT, rule, sample, apply, fresh_apply))
        remaining = sample.loc[apply(sample)["reason"].isna()].reset_index(drop=True)
    for rule in config["eligibility"]:
        apply, fresh_apply = _eligibility_apply(module, rule), _eligibility_apply(fresh, rule)
        details.append(probe_rule_detail("eligibility", rule, remaining, apply, fresh_apply))
    if config["outputs"]:
        details.extend(_probe_outputs(artifact, config, full, module, fresh))
    return details


def _probe_outputs(
    artifact: FittedPipelineArtifact, config: dict, full: pd.DataFrame, module: Any, fresh: Any
) -> list[dict[str, Any]]:
    """Probe output rules on up to 256 eligible rows taken from the whole holdout."""
    reasons = _scoring_eligibility(full, config, module)
    eligible = full.loc[reasons.isna().to_numpy()].head(_SAMPLE_ROWS).reset_index(drop=True)
    eligible, results = (
        _predictions(artifact, eligible) if len(eligible) else (eligible, pd.DataFrame())
    )
    details = []
    for rule in config["outputs"]:
        raw, previous = list(eligible.columns), list(results.columns)
        apply = _output_apply(module, rule, raw, previous)
        fresh_apply = _output_apply(fresh, rule, raw, previous)
        joined = pd.concat([eligible, results], axis=1)
        details.append(probe_rule_detail("outputs", rule, joined, apply, fresh_apply))
        if len(joined):
            results = pd.concat([results, apply(joined)], axis=1)
    return details


def rule_identity(kind: str, rule: dict[str, Any]) -> dict[str, Any]:
    """Bind evidence to the exact saved rule: version, function, params and columns."""
    return {
        "name": rule["name"],
        "kind": kind,
        "version": rule["version"],
        "function": rule["function"],
        "params_sha256": artifact_digest(rule["params"]).hex(),
        "rule_sha256": artifact_digest(rule).hex(),
    }


def rule_evidence(
    artifact: FittedPipelineArtifact, details: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Require every rule to pass every check and return its sealed evidence entries."""
    by_rule = {(detail["kind"], detail["name"]): detail for detail in details}
    evidence = []
    for kind, rule in scoring_rules(_config(artifact)):
        detail = by_rule.get((kind, rule["name"]), {})
        require_passed(_label(kind, rule), detail, {"status": "not_run"})
        evidence.append({**rule_identity(kind, rule), "checks": passed_checks()})
    return evidence


def require_row_local_pre_split(steps: list[dict[str, Any]]) -> None:
    """Admit only reused pre-split steps that look at one row at a time."""
    from ..integrations.databricks.training.fitting.pre_split import (  # noqa: PLC0415
        FIXED_TYPES,
    )

    for step in steps:
        kind = step["transformer"]
        if kind == "Deduplicate":
            raise ValueError(
                f"Reused pre-split step {step['name']!r} is Deduplicate, which depends on other "
                "rows; drop it from the scoring reuse or use SCORING_MODE='custom'."
            )
        if kind not in FIXED_TYPES and kind not in _ROW_LOCAL_FILTERS:
            raise ValueError(
                f"Reused pre-split step {step['name']!r} ({kind}) is a custom filter class; "
                "Spark and endpoints admit fixed built-in steps, DropMissingRows, "
                "ManualBounds and filter_step functions. Use filter_step or "
                "SCORING_MODE='custom'."
            )


def require_probed_rules(artifact: FittedPipelineArtifact) -> tuple[tuple[str, str, str, str], ...]:
    """Admit saved scoring rules only with matching, all-passed sealed evidence.

    Returns ``(name, node_type, config_json, rule_sha256)`` for each admitted rule.
    """
    config = _config(artifact)
    if config is None:
        return ()
    if type(config) is not dict or not set(RULE_KINDS) <= set(config) <= {*RULE_KINDS, PRE_SPLIT}:
        raise ValueError("Saved scoring configuration is malformed; retrain the model.")
    if PRE_SPLIT in config:
        require_row_local_pre_split(config[PRE_SPLIT]["steps"])
    rules = scoring_rules(config)
    if rules:
        require_flag(artifact.pipeline.config)
    return tuple(_admit_rule(artifact, kind, rule) for kind, rule in rules)


def _admit_rule(
    artifact: FittedPipelineArtifact, kind: str, rule: dict[str, Any]
) -> tuple[str, str, str, str]:
    """Match one saved rule with its sealed evidence, naming the rule on failure."""
    identity = rule_identity(kind, rule)
    try:
        entry = evidence_entry(
            artifact.pipeline.config.get(PROBE_KEY),
            artifact.manifest.project_source_sha256,
            "rules",
            {"name": rule["name"], "kind": kind},
        )
        if {key: entry.get(key) for key in identity} != identity:
            raise ValueError("Training probe evidence does not match this saved rule.")
        if entry.get("checks") != passed_checks():
            raise ValueError("Training probe evidence does not show every check passed.")
    except ValueError as exc:
        raise ValueError(f"{_label(kind, rule)}: {exc}") from exc
    config_json = json.dumps(_pack(rule), sort_keys=True)
    return rule["name"], f"{SCORING_RULE_NODE}:{kind}", config_json, identity["rule_sha256"]
