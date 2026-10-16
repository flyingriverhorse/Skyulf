"""Spark and endpoint admission evidence for model-set combined function rules.

A combined rule reads the same row's raw inputs and the other components'
predictions, so it can be row-local. Training lints each callback and probes it
with the same checks as custom steps on up to 256 ordered training rows:
the component predictions are computed first, then each rule runs on the rows
whose required components succeeded. ``perturb`` changes one row's inputs and
component predictions and requires every other row's output to stay the same.
Passing evidence is sealed in the set's composition config, which the set
digest covers. Final whole/chunked scoring also checks the composed outputs.
Declarative ``operation`` rules need that final check when any component uses
empirical admission: tolerated component noise can be amplified by arithmetic.
Every combined output must actually score at least 16 rows in the sample.
"""

from copy import deepcopy
from typing import Any, cast

import pandas as pd

from ._probe_frames import Frame, assert_same, copy_frame, float_tolerance
from ._probe_strategies import run_check
from .fitted_pipeline import load_pipeline
from .model_set import ModelSetArtifact
from .model_set_scoring import (
    _raw_input,
    _rule_reasons,
    _score_component,
    assemble_component_outputs,
    predict_model_set,
)
from .probed_rules import probe_rule_detail, rule_identity
from .probed_steps import (
    CHUNK_SIZES,
    LINT_VERSION,
    PIPELINE_CHECKS,
    PROBE_KEY,
    PROBE_VERSION,
    _check_envelope,
    evidence_entry,
    lint_failure,
    passed_checks,
    require_passed,
    require_pipeline_probe,
    require_sample_rows,
    require_scored_rows,
)
from .project_code import fresh_project_module, load_project_module
from .project_scoring import _check_rows, _resolve, _typed_column
from .row_local_lint import lint_callbacks

COMBINED = "combined"
SOURCE_KEY = "composition_source_sha256"
_RAW, _PREDICTED = "raw:", "predicted:"
_LIMIT_ROWS, _LIMIT_BYTES = 256, 8 * 1024 * 1024


def function_rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return combined rules that run saved Python rather than a declared operation."""
    return [rule for rule in config.get("outputs", []) if "operation" not in rule]


def requires_composition_probe(artifact: ModelSetArtifact) -> bool:
    """Require final parity for functions or arithmetic on empirically admitted components."""
    config = artifact.manifest.composition_config
    if function_rules(config):
        return True
    if not config.get("outputs"):
        return False
    return any(
        PROBE_KEY
        in load_pipeline(artifact.directory / "components" / component.branch).pipeline.config
        for component in artifact.manifest.components
    )


def _source(artifact: ModelSetArtifact) -> str:
    """Read the set's saved composition source."""
    return (artifact.directory / "composition.py").read_text(encoding="utf-8")


def _module(artifact: ModelSetArtifact) -> Any:
    """Load the set's saved composition source."""
    return load_project_module(_source(artifact))


def _label(rule: dict[str, Any]) -> str:
    """Name a combined rule the way every message does."""
    return f"Combined rule {rule['name']!r}"


def _lint(artifact: ModelSetArtifact, rules: list[dict[str, Any]]) -> None:
    """Statically reject batch-level code before the probe runs."""
    module = _module(artifact)
    for rule in rules:
        findings = lint_callbacks([_resolve(module, rule["function"])], frames=2)
        message = lint_failure(_label(rule), findings)
        if message is not None:
            raise ValueError(message)


def _component_predictions(artifact: ModelSetArtifact, sample: Any) -> tuple[pd.DataFrame, ...]:
    """Score every component on the sample exactly as set scoring does before composing."""
    raw = _raw_input(sample, artifact, _LIMIT_ROWS, _LIMIT_BYTES)
    keys = list(artifact.manifest.record_key_columns)
    result = raw[keys].copy(deep=True)
    for component in artifact.manifest.components:
        keyed, _ = _score_component(
            raw, sample, artifact, component, None, False, _LIMIT_ROWS, _LIMIT_BYTES
        )
        result = assemble_component_outputs(result, keyed, keys)
    return raw, result


def _apply(module: Any, rule: dict[str, Any]) -> Any:
    """Call one combined rule as composition does, on prefixed raw and prediction columns."""

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        """Give the callback fresh-index copies, then type and relabel its columns."""
        fresh = frame.reset_index(drop=True)
        raw = _split(fresh, _RAW)
        predictions = _split(fresh, _PREDICTED)
        values = _resolve(module, rule["function"])(raw, predictions, deepcopy(rule["params"]))
        _check_rows(values, fresh, pd.DataFrame)
        if list(values.columns) != [column["name"] for column in rule["columns"]]:
            raise ValueError("Composition callback columns must exactly match declared columns.")
        typed = pd.DataFrame(
            {c["name"]: _typed_column(values[c["name"]], c["dtype"]) for c in rule["columns"]}
        )
        if typed.isna().any().any():
            raise ValueError("Composition outputs cannot be missing for eligible rows.")
        typed.index = frame.index
        return typed

    return apply


def _split(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Recover one prefixed group of columns under its original names."""
    columns = [name for name in frame.columns if name.startswith(prefix)]
    return frame.loc[:, columns].rename(columns=lambda name: name[len(prefix) :]).copy(deep=True)


def _probe(artifact: ModelSetArtifact, rules: list[dict], sample: Any) -> list[dict[str, Any]]:
    """Probe each function rule on the rows whose required components succeeded."""
    module = _module(artifact)
    raw, predictions = _component_predictions(artifact, sample)
    joined = pd.concat([raw.add_prefix(_RAW), predictions.add_prefix(_PREDICTED)], axis=1)
    details = []
    with fresh_project_module(_source(artifact)) as fresh:
        for rule in rules:
            eligible = _rule_reasons(predictions, rule).isna()
            frame = joined.loc[eligible.to_numpy()].reset_index(drop=True)
            detail = probe_rule_detail(
                COMBINED, rule, frame, _apply(module, rule), _apply(fresh, rule), "training"
            )
            details.append(detail)
    return details


def _exact_output_columns(artifact: ModelSetArtifact) -> tuple[str, ...]:
    """Keep keys and component class labels exact even when their storage type is float."""
    names = list(artifact.manifest.record_key_columns)
    for component in artifact.manifest.components:
        local = load_pipeline(artifact.directory / "components" / component.branch)
        if local.manifest.task == "classification":
            names.extend(
                f"{component.branch}__{name}" for name in ("prediction", "threshold_prediction")
            )
    return tuple(names)


def _probe_outputs(artifact: ModelSetArtifact, sample: Any) -> dict[str, Any]:
    """Detect tolerated component noise amplified into a different composed decision."""
    raw = _raw_input(sample, artifact, _LIMIT_ROWS, _LIMIT_BYTES)

    def apply(frame: Frame) -> pd.DataFrame:
        """Use actual set scoring, which reloads and validates every saved component."""
        argument = copy_frame(frame)
        try:
            return predict_model_set(
                argument, artifact, max_rows=_LIMIT_ROWS, max_bytes=_LIMIT_BYTES
            )
        finally:
            assert_same(frame, argument, "input_mutation", exact=True)

    labels = _exact_output_columns(artifact)
    expected = None
    checks = []
    scored_rows = 0
    for name in PIPELINE_CHECKS:
        check, output = run_check(name, apply, raw, expected, exact_columns=labels)
        if check != {"name": name, "status": "passed"}:
            raise ValueError(
                f"Model-set partition probe failed {name}: {check.get('reason', 'missing evidence')}. "
                "Retrain with a composition that preserves partition parity."
            )
        if name == "full":
            expected = output
            scored_rows = _scored_output_rows(artifact, output, len(raw))
        checks.append(check)
    return {"rows": len(raw), "scored_rows": scored_rows, "checks": checks}


def _scored_output_rows(artifact: ModelSetArtifact, output: Any, rows: int) -> int:
    """Do not accept a composed rule when exclusions hide its final-output behavior."""
    count = min(
        int(output[f"{rule['name']}__scoring_status"].eq("predicted").sum())
        for rule in artifact.manifest.composition_config["outputs"]
    )
    return require_scored_rows(count, rows)


def sealed_composition_config(artifact: ModelSetArtifact, sample: Any) -> dict[str, Any]:
    """Probe combined functions and final outputs after empirically admitted components.

    Save the set again with this config; the set digest then covers the evidence.
    """
    config = artifact.manifest.composition_config
    config.pop(PROBE_KEY, None)
    rules = function_rules(config)
    if not requires_composition_probe(artifact):
        return config
    if rules:
        _lint(artifact, rules)
    require_sample_rows(len(sample), "training")
    details = (
        {detail["name"]: detail for detail in _probe(artifact, rules, sample)} if rules else {}
    )
    evidence = []
    for rule in rules:
        require_passed(_label(rule), details[rule["name"]], {"status": "not_run"})
        evidence.append({**rule_identity(COMBINED, rule), "checks": passed_checks()})
    config[PROBE_KEY] = {
        "version": PROBE_VERSION,
        "float_tolerance": float_tolerance(),
        "lint_version": LINT_VERSION,
        "rows": len(sample),
        "chunk_sizes": list(CHUNK_SIZES),
        SOURCE_KEY: artifact.manifest.composition_source_sha256,
        "rules": evidence,
        "pipeline": _probe_outputs(artifact, sample),
    }
    return config


def require_probed_composition(artifact: ModelSetArtifact) -> list[str]:
    """Validate required final evidence and return the admitted combined function names."""
    config = artifact.manifest.composition_config
    if not function_rules(config) and requires_composition_probe(artifact):
        probe = config.get(PROBE_KEY)
        _check_envelope(probe, artifact.manifest.composition_source_sha256, SOURCE_KEY)
        require_pipeline_probe(cast(dict, probe))
    admitted = []
    for rule in function_rules(config):
        identity = rule_identity(COMBINED, rule)
        try:
            entry = evidence_entry(
                config.get(PROBE_KEY),
                artifact.manifest.composition_source_sha256,
                "rules",
                {"name": rule["name"], "kind": COMBINED},
                source_key=SOURCE_KEY,
            )
            require_pipeline_probe(config[PROBE_KEY])
            if {key: entry.get(key) for key in identity} != identity:
                raise ValueError("Training probe evidence does not match this saved rule.")
            if entry.get("checks") != passed_checks():
                raise ValueError("Training probe evidence does not show every check passed.")
        except ValueError as exc:
            raise ValueError(f"{_label(rule)}: {exc}") from exc
        admitted.append(rule["name"])
    return admitted
