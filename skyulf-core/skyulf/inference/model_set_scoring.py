"""Bounded, keyed execution of pinned local components and saved composition rules."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

from ._manifest import ColumnSpec
from ._model_set_operations import apply_operation, validate_operation
from .fitted_pipeline import (
    load_pipeline as load_local_pipeline,
)
from .fitted_pipeline import (
    validate_pipeline_input as validate_local_input,
)
from .pipeline_scoring import (
    _preserve_history,
)
from .pipeline_scoring import (
    pipeline_history_session as local_history_session,
)
from .pipeline_scoring import (
    score_pipeline as score_local_pipeline,
)
from .project_code import load_project_module
from .project_scoring import _check_rows, _json_copy, _resolve, _typed_column, _validate_rule

if TYPE_CHECKING:
    from .model_set import ModelSetArtifact

# Fallbacks for direct Core/PyFunc calls. Bundle entrypoints pass workflow.json's
# max_rows and max_input_mb instead; these are frame budgets, not total process RAM.
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_NULLABLE = {"float64": "Float64", "int64": "Int64", "string": "string", "bool": "boolean"}


@dataclass(frozen=True)
class ModelSetPrediction:
    """Complete results and detached history proposals for one atomic publication."""

    frame: pd.DataFrame
    history: dict[str, Any]


def _outcomes(prefix: str) -> tuple[ColumnSpec, ...]:
    """Give each independent component or rule an explicit row outcome."""
    return tuple(
        ColumnSpec(name=f"{prefix}__{name}", dtype="string")
        for name in ("scoring_status", "exclusion_reason")
    )


def model_set_schema(
    components: Any, record_key_schema: Any, config: dict
) -> tuple[ColumnSpec, ...]:
    """Describe the complete collision-free keyed publication schema."""
    columns = list(record_key_schema)
    for component in components:
        columns.extend(
            ColumnSpec(name=f"{component.branch}__{col.name}", dtype=col.dtype)
            for col in component.output_schema
        )
        native = {f"{component.branch}__{col.name}" for col in component.output_schema}
        columns.extend(col for col in _outcomes(component.branch) if col.name not in native)
    for rule in config["outputs"]:
        columns.extend(ColumnSpec(**col) for col in rule["columns"])
        columns.extend(_outcomes(rule["name"]))
    names = [col.name.casefold() for col in columns]
    if len(names) != len(set(names)):
        raise ValueError("Model-set output names collide with keys, components or rule outcomes.")
    return tuple(columns)


def model_set_output_schema(artifact: ModelSetArtifact) -> tuple[ColumnSpec, ...]:
    """Return recorded key, component and rule outputs in stable publication order."""
    manifest = artifact.manifest
    return model_set_schema(
        manifest.components, manifest.record_key_schema, manifest.composition_config
    )


def _dependencies(rule: dict, branches: set[str]) -> None:
    """Reject missing, repeated or indirect composition dependencies."""
    required = rule.get("required_components")
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(x, str) for x in required)
    ):
        raise ValueError("Rule required_components must be a nonempty list of component names.")
    if len(required) != len(set(required)) or not set(required) <= branches:
        raise ValueError("Rule required_components must be unique known direct components.")


def validate_model_set_composition(
    config: Any, source: str, components: Any, record_key_schema: Any
) -> dict[str, Any]:
    """Validate saved callbacks or declarative arithmetic and explicit rule eligibility."""
    config = _composition_config(config)
    if not config["outputs"]:
        model_set_schema(components, record_key_schema, config)
        return config
    module = None
    branches = {component.branch for component in components}
    names: set[str] = set()
    for rule in config["outputs"]:
        if type(rule) is not dict:
            raise ValueError("Model-set composition rules must be objects.")
        _dependencies(rule, branches)
        if "operation" in rule:
            validate_operation(rule, components)
        else:
            if module is None:
                module = load_project_module(source)
            _validate_rule(
                {k: v for k, v in rule.items() if k != "required_components"},
                output=True,
                module=module,
            )
        name = rule["name"].casefold()
        if name in names:
            raise ValueError("Composition rule names must be unique.")
        names.add(name)
    model_set_schema(components, record_key_schema, config)
    return config


def _composition_config(config: Any) -> dict:
    """Normalize absence to independent outputs and reject implicit extra policies."""
    config = _json_copy({"outputs": []} if config is None else config)
    if (
        type(config) is not dict
        or set(config) != {"outputs"}
        or type(config["outputs"]) is not list
    ):
        raise ValueError("Model-set composition requires exactly an outputs list.")
    return config


def _validate_keys(frame: pd.DataFrame, keys: list[str]) -> None:
    """Require an unambiguous complete record identity before joining any result."""
    if not keys or len(keys) != len(set(keys)) or not set(keys) <= set(frame.columns):
        raise ValueError("Model-set record keys must be distinct existing columns.")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Model-set record keys must be non-null and unique.")


def assemble_component_outputs(
    current: pd.DataFrame, component: pd.DataFrame, record_keys: list[str]
) -> pd.DataFrame:
    """Join by validated keys, accepting reorder while rejecting changed key coverage."""
    _validate_keys(current, record_keys)
    _validate_keys(component, record_keys)
    if len(current) != len(component):
        raise ValueError("Component record key coverage differs from the requested rows.")
    overlapping = (set(current) & set(component)) - set(record_keys)
    if overlapping:
        raise ValueError("Component outputs cannot overwrite existing output columns.")
    membership = current[record_keys].merge(
        component[record_keys], on=record_keys, how="outer", indicator=True, validate="one_to_one"
    )
    if not membership["_merge"].eq("both").all():
        raise ValueError("Component record key coverage differs from the requested rows.")
    return current.merge(component, on=record_keys, how="left", sort=False, validate="one_to_one")


def _rule_reasons(predictions: pd.DataFrame, rule: dict) -> pd.Series:
    """Record the first unavailable required component without imposing global eligibility."""
    reasons = pd.Series(pd.NA, index=predictions.index, dtype="string")
    for branch in rule["required_components"]:
        excluded = predictions[f"{branch}__scoring_status"].eq("excluded") & reasons.isna()
        detail = predictions[f"{branch}__exclusion_reason"].astype("string")
        reasons.loc[excluded] = branch + ": " + detail.loc[excluded]
    return reasons


def compose_model_set_outputs(
    raw: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict,
    source: str,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> pd.DataFrame:
    """Run each saved rule only for rows where its direct dependencies succeeded."""
    result = predictions.copy(deep=True)
    if not config["outputs"]:
        return result
    for rule in config["outputs"]:
        reasons = _rule_reasons(predictions, rule)
        eligible = reasons.isna()
        for column in rule["columns"]:
            result[column["name"]] = pd.Series(
                pd.NA, index=raw.index, dtype=_NULLABLE[column["dtype"]]
            )
        if eligible.any():
            selected = raw.loc[eligible].reset_index(drop=True)
            values = _composition_values(selected, predictions.loc[eligible], rule, source)
            _check_rows(values, selected, pd.DataFrame)
            _bounded(values, max_rows, max_bytes)
            if list(values.columns) != [col["name"] for col in rule["columns"]]:
                raise ValueError(
                    "Composition callback columns must exactly match declared columns."
                )
            for column in rule["columns"]:
                typed = _typed_column(values[column["name"]], column["dtype"])
                if typed.isna().any():
                    raise ValueError("Composition outputs cannot be missing for eligible rows.")
                result.loc[eligible, column["name"]] = typed.to_numpy()
        result[f"{rule['name']}__scoring_status"] = pd.Series(
            "excluded", index=raw.index, dtype="string"
        ).mask(eligible, "predicted")
        result[f"{rule['name']}__exclusion_reason"] = reasons
        _bounded(result, max_rows, max_bytes)
    return result


def _composition_values(
    selected: pd.DataFrame, predictions: pd.DataFrame, rule: dict, source: str
) -> pd.DataFrame:
    """Keep declarative arithmetic independent of saved callback source loading."""
    predictions = predictions.reset_index(drop=True).copy(deep=True)
    if "operation" in rule:
        return apply_operation(predictions, rule)
    module = load_project_module(source)
    return _resolve(module, rule["function"])(
        selected.copy(deep=True), predictions, deepcopy(rule["params"])
    )


def _bounded(frame: pd.DataFrame | pl.DataFrame, max_rows: int, max_bytes: int) -> None:
    """Reject oversized native inputs and materialized results at every boundary."""
    if type(max_rows) is not int or type(max_bytes) is not int or min(max_rows, max_bytes) <= 0:
        raise ValueError("Model-set max_rows and max_bytes must be positive integers.")
    size = (
        frame.estimated_size()
        if isinstance(frame, pl.DataFrame)
        else int(frame.memory_usage(index=True, deep=True).sum())
    )
    if len(frame) > max_rows or size > max_bytes:
        raise ValueError("Model-set frame exceeds max_rows or max_bytes.")


def _pandas_input(frame: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    """Preserve nullable integer widths and values while assembling keyed outputs."""
    if isinstance(frame, pd.DataFrame):
        return frame.copy(deep=True)
    raw = frame.to_pandas()
    for column, dtype in frame.schema.items():
        if dtype.is_integer() and frame[column].null_count():
            raw[column] = pd.Series(frame[column].to_list(), dtype=str(dtype))
        elif dtype == pl.Date:
            raw[column] = frame[column].to_pandas(use_pyarrow_extension_array=True)
    return raw


def _raw_input(
    frame: Any, artifact: ModelSetArtifact, max_rows: int, max_bytes: int
) -> pd.DataFrame:
    """Bound the source before conversion and validate supported exact record keys."""
    if not isinstance(frame, (pd.DataFrame, pl.DataFrame)):
        raise TypeError("Model-set scoring requires a pandas or Polars DataFrame.")
    _bounded(frame, max_rows, max_bytes)
    raw = _pandas_input(frame)
    raw = raw.reset_index(drop=True)
    _bounded(raw, max_rows, max_bytes)
    if not all(isinstance(name, str) for name in raw.columns) or len(
        {name.casefold() for name in raw.columns}
    ) != len(raw.columns):
        raise ValueError("Model-set input columns must have unique string names.")
    keys = list(artifact.manifest.record_key_columns)
    _validate_keys(raw, keys)
    for column in artifact.manifest.record_key_schema:
        if column.dtype not in {"string", "int64", "bool"}:
            raise ValueError("Unsupported model-set record key dtype.")
        raw[column.name] = _typed_column(raw[column.name], column.dtype)
    return raw[[column.name for column in artifact.manifest.input_schema]]


def _component_outcomes(result: pd.DataFrame, component: Any) -> pd.DataFrame:
    """Validate complete declared predictions and explicit per-component exclusions."""
    if list(result.columns) != [col.name for col in component.output_schema]:
        raise ValueError("Component output columns disagree with their recorded schema.")
    result = result.copy(deep=True)
    if "scoring_status" not in result:
        result["scoring_status"] = pd.Series("predicted", index=result.index, dtype="string")
        result["exclusion_reason"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if not result["scoring_status"].isin(["predicted", "excluded"]).all():
        raise ValueError("Component scoring_status must describe every requested row.")
    excluded = result["scoring_status"].eq("excluded")
    reasons = result["exclusion_reason"]
    if reasons.loc[excluded].isna().any() or reasons.loc[~excluded].notna().any():
        raise ValueError("Component exclusion reasons disagree with row outcomes.")
    for column in component.output_schema:
        result[column.name] = _typed_column(result[column.name], column.dtype)
        if (
            column.name not in {"scoring_status", "exclusion_reason"}
            and result.loc[~excluded, column.name].isna().any()
        ):
            raise ValueError("Component outputs cannot be missing for predicted rows.")
    return result


def _score_component(
    raw: pd.DataFrame,
    artifact: ModelSetArtifact,
    component: Any,
    state: Any,
    bootstrap: bool,
    max_rows: int,
    max_bytes: int,
) -> tuple[pd.DataFrame, Any]:
    """Release one loaded component before advancing to the next model in the set."""
    local = load_local_pipeline(artifact.directory / "components" / component.branch)
    selected = raw[[column.name for column in component.input_schema]].copy(deep=True)
    _bounded(selected, max_rows, max_bytes)
    with local_history_session(local, state, bootstrap=bootstrap) as session:
        result = _predict_component(selected, local, component)
        _check_rows(result, selected, pd.DataFrame)
        result = _component_outcomes(result, component)
    result = result.rename(columns=lambda name: f"{component.branch}__{name}")
    keyed = pd.concat([raw[list(artifact.manifest.record_key_columns)], result], axis=1)
    _bounded(keyed, max_rows, max_bytes)
    return keyed, session.state if session is not None else None


def _predict_component(selected: pd.DataFrame, local: Any, component: Any) -> pd.DataFrame:
    """Validate empty inputs and preserve history without invoking an estimator."""
    if not selected.empty:
        return score_local_pipeline(selected, local)
    validate_local_input(selected, local)
    _preserve_history(local)
    return pd.DataFrame(
        {
            column.name: pd.Series(dtype=_NULLABLE[column.dtype])
            for column in component.output_schema
        }
    )


def _history_previous(artifact: ModelSetArtifact, state: Any, bootstrap: bool) -> dict:
    """Bind continuation to the complete immutable set before running components."""
    if state is None:
        return {}
    _validate_history_envelope(state)
    if (
        bootstrap
        or type(state) is not dict
        or set(state) != {"version", "model_set_sha256", "components"}
    ):
        raise ValueError("Invalid model-set temporal history or conflicting bootstrap.")
    if (
        state["version"] != 1
        or state["model_set_sha256"] != artifact.manifest.set_sha256
        or type(state["components"]) is not dict
    ):
        raise ValueError("Temporal history belongs to a different model set or format.")
    branches = {component.branch for component in artifact.manifest.components}
    if not set(state["components"]) <= branches:
        raise ValueError("Temporal history contains unknown components.")
    return deepcopy(state["components"])


def _validate_history_envelope(state: Any) -> None:
    """Bound incoming continuation and reject entries that would silently reset a model."""
    if type(state) is not dict or len(json.dumps(state, allow_nan=False).encode()) > 60 * 1024:
        raise ValueError("Invalid model-set history or history exceeds the 60 KiB receipt budget.")
    components = state.get("components")
    if type(components) is not dict or any(type(item) is not dict for item in components.values()):
        raise ValueError("Model-set temporal history requires complete component state objects.")


def score_model_set(
    frame: pd.DataFrame | pl.DataFrame,
    artifact: ModelSetArtifact,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    history_state: dict | None = None,
    bootstrap_history: bool = False,
) -> ModelSetPrediction:
    """Compute a complete bounded result and history proposal without external writes."""
    raw = _raw_input(frame, artifact, max_rows, max_bytes)
    _revalidate_package(artifact)
    previous = _history_previous(artifact, history_state, bootstrap_history)
    result = raw[list(artifact.manifest.record_key_columns)].copy(deep=True)
    proposed: dict[str, Any] = {}
    for component in artifact.manifest.components:
        keyed, state = _score_component(
            raw,
            artifact,
            component,
            previous.get(component.branch),
            bootstrap_history,
            max_rows,
            max_bytes,
        )
        if state is not None:
            if history_state is not None and component.branch not in previous:
                raise ValueError("Temporal history is missing a component's continuation.")
            proposed[component.branch] = state
        result = assemble_component_outputs(
            result, keyed, list(artifact.manifest.record_key_columns)
        )
        _bounded(result, max_rows, max_bytes)
    source = (
        (artifact.directory / "composition.py").read_text(encoding="utf-8")
        if artifact.manifest.composition_config["outputs"]
        else ""
    )
    config = validate_model_set_composition(
        artifact.manifest.composition_config,
        source,
        artifact.manifest.components,
        artifact.manifest.record_key_schema,
    )
    result = compose_model_set_outputs(
        raw, result, config, source, max_rows=max_rows, max_bytes=max_bytes
    )
    _bounded(result, max_rows, max_bytes)
    history = (
        {"version": 1, "model_set_sha256": artifact.manifest.set_sha256, "components": proposed}
        if proposed
        else {}
    )
    if len(json.dumps(history, allow_nan=False).encode()) > 60 * 1024:
        raise ValueError("Model-set temporal history exceeds the 60 KiB receipt budget.")
    if isinstance(frame, pd.DataFrame):
        result.index = frame.index
    return ModelSetPrediction(result, history)


def _revalidate_package(artifact: ModelSetArtifact) -> None:
    """Reject replaced package contents even if the caller retains an older artifact."""
    from .model_set import verify_model_set_files  # noqa: PLC0415

    verify_model_set_files(artifact)


def predict_model_set(
    frame: pd.DataFrame | pl.DataFrame,
    artifact: ModelSetArtifact,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> pd.DataFrame:
    """Return independent and composed predictions from one coherent saved model set."""
    return score_model_set(frame, artifact, max_rows=max_rows, max_bytes=max_bytes).frame
