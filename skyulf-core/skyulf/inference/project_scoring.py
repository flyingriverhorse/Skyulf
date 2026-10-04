"""Versioned, trusted project eligibility and output rules for local scoring.

Callbacks receive defensive pandas frames with a fresh RangeIndex, irrespective
of the model engine. Eligibility returns nullable reason strings; the first
reason wins. Output callbacks return exactly their declared columns and rows.
These are trusted executable model contents, not a sandbox or a purity proof.
Training and evaluation filters are intentionally outside this contract.
"""

import importlib
import json
import re
from collections.abc import Callable
from copy import deepcopy
from numbers import Integral, Real
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from .project_code import load_project_module

_DTYPES: dict[str, Literal["Float64", "Int64", "string", "boolean"]] = {
    "float64": "Float64",
    "int64": "Int64",
    "string": "string",
    "bool": "boolean",
}
_RESERVED = {
    "prediction",
    "run_id",
    "model_name",
    "model_version",
    "scoring_status",
    "exclusion_reason",
}


def _json_copy(value: Any) -> Any:
    """Reject non-JSON and non-finite rule parameters before saving them."""
    try:
        encoded = json.dumps(value, allow_nan=False)
        result = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scoring configuration requires finite JSON values.") from exc
    if result != value:
        raise ValueError("Scoring configuration requires JSON objects and arrays.")
    return result


def _identifier(value: Any) -> bool:
    """Recognize simple public names outside Skyulf's internal namespace."""
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is not None
        and not value.casefold().startswith("__skyulf_")
    )


def _resolve(module: ModuleType, path: str) -> Callable[..., Any]:
    """Resolve a relative dotted function solely within saved project code."""
    if not isinstance(path, str) or not all(_identifier(part) for part in path.split(".")):
        raise ValueError("Scoring function must be a relative dotted project name.")
    parent, separator, name = path.rpartition(".")
    try:
        owner = importlib.import_module(f"{module.__name__}.{parent}") if separator else module
        function = getattr(owner, name if separator else path)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Scoring function {path!r} is absent from saved project source.") from exc
    origin = getattr(function, "__module__", "")
    if not callable(function) or not (
        origin == module.__name__ or origin.startswith(module.__name__ + ".")
    ):
        raise ValueError("Scoring callbacks must be defined in saved project source.")
    return function


def _validate_rule(rule: Any, *, output: bool, module: ModuleType) -> None:
    """Require named, versioned functions with JSON parameters and known fields."""
    required = {"name", "version", "function", "params"}
    if output:
        required.add("columns")
    if type(rule) is not dict or set(rule) != required:
        raise ValueError(
            "Scoring rule fields must match name, version, function, params and output columns."
        )
    if not _identifier(rule["name"]):
        raise ValueError("Scoring rule name must be a simple identifier.")
    if not isinstance(rule["version"], str) or not rule["version"].strip():
        raise ValueError("Scoring rule requires a nonempty immutable version string.")
    if type(rule["params"]) is not dict:
        raise ValueError("Scoring rule params must be a JSON object.")
    _resolve(module, rule["function"])
    if output:
        _validate_columns(rule["columns"])


def _reserved(name: str) -> bool:
    """Protect prediction and publication metadata using case-insensitive names."""
    return (
        name.casefold() in _RESERVED
        or re.fullmatch(r"probability_\d+", name.casefold()) is not None
    )


def _validate_columns(columns: Any) -> None:
    """Require explicit supported scalar output types and unreserved names."""
    if type(columns) is not list or not columns:
        raise ValueError("Output rules need a nonempty columns list.")
    for column in columns:
        if type(column) is not dict or set(column) != {"name", "dtype"}:
            raise ValueError("Output column requires exactly name and dtype.")
        if not _identifier(column["name"]) or _reserved(column["name"]):
            raise ValueError("Output column name is invalid or reserved.")
        if column["dtype"] not in _DTYPES:
            raise ValueError("Output column dtype must be float64, int64, string or bool.")


def scoring_output_columns(config: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return declared business output names and types in their saved order."""
    return tuple(
        (column["name"], column["dtype"])
        for rule in config["outputs"]
        for column in rule["columns"]
    )


def validate_scoring_config(config: Any, source: str) -> dict[str, Any]:
    """Validate and defensively copy the exact scoring contract persisted at fit."""
    config = _json_copy(config)
    if type(config) is not dict or set(config) not in (
        {"eligibility", "outputs"},
        {"eligibility", "outputs", "pre_split"},
    ):
        raise ValueError("Scoring configuration requires eligibility and outputs lists.")
    module = load_project_module(source)
    if "pre_split" in config:
        from ..integrations.databricks.scoring_pre_split import (  # noqa: PLC0415
            validate_pre_split_scoring,
        )

        validate_pre_split_scoring(config["pre_split"], module)
    names: set[str] = set()
    for kind in ("eligibility", "outputs"):
        if type(config[kind]) is not list:
            raise ValueError("Scoring eligibility and outputs must be lists.")
        for rule in config[kind]:
            _validate_rule(rule, output=kind == "outputs", module=module)
            if rule["name"].casefold() in names:
                raise ValueError("Scoring rule names must be unique.")
            names.add(rule["name"].casefold())
    columns = [name.casefold() for name, dtype in scoring_output_columns(config)]
    if len(set(columns)) != len(columns):
        raise ValueError("Output column names must be unique across rules.")
    return config


def _check_rows(value: Any, frame: pd.DataFrame, kind: type) -> None:
    """Fail on changed row counts, reordered indexes or non-tabular callbacks."""
    if not isinstance(value, kind) or not value.index.equals(frame.index):
        raise ValueError("Scoring callback rows and index must exactly match its input.")


def _is_null(value: Any) -> bool:
    """Recognize scalar nullable values without accepting arrays or containers."""
    return value is None or value is pd.NA or (isinstance(value, Real) and bool(np.isnan(value)))


def _eligibility(frame: pd.DataFrame, config: dict[str, Any], module: ModuleType) -> pd.Series:
    """Collect the first exclusion reason without changing callback inputs."""
    reasons = pd.Series(pd.NA, index=frame.index, dtype="string")
    for rule in config["eligibility"]:
        values = _resolve(module, rule["function"])(frame.copy(deep=True), deepcopy(rule["params"]))
        _check_rows(values, frame, pd.Series)
        if any(
            not _is_null(value) and (not isinstance(value, str) or not value.strip())
            for value in values
        ):
            raise ValueError("Eligibility reason must be a nonempty string or null.")
        reasons = reasons.fillna(values.astype("string"))
    return reasons


def _scoring_eligibility(
    frame: pd.DataFrame, config: dict[str, Any], module: ModuleType
) -> pd.Series:
    """Evaluate pre-split first, then custom checks only on its original surviving inputs."""
    if "pre_split" not in config:
        return _eligibility(frame, config, module)
    from ..integrations.databricks.scoring_pre_split import (  # noqa: PLC0415
        pre_split_exclusion_reasons,
    )

    reasons = pre_split_exclusion_reasons(frame, config["pre_split"])
    remaining = reasons.isna()
    if remaining.any() and config["eligibility"]:
        selected = frame.loc[remaining].reset_index(drop=True)
        custom = _eligibility(selected, config, module)
        reasons.loc[remaining] = custom.to_numpy()
    return reasons


def _valid_scalar(value: Any, dtype: str) -> bool:
    """Reject lossy or semantic coercions for declared business output values."""
    if _is_null(value):
        return True
    if dtype == "string":
        return isinstance(value, str)
    if dtype == "bool":
        return isinstance(value, (bool, np.bool_))
    if isinstance(value, (bool, np.bool_)):
        return False
    if dtype == "int64":
        return isinstance(value, Integral) and -(2**63) <= value < 2**63
    return isinstance(value, Real) and bool(np.isfinite(value))


def _typed_column(series: pd.Series, dtype: str) -> pd.Series:
    """Validate scalar meaning before normalizing to a nullable pandas dtype."""
    if dtype not in _DTYPES or not all(_valid_scalar(value, dtype) for value in series):
        raise ValueError(
            f"Scoring column {series.name!r} has values incompatible with dtype {dtype!r}."
        )
    return series.astype(_DTYPES[dtype])


def _outputs(
    frame: pd.DataFrame, predictions: pd.DataFrame, config: dict[str, Any], module: ModuleType
) -> pd.DataFrame:
    """Run ordered rules, allowing later rules to read earlier declared outputs."""
    result = predictions.copy(deep=True)
    for rule in config["outputs"]:
        values = _resolve(module, rule["function"])(
            frame.copy(deep=True), result.copy(deep=True), deepcopy(rule["params"])
        )
        _check_rows(values, frame, pd.DataFrame)
        expected = [column["name"] for column in rule["columns"]]
        if list(values.columns) != expected:
            raise ValueError(
                "Output callback columns must exactly match declared columns in order."
            )
        for column in rule["columns"]:
            result[column["name"]] = _typed_column(values[column["name"]], column["dtype"])
    return result


def _validate_input_columns(frame: pd.DataFrame, row_keys: list[str]) -> None:
    """Require unambiguous input columns and explicit existing row keys."""
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Scoring input columns must be unique.")
    if len(set(row_keys)) != len(row_keys) or any(key not in frame for key in row_keys):
        raise ValueError("Scoring row keys must be distinct input columns.")


def _result_schema(
    frame: pd.DataFrame,
    config: dict[str, Any],
    row_keys: list[str],
    prediction_dtypes: dict[str, str],
) -> dict[str, str]:
    """Validate collision-free output names before any model or callback executes."""
    _validate_input_columns(frame, row_keys)
    protected = {str(name).casefold() for name in frame.columns} | {
        name.casefold() for name in prediction_dtypes
    }
    columns = scoring_output_columns(config)
    if any(name.casefold() in protected for name, dtype in columns):
        raise ValueError("Output columns cannot overwrite raw input or prediction columns.")
    if set(row_keys) & (set(prediction_dtypes) | _RESERVED):
        raise ValueError("Scoring row keys cannot overwrite reserved output columns.")
    schema = {**prediction_dtypes, **dict(columns)}
    if any(dtype not in _DTYPES for dtype in schema.values()):
        raise ValueError("Scoring output schema contains an unsupported dtype.")
    return schema


def _predict_eligible(
    frame: pd.DataFrame,
    eligible: pd.Series,
    native: pd.DataFrame | pl.DataFrame,
    predict: Callable[..., pd.DataFrame],
    config: dict[str, Any],
    module: ModuleType,
    prediction_dtypes: dict[str, str],
) -> pd.DataFrame:
    """Validate model row alignment before applying declared business outputs."""
    selected = frame.loc[eligible].reset_index(drop=True)
    model_frame = (
        native.filter(pl.Series(eligible.to_numpy()))
        if isinstance(native, pl.DataFrame)
        else selected.copy(deep=True)
    )
    predictions = predict(model_frame)
    _check_rows(predictions, selected, pd.DataFrame)
    if list(predictions.columns) != list(prediction_dtypes):
        raise ValueError("Prediction columns must exactly match their recorded schema.")
    if predictions.isna().any().any():
        raise ValueError("Estimator outputs must not contain missing values.")
    predictions = predictions.copy(deep=True)
    for name, dtype in prediction_dtypes.items():
        predictions[name] = _typed_column(predictions[name], dtype)
    return _outputs(selected, predictions, config, module)


def run_project_scoring(
    frame: pd.DataFrame | pl.DataFrame,
    predict: Callable[..., pd.DataFrame],
    *,
    source: str,
    config: dict[str, Any],
    row_keys: list[str],
    prediction_dtypes: dict[str, str],
) -> pd.DataFrame:
    """Return one keyed result per input, skipping model work for excluded rows.

    ``predict`` receives eligible rows in the input engine with a fresh index.
    It must return pandas predictions with that exact RangeIndex and schema.
    Callbacks never filter training data. Rule outputs run only on eligible rows.
    Empty and all-excluded batches do not invoke the model or output callbacks.
    """
    if not isinstance(frame, (pd.DataFrame, pl.DataFrame)):
        raise TypeError("Scoring requires a pandas or Polars DataFrame.")
    config = validate_scoring_config(config, source)
    raw = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame.copy(deep=True)
    original_index = raw.index
    raw = raw.reset_index(drop=True)
    schema = _result_schema(raw, config, row_keys, prediction_dtypes)
    module = load_project_module(source)
    reasons = _scoring_eligibility(raw, config, module)
    eligible = reasons.isna()
    result = raw[row_keys].copy(deep=True)
    for name, dtype in schema.items():
        result[name] = pd.Series(pd.NA, index=raw.index, dtype=_DTYPES[dtype])
    if eligible.any():
        scored = _predict_eligible(raw, eligible, frame, predict, config, module, prediction_dtypes)
        scored.index = raw.index[eligible]
        result.loc[eligible, list(schema)] = scored
    result["scoring_status"] = pd.Series(
        np.where(eligible, "predicted", "excluded"), index=raw.index, dtype="string"
    )
    result["exclusion_reason"] = reasons
    result.index = original_index
    return result
