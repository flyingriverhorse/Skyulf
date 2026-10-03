"""Plain-function preprocessing steps for project code.

Users write one ordinary pandas function instead of a Calculator/Applier pair.
Functions are saved by reference (``module:qualname``), so they must be
top-level ``def`` definitions in an importable or loaded project module.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ._helpers import select_rows_by_position
from .base import BaseApplier, BaseCalculator, apply_method, fit_method

__all__ = ["column_step", "filter_step", "fitted_step"]

COLUMN_STEP = "ColumnFunction"
FITTED_STEP = "FittedFunction"
FILTER_STEP = "RowFilterFunction"
_CODE_ONLY_TAG = "code_only"


def function_ref(fn: Callable[..., Any]) -> str:
    """Return a saved reference for a top-level function, rejecting lambdas and closures."""
    if not callable(fn) or isinstance(fn, type):
        raise ValueError("Step function must be a plain function defined with def.")
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", "")
    if not module or "<" in qualname:
        raise ValueError(
            f"Step function {qualname or fn!r} must be a top-level def in a module file; "
            "lambdas and nested functions cannot be saved with the model."
        )
    ref = f"{module}:{qualname}"
    if resolve_function(ref) is not fn:
        raise ValueError(f"Step function {ref} cannot be found again by name.")
    return ref


def resolve_function(ref: str) -> Callable[..., Any]:
    """Find a saved function in an already loaded project module or an importable module."""
    module_name, _, qualname = ref.partition(":")
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)
    target: Any = module
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def _outputs(output: str | list[str]) -> list[str]:
    """Normalize output column names and reject empty or duplicate names."""
    names = [output] if isinstance(output, str) else list(output)
    if not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("output must be a column name or a nonempty list of column names.")
    if len(set(names)) != len(names):
        raise ValueError("output column names must be unique.")
    return names


def _user_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Require JSON-compatible user parameters so they can be saved and digested."""
    params = {} if params is None else dict(params)
    _require_json(params, "params")
    return params


def _require_json(value: Any, label: str) -> None:
    """Reject values that cannot be saved as plain JSON evidence."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must contain only JSON values (str, number, bool, list, dict)."
        ) from exc


def _learned_state(state: Any) -> dict[str, Any]:
    """Return learned state exactly as it will look after saving and loading."""
    if not isinstance(state, dict):
        raise ValueError("learn function must return a dict of learned values.")
    _require_string_keys(state)
    try:
        text = json.dumps(state, allow_nan=False, default=_numpy_value)
    except ValueError as exc:
        raise ValueError(
            "Learned state contains NaN or infinity; replace them, e.g. with None or a default."
        ) from exc
    except TypeError as exc:
        raise ValueError(
            f"Learned state must contain only JSON values (str, number, bool, list, dict): {exc}"
        ) from exc
    return json.loads(text)


def _require_string_keys(value: Any) -> None:
    """Reject dict keys that saving would silently turn into strings."""
    if isinstance(value, dict):
        bad = [key for key in value if not isinstance(key, str)]
        if bad:
            raise ValueError(
                f"Learned state keys must be strings, got {bad[:3]!r}; group on "
                "df[col].astype(str) when learning and map df[col].astype(str) when applying."
            )
        for item in value.values():
            _require_string_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_string_keys(item)


def _numpy_value(value: Any) -> Any:
    """Convert NumPy scalars and arrays so beginners need not call int()/float()."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _pandas_view(X: Any) -> tuple[Any, pd.DataFrame]:
    """Return the caller's native frame and a private pandas copy for user code."""
    native = X.to_native() if hasattr(X, "to_native") else X
    if isinstance(native, pl.DataFrame):
        return native, native.to_pandas()
    return native, native.copy()


def _without_target(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Hide the pipeline target column so user code cannot read it by accident."""
    target = config.get("target_column")
    return frame.drop(columns=[target]) if target in frame.columns else frame


def _pandas_target(y: Any) -> Any:
    """Expose the target to learn functions as pandas, or None when absent."""
    if y is None:
        return None
    native = y.to_native() if hasattr(y, "to_native") else y
    return native.to_pandas() if isinstance(native, pl.Series) else native


def _call(ref: str, *args: Any, params: dict[str, Any]) -> Any:
    """Call a saved function, passing user params only when configured."""
    fn = resolve_function(ref)
    try:
        return fn(*args, params) if params else fn(*args)
    except Exception as exc:
        raise ValueError(f"Project function {ref} failed: {type(exc).__name__}: {exc}") from exc


def _as_frame(values: Any, frame: pd.DataFrame, outputs: list[str]) -> pd.DataFrame:
    """Validate the function result keeps every row in order and name its columns."""
    if isinstance(values, (np.ndarray, list)):
        # Plain arrays such as np.where(...) carry no index; they follow df row order.
        values = pd.DataFrame(np.asarray(values), index=frame.index)
    if isinstance(values, pd.Series):
        values = values.to_frame()
    if not isinstance(values, pd.DataFrame):
        raise ValueError("Step function must return a pandas Series, DataFrame or NumPy array.")
    if len(values) != len(frame):
        raise ValueError(
            f"Step function returned {len(values)} rows for {len(frame)} input rows; "
            "column steps must keep every row in the same order."
        )
    if not values.index.equals(frame.index):
        raise ValueError(
            "Step function returned rows with a different index or order than df "
            "(e.g. after sort_values or reset_index); return values aligned to df.index."
        )
    if values.shape[1] != len(outputs):
        raise ValueError(
            f"Step function returned {values.shape[1]} columns; output names {outputs}."
        )
    values = values.copy()
    values.columns = outputs
    return values


def _assign(X: Any, params: dict[str, Any], computed: Callable[[pd.DataFrame], Any]) -> Any:
    """Add computed columns to the caller's frame, leaving every other column untouched."""
    native, frame = _pandas_view(X)
    outputs = params["output"]
    if params.get("target_column") in outputs:
        raise ValueError(f"Output {outputs} must not overwrite the target column.")
    if not params["replace"]:
        clash = [name for name in outputs if name in frame.columns]
        if clash:
            raise ValueError(f"Columns {clash} already exist; pass replace=True to overwrite them.")
    values = _as_frame(computed(_without_target(frame, params)), frame, outputs)
    if isinstance(native, pl.DataFrame):
        return native.with_columns(pl.from_pandas(values).get_columns())
    return native.assign(**{name: values[name] for name in outputs})


@node_meta(
    id=COLUMN_STEP,
    name="Column Function",
    category="Project Code",
    description="Add columns computed by a project function; learns nothing from data.",
    params={"function": "", "output": [], "replace": False, "params": {}},
    tags=[_CODE_ONLY_TAG],
    learns_from_data=False,
)
class ColumnFunctionCalculator(BaseCalculator):
    """Freeze the configured function reference; nothing is learned."""

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> dict[str, Any]:
        """Return the validated configuration as the saved artifact."""
        resolve_function(config["function"])
        return dict(config)


class ColumnFunctionApplier(BaseApplier):
    """Apply the saved function to every split and inference batch identically."""

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:
        """Add or replace output columns without changing rows."""
        ref, user = params["function"], params["params"]
        return _assign(X, params, lambda frame: _call(ref, frame, params=user))


@node_meta(
    id=FITTED_STEP,
    name="Fitted Function",
    category="Project Code",
    description="Learn state with one project function and apply it with another.",
    params={"learn": "", "apply": "", "output": [], "replace": False, "params": {}},
    tags=[_CODE_ONLY_TAG],
    learns_from_data=True,
)
class FittedFunctionCalculator(BaseCalculator):
    """Run the learn function on the training rows of the current fold only."""

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> dict[str, Any]:
        """Save the JSON state returned by the learn function beside the configuration."""
        _, frame = _pandas_view(X)
        target = config.get("target_column")
        if y is None and target in frame.columns:
            # Pipelines keep the target inside X and name it in the config.
            y = frame[target]
        features = _without_target(frame, config)
        state = _call(config["learn"], features, _pandas_target(y), params=config["params"])
        return {**config, "state": _learned_state(state)}


class FittedFunctionApplier(BaseApplier):
    """Apply saved state; the apply function never sees the target."""

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:
        """Add or replace output columns using only the saved state."""
        ref, state, user = params["apply"], params["state"], params["params"]
        return _assign(X, params, lambda frame: _call(ref, frame, state, params=user))


@node_meta(
    id=FILTER_STEP,
    name="Row Filter Function",
    category="Project Code",
    description="Keep rows where a project function returns True; values are unchanged.",
    params={"function": "", "columns": [], "params": {}},
    tags=[_CODE_ONLY_TAG],
    learns_from_data=False,
)
class RowFilterFunctionCalculator(BaseCalculator):
    """Freeze the filter rule; nothing is learned."""

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> dict[str, Any]:
        """Return the validated configuration as the saved artifact."""
        resolve_function(config["function"])
        return dict(config)


class RowFilterFunctionApplier(BaseApplier):
    """Keep selected rows in their original order without editing any value."""

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:
        """Filter rows with a boolean mask from the saved function."""
        native, frame = _pandas_view(X)
        missing = [c for c in params["columns"] if c not in frame.columns]
        if missing:
            raise ValueError(
                f"Filter columns {missing} are missing; available: {list(frame.columns)}."
            )
        mask = _call(params["function"], frame, params=params["params"])
        keep = _mask(mask, frame)
        kept = (
            native.filter(pl.Series(keep)) if isinstance(native, pl.DataFrame) else native.loc[keep]
        )
        if y is None:
            return kept
        return kept, _filter_target(y, keep)


def _mask(mask: Any, frame: pd.DataFrame) -> list[bool]:
    """Require one non-null boolean per input row."""
    if (
        not isinstance(mask, pd.Series)
        or len(mask) != len(frame)
        or not mask.index.equals(frame.index)
    ):
        raise ValueError("Filter function must return a boolean Series with one value per row.")
    if not pd.api.types.is_bool_dtype(mask):
        raise ValueError(
            "Filter function must return only True/False values, e.g. df['age'] >= 18."
        )
    if mask.isna().any():
        raise ValueError(
            "Filter mask has missing values; decide those rows explicitly, e.g. (mask).fillna(False)."
        )
    return mask.astype(bool).tolist()


def _filter_target(y: Any, keep: list[bool]) -> Any:
    """Keep the target aligned with retained rows."""
    return select_rows_by_position(y, np.flatnonzero(keep))


NodeRegistry.register(COLUMN_STEP, ColumnFunctionApplier)(ColumnFunctionCalculator)
NodeRegistry.register(FITTED_STEP, FittedFunctionApplier)(FittedFunctionCalculator)
NodeRegistry.register(FILTER_STEP, RowFilterFunctionApplier)(RowFilterFunctionCalculator)


def column_step(
    name: str,
    function: Callable[..., Any],
    *,
    output: str | list[str],
    replace: bool = False,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a step that adds columns from ``function(df)`` or ``function(df, params)``."""
    return {
        "name": name,
        "transformer": COLUMN_STEP,
        "params": {
            "function": function_ref(function),
            "output": _outputs(output),
            "replace": bool(replace),
            "params": _user_params(params),
        },
    }


def fitted_step(
    name: str,
    learn: Callable[..., Any],
    apply: Callable[..., Any],
    *,
    output: str | list[str],
    replace: bool = False,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a step whose ``learn(df, y)`` state is reused by ``apply(df, state)``."""
    return {
        "name": name,
        "transformer": FITTED_STEP,
        "params": {
            "learn": function_ref(learn),
            "apply": function_ref(apply),
            "output": _outputs(output),
            "replace": bool(replace),
            "params": _user_params(params),
        },
    }


def filter_step(
    name: str,
    function: Callable[..., Any],
    *,
    columns: list[str],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a pre-split row filter that keeps rows where ``function(df)`` is True."""
    if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns):
        raise ValueError("filter columns must be a nonempty list of unique column names.")
    return {
        "name": name,
        "transformer": FILTER_STEP,
        "params": {
            "function": function_ref(function),
            "columns": list(columns),
            "params": _user_params(params),
        },
        "pre_split": {
            "effect": "filter",
            "required_columns": list(columns),
            "learns_from_data": False,
        },
    }
