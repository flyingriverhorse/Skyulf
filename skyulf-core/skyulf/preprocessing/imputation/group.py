"""Group imputer node: fill gaps with a statistic learned per group on training rows."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..._validation import raise_invalid_choice
from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...core.portable_state import _normalize
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, is_decimal_series, resolve_columns
from .._artifacts import GroupImputerArtifact
from .._fitted_validation import _columns, _fields, _scalar, fitted_columns
from .._helpers import (
    auto_detect_numeric_columns,
    promote_configured_columns_to_float64,
    to_pandas,
)
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _imputation_config

_NUMERIC_STRATEGIES = {"mean", "median"}
_STRATEGIES = (*sorted(_NUMERIC_STRATEGIES), "most_frequent")


def _native(X: Any) -> Any:
    """Unwrap Skyulf frame wrappers to the underlying pandas or polars frame."""
    return X.to_native() if hasattr(X, "to_native") else X


def _python_value(value: Any) -> Any:
    """Turn NumPy scalars into plain Python values so the artifact stays JSON-friendly."""
    return value.item() if isinstance(value, np.generic) else value


def _statistic(values: pd.Series, strategy: str) -> Any:
    """Return the strategy's statistic of the observed values, or None when none exist."""
    values = values.dropna()
    if values.empty:
        return None
    if strategy == "mean":
        return float(values.mean())
    if strategy == "median":
        return float(values.median())
    # Series.mode() is sorted, so a tie picks the smallest value like SimpleImputer.
    return _python_value(values.mode().iloc[0])


def _strategy(config: dict[str, Any]) -> str:
    """Normalize and validate the configured strategy."""
    strategy = config.get("strategy", "mean")
    strategy = "most_frequent" if strategy == "mode" else strategy
    if strategy not in _STRATEGIES:
        raise_invalid_choice(strategy, _STRATEGIES, "strategy")
    return strategy


def _require_group_column(columns: Any, group_by: str) -> None:
    """Fail clearly when the column that selects each row's group is absent."""
    if group_by not in columns:
        raise ValueError(
            f"GroupImputer group_by column '{group_by}' is missing from the data; "
            "it is needed to pick each row's fill value."
        )


def _fill_columns(X: Any, config: dict[str, Any], group_by: str, strategy: str) -> list[str]:
    """Resolve the columns to fill; mean/median need numeric ones and the group key is excluded."""
    if group_by in (config.get("columns") or []):
        raise ValueError(f"GroupImputer cannot fill its group_by column '{group_by}' by itself.")
    numeric = strategy in _NUMERIC_STRATEGIES
    detect = detect_numeric_columns if numeric else (lambda frame: list(frame.columns))
    columns = [column for column in resolve_columns(X, config, detect) if column != group_by]
    if numeric:
        allowed = set(auto_detect_numeric_columns(X))
        invalid = [column for column in columns if column not in allowed]
        if invalid:
            raise ValueError(
                f"GroupImputer strategy '{strategy}' requires numeric columns; "
                f"non-numeric columns selected: {invalid}."
            )
    return columns


def _training_frame(X: Any, columns: list[str], group_by: str) -> pd.DataFrame:
    """Convert only the filled columns and the group key to pandas."""
    selected = [*columns, group_by]
    if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
        return X.select(selected).to_pandas()
    return to_pandas(X)[selected]


def _group_values(frame: pd.DataFrame, column: str, group_by: str, strategy: str) -> list[list]:
    """Return [group key, value] pairs for groups that have at least one observed value."""
    values = frame[column]
    if strategy in _NUMERIC_STRATEGIES:
        values = pd.to_numeric(values)
    pairs = []
    for key, part in values.groupby(frame[group_by], dropna=True, sort=True, observed=True):
        value = _statistic(part, strategy)
        if value is not None:
            pairs.append([_python_value(key), value])
    return pairs


def _extend_categories(series: pd.Series, fills: pd.Series) -> pd.Series:
    """Allow learned fills absent from a categorical inference batch's vocabulary."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        new = [value for value in fills.dropna().unique() if value not in series.cat.categories]
        if new:
            return series.cat.add_categories(new)
    return series


class GroupImputerApplier(BaseApplier):
    """Fill gaps with the training value of each row's group, then with the global value.

    Rows whose group was unseen in training, whose group key is null, or whose
    group had no observed value fall back to the column's global training
    value. The scoring batch's own values are never used to compute fills.
    """

    @staticmethod
    def validate_fitted_state(raw: dict) -> dict:
        """Inspect this node's supported saved state without fitting or applying data."""
        return _group_state(raw)

    @staticmethod
    def resolve_fitted_config(raw: dict, state: dict) -> dict:
        """Bind inference configuration to this node's inspected fitted artifact."""
        return _imputation_config("GroupImputer", fitted_columns(raw, state), state)

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Fill the learned columns on the active engine; ``y`` passes through."""
        if not params.get("columns"):
            return X if y is None else (X, y)
        _require_group_column(_native(X).columns, params["group_by"])
        input_data = (X, y) if y is not None else X
        return apply_dual_engine(
            input_data, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        out = X.copy()
        keys = out[params["group_by"]].astype(object)
        numeric = params["strategy"] in _NUMERIC_STRATEGIES
        for column in params["columns"]:
            if column not in out.columns:
                continue
            series = out[column]
            group_fill = keys.map(dict(map(tuple, params["group_values"][column])))
            fallback = params["fill_values"].get(column)
            if fallback is not None:
                group_fill = group_fill.fillna(fallback)
            if numeric:
                if not pd.api.types.is_float_dtype(series) or is_decimal_series(series):
                    series = pd.to_numeric(series).astype("float64")
                group_fill = pd.to_numeric(group_fill).astype("float64")
            series = _extend_categories(series, group_fill)
            out[column] = series.where(series.notna(), group_fill.to_numpy())
        return out, y

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        numeric = params["strategy"] in _NUMERIC_STRATEGIES
        keys = pl.col(params["group_by"])
        key_dtype = X.schema[params["group_by"]]
        if isinstance(key_dtype, (pl.Categorical, pl.Enum)):
            keys = keys.cast(pl.Utf8)
            key_dtype = pl.String
        exprs = [
            _polars_fill(X.schema[column], column, keys, key_dtype, params, numeric)
            for column in params["columns"]
            if column in X.columns
        ]
        return X.with_columns(exprs), y


def _polars_group_mapping(pairs: list[list], key_dtype: Any) -> tuple[Any, list[Any]]:
    """Keep only exact key conversions before a native lookup can coerce identities.

    Compare learned scalars in Python so numeric equality (including booleans)
    survives, while text conversion, truncation and rounding cannot invent a
    match. Only learned groups cross Python; inference rows stay in Polars.
    """
    old, new = zip(*pairs, strict=True)
    if not (
        key_dtype.is_integer()
        or key_dtype.is_float()
        or key_dtype in (pl.String, pl.Boolean, pl.Null)
    ):
        return list(old), list(new)
    converted = pl.Series("group_keys", old, dtype=key_dtype, strict=False)
    keep = [
        actual is not None and original == actual
        for original, actual in zip(old, converted.to_list(), strict=True)
    ]
    return converted.filter(pl.Series(keep)), [
        value for value, match in zip(new, keep, strict=True) if match
    ]


def _polars_fill(
    dtype: Any,
    column: str,
    keys: Any,
    key_dtype: Any,
    params: dict[str, Any],
    numeric: bool,
) -> Any:
    """Build the expression that fills one column from its group value, then the global one."""
    out_dtype = pl.Float64 if numeric else dtype
    values = pl.col(column).cast(out_dtype)
    if out_dtype.is_float():
        values = values.fill_nan(None)
    pairs = params["group_values"][column]
    if pairs:
        old, new = _polars_group_mapping(pairs, key_dtype)
        values = values.fill_null(
            keys.replace_strict(old, new, default=None, return_dtype=out_dtype)
        )
    fallback = params["fill_values"].get(column)
    if fallback is not None:
        values = values.fill_null(pl.lit(fallback, dtype=out_dtype))
    return values.alias(column)


@NodeRegistry.register(
    "GroupImputer",
    GroupImputerApplier,
    execution_capabilities=tuple(
        ExecutionCapability(
            engine,
            "apply",
            execution_kind,
            "preserve",
            "row",
            config_match=(("strategy", strategy),),
        )
        for engine, execution_kind in (("pandas", "python_batch"), ("polars", "local"))
        for strategy in ("mean", "most_frequent")
    ),
)
@node_meta(
    id="GroupImputer",
    name="Group Imputer",
    category="Preprocessing",
    description=(
        "Fills missing values with the mean, median or most frequent value of each row's "
        "group (for example per industry), falling back to the overall value."
    ),
    params={"group_by": "", "strategy": "mean", "columns": []},
    learns_from_data=True,
)
class GroupImputerCalculator(BaseCalculator):
    """Learn per-group and global fill values from training rows.

    ``group_by`` names the column that defines the groups. ``strategy`` is
    ``mean``, ``median`` or ``most_frequent`` (``mode`` is an alias). Leave
    ``columns`` empty to fill every numeric column (mean/median) or every
    column (most_frequent), excluding the group key and the target.
    """

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Promote filled columns to ``float64`` for mean/median; keep the schema otherwise."""
        if config.get("strategy", "mean") not in _NUMERIC_STRATEGIES:
            return input_schema
        return promote_configured_columns_to_float64(input_schema, config)

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> GroupImputerArtifact:  # pylint: disable=arguments-differ
        """Compute ``[group, value]`` pairs and a global fallback for every filled column."""
        group_by = config.get("group_by")
        if not group_by:
            raise ValueError(
                "GroupImputer needs group_by: the column whose groups get their own fill value."
            )
        strategy = _strategy(config)
        X = _native(X)
        _require_group_column(X.columns, group_by)
        columns = _fill_columns(X, config, group_by, strategy)
        frame = _training_frame(X, columns, group_by)
        fill_values = {}
        for column in columns:
            values = frame[column]
            if strategy in _NUMERIC_STRATEGIES:
                values = pd.to_numeric(values)
            fill_values[column] = _statistic(values, strategy)
        return {
            "type": "group_imputer",
            "group_by": group_by,
            "strategy": strategy,
            "columns": columns,
            "group_values": {c: _group_values(frame, c, group_by, strategy) for c in columns},
            "fill_values": fill_values,
        }


def _group_pairs(pairs: Any, numeric: bool) -> None:
    """Require unique scalar group keys and finite per-group replacements."""
    if type(pairs) is not list:
        raise ValueError("Group replacements must be a list.")
    keys = set()
    for pair in pairs:
        if type(pair) is not list or len(pair) != 2:
            raise ValueError("Group replacements must contain key/value pairs.")
        key, value = pair
        _scalar(key)
        _scalar(value)
        if key is None or key in keys:
            raise ValueError("Group keys must be non-null and unique.")
        keys.add(key)
        if numeric and type(value) not in (int, float):
            raise ValueError("Group means must be numeric.")


def _group_state(raw: dict) -> dict:
    """Validate learned group maps and the global fallback without recomputing either."""
    state = _normalize(raw)
    _fields(state, {"type", "group_by", "strategy", "columns", "group_values", "fill_values"})
    columns = _columns(state["columns"])
    if state["type"] != "group_imputer" or state["strategy"] not in ("mean", "most_frequent"):
        raise ValueError("Unsupported group imputer state.")
    if type(state["group_by"]) is not str or state["group_by"] in columns:
        raise ValueError("Invalid group key.")
    _fields(state["group_values"], set(columns))
    _fields(state["fill_values"], set(columns))
    for column in columns:
        fallback = state["fill_values"][column]
        _scalar(fallback)
        if state["strategy"] == "mean" and type(fallback) not in (int, float, type(None)):
            raise ValueError("Global group means must be numeric.")
        _group_pairs(state["group_values"][column], state["strategy"] == "mean")
    return state
