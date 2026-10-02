"""Group imputer node: fill gaps with a statistic learned per group on training rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..._validation import raise_invalid_choice
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, is_decimal_series, resolve_columns
from .._artifacts import GroupImputerArtifact
from .._helpers import (
    auto_detect_numeric_columns,
    promote_configured_columns_to_float64,
    to_pandas,
)
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine

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


class GroupImputerApplier(BaseApplier):
    """Fill gaps with the training value of each row's group, then with the global value.

    Rows whose group was unseen in training, whose group key is null, or whose
    group had no observed value fall back to the column's global training
    value. The scoring batch's own values are never used to compute fills.
    """

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
            if numeric:
                if not pd.api.types.is_float_dtype(series) or is_decimal_series(series):
                    series = pd.to_numeric(series).astype("float64")
                group_fill = pd.to_numeric(group_fill).astype("float64")
            filled = series.where(series.notna(), group_fill.to_numpy())
            fallback = params["fill_values"].get(column)
            out[column] = filled if fallback is None else filled.fillna(fallback)
        return out, y

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        numeric = params["strategy"] in _NUMERIC_STRATEGIES
        keys = pl.col(params["group_by"])
        if isinstance(X.schema[params["group_by"]], (pl.Categorical, pl.Enum)):
            keys = keys.cast(pl.Utf8)
        exprs = [
            _polars_fill(X.schema[column], column, keys, params, numeric)
            for column in params["columns"]
            if column in X.columns
        ]
        return X.with_columns(exprs), y


def _polars_fill(dtype: Any, column: str, keys: Any, params: dict[str, Any], numeric: bool) -> Any:
    """Build the expression that fills one column from its group value, then the global one."""
    out_dtype = pl.Float64 if numeric else dtype
    values = pl.col(column).cast(out_dtype)
    if out_dtype.is_float():
        values = values.fill_nan(None)
    pairs = params["group_values"][column]
    if pairs:
        old, new = zip(*pairs, strict=True)
        values = values.fill_null(
            keys.replace_strict(list(old), list(new), default=None, return_dtype=out_dtype)
        )
    fallback = params["fill_values"].get(column)
    if fallback is not None:
        values = values.fill_null(pl.lit(fallback, dtype=out_dtype))
    return values.alias(column)


@NodeRegistry.register("GroupImputer", GroupImputerApplier)
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
