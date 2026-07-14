"""Value replacement node (mapping / to_replace)."""

from typing import Any

import pandas as pd

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns
from .._artifacts import ValueReplacementArtifact
from .._helpers import resolve_valid_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine


def _is_mapping_like(obj: Any) -> bool:
    return isinstance(obj, (dict, pd.Series)) or hasattr(obj, "items")


def _polars_mapping_exprs(valid: list[str], mapping: dict[str, Any]) -> list[Any]:
    import polars as pl

    is_nested = any(isinstance(v, dict) for v in mapping.values())
    exprs: list[Any] = []
    for col in valid:
        if is_nested and col in mapping:
            exprs.append(pl.col(col).replace_strict(mapping[col], default=pl.col(col)).alias(col))
        elif not is_nested:
            exprs.append(pl.col(col).replace_strict(mapping, default=pl.col(col)).alias(col))
    return exprs


def _polars_to_replace_exprs(valid: list[str], to_replace: Any, value: Any) -> list[Any]:
    import polars as pl

    exprs: list[Any] = []
    is_map = _is_mapping_like(to_replace)
    for col in valid:
        if is_map:
            exprs.append(pl.col(col).replace_strict(to_replace, default=pl.col(col)).alias(col))
        else:
            exprs.append(
                pl.col(col).replace_strict({to_replace: value}, default=pl.col(col)).alias(col)
            )
    return exprs


def _value_replacement_exprs_polars(
    valid: list[str],
    mapping: dict[str, Any] | None,
    to_replace: Any,
    value: Any,
) -> list[Any]:
    if mapping:
        return _polars_mapping_exprs(valid, mapping)
    if to_replace is None:
        return []
    return _polars_to_replace_exprs(valid, to_replace, value)


def _pandas_apply_mapping(
    df_out: pd.DataFrame, valid: list[str], mapping: dict[str, Any]
) -> pd.DataFrame:
    is_nested = any(isinstance(v, dict) for v in mapping.values())
    # `replace`'s implicit downcasting is deprecated (pandas GH#54710). Opt into
    # the future (no-silent-downcast) behavior for this call only, then apply
    # the same downcast explicitly via `infer_objects` to preserve current
    # dtype behavior without the warning.
    with pd.option_context("future.no_silent_downcasting", True):
        if is_nested:
            for col, map_dict in mapping.items():
                if col in valid:
                    df_out[col] = df_out[col].replace(map_dict).infer_objects()
        else:
            for col in valid:
                df_out[col] = df_out[col].replace(mapping).infer_objects()
    return df_out


def _pandas_apply_to_replace(
    df_out: pd.DataFrame, valid: list[str], to_replace: Any, value: Any
) -> pd.DataFrame:
    is_map = _is_mapping_like(to_replace)
    with pd.option_context("future.no_silent_downcasting", True):
        for col in valid:
            if is_map:
                df_out[col] = df_out[col].replace(to_replace).infer_objects()
            else:
                df_out[col] = df_out[col].replace(to_replace, value).infer_objects()
    return df_out


def _apply_value_replacement_pandas(
    df_out: pd.DataFrame,
    valid: list[str],
    mapping: dict[str, Any] | None,
    to_replace: Any,
    value: Any,
) -> pd.DataFrame:
    if mapping:
        return _pandas_apply_mapping(df_out, valid, mapping)
    if to_replace is None:
        return df_out
    return _pandas_apply_to_replace(df_out, valid, to_replace, value)


class ValueReplacementApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y
        exprs = _value_replacement_exprs_polars(
            valid,
            params.get("mapping"),
            params.get("to_replace"),
            params.get("value"),
        )
        return (X.with_columns(exprs) if exprs else X), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y
        df_out = X.copy()
        df_out = _apply_value_replacement_pandas(
            df_out,
            valid,
            params.get("mapping"),
            params.get("to_replace"),
            params.get("value"),
        )
        return df_out, _y


@NodeRegistry.register("ValueReplacement", ValueReplacementApplier)
@node_meta(
    id="ValueReplacement",
    name="Replace Values",
    category="Cleaning",
    description="Replace specified values with new values.",
    params={"columns": [], "mapping": {}},
)
class ValueReplacementCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        # Value mapping replaces values in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> ValueReplacementArtifact:  # pylint: disable=arguments-differ
        cols = resolve_columns(X, config)
        mapping = config.get("mapping")
        replacements = config.get("replacements")
        if replacements:
            mapping = {item["old"]: item["new"] for item in replacements}
        return {
            "type": "value_replacement",
            "columns": cols,
            "mapping": mapping,
            "to_replace": config.get("to_replace"),
            "value": config.get("value"),
        }
