"""Alias replacement node (boolean / country / custom canonicalization)."""

import re
import string
from typing import Any, Dict, Tuple

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from .._artifacts import AliasReplacementArtifact
from .._helpers import auto_detect_text_columns as _auto_detect_text_columns
from .._helpers import resolve_valid_columns
from .._schema import SkyulfSchema
from ._common import ALIAS_PUNCTUATION_TABLE, COMMON_BOOLEAN_ALIASES, COUNTRY_ALIAS_MAP


def _resolve_alias_type(config: Dict[str, Any]) -> str:
    """Resolve the alias-type alias and remap legacy values."""
    alias_type = config.get("alias_type") or config.get("mode", "boolean")
    if alias_type == "normalize_boolean":
        return "boolean"
    if alias_type == "canonicalize_country_codes":
        return "country"
    return alias_type


def _normalize_alias_custom_map(custom_map: Dict[Any, Any]) -> Dict[Any, Any]:
    """Lowercase + strip punctuation/spaces from string keys to match runtime cleaning."""
    if not custom_map:
        return custom_map
    normalized: Dict[Any, Any] = {}
    for k, v in custom_map.items():
        if isinstance(k, str):
            clean_k = k.lower().translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
            normalized[clean_k] = v
        else:
            normalized[k] = v
    return normalized


def _resolve_alias_mapping(alias_type: str, custom_map: Dict[str, str]) -> Dict[str, str]:
    if alias_type == "boolean":
        return COMMON_BOOLEAN_ALIASES
    if alias_type == "country":
        return COUNTRY_ALIAS_MAP
    if alias_type == "custom":
        return custom_map
    return {}


def _normalize_alias_pandas(val: Any, mapping: Dict[str, str]) -> Any:
    if not isinstance(val, str):
        return val
    clean = val.lower().translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
    return mapping.get(clean, val)


class AliasReplacementApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y

        mapping = _resolve_alias_mapping(
            params.get("alias_type", "boolean"), params.get("custom_map", {})
        )
        escaped_punct = re.escape(string.punctuation)

        exprs = []
        for col in valid:
            clean_expr = (
                pl.col(col)
                .cast(pl.String)
                .str.to_lowercase()
                .str.replace_all(f"[{escaped_punct}]", "")
                .str.replace_all(" ", "")
            )
            # Polars `replace(default=None)` returns null for non-matches, so we
            # coalesce back to the original value.
            mapped_expr = clean_expr.replace(mapping, default=None)
            final_expr = pl.coalesce([mapped_expr, pl.col(col)])
            exprs.append(final_expr.alias(col))
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y

        mapping = _resolve_alias_mapping(
            params.get("alias_type", "boolean"), params.get("custom_map", {})
        )

        df_out = X.copy()
        for col in valid:
            clean_series = (
                df_out[col]
                .astype(str)
                .str.lower()
                .str.translate(ALIAS_PUNCTUATION_TABLE)
                .str.replace(" ", "")
            )
            mapped_series = clean_series.map(mapping)
            df_out[col] = mapped_series.fillna(df_out[col])
        return df_out, _y


@NodeRegistry.register("AliasReplacement", AliasReplacementApplier)
@node_meta(
    id="AliasReplacement",
    name="Standardize Values",
    category="Cleaning",
    description="Standardize common variations in text values (e.g. Yes/No, Country names).",
    params={"columns": [], "domain": "boolean"},
)
class AliasReplacementCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Alias normalization replaces values in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> AliasReplacementArtifact:
        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, _auto_detect_text_columns)
        alias_type = _resolve_alias_type(config)
        custom_map = _normalize_alias_custom_map(
            config.get("custom_map") or config.get("custom_pairs", {})
        )

        return {
            "type": "alias_replacement",
            "columns": cols,
            "alias_type": alias_type,
            "custom_map": custom_map,
        }
