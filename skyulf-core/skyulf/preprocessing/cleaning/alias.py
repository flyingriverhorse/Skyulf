"""Alias replacement node (boolean / country / custom canonicalization)."""

import re
import string
from typing import Any

import polars as pl

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import AliasReplacementArtifact
from .._helpers import auto_detect_text_columns as _auto_detect_text_columns
from .._helpers import resolve_valid_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import ALIAS_PUNCTUATION_TABLE, COMMON_BOOLEAN_ALIASES, COUNTRY_ALIAS_MAP


def _resolve_alias_type(config: dict[str, Any]) -> str:
    """Resolve the alias-type alias and remap legacy values."""
    alias_type = config.get("alias_type") or config.get("mode", "boolean")
    if alias_type == "normalize_boolean":
        return "boolean"
    if alias_type == "canonicalize_country_codes":
        return "country"
    return alias_type


def _normalize_alias_custom_map(custom_map: dict[Any, Any]) -> dict[Any, Any]:
    """Lowercase + strip punctuation/spaces from string keys to match runtime cleaning."""
    if not custom_map:
        return custom_map
    normalized: dict[Any, Any] = {}
    for k, v in custom_map.items():
        if isinstance(k, str):
            clean_k = k.lower().translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
            normalized[clean_k] = v
        else:
            normalized[k] = v
    return normalized


def _resolve_alias_mapping(alias_type: str, custom_map: dict[str, str]) -> dict[str, str]:
    if alias_type == "boolean":
        return COMMON_BOOLEAN_ALIASES
    if alias_type == "country":
        return COUNTRY_ALIAS_MAP
    if alias_type == "custom":
        return custom_map
    return {}


def _normalize_alias_pandas(val: Any, mapping: dict[str, str]) -> Any:
    if not isinstance(val, str):
        return val
    clean = val.lower().translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
    return mapping.get(clean, val)


class AliasReplacementApplier(BaseApplier):
    """Canonicalise near-duplicate text values in the resolved columns.

    The pandas and polars paths must agree value-for-value: each cell is
    normalised (lowercased, punctuation and spaces stripped) and looked up in
    the alias map, and a cell that matches nothing is left exactly as it was.
    ``punctuation`` mode is the exception — it strips punctuation only,
    preserving case and spaces.
    """

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch the replacement to the pandas or polars path; ``y`` passes through."""
        return apply_dual_engine(
            X, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y

        alias_type = params.get("alias_type", "boolean")
        escaped_punct = re.escape(string.punctuation)

        if alias_type == "punctuation":
            # Punctuation mode strips punctuation only — case and spaces are
            # preserved, unlike the alias-mapping modes which normalise fully.
            exprs = [
                pl.col(col).cast(pl.String).str.replace_all(f"[{escaped_punct}]", "").alias(col)
                for col in valid
            ]
            return X.with_columns(exprs), _y

        mapping = _resolve_alias_mapping(alias_type, params.get("custom_map", {}))

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
            mapped_expr = clean_expr.replace_strict(mapping, default=None)
            final_expr = pl.coalesce([mapped_expr, pl.col(col)])
            exprs.append(final_expr.alias(col))
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y

        alias_type = params.get("alias_type", "boolean")

        if alias_type == "punctuation":
            df_out = X.copy()
            for col in valid:
                series = df_out[col]
                stripped = series.astype(str).str.translate(ALIAS_PUNCTUATION_TABLE)
                # astype(str) turns NaN into "nan"; restore the original NaN.
                df_out[col] = stripped.where(series.notna(), series)
            return df_out, _y

        mapping = _resolve_alias_mapping(alias_type, params.get("custom_map", {}))

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
    learns_from_data=False,
)
class AliasReplacementCalculator(BaseCalculator):
    """Resolve alias-replacement config into an artifact; nothing is learned from data."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Pass the input schema through, since only cell values are rewritten."""
        # Alias normalization replaces values in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> AliasReplacementArtifact:  # pylint: disable=arguments-differ
        """Build the artifact, auto-detecting text columns when none are named.

        An explicit empty column selection yields an empty artifact, which
        makes the applier a no-op. Legacy ``alias_type`` spellings are remapped
        and ``custom_map`` keys are normalised exactly like runtime values, so
        a hand-written map still matches.
        """
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
