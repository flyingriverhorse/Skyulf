"""Group infrequent textual categories using fixed training membership."""

import math
from collections import Counter
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._category_keys import category_key, uses_category_keys
from .._fitted_validation import local_state_fields
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _detect_target_column, detect_categorical_columns


def _replacement(value: Any) -> str:
    """Require a nonempty textual fallback, including in saved state."""
    if not isinstance(value, str) or not value:
        raise ValueError("RareCategoryGrouping replacement must be a nonempty string.")
    return value


def _options(config: dict) -> tuple[int, float, str]:
    """Validate conjunctive thresholds and the replacement label."""
    count = config.get("min_count", 2)
    share = config.get("min_share", 0.0)
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
        raise ValueError("RareCategoryGrouping min_count must be a positive integer.")
    if isinstance(share, (bool, np.bool_)) or not isinstance(share, Real):
        raise ValueError("RareCategoryGrouping min_share must be a finite number in [0, 1].")
    if not math.isfinite(share) or not 0 <= share <= 1:
        raise ValueError("RareCategoryGrouping min_share must be a finite number in [0, 1].")
    return int(count), float(share), _replacement(config.get("replacement", "Other"))


def _text_values(series: Any, col: str) -> list[str | None]:
    """Normalize nulls only; reject numeric and structured category values."""
    values = []
    for value in series:
        if isinstance(value, str):
            values.append(value)
        elif pd.api.types.is_scalar(value) and pd.isna(value):
            values.append(None)
        else:
            raise ValueError(
                f"RareCategoryGrouping column {col!r} requires text categories; "
                "use explicit Casting for numeric categories."
            )
    return values


def _fit(X: Any, y: Any, config: dict) -> dict:
    """Persist membership after applying both inclusive training thresholds."""
    count, share, replacement = _options(config)
    target = _detect_target_column(config, y)
    columns = [c for c in resolve_columns(X, config, detect_categorical_columns) if c != target]
    if not columns:
        return {}
    if not len(X):
        raise ValueError("RareCategoryGrouping requires training rows for selected columns.")
    kept = {}
    for col in columns:
        counts = Counter(category_key(v) for v in _text_values(X[col], col))
        kept[col] = [key for key, n in counts.items() if n >= count and n / len(X) >= share]
    return {
        "type": "rare_category_grouping",
        "columns": columns,
        "category_key_version": 1,
        "kept_categories": kept,
        "replacement": replacement,
    }


def _apply(X: Any, y: Any, state: dict) -> tuple[Any, Any]:
    """Preserve learned frequent text/nulls and replace all other values."""
    RareCategoryGroupingApplier.validate_inference_state(state)
    if not state:
        return X, y
    result = X.clone() if isinstance(X, pl.DataFrame) else X.copy()
    for col in state["columns"]:
        if col not in X.columns:
            continue
        kept = set(state["kept_categories"][col])
        values = [
            value if category_key(value) in kept else state["replacement"]
            for value in _text_values(X[col], col)
        ]
        if isinstance(X, pl.DataFrame):
            result = result.with_columns(pl.Series(col, values, dtype=pl.String))
        else:
            result[col] = pd.Series(values, index=X.index, dtype="string")
    return result, y


class RareCategoryGroupingApplier(BaseApplier):
    """Replay learned text-category membership without discovering new frequent values."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Reject malformed or ambiguous saved membership and fallback values."""
        fields = {"type", "columns", "category_key_version", "kept_categories", "replacement"}
        if not local_state_fields(raw, "rare_category_grouping", fields, allow_empty=True):
            return raw
        uses_category_keys(raw)
        _replacement(raw["replacement"])
        columns = raw["columns"]
        if type(columns) is not list or not columns or any(type(c) is not str for c in columns):
            raise ValueError("Fitted rare-grouping columns must be a nonempty ordered string list.")
        if len(columns) != len(set(columns)):
            raise ValueError("Fitted rare-grouping columns must be unique.")
        RareCategoryGroupingApplier._validate_membership(raw)
        return raw

    @staticmethod
    def _validate_membership(raw: dict) -> None:
        """Keep each learned category set attached to its exact source column."""
        kept = raw["kept_categories"]
        if type(kept) is not dict or set(kept) != set(raw["columns"]):
            raise ValueError("Fitted rare-grouping membership must match selected columns.")
        for keys in kept.values():
            if type(keys) is not list or any(type(key) is not str for key in keys):
                raise ValueError("Fitted rare-grouping keys must be ordered strings.")
            if len(keys) != len(set(keys)):
                raise ValueError("Fitted rare-grouping keys must be unique.")

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe validated row-local membership without granting worker execution."""
        if engine not in ("pandas", "polars"):
            return None
        RareCategoryGroupingApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, y: Any, params: dict) -> Any:
        """Replace selected columns with nullable strings, retaining index and target."""
        return apply_dual_engine(
            (X, y) if y is not None else X, params, {"pandas": _apply, "polars": _apply}
        )


@NodeRegistry.register("RareCategoryGrouping", RareCategoryGroupingApplier)
@node_meta(
    id="RareCategoryGrouping",
    name="Rare Category Grouping",
    category="Preprocessing",
    description="Keep frequent text categories and group rare or unseen values under a label.",
    params={"columns": [], "min_count": 2, "min_share": 0.0, "replacement": "Other"},
    learns_from_data=True,
)
class RareCategoryGroupingCalculator(BaseCalculator):
    """Learn text categories meeting both count and share thresholds, including null.

    Frequent missing values stay null; rare and unseen categories become the
    replacement string. A replacement already present in the data merges with
    grouped values. Numeric categories require explicit Casting before this node.
    """

    @fit_method
    def fit(self, X: Any, y: Any, config: dict) -> dict:
        """Persist fixed membership using training rows only."""
        if user_picked_no_columns(config):
            return {}
        return fit_dual_engine(
            (X, y) if y is not None else X, config, {"pandas": _fit, "polars": _fit}
        )


__all__ = ["RareCategoryGroupingApplier", "RareCategoryGroupingCalculator"]
