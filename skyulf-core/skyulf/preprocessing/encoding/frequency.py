"""Category counts or frequencies learned once and replayed on either engine."""

from collections import Counter
from typing import Any

import polars as pl

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._category_keys import (
    category_key,
    category_key_expr,
    category_keys_pandas,
    uses_category_keys,
)
from .._fitted_validation import local_state_fields
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _detect_target_column, detect_categorical_columns


def _mode(value: Any) -> str:
    """Require an explicit supported statistic without coercing malformed options."""
    if value not in ("frequency", "count"):
        raise ValueError("FrequencyEncoder mode must be 'frequency' or 'count'.")
    return value


def _fit(X: Any, y: Any, config: dict) -> dict:
    """Count canonical scalars, including missing, using only training rows."""
    mode = _mode(config.get("mode", "frequency"))
    target = _detect_target_column(config, y)
    columns = [c for c in resolve_columns(X, config, detect_categorical_columns) if c != target]
    if not columns:
        return {}
    if not len(X):
        raise ValueError("FrequencyEncoder requires training rows for selected columns.")
    return {
        "type": "frequency_encoder",
        "columns": columns,
        "mode": mode,
        "category_key_version": 1,
        "n_samples": len(X),
        "mappings": {c: dict(Counter(category_key(v) for v in X[c])) for c in columns},
    }


def _apply(X: Any, y: Any, state: dict) -> tuple[Any, Any]:
    """Replace each known category by its training statistic and unseen categories by zero."""
    FrequencyEncoderApplier.validate_inference_state(state)
    if not state:
        return X, y
    columns = [c for c in state["columns"] if c in X.columns]
    frequency = state["mode"] == "frequency"
    divisor = state["n_samples"] if frequency else 1
    if isinstance(X, pl.DataFrame):
        expressions = []
        for col in columns:
            expression = category_key_expr(col).replace_strict(
                state["mappings"][col], default=0, return_dtype=pl.Int64
            )
            expressions.append((expression / divisor if frequency else expression).alias(col))
        return X.with_columns(expressions), y
    result = X.copy()
    for col in columns:
        counts = category_keys_pandas(X[col]).map(state["mappings"][col]).fillna(0)
        result[col] = (counts / divisor).astype("float64") if frequency else counts.astype("int64")
    return result, y


class FrequencyEncoderApplier(BaseApplier):
    """Replay fixed category statistics without recomputing inference-batch frequencies."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Check exact learned counts and dimensions before granting local replay."""
        fields = {"type", "columns", "mode", "category_key_version", "n_samples", "mappings"}
        if not local_state_fields(raw, "frequency_encoder", fields, allow_empty=True):
            return raw
        uses_category_keys(raw)
        _mode(raw["mode"])
        columns = raw["columns"]
        if type(columns) is not list or not columns or any(type(c) is not str for c in columns):
            raise ValueError("Fitted frequency columns must be a nonempty ordered string list.")
        if len(columns) != len(set(columns)):
            raise ValueError("Fitted frequency columns must be unique.")
        FrequencyEncoderApplier._validate_counts(raw)
        return raw

    @staticmethod
    def _validate_counts(raw: dict) -> None:
        """Each column counts the same positive training population exactly once."""
        n_samples = raw["n_samples"]
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError("Fitted frequency n_samples must be a positive integer.")
        mappings = raw["mappings"]
        if type(mappings) is not dict or set(mappings) != set(raw["columns"]):
            raise ValueError("Fitted frequency mappings must match selected columns.")
        for mapping in mappings.values():
            FrequencyEncoderApplier._validate_column_counts(mapping, n_samples)

    @staticmethod
    def _validate_column_counts(mapping: Any, n_samples: int) -> None:
        """Inspect one categorical histogram without casting or correcting its entries."""
        if type(mapping) is not dict or any(type(key) is not str for key in mapping):
            raise ValueError("Fitted frequency categories must be string keys.")
        if any(type(count) is not int or count < 1 for count in mapping.values()):
            raise ValueError("Fitted frequency counts must be positive integers.")
        if sum(mapping.values()) != n_samples:
            raise ValueError("Fitted frequency counts disagree with n_samples.")

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe validated row-local lookups without granting worker execution."""
        if engine not in ("pandas", "polars"):
            return None
        FrequencyEncoderApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, y: Any, params: dict) -> Any:
        """Replace selected values while retaining row order, index, and target."""
        return apply_dual_engine(
            (X, y) if y is not None else X, params, {"pandas": _apply, "polars": _apply}
        )


@NodeRegistry.register("FrequencyEncoder", FrequencyEncoderApplier)
@node_meta(
    id="FrequencyEncoder",
    name="Frequency Encoder",
    category="Preprocessing",
    description="Replace categories with training frequencies or counts; unseen values become zero.",
    params={"columns": [], "mode": "frequency"},
    learns_from_data=True,
)
class FrequencyEncoderCalculator(BaseCalculator):
    """Learn counts including missing values; frequency divides by all training rows.

    ``mode='frequency'`` emits float64 and ``mode='count'`` emits int64.
    Omitted columns select categorical features, excluding the named target.
    An explicit empty selection is a no-op. Selected empty training data fails.
    """

    @fit_method
    def fit(self, X: Any, y: Any, config: dict) -> dict:
        """Persist category counts for the selected native dataframe columns."""
        if user_picked_no_columns(config):
            return {}
        return fit_dual_engine(
            (X, y) if y is not None else X, config, {"pandas": _fit, "polars": _fit}
        )


__all__ = ["FrequencyEncoderApplier", "FrequencyEncoderCalculator"]
