"""Feature-generation (math) node."""

import logging
from copy import deepcopy
from decimal import Decimal
from numbers import Real
from typing import Any, cast

import pandas as pd

from ..._validation import raise_invalid_choice
from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import FeatureGenerationArtifact
from .._fitted_validation import local_scalar, local_state_fields
from .._helpers import select_then_to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import (
    DEFAULT_EPSILON,
    FEATURE_MATH_ALLOWED_TYPES,
    _resolve_group_agg_cols,
    _similarity_backend,
    _validate_similarity_backend,
)
from ._pandas_ops import _PANDAS_AGG_METHODS, _featgen_apply_pandas
from ._polars_ops import _featgen_apply_polars

logger = logging.getLogger(__name__)


def _check_similarity_runtime(operations: list[dict[str, Any]]) -> None:
    """Keep fitted similarity implementations stable and identify unpinned legacy artifacts."""
    for op in operations:
        if op.get("operation_type") != "similarity":
            continue
        backend = op.get("similarity_backend")
        _validate_similarity_backend(backend)
        if backend is None:
            logger.warning(
                "Legacy similarity artifact has no saved backend; refit to keep feature values "
                "stable across environments."
            )


def _validate_operation_types(operations: list[dict[str, Any]]) -> None:
    """Reject unsupported operations before fitting or replaying an artifact."""
    for op in operations:
        op_type = op.get("operation_type", "arithmetic")
        if op_type == "polynomial":
            raise ValueError(
                "FeatureGeneration does not support operation_type='polynomial'; "
                "use the PolynomialFeatures node instead."
            )
        if op_type not in FEATURE_MATH_ALLOWED_TYPES:
            raise_invalid_choice(
                op_type, FEATURE_MATH_ALLOWED_TYPES, "FeatureGeneration operation_type"
            )


def _validate_group_mapping(op: dict) -> None:
    """Require fitted lookup dimensions while retaining unresolved fit-time no-ops."""
    if "group_agg_mapping" not in op:
        raise ValueError("FeatureGeneration group_agg requires fitted statistics; refit the node.")
    mapping = op["group_agg_mapping"]
    if mapping is None:
        return
    if type(mapping) is not dict or set(mapping) != {
        "group_column",
        "keys",
        "values",
        "null_value",
    }:
        raise ValueError("Fitted group aggregation mapping has unexpected fields.")
    if not isinstance(mapping["group_column"], str):
        raise ValueError("Fitted group aggregation requires a named group column.")
    _validate_group_values(mapping)


def _validate_group_values(mapping: dict) -> None:
    """Inspect the learned lookup vectors without recomputing any aggregate."""
    keys, values = mapping["keys"], mapping["values"]
    if type(keys) not in (list, tuple) or type(values) not in (list, tuple):
        raise ValueError("Fitted group keys and values must be ordered sequences.")
    if len(keys) != len(values):
        raise ValueError("Fitted group keys and values must have equal lengths.")
    if any(
        value is not None and not isinstance(value, Real)
        for value in [*values, mapping["null_value"]]
    ):
        raise ValueError("Fitted aggregate values must be numeric or missing.")


def _operation_context(op: dict, epsilon: Any, engine: str) -> str | None:
    """Retain pandas string rendering and uncertain numeric fallback dependencies."""
    kind = op.get("operation_type", "arithmetic")
    if kind == "similarity":
        if op.get("similarity_backend") is None:
            return None
        return "global" if engine == "pandas" else "row"
    return _numeric_operation_context(op, epsilon, kind)


def _numeric_operation_context(op: dict, epsilon: Any, kind: str) -> str | None:
    """Avoid promising independent batches for nonnumeric fill or epsilon fallbacks."""
    divides = kind == "ratio" or (kind == "arithmetic" and op.get("method") == "divide")
    if divides and not isinstance(epsilon, Real):
        return None
    fill = op.get("fillna")
    if kind == "arithmetic" and fill is not None and not isinstance(fill, Real):
        return None
    return "row"


class FeatureGenerationApplier(BaseApplier):
    """Append the columns described by a feature-generation artifact."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect saved operations, pinned similarity and learned group lookup shapes."""
        local_state_fields(
            raw, "feature_generation", {"type", "operations", "epsilon", "allow_overwrite"}
        )
        operations = raw["operations"]
        if type(operations) not in (list, tuple) or any(type(op) is not dict for op in operations):
            raise ValueError(
                "Fitted feature operations must be an ordered sequence of dictionaries."
            )
        if not isinstance(raw["allow_overwrite"], Decimal):
            local_scalar(raw["allow_overwrite"], "allow_overwrite")
        _validate_operation_types(operations)
        for op in operations:
            if op.get("operation_type") == "group_agg":
                _validate_group_mapping(op)
            if op.get("operation_type") == "similarity":
                _validate_similarity_backend(op.get("similarity_backend"))
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe saved feature replay without granting arbitrary partition execution."""
        if engine not in {"pandas", "polars"}:
            return None
        FeatureGenerationApplier.validate_inference_state(state)
        contexts = {_operation_context(op, state["epsilon"], engine) for op in state["operations"]}
        if None in contexts:
            return None
        context = "global" if "global" in contexts else "row"
        return ExecutionCapability(engine, "apply", "local", "preserve", context)

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Evaluate each configured operation and append its output column to ``X``.

        Operations run in order, so a later one can read a column an earlier one
        created. Unsupported operation types raise ``ValueError`` before any
        operation runs. A failure inside a supported operation is logged and
        skipped, and an output name that already exists is suffixed until
        unique unless ``allow_overwrite`` says otherwise.

        Datetime extraction uses ``output_column`` exactly for one configured
        source and feature. With multiple features it produces ``name_feature``;
        with multiple sources it produces ``name_source_feature``. Without an
        explicit name, outputs remain ``source_feature``, optionally preceded
        by ``output_prefix``. Every datetime output follows the collision policy.
        Saved pipelines affected by formerly ignored names or overwritten
        columns require refitting after upgrading.

        Group aggregates require training-fitted mappings. Legacy artifacts
        lacking those mappings must be refitted before inference.
        """
        _validate_operation_types(params.get("operations", []))
        _check_similarity_runtime(params.get("operations", []))
        if any(
            op.get("operation_type") == "group_agg" and "group_agg_mapping" not in op
            for op in params.get("operations", [])
        ):
            raise ValueError(
                "FeatureGeneration group_agg requires fitted statistics; refit the node."
            )
        return apply_dual_engine(
            X, params, {"polars": _featgen_apply_polars, "pandas": _featgen_apply_pandas}
        )


def _fit_group_aggregation(X: Any, op: dict[str, Any]) -> dict[str, Any] | None:
    """Learn one aggregate from training rows, keeping missing keys as a group."""
    resolved = _resolve_group_agg_cols(op, list(X.columns))
    if resolved is None:
        return None
    group_col, target_col, method = resolved
    if method not in _PANDAS_AGG_METHODS:
        return None

    frame = select_then_to_pandas(X, [group_col, target_col])
    target = frame[target_col]
    if method != "count":
        target = pd.to_numeric(target, errors="coerce")
    statistics = target.groupby(frame[group_col], dropna=False, sort=False, observed=True).agg(
        method
    )
    fitted: dict[str, Any] = {
        "group_column": group_col,
        "keys": [],
        "values": [],
        "null_value": None,
    }
    for key, value in statistics.items():
        group_key = cast(Any, key)
        numeric = None if pd.isna(value) else float(value)
        if pd.isna(group_key):
            fitted["null_value"] = numeric
        else:
            fitted["keys"].append(group_key.item() if hasattr(group_key, "item") else group_key)
            fitted["values"].append(numeric)
    return fitted


@NodeRegistry.register("FeatureGeneration", FeatureGenerationApplier)
@NodeRegistry.register("FeatureMath", FeatureGenerationApplier)
@NodeRegistry.register("FeatureGenerationNode", FeatureGenerationApplier)
@node_meta(
    id="FeatureGenerationNode",
    name="Feature Generation (Math)",
    category="Feature Engineering",
    description="Generate new features using mathematical operations.",
    params={"operations": []},
    learns_from_data=True,
)
class FeatureGenerationCalculator(BaseCalculator):
    """Record row-local operations and fit group aggregates from training rows."""

    @fit_method
    def fit(
        self,
        X: Any,
        _y: Any,
        config: dict[str, Any],
    ) -> FeatureGenerationArtifact:  # pylint: disable=arguments-differ
        """Fit aggregate mappings against the intermediate training feature frame.

        Operations execute in order while fitting so an aggregate can consume
        an earlier generated column. Its saved mapping supplies held-out rows;
        unknown keys stay missing, and no inference values are aggregated.
        Configurations containing only row-local operations need no data fit.
        Unsupported operation types raise ``ValueError``; polynomial expansion
        belongs in the separate ``PolynomialFeatures`` node.
        """
        _validate_operation_types(config.get("operations", []))
        params: FeatureGenerationArtifact = {
            "type": "feature_generation",
            "operations": deepcopy(config.get("operations", [])),
            "epsilon": config.get("epsilon", DEFAULT_EPSILON),
            "allow_overwrite": config.get("allow_overwrite", False),
        }
        for op in params["operations"]:
            if op.get("operation_type") == "similarity":
                op["similarity_backend"] = _similarity_backend()
        if not any(op.get("operation_type") == "group_agg" for op in params["operations"]):
            return params

        working = X
        for index, op in enumerate(params["operations"]):
            if op.get("operation_type") == "group_agg":
                op["group_agg_mapping"] = _fit_group_aggregation(working, op)
            # apply_method exposes the public (data, params) call signature.
            working = FeatureGenerationApplier().apply(  # pylint: disable=no-value-for-parameter
                working, {**params, "operations": [op], "_operation_offset": index}
            )
        return params
