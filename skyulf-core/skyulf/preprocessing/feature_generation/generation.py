"""Feature-generation (math) node."""

from copy import deepcopy
from typing import Any, cast

import pandas as pd

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import FeatureGenerationArtifact
from .._helpers import select_then_to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import DEFAULT_EPSILON, _resolve_group_agg_cols
from ._pandas_ops import _PANDAS_AGG_METHODS, _featgen_apply_pandas
from ._polars_ops import _featgen_apply_polars


class FeatureGenerationApplier(BaseApplier):
    """Append the columns described by a feature-generation artifact."""

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Evaluate each configured operation and append its output column to ``X``.

        Operations run in order, so a later one can read a column an earlier one
        created. A failing operation is logged and skipped rather than aborting
        the node, and an output name that already exists is suffixed until
        unique unless ``allow_overwrite`` says otherwise.

        Group aggregates require training-fitted mappings. Legacy artifacts
        lacking those mappings must be refitted before inference.
        """
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
        """
        params: FeatureGenerationArtifact = {
            "type": "feature_generation",
            "operations": deepcopy(config.get("operations", [])),
            "epsilon": config.get("epsilon", DEFAULT_EPSILON),
            "allow_overwrite": config.get("allow_overwrite", False),
        }
        if not any(op.get("operation_type") == "group_agg" for op in params["operations"]):
            return params

        working = X
        for index, op in enumerate(params["operations"]):
            if op.get("operation_type") == "group_agg":
                op["group_agg_mapping"] = _fit_group_aggregation(working, op)
            working = FeatureGenerationApplier().apply(
                working, {**params, "operations": [op], "_operation_offset": index}
            )
        return params
