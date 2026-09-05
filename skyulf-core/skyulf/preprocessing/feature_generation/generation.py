"""Feature-generation (math) node."""

from typing import Any

import pandas as pd

from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...registry import NodeRegistry
from .._artifacts import FeatureGenerationArtifact
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._common import DEFAULT_EPSILON
from ._pandas_ops import _featgen_apply_pandas
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
        """
        return apply_dual_engine(
            X, params, {"polars": _featgen_apply_polars, "pandas": _featgen_apply_pandas}
        )


@NodeRegistry.register("FeatureGeneration", FeatureGenerationApplier)
@NodeRegistry.register("FeatureMath", FeatureGenerationApplier)
@NodeRegistry.register("FeatureGenerationNode", FeatureGenerationApplier)
@node_meta(
    id="FeatureGenerationNode",
    name="Feature Generation (Math)",
    category="Feature Engineering",
    description="Generate new features using mathematical operations.",
    params={"operations": []},
    learns_from_data=False,
)
class FeatureGenerationCalculator(BaseCalculator):
    """Carry the operation configuration into the artifact; nothing is learned."""

    def fit(
        self,
        df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
        config: dict[str, Any],
    ) -> FeatureGenerationArtifact:
        """Copy the operation list and its guards from ``config`` into the artifact.

        ``df`` is never read. The node is registered ``learns_from_data=False``,
        so the train and test artifacts are identical and this calculator
        contributes no leakage surface of its own.
        """
        return {
            "type": "feature_generation",
            "operations": config.get("operations", []),
            "epsilon": config.get("epsilon", DEFAULT_EPSILON),
            "allow_overwrite": config.get("allow_overwrite", False),
        }
