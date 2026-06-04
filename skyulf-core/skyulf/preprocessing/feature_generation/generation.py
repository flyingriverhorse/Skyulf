"""Feature-generation (math) node."""

from typing import Any, Dict, Tuple, Union

import pandas as pd

from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...registry import NodeRegistry
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from .._artifacts import FeatureGenerationArtifact
from ._common import DEFAULT_EPSILON
from ._pandas_ops import _featgen_apply_pandas
from ._polars_ops import _featgen_apply_polars


class FeatureGenerationApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, _featgen_apply_polars, _featgen_apply_pandas)


@NodeRegistry.register("FeatureGeneration", FeatureGenerationApplier)
@NodeRegistry.register("FeatureMath", FeatureGenerationApplier)
@NodeRegistry.register("FeatureGenerationNode", FeatureGenerationApplier)
@node_meta(
    id="FeatureGenerationNode",
    name="Feature Generation (Math)",
    category="Feature Engineering",
    description="Generate new features using mathematical operations.",
    params={"operations": []},
)
class FeatureGenerationCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> FeatureGenerationArtifact:
        return {
            "type": "feature_generation",
            "operations": config.get("operations", []),
            "epsilon": config.get("epsilon", DEFAULT_EPSILON),
            "allow_overwrite": config.get("allow_overwrite", False),
        }
