"""Unified feature-selection facade node."""

import logging
from typing import Any, Callable, Dict, Mapping, Optional

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ..base import BaseApplier, BaseCalculator
from .correlation import CorrelationThresholdApplier, CorrelationThresholdCalculator
from .model_based import ModelBasedSelectionApplier, ModelBasedSelectionCalculator
from .univariate import UnivariateSelectionApplier, UnivariateSelectionCalculator
from .variance import VarianceThresholdApplier, VarianceThresholdCalculator

logger = logging.getLogger(__name__)


class FeatureSelectionApplier(BaseApplier):
    def apply(
        self,
        df: Any,
        params: Dict[str, Any],
    ) -> Any:
        # The params returned by the specific calculator carry a "type" tag
        # that selects the right concrete applier.
        type_name = params.get("type")

        applier: Optional[BaseApplier] = None
        if type_name == "variance_threshold":
            applier = VarianceThresholdApplier()
        elif type_name == "correlation_threshold":
            applier = CorrelationThresholdApplier()
        elif type_name == "univariate_selection":
            applier = UnivariateSelectionApplier()
        elif type_name == "model_based_selection":
            applier = ModelBasedSelectionApplier()

        if applier:
            return applier.apply(df, params)
        # Identity passthrough when no concrete applier matches.
        return df


_FS_CALCULATORS: Dict[str, Callable[[], BaseCalculator]] = {
    "variance_threshold": VarianceThresholdCalculator,
    "correlation_threshold": CorrelationThresholdCalculator,
    "select_k_best": UnivariateSelectionCalculator,
    "select_percentile": UnivariateSelectionCalculator,
    "generic_univariate_select": UnivariateSelectionCalculator,
    "select_fpr": UnivariateSelectionCalculator,
    "select_fdr": UnivariateSelectionCalculator,
    "select_fwe": UnivariateSelectionCalculator,
    "select_from_model": ModelBasedSelectionCalculator,
    "rfe": ModelBasedSelectionCalculator,
}


@NodeRegistry.register("feature_selection", FeatureSelectionApplier)
@node_meta(
    id="feature_selection",
    name="Feature Selection (Wrapper)",
    category="Feature Selection",
    description="General wrapper for feature selection strategies.",
    params={"method": "variance", "threshold": 0.0},
)
class FeatureSelectionCalculator(BaseCalculator):
    def fit(
        self,
        df: Any,
        config: Dict[str, Any],
    ) -> Mapping[str, Any]:
        method = config.get("method", "select_k_best")
        ctor = _FS_CALCULATORS.get(method)
        if ctor is None:
            logger.warning(f"Unknown feature selection method: {method}")
            return {}
        return ctor().fit(df, config)
