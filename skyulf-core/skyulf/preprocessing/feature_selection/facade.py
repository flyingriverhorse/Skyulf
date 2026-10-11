"""Unified feature-selection facade node."""

import logging
from collections.abc import Callable, Mapping
from typing import Any

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ..base import BaseApplier, BaseCalculator
from .correlation import CorrelationThresholdApplier, CorrelationThresholdCalculator
from .model_based import ModelBasedSelectionApplier, ModelBasedSelectionCalculator
from .univariate import UnivariateSelectionApplier, UnivariateSelectionCalculator
from .variance import VarianceThresholdApplier, VarianceThresholdCalculator

logger = logging.getLogger(__name__)

_FS_APPLIERS = {
    "variance_threshold": VarianceThresholdApplier,
    "correlation_threshold": CorrelationThresholdApplier,
    "univariate_selection": UnivariateSelectionApplier,
    "model_based_selection": ModelBasedSelectionApplier,
}


class FeatureSelectionApplier(BaseApplier):
    """Route a fitted selection artifact to the applier that produced it."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Delegate inspection to the same saved discriminator used by local apply."""
        if type(raw) is not dict:
            raise ValueError("Local fitted selection state must be a dictionary.")
        kind = raw.get("type")
        if kind is not None and not isinstance(kind, str):
            raise ValueError("Fitted selection type must be a string.")
        owner = _FS_APPLIERS.get(kind)
        if owner is not None:
            owner.validate_inference_state(raw)
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Reuse the selected owner's context; unmatched saved types remain passthroughs."""
        if engine not in {"pandas", "polars"}:
            return None
        FeatureSelectionApplier.validate_inference_state(state)
        owner = _FS_APPLIERS.get(state.get("type"))
        if owner is not None:
            return owner.inference_capability(state, engine=engine)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    def apply(
        self,
        df: Any,
        params: dict[str, Any],
    ) -> Any:
        """Apply whichever concrete selection applier the artifact's ``type`` names.

        An unrecognized ``type`` — including the empty artifact an unknown method
        produces at fit time — is an identity passthrough, so this facade can
        never drop data on its own.
        """
        # The params returned by the specific calculator carry a "type" tag
        # that selects the right concrete applier.
        type_name = params.get("type")

        applier: BaseApplier | None = None
        if type_name == "variance_threshold":
            applier = VarianceThresholdApplier()
        elif type_name == "correlation_threshold":
            applier = CorrelationThresholdApplier()
        elif type_name == "univariate_selection":
            applier = UnivariateSelectionApplier()
        elif type_name == "model_based_selection":
            applier = ModelBasedSelectionApplier()

        if applier:
            return applier.apply(df, params)  # pylint: disable=no-value-for-parameter
        # Identity passthrough when no concrete applier matches.
        return df


_FS_CALCULATORS: dict[str, Callable[[], BaseCalculator]] = {
    "variance": VarianceThresholdCalculator,
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
    learns_from_data=True,
)
class FeatureSelectionCalculator(BaseCalculator):
    """Resolve ``config["method"]`` to one of the four concrete selection nodes."""

    def fit(
        self,
        df: Any,
        config: dict[str, Any],
    ) -> Mapping[str, Any]:
        """Fit the concrete calculator the method alias maps to and return its artifact.

        ``_FS_CALCULATORS`` collapses eleven aliases — node ids such as
        ``variance`` alongside sklearn-style names such as ``select_k_best`` and
        ``rfe`` — onto four calculators, so several spellings reach the same fit.
        An unknown method logs a warning and returns an empty artifact rather
        than raising, which degrades the node to a passthrough.
        """
        method = config.get("method", "select_k_best")
        ctor = _FS_CALCULATORS.get(method)
        if ctor is None:
            logger.warning(f"Unknown feature selection method: {method}")
            return {}
        return ctor().fit(df, config)
