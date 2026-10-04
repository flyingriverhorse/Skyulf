"""Shared configured-model capability checks for runtime and integrations."""

from typing import Any

from ..registry import NodeRegistry
from ._class_weights import split_class_weight_params
from ._sample_weights import SampleWeightError, ensure_sample_weight_support
from .sklearn_wrapper import SklearnCalculator


def ensure_model_sample_weight_support(
    model_type: str, params: dict[str, Any] | None = None
) -> None:
    """Reject a configured model unless every fit can receive explicit row weights.

    Parameters use the same flat model-parameter mapping as pipeline modeling
    ``params``. Ensemble base/final selections and calibration are resolved by
    their calculators, so integration selectors share the runtime policy.
    """
    calculator = NodeRegistry.get_calculator(model_type)()
    if not isinstance(calculator, SklearnCalculator) or calculator.problem_type not in (
        "classification",
        "regression",
    ):
        raise SampleWeightError(f"{model_type} does not support sample_weight.")
    config = {"params": dict(params or {})}
    for resolver_name in ("_resolve_estimators", "_resolve_base_estimator"):
        resolver = getattr(calculator, resolver_name, None)
        if resolver is not None:
            config = resolver(config)
    resolved = calculator._resolve_fit_params(config)
    resolved, _ = split_class_weight_params(calculator.model_class, resolved)
    model = calculator.model_class(**calculator._filter_supported_params(resolved))
    ensure_sample_weight_support(model)


def model_supports_sample_weight(model_type: str, params: dict[str, Any] | None = None) -> bool:
    """Return whether the configured model meets the shared row-weight contract."""
    try:
        ensure_model_sample_weight_support(model_type, params)
    except (ValueError, TypeError, ImportError):
        return False
    return True
