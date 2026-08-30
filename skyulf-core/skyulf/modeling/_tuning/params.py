"""Search-space cleaning and estimator instantiation helpers for tuning.

Leaf module (F-18 split of ``engine.py``): no imports from sibling tuning
modules, so the refit, grid/random, and halving paths can all build
estimators without depending on the orchestrator.
"""

import inspect
from typing import Any

from sklearn.linear_model import LogisticRegression

from .._sklearn_compat import normalize_logistic_regression_params
from .schemas import TuningConfig


def clean_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively cleans the search space.
    - Converts "none" string to None.
    """
    cleaned: dict[str, Any] = {}
    for k, v in search_space.items():
        if isinstance(v, list):
            cleaned[k] = [None if x == "none" else x for x in v]
        elif isinstance(v, dict):
            cleaned[k] = clean_search_space(v)
        else:
            cleaned[k] = None if v == "none" else v
    return cleaned


def split_flat_and_nested_params(
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Splits ``params`` into flat constructor args and nested ``a__b`` keys."""
    flat = {k: v for k, v in params.items() if "__" not in str(k)}
    nested = {k: v for k, v in params.items() if "__" in str(k)}
    return flat, nested


def filter_params_to_signature(model_class: Any, flat: dict[str, Any]) -> dict[str, Any]:
    """Filters ``flat`` down to ``model_class``'s constructor params, unless it accepts ``**kwargs``."""
    sig = inspect.signature(model_class)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_kwargs:
        return flat
    return {k: v for k, v in flat.items() if k in sig.parameters}


def instantiate_model(model_class: Any, params: dict[str, Any]) -> Any:
    """Build an estimator, routing nested ``a__b`` keys through ``set_params``.

    Constructor args (no ``__``) are filtered to the model's signature
    (unless it accepts ``**kwargs``); nested keys — e.g. an ensemble's
    ``random_forest__n_estimators`` — are applied afterwards via
    ``set_params`` because sklearn estimators only accept them that way.
    """
    flat, nested = split_flat_and_nested_params(params)
    flat = filter_params_to_signature(model_class, flat)

    # LogisticRegression-only: sklearn >=1.8 deprecates the ``penalty``
    # constructor arg. The tuning engine builds estimators directly
    # (bypassing LogisticRegressionCalculator._resolve_fit_params), so a
    # ``penalty`` coming from the search space/best_params would otherwise
    # reach sklearn unnormalized and trigger the FutureWarning on every
    # fold fit and the final refit. Other models (e.g. SGDClassifier) also
    # have a ``penalty`` param with different, non-deprecated semantics,
    # so this must stay scoped to LogisticRegression specifically.
    if model_class is LogisticRegression:
        flat = normalize_logistic_regression_params(flat)

    model = model_class(**flat)
    if nested:
        model.set_params(**nested)
    return model


def seed_params(config: TuningConfig) -> dict[str, Any]:
    """The caller's seed as a params overlay, for every tuning-path model build."""
    return {"random_state": config.random_state} if config.random_state is not None else {}
