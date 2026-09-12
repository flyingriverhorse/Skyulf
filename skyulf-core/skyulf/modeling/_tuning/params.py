"""Search-space cleaning and estimator instantiation helpers for tuning.

Leaf module (``engine.py``): no imports from sibling tuning
modules, so the refit, grid/random, and halving paths can all build
estimators without depending on the orchestrator.
"""

import inspect
from dataclasses import replace
from typing import Any

from sklearn.linear_model import LogisticRegression

from .._sklearn_compat import normalize_logistic_regression_params
from .schemas import TuningConfig


def normalize_logistic_search_config(
    model_class: Any, defaults: dict[str, Any], config: TuningConfig
) -> TuningConfig:
    """Resolve nullable Elastic Net ratios before searchers bypass constructor normalization.

    An exclusively Elastic Net search uses the same 0.5 default as direct
    fitting. Search mixed penalties separately when their ratio is unspecified
    to avoid applying this default to unrelated candidates. Caller settings
    are preserved.
    """
    if model_class is not LogisticRegression:
        return config
    space = clean_search_space(config.search_space)
    penalties = space.get("penalty", [defaults.get("penalty")])
    penalties = getattr(penalties, "choices", penalties)
    if not isinstance(penalties, (list, tuple)) or "elasticnet" not in penalties:
        return config
    ratios = space.get("l1_ratio", [defaults.get("l1_ratio")])
    choices = getattr(ratios, "choices", ratios)
    # Explicit numeric distributions already supply a ratio. Categorical
    # distributions expose choices and must keep that type for CMA-ES routing.
    if not isinstance(choices, (list, tuple)) or all(ratio is not None for ratio in choices):
        return config
    if any(penalty != "elasticnet" for penalty in penalties):
        raise ValueError(
            "Logistic Regression: search elasticnet separately from other penalties "
            "when l1_ratio is omitted or None."
        )
    resolved: Any = [0.5 if ratio is None else ratio for ratio in choices]
    if hasattr(ratios, "choices"):
        resolved = type(ratios)(resolved)
    return replace(
        config,
        search_space={
            **config.search_space,
            "l1_ratio": resolved,
        },
    )


def clean_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    """Recursively cleans the search space.

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
