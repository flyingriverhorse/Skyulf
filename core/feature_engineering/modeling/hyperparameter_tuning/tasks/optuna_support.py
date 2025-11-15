"""Optuna-specific helpers for hyperparameter tuning tasks."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, cast

from ...shared import SearchConfiguration

logger = logging.getLogger(__name__)

CategoricalDistribution: Any
TPESampler: Any
OptunaSearchCV: Any = None
_HAS_OPTUNA = False
_OPTUNA_ACCEPTS_SAMPLER: Optional[bool] = None

try:  # optuna support is optional
    from optuna.distributions import CategoricalDistribution as _CategoricalDistribution
    from optuna.samplers import TPESampler as _TPESampler
except ImportError:  # pragma: no cover - optional dependency safeguard
    CategoricalDistribution = None
    TPESampler = None
else:
    CategoricalDistribution = _CategoricalDistribution
    TPESampler = _TPESampler
    _HAS_OPTUNA = True

if _HAS_OPTUNA:
    try:
        from optuna.integration import OptunaSearchCV as _OptunaSearchCV  # type: ignore[attr-defined]

        OptunaSearchCV = _OptunaSearchCV
    except ImportError:  # pragma: no cover - optuna>=3.4 fallback
        try:
            from optuna.integration.sklearn import OptunaSearchCV as _OptunaSearchCV  # type: ignore[attr-defined]

            OptunaSearchCV = _OptunaSearchCV
        except ImportError:  # pragma: no cover - optuna>=4 fallback
            try:
                from optuna_integration.sklearn import OptunaSearchCV as _OptunaSearchCV  # type: ignore[attr-defined]

                OptunaSearchCV = _OptunaSearchCV
            except ImportError:  # pragma: no cover - integration package missing
                OptunaSearchCV = cast(Any, None)

    if OptunaSearchCV is None:
        _HAS_OPTUNA = False

_OPTUNA_INTEGRATION_GUIDANCE = (
    "Optuna strategy requested but Optuna's sklearn integration is unavailable. "
    "Install 'optuna' together with 'optuna-integration' and restart the worker."
)


def _build_optuna_distributions(search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    if not _HAS_OPTUNA or CategoricalDistribution is None:
        raise RuntimeError(
            "Optuna is not installed. Install the optional dependency to use the optuna search strategy."
        )

    distributions: Dict[str, Any] = {}
    for key, candidates in search_space.items():
        if not candidates:
            continue

        unique_values: List[Any] = []
        seen: set[str] = set()
        for candidate in candidates:
            marker = repr(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            unique_values.append(candidate)

        if not unique_values:
            continue

        distributions[key] = CategoricalDistribution(unique_values)

    if not distributions:
        raise ValueError("Search space is empty after filtering unsupported hyperparameters for the optuna strategy.")

    return distributions


def _create_optuna_searcher(
    optuna_kwargs: Dict[str, Any],
    search_config: SearchConfiguration,
    _search_cls: Any = None,
    _sampler_cls: Any = None,
    _accepts_sampler_fn: Optional[Any] = None,
):
    """Create an Optuna-backed searcher with injectable dependencies for tests."""

    search_cls = _search_cls if _search_cls is not None else OptunaSearchCV
    sampler_cls = _sampler_cls if _sampler_cls is not None else TPESampler
    accepts_sampler = _accepts_sampler_fn if _accepts_sampler_fn is not None else _optuna_accepts_sampler

    if not _HAS_OPTUNA or search_cls is None:  # pragma: no cover - guarded by caller
        raise RuntimeError(_OPTUNA_INTEGRATION_GUIDANCE)

    kwargs: Dict[str, Any] = dict(optuna_kwargs)
    sampler_added = False

    if sampler_cls is not None and accepts_sampler():
        kwargs["sampler"] = sampler_cls(
            seed=search_config.random_state if search_config.random_state is not None else None
        )
        sampler_added = True

    try:
        return search_cls(**kwargs)
    except TypeError as exc:
        if sampler_added and "sampler" in str(exc):
            logger.debug("OptunaSearchCV rejected 'sampler'; retrying without sampler. Error: %s", exc)
            kwargs.pop("sampler", None)
            return search_cls(**kwargs)
        raise


def _optuna_accepts_sampler() -> bool:
    global _OPTUNA_ACCEPTS_SAMPLER

    if _OPTUNA_ACCEPTS_SAMPLER is not None:
        return _OPTUNA_ACCEPTS_SAMPLER

    if not _HAS_OPTUNA or OptunaSearchCV is None:
        _OPTUNA_ACCEPTS_SAMPLER = False
        return False

    try:
        signature = inspect.signature(OptunaSearchCV.__init__)  # type: ignore[attr-defined]
    except (TypeError, ValueError):  # pragma: no cover - dynamic loading edge cases
        _OPTUNA_ACCEPTS_SAMPLER = False
        return False

    _OPTUNA_ACCEPTS_SAMPLER = "sampler" in signature.parameters
    return _OPTUNA_ACCEPTS_SAMPLER


__all__ = [
    "CategoricalDistribution",
    "TPESampler",
    "OptunaSearchCV",
    "_HAS_OPTUNA",
    "_OPTUNA_INTEGRATION_GUIDANCE",
    "_build_optuna_distributions",
    "_create_optuna_searcher",
    "_optuna_accepts_sampler",
]
