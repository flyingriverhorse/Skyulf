"""The Optuna search strategy: lazy loader + OptunaSearchCV builder.

Leaf module (F-18 split of ``engine.py``). Optuna is an optional,
heavyweight dependency only needed when a caller actually requests
``strategy="optuna"`` tuning. Resolution (including its multi-path
sklearn-integration fallback chain) is deferred to
``_ensure_optuna_loaded()``, called only from ``build_optuna_searcher``,
so merely importing ``skyulf``/``skyulf.modeling`` never imports optuna
or emits its "OptunaSearchCV not found" warning for users who never use
this strategy.

The load state lives on one mutable object (not module globals) so
the loader needs no ``global`` statements and can guard the once-only
import with a lock against concurrent tuning runs.
"""

import logging
import threading
import warnings
from collections.abc import Callable
from typing import Any

from ..schemas import TuningConfig

logger = logging.getLogger(__name__)


class _OptunaLoadState:
    __slots__ = ("attempted", "has_optuna", "optuna_module", "search_cv")

    def __init__(self) -> None:
        self.attempted = False
        self.has_optuna = False
        self.optuna_module: Any = None
        self.search_cv: Any = None


_optuna_state = _OptunaLoadState()
_optuna_load_lock = threading.Lock()


def _ensure_optuna_loaded() -> bool:
    """Lazily import Optuna and resolve its sklearn-compatible OptunaSearchCV
    integration, memoizing the result under a lock so repeated (possibly
    concurrent) tuning calls don't re-attempt the multi-path fallback import.

    Populates ``_optuna_state`` on success; the ``build_optuna_*`` helpers
    read from it, and are only ever called from ``build_optuna_searcher``
    after this function has run. The legacy module-level names
    (``HAS_OPTUNA``/``OptunaSearchCV``/``optuna``/``_optuna_load_attempted``)
    remain readable via the module ``__getattr__`` below.
    """
    with _optuna_load_lock:
        if _optuna_state.attempted:
            return _optuna_state.has_optuna
        _optuna_state.attempted = True

        try:
            import optuna as _optuna  # ty: ignore[unresolved-import]  # noqa: PLC0415 - optional tuning extra; lazy loader pinned by F-14

            _optuna_state.optuna_module = _optuna
            _optuna_state.has_optuna = True
        except ImportError:
            return _optuna_state.has_optuna

        try:
            from optuna.integration import (  # ty: ignore[unresolved-import]  # noqa: PLC0415 - optional tuning extra; lazy loader pinned by F-14
                OptunaSearchCV as _OptunaSearchCV,
            )

            _optuna_state.search_cv = _OptunaSearchCV
        except ImportError:
            try:
                from optuna.integration.sklearn import (  # ty: ignore[unresolved-import]  # noqa: PLC0415 - optional tuning extra; lazy loader pinned by F-14
                    OptunaSearchCV as _OptunaSearchCV,
                )

                _optuna_state.search_cv = _OptunaSearchCV
            except ImportError:
                try:
                    from optuna_integration.sklearn import (  # ty: ignore[unresolved-import]  # noqa: PLC0415 - optional tuning extra; lazy loader pinned by F-14
                        OptunaSearchCV as _OptunaSearchCV,
                    )

                    _optuna_state.search_cv = _OptunaSearchCV
                except ImportError:
                    _optuna_state.has_optuna = False
                    logger.warning(
                        "Optuna installed but OptunaSearchCV not found. Install 'optuna-integration'."
                    )
        return _optuna_state.has_optuna


def __getattr__(name: str) -> Any:
    """Read-only views of the optuna load state under its legacy names (F-14)."""
    if name == "HAS_OPTUNA":
        return _optuna_state.has_optuna
    if name == "OptunaSearchCV":
        return _optuna_state.search_cv
    if name == "optuna":
        return _optuna_state.optuna_module
    if name == "_optuna_load_attempted":
        return _optuna_state.attempted
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_use_cmaes_numeric_list(v: Any, use_cmaes: bool) -> bool:
    """Returns whether ``v`` is a non-empty numeric list that should become a continuous range."""
    return (
        isinstance(v, list)
        and use_cmaes
        and bool(v)
        and all(isinstance(x, (int, float)) for x in v)
    )


def numeric_range_distribution(v: list) -> Any:
    """Builds an Optuna Int/FloatDistribution spanning the min/max of a numeric list."""
    optuna_mod = _optuna_state.optuna_module
    lo, hi = min(v), max(v)
    if all(isinstance(x, int) for x in v):
        return optuna_mod.distributions.IntDistribution(lo, hi)
    return optuna_mod.distributions.FloatDistribution(float(lo), float(hi))


def distribution_for_value(k: str, v: Any, use_cmaes: bool) -> Any:
    """Builds the Optuna distribution for a single search-space entry.

    Numeric lists become continuous ``IntDistribution``/``FloatDistribution`` under
    CMA-ES (so it samples the full range); everything else stays categorical.
    """
    if is_use_cmaes_numeric_list(v, use_cmaes):
        return numeric_range_distribution(v)
    if isinstance(v, list):
        return _optuna_state.optuna_module.distributions.CategoricalDistribution(v)
    return v


def build_optuna_distributions(search_space: dict[str, Any], use_cmaes: bool) -> dict[str, Any]:
    """Converts a raw search space into Optuna distributions.

    Numeric lists become continuous ``IntDistribution``/``FloatDistribution`` under
    CMA-ES (so it samples the full range); everything else stays categorical.
    """
    return {k: distribution_for_value(k, v, use_cmaes) for k, v in search_space.items()}


def build_optuna_sampler(sampler_name: str, random_state: Any) -> Any:
    """Builds the Optuna sampler for the configured sampler name (random/cmaes/tpe)."""
    optuna_mod = _optuna_state.optuna_module
    if sampler_name == "random":
        return optuna_mod.samplers.RandomSampler(seed=random_state)
    if sampler_name == "cmaes":
        # Suppress the fallback warning for genuinely categorical params
        # (strings, booleans, None) — those can never be continuous and
        # the random fallback for them is expected behaviour.
        return optuna_mod.samplers.CmaEsSampler(seed=random_state, warn_independent_sampling=False)
    return optuna_mod.samplers.TPESampler(seed=random_state)


def build_optuna_pruner(pruner_name: str) -> Any:
    """Builds the Optuna pruner for the configured pruner name (hyperband/none/median)."""
    optuna_mod = _optuna_state.optuna_module
    if pruner_name == "hyperband":
        return optuna_mod.pruners.HyperbandPruner()
    if pruner_name == "none":
        return optuna_mod.pruners.NopPruner()
    return optuna_mod.pruners.MedianPruner()


def build_optuna_searcher(
    config: TuningConfig,
    base_estimator: Any,
    cv: Any,
    scoring: Any,
    progress_callback: Callable[[int, int, float | None, dict | None], None] | None,
    log_callback: Callable[[str], None] | None,
) -> Any:
    """Builds an OptunaSearchCV searcher, wiring up distributions, sampler, pruner, and callbacks."""
    if not _ensure_optuna_loaded():
        raise ImportError(
            "Optuna is not installed. Please install 'optuna' and 'optuna-integration'."
        )

    # Convert search space to Optuna distributions.
    # CMA-ES needs continuous distributions — numeric lists become
    # IntDistribution or FloatDistribution so CMA-ES samples the full
    # range instead of treating discrete values as categories.
    # String / bool / None lists remain CategoricalDistribution; CMA-ES
    # falls back to RandomSampler for those (unavoidable) but we suppress
    # the noisy warning via warn_independent_sampling=False.
    strategy_params = getattr(config, "strategy_params", {})
    use_cmaes = strategy_params.get("sampler", "tpe") == "cmaes"
    distributions = build_optuna_distributions(config.search_space, use_cmaes)

    # Optuna callbacks
    callbacks = []
    if progress_callback:

        def _optuna_callback(study, trial):
            # Optuna doesn't know total trials upfront easily if not set, but we have config.n_trials
            # trial.value is the score (or None if failed/pruned)
            score = trial.value if trial.value is not None else None

            if log_callback:
                log_callback(f"Optuna Trial {trial.number + 1} finished. Mean CV Score: {score}")

            progress_callback(trial.number + 1, config.n_trials, score, trial.params)

        callbacks.append(_optuna_callback)

    # Sampler Selection
    sampler_name = strategy_params.get("sampler", "tpe")
    sampler = build_optuna_sampler(sampler_name, config.random_state)

    # Pruner Selection
    pruner_name = strategy_params.get("pruner", "median")
    pruner = build_optuna_pruner(pruner_name)

    study = _optuna_state.optuna_module.create_study(
        sampler=sampler, pruner=pruner, direction="maximize"
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"OptunaSearchCV is experimental.*",
            category=_optuna_state.optuna_module.exceptions.ExperimentalWarning,
        )
        return _optuna_state.search_cv(
            estimator=base_estimator,
            param_distributions=distributions,
            n_trials=config.n_trials,
            timeout=config.timeout,
            cv=cv,
            scoring=scoring,
            n_jobs=config.n_jobs,
            refit=False,
            verbose=0,
            callbacks=callbacks,
            study=study,
        )
