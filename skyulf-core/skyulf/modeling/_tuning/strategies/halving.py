"""The halving search strategies (HalvingGridSearchCV / HalvingRandomSearchCV).

Leaf module (F-18 split of ``engine.py``).
"""

from collections.abc import Callable
from typing import Any

import numpy as np

# Side-effect import (F401 by design): activates sklearn's experimental
# HalvingGridSearchCV / HalvingRandomSearchCV used by the halving strategies.
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)

from ..params import clean_search_space
from ..schemas import TuningConfig


def build_halving_searcher(
    config: TuningConfig,
    base_estimator: Any,
    cv: Any,
    scoring: Any,
    log_callback: Callable[[str], None] | None,
) -> Any:
    """Builds a HalvingGridSearchCV/HalvingRandomSearchCV searcher for the halving strategies."""
    strategy_params = getattr(config, "strategy_params", {})
    factor = strategy_params.get("factor", 3)
    resource = strategy_params.get("resource", "n_samples")
    min_resources = strategy_params.get("min_resources", "exhaust")

    # Halving search uses sklearn's internal scheduler and does NOT
    # expose per-trial callbacks (no equivalent of Optuna's callbacks=).
    # Emit a started log here so the Live Logs panel is never empty
    # while the search is running. Per-iteration progress is not
    # available without monkey-patching sklearn internals.
    if log_callback:
        space = clean_search_space(config.search_space)
        if config.strategy == "halving_grid":
            grid_size = int(np.prod([len(v) for v in space.values()] or [0]))
            log_callback(
                f"Starting halving_grid search "
                f"(grid_size={grid_size}, factor={factor}, "
                f"resource={resource}, min_resources={min_resources}). "
                f"sklearn HalvingGridSearchCV runs without per-trial callbacks; "
                f"this may take a while."
            )
        else:
            log_callback(
                f"Starting halving_random search "
                f"(n_candidates={config.n_trials}, factor={factor}, "
                f"resource={resource}, min_resources={min_resources}). "
                f"sklearn HalvingRandomSearchCV runs without per-trial callbacks; "
                f"this may take a while."
            )

    if isinstance(min_resources, str) and min_resources.isdigit():
        min_resources = int(min_resources)

    if config.strategy == "halving_grid":
        return HalvingGridSearchCV(
            estimator=base_estimator,
            param_grid=clean_search_space(config.search_space),
            scoring=scoring,
            cv=cv,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            refit=False,
            error_score=np.nan,
            factor=factor,
            resource=resource,
            min_resources=min_resources,
        )
    return HalvingRandomSearchCV(
        estimator=base_estimator,
        param_distributions=clean_search_space(config.search_space),
        n_candidates=config.n_trials,
        scoring=scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        refit=False,
        error_score=np.nan,
        factor=factor,
        resource=resource,
        min_resources=min_resources,
    )
