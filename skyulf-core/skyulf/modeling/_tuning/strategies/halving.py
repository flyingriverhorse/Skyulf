"""The halving search strategies (HalvingGridSearchCV / HalvingRandomSearchCV).

Leaf module (F-18 split of ``engine.py``).
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import numpy as np

# Side-effect import (F401 by design): activates sklearn's experimental
# HalvingGridSearchCV / HalvingRandomSearchCV used by the halving strategies.
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)

from ..fold_scoring import wrap_fold_scorer
from ..params import clean_search_space
from ..schemas import TuningConfig


class _CoverageResults:
    """Preserve scalar search selection while collecting fold coverage returned by workers."""

    def _format_results(
        self, candidate_params: Any, n_splits: int, out: Any, more_results: Any = None
    ):
        """Separate diagnostics before sklearn infers whether search scores are multimetric."""
        for record in out:
            scores = record["test_scores"]
            if isinstance(scores, dict):
                record["evaluation_coverage"] = scores
                record["test_scores"] = scores["score"]
            record.setdefault("evaluation_coverage", {})
            if isinstance(record.get("train_scores"), dict):
                record["train_scores"] = record["train_scores"]["score"]
        # sklearn >=1.7 inspects this same `out` after formatting to select its
        # scalar/multimetric mode. Keep the protected hook confined to this mixin.
        results = cast(Any, super())._format_results(candidate_params, n_splits, out, more_results)
        for key in ("input_rows", "scored_rows", "excluded_rows"):
            counts = np.asarray(
                [record["evaluation_coverage"].get(key, np.nan) for record in out]
            ).reshape(-1, n_splits)
            for fold in range(n_splits):
                results[f"split{fold}_test_{key}"] = counts[:, fold]
        return results


class _CoverageHalvingGridSearchCV(_CoverageResults, HalvingGridSearchCV):
    """Halving grid search with diagnostic row counts and unchanged scalar ranking."""


class _CoverageHalvingRandomSearchCV(_CoverageResults, HalvingRandomSearchCV):
    """Halving random search with diagnostic row counts and unchanged scalar ranking."""


def bound_sample_resources(config: TuningConfig, row_count: int) -> TuningConfig:
    """Cap sample ceilings to the current search population, including nested folds.

    An explicit minimum remains a requirement and fails when it cannot be met.
    Estimator resources such as tree counts are independent of training rows.
    The caller's requested policy stays unchanged for other folds and final refit.
    """
    params = config.strategy_params
    if params.get("resource", "n_samples") != "n_samples":
        return config
    minimum = _resource_count(params.get("min_resources", "exhaust"))
    maximum = _resource_count(params.get("max_resources", "auto"))
    _validate_sample_resources(minimum, maximum)
    available = row_count if maximum == "auto" else min(maximum, row_count)
    if type(minimum) is int and minimum > available:
        raise ValueError(
            f"Halving min_resources={minimum} exceeds the current sample budget ({available})."
        )
    if maximum == "auto" or maximum <= row_count:
        return config
    return replace(config, strategy_params={**params, "max_resources": available})


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
    requested_resource = resource
    min_resources = strategy_params.get("min_resources", "exhaust")
    max_resources = strategy_params.get("max_resources", "auto")
    if type(resource) is not str or not resource:
        raise ValueError("Halving resource must name n_samples or an estimator parameter.")
    min_resources = _resource_count(min_resources)
    max_resources = _resource_count(max_resources)
    resource = _validate_resource(base_estimator, resource, min_resources, max_resources)
    if type(min_resources) is int and type(max_resources) is int and min_resources > max_resources:
        raise ValueError("Halving min_resources cannot exceed max_resources.")
    space = clean_search_space(config.search_space)
    if resource in space or requested_resource in space:
        raise ValueError("Halving resource cannot also appear in search_space.")
    scoring = wrap_fold_scorer(base_estimator, scoring, include_coverage=True)

    _log_search_start(config, space, factor, resource, min_resources, max_resources, log_callback)

    if config.strategy == "halving_grid":
        return _CoverageHalvingGridSearchCV(
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
            max_resources=max_resources,
        )
    return _CoverageHalvingRandomSearchCV(
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
        max_resources=max_resources,
    )


def _resource_count(value: Any) -> Any:
    """Accept digit strings as resource counts while retaining sklearn sentinels."""
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _validate_resource(
    base_estimator: Any, resource: str, min_resources: Any, max_resources: Any
) -> str:
    """Validate sample budgets or resolve the declared estimator resource parameter."""
    if resource == "n_samples":
        _validate_sample_resources(min_resources, max_resources)
        return resource
    return _validate_estimator_resource(base_estimator, resource, min_resources, max_resources)


def _validate_sample_resources(min_resources: Any, max_resources: Any) -> None:
    """Allow sample-budget sentinels or strictly positive integer bounds."""
    if min_resources not in ("exhaust", "smallest") and (
        type(min_resources) is not int or min_resources <= 0
    ):
        raise ValueError("Halving min_resources must be positive or exhaust/smallest.")
    if max_resources != "auto" and (type(max_resources) is not int or max_resources <= 0):
        raise ValueError("Halving max_resources must be auto or a positive integer.")


def _validate_estimator_resource(
    base_estimator: Any, resource: str, min_resources: Any, max_resources: Any
) -> str:
    """Require numeric estimator budgets and route fold-wrapped parameter names."""
    if type(max_resources) is not int or max_resources <= 0:
        raise ValueError("Halving estimator resource requires explicit positive max_resources.")
    if type(min_resources) is not int or min_resources <= 0:
        raise ValueError("Halving estimator resource requires positive integer min_resources.")
    params = base_estimator.get_params(deep=True)
    routed = f"model__estimator__{resource}"
    resource = routed if routed in params else resource
    if resource not in params:
        raise ValueError("Halving resource is not an available estimator parameter.")
    return resource


def _log_search_start(
    config: TuningConfig,
    space: dict,
    factor: Any,
    resource: str,
    min_resources: Any,
    max_resources: Any,
    log_callback: Callable[[str], None] | None,
) -> None:
    """Report the scheduled halving search while sklearn owns iteration progress."""
    # Halving search uses sklearn's internal scheduler and does NOT
    # expose per-trial callbacks (no equivalent of Optuna's callbacks=).
    # Emit a started log here so the Live Logs panel is never empty
    # while the search is running. Per-iteration progress is not
    # available without monkey-patching sklearn internals.
    if log_callback:
        if config.strategy == "halving_grid":
            grid_size = int(np.prod([len(v) for v in space.values()] or [0]))
            log_callback(
                f"Starting halving_grid search "
                f"(grid_size={grid_size}, factor={factor}, "
                f"resource={resource}, min_resources={min_resources}, max_resources={max_resources}). "
                f"sklearn HalvingGridSearchCV runs without per-trial callbacks; "
                f"this may take a while."
            )
        else:
            log_callback(
                f"Starting halving_random search "
                f"(n_candidates={config.n_trials}, factor={factor}, "
                f"resource={resource}, min_resources={min_resources}, max_resources={max_resources}). "
                f"sklearn HalvingRandomSearchCV runs without per-trial callbacks; "
                f"this may take a while."
            )
