"""Search builder utilities for hyperparameter tuning."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
)

from ...config.hyperparameters import get_hyperparameters_for_model
from ...shared import SearchConfiguration, _filter_supported_parameters, _sanitize_logistic_regression_hyperparameters, _sanitize_parameters
from .optuna_support import (
    OptunaSearchCV,
    _HAS_OPTUNA,
    _OPTUNA_INTEGRATION_GUIDANCE,
    _build_optuna_distributions,
    _create_optuna_searcher,
)


def _prepare_search_parameters(
    job,
    node_config: Dict[str, Any],
    spec: Any,
    search_config: SearchConfiguration,
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    from ...shared import _coerce_none_strings  # local import to avoid cycles

    base_params = dict(spec.default_params)
    raw_baseline: Dict[str, Any] = {}

    if job.baseline_hyperparameters:
        raw_baseline.update(job.baseline_hyperparameters)
    else:
        baseline_config = node_config.get("baseline_hyperparameters") if isinstance(node_config, dict) else None
        if isinstance(baseline_config, str):
            try:
                raw_baseline.update(_sanitize_parameters(baseline_config))
            except ValueError as exc:
                raise ValueError("Baseline hyperparameters JSON is invalid") from exc

    raw_baseline = _coerce_none_strings(raw_baseline)

    metadata_fields = get_hyperparameters_for_model(spec.key)
    allowed_key_candidates = {field.get("name") for field in metadata_fields if isinstance(field, dict)}
    allowed_keys: set[str] = {key for key in allowed_key_candidates if isinstance(key, str)}
    allowed_keys.update(spec.default_params.keys())
    allowed_keys.update(search_config.search_space.keys())

    filtered_baseline = _filter_supported_parameters(raw_baseline, allowed_keys)
    base_params.update(filtered_baseline)

    filtered_search_space = _filter_supported_parameters(search_config.search_space, allowed_keys)
    if not filtered_search_space:
        raise ValueError(
            "Search space is empty after filtering unsupported hyperparameters for this estimator."
        )

    if spec.key == "logistic_regression":
        _sanitize_logistic_regression_hyperparameters(base_params, filtered_search_space)
        if not filtered_search_space:
            raise ValueError(
                "Search space is empty after normalizing logistic regression parameters. Use 'ovr' or 'multinomial' for multi_class."
            )

    return base_params, filtered_search_space


def _build_searcher(
    spec: Any,
    search_config: SearchConfiguration,
    base_params: Dict[str, Any],
    filtered_search_space: Dict[str, List[Any]],
    splitter,
):
    estimator = spec.factory(**base_params)
    search_kwargs = {
        "estimator": estimator,
        "scoring": search_config.scoring,
        "cv": splitter,
        "n_jobs": 1,
        "return_train_score": True,
        "refit": True,
    }

    if search_config.strategy == "grid":
        return GridSearchCV(param_grid=filtered_search_space, **search_kwargs)
    if search_config.strategy == "halving":
        return HalvingGridSearchCV(
            param_grid=filtered_search_space,
            resource="n_samples",
            **search_kwargs,
        )
    if search_config.strategy == "halving_random":
        halving_random_kwargs: Dict[str, Any] = {
            **search_kwargs,
            "param_distributions": filtered_search_space,
            "random_state": search_config.random_state,
        }
        if search_config.n_iterations is not None and search_config.n_iterations > 0:
            halving_random_kwargs["n_candidates"] = int(max(1, search_config.n_iterations))
        return HalvingRandomSearchCV(**halving_random_kwargs)
    if search_config.strategy == "optuna":
        if not _HAS_OPTUNA or OptunaSearchCV is None:
            raise RuntimeError(_OPTUNA_INTEGRATION_GUIDANCE)

        optuna_distributions = _build_optuna_distributions(filtered_search_space)
        iterations = search_config.n_iterations
        if iterations is None or iterations <= 0:
            iterations = min(30, max(len(optuna_distributions), 1) * 10)

        optuna_kwargs: Dict[str, Any] = {
            "estimator": estimator,
            "param_distributions": optuna_distributions,
            "n_trials": int(iterations),
            "cv": splitter,
            "scoring": search_config.scoring,
            "refit": True,
            "return_train_score": True,
            "n_jobs": 1,
        }
        if search_config.random_state is not None:
            optuna_kwargs["random_state"] = search_config.random_state

        return _create_optuna_searcher(optuna_kwargs, search_config)

    iterations = search_config.n_iterations
    if iterations is None or iterations <= 0:
        iterations = min(20, max(len(filtered_search_space), 1) * 5)
    return RandomizedSearchCV(
        param_distributions=filtered_search_space,
        n_iter=int(iterations),
        random_state=search_config.random_state,
        **search_kwargs,
    )


__all__ = ["_prepare_search_parameters", "_build_searcher"]
