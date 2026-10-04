"""Explicit temporal/group policy evaluation for fixed-model CV entrypoints."""

from copy import deepcopy
from typing import Any, cast

from ._cv_weights import preflight_weights, weight_kwargs
from ._tuning.cv_policy import policy_description, policy_splitter, prepare_policy_data
from ._tuning.engine import TuningCalculator
from ._tuning.schemas import TuningConfig
from .cross_validation import _aggregate_metrics, _run_cv_fold


def policy_config(
    cv_type: str,
    n_folds: int,
    shuffle: bool,
    random_state: int,
    time_column: str | None,
    options: dict[str, Any],
) -> TuningConfig:
    """Build the same split contract used by advanced tuning from ordinary CV controls."""
    return TuningConfig(
        cv_type=cast(Any, cv_type),
        cv_folds=n_folds,
        cv_shuffle=shuffle,
        cv_random_state=random_state,
        cv_time_column=time_column,
        cv_nested_type=options["cv_nested_type"],
        cv_group_column=options["group_column"],
        cv_gap=options["gap"],
        cv_test_size=options["test_size"],
        cv_max_train_size=options["max_train_size"],
        cv_inner_folds=options["inner_folds"],
    )


def perform_policy_cv(
    calculator: Any,
    X: Any,
    y: Any,
    model_config: dict[str, Any],
    policy: TuningConfig,
    preprocessing: Any,
    log_callback: Any,
    progress_callback: Any,
    sample_weight: Any = None,
) -> dict[str, Any]:
    """Evaluate fixed parameters under the same verified boundaries as tuning."""
    if policy.cv_type == "nested_cv":
        return _fixed_nested(
            calculator, X, y, model_config, policy, preprocessing, log_callback, sample_weight
        )
    X, y, metadata, positions = prepare_policy_data(
        X, y, policy, calculator.problem_type, preprocessing, return_positions=True
    )
    if sample_weight is not None:
        sample_weight = sample_weight[positions]
    cv = policy_splitter(policy, calculator.problem_type, y, metadata)
    preflight_weights(sample_weight, cv, X, y)
    folds = []
    for index, (train, test) in enumerate(cv.split(X, y)):
        result = _run_cv_fold(
            calculator,
            X,
            y,
            train,
            test,
            model_config,
            calculator.problem_type,
            index,
            policy.cv_folds,
            progress_callback,
            log_callback,
            deepcopy(preprocessing),
            sample_weight,
        )
        folds.append(result | {"split": cv.evidence[index]})
    return {
        "aggregated_metrics": _aggregate_metrics([f["metrics"] for f in folds]),
        "folds": folds,
        "cv_config": {
            "n_folds": policy.cv_folds,
            "cv_type": policy.cv_type,
            "shuffle": policy.cv_shuffle,
            "random_state": policy.cv_random_state,
        },
        "split_policy": policy_description(policy, calculator.problem_type),
    }


def _fixed_nested(
    calculator: Any,
    X: Any,
    y: Any,
    model_config: dict[str, Any],
    policy: TuningConfig,
    preprocessing: Any,
    log_callback: Any,
    sample_weight: Any = None,
) -> dict[str, Any]:
    """Evaluate a singleton recipe at both levels without introducing parameter choices."""
    calculator = deepcopy(calculator)
    model_config = deepcopy(model_config)
    calculator.prepare_tuning_params(model_config)
    params = calculator._resolve_fit_params(model_config)
    structural = set(calculator.STRUCTURAL_TUNING_KEYS)
    configured = model_config.get("params", model_config) or {}
    # Resolved estimator objects belong to calculator defaults, not reportable
    # candidate axes. Explicit estimator overrides still follow ordinary fitting.
    structural.update({"estimators", "estimator"} - configured.keys())
    policy.strategy = "grid"
    policy.metric = "accuracy" if calculator.problem_type == "classification" else "mse"
    policy.search_space = {
        name: [value] for name, value in params.items() if name not in structural
    }
    result = TuningCalculator(calculator).tune(
        X,
        y,
        policy,
        preprocessing=preprocessing,
        preprocessing_frames=(X, y) if preprocessing is not None else None,
        log_callback=log_callback,
        **weight_kwargs(sample_weight),
    )
    if result.nested_cv is None:
        raise ValueError("Nested fixed-model evaluation did not produce complete evidence.")
    return result.nested_cv | {"fixed_parameters": True}
