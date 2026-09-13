"""Resolve model aliases identically for training and data-free capability inspection."""

from skyulf.registry import NodeRegistry


def get_model_components(algorithm: str, *, task_type: str | None = None):
    """Build fresh calculator/applier components, resolving ambiguous aliases by task."""
    algo = algorithm.lower().replace(" ", "_").replace("-", "_")
    alias_map = {
        "logisticregression": "logistic_regression",
        "randomforestclassifier": "random_forest_classifier",
        "ridgeregression": "ridge_regression",
        "ridge": "ridge_regression",
        "randomforestregressor": "random_forest_regressor",
    }
    task_alias_map = {
        "classification": "random_forest_classifier",
        "regression": "random_forest_regressor",
    }
    if algo == "random_forest":
        if task_type not in task_alias_map:
            raise ValueError(
                "Ambiguous algorithm alias 'random_forest'; provide "
                "task_type='classification' or task_type='regression'"
            )
        registry_id = task_alias_map[task_type]
    else:
        registry_id = alias_map.get(algo, algo)
    try:
        calculator_cls = NodeRegistry.get_calculator(registry_id)
        applier_cls = NodeRegistry.get_applier(registry_id)
    except ValueError:
        raise ValueError(f"Unknown algorithm: {algorithm} (Registry ID: {registry_id})") from None
    calculator = calculator_cls()
    if task_type and getattr(calculator, "problem_type", None) != task_type:
        raise ValueError(
            f"Algorithm '{algorithm}' resolves to a {calculator.problem_type} "
            f"model, incompatible with task_type='{task_type}'"
        )
    return calculator, applier_cls()
