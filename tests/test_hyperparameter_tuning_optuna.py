from core.feature_engineering.nodes.modeling.hyperparameter_tuning_tasks import (
    _create_optuna_searcher,
    SearchConfiguration,
)
from core.feature_engineering.nodes.modeling.model_training_tasks import CrossValidationConfig


def _make_search_configuration(random_state: int | None = 7) -> SearchConfiguration:
    return SearchConfiguration(
        strategy="optuna",
        selected_strategy="optuna",
        search_space={"mock": [1, 2]},
        n_iterations=5,
        scoring=None,
        random_state=random_state,
        cross_validation=CrossValidationConfig(
            enabled=True,
            strategy="auto",
            folds=3,
            shuffle=False,
            random_state=random_state,
            refit_strategy="train_only",
        ),
    )


def test_create_optuna_searcher_attaches_sampler_when_supported(monkeypatch):
    captured_kwargs: dict[str, object] = {}

    class DummySearchCV:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    class DummySampler:
        def __init__(self, seed=None):
            captured_kwargs["sampler_seed"] = seed

    from core.feature_engineering.nodes.modeling import hyperparameter_tuning_tasks as tasks_module

    monkeypatch.setattr(tasks_module, "OptunaSearchCV", DummySearchCV)
    monkeypatch.setattr(tasks_module, "TPESampler", DummySampler)
    monkeypatch.setattr(tasks_module, "_optuna_accepts_sampler", lambda: True)
    monkeypatch.setattr(tasks_module, "_HAS_OPTUNA", True)

    search_config = _make_search_configuration(random_state=11)

    optuna_kwargs = {
        "estimator": object(),
        "param_distributions": {"mock": [1, 2]},
        "n_trials": 3,
        "cv": object(),
        "scoring": None,
        "refit": True,
        "return_train_score": True,
        "n_jobs": 1,
    }

    _create_optuna_searcher(optuna_kwargs, search_config)

    assert "sampler" in captured_kwargs
    assert captured_kwargs["sampler_seed"] == 11


def test_create_optuna_searcher_retries_without_sampler(monkeypatch):
    call_kwargs: list[dict[str, object]] = []

    class DummySearchCV:
        def __init__(self, **kwargs):
            call_kwargs.append(dict(kwargs))
            if "sampler" in kwargs:
                raise TypeError("unexpected keyword argument 'sampler'")
            self.kwargs = kwargs

    class DummySampler:
        def __init__(self, seed=None):
            self.seed = seed

    from core.feature_engineering.nodes.modeling import hyperparameter_tuning_tasks as tasks_module

    monkeypatch.setattr(tasks_module, "OptunaSearchCV", DummySearchCV)
    monkeypatch.setattr(tasks_module, "TPESampler", DummySampler)
    monkeypatch.setattr(tasks_module, "_optuna_accepts_sampler", lambda: True)
    monkeypatch.setattr(tasks_module, "_HAS_OPTUNA", True)

    search_config = _make_search_configuration(random_state=13)

    optuna_kwargs = {
        "estimator": object(),
        "param_distributions": {"mock": [1, 2]},
        "n_trials": 4,
        "cv": object(),
        "scoring": None,
        "refit": True,
        "return_train_score": True,
        "n_jobs": 1,
    }

    searcher = _create_optuna_searcher(optuna_kwargs, search_config)

    assert isinstance(searcher, DummySearchCV)
    assert len(call_kwargs) == 2
    assert "sampler" in call_kwargs[0]
    assert "sampler" not in call_kwargs[1]
