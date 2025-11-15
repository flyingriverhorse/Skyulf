import json
from importlib import import_module

import pandas as pd
import pytest
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    HalvingGridSearchCV,
)

from core.feature_engineering.modeling.hyperparameter_tuning.tasks import _build_optuna_distributions
from core.feature_engineering.modeling.shared import _coerce_search_space
from core.feature_engineering.modeling.training.registry import get_model_spec
from core.feature_engineering.modeling.training.tasks import _classification_metrics


def _build_frame():
    # Small deterministic dataset with numeric-only features for compatibility
    return pd.DataFrame(
        {
            "feature_numeric": [0.0, 1.0, 0.2, 0.8, 0.3, 0.9],
            "feature_indicator": [0, 0, 1, 1, 2, 2],
            "target": ["no", "no", "no", "yes", "yes", "yes"],
        }
    )


def _get_spec():
    return get_model_spec("random_forest_classifier")


def test_grid_search_works_and_applies_params():
    frame = _build_frame()
    X_train = frame[["feature_numeric", "feature_indicator"]]
    y_train = frame["target"].map({"no": 0, "yes": 1})

    spec = _get_spec()

    raw_search_space = json.dumps({"n_estimators": [5, 10], "max_depth": [None, 3]})
    search_space = _coerce_search_space(raw_search_space)
    assert "max_depth" in search_space

    allowed_keys = set(spec.default_params.keys()) | set(search_space.keys())
    filtered_space = {k: v for k, v in search_space.items() if k in allowed_keys}

    searcher = GridSearchCV(
        estimator=spec.factory(**spec.default_params),
        param_grid=filtered_space,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0),
        n_jobs=1,
        return_train_score=True,
        scoring="accuracy",
    )
    searcher.fit(X_train, y_train)

    best_params = searcher.best_params_
    applied_params = {**spec.default_params, **best_params}

    trained_model = spec.factory(**applied_params)
    trained_model.fit(X_train, y_train)

    metrics = _classification_metrics(trained_model, X_train.to_numpy(dtype=float), y_train.to_numpy())
    assert metrics["accuracy"] >= 0.5

    for key, value in best_params.items():
        assert trained_model.get_params()[key] == value


def test_randomized_search_simple():
    frame = _build_frame()
    X_train = frame[["feature_numeric", "feature_indicator"]]
    y_train = frame["target"].map({"no": 0, "yes": 1})

    spec = _get_spec()

    raw_search_space = json.dumps({"n_estimators": [5, 10, 20], "max_depth": [None, 2, 3]})
    search_space = _coerce_search_space(raw_search_space)

    allowed_keys = set(spec.default_params.keys()) | set(search_space.keys())
    filtered_space = {k: v for k, v in search_space.items() if k in allowed_keys}

    searcher = RandomizedSearchCV(
        estimator=spec.factory(**spec.default_params),
        param_distributions=filtered_space,
        n_iter=2,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0),
        random_state=0,
        n_jobs=1,
        scoring="accuracy",
        return_train_score=True,
    )
    searcher.fit(X_train, y_train)

    assert hasattr(searcher, "best_params_")


def test_halving_grid_search_runs():
    frame = _build_frame()
    X_train = frame[["feature_numeric", "feature_indicator"]]
    y_train = frame["target"].map({"no": 0, "yes": 1})

    spec = _get_spec()

    raw_search_space = json.dumps({"n_estimators": [5, 10], "max_depth": [None, 3]})
    search_space = _coerce_search_space(raw_search_space)

    allowed_keys = set(spec.default_params.keys()) | set(search_space.keys())
    filtered_space = {k: v for k, v in search_space.items() if k in allowed_keys}

    searcher = HalvingGridSearchCV(
        estimator=spec.factory(**spec.default_params),
        param_grid=filtered_space,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0),
        factor=2,
        resource="n_samples",
        min_resources=2,
        max_resources=X_train.shape[0],
        n_jobs=1,
        scoring="accuracy",
    )

    # Should run without error and produce best_params_
    searcher.fit(X_train, y_train)
    assert hasattr(searcher, "best_params_")


def test_optuna_search_if_available():
    # Skip gracefully if optuna or its sklearn integration isn't available
    pytest.importorskip("optuna", reason="Optuna not installed")

    OptunaSearchCV = None
    for module_name in ("optuna_integration.sklearn", "optuna.integration.sklearn", "optuna.integration"):
        try:
            m = import_module(module_name)
            if hasattr(m, "OptunaSearchCV"):
                OptunaSearchCV = getattr(m, "OptunaSearchCV")
                break
        except Exception:
            continue

    if OptunaSearchCV is None:
        pytest.skip("OptunaSearchCV not available in installed optuna packages")

    frame = _build_frame()
    X_train = frame[["feature_numeric", "feature_indicator"]]
    y_train = frame["target"].map({"no": 0, "yes": 1})

    spec = _get_spec()

    raw_search_space = json.dumps({"n_estimators": [5, 10], "max_depth": [None, 3]})
    search_space = _coerce_search_space(raw_search_space)

    allowed_keys = set(spec.default_params.keys()) | set(search_space.keys())
    filtered_space = {k: v for k, v in search_space.items() if k in allowed_keys}

    # Optuna expects distributions; the integration will accept categorical lists in many wrappers.
    distributions = _build_optuna_distributions(filtered_space)

    optuna_search = OptunaSearchCV(
        spec.factory(**spec.default_params),
        distributions,
        n_trials=5,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0),
        scoring="accuracy",
        n_jobs=1,
        random_state=0,
    )

    optuna_search.fit(X_train, y_train)
    assert hasattr(optuna_search, "best_params_")

