import json

import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN
from core.feature_engineering.nodes.modeling.hyperparameter_tuning_tasks import (
    _coerce_search_space,
    _filter_supported_parameters,
    _sanitize_logistic_regression_hyperparameters,
)
from core.feature_engineering.nodes.modeling.model_training_registry import get_model_spec
from core.feature_engineering.nodes.modeling.model_training_tasks import (
    _classification_metrics,
    _prepare_training_data,
)


def _build_frame():
    return pd.DataFrame(
        {
            "feature_numeric": [1.0, 2.0, 3.0, 4.0],
            "feature_categorical": ["a", "b", "a", "c"],
            "target": ["yes", "no", "yes", "no"],
            SPLIT_TYPE_COLUMN: ["train", "train", "test", "test"],
        }
    )


def test_prepare_training_data_with_split_classification():
    frame = _build_frame()

    X_train, y_train, X_validation, y_validation, X_test, y_test, feature_columns, target_meta = _prepare_training_data(frame, "target")

    assert list(feature_columns) == ["feature_numeric", "feature_categorical"]
    assert X_train.shape == (2, 2)
    assert y_train.tolist() == [1, 0]
    assert X_test is not None and y_test is not None
    assert X_test.shape == (2, 2)
    assert y_test.tolist() == [1, 0]
    assert target_meta["dtype"] == "categorical"
    assert target_meta["categories"] == ["no", "yes"]


def test_prepare_training_data_without_split_builds_test_none():
    frame = _build_frame().drop(columns=[SPLIT_TYPE_COLUMN])

    X_train, y_train, X_validation, y_validation, X_test, y_test, feature_columns, _ = _prepare_training_data(frame, "target")

    assert X_train.shape == (4, 2)
    assert y_train.tolist() == [1, 0, 1, 0]
    assert X_test is None
    assert y_test is None
    assert feature_columns == ["feature_numeric", "feature_categorical"]


def test_prepare_training_data_without_features_raises():
    frame = pd.DataFrame(
        {
            "target": ["yes", "no"],
            SPLIT_TYPE_COLUMN: ["train", "train"],
        }
    )

    with pytest.raises(ValueError, match="No feature columns available for training"):
        _prepare_training_data(frame, "target")


def test_hyperparameter_tuning_results_applied_in_training():
    frame = pd.DataFrame(
        {
            "feature_numeric": [0.0, 1.0, 0.2, 0.8, 0.3, 0.9],
            "feature_categorical": [
                "low",
                "low",
                "medium",
                "medium",
                "high",
                "high",
            ],
            "target": ["no", "no", "no", "yes", "yes", "yes"],
            SPLIT_TYPE_COLUMN: ["train"] * 6,
        }
    )

    X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = _prepare_training_data(frame, "target")

    spec = get_model_spec("random_forest_classifier")

    raw_search_space = json.dumps({
        "n_estimators": [5, 10],
        "max_depth": ["None", 3],
    })
    search_space = _coerce_search_space(raw_search_space)
    assert None in search_space["max_depth"], "Search space should treat 'None' string as None"

    allowed_keys = set(spec.default_params.keys()) | set(search_space.keys())
    filtered_space = _filter_supported_parameters(search_space, allowed_keys)

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
    applied_params = _filter_supported_parameters({**spec.default_params, **best_params}, allowed_keys)

    trained_model = spec.factory(**applied_params)
    trained_model.fit(X_train, y_train)

    metrics = _classification_metrics(
        trained_model,
        X_train.to_numpy(dtype=float),
        y_train.to_numpy(),
    )

    assert metrics["accuracy"] >= 0.8
    for key, value in best_params.items():
        assert trained_model.get_params()[key] == value


def test_logistic_regression_hyperparameters_are_sanitized():
    base_params = {
        "solver": "sag",
        "penalty": "elasticnet",
        "l1_ratio": 0.5,
    }
    search_space = {
        "solver": ["sag", "newton-cg", "lbfgs", "saga"],
        "penalty": ["elasticnet", "l1", "l2", "none"],
        "l1_ratio": [0.3, 0.7],
    }

    _sanitize_logistic_regression_hyperparameters(base_params, search_space)

    # Baseline configuration should be coerced to safe combinations
    assert base_params["solver"] in {"lbfgs", "saga"}
    assert base_params["penalty"] in {"l2", "none"}
    assert "l1_ratio" not in base_params

    # Search space should only contain allowed solver/penalty values and drop unsupported params
    assert set(search_space["solver"]).issubset({"lbfgs", "saga"})
    assert set(search_space["penalty"]).issubset({"l2", "none"})
    assert "l1_ratio" not in search_space
