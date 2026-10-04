"""Tuning must apply the same class weighting as a direct calculator fit."""

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from skyulf.modeling import classification
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig

STRATEGIES = ["grid", "random", "halving_grid", "halving_random", "optuna"]


class DuplicateMinority:
    """Resample training rows while leaving validation and serving rows intact."""

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Change the training class frequencies before model fitting."""
        minority = np.asarray(y) == 1
        return (
            pd.concat([X, X.loc[minority]], ignore_index=True),
            pd.concat([y, y.loc[minority]], ignore_index=True),
        )

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Preserve held-out rows so scorers keep their original target alignment."""
        return X, y


def _data() -> tuple[pd.DataFrame, pd.Series]:
    """Create the imbalanced dataset that exposes ignored class weighting."""
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        weights=[0.9, 0.1],
        flip_y=0,
        random_state=17,
    )
    return pd.DataFrame(X), pd.Series(y)


def _config(strategy: Any, params: dict[str, Any]) -> TuningConfig:
    """Run one deterministic candidate on two full-resource folds."""
    if strategy == "optuna":
        pytest.importorskip("optuna_integration.sklearn")
    return TuningConfig(
        strategy=strategy,
        metric="accuracy",
        search_space={key: [value] for key, value in params.items()},
        n_trials=1,
        cv_folds=2,
        cv_shuffle=False,
        random_state=7,
        strategy_params={"min_resources": 200},
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("resample", [False, True])
@pytest.mark.parametrize(
    "calculator_name", ["GradientBoostingClassifierCalculator", "XGBClassifierCalculator"]
)
def test_tuning_applies_class_weights_to_every_fold_and_refit(
    monkeypatch, strategy, resample, calculator_name
) -> None:
    """Every strategy must weight post-preprocessing train labels and save a balanced model."""
    if calculator_name == "XGBClassifierCalculator":
        pytest.importorskip("xgboost")
    calculator = getattr(classification, calculator_name)()
    X, y = _data()
    params = {"n_estimators": 10, "max_depth": 2, "class_weight": "balanced"}
    preprocessor = DuplicateMinority() if resample else None
    X_fit, y_fit = preprocessor.fit_transform(X, y) if preprocessor else (X, y)
    direct = calculator.fit(X_fit, y_fit, {"params": {**params, "random_state": 7}})
    plain = calculator.fit(
        X_fit, y_fit, {"params": {**params, "random_state": 7, "class_weight": None}}
    )
    fitted: list[tuple[np.ndarray, Any]] = []
    original_fit = calculator.model_class.fit

    @wraps(original_fit)
    def recording_fit(model, features, labels, *args, **kwargs):
        """Observe real estimator inputs without replacing their training behavior."""
        fitted.append((np.asarray(labels).copy(), kwargs.get("sample_weight")))
        return original_fit(model, features, labels, *args, **kwargs)

    monkeypatch.setattr(calculator.model_class, "fit", recording_fit)
    model, result = TuningCalculator(calculator).fit(
        X, y, _config(strategy, params), preprocessing=preprocessor
    )

    assert result.best_params == params
    assert result.trials[0]["params"] == params
    assert len(fitted) == 3
    for labels, sample_weight in fitted:
        assert sample_weight is not None, "class_weight must reach every native estimator fit"
        counts = np.bincount(labels)
        expected = len(labels) / (2 * counts[labels])
        np.testing.assert_array_equal(sample_weight, expected)
    np.testing.assert_array_equal(model.predict_proba(X), direct.predict_proba(X))
    assert np.max(np.abs(model.predict_proba(X) - plain.predict_proba(X))) > 0.1
    assert type(model) is calculator.model_class
    assert "class_weight" not in model.get_params()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_tuning_preserves_native_class_weight(strategy) -> None:
    """Native weighting must remain on the estimator without applying it a second time."""
    X, y = _data()
    calculator = classification.RandomForestClassifierCalculator()
    params = {"n_estimators": 10, "max_depth": 2, "class_weight": "balanced", "n_jobs": 1}
    direct = calculator.fit(X, y, {"params": {**params, "random_state": 7}})
    model, result = TuningCalculator(calculator).fit(X, y, _config(strategy, params))

    assert result.best_params == params
    np.testing.assert_array_equal(model.predict_proba(X), direct.predict_proba(X))
    assert model.get_params()["class_weight"] == "balanced"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_tuning_balanced_holdout_ignores_validation_class_frequencies(strategy) -> None:
    """Holdout labels must not influence the class weights used in candidate training."""
    X, y = _data()
    params = {"n_estimators": 10, "max_depth": 2, "class_weight": "balanced"}
    calculator = classification.GradientBoostingClassifierCalculator()
    direct = calculator.fit(X, y, {"params": {**params, "random_state": 7}})
    X_val = X.iloc[:40].copy()
    y_val = pd.Series(np.tile([0, 1], 20))
    config = _config(strategy, params)
    config.strategy_params["min_resources"] = 240
    model, result = TuningCalculator(calculator).fit(X, y, config, validation_data=(X_val, y_val))

    assert result.best_score == pytest.approx(np.mean(direct.predict(X_val) == y_val))
    np.testing.assert_array_equal(model.predict_proba(X), direct.predict_proba(X))


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("class_weight", [{0: 0.5, 1: 3.0}, "None", "none", "", None])
def test_tuning_matches_direct_class_weight_options(strategy, class_weight) -> None:
    """Explicit maps and UI no-weight sentinels must retain the direct-fit semantics."""
    X, y = _data()
    params = {"n_estimators": 10, "max_depth": 2, "class_weight": class_weight}
    calculator = classification.GradientBoostingClassifierCalculator()
    direct = calculator.fit(X, y, {"params": {**params, "random_state": 7}})
    model, result = TuningCalculator(calculator).fit(X, y, _config(strategy, params))

    assert result.n_trials == 1
    np.testing.assert_array_equal(model.predict_proba(X), direct.predict_proba(X))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_tuning_applies_class_weight_from_calculator_defaults(strategy) -> None:
    """Fixed calculator weighting must survive searches that tune other parameters."""
    X, y = _data()
    calculator = classification.GradientBoostingClassifierCalculator()
    calculator.default_params["class_weight"] = "balanced"
    params = {"n_estimators": 10, "max_depth": 2}
    direct = calculator.fit(X, y, {"params": {**params, "random_state": 7}})
    model, result = TuningCalculator(calculator).fit(X, y, _config(strategy, params))

    assert result.best_params == params
    np.testing.assert_array_equal(model.predict_proba(X), direct.predict_proba(X))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_tuning_rejects_class_weights_without_sample_weight_support(strategy) -> None:
    """Models unable to honor weighting must fail instead of reporting an unweighted winner."""
    X, y = _data()
    calculator = classification.KNeighborsClassifierCalculator()

    with pytest.raises(ValueError, match="class weighting cannot be applied|All trials failed"):
        TuningCalculator(calculator).fit(X, y, _config(strategy, {"class_weight": "balanced"}))
