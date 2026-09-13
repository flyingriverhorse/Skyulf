"""A tuning score must describe the same Logistic Regression penalty as refitting."""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import LogisticRegressionCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Strong regularization makes a mistaken Elastic Net mix observably different."""
    X, y = make_classification(n_samples=80, n_features=6, random_state=4)
    return pd.DataFrame(X, columns=list("abcdef")), pd.Series(y, name="target")


@pytest.mark.parametrize("penalty,ratio", [("l1", 1.0), ("l2", 0.0)])
@pytest.mark.parametrize("supplied_ratio", [None, 0.5])
def test_explicit_penalty_controls_direct_fit(
    classification_data, penalty: str, ratio: float, supplied_ratio: float | None
) -> None:
    """A stale or nullable Elastic Net setting must not change a selected L1/L2 model."""
    X, y = classification_data
    settings = {"solver": "saga", "C": 0.1, "random_state": 2, "max_iter": 2000}
    params = {**settings, "penalty": penalty, "l1_ratio": supplied_ratio}
    before = deepcopy(params)
    actual = LogisticRegressionCalculator().fit(X, y, {"params": params})
    expected = LogisticRegression(**settings, l1_ratio=ratio).fit(X, y)

    assert params == before
    np.testing.assert_allclose(actual.coef_, expected.coef_, atol=1e-12)
    np.testing.assert_allclose(actual.predict_proba(X), expected.predict_proba(X), atol=1e-12)


@pytest.mark.parametrize(
    "strategy,pruner",
    [
        ("grid", "none"),
        ("random", "none"),
        ("halving_grid", "none"),
        ("halving_random", "none"),
        ("optuna", "none"),
        ("optuna", "median"),
    ],
)
@pytest.mark.parametrize("penalty,ratio", [("l1", 1.0), ("l2", 0.0)])
@pytest.mark.parametrize("fixed_penalty", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_search_scores_and_refit_match_selected_penalty(
    classification_data,
    strategy: Any,
    pruner: str,
    penalty: str,
    ratio: float,
    fixed_penalty: bool,
    wrapped: bool,
) -> None:
    """Every candidate path must agree with independent CV and the returned final model."""
    X, y = classification_data
    settings = {"solver": "saga", "C": 0.1, "max_iter": 2000, "tol": 1e-10}
    calculator = LogisticRegressionCalculator()
    space = {key: [value] for key, value in {**settings, "l1_ratio": 0.5}.items()}
    if fixed_penalty:
        calculator.default_params["penalty"] = penalty
    else:
        space["penalty"] = [penalty]
    config = TuningConfig(
        strategy=strategy,
        search_space=space,
        n_trials=1,
        n_jobs=1,
        cv_type="stratified_k_fold",
        cv_folds=2,
        cv_random_state=3,
        random_state=2,
        metric="neg_log_loss",
        strategy_params={"min_resources": 80, "factor": 2, "pruner": pruner},
    )
    preprocessing = (
        FeatureEngineerFoldAdapter(
            [{"name": "scale", "transformer": "StandardScaler", "params": {}}],
            target_column="target",
        )
        if wrapped
        else None
    )
    original = deepcopy(config), deepcopy(calculator.default_params)
    expected = LogisticRegression(**settings, l1_ratio=ratio, random_state=2)
    reference = make_pipeline(StandardScaler(), expected) if wrapped else expected
    score = cross_val_score(
        reference,
        X,
        y,
        scoring="neg_log_loss",
        cv=StratifiedKFold(2, shuffle=True, random_state=3),
    ).mean()
    model, result = TuningCalculator(calculator).fit(X, y, config, preprocessing=preprocessing)
    reference.fit(X, y)

    assert (config, calculator.default_params) == original
    # Halving resamples even a full-size fold, changing SAGA's row order.
    assert result.best_score == pytest.approx(score, abs=1e-8)
    assert isinstance(model, LogisticRegression)
    assert isinstance(expected.coef_, np.ndarray)
    np.testing.assert_allclose(model.coef_, expected.coef_, atol=1e-12)


@pytest.mark.parametrize("strategy", ["grid", "halving_grid", "optuna"])
def test_search_can_leave_an_unpenalized_fixed_default(classification_data, strategy: Any) -> None:
    """A translated infinite C must not disable a candidate's explicitly selected L2 penalty."""
    X, y = classification_data
    settings = {"solver": "saga", "C": 0.1, "max_iter": 5000, "tol": 1e-10}
    calculator = LogisticRegressionCalculator()
    calculator.default_params.update({**settings, "penalty": None})
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy=strategy,
            search_space={"penalty": ["l2"], "l1_ratio": [0.5]},
            n_trials=1,
            cv_type="stratified_k_fold",
            cv_folds=2,
            cv_random_state=3,
            random_state=2,
            metric="neg_log_loss",
            strategy_params={"min_resources": 80},
        ),
    )
    expected = LogisticRegression(**settings, l1_ratio=0.0, random_state=2)
    scores = cross_val_score(
        expected, X, y, scoring="neg_log_loss", cv=StratifiedKFold(2, shuffle=True, random_state=3)
    )
    expected.fit(X, y)

    assert result.best_score == pytest.approx(scores.mean(), abs=1e-8)
    assert isinstance(model, LogisticRegression)
    assert isinstance(expected.coef_, np.ndarray)
    np.testing.assert_allclose(model.coef_, expected.coef_, atol=1e-12)


@pytest.mark.parametrize("strategy", ["grid", "halving_grid", "optuna"])
def test_ratio_without_a_public_penalty_keeps_its_meaning(
    classification_data, strategy: Any
) -> None:
    """Native ratio-only configuration must remain a supported intentional Elastic Net mix."""
    X, y = classification_data
    settings = {"solver": "saga", "C": 0.1, "l1_ratio": 0.25, "max_iter": 5000, "tol": 1e-10}
    model, result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X,
        y,
        TuningConfig(
            strategy=strategy,
            search_space={key: [value] for key, value in settings.items()},
            n_trials=1,
            cv_type="stratified_k_fold",
            cv_folds=2,
            cv_random_state=3,
            random_state=2,
            metric="neg_log_loss",
            strategy_params={"min_resources": 80},
        ),
    )
    expected = LogisticRegression(**settings, random_state=2)
    scores = cross_val_score(
        expected, X, y, scoring="neg_log_loss", cv=StratifiedKFold(2, shuffle=True, random_state=3)
    )
    expected.fit(X, y)

    assert result.best_score == pytest.approx(scores.mean(), abs=1e-8)
    assert isinstance(model, LogisticRegression)
    assert isinstance(expected.coef_, np.ndarray)
    np.testing.assert_allclose(model.coef_, expected.coef_, atol=1e-12)


@pytest.mark.parametrize("strategy", ["grid", "halving_grid", "optuna"])
def test_mixed_penalty_candidates_keep_independent_scores(strategy: Any) -> None:
    """Each recorded trial must measure its selected penalty even when ratios cross penalties."""
    # Overlapping noisy classes keep the unpenalized optimum finite.
    X, y = make_classification(
        n_samples=80, n_features=6, n_redundant=0, class_sep=0.5, flip_y=0.2, random_state=4
    )
    settings = {"solver": "saga", "C": 1.0, "max_iter": 5000, "tol": 1e-10}
    _, result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X,
        y,
        TuningConfig(
            strategy=strategy,
            search_space={
                **{key: [value] for key, value in settings.items()},
                "penalty": ["l1", "l2", "elasticnet", None],
                "l1_ratio": [0.25, 0.75],
            },
            n_trials=8,
            cv_type="stratified_k_fold",
            cv_folds=2,
            cv_random_state=3,
            random_state=2,
            metric="neg_log_loss",
            strategy_params={"min_resources": 80, "pruner": "none"},
        ),
    )
    expected_scores = {}
    for penalty, ratio in [(p, r) for p in ("l1", "l2", "elasticnet", None) for r in (0.25, 0.75)]:
        effective_ratio = {"l1": 1.0, "l2": 0.0}.get(penalty, ratio)
        params = {**settings, "l1_ratio": effective_ratio, "random_state": 2}
        if penalty is None:
            params["C"] = np.inf
        expected_scores[penalty, ratio] = cross_val_score(
            LogisticRegression(**params),
            X,
            y,
            scoring="neg_log_loss",
            cv=StratifiedKFold(2, shuffle=True, random_state=3),
        ).mean()

    assert len(result.trials) == 8
    for trial in result.trials:
        params = trial["params"]
        assert trial["score"] == pytest.approx(
            expected_scores[params["penalty"], params["l1_ratio"]], abs=1e-8
        )


@pytest.mark.parametrize("strategy", ["grid", "halving_grid", "optuna"])
def test_search_restores_configured_ratio_when_entering_elasticnet(
    classification_data, strategy: Any
) -> None:
    """An L2 constructor's translated zero must not overwrite a later Elastic Net candidate's mix."""
    X, y = classification_data
    settings = {"solver": "saga", "C": 0.1, "l1_ratio": 0.3, "max_iter": 5000, "tol": 1e-10}
    calculator = LogisticRegressionCalculator()
    calculator.default_params.update({**settings, "penalty": "l2"})
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy=strategy,
            search_space={"penalty": ["elasticnet"]},
            n_trials=1,
            cv_type="stratified_k_fold",
            cv_folds=2,
            cv_random_state=3,
            random_state=2,
            metric="neg_log_loss",
            strategy_params={"min_resources": 80},
        ),
    )
    expected = LogisticRegression(**settings, random_state=2)
    score = cross_val_score(
        expected, X, y, scoring="neg_log_loss", cv=StratifiedKFold(2, shuffle=True, random_state=3)
    ).mean()
    expected.fit(X, y)

    assert result.best_score == pytest.approx(score, abs=1e-8)
    assert isinstance(model, LogisticRegression)
    assert isinstance(expected.coef_, np.ndarray)
    np.testing.assert_allclose(model.coef_, expected.coef_, atol=1e-12)


@pytest.mark.parametrize("strategy", ["grid", "halving_grid", "optuna"])
def test_invalid_penalty_cannot_silently_become_l2(classification_data, strategy: Any) -> None:
    """A misspelled public penalty must fail instead of returning a different valid model."""
    X, y = classification_data
    with pytest.raises(ValueError, match="penalty"):
        TuningCalculator(LogisticRegressionCalculator()).fit(
            X,
            y,
            TuningConfig(
                strategy=strategy,
                search_space={"penalty": ["typo"]},
                cv_folds=2,
                n_trials=1,
            ),
        )
