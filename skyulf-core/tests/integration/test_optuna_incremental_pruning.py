"""Optuna prunes only estimator paths that can incrementally fit the real fold input."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.fold_pipeline import FoldAwareModelStep
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling._tuning.strategies import optuna as strategy
from skyulf.modeling.classification import SGDClassifierCalculator

optuna = pytest.importorskip("optuna")


def _search(estimator, *, pruner="hyperband", search_space=None, strategy_params=None):
    """Construct the real searcher with a small estimator epoch budget."""
    messages = []
    config = TuningConfig(
        strategy="optuna",
        n_trials=2,
        search_space=search_space or {},
        strategy_params={"pruner": pruner, **(strategy_params or {})},
    )
    search = strategy.build_optuna_searcher(
        config,
        estimator,
        StratifiedKFold(2, shuffle=True, random_state=7),
        "accuracy",
        None,
        messages.append,
    )
    return search, messages


def _classification_data():
    """Use a deterministic small dataset with both classes in each fold."""
    X, y = make_classification(
        n_samples=40, n_features=4, n_informative=3, n_redundant=0, random_state=7
    )
    return pd.DataFrame(X, columns=list("abcd")), pd.Series(y)


def test_selected_hyperband_reports_sgd_intermediate_scores():
    """Selecting a pruner must activate real per-epoch CV reporting for direct SGD."""
    search, _ = _search(SGDClassifier(max_iter=3, random_state=7))
    X, y = _classification_data()
    search.fit(X, y)
    assert search.enable_pruning is True
    assert isinstance(search.study_.pruner, optuna.pruners.HyperbandPruner)
    assert all(trial.intermediate_values for trial in search.study_.trials)
    assert all(
        0 <= score <= 1
        for trial in search.study_.trials
        for score in trial.intermediate_values.values()
    )
    assert all(max(trial.intermediate_values) < 3 for trial in search.study_.trials)


def test_real_pruner_can_stop_an_incremental_trial(monkeypatch):
    """Intermediate scores must reach Optuna's pruning decision and stop actual trials."""
    monkeypatch.setattr(
        strategy, "build_optuna_pruner", lambda name: optuna.pruners.ThresholdPruner(lower=1.1)
    )
    search, _ = _search(SGDClassifier(max_iter=4, random_state=7))
    X, y = _classification_data()
    search.fit(X, y)
    assert [trial.state for trial in search.study_.trials] == [optuna.trial.TrialState.PRUNED] * 2
    assert all(list(trial.intermediate_values) == [0] for trial in search.study_.trials)


@pytest.mark.parametrize("params", [{"pruner": "none"}, {"strategy_params": {"pruning": False}}])
def test_explicitly_disabled_pruning_keeps_regular_fit(params):
    """None and the documented explicit false switch must keep ordinary fit behavior."""
    search, _ = _search(SGDClassifier(max_iter=3, random_state=7), **params)
    X, y = _classification_data()
    search.fit(X, y)
    assert search.enable_pruning is False
    assert all(not trial.intermediate_values for trial in search.study_.trials)
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in search.study_.trials)


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(),
        Pipeline(
            [("scale", StandardScaler()), ("model", SGDClassifier(max_iter=3, random_state=7))]
        ),
        FoldAwareModelStep(estimator=SGDClassifier(max_iter=3, random_state=7)),
    ],
)
def test_unsupported_outer_estimators_keep_fit_and_explain_skipped_pruning(estimator):
    """An incremental inner model must never bypass a pipeline or fold wrapper."""
    search, messages = _search(estimator)
    X, y = _classification_data()
    search.fit(X, y)
    assert search.enable_pruning is False
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in search.study_.trials)
    assert any(
        "pruning disabled" in message.lower() and "incremental training" in message
        for message in messages
    )


@pytest.mark.parametrize(
    "fixed, space",
    [
        ({"early_stopping": True}, {}),
        ({"class_weight": "balanced"}, {}),
        ({}, {"early_stopping": [False, True]}),
        ({}, {"class_weight": [None, "balanced"]}),
        ({}, {"max_iter": [2, 3]}),
    ],
)
def test_incompatible_partial_fit_options_disable_pruning(fixed, space):
    """Partial-fit restrictions must not turn previously valid fit candidates into failed trials."""
    search, messages = _search(
        SGDClassifier(max_iter=3, random_state=7, **fixed), search_space=space
    )
    X, y = _classification_data()
    search.fit(X, y)
    assert search.enable_pruning is False
    assert any("pruning disabled" in message.lower() for message in messages)
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in search.study_.trials)


def test_sgd_tuning_preserves_fold_preprocessing_and_reports_unavailable_pruning():
    """Selecting Hyperband must preserve scaler refits inside each original training fold."""
    fit_rows = []

    class FoldScaler:
        """Run real scaling while recording which source rows fitted its statistics."""

        def fit_transform(self, X, y):
            """Fit statistics only on the fold's training rows."""
            fit_rows.append(frozenset(X.index))
            self.scaler_ = StandardScaler().fit(X)
            return pd.DataFrame(self.scaler_.transform(X), index=X.index, columns=X.columns), y

        def transform(self, X, y):
            """Score held-out rows with the already fitted scaler."""
            return pd.DataFrame(self.scaler_.transform(X), index=X.index, columns=X.columns), y

    X, y = _classification_data()
    calculator = SGDClassifierCalculator()
    calculator.default_params["max_iter"] = 3
    messages = []
    config = TuningConfig(
        strategy="optuna",
        n_trials=2,
        cv_folds=2,
        cv_type="stratified_k_fold",
        cv_random_state=7,
        search_space={"alpha": [0.001]},
        strategy_params={"pruner": "hyperband"},
    )
    result = TuningCalculator(calculator).tune(
        X.to_numpy(),
        y.to_numpy(),
        config,
        preprocessing=FoldScaler(),
        preprocessing_frames=(X, y),
        log_callback=messages.append,
    )
    folds = StratifiedKFold(2, shuffle=True, random_state=7).split(X, y)
    expected = {frozenset(train) for train, _ in folds}
    assert set(fit_rows) == expected
    assert len(fit_rows) == 4
    assert result.n_trials == 2
    assert np.isfinite(result.best_score)
    assert any(
        "pruning disabled" in message.lower() and "Pipeline" in message for message in messages
    )


def test_incremental_statistics_models_keep_single_fit_per_fold():
    """GaussianNB must not count every fold's observations another 1,000 times."""
    fitted_counts = []
    partial_calls = []

    class ObservedGaussianNB(GaussianNB):
        """Observe real GaussianNB training without changing its statistical update behavior."""

        def fit(self, X, y, sample_weight=None):
            """Record how many observations the ordinarily fitted model has counted."""
            result = super().fit(X, y, sample_weight=sample_weight)
            fitted_counts.append(sum(self.class_count_))
            return result

        def partial_fit(self, X, y, classes=None, sample_weight=None):
            """Record public incremental updates so accidental epoch repetition is visible."""
            partial_calls.append(len(X))
            return super().partial_fit(X, y, classes=classes, sample_weight=sample_weight)

    search, messages = _search(ObservedGaussianNB())
    assert search.enable_pruning is False
    X, y = _classification_data()
    search.fit(X, y)
    expected = cross_val_score(
        GaussianNB(), X, y, cv=StratifiedKFold(2, shuffle=True, random_state=7)
    ).mean()
    assert search.best_score_ == pytest.approx(expected)
    assert fitted_counts == [20, 20, 20, 20]
    assert partial_calls == []
    assert any(
        "pruning disabled" in message.lower() and "max_iter" in message for message in messages
    )


def test_public_tuning_calculator_activates_sgd_pruning(monkeypatch):
    """The public tuning path must retain incremental reporting after estimator preparation."""
    searches = []
    real_builder = strategy.build_optuna_searcher

    def capture_search(*args, **kwargs):
        """Retain the real searcher for inspecting the study after public tuning completes."""
        search = real_builder(*args, **kwargs)
        searches.append(search)
        return search

    monkeypatch.setattr(strategy, "build_optuna_searcher", capture_search)
    calculator = SGDClassifierCalculator()
    calculator.default_params["max_iter"] = 3
    X, y = _classification_data()
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        config=TuningConfig(
            strategy="optuna",
            n_trials=2,
            cv_folds=2,
            search_space={"alpha": [0.001]},
            strategy_params={"pruner": "hyperband"},
        ),
    )
    assert model.predict(X).shape == (40,)
    assert np.isfinite(result.best_score)
    assert searches[0].enable_pruning is True
    assert all(trial.intermediate_values for trial in searches[0].study_.trials)


def test_mlp_solver_search_uses_regular_fit_for_lbfgs_candidates():
    """A solver choice that removes partial_fit must not crash incremental trials."""
    search, messages = _search(
        MLPClassifier(hidden_layer_sizes=(3,), max_iter=3, random_state=7),
        search_space={"solver": ["lbfgs"]},
    )
    assert search.enable_pruning is False
    X, y = _classification_data()
    search.fit(X, y)
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in search.study_.trials)
    assert all(not trial.intermediate_values for trial in search.study_.trials)
    assert any("lbfgs" in message for message in messages)
