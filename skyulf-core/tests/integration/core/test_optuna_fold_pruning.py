"""Custom Optuna CV search keeps pruning and completed scores trustworthy."""

import importlib
from concurrent.futures import CancelledError

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import PredefinedSplit, StratifiedKFold, cross_val_score

optuna = pytest.importorskip("optuna")


def _data():
    """Keep source indexes distinct from positional CV indices."""
    X, y = make_classification(
        n_samples=36, n_features=4, n_informative=3, n_redundant=0, random_state=7
    )
    return pd.DataFrame(X, index=np.arange(100, 136)), pd.Series(y, index=np.arange(100, 136))


def _search(estimator=None, *, pruner=None, cv=None, **kwargs):
    """Build the production searcher with controlled sampler and pruning decisions."""
    module = importlib.import_module("skyulf.modeling._tuning.strategies.optuna_search")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=7),
        pruner=pruner if pruner is not None else optuna.pruners.NopPruner(),
    )
    return module.OptunaPruningSearchCV(
        estimator=estimator
        if estimator is not None
        else RandomForestClassifier(n_estimators=3, random_state=7),
        param_distributions=kwargs.pop("param_distributions", {}),
        cv=cv if cv is not None else StratifiedKFold(3, shuffle=True, random_state=7),
        scoring=kwargs.pop("scoring", "accuracy"),
        study=study,
        n_trials=kwargs.pop("n_trials", 2),
        timeout=None,
        n_jobs=kwargs.pop("n_jobs", 1),
        callbacks=kwargs.pop("callbacks", []),
        mode=kwargs.pop("mode", "folds"),
        iteration_budget=kwargs.pop("iteration_budget", 1),
        **kwargs,
    )


def test_random_forest_pruning_stops_between_folds_and_reports_callbacks():
    """A pruned forest candidate must skip its remaining real model fits."""
    fitted_models = []
    callbacks = []

    class ObservedForest(RandomForestClassifier):
        """Record real fits while retaining each model for identity checks."""

        def fit(self, X, y, sample_weight=None):
            """Let a full forest fit finish before the next pruning decision."""
            fitted_models.append(self)
            return super().fit(X, y, sample_weight=sample_weight)

    search = _search(
        ObservedForest(n_estimators=3, random_state=7),
        pruner=optuna.pruners.ThresholdPruner(lower=1.1),
        callbacks=[lambda study, trial: callbacks.append(trial.state)],
    )
    search.fit(*_data())
    assert len(fitted_models) == 2
    assert callbacks == [optuna.trial.TrialState.PRUNED] * 2
    assert all(list(trial.intermediate_values) == [0] for trial in search.study_.trials)
    with pytest.raises(ValueError, match="pruned"):
        _ = search.best_params_
    assert search.n_trials_ == 2


@pytest.mark.parametrize("split_kind", ["stratified", "iterable", "predefined"])
def test_unpruned_search_preserves_full_cv_scores_and_positional_slicing(split_kind):
    """A custom search must score every supplied split like ordinary sklearn CV."""
    X, y = _data()
    cv = StratifiedKFold(3, shuffle=True, random_state=7)
    if split_kind == "predefined":
        cv = PredefinedSplit(np.repeat([0, 1, 2], 12))
    splits = list(cv.split(X, y))
    expected_folds = cross_val_score(
        RandomForestClassifier(n_estimators=3, random_state=7), X, y, cv=splits
    )
    supplied_cv = iter(splits) if split_kind == "iterable" else cv
    search = _search(cv=supplied_cv)
    search.fit(X, y)
    assert search.best_score_ == pytest.approx(expected_folds.mean())
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in search.study_.trials)
    for trial in search.study_.trials:
        assert list(trial.intermediate_values) == [0, 1, 2]
        assert list(trial.intermediate_values.values()) == pytest.approx(
            np.cumsum(expected_folds) / np.arange(1, 4)
        )


def test_pruner_cannot_discard_a_trial_after_its_final_fold():
    """A completed CV score remains a candidate when pruning could save no further work."""

    class FinalFoldPruner(optuna.pruners.BasePruner):
        """Reject a trial only after all three scores exist."""

        def prune(self, study, trial):
            """Expose an erroneous pruning check at the final completed fold."""
            return trial.last_step == 2

    search = _search(pruner=FinalFoldPruner())
    search.fit(*_data())
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in search.study_.trials)
    assert np.isfinite(search.best_score_)


def test_failed_later_fold_invalidates_the_entire_candidate():
    """An attractive partial CV score must not win when a later fold cannot score."""
    X, y = _data()
    splits = list(PredefinedSplit(np.repeat([0, 1, 2], 12)).split(X, y))

    def scorer(model, held_out, target):
        """Fail only the second fold of one otherwise trainable candidate."""
        if model.max_depth == 1 and held_out.index.min() == 112:
            raise ValueError("second fold cannot score")
        return accuracy_score(target, model.predict(held_out))

    search = _search(
        cv=splits,
        scoring=scorer,
        param_distributions={"max_depth": optuna.distributions.CategoricalDistribution([1, 2])},
    )
    search.study.enqueue_trial({"max_depth": 1})
    search.study.enqueue_trial({"max_depth": 2})
    search.fit(X, y)
    assert [trial.state for trial in search.study_.trials] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert list(search.study_.trials[0].intermediate_values) == [0]
    assert search.best_params_ == {"max_depth": 2}


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_fold_scores_fail_instead_of_becoming_candidates(bad_score):
    """Every completed fold score must be finite before the candidate may win."""
    search = _search(scoring=lambda model, X, y: bad_score)
    search.fit(*_data())
    assert all(trial.state == optuna.trial.TrialState.FAIL for trial in search.study_.trials)
    with pytest.raises(ValueError, match="No trials are completed yet"):
        _ = search.best_score_


def test_parallel_trials_and_folds_do_not_share_fitted_estimators():
    """Reusing one fitted model would leak candidate or fold state under parallel tuning."""
    fitted_models = []

    class ObservedForest(RandomForestClassifier):
        """Reject a second fit of the same estimator instance."""

        def fit(self, X, y, sample_weight=None):
            """Keep each real estimator alive until identity assertions finish."""
            if hasattr(self, "estimators_"):
                raise ValueError("estimator reused across folds")
            fitted_models.append(self)
            return super().fit(X, y, sample_weight=sample_weight)

    original = ObservedForest(n_estimators=3, random_state=7)
    search = _search(original, n_jobs=2, n_trials=4)
    search.fit(*_data())
    assert len(fitted_models) == 12
    assert len({id(model) for model in fitted_models}) == 12
    assert not hasattr(original, "estimators_")
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in search.study_.trials)


@pytest.mark.parametrize("error", [KeyboardInterrupt, CancelledError])
def test_cancellation_propagates_without_continuing_other_trials(error):
    """A cancelled user run must not silently train the remaining candidates."""

    def cancelled_score(model, X, y):
        """Represent a cancellation delivered while evaluating a fitted fold."""
        raise error("stop tuning")

    search = _search(scoring=cancelled_score, n_trials=3)
    with pytest.raises(error, match="stop tuning"):
        search.fit(*_data())
    assert len(search.study_.trials) == 1


def _booster(kind, n_estimators):
    """Use real optional boosters with tiny deterministic training budgets."""
    if kind == "xgboost":
        xgboost = pytest.importorskip("xgboost")
        return xgboost.XGBClassifier(
            n_estimators=n_estimators, max_depth=2, random_state=7, n_jobs=1
        )
    lightgbm = pytest.importorskip("lightgbm")
    return lightgbm.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=2,
        min_child_samples=2,
        random_state=7,
        n_jobs=1,
        verbosity=-1,
    )


@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
@pytest.mark.filterwarnings("ignore:.*valid feature names.*:UserWarning")
def test_native_reporting_uses_common_fold_steps_for_different_iteration_budgets(kind):
    """Candidate-specific iteration counts must not move the next fold's reporting steps."""
    X, y = _data()
    cv = StratifiedKFold(2, shuffle=True, random_state=7)
    search = _search(
        _booster(kind, 2),
        cv=cv,
        mode="iterations",
        iteration_budget=4,
        param_distributions={"n_estimators": optuna.distributions.IntDistribution(2, 4, step=2)},
    )
    search.study.enqueue_trial({"n_estimators": 2})
    search.study.enqueue_trial({"n_estimators": 4})
    search.fit(X, y)
    assert list(search.study_.trials[0].intermediate_values) == [0, 1, 4, 5]
    assert list(search.study_.trials[1].intermediate_values) == list(range(8))
    for trial in search.study_.trials:
        count = trial.params["n_estimators"]
        expected_folds = cross_val_score(_booster(kind, count), X, y, cv=cv)
        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.intermediate_values[count - 1] == pytest.approx(expected_folds[0])
        assert trial.intermediate_values[4 + count - 1] == pytest.approx(expected_folds.mean())
        assert trial.value == pytest.approx(expected_folds.mean())


@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
@pytest.mark.filterwarnings("ignore:.*valid feature names.*:UserWarning")
def test_native_pruning_preserves_final_score_for_a_shorter_candidate(kind):
    """A shared larger budget must not allow pruning after the candidate has fully trained."""

    class FinalIterationPruner(optuna.pruners.BasePruner):
        """Try pruning after the shorter candidate completes its second fold."""

        def prune(self, study, trial):
            """Expose a final-iteration check that mistakenly uses the common budget."""
            return trial.last_step == 5

    search = _search(
        _booster(kind, 2),
        cv=StratifiedKFold(2),
        mode="iterations",
        iteration_budget=4,
        pruner=FinalIterationPruner(),
        n_trials=1,
    )
    search.fit(*_data())
    assert search.study_.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert np.isfinite(search.best_score_)
