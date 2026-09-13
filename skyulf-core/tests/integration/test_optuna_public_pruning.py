"""Public tuning preserves fold isolation, pruning progress, and winner-only refit."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from skyulf.modeling import classification
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling._tuning.strategies import optuna as strategy

optuna = pytest.importorskip("optuna")


def _data():
    """Keep row identity visible through fold preprocessing and final refit."""
    X, y = make_classification(
        n_samples=36, n_features=4, n_informative=3, n_redundant=0, random_state=7
    )
    index = np.arange(100, 136)
    return pd.DataFrame(X, index=index, columns=list("abcd")), pd.Series(y, index=index)


def _calculator(kind):
    """Use each real public calculator with bounded single-threaded tree budgets."""
    if kind != "random_forest":
        pytest.importorskip(kind)
    name = {
        "random_forest": "RandomForestClassifierCalculator",
        "xgboost": "XGBClassifierCalculator",
        "lightgbm": "LGBMClassifierCalculator",
    }[kind]
    calculator = getattr(classification, name)()
    calculator.default_params.update(n_estimators=2, max_depth=2, n_jobs=1)
    if kind == "lightgbm":
        calculator.default_params["min_child_samples"] = 2
    return calculator


def _config():
    """Use reproducible folds and distinct parameters for the complete and pruned trial."""
    return TuningConfig(
        strategy="optuna",
        n_trials=2,
        search_space={"n_estimators": [2, 4]},
        cv_folds=2,
        cv_type="stratified_k_fold",
        cv_random_state=7,
        random_state=7,
        strategy_params={"pruner": "hyperband"},
    )


def _capture_searches(monkeypatch):
    """Observe actual studies while fixing candidate order without replacing search behavior."""
    searches = []
    real_builder = strategy.build_optuna_searcher

    def capture(*args, **kwargs):
        """Queue the same candidates through both direct and fold-wrapper parameter paths."""
        search = real_builder(*args, **kwargs)
        key = next(iter(search.param_distributions))
        search.study.enqueue_trial({key: 2})
        search.study.enqueue_trial({key: 4})
        searches.append(search)
        return search

    monkeypatch.setattr(strategy, "build_optuna_searcher", capture)
    return searches


def _observe_fits(monkeypatch, calculator):
    """Keep real model fitting while exposing fold and final-refit training sizes."""
    fits = []
    real_fit = calculator.model_class.fit

    def observed_fit(self, X, y, **kwargs):
        """Record attempted real fits, including a native fit interrupted by pruning."""
        fits.append((self, len(X), self.n_estimators))
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(calculator.model_class, "fit", observed_fit)
    return fits


def _observed_scaler(fit_rows):
    """Share observations across cloned workers without sharing their fitted scaling state."""

    class FoldScaler:
        """Fit real scaling statistics only on the rows given to this worker."""

        def fit_transform(self, X, y):
            """Expose exactly which source rows contributed to fold statistics."""
            fit_rows.append(frozenset(X.index))
            self.scaler_ = StandardScaler().fit(X)
            return pd.DataFrame(self.scaler_.transform(X), index=X.index, columns=X.columns), y

        def transform(self, X, y):
            """Use training statistics while retaining aligned validation targets."""
            return pd.DataFrame(self.scaler_.transform(X), index=X.index, columns=X.columns), y

    return FoldScaler()


@pytest.mark.parametrize("kind", ["random_forest", "xgboost", "lightgbm"])
@pytest.mark.parametrize("with_preprocessing", [False, True])
@pytest.mark.filterwarnings("ignore:.*valid feature names.*:UserWarning")
def test_public_tuning_prunes_one_candidate_and_refits_only_the_completed_winner(
    monkeypatch, kind, with_preprocessing
):
    """Pruned progress must remain visible while only complete CV candidates reach final refit."""

    class SecondTrialPruner(optuna.pruners.BasePruner):
        """Keep one full candidate and deterministically stop the next at its first checkpoint."""

        def prune(self, study, trial):
            """Use trial identity so native and fold checkpoints have the same outcome."""
            return trial.number == 1

    monkeypatch.setattr(strategy, "build_optuna_pruner", lambda name: SecondTrialPruner())
    calculator = _calculator(kind)
    fits = _observe_fits(monkeypatch, calculator)
    searches = _capture_searches(monkeypatch)
    fit_rows = []
    preprocessing = _observed_scaler(fit_rows) if with_preprocessing else None
    progress = []
    X, y = _data()
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        config=_config(),
        preprocessing=preprocessing,
        progress_callback=lambda *event: progress.append(event),
    )
    trials = searches[0].study_.trials
    assert [trial.state for trial in trials] == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    ]
    assert list(trials[1].intermediate_values) == [0]
    assert searches[0].mode == ("folds" if kind == "random_forest" else "iterations")
    assert [(rows, count) for _, rows, count in fits] == [(18, 2), (18, 2), (18, 4), (36, 2)]
    assert fits[-1][0] is model
    assert result.best_params == {"n_estimators": 2}
    assert result.best_score == pytest.approx(trials[0].value)
    assert result.trials == [{"params": {"n_estimators": 2}, "score": trials[0].value}]
    assert result.n_trials == 1
    assert len(progress) == 2
    assert [event[:2] for event in progress] == [(1, 2), (2, 2)]
    assert progress[0][2] == pytest.approx(trials[0].value)
    assert progress[1][2] is None
    if preprocessing is not None:
        expected_rows = [
            frozenset(X.iloc[train].index)
            for train, _ in StratifiedKFold(2, shuffle=True, random_state=7).split(X, y)
        ]
        assert fit_rows == [*expected_rows, expected_rows[0], frozenset(X.index)]
        X, _ = preprocessing.transform(X, y)
    assert model.predict(X.to_numpy()).shape == (36,)
    assert np.isfinite(model.predict_proba(X.to_numpy())).all()


@pytest.mark.parametrize("kind", ["random_forest", "xgboost", "lightgbm"])
def test_public_tuning_never_refits_when_every_candidate_is_pruned(monkeypatch, kind):
    """All-pruned searches must explain their outcome without training a partial-score winner."""
    monkeypatch.setattr(
        strategy, "build_optuna_pruner", lambda name: optuna.pruners.ThresholdPruner(lower=1.1)
    )
    calculator = _calculator(kind)
    fits = _observe_fits(monkeypatch, calculator)
    searches = _capture_searches(monkeypatch)
    X, y = _data()
    with pytest.raises(ValueError, match="pruned") as error:
        TuningCalculator(calculator).fit(X, y, config=_config())
    assert "All trials failed" not in str(error.value)
    assert [(rows, count) for _, rows, count in fits] == [(18, 2), (18, 4)]
    assert all(trial.state == optuna.trial.TrialState.PRUNED for trial in searches[0].study_.trials)
