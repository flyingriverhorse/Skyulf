"""Pin positional training weights across CV, search and final refits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from skyulf.modeling._tuning.engine import TuningCalculator, TuningConfig
from skyulf.modeling.regression import RidgeRegressionApplier, RidgeRegressionCalculator


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "optuna"])
def test_weighted_search_and_refit_match_sklearn(strategy):
    """Every strategy must train and refit with the caller's original weights."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    X = pd.DataFrame({"row": np.arange(40, dtype=float)})
    y = pd.Series(np.sin(X.row) + X.row / 8)
    weights = np.arange(1, 41, dtype=float)
    config = TuningConfig(
        strategy=strategy,
        search_space={"alpha": [1.0]},
        cv_folds=2,
        n_trials=1,
        n_jobs=1,
        metric="mse",
    )
    model, _ = TuningCalculator(RidgeRegressionCalculator()).fit(
        X, y, config, sample_weight=weights
    )
    expected = Ridge(alpha=1.0).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(model.predict(X.to_numpy()), expected.predict(X))


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
@pytest.mark.parametrize(
    "policy",
    ["k_fold", "group_k_fold", "time_series_split", "nested_cv", "nested_group", "nested_time"],
)
def test_actual_training_rows_keep_weights(monkeypatch, strategy, policy):
    """Duplicate labels, temporal permutations and resource subsamples cannot detach weights."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    rng = np.random.default_rng(21)
    ids = rng.permutation(72)
    X = pd.DataFrame({"row": ids.astype(float)}, index=np.zeros(72))
    if policy in {"group_k_fold", "nested_group"}:
        X["group"] = ids % 6
    if policy in {"time_series_split", "nested_time"}:
        X["time"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(ids, unit="D")
    y = pd.Series(np.sin(ids) + ids / 10, index=X.index)
    calls = []
    original = Ridge.fit

    def record(self, X, y, sample_weight=None):
        """Record the estimator's actual fit payload before delegating to sklearn."""
        rows = np.asarray(X)[:, 0]
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, rows + 1)
        calls.append(rows.copy())
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(Ridge, "fit", record)
    config = TuningConfig(
        strategy=strategy,
        search_space={"alpha": [0.1, 1.0]},
        n_trials=2,
        cv_folds=3,
        cv_inner_folds=2,
        cv_type="nested_cv" if policy.startswith("nested_") else policy,
        cv_shuffle=policy not in {"time_series_split", "nested_time"},
        cv_nested_type={"nested_group": "group_k_fold", "nested_time": "time_series_split"}.get(
            policy, "auto"
        ),
        cv_group_column="group" if policy in {"group_k_fold", "nested_group"} else None,
        cv_time_column="time" if policy in {"time_series_split", "nested_time"} else None,
        n_jobs=1,
        metric="mse",
        strategy_params={"min_resources": 12},
    )
    tuner = TuningCalculator(RidgeRegressionCalculator())
    tuner.fit(X, y, config, sample_weight=ids + 1.0)
    assert len(calls) > 1
    np.testing.assert_array_equal(
        calls[-1], np.sort(ids) if policy in {"time_series_split", "nested_time"} else ids
    )


@pytest.mark.parametrize("policy", ["k_fold", "group_k_fold", "time_series_split", "nested_cv"])
def test_ordinary_cv_weights(monkeypatch, policy):
    """Fixed-model CV uses each actual fold's positional training weights."""
    from skyulf.modeling.cross_validation import perform_cross_validation

    ids = np.random.default_rng(5).permutation(48)
    X = pd.DataFrame({"row": ids.astype(float)}, index=np.zeros(48))
    options: dict[str, Any] = {}
    if policy == "group_k_fold":
        X["group"] = ids % 6
        options["group_column"] = "group"
    if policy == "time_series_split":
        X["time"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(ids, unit="D")
        options["time_column"] = "time"
    calls = []
    original = Ridge.fit

    def record(self, X, y, sample_weight=None):
        """Verify weights at the concrete sklearn boundary."""
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, np.asarray(X)[:, 0] + 1)
        calls.append(len(X))
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(Ridge, "fit", record)
    result = perform_cross_validation(
        RidgeRegressionCalculator(),
        RidgeRegressionApplier(),
        X,
        pd.Series(ids / 10),
        {},
        n_folds=3,
        cv_type=policy,
        sample_weight=ids + 1.0,
        **options,
    )
    assert len(result["folds"]) == 3
    assert len(calls) >= 3


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "optuna"])
def test_zero_weight_training_fold_fails_before_any_fit(monkeypatch, strategy):
    """Global positive mass does not authorize zero-total fold fits or a surviving winner."""
    from skyulf.modeling._sample_weights import SampleWeightError

    if strategy == "optuna":
        pytest.importorskip("optuna")
    X = pd.DataFrame({"row": np.arange(24, dtype=float)})
    calls = []

    def record(*args, **kwargs):
        """Make any fit before deterministic preflight rejection observable."""
        calls.append(True)
        raise AssertionError("fit must not run")

    monkeypatch.setattr(Ridge, "fit", record)
    config = TuningConfig(
        strategy=strategy,
        search_space={"alpha": [1.0]},
        cv_folds=2,
        cv_shuffle=False,
        n_jobs=1,
        n_trials=1,
        metric="mse",
    )
    with pytest.raises(SampleWeightError, match="positive finite total"):
        TuningCalculator(RidgeRegressionCalculator()).fit(
            X, X.row, config, sample_weight=[1.0] + [0.0] * 23
        )
    assert calls == []


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_halving_rejects_actual_zero_weight_resource_subset(n_jobs):
    """Positive full folds cannot hide a zero-mass resource-sampled training subset."""
    from skyulf.modeling._sample_weights import SampleWeightError

    X = pd.DataFrame({"row": np.arange(80, dtype=float)})
    weights = np.zeros(80)
    weights[[0, 40]] = 1
    config = TuningConfig(
        strategy="halving_grid",
        search_space={"alpha": [0.1, 1.0]},
        cv_folds=2,
        cv_shuffle=False,
        n_jobs=n_jobs,
        metric="mse",
        strategy_params={"min_resources": 8},
    )
    with pytest.raises(SampleWeightError, match="positive finite total"):
        TuningCalculator(RidgeRegressionCalculator()).fit(X, X.row, config, sample_weight=weights)


@pytest.mark.parametrize("strategy", ["halving_grid", "halving_random"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_weighted_halving_continues_after_invalid_candidate(strategy, n_jobs):
    """Adding valid weights must not turn an invalid solver candidate into a fatal search."""
    from sklearn.datasets import make_classification

    from skyulf.modeling.classification import LogisticRegressionCalculator

    features, labels = make_classification(n_samples=80, n_features=4, random_state=2)
    X, y = pd.DataFrame(features), pd.Series(labels)
    config = TuningConfig(
        strategy=strategy,
        search_space={"solver": ["lbfgs", "liblinear"], "penalty": ["l1", "l2"]},
        cv_folds=2,
        n_jobs=n_jobs,
        n_trials=4,
        metric="accuracy",
        strategy_params={"min_resources": 80, "max_resources": 80},
    )
    tuner = TuningCalculator(LogisticRegressionCalculator())
    with pytest.warns(UserWarning):
        expected_model, expected_result = tuner.fit(X, y, config)
    with pytest.warns(UserWarning):
        model, result = tuner.fit(X, y, config, sample_weight=np.ones(80))
    assert result.best_score == expected_result.best_score
    assert result.best_params == expected_result.best_params
    np.testing.assert_allclose(model.predict_proba(X), expected_model.predict_proba(X))


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
def test_native_class_weights_reject_zero_effective_fold(strategy):
    """A positive full product cannot authorize a native zero-mass training fold."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    from sklearn.tree import DecisionTreeClassifier

    from skyulf.modeling._sample_weights import SampleWeightError
    from skyulf.modeling.sklearn_wrapper import SklearnCalculator

    X = pd.DataFrame({"row": np.arange(24, dtype=float)})
    y = pd.Series(np.arange(24) % 2)
    weights = np.asarray(y == 0, dtype=float)
    weights[1] = 1.0
    calculator = SklearnCalculator(
        DecisionTreeClassifier, {"class_weight": {0: 0.0, 1: 1.0}}, "classification"
    )
    config = TuningConfig(
        strategy=strategy,
        search_space={"max_depth": [1, 2]},
        cv_folds=2,
        cv_shuffle=False,
        n_jobs=1,
        n_trials=2,
        metric="accuracy",
        strategy_params={"min_resources": 24, "max_resources": 24},
    )
    with pytest.raises(SampleWeightError, match="positive finite total"):
        TuningCalculator(calculator).fit(X, y, config, sample_weight=weights)


def test_holdout_weights_exclude_validation_rows(monkeypatch):
    """Placeholder weights for the concatenated validation partition never enter a fit."""
    X = pd.DataFrame({"row": np.arange(20, dtype=float)})
    validation = pd.DataFrame({"row": np.arange(100, 108, dtype=float)})
    calls = []
    original = Ridge.fit

    def record(self, X, y, sample_weight=None):
        """Detect validation leakage through either sample rows or their placeholders."""
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, np.asarray(X)[:, 0] + 1)
        calls.append(len(X))
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(Ridge, "fit", record)
    config = TuningConfig(
        strategy="halving_grid", search_space={"alpha": [1.0]}, cv_folds=2, n_jobs=1, metric="mse"
    )
    TuningCalculator(RidgeRegressionCalculator()).fit(
        X, X.row, config, validation_data=(validation, validation.row), sample_weight=X.row + 1
    )
    assert calls == [20, 20]


@pytest.mark.parametrize("strategy", ["grid", "optuna"])
def test_nested_threshold_fresh_fits_are_weighted(monkeypatch, strategy):
    """Threshold OOF, outer winners and final refit must all retain row weights."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    from sklearn.linear_model import LogisticRegression

    from skyulf.modeling.classification import LogisticRegressionCalculator

    ids = np.random.default_rng(5).permutation(72)
    X = pd.DataFrame({"row": ids.astype(float)})
    y = pd.Series(ids % 2)
    calls = []
    original = LogisticRegression.fit

    def record(self, X, y, sample_weight=None):
        """Observe every real search, OOF and winner fit with the same row identity."""
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, np.asarray(X)[:, 0] + 1)
        calls.append(len(X))
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LogisticRegression, "fit", record)
    config = TuningConfig(
        strategy=strategy,
        search_space={"C": [1.0]},
        n_trials=1,
        cv_type="nested_cv",
        cv_folds=3,
        cv_inner_folds=2,
        tune_threshold=True,
        metric="f1",
        n_jobs=1,
    )
    _, result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X, y, config, sample_weight=ids + 1.0
    )
    assert len(calls) == 20
    assert result.decision_thresholds is not None


def test_scores_match_unweighted_sklearn_cv():
    """Weights alter training but never selection scores on the held-out observations."""
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    X = pd.DataFrame({"row": np.arange(30, dtype=float)})
    y = pd.Series(np.sin(X.row))
    weights = X.row.to_numpy() + 1
    config = TuningConfig(
        strategy="grid", search_space={"alpha": [1.0]}, cv_folds=3, cv_shuffle=False, metric="mse"
    )
    result = TuningCalculator(RidgeRegressionCalculator()).tune(X, y, config, sample_weight=weights)
    scores = []
    for train, test in KFold(3).split(X):
        model = Ridge(alpha=1.0).fit(X.iloc[train], y.iloc[train], sample_weight=weights[train])
        scores.append(-mean_squared_error(y.iloc[test], model.predict(X.iloc[test])))
    assert result.best_score == pytest.approx(np.mean(scores))


def test_weighted_incremental_optuna_uses_full_fold_fit(monkeypatch):
    """Weighted incremental estimators use the audited custom fold route."""
    pytest.importorskip("optuna")
    from sklearn.linear_model import SGDClassifier

    from skyulf.modeling.classification import SGDClassifierCalculator

    X = pd.DataFrame({"row": np.arange(40, dtype=float)})
    y = pd.Series(np.arange(40) % 2)
    calls = []
    original = SGDClassifier.fit

    def record(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        """Only full fit may consume the weighted payload under this V1 route."""
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, np.asarray(X)[:, 0] + 1)
        calls.append(len(X))
        return original(self, X, y, coef_init, intercept_init, sample_weight)

    def reject(*args, **kwargs):
        """Make an accidental external partial_fit route fail immediately."""
        raise AssertionError("weighted incremental route is not supported")

    monkeypatch.setattr(SGDClassifier, "fit", record)
    monkeypatch.setattr(SGDClassifier, "partial_fit", reject)
    config = TuningConfig(
        strategy="optuna",
        search_space={"max_iter": [30]},
        n_trials=1,
        cv_folds=2,
        metric="accuracy",
        n_jobs=1,
    )
    TuningCalculator(SGDClassifierCalculator()).fit(X, y, config, sample_weight=X.row + 1)
    assert calls == [20, 20, 40]


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_native_optuna_weights_and_unweighted_validation(monkeypatch, library):
    """Native iteration pruning weights training only, including the final refit."""
    pytest.importorskip("optuna")
    module = pytest.importorskip(library)
    from skyulf.modeling.sklearn_wrapper import SklearnCalculator

    cls = module.XGBRegressor if library == "xgboost" else module.LGBMRegressor
    original = cls.fit
    calls = []

    def record(self, X, y, *, sample_weight=None, **kwargs):
        """Inspect training vectors and reject any weighted validation arguments."""
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, np.asarray(X)[:, 0] + 1)
        assert "sample_weight_eval_set" not in kwargs
        assert "eval_sample_weight" not in kwargs
        calls.append(len(X))
        return original(self, X, y, sample_weight=sample_weight, **kwargs)

    monkeypatch.setattr(cls, "fit", record)
    defaults = {"n_estimators": 3, "max_depth": 2, "n_jobs": 1}
    if library == "lightgbm":
        defaults["verbosity"] = -1
    calculator = SklearnCalculator(cls, defaults, "regression")
    X = pd.DataFrame({"row": np.arange(40, dtype=float)})
    config = TuningConfig(
        strategy="optuna",
        search_space={"n_estimators": [3]},
        n_trials=1,
        cv_folds=2,
        metric="mse",
        n_jobs=1,
    )
    TuningCalculator(calculator).fit(X, pd.Series(np.sin(X.row)), config, sample_weight=X.row + 1)
    assert calls == [20, 20, 40]


@pytest.mark.parametrize("strategy", ["grid", "optuna", "halving_grid"])
def test_invalid_combined_class_weights_propagate(strategy):
    """A zero-total class/user product is fatal, never a discarded candidate."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    from sklearn.naive_bayes import GaussianNB

    from skyulf.modeling._sample_weights import SampleWeightError
    from skyulf.modeling.sklearn_wrapper import SklearnCalculator

    calculator = SklearnCalculator(GaussianNB, {"class_weight": {0: 0.0, 1: 0.0}}, "classification")
    X = pd.DataFrame({"row": np.arange(24, dtype=float)})
    config = TuningConfig(
        strategy=strategy,
        search_space={},
        n_trials=1,
        cv_folds=2,
        cv_type="stratified_k_fold",
        n_jobs=1,
        metric="accuracy",
    )
    with pytest.raises(SampleWeightError, match="positive finite total"):
        TuningCalculator(calculator).fit(
            X, pd.Series(np.arange(24) % 2), config, sample_weight=np.ones(24)
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("entrypoint", ["fit", "tune"])
def test_temporal_direct_entrypoints_align_weights(monkeypatch, engine, entrypoint):
    """Both direct APIs sort the actual feature, target and weight positions together."""
    ids = np.random.default_rng(12).permutation(30)
    X = pd.DataFrame(
        {
            "row": ids.astype(float),
            "time": pd.Timestamp("2025-01-01") + pd.to_timedelta(ids, unit="D"),
        }
    )
    y = pd.Series(ids / 10)
    if engine == "polars":
        import polars as pl

        X, y = pl.from_pandas(X), pl.Series(y)
    calls = []
    original = Ridge.fit

    def record(self, X, y, sample_weight=None):
        """Training subsets must reflect ascending chronology and original row identity."""
        rows = np.asarray(X)[:, 0]
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, rows + 1)
        assert np.all(np.diff(rows) >= 0)
        calls.append(len(rows))
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(Ridge, "fit", record)
    config = TuningConfig(
        strategy="grid",
        search_space={"alpha": [1.0]},
        cv_folds=2,
        cv_type="time_series_split",
        cv_time_column="time",
        metric="mse",
    )
    getattr(TuningCalculator(RidgeRegressionCalculator()), entrypoint)(
        X, y, config, sample_weight=ids + 1.0
    )
    assert len(calls) == (3 if entrypoint == "fit" else 2)
