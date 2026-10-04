"""Native pruning must score the requested metric on isolated validation folds."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import f1_score, get_scorer, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skyulf.modeling._tuning.fold_pipeline import FoldAwareModelStep
from skyulf.modeling._tuning.strategies.optuna_folds import fit_and_score_fold


def _estimator(library: str, regression: bool = False, rounds: int = 5) -> Any:
    """Use real optional boosting implementations with a small fixed budget."""
    module = pytest.importorskip(library)
    if library == "xgboost":
        cls = module.XGBRegressor if regression else module.XGBClassifier
        return cls(n_estimators=rounds, max_depth=2, n_jobs=1, random_state=9)
    cls = module.LGBMRegressor if regression else module.LGBMClassifier
    return cls(
        n_estimators=rounds,
        max_depth=2,
        min_child_samples=3,
        n_jobs=1,
        random_state=9,
        verbosity=-1,
    )


def _data(regression: bool = False) -> tuple[Any, Any, Any, Any]:
    """Provide noisy, nontrivial heldout scores whose signed metrics differ."""
    generator = make_regression if regression else make_classification
    X, y = generator(n_samples=90, n_features=5, random_state=18)
    return train_test_split(pd.DataFrame(X), pd.Series(y), test_size=0.3, random_state=4)


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("metric", ["f1", "roc_auc", "neg_mean_squared_error"])
def test_each_native_iteration_uses_the_requested_signed_scorer(library, metric):
    """Native logloss cannot stand in for F1, AUC, or a signed regression loss."""
    regression = metric == "neg_mean_squared_error"
    X_train, X_valid, y_train, y_valid = _data(regression)
    estimator = _estimator(library, regression)
    scorer = get_scorer(metric)
    reports = []
    result = fit_and_score_fold(
        estimator,
        X_train,
        y_train,
        X_valid,
        y_valid,
        scorer,
        lambda score, epoch: reports.append((epoch, score)),
    )
    expected = []
    for rounds in range(1, 6):
        fitted = clone(estimator).set_params(n_estimators=rounds).fit(X_train, y_train)
        expected.append(scorer(fitted, X_valid, y_valid))
    assert [epoch for epoch, _ in reports] == list(range(5))
    assert [score for _, score in reports] == pytest.approx(expected)
    assert result == pytest.approx(expected[-1])
    assert not hasattr(estimator, "n_features_in_")


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_trial_pruning_interrupts_native_fit(library):
    """A pruning decision must stop inside fitting before the remaining trees exist."""
    optuna = pytest.importorskip("optuna")
    X_train, X_valid, y_train, y_valid = _data()
    reports = []

    def report(score, iteration):
        """Reject after the second actual training round."""
        reports.append((iteration, score))
        if iteration == 1:
            raise optuna.TrialPruned("deterministic pruning")

    with pytest.raises(optuna.TrialPruned, match="deterministic pruning"):
        fit_and_score_fold(
            _estimator(library), X_train, y_train, X_valid, y_valid, get_scorer("f1"), report
        )
    assert [iteration for iteration, _ in reports] == [0, 1]


class EncodingScaler:
    """A real train-only scaler with reversible target encoding."""

    def fit_transform(self, X, y):
        """Learn the scale and label vocabulary solely from training rows."""
        self.scaler_ = StandardScaler().fit(X)
        self.mapping_ = {value: index for index, value in enumerate(np.unique(y))}
        return self.scaler_.transform(X), y.map(self.mapping_)

    def transform(self, X, y):
        """Transform validation labels with the training mapping when supplied."""
        target = y.map(self.mapping_) if y is not None else None
        return self.scaler_.transform(X), target


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("labels", [(1, 2), ("negative", "positive")])
@pytest.mark.parametrize("metric", ["f1", "roc_auc"])
def test_encoded_fold_keeps_original_labels_in_native_scores(library, labels, metric):
    """Validation encoding must preserve the scorer's original positive label."""
    X_train, X_valid, y_train, y_valid = _data()
    y_train = y_train.map(dict(enumerate(labels)))
    y_valid = y_valid.map(dict(enumerate(labels)))
    estimator = Pipeline(
        [
            (
                "model",
                FoldAwareModelStep(estimator=_estimator(library), preprocessor=EncodingScaler()),
            )
        ]
    )
    scorer = make_scorer(f1_score, pos_label=labels[1]) if metric == "f1" else get_scorer(metric)
    expected = scorer(clone(estimator).fit(X_train, y_train), X_valid, y_valid)
    reports = []
    result = fit_and_score_fold(
        estimator,
        X_train,
        y_train,
        X_valid,
        y_valid,
        scorer,
        lambda score, epoch: reports.append(score),
    )
    assert result == pytest.approx(expected)
    assert reports[-1] == pytest.approx(expected)
    assert len(reports) == 5
    assert not hasattr(estimator.named_steps["model"].preprocessor, "scaler_")


class FilteringScaler:
    """Filter matching validation targets while exposing real scaler behavior."""

    def fit_transform(self, X, y):
        """Fit on the training rows before applying the same row filter."""
        self.scaler_ = StandardScaler().fit(X)
        return self.transform(X, y)

    def transform(self, X, y):
        """Keep even-indexed rows and their aligned targets."""
        keep = X.index % 2 == 0
        return self.scaler_.transform(X.loc[keep]), y.loc[keep]


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("reporting", [False, True])
def test_validation_row_filter_and_train_only_scaling_stay_aligned(library, reporting):
    """Validation rows cannot enter scaler fitting or become detached from y."""
    X_train = pd.DataFrame({"value": np.arange(20, dtype=float)})
    X_valid = pd.DataFrame({"value": np.arange(100, 108, dtype=float)}, index=range(20, 28))
    y_train = pd.Series(np.arange(20, dtype=float), index=X_train.index)
    y_valid = pd.Series(np.arange(100, 108, dtype=float), index=X_valid.index)
    model = _estimator(library, regression=True)
    estimator = Pipeline([("model", FoldAwareModelStep(model, FilteringScaler()))])
    train_mask, valid_mask = X_train.index % 2 == 0, X_valid.index % 2 == 0
    scaler = StandardScaler().fit(X_train)
    expected_model = clone(model).fit(
        scaler.transform(X_train.loc[train_mask]), y_train[train_mask]
    )
    expected = get_scorer("neg_mean_squared_error")(
        expected_model, scaler.transform(X_valid.loc[valid_mask]), y_valid[valid_mask]
    )

    def scorer(estimator, X, y):
        """Observe validation values so tree scale invariance cannot hide leakage."""
        np.testing.assert_allclose(X, scaler.transform(X_valid.loc[valid_mask]))
        np.testing.assert_array_equal(y, y_valid[valid_mask])
        return get_scorer("neg_mean_squared_error")(estimator, X, y)

    reports = []
    report = (lambda score, iteration: reports.append(score)) if reporting else None
    actual = fit_and_score_fold(estimator, X_train, y_train, X_valid, y_valid, scorer, report)
    assert actual == pytest.approx(expected)
    assert len(reports) == (5 if reporting else 0)


def test_arbitrary_pipeline_preserves_all_steps_and_ordinary_fit():
    """Unrecognized pipelines must retain their complete sklearn execution semantics."""
    from sklearn.linear_model import LogisticRegression

    X_train, X_valid, y_train, y_valid = _data()
    estimator = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
    scorer = get_scorer("roc_auc")
    expected = scorer(clone(estimator).fit(X_train, y_train), X_valid, y_valid)
    reports = []
    actual = fit_and_score_fold(
        estimator,
        X_train,
        y_train,
        X_valid,
        y_valid,
        scorer,
        lambda score, epoch: reports.append(score),
    )
    assert actual == pytest.approx(expected)
    assert reports == []
    assert not hasattr(estimator.named_steps["scale"], "mean_")


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("reporting", [False, True])
def test_native_callbacks_are_retained_without_mutating_the_template(library, reporting):
    """Adding pruning cannot discard user callbacks or attach them to later refits."""
    X_train, X_valid, y_train, y_valid = _data()
    model = _estimator(library)
    observed = []
    if library == "xgboost":
        from xgboost.callback import TrainingCallback

        class Observer(TrainingCallback):
            """Record real native rounds through the preexisting callback."""

            def after_iteration(self, model, epoch, evals_log):
                """Observe callbacks cloned with the template estimator."""
                observed.append(epoch)
                return False

        callback = Observer()
    else:

        def callback(env):
            """Observe the user callback that must accompany native pruning."""
            observed.append(env.iteration)

    model.set_params(callbacks=[callback])
    reports = []
    report = (lambda score, epoch: reports.append(score)) if reporting else None
    result = fit_and_score_fold(model, X_train, y_train, X_valid, y_valid, get_scorer("f1"), report)
    assert np.isfinite(result)
    assert observed == list(range(5))
    assert model.get_params()["callbacks"] == [callback]
    assert len(reports) == (5 if reporting else 0)


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("reporting", [False, True])
def test_fold_class_weights_reach_native_fitting(library, reporting):
    """Converting fold-local class weights must survive the native callback path."""
    from sklearn.utils.class_weight import compute_sample_weight

    X_train, X_valid, y_train, y_valid = _data()
    model = _estimator(library)
    class_weight = {0: 1, 1: 12}
    estimator = Pipeline([("model", FoldAwareModelStep(model, class_weight=class_weight))])
    scorer = get_scorer("neg_log_loss")
    expected_model = clone(model).fit(
        X_train, y_train, sample_weight=compute_sample_weight(class_weight, y_train)
    )
    expected = scorer(expected_model, X_valid, y_valid)
    unweighted = scorer(clone(model).fit(X_train, y_train), X_valid, y_valid)
    reports = []
    report = (lambda score, epoch: reports.append(score)) if reporting else None
    actual = fit_and_score_fold(estimator, X_train, y_train, X_valid, y_valid, scorer, report)
    assert abs(expected - unweighted) > 0.01
    assert actual == pytest.approx(expected)
    assert not reporting or reports[-1] == pytest.approx(expected)


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("reporting", [False, True])
def test_skyulf_lightgbm_sampling_subclasses_preserve_native_params(regression, reporting):
    """Pruning must retain automatic sampling and native class-weight policy."""
    pytest.importorskip("lightgbm")
    from skyulf.modeling.classification import _SamplingLGBMClassifier
    from skyulf.modeling.regression import _SamplingLGBMRegressor

    X_train, X_valid, y_train, y_valid = _data(regression)
    cls = _SamplingLGBMRegressor if regression else _SamplingLGBMClassifier
    model = cls(
        n_estimators=5,
        min_child_samples=3,
        subsample=0.6,
        n_jobs=1,
        random_state=9,
        verbosity=-1,
        class_weight=None if regression else {0: 1, 1: 8},
    )
    scorer = get_scorer("neg_mean_squared_error" if regression else "neg_log_loss")
    expected = scorer(clone(model).fit(X_train, y_train), X_valid, y_valid)
    reports = []
    report = (lambda score, epoch: reports.append(score)) if reporting else None
    actual = fit_and_score_fold(model, X_train, y_train, X_valid, y_valid, scorer, report)
    assert actual == pytest.approx(expected)
    assert not reporting or reports[-1] == pytest.approx(expected)
    assert model.subsample_freq == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_non_finite_fold_scores_fail_instead_of_becoming_trial_results(invalid):
    """Undefined metrics cannot enter best-trial selection as successful scores."""
    from sklearn.linear_model import LinearRegression

    X_train, X_valid, y_train, y_valid = _data(regression=True)
    with pytest.raises(ValueError, match="non-finite"):
        fit_and_score_fold(
            LinearRegression(), X_train, y_train, X_valid, y_valid, lambda estimator, X, y: invalid
        )
