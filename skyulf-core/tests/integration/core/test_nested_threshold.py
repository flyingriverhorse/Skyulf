"""Training-only threshold selection and threshold-aware outer evaluation."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from skyulf.data.dataset import SplitDataset
from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator
from skyulf.modeling._tuning.nested_threshold import (
    score_nested_threshold,
    select_nested_threshold,
)
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.pipeline import SkyulfPipeline
from skyulf.registry import NodeRegistry


class _ReverseLabels:
    """Exercise a valid target encoding whose order differs from the raw labels."""

    def fit_transform(self, X, y):
        """Encode the raw positive class as zero without changing any rows."""
        return X, pd.Series(np.where(np.asarray(y) == "yes", 0, 1), index=y.index)

    def transform(self, X, y):
        """Leave features unchanged during inference."""
        return X, y


class _ResamplePositive:
    """Increase training positive rows while preserving raw labels and inference rows."""

    def fit_transform(self, X, y):
        """Append positive training rows so fold-local weights must follow resampling."""
        positive = np.asarray(y) == "yes"
        return pd.concat([X, X[positive]]), pd.concat([y, y[positive]])

    def transform(self, X, y):
        """Preserve held-out observations exactly once."""
        return X, y


def _rows():
    """Provide an imbalanced string target with overlapping class probabilities."""
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=2, weights=[0.8, 0.2], random_state=17
    )
    return pd.DataFrame(X), pd.Series(np.where(y, "yes", "no"))


def _tuner(model=LogisticRegression):
    """Expose the calculator contract without sharing production fitting logic."""
    return SimpleNamespace(
        problem_type="classification",
        model_calculator=SimpleNamespace(model_class=model, default_params={}),
    )


def test_threshold_matches_independent_oof_reference():
    """A threshold must reflect fresh held-out inner predictions, with raw positive labels."""
    X, y = _rows()
    cv = StratifiedKFold(3, shuffle=True, random_state=13)
    config = TuningConfig(metric="f1", random_state=23)
    result = select_nested_threshold(_tuner(), X, y, config, {"C": 0.2}, cv)
    probabilities = np.empty(len(y))
    for train, test in cv.split(X, y):
        model = LogisticRegression(C=0.2, random_state=23).fit(X.iloc[train], y.iloc[train])
        probabilities[test] = model.predict_proba(X.iloc[test])[:, 1]
    candidates = np.linspace(0, 1, 103)[1:-1]
    scores = [
        f1_score(y, np.where(probabilities >= t, "yes", "no"), pos_label="yes") for t in candidates
    ]
    best = min(
        zip(candidates, scores, strict=True), key=lambda pair: (-pair[1], abs(pair[0] - 0.5))
    )[0]
    assert result["decision_thresholds"]["yes"] == pytest.approx(best)
    assert result["oof_rows"] == len(y)
    assert result["positive_class"] == "yes"


def test_threshold_handles_resampled_rows_and_nonnative_class_weights():
    """OOF prediction rows stay untouched while weights derive from each resampled train fold."""
    X, y = _rows()
    cv = StratifiedKFold(3, shuffle=True, random_state=13)
    tuner = _tuner(GaussianNB)
    tuner.model_calculator.default_params = {"class_weight": "balanced"}
    result = select_nested_threshold(
        tuner, X, y, TuningConfig(metric="f1"), {}, cv, _ResamplePositive()
    )
    proba = np.empty(len(y))
    for train, test in cv.split(X, y):
        tx, ty = X.iloc[train], y.iloc[train]
        positive = ty == "yes"
        tx, ty = pd.concat([tx, tx[positive]]), pd.concat([ty, ty[positive]])
        model = GaussianNB().fit(tx, ty, sample_weight=compute_sample_weight("balanced", ty))
        proba[test] = model.predict_proba(X.iloc[test])[:, 1]
    grid = np.linspace(0, 1, 103)[1:-1]
    cutoff = min(
        grid,
        key=lambda cut: (
            -f1_score(y, np.where(proba >= cut, "yes", "no"), pos_label="yes"),
            abs(cut - 0.5),
        ),
    )
    assert result["decision_thresholds"]["yes"] == pytest.approx(cutoff)
    assert result["oof_rows"] == len(y)


@pytest.mark.parametrize("metric", ["f1", "roc_auc"])
def test_outer_score_matches_threshold_or_probability_reference(metric):
    """Outer labels affect evaluation only, while ranking metrics keep original probabilities."""
    X, y = _rows()
    config = TuningConfig(metric=metric, random_state=23)
    train_x, train_y = X.iloc[:90], y.iloc[:90]
    test_x, test_y = X.iloc[90:], y.iloc[90:]
    selection = select_nested_threshold(_tuner(), train_x, train_y, config, {}, StratifiedKFold(3))
    model = LogisticRegression(random_state=23).fit(train_x, train_y)
    proba = model.predict_proba(test_x)[:, 1]
    if metric == "f1":
        pred = np.where(proba >= selection["decision_thresholds"]["yes"], "yes", "no")
        expected = f1_score(test_y, pred, pos_label="yes")
    else:
        expected = roc_auc_score(test_y, proba)
    actual = score_nested_threshold(
        _tuner(), train_x, train_y, test_x, test_y, config, {}, selection
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["roc_auc", "pr_auc", "log_loss"])
def test_threshold_probability_scoring_supports_legacy_classifier_detection(metric, monkeypatch):
    """Supported sklearn 1.4/1.5 scorers must recognize the threshold wrapper as a classifier."""
    from sklearn import base
    from sklearn.utils import _response

    X, y = _rows()
    config = TuningConfig(metric=metric)
    selection = {"decision_thresholds": {"no": 0.5, "yes": 0.5}}
    arguments = (
        _tuner(),
        X.iloc[:90],
        y.iloc[:90],
        X.iloc[90:],
        y.iloc[90:],
        config,
        {},
        selection,
    )
    expected = score_nested_threshold(*arguments)

    def legacy_is_classifier(estimator):
        """Use the actual estimator recognition rule from sklearn 1.4 and 1.5."""
        return getattr(estimator, "_estimator_type", None) == "classifier"

    monkeypatch.setattr(_response, "is_classifier", legacy_is_classifier)
    monkeypatch.setattr(base, "is_classifier", legacy_is_classifier)
    assert score_nested_threshold(*arguments) == pytest.approx(expected)


def test_time_oof_reports_partial_coverage():
    """Temporal warmup rows cannot produce valid OOF predictions or enter threshold selection."""
    X, y = _rows()
    result = select_nested_threshold(_tuner(), X, y, TuningConfig(), {}, TimeSeriesSplit(3))
    assert result["oof_rows"] == 90
    assert result["training_rows"] == 120


def test_threshold_evidence_supports_json_numeric_labels():
    """Persisted report keys must use native labels even when numpy supplies the targets."""
    X, y = _rows()
    numeric = np.where(y == "yes", 2, 1)
    result = select_nested_threshold(
        _tuner(), X, numeric, TuningConfig(metric="f1"), {}, StratifiedKFold(3)
    )
    restored = json.loads(json.dumps(result))
    assert restored["positive_class"] == 2
    assert set(restored["decision_thresholds"]) == {"1", "2"}


def test_probability_unavailable_fails_loudly():
    """An unsupported model cannot silently report threshold-enabled nested evaluation."""
    X, y = _rows()
    with pytest.raises(ValueError, match="predict_proba"):
        select_nested_threshold(_tuner(LinearSVC), X, y, TuningConfig(), {}, StratifiedKFold(3))


@pytest.mark.parametrize("metric", ["roc_auc", "pr_auc", "log_loss"])
def test_probability_scores_preserve_raw_positive_class_after_reverse_encoding(metric):
    """A valid encoding must not invert ranking probabilities or the positive class."""
    X, y = _rows()
    config = TuningConfig(metric=metric)
    selection = {"decision_thresholds": {"no": 0.7, "yes": 0.3}}
    raw = score_nested_threshold(
        _tuner(), X.iloc[:90], y.iloc[:90], X.iloc[90:], y.iloc[90:], config, {}, selection
    )
    encoded = score_nested_threshold(
        _tuner(),
        X.iloc[:90],
        y.iloc[:90],
        X.iloc[90:],
        y.iloc[90:],
        config,
        {},
        selection,
        preprocessing=_ReverseLabels(),
    )
    assert encoded == pytest.approx(raw)


def test_nonfinite_candidate_metric_fails(monkeypatch):
    """An all-NaN objective must not become an apparently selected default threshold."""
    X, y = _rows()
    monkeypatch.setattr(
        "skyulf.modeling._tuning.nested_threshold.resolve_threshold_metric",
        lambda *args, **kwargs: (lambda yt, yp: float("nan"), "f1"),
    )
    with pytest.raises(ValueError, match="finite"):
        select_nested_threshold(_tuner(), X, y, TuningConfig(), {}, StratifiedKFold(3))


def test_nonfinite_probabilities_fail(monkeypatch):
    """Invalid model probabilities cannot yield a misleading finite cutoff artifact."""
    X, y = _rows()
    monkeypatch.setattr(
        LogisticRegression, "predict_proba", lambda self, X: np.full((len(X), 2), np.nan)
    )
    with pytest.raises(ValueError, match="probabilities must be finite"):
        select_nested_threshold(_tuner(), X, y, TuningConfig(), {}, StratifiedKFold(3))


def test_threshold_selection_rejects_multiclass():
    """The declared binary policy cannot silently optimize unsupported multiclass decisions."""
    X, _ = _rows()
    with pytest.raises(ValueError, match="binary classification"):
        select_nested_threshold(
            _tuner(), X, np.arange(len(X)) % 3, TuningConfig(), {}, StratifiedKFold(3)
        )


def test_threshold_outer_rejects_unknown_labels():
    """Outer label mismatches must fail instead of being coerced into trained classes."""
    X, y = _rows()
    with pytest.raises(ValueError, match="labels do not match"):
        score_nested_threshold(
            _tuner(),
            X,
            y,
            X.iloc[:10],
            ["unknown"] * 10,
            TuningConfig(),
            {},
            {"decision_thresholds": {"no": 0.5, "yes": 0.5}},
        )


@pytest.mark.parametrize("encode", [False, True])
def test_pipeline_nested_threshold_ignores_holdout_labels_and_survives_reload(tmp_path, encode):
    """Held-out labels cannot select thresholds; serialized predictions retain class mapping."""
    X, y = _rows()
    X.columns = ["a", "b", "c", "d"]
    data = X.assign(target=y)
    steps = [{"name": "scale", "transformer": "StandardScaler", "params": {}}]
    if encode:
        steps.append(
            {"name": "encode", "transformer": "LabelEncoder", "params": {"columns": ["target"]}}
        )
    config = {
        "preprocessing": steps,
        "modeling": {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "logistic_regression"},
            "strategy": "grid",
            "metric": "f1",
            "search_space": {"C": [0.2, 1.0]},
            "cv_type": "nested_cv",
            "cv_folds": 3,
            "cv_inner_folds": 2,
            "tune_threshold": True,
        },
    }
    pipelines = []
    for flip in (False, True):
        validation = data.iloc[90:105].copy()
        test = data.iloc[105:].copy()
        if flip:
            validation["target"] = np.where(validation.target == "yes", "no", "yes")
            test["target"] = np.where(test.target == "yes", "no", "yes")
        pipeline = SkyulfPipeline(config)
        pipeline.fit(SplitDataset(train=data.iloc[:90], validation=validation, test=test), "target")
        pipelines.append(pipeline)
    assert pipelines[0].model_estimator is not None
    assert pipelines[1].model_estimator is not None
    assert isinstance(pipelines[0].model_estimator.model, tuple)
    assert isinstance(pipelines[1].model_estimator.model, tuple)
    model, result = pipelines[0].model_estimator.model
    _, changed = pipelines[1].model_estimator.model
    assert result.decision_thresholds is not None
    assert result.decision_thresholds == changed.decision_thresholds
    assert result.best_params == changed.best_params
    assert result.nested_cv == changed.nested_cv
    assert set(result.decision_thresholds) == set(model.classes_)
    path = tmp_path / "nested.pkl"
    pipelines[0].save(str(path))
    loaded = SkyulfPipeline.load(str(path))
    raw_X = X.iloc[105:]
    proba = np.asarray(loaded._predict_proba_transformed(loaded.feature_engineer.transform(raw_X)))
    cutoff = result.decision_thresholds[model.classes_[1]]
    expected = np.where(proba[:, 1] >= cutoff, model.classes_[1], model.classes_[0])
    np.testing.assert_array_equal(loaded.predict(raw_X), expected)
    np.testing.assert_array_equal(loaded.predict(raw_X, use_tuned_thresholds=True), expected)
    np.testing.assert_array_equal(loaded.predict(raw_X), pipelines[0].predict(raw_X))
    transformed = loaded.feature_engineer.transform(raw_X)
    applier = TuningApplier(NodeRegistry.get_applier("logistic_regression")())
    disabled = applier.predict(transformed, (model, replace(result, decision_thresholds=None)))
    np.testing.assert_array_equal(disabled, model.predict(np.asarray(transformed)))


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
def test_every_strategy_selects_training_only_final_threshold(strategy):
    """Every strategy must evaluate fold thresholds and select a separate final threshold."""
    X, y = _rows()
    tuner = TuningCalculator(NodeRegistry.get_calculator("decision_tree_classifier")())
    config = TuningConfig(
        strategy=strategy,
        metric="f1",
        cv_type="nested_cv",
        cv_folds=3,
        cv_inner_folds=2,
        tune_threshold=True,
        search_space={"max_depth": [1, 3]},
        n_trials=2,
        strategy_params={"min_resources": 40, "factor": 2, "pruner": "none"},
    )
    model, result = tuner.fit(X, y, config)
    assert result.decision_thresholds is not None
    assert set(result.decision_thresholds) == set(model.classes_)
    assert result.nested_cv is not None
    assert len(result.nested_cv["folds"]) == 3
    for fold in result.nested_cv["folds"]:
        assert np.isfinite(fold["outer_score"])
        assert fold["threshold_selection"]["oof_rows"] == fold["train_rows"]
        assert fold["threshold_selection"]["selection"] == "inner_oof"
    assert result.nested_cv["threshold_selection"]["training_rows"] == len(X)
    assert (
        result.nested_cv["threshold_selection"]["decision_thresholds"] == result.decision_thresholds
    )
    assert result.decision_threshold_metric == "f1"


@pytest.mark.parametrize("policy", ["time_series_split", "group_k_fold", "stratified_group_k_fold"])
def test_nested_threshold_composes_with_time_and_group_policy(policy):
    """Threshold OOF fits must honor the same named temporal or group policy as the searches."""
    X, y = _rows()
    y = pd.Series(np.where(np.arange(len(X)) % 4 == 0, "yes", "no"))
    X.columns = ["a", "b", "c", "d"]
    config = TuningConfig(
        strategy="grid",
        metric="f1",
        cv_type="nested_cv",
        cv_nested_type=policy,
        cv_folds=3,
        cv_inner_folds=2,
        tune_threshold=True,
        search_space={"C": [0.2]},
    )
    if policy == "time_series_split":
        X["time"] = pd.date_range("2024-01-01", periods=len(X), freq="h")
        config.cv_time_column = "time"
        config.cv_gap = 1
        config.cv_shuffle = False
    else:
        X["group"] = np.arange(len(X)) // 4
        config.cv_group_column = "group"
    model, result = TuningCalculator(NodeRegistry.get_calculator("logistic_regression")()).fit(
        X, y, config
    )
    assert model.n_features_in_ == 4
    assert result.decision_thresholds is not None
    assert result.nested_cv is not None
    for fold in result.nested_cv["folds"]:
        assert np.isfinite(fold["outer_score"])
        selection = fold["threshold_selection"]
        assert selection["oof_rows"] == sum(part["test_rows"] for part in fold["inner_splits"])
        assert selection["training_rows"] == fold["train_rows"]
        if policy == "time_series_split":
            assert selection["oof_rows"] < fold["train_rows"]
            assert fold["split"]["train_end"] < fold["split"]["test_start"]
        else:
            assert selection["oof_rows"] == fold["train_rows"]


@pytest.mark.parametrize(
    "family,calibrate",
    [("voting_classifier", False), ("voting_classifier", True), ("stacking_classifier", False)],
)
def test_nested_threshold_supports_probability_ensembles(family, calibrate):
    """Eligible ensemble and calibration fits must retain binary threshold artifacts."""
    X, y = _rows()
    calculator = NodeRegistry.get_calculator(family)()
    calculator.prepare_tuning_params(
        {
            "params": {
                "base_estimators": ["logistic_regression", "gaussian_nb"],
                "voting": "soft",
                "cv": 2,
                "calibrate_base_models": calibrate,
                "calibration_cv": 2,
            }
        }
    )
    config = TuningConfig(
        strategy="grid",
        metric="f1",
        cv_type="nested_cv",
        cv_folds=3,
        cv_inner_folds=2,
        tune_threshold=True,
        search_space={},
    )
    model, result = TuningCalculator(calculator).fit(X, y, config)
    assert result.decision_thresholds is not None
    assert set(result.decision_thresholds) == {"no", "yes"}
    assert model.predict_proba(X).shape == (len(X), 2)


def _reference_threshold(X, y, params, cv):
    """Recompute a cutoff independently from sklearn fits and explicit F1 grid scores."""
    proba = np.empty(len(y))
    for train, test in cv.split(X, y):
        classifier = LogisticRegression(**params).fit(X.iloc[train], y.iloc[train])
        proba[test] = classifier.predict_proba(X.iloc[test])[:, 1]
    candidates = np.linspace(0, 1, 103)[1:-1]
    scores = [
        f1_score(y, np.where(proba >= cut, "yes", "no"), pos_label="yes") for cut in candidates
    ]
    return min(
        zip(candidates, scores, strict=True), key=lambda pair: (-pair[1], abs(pair[0] - 0.5))
    )[0]


def test_complete_nested_threshold_matches_independent_searches():
    """Each outer score must evaluate its own independent search and training-only cutoff."""
    X, y = _rows()
    config = TuningConfig(
        strategy="grid",
        metric="f1",
        cv_type="nested_cv",
        cv_folds=3,
        cv_inner_folds=2,
        cv_random_state=11,
        random_state=23,
        tune_threshold=True,
        search_space={"C": [0.2, 1.0]},
    )
    calculator = NodeRegistry.get_calculator("logistic_regression")()
    _, result = TuningCalculator(calculator).fit(X, y, config)
    assert result.nested_cv is not None
    outer = StratifiedKFold(3, shuffle=True, random_state=11)
    inner = StratifiedKFold(2, shuffle=True, random_state=11)
    scorer = make_scorer(f1_score, pos_label="yes")
    defaults = {**calculator.default_params, "random_state": 23}
    for fold, (train, test) in zip(result.nested_cv["folds"], outer.split(X, y), strict=True):
        train_x, train_y = X.iloc[train], y.iloc[train]
        search = GridSearchCV(
            LogisticRegression(**defaults), {"C": [0.2, 1.0]}, cv=inner, scoring=scorer
        ).fit(train_x, train_y)
        cutoff = _reference_threshold(train_x, train_y, defaults | search.best_params_, inner)
        pred = np.where(search.predict_proba(X.iloc[test])[:, 1] >= cutoff, "yes", "no")
        assert fold["best_params"] == search.best_params_
        assert fold["threshold_selection"]["decision_thresholds"]["yes"] == pytest.approx(cutoff)
        assert fold["outer_score"] == pytest.approx(f1_score(y.iloc[test], pred, pos_label="yes"))
    final = GridSearchCV(
        LogisticRegression(**defaults), {"C": [0.2, 1.0]}, cv=inner, scoring=scorer
    ).fit(X, y)
    assert result.best_params == final.best_params_
    assert result.decision_thresholds is not None
    assert result.decision_thresholds["yes"] == pytest.approx(
        _reference_threshold(X, y, defaults | final.best_params_, inner)
    )
